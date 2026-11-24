// Multi-Model Concurrency Benchmark - Phase 9.2.2.1
//
// Verifies that different ModelInstances can execute inference in parallel
// using the separated aggregator + execution worker architecture.
//
// Scenarios:
//   A: Single model, 40 requests (baseline)
//   B: Two models, 40 requests each (80 total)
//   C: Three models, 40 requests each (120 total)
//
// Records per-model throughput, latency percentiles, and concurrent execution.

#if defined(VIE_HAS_ONNXRUNTIME)

#include "vie/decoder/detection.hpp"
#include "vie/model/manifest.hpp"
#include "vie/preprocess/image_input.hpp"
#include "vie/runtime/runtime.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

struct PerModelStats {
  std::string model_id;
  int request_count = 0;
  std::vector<double> latencies_ms;
  double throughput_rps = 0.0;
};

struct ScenarioResult {
  std::string name;
  int total_requests = 0;
  int worker_count = 0;
  double total_time_ms = 0.0;
  double aggregate_throughput_rps = 0.0;
  double avg_latency_ms = 0.0;
  double p50_latency_ms = 0.0;
  double p95_latency_ms = 0.0;
  double max_latency_ms = 0.0;
  std::size_t total_batches = 0;
  std::size_t total_single_infers = 0;
  std::size_t total_aggregated_requests = 0;
  std::vector<PerModelStats> per_model;
};

double percentile(std::vector<double>& data, double p) {
  if (data.empty()) return 0.0;
  std::sort(data.begin(), data.end());
  double index = p / 100.0 * (static_cast<double>(data.size()) - 1.0);
  auto lower = static_cast<std::size_t>(index);
  auto upper = lower + 1;
  if (upper >= data.size()) upper = data.size() - 1;
  double weight = index - static_cast<double>(lower);
  return data[lower] * (1.0 - weight) + data[upper] * weight;
}

std::string vie_fixture(const std::string& rel) {
  return std::string(VIE_SOURCE_DIR) + "/" + rel;
}

vie::ModelConfig make_model_config(const std::string& model_path, const std::string& name) {
  vie::ModelManifest manifest;
  manifest.model_name = name;
  manifest.model_family = "yolov8";
  manifest.num_classes = 80;
  manifest.model_path = model_path;
  manifest.task = vie::TaskType::Detection;
  manifest.model_format = vie::ModelFormat::Onnx;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 640, 640};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  vie::OutputDesc out;
  out.name = "predictions";
  out.shape = {1, 84, 8400};
  out.dtype = vie::DataType::Fp32;
  out.layout = vie::Layout::Unknown;
  out.semantic = "predictions";
  manifest.outputs.push_back(out);
  manifest.decoder.protocol = "yolov8_detect_raw";
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.class_selection = "best_class";
  manifest.decoder.scores_are_probabilities = true;
  manifest.nms.enabled = true;
  manifest.nms.confidence_threshold = 0.25f;
  manifest.nms.iou_threshold = 0.45f;
  manifest.nms.max_detections = 20;
  manifest.nms.pre_nms_top_k = 300;
  manifest.nms.class_aware = true;

  vie::ModelConfig config;
  config.model_path = model_path;
  config.task = vie::TaskType::Detection;
  config.backend = vie::BackendKind::OnnxRuntime;
  config.device = vie::DeviceType::Cpu;
  config.device_id = 0;
  config.manifest = manifest;
  return config;
}

struct OwnedPixels {
  std::vector<uint8_t> data;
};

vie::ImageInput make_test_image(OwnedPixels& owner, int w = 640, int h = 640, int c = 3) {
  owner.data.resize(static_cast<std::size_t>(w) * h * c, 128);
  vie::ImageInput img;
  img.data = owner.data.data();
  img.width = w;
  img.height = h;
  img.channels = c;
  img.row_stride_bytes = w * c;
  img.device = vie::DeviceType::Cpu;
  img.device_id = 0;
  img.color = vie::ColorOrder::Bgr;
  img.layout = vie::ImageLayout::Hwc;
  img.dtype = vie::DataType::Uint8;
  img.ownership = vie::ImageOwnership::Borrowed;
  return img;
}

ScenarioResult run_scenario(const std::string& name,
                            const std::vector<std::string>& model_ids,
                            int requests_per_model,
                            int worker_count,
                            const std::string& model_path) {
  ScenarioResult result;
  result.name = name;
  result.total_requests = static_cast<int>(model_ids.size()) * requests_per_model;
  result.worker_count = worker_count;

  // Configure runtime for concurrency
  vie::RuntimeConfig config;
  config.thread_pool_size = 4;
  config.request_queue_capacity = 256;
  config.inference_worker_count = worker_count;
  config.defer_start = false;
  config.batch_aggregation.enabled = true;
  config.batch_aggregation.max_batch_size = 1;  // Fixed-batch model
  config.batch_aggregation.max_wait_time = std::chrono::milliseconds(0);
  config.batch_aggregation.max_pending_requests = 512;
  config.batch_execution_queue_capacity = 128;

  vie::Runtime runtime(config);

  // Load models
  for (const auto& id : model_ids) {
    runtime.models().load(id, make_model_config(model_path, id));
  }

  // Warmup: one request per model
  for (const auto& id : model_ids) {
    OwnedPixels px;
    runtime.submit(id, make_test_image(px)).get();
  }

  // Initialize per-model stats
  std::map<std::string, PerModelStats> stats_map;
  for (const auto& id : model_ids) {
    stats_map[id].model_id = id;
    stats_map[id].request_count = requests_per_model;
  }

  // Submit all requests and measure latency at completion
  auto bench_start = std::chrono::steady_clock::now();

  // Submit all requests
  struct RequestEntry {
    std::string model_id;
    std::shared_ptr<OwnedPixels> pixels;
    std::future<vie::DetectionResult> future;
    std::chrono::steady_clock::time_point submit_time;
  };
  std::vector<RequestEntry> entries;
  entries.reserve(result.total_requests);

  for (const auto& id : model_ids) {
    for (int i = 0; i < requests_per_model; ++i) {
      RequestEntry entry;
      entry.model_id = id;
      entry.pixels = std::make_shared<OwnedPixels>();
      entry.submit_time = std::chrono::steady_clock::now();
      entry.future = runtime.submit(id, make_test_image(*entry.pixels));
      entries.push_back(std::move(entry));
    }
  }

  // Collect results
  for (auto& entry : entries) {
    entry.future.get();
    auto end = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::milli>(end - entry.submit_time).count();
    stats_map[entry.model_id].latencies_ms.push_back(latency);
  }

  auto bench_end = std::chrono::steady_clock::now();
  result.total_time_ms = std::chrono::duration<double, std::milli>(bench_end - bench_start).count();

  // Gather metrics
  auto metrics = runtime.snapshot_metrics();
  result.total_batches = metrics.total_batches;
  result.total_single_infers = metrics.total_single_infer_calls;
  result.total_aggregated_requests = metrics.total_aggregated_requests;

  // Compute per-model stats
  std::vector<double> all_latencies;
  for (auto& [id, stats] : stats_map) {
    if (!stats.latencies_ms.empty()) {
      double total = std::accumulate(stats.latencies_ms.begin(), stats.latencies_ms.end(), 0.0);
      stats.throughput_rps = stats.latencies_ms.size() / (total / 1000.0);
    }
    all_latencies.insert(all_latencies.end(), stats.latencies_ms.begin(), stats.latencies_ms.end());
    result.per_model.push_back(std::move(stats));
  }

  if (!all_latencies.empty()) {
    result.avg_latency_ms = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0) / all_latencies.size();
    result.p50_latency_ms = percentile(all_latencies, 50.0);
    result.p95_latency_ms = percentile(all_latencies, 95.0);
    result.max_latency_ms = *std::max_element(all_latencies.begin(), all_latencies.end());
  }

  result.aggregate_throughput_rps = result.total_requests / (result.total_time_ms / 1000.0);

  return result;
}

void print_scenario(const ScenarioResult& r) {
  std::cout << "\n============================================================\n";
  std::cout << "Scenario: " << r.name << "\n";
  std::cout << "============================================================\n";
  std::cout << "Total requests: " << r.total_requests << "\n";
  std::cout << "Execution workers: " << r.worker_count << "\n";
  std::cout << "Total time: " << std::fixed << std::setprecision(2) << r.total_time_ms << " ms\n";
  std::cout << "Aggregate throughput: " << std::fixed << std::setprecision(2)
            << r.aggregate_throughput_rps << " req/s\n";
  std::cout << "Avg latency: " << std::fixed << std::setprecision(3) << r.avg_latency_ms << " ms\n";
  std::cout << "P50 latency: " << std::fixed << std::setprecision(3) << r.p50_latency_ms << " ms\n";
  std::cout << "P95 latency: " << std::fixed << std::setprecision(3) << r.p95_latency_ms << " ms\n";
  std::cout << "Max latency: " << std::fixed << std::setprecision(3) << r.max_latency_ms << " ms\n";
  std::cout << "Total execution units: " << r.total_batches << "\n";
  std::cout << "  Single infers (batch=1): " << r.total_single_infers << "\n";
  std::cout << "  Total aggregated requests: " << r.total_aggregated_requests << "\n";

  for (const auto& m : r.per_model) {
    std::cout << "\n  [" << m.model_id << "]\n";
    std::cout << "    Requests: " << m.request_count << "\n";
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2) << m.throughput_rps << " req/s\n";
    if (!m.latencies_ms.empty()) {
      double avg = std::accumulate(m.latencies_ms.begin(), m.latencies_ms.end(), 0.0) / m.latencies_ms.size();
      std::cout << "    Avg latency: " << std::fixed << std::setprecision(3) << avg << " ms\n";
    }
  }
}

}  // namespace

int main() {
  std::string model_path = vie_fixture("testdata/onnx/yolov8n.onnx");
  const int requests_per_model = 40;

  std::cout << "Multi-Model Concurrency Benchmark (Phase 9.2.2.1)\n";
  std::cout << "Model: " << model_path << "\n";
  std::cout << "Requests per model: " << requests_per_model << "\n";

  // Scenario A: Single model
  auto result_a = run_scenario("A: Single Model (40 reqs, 2 workers)",
                               {"model_a"}, requests_per_model, 2, model_path);
  print_scenario(result_a);

  // Scenario B: Two models
  auto result_b = run_scenario("B: Two Models (80 reqs, 4 workers)",
                               {"model_a", "model_b"}, requests_per_model, 4, model_path);
  print_scenario(result_b);

  // Scenario C: Three models
  auto result_c = run_scenario("C: Three Models (120 reqs, 4 workers)",
                               {"model_a", "model_b", "model_c"}, requests_per_model, 4, model_path);
  print_scenario(result_c);

  // Summary
  std::cout << "\n============================================================\n";
  std::cout << "SUMMARY\n";
  std::cout << "============================================================\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Scenario A (1 model):  " << result_a.aggregate_throughput_rps << " req/s, "
            << result_a.total_time_ms << " ms total, P95=" << result_a.p95_latency_ms << " ms\n";
  std::cout << "Scenario B (2 models): " << result_b.aggregate_throughput_rps << " req/s, "
            << result_b.total_time_ms << " ms total, P95=" << result_b.p95_latency_ms << " ms\n";
  std::cout << "Scenario C (3 models): " << result_c.aggregate_throughput_rps << " req/s, "
            << result_c.total_time_ms << " ms total, P95=" << result_c.p95_latency_ms << " ms\n";

  double speedup_b = (result_b.total_requests / result_b.total_time_ms) /
                     (result_a.total_requests / result_a.total_time_ms);
  double speedup_c = (result_c.total_requests / result_c.total_time_ms) /
                     (result_a.total_requests / result_a.total_time_ms);
  std::cout << "\nConcurrency speedup (B/A): " << speedup_b << "x\n";
  std::cout << "Concurrency speedup (C/A): " << speedup_c << "x\n";

  if (speedup_b > 1.3) {
    std::cout << "RESULT: Two-model parallelism shows >30% throughput improvement.\n";
  } else {
    std::cout << "RESULT: Two-model speedup is " << speedup_b
              << "x (expected >1.3x for effective parallelism).\n";
  }

  // Write results to file
  std::ofstream out("multi_model_benchmark_results.csv");
  if (out.is_open()) {
    out << "scenario,total_requests,workers,total_time_ms,throughput_rps,avg_latency_ms,"
        << "p50_latency_ms,p95_latency_ms,max_latency_ms,total_batches,single_infers\n";
    for (const auto* r : {&result_a, &result_b, &result_c}) {
      out << r->name << "," << r->total_requests << "," << r->worker_count << ","
          << r->total_time_ms << "," << r->aggregate_throughput_rps << ","
          << r->avg_latency_ms << "," << r->p50_latency_ms << "," << r->p95_latency_ms << ","
          << r->max_latency_ms << "," << r->total_batches << "," << r->total_single_infers << "\n";
    }
    out.close();
    std::cout << "\nResults written to multi_model_benchmark_results.csv\n";
  }

  return 0;
}

#else  // VIE_HAS_ONNXRUNTIME

#include <iostream>
int main() {
  std::cout << "Multi-model benchmark requires ONNX Runtime. Skipping.\n";
  return 0;
}

#endif  // VIE_HAS_ONNXRUNTIME
