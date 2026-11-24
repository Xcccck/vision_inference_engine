// Batch Aggregator Benchmark - Phase 9.2.2
// Compares throughput/latency with batch aggregation enabled vs disabled
// using the real dynamic batch model (yolov8n_batch.onnx).

#if defined(VIE_HAS_ONNXRUNTIME)

#include "vie/decoder/detection.hpp"
#include "vie/model/manifest.hpp"
#include "vie/preprocess/image_input.hpp"
#include "vie/runtime/runtime.hpp"

#include <algorithm>
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

struct BenchmarkResult {
  std::string scenario;
  int request_count;
  int concurrent_threads;
  double total_time_ms;
  double throughput_rps;
  double avg_latency_ms;
  double p50_latency_ms;
  double p95_latency_ms;
  double max_latency_ms;
  std::size_t total_batches;
  std::size_t max_batch_size_achieved;
  std::size_t timeout_flush_count;
  std::size_t full_flush_count;
};

double percentile(std::vector<double>& data, double p) {
  if (data.empty()) return 0.0;
  std::sort(data.begin(), data.end());
  double index = p / 100.0 * (data.size() - 1);
  auto lower = static_cast<std::size_t>(index);
  auto upper = lower + 1;
  if (upper >= data.size()) upper = data.size() - 1;
  double weight = index - lower;
  return data[lower] * (1.0 - weight) + data[upper] * weight;
}

void print_result(const BenchmarkResult& r) {
  std::cout << "\n=== " << r.scenario << " ===\n";
  std::cout << "Requests: " << r.request_count << "\n";
  std::cout << "Concurrent threads: " << r.concurrent_threads << "\n";
  std::cout << "Total time: " << std::fixed << std::setprecision(2) << r.total_time_ms << " ms\n";
  std::cout << "Throughput: " << std::fixed << std::setprecision(2) << r.throughput_rps << " req/s\n";
  std::cout << "Avg latency: " << std::fixed << std::setprecision(3) << r.avg_latency_ms << " ms\n";
  std::cout << "P50 latency: " << std::fixed << std::setprecision(3) << r.p50_latency_ms << " ms\n";
  std::cout << "P95 latency: " << std::fixed << std::setprecision(3) << r.p95_latency_ms << " ms\n";
  std::cout << "Max latency: " << std::fixed << std::setprecision(3) << r.max_latency_ms << " ms\n";
  std::cout << "Total batches: " << r.total_batches << "\n";
  std::cout << "Max batch size achieved: " << r.max_batch_size_achieved << "\n";
  std::cout << "Timeout flushes: " << r.timeout_flush_count << "\n";
  std::cout << "Full flushes: " << r.full_flush_count << "\n";
}

std::string vie_fixture(const std::string& rel) {
  return std::string(VIE_SOURCE_DIR) + "/" + rel;
}

vie::ModelConfig make_batch_model_config(const std::string& model_path) {
  vie::ModelManifest manifest;
  manifest.model_name = "yolov8n_batch";
  manifest.model_family = "yolov8";
  manifest.num_classes = 80;
  manifest.model_path = model_path;
  manifest.task = vie::TaskType::Detection;
  manifest.model_format = vie::ModelFormat::Onnx;
  manifest.input.name = "images";
  manifest.input.shape = {-1, 3, 640, 640};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  vie::OutputDesc out;
  out.name = "output0";
  out.shape = {-1, 84, 8400};
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

vie::ImageInput make_test_image(OwnedPixels& owner) {
  owner.data.resize(640 * 640 * 3, 128);
  vie::ImageInput img;
  img.data = owner.data.data();
  img.width = 640;
  img.height = 640;
  img.channels = 3;
  img.row_stride_bytes = 640 * 3;
  img.device = vie::DeviceType::Cpu;
  img.device_id = 0;
  img.color = vie::ColorOrder::Bgr;
  img.layout = vie::ImageLayout::Hwc;
  img.dtype = vie::DataType::Uint8;
  img.ownership = vie::ImageOwnership::Borrowed;
  return img;
}

BenchmarkResult run_benchmark(const std::string& scenario,
                               vie::RuntimeConfig& config,
                               const std::string& model_path,
                               int request_count,
                               int concurrent_threads) {
  config.defer_start = false;
  vie::Runtime runtime(config);
  runtime.models().load("batch_model", make_batch_model_config(model_path));

  // Warmup
  {
    OwnedPixels pixels;
    auto image = make_test_image(pixels);
    auto future = runtime.submit("batch_model", image);
    future.get();
  }

  std::vector<double> latencies;
  std::mutex lat_mutex;

  auto benchmark_start = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  for (int t = 0; t < concurrent_threads; ++t) {
    workers.emplace_back([&, t] {
      for (int i = t; i < request_count; i += concurrent_threads) {
        auto req_start = std::chrono::steady_clock::now();
        OwnedPixels pixels;
        auto image = make_test_image(pixels);
        auto future = runtime.submit("batch_model", image);
        try {
          auto result = future.get();
          auto req_end = std::chrono::steady_clock::now();
          double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
          std::lock_guard<std::mutex> lock(lat_mutex);
          latencies.push_back(latency);
        } catch (...) {
          auto req_end = std::chrono::steady_clock::now();
          double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
          std::lock_guard<std::mutex> lock(lat_mutex);
          latencies.push_back(latency);
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  auto benchmark_end = std::chrono::steady_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(benchmark_end - benchmark_start).count();

  auto metrics = runtime.snapshot_metrics();

  BenchmarkResult result;
  result.scenario = scenario;
  result.request_count = static_cast<int>(latencies.size());
  result.concurrent_threads = concurrent_threads;
  result.total_time_ms = total_ms;
  result.throughput_rps = latencies.size() / (total_ms / 1000.0);
  result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  result.p50_latency_ms = percentile(latencies, 50);
  result.p95_latency_ms = percentile(latencies, 95);
  result.max_latency_ms = latencies.empty() ? 0.0 : *std::max_element(latencies.begin(), latencies.end());
  result.total_batches = metrics.total_batches;
  result.max_batch_size_achieved = metrics.max_batch_size_achieved;
  result.timeout_flush_count = metrics.batch_timeout_flush_count;
  result.full_flush_count = metrics.batch_full_flush_count;

  return result;
}

}  // namespace

int main() {
  std::cout << "VIE Batch Aggregator Benchmark - Phase 9.2.2\n";
  std::cout << "==============================================\n";

  std::string model_path = vie_fixture("testdata/onnx_dynamic/yolov8n_batch.onnx");

  // Check if model exists
  {
    std::ifstream f(model_path);
    if (!f.good()) {
      std::cerr << "\n[SKIP] Dynamic batch model not found: " << model_path << "\n";
      std::cerr << "This benchmark requires testdata/onnx_dynamic/yyolov8n_batch.onnx\n";
      return 0;
    }
  }

  const int request_count = 40;
  const int concurrent_threads = 4;

  std::cout << "\nModel: " << model_path << "\n";
  std::cout << "Requests: " << request_count << "\n";
  std::cout << "Concurrent threads: " << concurrent_threads << "\n";

  // Scenario A: Batch aggregation DISABLED (max_batch_size=1)
  {
    vie::RuntimeConfig config;
    config.thread_pool_size = 2;
    config.request_queue_capacity = 100;
    config.inference_worker_count = 1;
    config.batch_aggregation.enabled = true;
    config.batch_aggregation.max_batch_size = 1;
    config.batch_aggregation.max_wait_time = std::chrono::milliseconds(0);
    config.batch_aggregation.max_pending_requests = 200;

    auto result = run_benchmark("A: Batch DISABLED (max_batch_size=1)", config, model_path,
                                 request_count, concurrent_threads);
    print_result(result);
  }

  // Scenario B: Batch aggregation ENABLED (max_batch_size=4, short wait)
  {
    vie::RuntimeConfig config;
    config.thread_pool_size = 2;
    config.request_queue_capacity = 100;
    config.inference_worker_count = 1;
    config.batch_aggregation.enabled = true;
    config.batch_aggregation.max_batch_size = 4;
    config.batch_aggregation.max_wait_time = std::chrono::milliseconds(5);
    config.batch_aggregation.max_pending_requests = 200;

    auto result = run_benchmark("B: Batch ENABLED (max_batch_size=4, 5ms wait)", config, model_path,
                                 request_count, concurrent_threads);
    print_result(result);
  }

  // Scenario C: Batch aggregation ENABLED (max_batch_size=8, longer wait)
  {
    vie::RuntimeConfig config;
    config.thread_pool_size = 2;
    config.request_queue_capacity = 100;
    config.inference_worker_count = 1;
    config.batch_aggregation.enabled = true;
    config.batch_aggregation.max_batch_size = 8;
    config.batch_aggregation.max_wait_time = std::chrono::milliseconds(20);
    config.batch_aggregation.max_pending_requests = 200;

    auto result = run_benchmark("C: Batch ENABLED (max_batch_size=8, 20ms wait)", config, model_path,
                                 request_count, concurrent_threads);
    print_result(result);
  }

  std::cout << "\n\nBenchmark complete.\n";
  std::cout << "\nKey observations:\n";
  std::cout << "- Scenario A processes each image individually (no batching)\n";
  std::cout << "- Scenario B/C group images into batches, reducing per-image overhead\n";
  std::cout << "- Higher max_batch_size may improve throughput but increase latency\n";
  std::cout << "- max_wait_time controls how long to wait for more requests before flushing\n";

  return 0;
}

#else  // !VIE_HAS_ONNXRUNTIME

#include <iostream>

int main() {
  std::cout << "Batch Aggregator Benchmark requires ONNXRuntime backend.\n";
  std::cout << "Rebuild with -DVIE_ENABLE_ONNXRUNTIME=ON\n";
  return 0;
}

#endif  // VIE_HAS_ONNXRUNTIME
