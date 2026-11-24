// Real Model Benchmark - Phase 8.3.1
// Uses actual ONNX models with VIE Runtime.
//
// Statistics Definitions:
//   request_latency = per-request end-to-end wall time (steady_clock)
//     - Start: immediately before model->infer() call
//     - End: immediately after infer() returns (or throws)
//     - Includes: preprocessing, backend execution, decoding, NMS, mutex wait
//   throughput = completed_requests / total_wall_time_seconds
//   mean_latency = sum(request_latency_samples) / sample_count
//   p50/p95: nearest-rank percentile on sorted per-request samples
//     rank = ceil(p/100 * N), value = sorted[rank-1]
//   total_wall_time: measured from before first request to after last future.get()

#include "vie/runtime/runtime.hpp"
#include "vie/model/model_config.hpp"
#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/core/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

// ============================================================================
// Statistics
// ============================================================================

struct LatencyStats {
  std::size_t sample_count = 0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
};

// Nearest-rank percentile: rank = ceil(p/100 * N), value = sorted[rank-1]
// Requires data pre-sorted. Returns 0.0 for empty input.
double nearest_rank_percentile(const std::vector<double>& sorted_data, double p) {
  if (sorted_data.empty()) return 0.0;
  std::size_t n = sorted_data.size();
  std::size_t rank = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(n)));
  if (rank == 0) rank = 1;          // minimum rank is 1
  if (rank > n) rank = n;           // clamp to last
  return sorted_data[rank - 1];     // convert 1-based rank to 0-based index
}

LatencyStats compute_stats(std::vector<double> latencies) {
  LatencyStats stats;
  if (latencies.empty()) {
    throw std::runtime_error("compute_stats: empty sample set, refusing to output fake zero metrics");
  }

  // Validate all samples are finite
  for (double v : latencies) {
    if (!std::isfinite(v)) {
      throw std::runtime_error("compute_stats: non-finite latency sample detected");
    }
  }

  std::sort(latencies.begin(), latencies.end());

  stats.sample_count = latencies.size();
  stats.min_ms = latencies.front();
  stats.max_ms = latencies.back();
  stats.mean_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(latencies.size());
  stats.p50_ms = nearest_rank_percentile(latencies, 50.0);
  stats.p95_ms = nearest_rank_percentile(latencies, 95.0);

  // Consistency validation: min <= p50 <= p95 <= max
  if (stats.min_ms > stats.p50_ms || stats.p50_ms > stats.p95_ms || stats.p95_ms > stats.max_ms) {
    throw std::runtime_error("compute_stats: consistency check failed (min<=p50<=p95<=max)");
  }
  // Mean must be between min and max
  if (stats.mean_ms < stats.min_ms || stats.mean_ms > stats.max_ms) {
    throw std::runtime_error("compute_stats: mean not within [min, max]");
  }

  return stats;
}

// ============================================================================
// Result
// ============================================================================

struct BenchmarkResult {
  std::string scenario;
  std::string backend_type;
  int worker_count;
  int instance_count;
  int total_requests;
  int warmup_count;
  int completed_requests;
  int failed_requests;
  double total_wall_time_ms;
  double throughput_rps;
  LatencyStats latency_stats;
};

void print_result(const BenchmarkResult& r) {
  std::cout << "\n=== " << r.scenario << " ===\n";
  std::cout << "Backend:              " << r.backend_type << "\n";
  std::cout << "Workers:              " << r.worker_count << "\n";
  std::cout << "Instances:            " << r.instance_count << "\n";
  std::cout << "Requests (total):     " << r.total_requests << "\n";
  std::cout << "Requests completed:   " << r.completed_requests << "\n";
  std::cout << "Requests failed:      " << r.failed_requests << "\n";
  std::cout << "Warmup:               " << r.warmup_count << "\n";
  std::cout << "Total wall time:      " << std::fixed << std::setprecision(2)
            << r.total_wall_time_ms << " ms\n";
  std::cout << "Throughput:           " << std::fixed << std::setprecision(2)
            << r.throughput_rps << " req/s\n";
  std::cout << "Latency samples:      " << r.latency_stats.sample_count << "\n";
  std::cout << "Mean latency:         " << std::fixed << std::setprecision(3)
            << r.latency_stats.mean_ms << " ms\n";
  std::cout << "Min latency:          " << std::fixed << std::setprecision(3)
            << r.latency_stats.min_ms << " ms\n";
  std::cout << "P50 latency:          " << std::fixed << std::setprecision(3)
            << r.latency_stats.p50_ms << " ms\n";
  std::cout << "P95 latency:          " << std::fixed << std::setprecision(3)
            << r.latency_stats.p95_ms << " ms\n";
  std::cout << "Max latency:          " << std::fixed << std::setprecision(3)
            << r.latency_stats.max_ms << " ms\n";

  // Cross-validation
  if (static_cast<int>(r.latency_stats.sample_count) != r.completed_requests) {
    std::cerr << "WARNING: sample_count (" << r.latency_stats.sample_count
              << ") != completed_requests (" << r.completed_requests << ")\n";
  }
}

// ============================================================================
// Manifest & Image setup (unchanged from Phase 8.3)
// ============================================================================

vie::ModelManifest create_manifest(const std::string& model_path) {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(model_path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  auto manifest = vie::ManifestGenerator{}.generate(report, result, model_path);

  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  manifest.provenance["input.color"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["input.normalization"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["input.resize"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["input.interpolation"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["input.padding_value"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};

  manifest.decoder.class_selection = "best_class";
  manifest.nms.enabled = true;
  manifest.nms.confidence_threshold = 0.25f;
  manifest.nms.iou_threshold = 0.45f;
  manifest.nms.max_detections = 20;
  manifest.nms.pre_nms_top_k = 300;
  manifest.nms.class_aware = true;
  manifest.provenance["decoder.class_selection"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["nms.enabled"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["nms.confidence_threshold"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["nms.iou_threshold"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["nms.max_detections"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["nms.pre_nms_top_k"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};
  manifest.provenance["nms.class_aware"] = vie::FieldProvenance{vie::Certainty::User, {"benchmark"}};

  return manifest;
}

// CUDA image memory owner for cleanup
struct CudaImageBuffer {
  uint8_t* d_ptr = nullptr;
  std::vector<uint8_t> host_pixels;

  ~CudaImageBuffer() {
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
    if (d_ptr) {
      cudaFree(d_ptr);
      d_ptr = nullptr;
    }
#endif
  }
};

vie::ImageInput create_dummy_image(CudaImageBuffer& buf, int width, int height,
                                    vie::DeviceType device, int device_id) {
  buf.host_pixels.resize(static_cast<std::size_t>(width) * height * 3);
  for (std::size_t i = 0; i < buf.host_pixels.size(); ++i) {
    buf.host_pixels[i] = static_cast<uint8_t>(i % 256);
  }

  vie::ImageInput image;
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  image.device = device;
  image.device_id = device_id;

  if (device == vie::DeviceType::Cuda) {
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
    cudaError_t err = cudaMalloc(&buf.d_ptr, buf.host_pixels.size());
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    err = cudaMemcpy(buf.d_ptr, buf.host_pixels.data(), buf.host_pixels.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(buf.d_ptr);
      buf.d_ptr = nullptr;
      throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(err));
    }
    image.data = buf.d_ptr;
#else
    throw std::runtime_error("CUDA not available in this build");
#endif
  } else {
    image.data = buf.host_pixels.data();
  }

  return image;
}

// ============================================================================
// Scenario runner
// ============================================================================

// Shared constants for consistency across scenarios
constexpr int kWarmup = 10;
constexpr int kRequestsPerThread = 25;

BenchmarkResult run_scenario_a(const std::string& model_path, vie::DeviceType device,
                                const std::string& backend_type, const vie::ModelManifest& manifest,
                                const vie::ImageInput& image) {
  const int workers = 1;
  const int instances = 1;
  const int total_requests = workers * kRequestsPerThread;

  std::cout << "\n\n=== Scenario A: Single instance, single thread ===\n";

  vie::RuntimeConfig config;
  config.thread_pool_size = static_cast<std::size_t>(workers);
  config.default_device = device;
  vie::Runtime runtime(config);

  vie::ModelConfig model_config;
  model_config.model_path = model_path;
  model_config.device = device;
  model_config.device_id = 0;
  model_config.manifest = manifest;
  if (device == vie::DeviceType::Cuda) {
    model_config.cuda_cpu_ep_fallback = true;
  }

  auto model = runtime.models().load("model_a", model_config);

  // Warmup (not included in timing)
  for (int i = 0; i < kWarmup; ++i) {
    model->infer(image);
  }

  // Benchmark
  std::vector<double> latencies;
  int failed = 0;

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < total_requests; ++i) {
    auto req_start = std::chrono::steady_clock::now();
    try {
      model->infer(image);
      auto req_end = std::chrono::steady_clock::now();
      latencies.push_back(std::chrono::duration<double, std::milli>(req_end - req_start).count());
    } catch (const std::exception& e) {
      ++failed;
      std::cerr << "Request " << i << " failed: " << e.what() << "\n";
    }
  }
  auto end = std::chrono::steady_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  int completed = static_cast<int>(latencies.size());

  BenchmarkResult result;
  result.scenario = "A: Single instance, single thread";
  result.backend_type = backend_type;
  result.worker_count = workers;
  result.instance_count = instances;
  result.total_requests = total_requests;
  result.warmup_count = kWarmup;
  result.completed_requests = completed;
  result.failed_requests = failed;
  result.total_wall_time_ms = total_ms;
  result.throughput_rps = static_cast<double>(completed) / (total_ms / 1000.0);
  result.latency_stats = compute_stats(std::move(latencies));
  return result;
}

BenchmarkResult run_scenario_b(const std::string& model_path, vie::DeviceType device,
                                const std::string& backend_type, const vie::ModelManifest& manifest,
                                const vie::ImageInput& image) {
  const int workers = 4;
  const int instances = 1;
  const int total_requests = workers * kRequestsPerThread;

  std::cout << "\n\n=== Scenario B: Single instance, 4 threads ===\n";

  vie::RuntimeConfig config;
  config.thread_pool_size = static_cast<std::size_t>(workers);
  config.default_device = device;
  vie::Runtime runtime(config);

  vie::ModelConfig model_config;
  model_config.model_path = model_path;
  model_config.device = device;
  model_config.device_id = 0;
  model_config.manifest = manifest;
  if (device == vie::DeviceType::Cuda) {
    model_config.cuda_cpu_ep_fallback = true;
  }

  auto model = runtime.models().load("model_b", model_config);

  // Warmup
  for (int i = 0; i < kWarmup; ++i) {
    model->infer(image);
  }

  // Benchmark: 4 threads submit to same single instance
  std::vector<double> latencies;
  std::mutex lat_mutex;
  std::atomic<int> failed_count{0};

  auto start = std::chrono::steady_clock::now();

  std::vector<std::future<void>> futures;
  for (int t = 0; t < workers; ++t) {
    futures.push_back(runtime.thread_pool().submit([&, t] {
      for (int i = 0; i < kRequestsPerThread; ++i) {
        auto req_start = std::chrono::steady_clock::now();
        try {
          model->infer(image);
          auto req_end = std::chrono::steady_clock::now();
          double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
          std::lock_guard<std::mutex> lock(lat_mutex);
          latencies.push_back(latency);
        } catch (const std::exception&) {
          ++failed_count;
        }
      }
      (void)t;
    }));
  }

  // Wait for all tasks to complete BEFORE measuring end time
  for (auto& f : futures) {
    f.get();
  }

  auto end = std::chrono::steady_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  int completed = static_cast<int>(latencies.size());
  int failed = failed_count.load();

  BenchmarkResult result;
  result.scenario = "B: Single instance, 4 threads (serialized by mutex)";
  result.backend_type = backend_type;
  result.worker_count = workers;
  result.instance_count = instances;
  result.total_requests = total_requests;
  result.warmup_count = kWarmup;
  result.completed_requests = completed;
  result.failed_requests = failed;
  result.total_wall_time_ms = total_ms;
  result.throughput_rps = static_cast<double>(completed) / (total_ms / 1000.0);
  result.latency_stats = compute_stats(std::move(latencies));
  return result;
}

BenchmarkResult run_scenario_c(const std::string& model_path, vie::DeviceType device,
                                const std::string& backend_type, const vie::ModelManifest& manifest,
                                const vie::ImageInput& image) {
  const int workers = 4;
  const int instances = 2;
  const int total_requests = workers * kRequestsPerThread;

  std::cout << "\n\n=== Scenario C: 2 instances, 4 threads ===\n";

  vie::RuntimeConfig config;
  config.thread_pool_size = static_cast<std::size_t>(workers);
  config.default_device = device;
  vie::Runtime runtime(config);

  vie::ModelConfig model_config;
  model_config.model_path = model_path;
  model_config.device = device;
  model_config.device_id = 0;
  model_config.manifest = manifest;
  if (device == vie::DeviceType::Cuda) {
    model_config.cuda_cpu_ep_fallback = true;
  }

  auto model1 = runtime.models().load("model_c1", model_config);
  auto model2 = runtime.models().load("model_c2", model_config);

  // Warmup both instances
  for (int i = 0; i < kWarmup; ++i) {
    model1->infer(image);
    model2->infer(image);
  }

  // Benchmark: threads 0,2 -> model1; threads 1,3 -> model2
  std::vector<double> latencies;
  std::mutex lat_mutex;
  std::atomic<int> failed_count{0};

  auto start = std::chrono::steady_clock::now();

  std::vector<std::future<void>> futures;
  for (int t = 0; t < workers; ++t) {
    // Explicit instance assignment: even threads use model1, odd use model2
    std::shared_ptr<vie::ModelInstance> assigned_model = (t % 2 == 0) ? model1 : model2;
    futures.push_back(runtime.thread_pool().submit([&, t, assigned_model] {
      for (int i = 0; i < kRequestsPerThread; ++i) {
        auto req_start = std::chrono::steady_clock::now();
        try {
          assigned_model->infer(image);
          auto req_end = std::chrono::steady_clock::now();
          double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
          std::lock_guard<std::mutex> lock(lat_mutex);
          latencies.push_back(latency);
        } catch (const std::exception&) {
          ++failed_count;
        }
      }
      (void)t;
    }));
  }

  for (auto& f : futures) {
    f.get();
  }

  auto end = std::chrono::steady_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  int completed = static_cast<int>(latencies.size());
  int failed = failed_count.load();

  BenchmarkResult result;
  result.scenario = "C: 2 instances, 4 threads (2 per instance)";
  result.backend_type = backend_type;
  result.worker_count = workers;
  result.instance_count = instances;
  result.total_requests = total_requests;
  result.warmup_count = kWarmup;
  result.completed_requests = completed;
  result.failed_requests = failed;
  result.total_wall_time_ms = total_ms;
  result.throughput_rps = static_cast<double>(completed) / (total_ms / 1000.0);
  result.latency_stats = compute_stats(std::move(latencies));
  return result;
}

}  // namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [cpu|cuda]\n";
    std::cerr << "\nTiming boundaries:\n";
    std::cerr << "  - Per-request latency includes: preprocess + backend execute + decode + NMS\n";
    std::cerr << "  - CUDA path: input resides in GPU memory (H2D done once before benchmark)\n";
    std::cerr << "  - CPU path: input resides in host memory\n";
    std::cerr << "  - D2H for CUDA outputs is handled inside the backend\n";
    std::cerr << "  - Concurrent scenarios include mutex wait time in per-request latency\n";
    std::cerr << "  - Percentile: nearest-rank, rank = ceil(p/100 * N)\n";
    return 1;
  }

  std::string model_path = argv[1];
  std::string device_str = (argc >= 3) ? argv[2] : "cpu";

  vie::DeviceType device = vie::DeviceType::Cpu;
  std::string backend_type = "OnnxRuntime CPU EP";

  if (device_str == "cuda") {
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
    device = vie::DeviceType::Cuda;
    backend_type = "OnnxRuntime CUDA EP (with CPU fallback enabled)";
#else
    std::cerr << "CUDA backend not available in this build\n";
    return 1;
#endif
  }

  std::cout << "VIE Real Model Benchmark - Phase 8.3.1\n";
  std::cout << "=======================================\n";
  std::cout << "Model: " << model_path << "\n";
  std::cout << "Device: " << device_str << "\n";
  std::cout << "Backend: " << backend_type << "\n";
  std::cout << "\nTiming Boundaries:\n";
  std::cout << "  - Per-request latency = end-to-end from before infer() to after infer() returns\n";
  std::cout << "  - Includes: preprocessing, backend execute, decoder, NMS, mutex wait\n";
  std::cout << "  - CUDA input: pre-allocated GPU memory, H2D done once before benchmark loop\n";
  std::cout << "  - CPU input: host memory pointer\n";
  std::cout << "  - D2H of CUDA output handled inside Backend::execute()\n";
  std::cout << "  - Percentile method: nearest-rank (rank = ceil(p/100*N))\n";
  std::cout << "  - Same warmup (" << kWarmup << ") and same requests/thread ("
            << kRequestsPerThread << ") for all scenarios\n";

  try {
    // Create manifest
    std::cout << "\nCreating manifest...\n";
    auto manifest = create_manifest(model_path);
    std::cout << "Manifest created successfully\n";

    // Create dummy image (640x640 RGB, allocated once, reused across all scenarios)
    CudaImageBuffer img_buf;
    auto image = create_dummy_image(img_buf, 640, 640, device, 0);
    if (device == vie::DeviceType::Cuda) {
      std::cout << "Input image allocated in GPU memory (H2D done once before benchmark)\n";
    } else {
      std::cout << "Input image in host memory\n";
    }

    // Run all scenarios
    {
      auto result = run_scenario_a(model_path, device, backend_type, manifest, image);
      print_result(result);
    }

    {
      auto result = run_scenario_b(model_path, device, backend_type, manifest, image);
      print_result(result);
    }

    {
      auto result = run_scenario_c(model_path, device, backend_type, manifest, image);
      print_result(result);
    }

    // Summary
    std::cout << "\n\n=== Summary ===\n";
    std::cout << "Scenarios A/B/C use identical:\n";
    std::cout << "  - Model file: " << model_path << "\n";
    std::cout << "  - Input: 640x640 RGB, same pixel content, same device\n";
    std::cout << "  - Warmup: " << kWarmup << " iterations\n";
    std::cout << "  - Per-thread requests: " << kRequestsPerThread << "\n";
    std::cout << "  - Percentile: nearest-rank\n";
    std::cout << "  - Timing: per-request steady_clock, includes all pipeline stages\n";
    std::cout << "\nBenchmark complete.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "\nFATAL Error: " << e.what() << "\n";
    return 1;
  }
}
