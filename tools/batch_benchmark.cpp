// Phase 9.1 Batch Benchmark
// Measures VIE Batch inference throughput for a dynamic-batch YOLOv8n ONNX
// model at Batch = {1, 2, 4, 8}. Runs the same pipeline the E2E test uses:
// InferenceEngine::infer_batch() on a CPU or CUDA device.
//
// Statistics:
//   - Batch latency: wall time of one infer_batch() call (steady_clock,
//     from just before the call to just after it returns or throws).
//     Includes preprocess + execute + decode + NMS for all N samples.
//   - Per-image latency: batch_latency (reported as derived, not measured).
//   - Throughput: completed_images / total_wall_time_seconds (Images/s)
//     and completed_batches / total_wall_time_seconds (Batches/s).
//   - Percentile: nearest-rank, rank = ceil(p/100 * N).
//   - Warmup = 10 batches, timed = 30 batches for every batch size.
//
// Timing boundary caveats:
//   - CUDA path: the same pre-allocated GPU image buffer is reused across
//     N slots, so H2D is NOT inside the measured region. Preprocess runs on
//     GPU; D2H happens inside the Backend/decoder.
//   - CPU path: host memory pointers, no H2D or D2H.
//   - Batch latency is NOT a single-image latency. Do not read "Batch=4
//     p50 = X ms" as "one image takes X ms"; use per-image mean for that.

#include "vie/engine/inference_engine.hpp"
#include "vie/core/tensor.hpp"
#include "vie/model/model_config.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

constexpr int kWarmup = 10;
constexpr int kTimed = 30;

struct LatencyStats {
  std::size_t sample_count = 0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
};

double nearest_rank_percentile(const std::vector<double>& sorted_data, double p) {
  if (sorted_data.empty()) return 0.0;
  std::size_t n = sorted_data.size();
  std::size_t rank = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(n)));
  if (rank == 0) rank = 1;
  if (rank > n) rank = n;
  return sorted_data[rank - 1];
}

LatencyStats compute_stats(std::vector<double> latencies) {
  if (latencies.empty()) {
    throw std::runtime_error("compute_stats: empty sample set");
  }
  for (double v : latencies) {
    if (!std::isfinite(v)) throw std::runtime_error("compute_stats: non-finite sample");
  }
  std::sort(latencies.begin(), latencies.end());
  LatencyStats stats;
  stats.sample_count = latencies.size();
  stats.min_ms = latencies.front();
  stats.max_ms = latencies.back();
  stats.mean_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                  static_cast<double>(latencies.size());
  stats.p50_ms = nearest_rank_percentile(latencies, 50.0);
  stats.p95_ms = nearest_rank_percentile(latencies, 95.0);
  if (stats.min_ms > stats.p50_ms || stats.p50_ms > stats.p95_ms || stats.p95_ms > stats.max_ms) {
    throw std::runtime_error("compute_stats: consistency check failed");
  }
  return stats;
}

bool file_exists(const std::string& path) {
  struct stat buffer {};
  return ::stat(path.c_str(), &buffer) == 0;
}

// Builds a Manifest for the dynamic-batch model with batch_dim == -1 so the
// engine can accept any positive N. The recognizer does not resolve a fully
// dynamic ONNX in this phase, so the manifest is constructed here in the
// same shape the E2E test uses.
vie::ModelConfig build_config(const std::string& model_path, vie::DeviceType device,
                              bool use_cuda_backend) {
  using namespace vie;
  ModelManifest manifest;
  manifest.model_name = "yolov8n_batch";
  manifest.model_family = "yolov8";
  manifest.num_classes = 80;
  manifest.task = TaskType::Detection;
  manifest.model_format = ModelFormat::Onnx;
  manifest.model_path = model_path;
  manifest.input.name = "images";
  manifest.input.shape = {-1, 3, 640, 640};
  manifest.input.dtype = DataType::Fp32;
  manifest.input.layout = Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  for (const char* field : {"model_name", "model_family", "num_classes", "task", "model_format",
                            "model_path", "input.name", "input.shape", "input.dtype",
                            "input.layout", "input.color", "input.normalization", "input.resize",
                            "input.interpolation", "input.padding_value"}) {
    manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
  }
  OutputDesc out;
  out.name = "output0";
  out.shape = {-1, 84, 8400};
  out.dtype = DataType::Fp32;
  out.layout = Layout::Unknown;
  out.semantic = "predictions";
  manifest.outputs.push_back(out);
  manifest.decoder.protocol = "yolov8_detect_raw";
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.class_selection = "best_class";
  manifest.decoder.scores_are_probabilities = true;
  for (const char* field : {"decoder.protocol", "decoder.box_format", "decoder.box_decode",
                              "decoder.box_channels", "decoder.channel_axis",
                              "decoder.class_selection", "decoder.scores_are_probabilities"}) {
    manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
  }
  manifest.nms.enabled = true;
  manifest.nms.confidence_threshold = 0.25f;
  manifest.nms.iou_threshold = 0.45f;
  manifest.nms.max_detections = 20;
  manifest.nms.pre_nms_top_k = 300;
  manifest.nms.class_aware = true;
  for (const char* field : {"nms.enabled", "nms.confidence_threshold", "nms.iou_threshold",
                              "nms.max_detections", "nms.pre_nms_top_k", "nms.class_aware"}) {
    manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
  }

  ModelConfig config;
  config.model_path = model_path;
  config.task = TaskType::Detection;
  config.backend = use_cuda_backend ? BackendKind::OnnxRuntimeCuda : BackendKind::OnnxRuntime;
  config.device = device;
  config.device_id = 0;
  config.cuda_cpu_ep_fallback = use_cuda_backend;
  config.manifest = manifest;
  return config;
}

// Holds either a host or a device pixel buffer for one 640x640 RGB image.
struct ImageBuffer {
  std::vector<std::uint8_t> host;
  std::uint8_t* device = nullptr;
  vie::DeviceType kind = vie::DeviceType::Cpu;
  ~ImageBuffer() {
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
    if (device != nullptr) cudaFree(device);
#endif
  }
};

ImageBuffer make_image_buffer(vie::DeviceType device) {
  ImageBuffer buf;
  buf.kind = device;
  buf.host.assign(640u * 640u * 3u, static_cast<std::uint8_t>(128));
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
  if (device == vie::DeviceType::Cuda) {
    if (cudaMalloc(reinterpret_cast<void**>(&buf.device), buf.host.size()) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed for image buffer");
    }
    if (cudaMemcpy(buf.device, buf.host.data(), buf.host.size(), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      throw std::runtime_error("cudaMemcpy H2D failed");
    }
  }
#else
  if (device == vie::DeviceType::Cuda) {
    throw std::runtime_error("CUDA not compiled in");
  }
#endif
  return buf;
}

std::vector<vie::ImageInput> build_batch(std::size_t n, const ImageBuffer& buf) {
  std::vector<vie::ImageInput> images;
  images.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    vie::ImageInput image;
    image.data = (buf.kind == vie::DeviceType::Cuda)
                     ? static_cast<void*>(const_cast<std::uint8_t*>(buf.device))
                     : static_cast<void*>(const_cast<std::uint8_t*>(buf.host.data()));
    image.width = 640;
    image.height = 640;
    image.channels = 3;
    image.color = vie::ColorOrder::Rgb;
    image.layout = vie::ImageLayout::Hwc;
    image.dtype = vie::DataType::Uint8;
    image.device = buf.kind;
    image.device_id = 0;
    images.push_back(image);
  }
  return images;
}

struct BatchResult {
  int batch_size = 0;
  std::size_t completed_batches = 0;
  std::size_t failed_batches = 0;
  double total_wall_seconds = 0.0;
  LatencyStats batch_latency;
};

BatchResult run_batch_size(vie::InferenceEngine& engine, const ImageBuffer& buf, int n) {
  BatchResult result;
  result.batch_size = n;
  auto images = build_batch(static_cast<std::size_t>(n), buf);

  for (int i = 0; i < kWarmup; ++i) {
    try {
      engine.infer_batch(images);
    } catch (const std::exception& e) {
      // Warmup rejection means the pipeline cannot serve this batch size at
      // all; report a clean failure and let main() continue with the next
      // size so Batch=1 CUDA still prints while Batch=2/4 CUDA is marked
      // Unsupported.
      std::cerr << "  warmup rejected at batch size " << n << ": " << e.what() << "\n";
      return result;
    }
  }

  std::vector<double> samples;
  samples.reserve(kTimed);
  const auto wall_start = std::chrono::steady_clock::now();
  for (int i = 0; i < kTimed; ++i) {
    const auto start = std::chrono::steady_clock::now();
    try {
      engine.infer_batch(images);
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
      ++result.completed_batches;
    } catch (const std::exception& e) {
      ++result.failed_batches;
      std::cerr << "  batch #" << i << " failed: " << e.what() << "\n";
      // A decoder that rejects Batch>1 (e.g. yolov8_detect_raw on CUDA in
      // Phase 9.1) fails on every iteration; bail out so the harness can
      // report "Unsupported" instead of a bogus sample set.
      if (i == 0 && result.completed_batches == 0) {
        result.total_wall_seconds = 0.0;
        return result;
      }
    }
  }
  const auto wall_end = std::chrono::steady_clock::now();
  result.total_wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
  if (!samples.empty()) {
    result.batch_latency = compute_stats(std::move(samples));
  }
  return result;
}

void print_batch_result(const BatchResult& r) {
  std::cout << "\n-- Batch Size " << r.batch_size << " --\n";
  std::cout << "  completed_batches : " << r.completed_batches << "\n";
  std::cout << "  failed_batches    : " << r.failed_batches << "\n";
  std::cout << "  completed_images  : " << (r.completed_batches * static_cast<std::size_t>(r.batch_size)) << "\n";
  std::cout << "  total_wall_seconds: " << std::fixed << std::setprecision(3)
            << r.total_wall_seconds << " s\n";
  if (r.completed_batches == 0) {
    std::cout << "  Status            : Unsupported or no data\n";
    return;
  }
  const double images_per_second =
      static_cast<double>(r.completed_batches) * static_cast<double>(r.batch_size) /
      r.total_wall_seconds;
  const double batches_per_second = static_cast<double>(r.completed_batches) / r.total_wall_seconds;
  std::cout << "  Images/s          : " << std::fixed << std::setprecision(2) << images_per_second << "\n";
  std::cout << "  Batches/s         : " << std::fixed << std::setprecision(2) << batches_per_second << "\n";
  std::cout << "  Batch Mean ms     : " << std::fixed << std::setprecision(2)
            << r.batch_latency.mean_ms << "\n";
  std::cout << "  Batch P50 ms      : " << std::fixed << std::setprecision(2)
            << r.batch_latency.p50_ms << "\n";
  std::cout << "  Batch P95 ms      : " << std::fixed << std::setprecision(2)
            << r.batch_latency.p95_ms << "\n";
  std::cout << "  Batch min ms      : " << std::fixed << std::setprecision(2)
            << r.batch_latency.min_ms << "\n";
  std::cout << "  Batch max ms      : " << std::fixed << std::setprecision(2)
            << r.batch_latency.max_ms << "\n";
  std::cout << "  Per-image mean ms : " << std::fixed << std::setprecision(2)
            << (r.batch_latency.mean_ms / static_cast<double>(r.batch_size)) << "\n";
  std::cout << "  (Per-image latency derived from batch_mean/batch_size; it is a throughput "
               "measure, not a single-request latency.)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <yolov8n_batch.onnx> [cpu|cuda] [1 2 4 8]\n";
    std::cerr << "  Requires a dynamic-batch YOLOv8 ONNX model with shape [batch, 3, 640, 640].\n";
    return 1;
  }
  std::string model_path = argv[1];
  std::string device_str = (argc >= 3) ? argv[2] : "cpu";
  std::vector<int> sizes = {1, 2, 4};
  if (argc >= 4) {
    sizes.clear();
    for (int i = 3; i < argc; ++i) sizes.push_back(std::stoi(argv[i]));
  }
  if (!file_exists(model_path)) {
    std::cerr << "model not found: " << model_path << "\n";
    return 1;
  }

  bool use_cuda = (device_str == "cuda");
  vie::DeviceType device = use_cuda ? vie::DeviceType::Cuda : vie::DeviceType::Cpu;
#if !defined(VIE_HAS_ONNXRUNTIME_CUDA)
  if (use_cuda) {
    std::cerr << "this build does not include CUDA; run with 'cpu' instead\n";
    return 1;
  }
#endif

  std::cout << "VIE Batch Benchmark - Phase 9.1\n";
  std::cout << "================================\n";
  std::cout << "Model : " << model_path << "\n";
  std::cout << "Device: " << device_str << "\n";
  std::cout << "Warmup: " << kWarmup << " batches, Timed: " << kTimed
            << " batches per size\n";
  std::cout << "Percentile: nearest-rank\n";

  try {
    vie::InferenceEngine engine;
    engine.load(build_config(model_path, device, use_cuda));
    ImageBuffer buf = make_image_buffer(device);
    for (int n : sizes) {
      BatchResult r = run_batch_size(engine, buf, n);
      print_batch_result(r);
    }
    std::cout << "\nBenchmark complete.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\nFATAL: " << e.what() << "\n";
    return 1;
  }
}
