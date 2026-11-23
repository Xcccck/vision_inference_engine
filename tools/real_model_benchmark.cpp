// Real Model Benchmark - Phase 8.3
// Uses actual ONNX models with VIE Runtime

#include "vie/runtime/runtime.hpp"
#include "vie/model/model_config.hpp"
#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/core/tensor.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <vector>

namespace {

struct BenchmarkResult {
  std::string scenario;
  std::string backend_type;
  int worker_count;
  int instance_count;
  int request_count;
  int warmup_count;
  double total_time_ms;
  double throughput_rps;
  double avg_latency_ms;
  double p50_latency_ms;
  double p95_latency_ms;
};

void print_result(const BenchmarkResult& r) {
  std::cout << "\n=== " << r.scenario << " ===\n";
  std::cout << "Backend: " << r.backend_type << "\n";
  std::cout << "Workers: " << r.worker_count << "\n";
  std::cout << "Instances: " << r.instance_count << "\n";
  std::cout << "Requests: " << r.request_count << "\n";
  std::cout << "Warmup: " << r.warmup_count << "\n";
  std::cout << "Total time: " << std::fixed << std::setprecision(2) << r.total_time_ms << " ms\n";
  std::cout << "Throughput: " << std::fixed << std::setprecision(2) << r.throughput_rps << " req/s\n";
  std::cout << "Avg latency: " << std::fixed << std::setprecision(3) << r.avg_latency_ms << " ms\n";
  std::cout << "P50 latency: " << std::fixed << std::setprecision(3) << r.p50_latency_ms << " ms\n";
  std::cout << "P95 latency: " << std::fixed << std::setprecision(3) << r.p95_latency_ms << " ms\n";
}

double percentile(std::vector<double>& data, double p) {
  if (data.empty()) return 0.0;
  std::sort(data.begin(), data.end());
  double index = p / 100.0 * (data.size() - 1);
  size_t lower = static_cast<size_t>(index);
  size_t upper = lower + 1;
  if (upper >= data.size()) upper = data.size() - 1;
  double weight = index - lower;
  return data[lower] * (1.0 - weight) + data[upper] * weight;
}

vie::ModelManifest create_manifest(const std::string& model_path) {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(model_path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  auto manifest = vie::ManifestGenerator{}.generate(report, result, model_path);
  
  // Fill in required preprocess fields
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
  
  // Fill in detection policy
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

vie::ImageInput create_dummy_image(std::vector<uint8_t>& pixels, int width, int height) {
  pixels.resize(width * height * 3);
  // Fill with dummy data
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<uint8_t>(i % 256);
  }
  
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [cpu|cuda]\n";
    return 1;
  }

  std::string model_path = argv[1];
  std::string device_str = (argc >= 3) ? argv[2] : "cpu";
  
  vie::DeviceType device = vie::DeviceType::Cpu;
  std::string backend_type = "OnnxRuntime CPU";
  
  if (device_str == "cuda") {
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
    device = vie::DeviceType::Cuda;
    backend_type = "OnnxRuntime CUDA";
#else
    std::cerr << "CUDA backend not available in this build\n";
    return 1;
#endif
  }

  std::cout << "VIE Real Model Benchmark - Phase 8.3\n";
  std::cout << "=====================================\n";
  std::cout << "Model: " << model_path << "\n";
  std::cout << "Device: " << device_str << "\n";
  std::cout << "Backend: " << backend_type << "\n";

  try {
    // Create manifest
    std::cout << "\nCreating manifest...\n";
    auto manifest = create_manifest(model_path);
    std::cout << "Manifest created successfully\n";

    // Create dummy image (640x640 RGB)
    std::vector<uint8_t> pixels;
    auto image = create_dummy_image(pixels, 640, 640);
    std::cout << "Dummy image created (640x640 RGB)\n";

    // Scenario A: Single instance, single thread
    {
      std::cout << "\n\n=== Scenario A: Single instance, single thread ===\n";
      
      vie::RuntimeConfig config;
      config.thread_pool_size = 1;
      config.default_device = device;
      vie::Runtime runtime(config);
      
      vie::ModelConfig model_config;
      model_config.model_path = model_path;
      model_config.device = device;
      model_config.device_id = 0;
      model_config.manifest = manifest;
      
      auto model = runtime.models().load("model_a", model_config);
      
      // Warmup
      const int warmup = 5;
      for (int i = 0; i < warmup; ++i) {
        model->infer(image);
      }
      
      // Benchmark
      const int requests = 50;
      std::vector<double> latencies;
      
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < requests; ++i) {
        auto req_start = std::chrono::steady_clock::now();
        model->infer(image);
        auto req_end = std::chrono::steady_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(req_end - req_start).count());
      }
      auto end = std::chrono::steady_clock::now();
      
      double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
      
      BenchmarkResult result;
      result.scenario = "A: Single instance, single thread";
      result.backend_type = backend_type;
      result.worker_count = 1;
      result.instance_count = 1;
      result.request_count = requests;
      result.warmup_count = warmup;
      result.total_time_ms = total_ms;
      result.throughput_rps = requests / (total_ms / 1000.0);
      result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
      result.p50_latency_ms = percentile(latencies, 50);
      result.p95_latency_ms = percentile(latencies, 95);
      
      print_result(result);
    }

    // Scenario B: Single instance, multiple threads
    {
      std::cout << "\n\n=== Scenario B: Single instance, 4 threads ===\n";
      
      vie::RuntimeConfig config;
      config.thread_pool_size = 4;
      config.default_device = device;
      vie::Runtime runtime(config);
      
      vie::ModelConfig model_config;
      model_config.model_path = model_path;
      model_config.device = device;
      model_config.device_id = 0;
      model_config.manifest = manifest;
      
      auto model = runtime.models().load("model_b", model_config);
      
      // Warmup
      const int warmup = 5;
      for (int i = 0; i < warmup; ++i) {
        model->infer(image);
      }
      
      // Benchmark
      const int requests = 100;
      const int threads = 4;
      std::vector<double> latencies;
      std::mutex lat_mutex;
      
      auto start = std::chrono::steady_clock::now();
      
      std::vector<std::future<void>> futures;
      for (int t = 0; t < threads; ++t) {
        futures.push_back(runtime.thread_pool().submit([&, t] {
          for (int i = t * (requests / threads); i < (t + 1) * (requests / threads); ++i) {
            auto req_start = std::chrono::steady_clock::now();
            model->infer(image);
            auto req_end = std::chrono::steady_clock::now();
            
            double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
            std::lock_guard<std::mutex> lock(lat_mutex);
            latencies.push_back(latency);
          }
        }));
      }
      
      for (auto& f : futures) {
        f.get();
      }
      
      auto end = std::chrono::steady_clock::now();
      double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
      
      BenchmarkResult result;
      result.scenario = "B: Single instance, 4 threads";
      result.backend_type = backend_type;
      result.worker_count = threads;
      result.instance_count = 1;
      result.request_count = requests;
      result.warmup_count = warmup;
      result.total_time_ms = total_ms;
      result.throughput_rps = requests / (total_ms / 1000.0);
      result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
      result.p50_latency_ms = percentile(latencies, 50);
      result.p95_latency_ms = percentile(latencies, 95);
      
      print_result(result);
    }

    // Scenario C: Multiple instances (2)
    {
      std::cout << "\n\n=== Scenario C: 2 instances, 4 threads ===\n";
      
      vie::RuntimeConfig config;
      config.thread_pool_size = 4;
      config.default_device = device;
      vie::Runtime runtime(config);
      
      vie::ModelConfig model_config;
      model_config.model_path = model_path;
      model_config.device = device;
      model_config.device_id = 0;
      model_config.manifest = manifest;
      
      auto model1 = runtime.models().load("model_c1", model_config);
      auto model2 = runtime.models().load("model_c2", model_config);
      
      // Warmup
      const int warmup = 5;
      for (int i = 0; i < warmup; ++i) {
        model1->infer(image);
        model2->infer(image);
      }
      
      // Benchmark
      const int requests = 100;
      const int threads = 4;
      std::vector<double> latencies;
      std::mutex lat_mutex;
      
      auto start = std::chrono::steady_clock::now();
      
      std::vector<std::future<void>> futures;
      for (int t = 0; t < threads; ++t) {
        futures.push_back(runtime.thread_pool().submit([&, t] {
          auto model = (t % 2 == 0) ? model1 : model2;
          for (int i = t * (requests / threads); i < (t + 1) * (requests / threads); ++i) {
            auto req_start = std::chrono::steady_clock::now();
            model->infer(image);
            auto req_end = std::chrono::steady_clock::now();
            
            double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
            std::lock_guard<std::mutex> lock(lat_mutex);
            latencies.push_back(latency);
          }
        }));
      }
      
      for (auto& f : futures) {
        f.get();
      }
      
      auto end = std::chrono::steady_clock::now();
      double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
      
      BenchmarkResult result;
      result.scenario = "C: 2 instances, 4 threads";
      result.backend_type = backend_type;
      result.worker_count = threads;
      result.instance_count = 2;
      result.request_count = requests;
      result.warmup_count = warmup;
      result.total_time_ms = total_ms;
      result.throughput_rps = requests / (total_ms / 1000.0);
      result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
      result.p50_latency_ms = percentile(latencies, 50);
      result.p95_latency_ms = percentile(latencies, 95);
      
      print_result(result);
    }

    std::cout << "\n\nBenchmark complete.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
