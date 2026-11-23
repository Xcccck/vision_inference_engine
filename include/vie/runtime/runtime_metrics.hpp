#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace vie {

// Metrics collected during runtime
struct RuntimeMetrics {
  std::size_t total_inferences = 0;
  std::size_t failed_inferences = 0;
  std::size_t total_preprocess_calls = 0;
  std::size_t total_models_loaded = 0;
  std::size_t total_models_unloaded = 0;
  double total_inference_time_ms = 0.0;
  double total_preprocess_time_ms = 0.0;
  std::map<std::string, std::size_t> per_model_inference_count;
  std::map<std::string, double> per_model_inference_time_ms;

  void reset() {
    total_inferences = 0;
    failed_inferences = 0;
    total_preprocess_calls = 0;
    total_models_loaded = 0;
    total_models_unloaded = 0;
    total_inference_time_ms = 0.0;
    total_preprocess_time_ms = 0.0;
    per_model_inference_count.clear();
    per_model_inference_time_ms.clear();
  }
};

// Thread-safe metrics collector
class MetricsCollector {
 public:
  void record_inference(const std::string& model_id, double time_ms, bool success);
  void record_preprocess(double time_ms);
  void record_model_load();
  void record_model_unload();
  RuntimeMetrics snapshot() const;

 private:
  mutable std::mutex mutex_;
  RuntimeMetrics metrics_;
};

}  // namespace vie
