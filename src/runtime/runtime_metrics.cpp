#include "vie/runtime/runtime_metrics.hpp"

namespace vie {

void MetricsCollector::record_inference(const std::string& model_id, double time_ms, bool success) {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.total_inferences++;
  if (!success) {
    metrics_.failed_inferences++;
  }
  metrics_.total_inference_time_ms += time_ms;
  metrics_.per_model_inference_count[model_id]++;
  metrics_.per_model_inference_time_ms[model_id] += time_ms;
}

void MetricsCollector::record_preprocess(double time_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.total_preprocess_calls++;
  metrics_.total_preprocess_time_ms += time_ms;
}

void MetricsCollector::record_model_load() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.total_models_loaded++;
}

void MetricsCollector::record_model_unload() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.total_models_unloaded++;
}

RuntimeMetrics MetricsCollector::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return metrics_;
}

}  // namespace vie
