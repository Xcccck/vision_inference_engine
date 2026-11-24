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

  // Phase 9.2.1: Async request metrics
  std::size_t submitted_requests = 0;
  std::size_t accepted_requests = 0;
  std::size_t rejected_requests = 0;
  std::size_t completed_requests = 0;
  std::size_t failed_requests = 0;
  double total_queue_wait_time_ms = 0.0;
  double total_execution_time_ms = 0.0;
  double total_e2e_time_ms = 0.0;
  std::size_t current_queue_size = 0;
  std::size_t max_queue_size = 0;

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
    // Phase 9.2.1
    submitted_requests = 0;
    accepted_requests = 0;
    rejected_requests = 0;
    completed_requests = 0;
    failed_requests = 0;
    total_queue_wait_time_ms = 0.0;
    total_execution_time_ms = 0.0;
    total_e2e_time_ms = 0.0;
    current_queue_size = 0;
    max_queue_size = 0;
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

  // Phase 9.2.1: Async request metrics
  void record_request_submitted();
  void record_request_accepted();
  void record_request_rejected();
  void record_request_completed(double queue_wait_ms, double execution_ms, double e2e_ms, bool success);
  void record_queue_size(std::size_t size);

 private:
  mutable std::mutex mutex_;
  RuntimeMetrics metrics_;
};

}  // namespace vie
