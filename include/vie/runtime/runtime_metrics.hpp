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

  // Phase 9.2.2: Batch aggregation metrics
  std::size_t total_batches = 0;               // Total execution units (includes batch_size=1 single infers)
  std::size_t total_single_infer_calls = 0;    // Subset of total_batches where batch_size==1
  std::size_t total_aggregated_requests = 0;   // Sum of all batch sizes across execution units
  std::size_t batch_full_flush_count = 0;      // Batches flushed due to max_batch_size
  std::size_t batch_timeout_flush_count = 0;   // Batches flushed due to timeout
  std::size_t batch_shutdown_flush_count = 0;  // Batches flushed during shutdown
  std::size_t batch_execution_failures = 0;    // Failed infer_batch() calls
  std::size_t max_batch_size_achieved = 0;     // Largest batch actually executed
  std::map<int, std::size_t> batch_size_histogram;  // batch_size -> count

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
    // Phase 9.2.2
    total_batches = 0;
    total_single_infer_calls = 0;
    total_aggregated_requests = 0;
    batch_full_flush_count = 0;
    batch_timeout_flush_count = 0;
    batch_shutdown_flush_count = 0;
    batch_execution_failures = 0;
    max_batch_size_achieved = 0;
    batch_size_histogram.clear();
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

  // Phase 9.2.2: Batch aggregation metrics
  void record_batch_executed(std::size_t batch_size);
  void record_single_infer();
  void record_batch_full_flush();
  void record_batch_timeout_flush();
  void record_batch_shutdown_flush();
  void record_batch_execution_failure();

 private:
  mutable std::mutex mutex_;
  RuntimeMetrics metrics_;
};

}  // namespace vie
