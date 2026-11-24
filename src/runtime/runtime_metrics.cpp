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

// Phase 9.2.1: Async request metrics
void MetricsCollector::record_request_submitted() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.submitted_requests++;
}

void MetricsCollector::record_request_accepted() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.accepted_requests++;
}

void MetricsCollector::record_request_rejected() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.rejected_requests++;
}

void MetricsCollector::record_request_completed(double queue_wait_ms, double execution_ms, double e2e_ms, bool success) {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.completed_requests++;
  if (!success) {
    metrics_.failed_requests++;
  }
  metrics_.total_queue_wait_time_ms += queue_wait_ms;
  metrics_.total_execution_time_ms += execution_ms;
  metrics_.total_e2e_time_ms += e2e_ms;
}

void MetricsCollector::record_queue_size(std::size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.current_queue_size = size;
  if (size > metrics_.max_queue_size) {
    metrics_.max_queue_size = size;
  }
}

// Phase 9.2.2: Batch aggregation metrics
void MetricsCollector::record_batch_executed(std::size_t batch_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.total_batches++;
  metrics_.total_aggregated_requests += batch_size;
  if (batch_size > metrics_.max_batch_size_achieved) {
    metrics_.max_batch_size_achieved = batch_size;
  }
  metrics_.batch_size_histogram[static_cast<int>(batch_size)]++;
}

void MetricsCollector::record_single_infer() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.total_single_infer_calls++;
}

void MetricsCollector::record_batch_full_flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.batch_full_flush_count++;
}

void MetricsCollector::record_batch_timeout_flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.batch_timeout_flush_count++;
}

void MetricsCollector::record_batch_shutdown_flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.batch_shutdown_flush_count++;
}

void MetricsCollector::record_batch_execution_failure() {
  std::lock_guard<std::mutex> lock(mutex_);
  metrics_.batch_execution_failures++;
}

}  // namespace vie
