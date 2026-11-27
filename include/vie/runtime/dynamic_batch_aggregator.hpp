#pragma once

#include "vie/runtime/batch_aggregation_config.hpp"
#include "vie/runtime/inference_request.hpp"
#include "vie/runtime/request_queue.hpp"
#include "vie/runtime/runtime_metrics.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vie {

class ModelInstance;

// A formed batch of requests ready for execution. All requests share the
// same ModelInstance and are compatible for batch inference.
struct BatchWorkItem {
  std::vector<InferenceRequest> requests;
  std::shared_ptr<ModelInstance> model;  // Same for all requests
  std::chrono::steady_clock::time_point created_at;
};

// Dynamically aggregates individual inference requests into batches.
// Groups requests by ModelInstance identity and effective inference config
// (ResolvedInferenceConfig::fingerprint()), so two requests with different
// thresholds never share a batch even when they target the same model.
// Flushes a batch when:
//   - max_batch_size is reached (immediate flush),
//   - max_wait_time elapses since the oldest request in the group (timeout flush),
//   - shutdown is initiated (drain flush).
//
// Thread safety: This class is designed to run on a single aggregator thread.
// The add/wait/flush interface is NOT thread-safe for concurrent callers.
class DynamicBatchAggregator {
 public:
  DynamicBatchAggregator(BatchAggregationConfig config,
                         std::shared_ptr<MetricsCollector> metrics);

  // Add a request to the appropriate pending group. If the group reaches
  // max_batch_size, returns the formed batch. Otherwise returns nullopt.
  std::optional<BatchWorkItem> add(InferenceRequest request);

  // Check all pending groups and flush any whose deadline has expired.
  // Returns the first ready batch, or nullopt if none are ready.
  std::optional<BatchWorkItem> flush_expired();

  // Flush ALL pending groups regardless of deadline. Used during shutdown.
  // Returns all formed batches (one per non-empty group).
  std::vector<BatchWorkItem> flush_all();

  // Compute the earliest deadline among all pending groups.
  // Returns steady_clock::time_point::max() if no groups are pending.
  std::chrono::steady_clock::time_point earliest_deadline() const;

  // Total number of requests currently held in pending groups.
  std::size_t pending_count() const;

  // Number of pending groups.
  std::size_t group_count() const;

  // Whether aggregation is enabled.
  bool enabled() const noexcept { return config_.enabled; }

  // Effective max batch size (1 if disabled, or for CUDA models).
  std::size_t effective_max_batch_size(bool is_cuda) const;

  const BatchAggregationConfig& config() const noexcept { return config_; }

 private:
  // One pending batch group, keyed by ModelInstance identity and effective
  // inference config fingerprint.
  struct PendingGroup {
    std::shared_ptr<ModelInstance> model;
    std::string config_key;
    std::vector<InferenceRequest> requests;
    std::chrono::steady_clock::time_point deadline;  // When to flush at latest
  };

  // Find or create a group for the given model instance and effective config.
  // Returns nullptr if adding would exceed max_pending_requests.
  PendingGroup* find_or_create_group(const std::shared_ptr<ModelInstance>& model,
                                     const std::string& config_key);

  // Pop a group out of the pending map (for flush).
  BatchWorkItem extract_group(PendingGroup& group);

  BatchAggregationConfig config_;
  std::shared_ptr<MetricsCollector> metrics_;

  // Pending groups keyed by raw ModelInstance pointer (stable identity).
  // We use a vector since the number of concurrent groups is small.
  std::vector<PendingGroup> groups_;
};

}  // namespace vie
