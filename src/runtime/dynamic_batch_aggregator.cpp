#include "vie/runtime/dynamic_batch_aggregator.hpp"

#include "vie/core/error.hpp"
#include "vie/runtime/model_instance.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace vie {

DynamicBatchAggregator::DynamicBatchAggregator(BatchAggregationConfig config,
                                               std::shared_ptr<MetricsCollector> metrics)
    : config_(config), metrics_(std::move(metrics)) {
  if (config_.max_batch_size < 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "BatchAggregationConfig.max_batch_size must be >= 1");
  }
  if (config_.max_pending_requests < 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "BatchAggregationConfig.max_pending_requests must be >= 1");
  }
  // Negative durations are clamped to zero
  if (config_.max_wait_time.count() < 0) {
    config_.max_wait_time = std::chrono::milliseconds{0};
  }
}

std::size_t DynamicBatchAggregator::effective_max_batch_size(bool is_cuda) const {
  if (!config_.enabled) return 1;
  // CUDA models: force batch_size=1 (CUDA decoder rejects Batch>1)
  if (is_cuda) return 1;
  return config_.max_batch_size;
}

DynamicBatchAggregator::PendingGroup*
DynamicBatchAggregator::find_or_create_group(const std::shared_ptr<ModelInstance>& model,
                                             const std::string& config_key) {
  // Check total pending count against limit
  if (pending_count() >= config_.max_pending_requests) {
    return nullptr;  // Backpressure: too many pending requests
  }

  // Look for an existing group with the same ModelInstance and effective
  // config. Same model + different policy must stay in separate batches.
  for (auto& group : groups_) {
    if (group.model.get() == model.get() && group.config_key == config_key) {
      return &group;
    }
  }

  // Create a new group
  groups_.push_back(PendingGroup{});
  auto& group = groups_.back();
  group.model = model;
  group.config_key = config_key;
  group.deadline = std::chrono::steady_clock::now() + config_.max_wait_time;
  return &group;
}

std::optional<BatchWorkItem> DynamicBatchAggregator::add(InferenceRequest request) {
  auto* group = find_or_create_group(request.model, request.config_key);
  if (!group) {
    // Backpressure: cannot accept more pending requests.
    // This should not normally happen because the RequestQueue is already bounded,
    // but we handle it by completing the request with an error.
    try {
      request.promise.set_exception(std::make_exception_ptr(
          VieError(ErrorCode::RequestQueueFull,
                   "Aggregator pending limit reached (" +
                       std::to_string(config_.max_pending_requests) + ")")));
    } catch (...) {}
    if (metrics_) {
      metrics_->record_request_rejected();
    }
    return std::nullopt;
  }

  group->requests.push_back(std::move(request));

  // Determine effective max batch size based on device
  bool is_cuda = false;
  if (!group->requests.empty() && group->requests.back().image.device == DeviceType::Cuda) {
    is_cuda = true;
  }
  const std::size_t max_batch = effective_max_batch_size(is_cuda);

  // Check if the group has reached max batch size
  if (group->requests.size() >= max_batch) {
    BatchWorkItem batch = extract_group(*group);
    // Remove the empty group
    groups_.erase(std::remove_if(groups_.begin(), groups_.end(),
                                  [](const PendingGroup& g) { return g.requests.empty(); }),
                  groups_.end());
    if (metrics_) {
      metrics_->record_batch_full_flush();
    }
    return batch;
  }

  return std::nullopt;
}

std::optional<BatchWorkItem> DynamicBatchAggregator::flush_expired() {
  const auto now = std::chrono::steady_clock::now();

  for (auto& group : groups_) {
    if (!group.requests.empty() && now >= group.deadline) {
      BatchWorkItem batch = extract_group(group);
      if (metrics_) {
        metrics_->record_batch_timeout_flush();
      }
      // Remove the empty group
      groups_.erase(std::remove_if(groups_.begin(), groups_.end(),
                                    [](const PendingGroup& g) { return g.requests.empty(); }),
                    groups_.end());
      return batch;
    }
  }
  return std::nullopt;
}

std::vector<BatchWorkItem> DynamicBatchAggregator::flush_all() {
  std::vector<BatchWorkItem> batches;
  batches.reserve(groups_.size());

  for (auto& group : groups_) {
    if (!group.requests.empty()) {
      batches.push_back(extract_group(group));
      if (metrics_) {
        metrics_->record_batch_shutdown_flush();
      }
    }
  }
  groups_.clear();
  return batches;
}

std::chrono::steady_clock::time_point DynamicBatchAggregator::earliest_deadline() const {
  auto earliest = std::chrono::steady_clock::time_point::max();
  for (const auto& group : groups_) {
    if (!group.requests.empty() && group.deadline < earliest) {
      earliest = group.deadline;
    }
  }
  return earliest;
}

std::size_t DynamicBatchAggregator::pending_count() const {
  std::size_t total = 0;
  for (const auto& group : groups_) {
    total += group.requests.size();
  }
  return total;
}

std::size_t DynamicBatchAggregator::group_count() const {
  return groups_.size();
}

BatchWorkItem DynamicBatchAggregator::extract_group(PendingGroup& group) {
  BatchWorkItem batch;
  batch.model = group.model;
  batch.requests = std::move(group.requests);
  batch.created_at = std::chrono::steady_clock::now();
  group.requests.clear();
  return batch;
}

}  // namespace vie
