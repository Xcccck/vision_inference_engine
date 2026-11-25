#pragma once

#include "vie/runtime/dynamic_batch_aggregator.hpp"  // BatchWorkItem
#include "vie/runtime/execution_resource.hpp"
#include "vie/runtime/execution_scheduling_config.hpp"
#include "vie/runtime/runtime_metrics.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <map>
#include <memory>
#include <mutex>

namespace vie {

class ExecutionResourceManager;

// Phase 9.3: RAII handle for one held resource permit. The permit is acquired
// by ExecutionResourceManager::pop() and released exactly once, when this
// object is destroyed (normal return, exception, or worker unwind). This makes
// it impossible to leak a permit on any control-flow path and forbids a manual
// release racing with the destructor.
class ResourceLease {
 public:
  ResourceLease() = default;
  ResourceLease(ResourceLease&& other) noexcept;
  ResourceLease& operator=(ResourceLease&& other) noexcept;
  ~ResourceLease();

  ResourceLease(const ResourceLease&) = delete;
  ResourceLease& operator=(const ResourceLease&) = delete;

  bool held() const noexcept { return manager_ != nullptr; }
  const ExecutionResource& resource() const noexcept { return resource_; }

 private:
  friend class ExecutionResourceManager;
  ResourceLease(ExecutionResourceManager* manager, ExecutionResource resource)
      : manager_(manager), resource_(resource) {}

  ExecutionResourceManager* manager_ = nullptr;
  ExecutionResource resource_{};
};

// A batch handed to a worker together with the permit reserved for it. The
// lease releases the permit when this object leaves scope (after execute_batch).
struct ScheduledBatch {
  BatchWorkItem item;
  ResourceLease lease;
};

// Phase 9.3: Device-aware execution scheduler and admission controller.
//
// Responsibilities:
//   * Route formed batches into a per-ExecutionResource lane (CPU, each CUDA
//     device id) instead of one shared FIFO.
//   * Enforce an independent concurrency limit per resource:
//       - CUDA device N: cuda_max_concurrent_per_device (default 1).
//       - CPU: cpu_max_concurrent, or unbounded when 0 (worker-count governed).
//   * Prevent head-of-line blocking: pop() only hands out a batch whose lane
//     currently has a free permit. A worker never parks on a saturated GPU lane
//     while other lanes still have eligible work, so a full GPU device cannot
//     starve CPU requests or a different GPU device.
//   * Preserve backpressure: the total number of staged (not-yet-started)
//     batches across all lanes is bounded by one shared capacity.
//   * Never silently admit an Unknown resource.
//
// It is a resource gate only. It does not create CUDA streams, session pools,
// or claim any real GPU kernel concurrency.
class ExecutionResourceManager {
 public:
  ExecutionResourceManager(ExecutionSchedulingConfig config, std::size_t fallback_capacity,
                           std::shared_ptr<MetricsCollector> metrics = nullptr);

  ExecutionResourceManager(const ExecutionResourceManager&) = delete;
  ExecutionResourceManager& operator=(const ExecutionResourceManager&) = delete;

  // Validates the configuration; throws VieError(InvalidArgument) on an illegal
  // cuda_max_concurrent_per_device (== 0). Called from the constructor and
  // exposed so config-only tests can assert the failure.
  static void validate(const ExecutionSchedulingConfig& config);

  // Idempotently create a lane for the resource. Unknown throws.
  void register_resource(const ExecutionResource& resource);

  // Whether a lane exists for the resource.
  bool knows(const ExecutionResource& resource) const;

  // Whether shutdown() has been called (no new batches will be accepted).
  bool is_shutdown() const;

  // Block until the resource lane has room, then stage the batch. Returns false
  // if the manager has been shut down before the batch was accepted.
  bool push(const ExecutionResource& resource, BatchWorkItem item);

  // Block until some lane has a staged batch AND a free permit, then reserve a
  // permit and return that batch (with its lease). Returns false once the
  // manager is shut down and every lane is fully drained.
  bool pop(ScheduledBatch& out);

  // Return a permit (called by ResourceLease's destructor).
  void release(const ExecutionResource& resource);

  // Stop accepting pushes; wake all workers so they drain remaining staged
  // batches and then exit. Safe to call from multiple threads.
  void shutdown();

  // ---- Observability ---------------------------------------------------------
  std::size_t cap(const ExecutionResource& resource) const;
  std::size_t active(const ExecutionResource& resource) const;
  std::size_t max_active(const ExecutionResource& resource) const;
  std::size_t queued(const ExecutionResource& resource) const;  // staged, waiting
  std::size_t total_queued() const;
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  struct Lane {
    std::deque<BatchWorkItem> ready;
    std::size_t active = 0;
    std::size_t cap = 1;
    std::size_t max_active = 0;
  };

  // Create a lane assuming mutex_ is already held. Unknown throws.
  void register_resource_locked(const ExecutionResource& resource);

  ExecutionSchedulingConfig config_;
  std::size_t capacity_;
  std::shared_ptr<MetricsCollector> metrics_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool shutdown_ = false;
  std::size_t total_staged_ = 0;  // bounded by capacity_ (backpressure)
  std::map<ExecutionResource, Lane, bool (*)(const ExecutionResource&, const ExecutionResource&)>
      lanes_{[](const ExecutionResource& a, const ExecutionResource& b) {
        if (a.type != b.type) return a.type < b.type;
        return a.device_id < b.device_id;
      }};
};

}  // namespace vie
