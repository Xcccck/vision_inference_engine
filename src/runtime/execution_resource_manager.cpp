#include "vie/runtime/execution_resource_manager.hpp"

#include "vie/core/error.hpp"

#include <chrono>
#include <utility>

namespace vie {

// =============================================================================
// ResourceLease (RAII)
// =============================================================================

ResourceLease::ResourceLease(ResourceLease&& other) noexcept
    : manager_(other.manager_), resource_(other.resource_) {
  other.manager_ = nullptr;
}

ResourceLease& ResourceLease::operator=(ResourceLease&& other) noexcept {
  if (this != &other) {
    if (manager_ != nullptr) {
      manager_->release(resource_);
    }
    manager_ = other.manager_;
    resource_ = other.resource_;
    other.manager_ = nullptr;
  }
  return *this;
}

ResourceLease::~ResourceLease() {
  if (manager_ != nullptr) {
    manager_->release(resource_);
    manager_ = nullptr;
  }
}

// =============================================================================
// ExecutionResourceManager
// =============================================================================

ExecutionResourceManager::ExecutionResourceManager(ExecutionSchedulingConfig config,
                                                   std::size_t fallback_capacity,
                                                   std::shared_ptr<MetricsCollector> metrics)
    : config_(config), metrics_(std::move(metrics)) {
  validate(config_);
  std::size_t cap = config_.lane_capacity;
  if (cap == 0) {
    cap = fallback_capacity;
  }
  if (cap == 0) {
    cap = 1;  // Defensive: never a zero-capacity staging pool.
  }
  capacity_ = cap;
}

void ExecutionResourceManager::validate(const ExecutionSchedulingConfig& config) {
  if (config.cuda_max_concurrent_per_device == 0) {
    throw VieError(ErrorCode::InvalidArgument,
                   "cuda_max_concurrent_per_device must be >= 1 (0 does not mean unlimited)");
  }
}

void ExecutionResourceManager::register_resource(const ExecutionResource& resource) {
  std::lock_guard<std::mutex> lock(mutex_);
  register_resource_locked(resource);
}

void ExecutionResourceManager::register_resource_locked(const ExecutionResource& resource) {
  if (!resource.valid()) {
    throw VieError(ErrorCode::Unsupported,
                   "cannot schedule execution on an Unknown execution resource");
  }
  if (lanes_.find(resource) != lanes_.end()) {
    return;  // Idempotent.
  }
  Lane lane;
  if (resource.type == ExecutionResourceType::Cpu) {
    // 0 means unbounded: CPU parallelism stays governed by the worker pool and
    // the per-ModelInstance mutex, preserving pre-Phase-9.3 behavior.
    lane.cap = (config_.cpu_max_concurrent == 0) ? SIZE_MAX : config_.cpu_max_concurrent;
  } else {  // Cuda
    lane.cap = config_.cuda_max_concurrent_per_device;
  }
  lanes_.emplace(resource, std::move(lane));
}

bool ExecutionResourceManager::knows(const ExecutionResource& resource) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return lanes_.find(resource) != lanes_.end();
}

bool ExecutionResourceManager::is_shutdown() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return shutdown_;
}

bool ExecutionResourceManager::push(const ExecutionResource& resource, BatchWorkItem item) {
  // Unknown resources are rejected before touching any state.
  if (!resource.valid()) {
    return false;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (lanes_.find(resource) == lanes_.end()) {
    // Create the lane lazily with the correct configured cap so an unbounded
    // permit is never silently granted for a CUDA device.
    register_resource_locked(resource);
  }
  // Shared-pool backpressure: block while the total staged count reaches the
  // configured capacity. Splitting into lanes must not admit more work than a
  // single bounded queue would have.
  cv_.wait(lock, [this] { return shutdown_ || total_staged_ < capacity_; });
  if (shutdown_) {
    return false;
  }
  lanes_[resource].ready.push_back(std::move(item));
  ++total_staged_;
  cv_.notify_all();  // Wake a worker that can take this batch.
  return true;
}

bool ExecutionResourceManager::pop(ScheduledBatch& out) {
  // Release any permit the caller's ScheduledBatch already holds BEFORE taking
  // mutex_. ResourceLease's move-assignment calls release(), which locks
  // mutex_; doing that while already holding it (below, when we hand back the
  // new lease) would self-deadlock on the non-recursive mutex. Resetting here
  // makes pop() safe to call repeatedly on a reused ScheduledBatch and frees
  // the prior permit so the wait predicate can admit the next batch.
  out.lease = ResourceLease{};
  std::unique_lock<std::mutex> lock(mutex_);
  const auto admission_start = std::chrono::steady_clock::now();

  // Wait for any lane that has a staged batch and a free permit. Because we
  // only require *some* eligible lane, a worker is never parked on a saturated
  // GPU lane while CPU (or another device) work is available: this is what
  // prevents head-of-line blocking across resources.
  cv_.wait(lock, [this] {
    if (shutdown_ && total_staged_ == 0) {
      return true;  // Drain complete.
    }
    for (const auto& [res, lane] : lanes_) {
      if (!lane.ready.empty() && lane.active < lane.cap) {
        return true;
      }
    }
    return false;
  });

  Lane* chosen = nullptr;
  ExecutionResource chosen_res{};
  for (auto& [res, lane] : lanes_) {
    if (!lane.ready.empty() && lane.active < lane.cap) {
      chosen = &lane;
      chosen_res = res;
      break;  // lanes_ is ordered Cpu<Cuda by device_id; deterministic pick.
    }
  }

  if (chosen == nullptr) {
    // Only reachable when shutdown_ && total_staged_ == 0.
    return false;
  }

  out.item = std::move(chosen->ready.front());
  chosen->ready.pop_front();
  --total_staged_;
  ++chosen->active;
  if (chosen->active > chosen->max_active) {
    chosen->max_active = chosen->active;
  }
  out.lease = ResourceLease(this, chosen_res);

  const double wait_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - admission_start)
          .count();
  // A staged slot freed; unblock a waiting producer.
  cv_.notify_all();
  if (metrics_) {
    // Release our mutex before touching the collector to avoid coupling the
    // scheduler lock with the metrics lock.
    lock.unlock();
    metrics_->record_resource_admission(chosen_res.key(), wait_ms);
  }
  return true;
}

void ExecutionResourceManager::release(const ExecutionResource& resource) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = lanes_.find(resource);
  if (it == lanes_.end() || it->second.active == 0) {
    return;  // Defensive: releasing an unheld permit.
  }
  --it->second.active;
  cv_.notify_all();  // A permit freed; a parked worker may now proceed.
}

void ExecutionResourceManager::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  shutdown_ = true;
  cv_.notify_all();  // Wake producers (they will see shutdown and return false).
}

std::size_t ExecutionResourceManager::cap(const ExecutionResource& resource) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = lanes_.find(resource);
  return (it == lanes_.end()) ? 0 : it->second.cap;
}

std::size_t ExecutionResourceManager::active(const ExecutionResource& resource) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = lanes_.find(resource);
  return (it == lanes_.end()) ? 0 : it->second.active;
}

std::size_t ExecutionResourceManager::max_active(const ExecutionResource& resource) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = lanes_.find(resource);
  return (it == lanes_.end()) ? 0 : it->second.max_active;
}

std::size_t ExecutionResourceManager::queued(const ExecutionResource& resource) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = lanes_.find(resource);
  return (it == lanes_.end()) ? 0 : it->second.ready.size();
}

std::size_t ExecutionResourceManager::total_queued() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_staged_;
}

}  // namespace vie
