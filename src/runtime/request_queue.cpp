#include "vie/runtime/request_queue.hpp"

#include <chrono>

namespace vie {

RequestQueue::RequestQueue(std::size_t capacity) : capacity_(capacity) {
  if (capacity == 0) {
    throw std::invalid_argument("RequestQueue capacity must be at least 1");
  }
}

bool RequestQueue::push(InferenceRequest request) {
  std::unique_lock<std::mutex> lock(mutex_);
  
  // Reject if shutdown or full
  if (shutdown_ || queue_.size() >= capacity_) {
    return false;
  }
  
  queue_.push(std::move(request));
  not_empty_.notify_one();
  return true;
}

bool RequestQueue::wait_pop(InferenceRequest& request) {
  std::unique_lock<std::mutex> lock(mutex_);
  
  // Wait until there's a request or shutdown is initiated
  not_empty_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
  
  // If queue is empty and shutdown, return false
  if (queue_.empty()) {
    return false;
  }
  
  request = std::move(queue_.front());
  queue_.pop();
  not_full_.notify_one();
  return true;
}

// Phase 9.2.2: Timed wait variant for aggregator deadline-driven polling.
bool RequestQueue::wait_pop_until(std::chrono::steady_clock::time_point deadline,
                                  InferenceRequest& request) {
  std::unique_lock<std::mutex> lock(mutex_);

  // Wait until: item available, shutdown, or deadline reached
  if (!not_empty_.wait_until(lock, deadline,
                              [this] { return !queue_.empty() || shutdown_; })) {
    // Timed out: deadline reached with no item and no shutdown
    return false;
  }

  // If queue is empty (shutdown with nothing queued), return false
  if (queue_.empty()) {
    return false;
  }

  request = std::move(queue_.front());
  queue_.pop();
  not_full_.notify_one();
  return true;
}

void RequestQueue::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  shutdown_ = true;
  not_empty_.notify_all();
  not_full_.notify_all();
}

bool RequestQueue::is_shutdown() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return shutdown_;
}

std::size_t RequestQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

bool RequestQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

}  // namespace vie
