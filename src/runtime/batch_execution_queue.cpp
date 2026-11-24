#include "vie/runtime/batch_execution_queue.hpp"

namespace vie {

BatchExecutionQueue::BatchExecutionQueue(std::size_t capacity) : capacity_(capacity) {
  if (capacity == 0) {
    throw std::invalid_argument("BatchExecutionQueue capacity must be at least 1");
  }
}

bool BatchExecutionQueue::push(BatchWorkItem batch) {
  std::unique_lock<std::mutex> lock(mutex_);

  // Block while the queue is full and not shut down.
  // If shutdown is initiated while we're waiting, wake up and reject.
  not_full_.wait(lock, [this] { return queue_.size() < capacity_ || shutdown_; });

  if (shutdown_) {
    return false;
  }

  queue_.push(std::move(batch));
  not_empty_.notify_one();
  return true;
}

bool BatchExecutionQueue::wait_pop(BatchWorkItem& batch) {
  std::unique_lock<std::mutex> lock(mutex_);

  not_empty_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

  if (queue_.empty()) {
    return false;  // Shutdown with empty queue
  }

  batch = std::move(queue_.front());
  queue_.pop();
  not_full_.notify_one();
  return true;
}

void BatchExecutionQueue::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  shutdown_ = true;
  not_empty_.notify_all();
  not_full_.notify_all();
}

bool BatchExecutionQueue::is_shutdown() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return shutdown_;
}

std::size_t BatchExecutionQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

bool BatchExecutionQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

}  // namespace vie
