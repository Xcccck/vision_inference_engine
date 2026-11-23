#include "vie/runtime/thread_pool.hpp"

namespace vie {

ThreadPool::ThreadPool(std::size_t worker_count) {
  for (std::size_t i = 0; i < worker_count; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
  }
  condition_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void ThreadPool::worker_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return shutdown_ || !tasks_.empty(); });
      if (shutdown_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

std::size_t ThreadPool::pending_tasks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tasks_.size();
}

}  // namespace vie
