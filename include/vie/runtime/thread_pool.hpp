#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace vie {

// Lightweight fixed-size thread pool. Tasks are submitted and return
// std::future for result retrieval. Exceptions propagate through the future.
// Shutdown joins all workers; pending tasks complete before destruction.
class ThreadPool {
 public:
  explicit ThreadPool(std::size_t worker_count);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Submit a callable and return a future for its result. If the callable
  // throws, the exception is stored in the future and rethrown on get().
  template <typename F>
  auto submit(F&& func) -> std::future<std::invoke_result_t<F>>;

  std::size_t worker_count() const noexcept { return workers_.size(); }
  std::size_t pending_tasks() const;

 private:
  void worker_loop();

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool shutdown_ = false;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
};

template <typename F>
auto ThreadPool::submit(F&& func) -> std::future<std::invoke_result_t<F>> {
  using ReturnType = std::invoke_result_t<F>;
  auto task = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(func));
  std::future<ReturnType> future = task->get_future();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_) {
      throw std::runtime_error("submit on a stopped ThreadPool");
    }
    tasks_.emplace([task]() { (*task)(); });
  }
  condition_.notify_one();
  return future;
}

}  // namespace vie
