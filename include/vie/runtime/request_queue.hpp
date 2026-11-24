#pragma once

#include "vie/runtime/inference_request.hpp"

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace vie {

// Thread-safe bounded request queue. Supports graceful shutdown: after
// shutdown(), push() rejects new requests, and wait_pop() wakes all waiting
// workers. Already-queued requests continue to be processed until the queue
// is empty.
class RequestQueue {
 public:
  explicit RequestQueue(std::size_t capacity);
  ~RequestQueue() = default;

  RequestQueue(const RequestQueue&) = delete;
  RequestQueue& operator=(const RequestQueue&) = delete;

  // Push a request into the queue. Returns false if the queue is full or
  // has been shut down. Non-blocking.
  bool push(InferenceRequest request);

  // Pop a request from the queue. Returns false if the queue is empty and
  // has been shut down. Blocks if the queue is empty but not shut down.
  bool wait_pop(InferenceRequest& request);

  // Initiate graceful shutdown. Rejects new pushes and wakes all waiting
  // workers. Already-queued requests remain available for processing.
  void shutdown();

  // Check if the queue has been shut down.
  bool is_shutdown() const;

  // Current number of requests in the queue.
  std::size_t size() const;

  // Check if the queue is empty.
  bool empty() const;

  // Maximum capacity.
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  bool shutdown_ = false;
  std::queue<InferenceRequest> queue_;
};

}  // namespace vie
