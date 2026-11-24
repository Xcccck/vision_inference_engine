#pragma once

#include "vie/runtime/dynamic_batch_aggregator.hpp"

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace vie {

// Bounded thread-safe queue connecting the aggregator (producer) to execution
// workers (consumers). The aggregator pushes formed BatchWorkItems; workers
// pop and execute them. Different workers can execute batches for different
// ModelInstances in parallel (ModelInstance mutex still serializes same-model).
//
// Backpressure: push() blocks when the queue is full, which naturally blocks
// the aggregator and propagates back to the RequestQueue.
class BatchExecutionQueue {
 public:
  explicit BatchExecutionQueue(std::size_t capacity);
  ~BatchExecutionQueue() = default;

  BatchExecutionQueue(const BatchExecutionQueue&) = delete;
  BatchExecutionQueue& operator=(const BatchExecutionQueue&) = delete;

  // Push a batch for execution. Blocks if the queue is full. Returns false
  // if the queue has been shut down.
  bool push(BatchWorkItem batch);

  // Pop a batch for execution. Blocks until a batch is available or the
  // queue is shut down and empty. Returns false when no more batches will
  // arrive (queue empty + shutdown).
  bool wait_pop(BatchWorkItem& batch);

  // Initiate graceful shutdown. Rejects new pushes and wakes all waiting
  // workers. Already-queued batches remain available for processing.
  void shutdown();

  bool is_shutdown() const;
  std::size_t size() const;
  bool empty() const;
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  bool shutdown_ = false;
  std::queue<BatchWorkItem> queue_;
};

}  // namespace vie
