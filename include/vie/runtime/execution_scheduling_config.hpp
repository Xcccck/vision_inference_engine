#pragma once

#include <cstddef>

namespace vie {

// Phase 9.3: Device-aware execution scheduling configuration.
//
// This controls the Runtime's resource-level admission policy, NOT the true
// parallel capability of any backend. In particular cuda_max_concurrent_per_device
// is a conservative Runtime gate; it does not imply CUDA kernel concurrency,
// multiple CUDA streams, or a session pool. With the current ONNX Runtime CUDA
// EP (single stream) the default of 1 is the safe choice.
struct ExecutionSchedulingConfig {
  // Maximum number of batches the scheduler will run concurrently on the same
  // CUDA device. Must be >= 1. Zero is invalid (it does NOT mean unlimited).
  // Default 1 is the conservative policy required by Phase 9.3.
  std::size_t cuda_max_concurrent_per_device = 1;

  // Concurrency limit for the CPU execution lane. Zero means "unbounded":
  // CPU parallelism stays governed only by the worker-thread pool (and the
  // per-ModelInstance mutex), which is the pre-Phase-9.3 behavior. A non-zero
  // value imposes an explicit ceiling. Default 0 preserves existing behavior.
  std::size_t cpu_max_concurrent = 0;

  // Capacity of the scheduler's per-resource staging queues, counted as a
  // single shared pool across all lanes so splitting by resource never
  // increases total admitted batches beyond the configured backpressure.
  // Must be >= 1. If 0, the value falls back to batch_execution_queue_capacity
  // supplied to the Runtime.
  std::size_t lane_capacity = 0;
};

}  // namespace vie
