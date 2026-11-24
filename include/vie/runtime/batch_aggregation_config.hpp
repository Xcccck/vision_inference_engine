#pragma once

#include <chrono>
#include <cstddef>

namespace vie {

// Configuration for the dynamic batch aggregator (Phase 9.2.2).
// Controls how single-image requests are automatically grouped into
// batch inference calls.
struct BatchAggregationConfig {
  // Enable or disable automatic batch aggregation. When false, each
  // request is executed individually (equivalent to batch_size=1).
  bool enabled = true;

  // Maximum number of requests per batch. Must be >= 1.
  std::size_t max_batch_size = 4;

  // Maximum time to wait for more compatible requests before flushing
  // a partial batch. Must be >= 0ms. Uses steady_clock.
  std::chrono::milliseconds max_wait_time{5};

  // Maximum number of requests the aggregator holds pending at once.
  // Must be >= 1. Provides backpressure against unbounded accumulation.
  std::size_t max_pending_requests = 256;
};

}  // namespace vie
