# Phase 9.2.2: Dynamic Batch Aggregator

## Overview

Phase 9.2.2 implements **automatic dynamic batch aggregation** on top of the Phase 9.2.1 async request queue. The core promise: **callers still submit single-image async requests via `runtime.submit(model_id, image)`, but the Runtime transparently groups compatible requests into batches and executes them via `ModelInstance::infer_batch()`**, returning individual results to each caller's future.

This phase delivers the actual throughput benefit of batch inference without requiring any changes to caller code.

## Architecture

### Components

```
                         ┌─────────────────────────────┐
  submit() ──────────────>  RequestQueue (bounded)     │
                         └──────────┬──────────────────┘
                                    │ wait_pop / wait_pop_until
                                    v
                         ┌─────────────────────────────┐
                         │  Aggregator Thread           │
                         │  (aggregator_loop)           │
                         │                              │
                         │  1. Pull requests from queue │
                         │  2. Group by ModelInstance   │
                         │  3. Flush on:                │
                         │     - batch full             │
                         │     - deadline expired       │
                         │     - shutdown               │
                         │  4. Execute infer_batch()    │
                         │  5. Split results to promises│
                         └─────────────────────────────┘
```

### Key Design Decisions

1. **Single Aggregator Thread**: The aggregator is not thread-safe by design. A single thread pulls from the RequestQueue, forms batches, and executes them. This avoids complex locking while maintaining correctness.

2. **Grouping Key = ModelInstance Pointer Identity**: Requests targeting the same `ModelInstance` (same `shared_ptr::get()`) are grouped together. This is the natural compatibility key since:
   - Same model weights and input shape
   - Same preprocessor configuration
   - Same backend and device

3. **CUDA Forces batch_size=1**: When the model's device is CUDA, `effective_max_batch_size()` returns 1 regardless of configuration. The CUDA decoder explicitly rejects batch>1, so we never attempt it.

4. **Batch=1 Uses infer() Not infer_batch()**: When a flushed batch contains only 1 request, the aggregator calls `model->infer()` instead of `model->infer_batch()`. This ensures compatibility with fixed-batch models (e.g., `yolov8n.onnx` with input shape `[1,3,640,640]`).

5. **Backpressure via max_pending_requests**: The aggregator rejects new requests when its internal pending count reaches the configured limit, preventing unbounded memory growth.

### Batch Formation Strategy

The aggregator uses a deadline-driven approach:

1. Each new request starts a **pending group** with a deadline = `now() + max_wait_time`
2. If a group reaches `max_batch_size`, it is **immediately flushed** (full flush)
3. If a group's deadline expires before filling, it is **timeout-flushed**
4. On shutdown, all pending groups are **shutdown-flushed**

This balances latency (don't wait forever for a full batch) against throughput (batch as many as possible within the wait window).

## New Files

| File | Description |
|------|-------------|
| `include/vie/runtime/batch_aggregation_config.hpp` | Configuration struct for batch aggregation |
| `include/vie/runtime/dynamic_batch_aggregator.hpp` | Aggregator class and `BatchWorkItem` struct |
| `src/runtime/dynamic_batch_aggregator.cpp` | Aggregator implementation |
| `tests/test_dynamic_batch_aggregator.cpp` | 20 tests (unit + integration) |
| `tools/batch_aggregator_benchmark.cpp` | Throughput/latency benchmark |
| `docs/research/PHASE_9_2_2_DYNAMIC_BATCH_AGGREGATOR.md` | This document |

## Modified Files

| File | Changes |
|------|---------|
| `include/vie/runtime/request_queue.hpp` | Added `wait_pop_until(deadline)` |
| `src/runtime/request_queue.cpp` | Implemented timed wait |
| `include/vie/runtime/runtime_metrics.hpp` | Added batch-level metrics fields |
| `src/runtime/runtime_metrics.cpp` | Added batch metrics recording methods |
| `include/vie/runtime/runtime_config.hpp` | Added `BatchAggregationConfig` field |
| `include/vie/runtime/runtime.hpp` | Added `aggregator_` member, `aggregator_loop()`, `execute_batch()` |
| `src/runtime/runtime.cpp` | Replaced `worker_loop()` with aggregator-based loop |
| `tests/test_async_runtime.cpp` | Updated config for single-threaded aggregator |
| `CMakeLists.txt` | Added new source files, test file, benchmark target |

## Configuration

### BatchAggregationConfig

```cpp
struct BatchAggregationConfig {
  bool enabled = true;                                    // Enable/disable aggregation
  std::size_t max_batch_size = 4;                         // Max images per batch
  std::chrono::milliseconds max_wait_time{5};             // Max wait before flushing
  std::size_t max_pending_requests = 256;                 // Backpressure limit
};
```

### RuntimeConfig Extension

```cpp
struct RuntimeConfig {
  // ... existing fields ...
  BatchAggregationConfig batch_aggregation;  // Phase 9.2.2
};
```

## Metrics

### New Batch-Level Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `total_batches` | `size_t` | Number of `infer_batch()` calls |
| `total_single_infer_calls` | `size_t` | Number of single-image `infer()` calls |
| `total_aggregated_requests` | `size_t` | Sum of all batch sizes |
| `batch_full_flush_count` | `size_t` | Batches flushed due to `max_batch_size` |
| `batch_timeout_flush_count` | `size_t` | Batches flushed due to timeout |
| `batch_shutdown_flush_count` | `size_t` | Batches flushed during shutdown |
| `batch_execution_failures` | `size_t` | Failed `infer_batch()` calls |
| `max_batch_size_achieved` | `size_t` | Largest batch actually executed |
| `batch_size_histogram` | `map<int, size_t>` | Distribution of batch sizes |

## API

### RequestQueue Extensions

```cpp
// Timed wait: returns false if deadline expires before an item is available
bool wait_pop_until(std::chrono::steady_clock::time_point deadline,
                    InferenceRequest& request);
```

### DynamicBatchAggregator

```cpp
class DynamicBatchAggregator {
 public:
  DynamicBatchAggregator(BatchAggregationConfig config,
                          std::shared_ptr<MetricsCollector> metrics);

  // Add a request. Returns a BatchWorkItem if the group is full.
  std::optional<BatchWorkItem> add(InferenceRequest request);

  // Flush the first expired group (deadline passed).
  std::optional<BatchWorkItem> flush_expired();

  // Flush all pending groups (for shutdown).
  std::vector<BatchWorkItem> flush_all();

  // Earliest deadline among pending groups.
  std::chrono::steady_clock::time_point earliest_deadline() const;

  // Total pending requests across all groups.
  std::size_t pending_count() const;

  // Number of distinct model groups.
  std::size_t group_count() const;

  // Whether aggregation is enabled.
  bool enabled() const noexcept;

  // Effective max batch size (forces 1 for CUDA or disabled).
  std::size_t effective_max_batch_size(bool is_cuda) const;
};
```

## Testing

### Test Coverage (20 tests)

**Unit Tests (DynamicBatchAggregatorTest)**:
- `EnabledReturnsTrue` / `DisabledWhenConfigFalse` / `CudaForcesBatchSizeOne`
- `AddSingleRequestDoesNotFlush` / `AddReachesMaxBatchSizeFlushes`
- `DifferentModelsCreateSeparateGroups`
- `FlushExpiredRespectsDeadline` / `FlushAllReturnsAllGroups`
- `EarliestDeadlineReturnsMaxWhenEmpty` / `EarliestDeadlineReturnsSoonestGroup`
- `BackpressureRejectsWhenFull`

**Queue Tests (RequestQueueBatchTest)**:
- `WaitPopUntilReturnsOnItem` / `WaitPopUntilTimesOut` / `WaitPopUntilWakesOnShutdown`

**Integration Tests (BatchRuntimeIntegrationTest)**:
- `BatchModelExecutesMultipleRequests` — Real batch model, 4 concurrent requests
- `SingleBatchFlushesOnTimeout` — Timeout-based flush with single request
- `FixedBatchModelWorksWithBatchSizeOne` — Backward compatibility with fixed models
- `FutureMappingIsCorrect` — Each future gets exactly one result
- `GracefulShutdownFlushesAllPending` — No requests lost on shutdown
- `MetricsRecordBatchExecution` — Batch metrics are recorded correctly

### Four-Build Regression

| Build | Tests | Passed | Skipped | Failed |
|-------|-------|--------|---------|--------|
| CUDA OFF + ORT OFF | 215 | 208 | 7 | 0 |
| CUDA OFF + ORT CPU | 265 | 258 | 7 | 0 |
| CUDA ON + ORT OFF | 298 | 291 | 7 | 0 |
| CUDA ON + ORT GPU | 311 | 303 | 8 | 0 |

## Benchmark Results

Using `yolov8n_batch.onnx` (dynamic batch), 40 requests, 4 concurrent threads:

| Scenario | Throughput | Avg Latency | Batches | Max Batch Size |
|----------|-----------|-------------|---------|----------------|
| A: Disabled (batch=1) | 11.19 req/s | 344.8 ms | 41 | 1 |
| B: Enabled (batch=4, 5ms) | **12.62 req/s** | 316.9 ms | 11 | 4 |
| C: Enabled (batch=8, 20ms) | 11.95 req/s | 334.6 ms | 11 | 4 |

**Scenario B achieves ~12.6% throughput improvement** by grouping requests into batches of 4, reducing per-image preprocessing and backend overhead.

## Backward Compatibility

- **Fixed-batch models**: When `max_batch_size=1`, the aggregator calls `infer()` instead of `infer_batch()`, maintaining full compatibility with models that have fixed batch dimensions.
- **Existing async API**: Callers continue to use `runtime.submit(model_id, image)` — no code changes required.
- **Phase 9.2.1 tests**: All existing `AsyncRuntimeTest` tests pass with the aggregator enabled (configured with `max_batch_size=1`).

## Thread Safety

- **RequestQueue**: Thread-safe (mutex + condition variables)
- **DynamicBatchAggregator**: NOT thread-safe — accessed only by the single aggregator thread
- **Runtime**: Thread-safe for `submit()` from multiple callers; `shutdown()` is idempotent
- **ModelInstance::infer_batch()**: Serialized by internal mutex (same as `infer()`)

## Limitations and Future Work

1. **Single aggregator thread**: Limits parallelism for batch formation. Future work could explore per-model aggregator threads.
2. **No batch size auto-tuning**: The `max_batch_size` is statically configured. Future work could dynamically adjust based on model capacity.
3. **No cross-model batching**: Requests for different models are never batched together, even if they have compatible input shapes. This is by design (different weights/preprocessing).
4. **Timeout flush may sub-optimal**: A group with 1 request flushes after `max_wait_time` even if more requests are imminent. Tuning guidance is provided in the configuration section.
