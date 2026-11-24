# Runtime Architecture

## Overview

The VIE Runtime layer provides thread-safe model management, concurrent inference execution, and performance metrics collection. It builds on top of the core `InferenceEngine` to enable multi-model scenarios and parallel workloads.

## Core Components

### ThreadPool

**Location**: `include/vie/runtime/thread_pool.hpp`, `src/runtime/thread_pool.cpp`

Fixed-size thread pool for asynchronous task execution.

**Features**:
- Configurable worker count
- `std::future`-based task submission
- Exception propagation through futures
- Graceful shutdown (completes pending tasks)
- Task queue tracking

**Usage**:
```cpp
vie::ThreadPool pool(4);  // 4 worker threads
auto future = pool.submit([] { return compute(); });
auto result = future.get();
```

**Thread Safety**: Fully thread-safe. Multiple threads can submit tasks concurrently.

### MetricsCollector

**Location**: `include/vie/runtime/runtime_metrics.hpp`, `src/runtime/runtime_metrics.cpp`

Thread-safe collector for runtime performance metrics.

**Tracked Metrics**:
- Total inference count and time
- Failed inference count
- Preprocess call count and time
- Model load/unload count
- Per-model inference statistics
- **Phase 9.2.1**: Request lifecycle metrics (submitted/accepted/rejected/completed/failed)
- **Phase 9.2.1**: Queue wait time, execution time, end-to-end time
- **Phase 9.2.1**: Current and max queue size

**Usage**:
```cpp
auto metrics = std::make_shared<vie::MetricsCollector>();
metrics->record_inference("model1", 15.3, true);
auto snapshot = metrics->snapshot();
```

**Thread Safety**: All methods are thread-safe (internal mutex).

### ModelInstance

**Location**: `include/vie/runtime/model_instance.hpp`, `src/runtime/model_instance.cpp`

Wraps a single `InferenceEngine` with thread-safe access and automatic metrics recording.

**Features**:
- Serializes concurrent inference calls (internal mutex)
- Automatically records inference timing and success/failure
- Exposes all `InferenceEngine` infer methods
- Thread-safe preprocess and execute

**Usage**:
```cpp
vie::ModelConfig config;
config.model_path = "model.onnx";
auto instance = std::make_shared<vie::ModelInstance>("id", config, metrics);
auto result = instance->infer(image);  // Thread-safe
```

**Thread Safety**: All public methods are thread-safe. Concurrent calls are serialized.

**Design Decision**: Each `ModelInstance` owns its own `InferenceEngine` (and thus its own Backend and Decoder). This ensures complete isolation between model instances.

### ModelManager

**Location**: `include/vie/runtime/runtime.hpp`, `src/runtime/runtime.cpp`

Manages multiple `ModelInstance` objects by ID.

**Features**:
- Load/unload models by ID
- Prevents duplicate loads
- Enforces maximum model count
- Thread-safe access to instances

**Usage**:
```cpp
auto metrics = std::make_shared<vie::MetricsCollector>();
vie::ModelManager manager(metrics, 16);  // Max 16 models
auto instance = manager.load("model1", config);
auto retrieved = manager.get("model1");
manager.unload("model1");
```

**Thread Safety**: All methods are thread-safe (internal mutex).

### Runtime

**Location**: `include/vie/runtime/runtime.hpp`, `src/runtime/runtime.cpp`

Top-level coordinator that owns ThreadPool, MetricsCollector, ModelManager, and the async RequestQueue.

**Features**:
- Single entry point for runtime configuration
- Owns and coordinates all runtime components
- Provides access to thread pool and model manager
- Snapshots metrics
- **Phase 9.2.1**: Async request submission via `submit(model_id, image)` returning `std::future<DetectionResult>`
- **Phase 9.2.1**: Bounded RequestQueue with backpressure and graceful shutdown
- **Phase 9.2.1**: Background inference worker threads

**Usage (synchronous via ThreadPool)**:
```cpp
vie::RuntimeConfig config;
config.thread_pool_size = 4;
config.enable_profiling = true;

vie::Runtime runtime(config);
auto model = runtime.models().load("id", model_config);
auto future = runtime.thread_pool().submit([model] {
  return model->infer(image);
});
auto metrics = runtime.snapshot_metrics();
```

**Usage (asynchronous via RequestQueue, Phase 9.2.1)**:
```cpp
vie::RuntimeConfig config;
config.request_queue_capacity = 1024;
config.inference_worker_count = 2;

vie::Runtime runtime(config);
runtime.models().load("det", det_config);

// Submit async request, get future
auto future = runtime.submit("det", image);
DetectionResult result = future.get();

// Graceful shutdown
runtime.shutdown();
```

**Thread Safety**: Access to sub-components is thread-safe (see their documentation).

## Concurrency Model

### Different Models

Multiple `ModelInstance` objects can run inference concurrently without sharing VIE-owned mutable state. Each instance owns:
- Its own `InferenceEngine`
- Its own Backend (with independent ONNX Runtime session)
- Its own Decoder
- Its own mutex (for serializing same-model calls)

**Parallelism**: Different instances can be driven from different worker threads in parallel. Phase 8.3.1 real-model benchmark (Scenario C) observed throughput gains of ~50% on CPU EP and ~21% on CUDA EP when going from 1 to 2 instances. **Note for CUDA**: the backend's `execute()` calls `cudaDeviceSynchronize()` twice per run, which is a device-wide barrier. Whether GPU kernels from two independent Sessions actually overlap in wall-clock time was NOT verified by a profiler; the measured throughput gain is attributable to Session-level concurrency plus CPU-side overlap of preprocess/decode. Do not describe this as "GPU kernel overlap is achieved".

### Same Model

Concurrent calls to the same `ModelInstance` are serialized by its internal mutex. This is because:
- The Backend's `execute()` method modifies timing state (not thread-safe)
- The `InferenceEngine` is documented as not thread-safe for concurrent infer

**Parallelism**: None for the same model instance. Create multiple instances if parallel execution of the same model is needed.

### Thread Pool

The `ThreadPool` executes tasks on worker threads. Tasks can call inference methods on different `ModelInstance` objects in parallel.

**Recommendation**: Set `thread_pool_size` based on:
- Number of different models you want to run concurrently
- Available CPU cores
- Whether models are CPU or CUDA (CUDA models can share GPU more efficiently)

## BackendRegistry Thread Safety

**Location**: `include/vie/backend/backend_registry.hpp`, `src/backend/backend_registry.cpp`

The `BackendRegistry` has been updated to include a mutex, making all methods thread-safe. This aligns with `DecoderRegistry` which already had a mutex.

**Impact**: Minimal performance overhead. Registry operations (register, create) are infrequent compared to inference.

## Lifecycle

### Model Loading

1. `Runtime` is constructed with `RuntimeConfig`
2. `ModelManager::load()` creates a new `ModelInstance`
3. `ModelInstance` constructor:
   - Creates an `InferenceEngine`
   - Calls `engine.load(config)` (loads model, creates Backend and Decoder)
   - Records model load in `MetricsCollector`
4. Returns `shared_ptr<ModelInstance>`

### Inference

1. User calls `instance->infer(image)` (or other infer method)
2. `ModelInstance` acquires mutex (serializes concurrent calls)
3. `InferenceGuard` records start time
4. `engine.infer(image)` executes:
   - Preprocess
   - Backend execute
   - Decode
5. `InferenceGuard` destructor records metrics (time, success/failure)
6. Mutex released
7. Result returned

### Model Unloading

1. User calls `manager.unload(id)`.
2. `ModelManager` removes the entry from its map under `mutex_`. Any `shared_ptr` that a caller obtained earlier is unaffected.
3. Inference in flight on a previously-obtained `shared_ptr` continues to succeed. The model's resources are NOT released at the moment `unload()` returns.
4. When the last outstanding `shared_ptr` drops:
   - `ModelInstance` destructor runs.
   - `InferenceEngine` destructor runs.
   - Backend destructor runs (releases ONNX Runtime session; for CUDA this calls `quiesce()` which does `cudaDeviceSynchronize()` before dropping the session).
   - Decoder destructor runs.

`unload()` is therefore "remove from the manager and forbid new lookups", not "cancel in-flight inference" and not "synchronously free CUDA memory". Callers that hold a reference keep the model alive until they release it.

### Runtime Shutdown

1. `Runtime::shutdown()` is called (or `Runtime` destructor runs)
2. `running_` flag set to false
3. `RequestQueue::shutdown()` rejects new pushes, wakes all waiting workers
4. Workers continue processing remaining queued requests (graceful)
5. When queue is empty, `wait_pop()` returns false, workers exit
6. All worker threads joined
7. `ModelManager` destructor clears (unloads all models)
8. `ThreadPool` destructor runs:
   - Sets `shutdown_` flag
   - Notifies all workers
   - Workers complete pending tasks and exit
   - All worker threads joined

## Configuration

### RequestQueue (Phase 9.2.1)

**Location**: `include/vie/runtime/request_queue.hpp`, `src/runtime/request_queue.cpp`

Thread-safe bounded queue for async inference requests. Supports graceful shutdown.

**Features**:
- Bounded capacity with non-blocking push (rejects when full)
- Blocking wait_pop for worker threads
- Graceful shutdown: rejects new pushes, wakes waiting workers, drains queued requests
- `InferenceRequest` carries pre-resolved `shared_ptr<ModelInstance>` for model lifecycle safety

**Thread Safety**: Fully thread-safe via `std::mutex` + `std::condition_variable`.

### RuntimeConfig

```cpp
struct RuntimeConfig {
  std::size_t thread_pool_size = 4;
  DeviceType default_device = DeviceType::Cpu;
  int default_device_id = 0;
  bool enable_profiling = false;
  std::size_t max_loaded_models = 16;
  // Phase 9.2.1
  std::size_t request_queue_capacity = 1024;
  std::size_t inference_worker_count = 2;
  bool defer_start = false;
};
```

**thread_pool_size**: Number of worker threads. Set based on concurrency needs and available cores.

**default_device**: Default device for models that don't specify.

**default_device_id**: Default device ID (GPU index).

**enable_profiling**: Reserved for future detailed profiling.

**max_loaded_models**: Maximum number of models that can be loaded simultaneously.

## Performance Considerations

### Memory

- Each `ModelInstance` owns a complete `InferenceEngine` (Backend + Decoder)
- ONNX Runtime sessions are not shared between instances
- CUDA models: each instance creates its own CUDA stream
- Tensors are copied (not shared) between Backend and VIE

### CPU

- Different models run in parallel (no contention)
- Same model runs are serialized (mutex overhead)
- Thread pool workers are always running (no idle shutdown)

### Recommendations

1. **Load different models**: Maximizes parallelism
2. **Multiple instances of same model**: If you need parallel execution of the same model, load it multiple times with different IDs
3. **Thread pool size**: Match to number of concurrent models, not total models
4. **CUDA**: Multiple CUDA models share the GPU naturally (CUDA streams handle concurrency)

## Error Handling

- `ModelManager::load()` throws `VieError` if:
  - Model ID already exists
  - Maximum model count reached
  - Model loading fails
- `ModelManager::unload()` throws `VieError` if model ID not found
- `ModelInstance::infer()` propagates exceptions from `InferenceEngine`
- Failed inferences are recorded in metrics (success=false)
- **Phase 9.2.1**: `Runtime::submit()` throws `VieError` synchronously if:
  - Runtime is not running (`RuntimeNotStarted`)
  - Model ID is not loaded (`InvalidModelId`)
  - Request queue is full (`RequestQueueFull`)
- **Phase 9.2.1**: Execution errors propagate asynchronously through `std::future`

## Example

See `examples/multi_model_runtime.cpp` for a complete working example.

```cpp
// Create runtime
vie::RuntimeConfig config;
config.thread_pool_size = 4;
vie::Runtime runtime(config);

// Load models
auto model1 = runtime.models().load("det", det_config);
auto model2 = runtime.models().load("seg", seg_config);

// Submit concurrent inference
auto task1 = runtime.thread_pool().submit([model1, image1] {
  return model1->infer(image1);
});
auto task2 = runtime.thread_pool().submit([model2, image2] {
  return model2->infer_segmentation(image2, seg_runtime);
});

// Wait for results
auto result1 = task1.get();
auto result2 = task2.get();

// Check metrics
auto metrics = runtime.snapshot_metrics();
std::cout << "Total inferences: " << metrics.total_inferences << "\n";
```

## Future Enhancements

- **Automatic Batch Aggregation**: Request scheduler that transparently batches incoming single-image requests (Phase 9.2.2 - next)
- **Priority queues**: High-priority tasks preempt low-priority
- **Dynamic thread pool**: Adjust worker count based on load
- **Model sharing**: Share Backend/Decoder between instances (read-only)
- **Detailed profiling**: Per-layer timing, memory tracking
- **Async I/O**: Overlap data loading with inference
