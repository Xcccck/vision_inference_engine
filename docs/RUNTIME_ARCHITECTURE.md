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
- **Phase 9.2.2**: Batch aggregation metrics (total_batches, batch_size_histogram, flush counts)
- **Phase 9.3**: Resource scheduling metrics (total_resource_admissions, total_resource_wait_ms, per_resource_admissions/wait_ms keyed by "cpu"/"cuda:N")

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
- **Phase 9.3.2**: `infer_async()` / `infer_batch_async()` are the async-only
  execution boundary used by `Runtime::submit()`'s worker. On a CUDA instance fed
  a Runtime-owned CPU image they perform a `CudaInputBuffer`-owned H2D before
  inference; otherwise (CPU instances, already-on-device images, and all
  synchronous `infer()`/`infer_batch()` callers) they behave exactly like the
  synchronous methods. The synchronous APIs are unchanged.
- **Phase 9.3.2.1**: `end_profiling()` flushes ORT profiling for *this* instance's
  real engine session and returns the Chrome-trace path (empty unless the model was
  loaded with `ModelConfig::ort_enable_profiling`). It shares the instance mutex, so
  it never races an in-flight run and never changes inference results.

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
- **Phase 9.2.2**: Dynamic batch aggregation — transparently groups compatible requests into batches
- **Phase 9.2.2**: Single aggregator thread with deadline-driven flush strategy
- **Phase 9.2.2**: CUDA models automatically forced to batch_size=1
- **Phase 9.2.2.3**: `submit()` (async RequestQueue path) accepts CPU/Borrowed `ImageInput` only; CUDA device inputs are rejected synchronously before entering the queue (see "Input Device Support" below)
- **Phase 9.3**: Device-aware execution scheduling — formed batches are routed into per-`ExecutionResource` lanes with independent concurrency limits (see "Device-aware Execution Scheduling" below)

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
config.inference_worker_count = 1;  // Phase 9.2.2: single aggregator thread

vie::Runtime runtime(config);
runtime.models().load("det", det_config);

// Submit async request, get future
auto future = runtime.submit("det", image);
DetectionResult result = future.get();

// Graceful shutdown
runtime.shutdown();
```

**Usage (with batch aggregation, Phase 9.2.2)**:
```cpp
vie::RuntimeConfig config;
config.batch_aggregation.enabled = true;
config.batch_aggregation.max_batch_size = 4;
config.batch_aggregation.max_wait_time = std::chrono::milliseconds(5);

vie::Runtime runtime(config);
runtime.models().load("det", det_config);

// Requests are automatically batched
auto f1 = runtime.submit("det", image1);
auto f2 = runtime.submit("det", image2);
// Both may be executed as a single infer_batch() call
```

**Thread Safety**: Access to sub-components is thread-safe (see their documentation).

### Input Device Support (Phase 9.2.2.3)

`Runtime` exposes two different submission paths with different device guarantees:

| API | CPU `ImageInput` | CUDA `ImageInput` |
|---|---|---|
| `Runtime::submit(model_id, image)` (async, RequestQueue) | Supported | **Rejected** with `ErrorCode::Unsupported` |
| `ModelInstance::infer()` / `InferenceEngine::infer()` (synchronous) | Supported | Supported (unchanged) |

**Why the async path rejects CUDA input**: `ImageInput::data` for a CUDA image is a raw device pointer (`cudaMalloc`'d by the caller) with no ownership transfer. `InferenceRequest::adopt_cpu_pixels()` deep-copies only CPU pixel bytes into `owned_pixels`; there is no equivalent safe, cheap device-side copy that `Runtime` owns. Since the async worker may run the request at an arbitrary later time, the caller is free to `cudaFree()`/reuse the buffer before the worker reads it, which would cause a GPU-side use-after-free or silently corrupted input. Rather than introduce a partial, unproven CUDA buffer-ownership mechanism (which would also require new `InferenceRequest` members to depend on CUDA headers, breaking non-CUDA builds), the async API's device contract is explicitly narrowed to CPU only.

**Rejection semantics**:
- Check happens at the very start of `Runtime::submit()`, before `models_.get(model_id)` lookup and before `RequestQueue::push()` — a rejected request never increments `accepted_requests` and never produces a `std::future`.
- Throws `VieError(ErrorCode::Unsupported, ...)` synchronously; the message names the actual rejected `DeviceType` (`to_string(image.device)`) so callers can distinguish an unsupported *device type* from other `submit()` errors (`RuntimeNotStarted`, `InvalidModelId`, `RequestQueueFull`).
- Only affects the async `submit(model_id, image)` overload; `Runtime::thread_pool().submit(...)` (raw lambda task submission, unchanged) and direct `ModelInstance::infer()`/`ModelInstance::infer_batch()` CUDA calls are unaffected.

**Migrating existing CUDA async callers**: submit work through `Runtime::thread_pool().submit([model, image]{ return model->infer(image); })` instead — this keeps the call fully synchronous from `Runtime`'s RequestQueue/aggregator/batch-worker pipeline's point of view, so the caller's own buffer-lifetime discipline (holding the CUDA buffer alive until `future.get()`) remains the single, explicit contract, exactly as before this phase.

**Does a *CPU* `ImageInput` reach a *CUDA* model through async `submit()`?** **Yes, as of Phase 9.3.2.** `submit()` still enqueues the CPU image and the Phase 9.3 scheduler still routes it into the model's `cuda:N` lane and grants a permit; the async execution worker (`ModelInstance::infer_async()`) now performs a **Runtime-owned H2D** into a `CudaInputBuffer` before preprocess, so the CUDA backend executes and returns a real result. This does **not** change the row above: an *external* CUDA `ImageInput` is still rejected at `submit()` — only the pixels `Runtime` deep-copied at submit are ever uploaded. See "CPU-to-CUDA Input Staging (Phase 9.3.2)" and the traced Phase 9.3.1 call chain below.

**Known limitation**: two timing-based multi-model concurrency tests (`MultiModelConcurrencyTest.DifferentModelsExecuteInParallel`, `RuntimeConcurrencySemanticsTest.DifferentModelsHaveOverlappingExecution`) remain conditionally skipped in the ORT-CUDA (Build D) configuration only — the ONNX Runtime CUDA Execution Provider serializes inference through a single CUDA stream, making wall-clock overlap assertions unreliable in that build. Mock-backend concurrency tests (which do not depend on real GPU timing) continue to run and prove scheduling parallelism in every build.

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
- **Phase 9.2.2**: Timed `wait_pop_until(deadline)` for aggregator deadline-driven polling
- Graceful shutdown: rejects new pushes, wakes waiting workers, drains queued requests
- `InferenceRequest` carries pre-resolved `shared_ptr<ModelInstance>` for model lifecycle safety

**Thread Safety**: Fully thread-safe via `std::mutex` + `std::condition_variable`.

### DynamicBatchAggregator (Phase 9.2.2)

**Location**: `include/vie/runtime/dynamic_batch_aggregator.hpp`, `src/runtime/dynamic_batch_aggregator.cpp`

Automatically groups compatible single-image requests into batches for `infer_batch()` execution.

**Features**:
- Groups requests by `ModelInstance` pointer identity **and** the effective inference config fingerprint (`ResolvedInferenceConfig::fingerprint()`, Phase 12). Two requests with different thresholds never share a batch, even for the same model.
- Flush on batch full (`max_batch_size` reached)
- Flush on timeout (`max_wait_time` elapsed since first request in group)
- Flush on shutdown (all pending groups drained)
- CUDA models forced to `batch_size=1` (CUDA decoder rejects batch>1)
- Batch=1 uses `infer()` instead of `infer_batch()` for fixed-model compatibility
- Backpressure via `max_pending_requests` limit

**Thread Safety**: NOT thread-safe — designed for single aggregator thread access only.

### Device-aware Execution Scheduling (Phase 9.3)

**Location**: `include/vie/runtime/execution_resource.hpp`, `include/vie/runtime/execution_scheduling_config.hpp`, `include/vie/runtime/execution_resource_manager.hpp`, `src/runtime/execution_resource_manager.cpp`

Phase 9.3 adds a resource-aware admission layer between the aggregator and the execution workers. It replaces the single FIFO `BatchExecutionQueue` as the hand-off point: formed batches are routed into a lane keyed by their model's `ExecutionResource`, and a worker receives a batch only when that lane currently has a free concurrency permit.

**ExecutionResource** — what identifies the resource:

```cpp
enum class ExecutionResourceType { Cpu, Cuda, Unknown };
struct ExecutionResource { ExecutionResourceType type; int device_id; };
```

1. **Definition & source of truth**: The resource is derived from the *real execution path*, NOT from `ImageInput.device`. `InferenceEngine::load()` commits a single `DeviceContext` (`engine.device()`), and preprocess / backend execute / decode all use that same committed device (verified in the Phase 9.3 audit). `resolve_backend()` guarantees `OnnxRuntimeCuda ⇔ DeviceType::Cuda` and forces `OnnxRuntime ⇒ Cpu`; the Mock backend never runs on CUDA. So for a loaded engine, `device()` is the authoritative execution resource of the whole request.
2. **ModelInstance mapping**: `ModelInstance::execution_resource()` returns `execution_resource_from_device(engine_.device())` for a loaded engine, and `Unknown` if the engine was never loaded. A `ModelManager`-stored instance is always loaded, so `Unknown` cannot legitimately reach the scheduler; it is rejected defensively rather than being treated as CPU.
3. **Device ID**: CUDA resources are distinguished by `device_id`; `cuda:0` and `cuda:1` are separate lanes with separate permits. CPU collapses to a single `cpu` lane.

**Scheduling policy** (per lane):

| Resource | Concurrency limit | Default |
|---|---|---|
| CPU lane | `cpu_max_concurrent`, or **unbounded** when `0` | `0` (unbounded) — preserves pre-9.3 behavior |
| Each CUDA device | `cuda_max_concurrent_per_device` (must be ≥ 1) | `1` (conservative) |
| Unknown | — | Rejected (`Unsupported`); never silently CPU |

- **CPU**: There is deliberately **no** global CPU serial lock. The CPU lane is unbounded by default, so CPU parallelism stays governed only by `inference_worker_count` and the per-`ModelInstance` mutex — exactly as before Phase 9.3. A non-zero `cpu_max_concurrent` imposes an explicit ceiling when desired.
- **CUDA**: The default cap of 1 is a conservative Runtime-level **admission gate**, NOT a claim of GPU kernel concurrency, multi-stream execution, or a session pool. With the current single-stream ORT CUDA EP, admitting more than one concurrent batch per device has no proven wall-clock benefit, so the safe default serializes them.

**Head-of-line blocking avoidance**: `pop()` waits until *some* lane has a staged batch **and** a free permit, then hands back that batch. Because the predicate only requires an *eligible* lane (not a specific one), a worker never parks on a saturated GPU lane while CPU or another device still has runnable work. `CpuNotBlockedBySaturatedCudaLane` proves a CPU batch is popped and executed while the CUDA lane is pinned at its cap; `DifferentCudaDevicesAreIndependent` proves `cuda:0` saturation does not block `cuda:1`.

**Backpressure preserved**: permits gate *execution*, but staging is bounded by one shared pool (`total_staged_ < capacity_`) across all lanes, so splitting into lanes never admits more in-flight batches than the single bounded queue did. `lane_capacity` defaults (falls back) to `batch_execution_queue_capacity`.

**Batch ↔ permit relationship**: One `BatchWorkItem` (regardless of batch size) occupies exactly **one** permit of its resource lane for its whole execution. Requests in a batch are never re-ordered and each keeps its own `GeometryTransform`/result mapping. CUDA batch stays forced to size 1 (Phase 9.2.2 unchanged).

**Permit lifecycle & exception safety (RAII)**: `pop()` reserves a permit and returns a `ScheduledBatch { BatchWorkItem item; ResourceLease lease; }`. `ResourceLease` is move-only; its destructor calls `release()`. The permit is therefore returned exactly once on every path — normal return, exception unwinding, or worker exit — and there is no way to double-release (a manual release racing the destructor is impossible). `pop()` releases any permit already held by a reused `ScheduledBatch` *before* taking the internal mutex, so it never self-deadlocks. Execution order is always *permit acquired (pop) → then `ModelInstance` mutex (inside `execute_batch`)*, so the scheduler lock is never held while acquiring an instance lock — no lock-order inversion.

**Shutdown**: `Runtime::shutdown()` joins the aggregator first (so every formed batch is already pushed), then calls `resource_mgr_->shutdown()`, which only sets the flag and wakes everyone. Workers keep draining already-staged batches (their futures still complete) and exit only once every lane is empty. No accepted request is stranded, no permit leaks, and there is no deadlock (`ShutdownDrainsAllAcceptedBatches`).

**Metrics** (see `MetricsCollector`): `total_resource_admissions`, `total_resource_wait_ms`, and per-resource `per_resource_admissions` / `per_resource_wait_ms`. Resource *wait* time (time a batch spent parked waiting for a permit) is tracked separately from and never folded into backend execution time or GPU kernel time. The manager records admissions after releasing its own lock, so the scheduler mutex never nests with the metrics mutex.

**Thread Safety**: `ExecutionResourceManager` is fully thread-safe (single internal mutex + condition variable). Producers call `push()`, workers call `pop()`; permits return via `ResourceLease` destructors on worker threads.

**Explicitly NOT implemented in Phase 9.3**: CUDA multi-stream, ORT session pools, and GPU auto load-balancing remain future work. This layer is resource *admission* only.

### CUDA Runtime Scheduling Integration Audit (Phase 9.3.1)

Phase 9.3.1 verifies whether the device-aware scheduler is *actually wired into the
Runtime's CUDA execution path*, and corrects a possible misreading of the
Phase 9.2.2.3 CUDA-input rejection. The finding is an **audit conclusion, not a
new feature**: with the current code, a CUDA-loaded `ModelInstance` is correctly
recognised and routed into its `cuda:N` lane (permit acquired and released), but
the *async* `Runtime::submit()` pipeline cannot drive the CUDA backend to a
successful inference, because of a deliberate input-ownership boundary — not a
scheduling defect.

**Actual CPU-image → CUDA-backend call chain (Build D, traced):**

```text
Runtime::submit(model_id, image)      // runtime.cpp: rejects image.device != Cpu
  -> models_.get(id)                  // CUDA-loaded ModelInstance
  -> model->execution_resource()       // == cuda:N (valid)  -> lane registered
  -> RequestQueue::push                // InferenceRequest::adopt_cpu_pixels(): device stays Cpu
  -> DynamicBatchAggregator            // CUDA forced batch_size=1 (unchanged)
  -> dispatch_batch: resource=cuda:N   // routed into the cuda:N lane
  -> ExecutionResourceManager::pop()   // acquires 1 permit, records per_resource_admissions["cuda:N"]
  -> execute_batch -> ModelInstance::infer(cpu_image)
       -> InferenceEngine::infer -> preprocess(image)  // preprocess.cpp, uses engine device_ (Cuda)
            -> ImagePreprocessor::execute(cpu_image, cuda_device)  // preprocess.cpp:284
                 THROW InvalidArgument "image device cpu does not match execution device cuda"
  -> promise set_exception -> permit released (ResourceLease dtor) -> future.get() throws
```

**Why a CPU `ImageInput` cannot reach the CUDA backend today (Conclusion B):**

1. `Runtime::submit()` accepts only `image.device == Cpu` (Phase 9.2.2.3).
2. `adopt_cpu_pixels()` keeps `image.device == Cpu` (it never moves pixels to device).
3. `InferenceEngine::infer()` preprocesses with the engine's committed `device_`
   (CUDA), and `ImagePreprocessor::execute()` **requires** `image.device == device.device()`
   and throws otherwise (`src/preprocess/preprocess.cpp:284`).
4. `preprocess_cuda()` additionally assumes the image is *already on device*
   ("H2D cost stays outside this function", `src/preprocess/preprocess_cuda.cu:109`).

So the only missing step is a Runtime-owned H2D copy plus a device-buffer lifetime
manager. Introducing it would mean a **new CUDA buffer-ownership system inside
`InferenceRequest`/`Runtime`** — exactly what Phase 9.2.2.3 declined and Phase 9.3.1
scoped out. It is therefore left as future work, and this audit does **not** change
the CUDA raw-device-pointer rejection.

**What this means for the scheduler**: `submit()` of a *CPU* image to a *CUDA* model
still enqueues and is admitted to the `cuda:N` lane (an admission *is* recorded), so
the cross-resource routing and the permit acquire/release lifecycle are exercised on
a genuine CUDA resource lane even though the inference itself fails at the
preprocess boundary. This is the point Phase 9.3 could only prove at unit level;
Phase 9.3.1 proves it end-to-end through `Runtime::submit()` with a real
CUDA-loaded model.

**Terminology guardrail**: *CUDA `ImageInput` is not accepted by `submit()`* ≠ *CUDA
backends cannot be scheduled*. The scheduler fully supports CUDA resources and lanes;
what is not yet supported is feeding the CUDA backend through the **async** input path.
The **synchronous** CUDA path (`ModelInstance::infer`/`infer_batch` with a caller-owned
on-device image) is fully supported and is what executes the real CUDA EP.

**Runtime permit vs GPU kernel concurrency (unchanged)**: the `cuda_max_concurrent_per_device`
= 1 gate is a Runtime admission limit, not a GPU kernel-stream guarantee. It is proven
by the manager-level tests (`CudaCapEnforcedSerializesSameDevice`, max active permit == 1;
`DifferentCudaDevicesAreIndependent`; `LeaseReleasesOnException`) plus the new Build-D
runtime tests below. The public Runtime surface exposes resource admissions
(`per_resource_admissions`), not live permit counts; the authoritative permit-count
assertions therefore stay at the manager level.

**Real-CUDA test coverage added in Phase 9.3.1** (`test_execution_scheduling.cpp`,
Section 4, compiled only under `VIE_HAS_ONNXRUNTIME_CUDA`):
- `LoadedCudaModelReportsCudaResource` — a real CUDA-loaded model reports `cuda:0`.
- `CudaBackendExecutesSynchronouslyOnRealGpu` — a genuine on-device image runs through
  the CUDA-configured engine and returns a result tagged `execution_device == Cuda`
  (proves the CUDA path really executes; the strict "every node on CUDAExecutionProvider"
  guarantee is the `identity.onnx` backend test — `yolov8n` loads with
  `cuda_cpu_ep_fallback=true` because two `Resize` nodes are placed on the CPU EP
  while its four `Memcpy` kernels stay on CUDA; see "CUDA Execution Provider
  Placement Verification (Phase 9.3.2.1)" for the node-level evidence).
- `CudaModelCpuImageIsRoutedToCudaLaneAndReleasesPermit` — the async boundary: a CPU
  image on a CUDA model is admitted to `cuda:0`, fails at preprocess, and its permit is
  released without hanging (`failed_requests >= 1`, `per_resource_admissions["cuda:0"] >= 1`).
- `CpuAndCudaLanesAreIndependentInRuntime` — CPU work completes while a real CUDA lane
  is present, and both `cpu` and `cuda:0` admissions are recorded.

The CUDA `ImageInput` rejection itself is unchanged and still regression-tested
(`CudaInputSubmitStillRejected`, `PreprocessBatchCuda.CudaRejectsHostImageInputs`).

### CPU-to-CUDA Input Staging (Phase 9.3.2)

Phase 9.3.2 closes the single gap Phase 9.3.1 identified: it adds a **Runtime-owned
H2D step** so a CPU `ImageInput` can now drive a CUDA `ModelInstance` through the
**asynchronous** `submit()` path to a real result. It does **not** relax the
external-CUDA-`ImageInput` ban and does **not** change the synchronous APIs.

**Where staging lives.** A new async-only boundary, `ModelInstance::infer_async()` /
`infer_batch_async()` (called only by `Runtime`'s execution worker via
`execute_batch()`), performs the upload. When the instance's committed engine device
is CUDA and the incoming image is a Runtime-owned CPU image, it uploads the pixels
to the device and runs the unchanged synchronous `InferenceEngine::infer()` on the
staged image. The plain synchronous `infer()`/`infer_batch()` still require the image
device to match the engine device and never stage — so `ModelInstance::infer()` /
`InferenceEngine::infer()` semantics are untouched (§1.2). CPU instances do no staging.

**Ownership and lifetime.** `CudaInputBuffer` (`include/vie/runtime/cuda_input_staging.hpp`,
`src/runtime/cuda_input_staging.cpp`) is a move-only RAII owner of exactly one staged
input. Internally it reuses the existing device-memory facility: `Tensor::from_host()`
allocates a `cudaMalloc` buffer via `TensorStorage`, issues a **stream-ordered H2D on
the engine's own `DeviceContext` stream** (`cudaMemcpyAsync` + `cudaStreamSynchronize`),
and returns a `Tensor` whose `shared_ptr<TensorStorage>` `cudaFree`s the memory on
destruction. No raw device pointer is ever the sole owner, and none is handed to the
caller. Because `InferenceEngine::infer()` is host-synchronous (the backend/D2H
complete before it returns), the buffer is held for the whole
`staging → preprocess → backend → decoder → D2H` span by the local `CudaInputBuffer`
in `infer_async()`; it is freed on normal return **and** on every exception path via
RAII (§4/§6.4). CUDA Batch stays **1** (the aggregator still forces size 1 upstream);
`infer_batch_async()` stages each image into its **own** buffer, never shared.

**Sizing/validation.** `cuda_stage_byte_count()` is CUDA-free and computes the exact
transfer span from the validated image layout — `row_bytes = row_stride_bytes ?:
width*channels*dtype_size`, `total = row_bytes*height` — with checked multiplies and
rejects of null data, non-positive dimensions, unsupported dtype, and a stride smaller
than one packed row, plus integer-overflow detection. It never uses an unvalidated
`width*height*channels`. `InferenceRequest::adopt_cpu_pixels()` repacks every source
row to its packed width before the request enters the queue (ROI padding is dropped,
not copied), so `CudaInputBuffer::stage()` always sees a tightly packed CPU image.
The on-device image keeps width/height/channels/pixel-format with a packed row stride
and is tagged `device == Cuda` (never a CPU pointer mislabeled as CUDA).

**Stream order.** The H2D runs on the engine's committed stream and is synchronized
before `stage()` returns; the subsequent preprocess/backend work is issued on the same
stream, so ordering is guaranteed by stream order plus the synchronous upload, with no
extra global `cudaDeviceSynchronize` introduced by the staging layer (§3.3, §7).

**Why external CUDA `ImageInput` is still rejected.** The `submit()`-time ban
(`image.device != Cpu` → `ErrorCode::Unsupported`, §6.3/§12.1) survives: an external
bare device pointer has no provable ownership across the unbounded queue delay. Only
pixels the Runtime itself deep-copied at submit are ever uploaded, so there is no
GPU-side use-after-free. This is a different input path and is kept distinct.

**CPU-only build isolation (§8).** `cuda_input_staging.hpp`/`.cpp` include no CUDA
headers; they build only on `Tensor`/`DeviceContext`/`ImageInput`. In Build A/B
(CUDA OFF) the engine can never report a CUDA device, so `infer_async()` takes the
plain synchronous path and `stage()` is unreachable. Verified at **link** level: the
CPU-only `libvie.a`/`vie_tests` contain no raw CUDA runtime symbols (`cudaMalloc`/
`cudaMemcpy`/…), and `ldd` shows no CUDA libraries; the CUDA allocation entry points
are the existing CPU-only stubs that throw `Unsupported`.

**Real-CUDA test coverage added in Phase 9.3.2** (`test_execution_scheduling.cpp`):
- CPU-only (`CudaInputStagingUnitTest`, all builds): byte-count layout/stride,
  overflow, non-positive/unknown-dtype/null rejection, and the move-only/empty
  ownership contract.
- Build D (`CudaSchedulingRuntimeTest`):
  - `AsyncCpuImageDrivesRealCudaDetectionNonEmpty` — **the core A1/A3 proof**: a
    real bus.ppm submitted to a CUDA `yolov8n` returns a **non-empty** detection
    set through the async path with `execution_device == Cuda` (no silent CPU
    fallback).
  - `CudaModelCpuImageIsStagedAndSucceedsAsync` / `CpuAndCudaLanesAreIndependentInRuntime`
    — the Phase 9.3.1 failure assertions are now **success** assertions
    (`failed_requests == 0`, permit admitted on `cuda:0`).
  - `MutatingSourceAfterSubmitDoesNotChangeResult` / `FreeingSourceAfterSubmitStillSucceeds`
    — §9.4 ownership: the deep copy at submit, not the caller buffer, feeds staging.
  - `CudaInputBufferStagesAndCopiesExactly` (H2D byte-for-byte round-trip),
    `...MovePreservesOwnership`, `...RejectsNonCpuSource`,
    `...RepeatedStageFreesWithoutLeak` — §9.2 buffer semantics.
  - `StagingFailureRejectsFutureAndLaterRequestSucceeds` — §9.5 exception path.
  - `ConcurrentCudaRequestsDoNotShareInputBuffer` — §9.6 (each request decodes its
    own staged copy).

### CUDA Execution Provider Placement Verification (Phase 9.3.2.1)

Phase 9.3.2 proved a CPU image drives a real CUDA `ModelInstance` to a non-empty
result and tagged `execution_device == Cuda`. That tag identifies the **device
the Runtime committed and the engine was configured for**; it is **not** proof
that every ONNX graph node was placed on the `CUDAExecutionProvider`. With
`cuda_cpu_ep_fallback = true` ORT is explicitly allowed to keep some nodes on
CPU, so the placement was unmeasured. Phase 9.3.2.1 closes that evidence gap with
**node-level ONNX Runtime profiling** of the *real asynchronous Runtime session*.

**Four distinct concepts — never conflated:**
- `execution_device` — the device the Runtime lane/engine ran on (Phase 9.3.2).
- **EP placement** — which `ExecutionProvider` ORT assigned each *graph node* to
  (this phase; read from `args.provider` in the profile).
- **CUDA kernel execution** — GPU kernel time recorded for a CUDA-placed node.
- **Host/device memory transfer** — ORT-inserted `Memcpy*` movement between EPs.

**Profiling mechanism (opt-in, default off, production path unchanged):**
- ORT version: **1.20.1** (`/opt/onnxruntime-linux-x64-gpu-1.20.1`).
- API: `Ort::SessionOptions::EnableProfiling(prefix)` is called on the **same**
  `SessionOptions` that builds the inference session (never a second session),
  and `Ort::Session::EndProfilingAllocated()` flushes it and returns the
  Chrome-trace JSON path.
- Plumbing: `ModelConfig::ort_enable_profiling` + `ort_profile_prefix` →
  `InferenceEngine::load` → `OnnxRuntimeCudaBackend::set_profiling` → session;
  flush via `InferenceEngine::end_profiling()` / `ModelInstance::end_profiling()`
  (which serializes on the instance mutex so it runs strictly after any in-flight
  inference). Both flags default off, so `EnableProfiling` is never called and the
  normal Session is byte-for-byte identical. CPU-only builds return `""`.
- Output is written to a configurable directory (tests use
  `std::filesystem::temp_directory_path()`), never the source tree.
- **Lifecycle:** profiling is fixed at *session creation* (there is deliberately
  no runtime on/off toggle, which would need unsafe concurrent control). Enable
  = set `ort_enable_profiling` before `load`; events accrue across every
  `Session::Run`; flush = `end_profiling()` (`EndProfilingAllocated`) which must
  be called *before* the Session is destroyed to guarantee a complete trace. If
  flush fails it surfaces as a `VieError`; a missing/unwritable profile is never
  reported as success.
- **Disable / default:** both config flags default off, so `EnableProfiling` is
  never called, no file is produced, and `end_profiling()` returns `""`.
- **Concurrent-session file isolation:** each enabled session gets a distinct
  `ort_profile_prefix` (tests embed the model/id in the prefix under the temp
  dir), and ORT appends its own pid/timestamp suffix, so two concurrently
  profiled sessions never overwrite the same trace file (§4.2).

**Parser** (`include/vie/diagnostics/ort_profile_parser.hpp`,
`src/diagnostics/ort_profile_parser.cpp`, CUDA-free / ORT-free, built into every
build; uses the existing `nlohmann::json` dependency internally, plain structs
publicly). Real ORT 1.20.1 emits a **bare top-level JSON array**; each executed
node appears as three events (`*_fence_before`, `*_kernel_time`, `*_fence_after`),
and **only `*_kernel_time` carries `args.provider`**. Two independent views are
kept so nothing is conflated (§5.4):
- **Provider view (executed nodes)** = events with a string `args.provider` (the
  authoritative placement). `*_fence_*` markers and Session/thread events are
  never counted as executed nodes. Provider substring classification: contains
  `cuda` → CUDA, `cpu` → CPU, empty → **Unknown** (never silently folded into
  CPU), otherwise Other.
- **Category view (raw events)** = every trace entry bucketed by its `cat` field
  (`category_node_events`, `category_session_events`, `category_threadpool_events`,
  `category_memcpy_events`, `category_other_events`, `category_unknown_events`).
  Because each executed node emits three Node-category events, the raw
  `category_node_events` (537) far exceeds the executed-node count (179) — this
  is why Profile Node Events must never be read as the original node count.
- A **memcpy/transfer event** = a provider-carrying node whose op/name contains
  `memcpy`; it is tallied BOTH as a placed node and as a transfer, so the
  placement view and the transfer view never contradict each other.
- **Incomplete profiles (§5.5):** a Node-category event that is neither a fence
  marker nor carries a provider is an executed node we cannot attribute; it is
  kept as `unknown_provider_events` (never guessed) and sets `incomplete = true`.
  `unknown_provider_events > 0` ⇒ placement is at best INCONCLUSIVE.
- **Original ONNX Nodes** are not derivable from a profile; the parser leaves
  `original_onnx_nodes_known = false` and only the CLI/caller can supply it, so
  the count is never fabricated (§5.4, §6).

**Provider Placement auditor CLI** (`tools/vie_ort_profile_inspect.cpp`, target
`vie_ort_profile_inspect`, built in every configuration, CPU-only — it reads an
already-generated profile file and never loads a model, creates a Session, or
touches CUDA): `vie_ort_profile_inspect <profile.json> [--onnx-nodes N]` prints
the `Profile / Events / Provider Placement / Graph Information / Conclusion`
report (§6). The `Conclusion` line distinguishes *all-CUDA*, *confirmed CPU EP
placement*, and *INCONCLUSIVE (unknown nodes)*, and never treats a memcpy event
as a CPU fallback. Exit code is non-zero for an unparseable file or an
incomplete (unattributable-node) profile.

**Measured results (`test_execution_scheduling.cpp`, Build D, bus.ppm 810×1080,
async `submit()`, single graph execution, same session):**

| Configuration | placed nodes | CUDA | CPU | Other | Unknown | Transfer (memcpy) | raw Node-cat events | incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| `yolov8n`, `cuda_cpu_ep_fallback = true` | 179 | 177 (98.88%) | 2 (1.12%) | 0 | 0 | 4 | 537 | no |
| `identity.onnx`, strict CUDA (control) | ≥1 | all | 0 | 0 | 0 | 0 | ≥3 | no |

- **A3 acceptance = FAIL.** The node-level evidence confirms two CPU EP nodes
  exist for `yolov8n` while `fallback = false` cannot even create the Session.
  Per the acceptance rule, a confirmed CPU EP Node Event means placement is NOT
  fully CUDA — the honest verdict is FAIL, not PASS. (Total trace events 541 =
  537 Node + 4 Session; every one of the 179 executed nodes is attributed, so
  `unknown = 0` and `incomplete = no` — the FAIL is a genuine CPU placement, not
  a parsing gap.)

- The **only two CPU-placed nodes** are `/model.10/Resize` and `/model.13/Resize`
  (`op_type == Resize`). These — not the memcpys — are the genuine CPU fallback and
  the reason `cuda_cpu_ep_fallback` must be `true`.
- The **four "memcpy nodes"** are `Memcpy` / `Memcpy_token_0..2` with op types
  `MemcpyFromHost` / `MemcpyToHost`, and every one is attributed to
  **`CUDAExecutionProvider`**. They are ORT-inserted host/device copy kernels
  that shuttle tensors to and from the two CPU `Resize` nodes; a `Memcpy` event is
  **not** evidence of a CPU compute fallback.
- Strict mode (`cuda_cpu_ep_fallback = false`) fails at **session creation**
  (`ModelInstance::load`, never reaching `submit`) with ORT:
  `"This session contains graph nodes that are assigned to the default CPU EP, but
  fallback to CPU EP has been explicitly disabled by the user."` No inference
  profile exists in that case, and none is fabricated.
- The identity strict-CUDA control proves the profiler genuinely recognizes CUDA
  placement (0 CPU / 0 unknown / 0 transfer), so a CPU node in the `yolov8n` trace
  is a real signal rather than a parser artifact.

**What is now proven / still not proven:** a `yolov8n` async CUDA inference
places 177 of 179 nodes on CUDA and 2 `Resize` nodes on CPU, with four CUDA-side
memcpy transfers. Profiling is diagnostic only; its ON timings are not product
performance numbers, and the default production path keeps it off.

**Tests added (Phase 9.3.2.1):**
- `OrtProfileParserTest` (22, all builds): array/object roots, CUDA/CPU/Other/
  Unknown classification, fence & Session exclusion, memcpy dual tally, missing
  args/provider, empty/invalid JSON, repeated-node vs unique-name, raw category
  tallies (Node/Thread/Memcpy/Other), provider-less Node event → Unknown +
  `incomplete`, complete-profile → not incomplete, `original_onnx_nodes` left
  unknown, file I/O.
- `CudaSchedulingRuntimeTest.Yolov8nFallbackOnAsyncProfileIsCudaWithResizeFallback`
  — the core node-level EP audit of the real async Runtime session (asserts
  `unknown == 0`, `incomplete == false`, `category_node_events > total_node_events`).
- `...ProfilingDisabledByDefaultEmitsNoProfile` — default path unchanged.
- `...Yolov8nStrictCudaFallbackOffFailsAtLoad` — strict failure stage + message.
- `OnnxRuntimeCudaBackend.StrictIdentityProfilePlacesAllNodesOnCuda` — positive
  strict-CUDA control proving the profiler reads CUDA placement.

### YOLOv8n Resize CUDA Compatibility Investigation (Phase 9.3.3)

This phase is an **investigation**, not a production change. It answers *why* the
two `Resize` nodes land on the CPU EP and whether strict CUDA is reachable. No
Runtime / Backend / Aggregator / Staging / Decoder code was modified; only
read-only diagnostic scripts under `tools/diagnostics/resize_cuda_compat/` and
this document were touched. Baseline: `yolov8n.onnx` SHA256
`65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5`.

**Root cause — CONFIRMED (category A: CUDA EP operator-coverage limit, gated by
opset version).** ORT **1.20.1** registers a CUDA `Resize` kernel for the
`Resize` v13–v18 schemas but **not** for the `Resize` v19 schema (opset ≥ 19,
which added `antialias`). `yolov8n.onnx` is exported at **opset 20**, so its
`Resize` nodes resolve to the unregistered v19/v20 kernel and are assigned to the
default CPU EP. Verbose ORT logs state it directly:
`CUDA kernel not found in registries for Op type: Resize`. This is *not* caused
by dynamic shapes, `scales`-vs-`sizes`, graph partitioning, the ONNX structure,
or a VIE backend misconfiguration (each ruled out below).

**The two nodes (read from the raw ONNX, not inferred from the profile):**

| item | `/model.10/Resize` | `/model.13/Resize` |
|---|---|---|
| Op / domain / opset | Resize / ai.onnx / 20 | Resize / ai.onnx / 20 |
| mode / coord / nearest | nearest / asymmetric / floor | nearest / asymmetric / floor |
| cubic_coeff_a | -0.75 | -0.75 |
| X (dtype/shape) | float32 `[1,256,20,20]` (from Mul) | float32 `[1,128,40,40]` (from Mul) |
| roi | omitted (`""`) | omitted (`""`) |
| scales | **constant initializer** `[1,1,2,2]` (shared `/model.10/Constant_output_0`) | same constant |
| sizes | absent | absent |
| output | float32 `[1,256,40,40]` | float32 `[1,128,80,80]` |
| producer / consumer | Mul → Resize → Concat | Mul → Resize → Concat |

`onnx.checker` and strict `shape_inference` both PASS. Static shapes, constant
`scales`, standard nearest — the most-supported configuration, yet still CPU.

**Minimal repro (`resize_cuda_probe`, same ORT 1.20.1 GPU + same strict config
key as production).** Isolated single-node Resize models change one factor at a
time; every valid variant at opset 20 reproduces `kernel not found` → CPU, while
the SAME model at opset 13–18 runs the Resize on CUDA:

| opset | 13 | 15 | 16 | 17 | 18 | 19 | 20 | 21 |
|---|---|---|---|---|---|---|---|---|
| CUDA `Resize` kernel found | yes | yes | yes | yes | yes | **no** | **no** | **no** |
| provider placement (profile) | CUDA | CUDA | CUDA | CUDA | CUDA | CPU | CPU | CPU |

Independent of `mode` (nearest/linear), `coordinate_transformation_mode`
(asymmetric/half_pixel), rank (3D/4D), constant-vs-dynamic `scales`, `scales`-vs
-`sizes`, or static-vs-dynamic batch — the ONLY variable that flips placement is
the **opset**. A single isolated node reproduces it, which rules out graph
partitioning (D) and export structure (C); the production probe call mirrors
VIE exactly, which rules out a VIE backend defect (F).

**The fix is proven end-to-end on the real model.** Version-converting the
baseline to opset 18 (`downgrade_opset.py`, read-only, writes a copy):
- Strict CUDA (fallback **OFF**): `SESSION_CREATE=OK`, real inference runs.
- Profile: **175/175 nodes on CUDA, CPU=0, Unknown=0, and the four `Memcpy`
  transfer events are now 0** — those memcpys were exactly the CPU↔CUDA
  boundary shuttles around the two CPU `Resize` nodes.
- Numerical equivalence of the conversion (opset 20 vs opset 18, CPU EP, same
  input): `MaxAbsErr = 0.000e+00`, `MeanAbsErr = 0.000e+00`, no NaN/Inf.
- Steady-state latency (profiling OFF, 80 runs, same GPU): opset-20 fallback-ON
  ≈ 5.12 ms mean / 6.82 ms p95 vs opset-18 strict-CUDA ≈ 3.94 ms mean /
  5.02 ms p95 (~23% faster, tighter tail) — the CPU `Resize` + memcpys are not
  free, so "just accept the fallback" costs measurable latency.

**Option assessment:**
- **A — adjust export opset (≤18, ≥13): RECOMMENDED.** Evidence-backed, zero
  semantic change (`MaxAbsErr=0`), removes the CPU fallback and the memcpys,
  enables strict CUDA, and even improves latency. Applies to any model, so it
  is a model-export/deployment step, not YOLO-specific logic in the backend.
- **B — ONNX graph optimization:** cannot create a missing CUDA kernel; folding
  the constant `scales` does not change placement. Not useful for this cause.
- **C — upgrade ORT:** plausible (newer builds may register a Resize-v19 CUDA
  kernel) but **untested here** — only the 1.20.1 GPU package is available and
  the Python ORT in this env is CPU-only. Acknowledged evidence gap.
- **D — custom CUDA Resize op:** unnecessary; a kernel already exists for the
  v18 schema. Would add YOLO-specific complexity to a general backend — rejected.
- **E — accept the CPU fallback:** works only with fallback ON, is *not*
  GPU-only, and costs the ~23% latency + memcpys measured above. Acceptable as a
  stopgap, inferior to A.

**Recommended next step:** adopt **Option A** — export/deploy YOLOv8n (and similar
Ultralytics models) at **opset 18**, which turns the Phase 9.3.2.1 **A3 = FAIL
into PASS** with no VIE code change. Re-run this probe to confirm placement if
the ORT version changes (upgrading ORT is an untested alternative).

**Diagnostic scripts (read-only, non-production):** `inspect_resize_nodes.py`
(Phase-2 node dump), `build_minimal_resize_model.py` + `opset_sweep.py` (minimal
variants), `resize_cuda_probe.cpp` (CUDA strict/profiling/verbose-log/optimize-
dump/perf probe mirroring the production ORT calls), `downgrade_opset.py`
(opset-18 copy for the fix test). They write only to an explicit output dir and
never modify the baseline model or test data.

### Model Deployment Compatibility Workflow (Phase 9.3.4)

Phase 9.3.3 explained *why* the two YOLOv8n `Resize` nodes landed on the CPU EP
(ORT 1.20.1 has no CUDA `Resize` v19 kernel, so an opset-20 `Resize` falls back
to CPU) and proved that converting the model to opset 18 yields a strict-CUDA,
numerically identical run. Phase 9.3.4 turns that finding into a repeatable,
model-agnostic workflow made of independent tools. **No Runtime, Backend,
Engine, Staging, Aggregator or Decoder code was changed** to add it.

**Model export vs model conversion.** *Export* (training frameworks -> ONNX)
decides the opset a model is written with and is owned by the model producer.
*Conversion* (one ONNX opset -> another) is a deployment-time repair for a
specific ORT/kernel gap. They are separate responsibilities: VIE loads whatever
ONNX it is given and does not re-author models.

**Why VIE never auto-converts.** `InferenceEngine::load()` / the backends do not
rewrite opsets, silently lower versions, overwrite the user's file, or enable
CPU fallback to mask an incompatibility (see "Strict CUDA execution" above). An
incompatible operator is reported, not hidden. Conversion is always a separate,
explicit, user-invoked step so the original artifact and the exact applied
transform stay auditable.

**Explicit opset conversion** -- `convert_onnx_opset.py` (standalone CLI, python
`onnx.version_converter` only; never imported by the C++ Runtime). It refuses
`output == input`, refuses an existing output without `--overwrite`, never
touches the input, records original/converted SHA256, the source/target opset
and the converter version, then runs `check_model` and `infer_shapes`. It exits
non-zero and reports distinct states for *conversion failed*, *converted but
checker failed* and *converted but shape-inference failed* -- it never swallows
an error and claims success.

```
python3 tools/diagnostics/model_deployment/convert_onnx_opset.py \
  --input testdata/onnx/yolov8n.onnx \
  --output /tmp/yolov8n_opset18.onnx --target-opset 18
```

**Numerical equivalence** is decided by data, not by node counts. The
orchestrator feeds the *same* real preprocessed `bus.ppm` tensor through the
original and converted model on the CPU EP and reports MaxAbs / MeanAbs / RMSE /
MaxRel / MeanRel plus NaN/Inf counts and a shape/dtype compatibility gate
(default tolerance `MaxAbs<=1e-5`, `MeanAbs<=1e-6`; never widened to pass). The
final `DetectionResult` is compared via VIE's *own* decoder + NMS (reused
through `vie_deploy_verify`, not reimplemented) with IoU-based matching rather
than array-index equality.

**Strict CUDA provider placement** reuses the sanctioned GPU path: `vie_deploy_verify`
loads the converted model into a real `InferenceEngine` (CPU pixels are staged to
the device for the CUDA run), enables ORT profiling through
`ModelConfig::ort_enable_profiling`, runs a real inference and calls
`end_profiling()`. The produced trace is classified by `vie_ort_profile_inspect
--json`, which shares the exact `ort_profile_parser` the CUDA E2E tests use --
placement is never inferred from the CPU-only python `onnxruntime`. The GPU-only
rule is: session created, inference succeeded, profile complete, and CUDA nodes
== total nodes with CPU nodes == 0 and Unknown == 0; `Memcpy` events are reported
separately and are not counted as CPU nodes.

```
python3 tools/diagnostics/model_deployment/validate_model_deployment.py \
  --model testdata/onnx/yolov8n.onnx --run-conversion --target-opset 18 \
  --target-device cuda --report /tmp/yolov8n_deployment_report.json
```

**Reading the verdict.** `PASS` means every required check ran and cleared; a
strict-CUDA deployment only reaches it with model validity + numeric equivalence
+ detection equivalence + all-node-CUDA placement + profile completeness. Any
required check left unrun is `INCOMPLETE` (0 is never used to fake a run), an
explicit failure is `FAIL`, and a missing capability is `UNSUPPORTED`. Creating
a session, passing `check_model`, or a successful `version_converter` run are
never, on their own, treated as deployment compatibility.

**Verified case (YOLOv8n, ORT 1.20.1 GPU, RTX 4080, CUDA 12.6).** opset 20 ->
strict CUDA **fails session creation** with the CPU-EP-disabled message (the
`Resize` v19 gap; classified `cuda_ep_unsupported_op_would_fallback_to_cpu`, not
a VIE fault). The opset-18 conversion of the *same* model: checker + shape
inference pass, CPU numeric comparison **MaxAbs = MeanAbs = RMSE = 0.0**, VIE
detections identical (5/5 matched, 0 error), strict-CUDA placement
**175/175 nodes on CUDA, 0 CPU, 0 Unknown, 0 Memcpy**, and full-pipeline latency
CPU ~45 ms vs strict-CUDA ~6 ms.

**Why re-validate after an ORT change.** The kernel-coverage boundary (which
`Resize` schema versions the CUDA EP registers) is a property of the ORT build,
not of the model. A different ORT version can move or remove the opset-19 gap,
so opset 18 is recorded *only* as the verified remedy for this YOLOv8n + ORT
1.20.1 combination -- never as a rule for all YOLO, all ONNX models, all ORT
versions, or a guarantee that conversion preserves semantics (that is what the
numeric and detection checks are for). Re-run the workflow after any ORT upgrade.

**Tools:** `convert_onnx_opset.py`, `validate_model_deployment.py` (orchestrator +
JSON report), `vie_deploy_verify.cpp` (CMake target reusing the VIE engine),
`yolov8n.manifest.json` (fixture manifest), and `test_model_deployment.py`
(pytest; conversion/decision/maths tests are CPU-only, real-model CUDA tests
skip when no GPU/tools are present -- a skip is never a pass). Build the native
tools with `cmake --build <build> --target vie_deploy_verify vie_ort_profile_inspect`.

### Semantic Segmentation Task (Phase 9.4)

Phase 9.4 introduces an independent `TaskType::SemanticSegmentation`, strictly
distinguished from the existing `TaskType::Segmentation` (which is instance
segmentation from YOLOv8-seg). The task turns a dense per-pixel class-logit
tensor into a per-pixel class-id map restored to the original image grid.

**Recognizer.** `Deeplabv3SemanticSegmentationRecognizer` accepts only graphs
whose single 4-D output has (a) axis 1 in `[2, num_classes]`, (b) axes 2/3
static and `>= 1`, (c) an axis-1 count matching a terminal 1x1 Conv weight's
first dimension, and (d) a head path from output to Conv that walks only
`Resize`/`Upsample` and shape/passthrough nodes. `Softmax`, `LogSoftmax`,
`ArgMax`, or any pooling on that head path disqualifies the graph, so the
recognizer never accepts a post-processed (already-argmaxed or already-
softmaxed) segmentation model. SE-block `GlobalAveragePool`s live inside the
backbone and are therefore allowed. When it also matches any other task the
registry returns `Ambiguous`; the model name is never used as evidence.

**Manifest.** Schema 1.0 gained a `semantic_segmentation` block (`output_name`,
`channel_axis`, `logits_semantics`, `output_interpolation`,
`class_id_interpolation`). `ManifestGenerator` fills them with Inferred/Observed
provenance; `input.color`, `input.normalization`, `input.resize`, and
`input.interpolation` stay Unresolved until the caller records the transform
contract as User provenance (see `vie_semantic_verify.cpp`, which pins RGB +
ImageNet mean/std + stretch bilinear with a User entry so
`ensure_provenance()` accepts the manifest).

**Decoder.** `SemanticSegmentationDecoder` is the CPU implementation; Phase 13 adds
`SemanticSegmentationCudaDecoder` and a device dispatcher (see the Phase 13
section below). The CPU class still takes
class-logits in NCHW (`channel_axis` from the manifest), and for every
original pixel computes `mx = ox * scale_x + geometry_model_origin_x` and
samples each channel bilinearly. Before sampling it resizes the logits to
the preprocessed grid bilinearly when their spatial shape differs. ArgMax
uses a strict `>` comparison so ties resolve to the smallest class id
deterministically. NaN / Inf inputs raise `InvalidArgument`; wrong rank,
wrong channel count, or wrong dtype raise `InvalidArgument`; a CUDA tensor
raises `Unsupported` (there is no CUDA decoder pretending to be CPU).

**Engine.** `InferenceEngine::infer_semantic_segmentation(image, timing*)`
follows the same shape as `infer_classification`: preprocess -> execute ->
decode. On the CPU path the full logits are copied back (recording
`d2h_bytes` / `d2h_ms`) before the CPU decoder; on the CUDA path the decoder
downsamples on the device and only the compact class map crosses the bus
(Phase 13). `SemanticSegmentationResult` carries class_ids, original
width/height, logits width/height, num_classes, and `execution_device` (Cpu or
Cuda, the device the decoder actually ran on). `infer()` and `infer_batch()`
throw `Unsupported` for this task so callers cannot mistake it for
detection. `ModelInstance::infer_semantic_segmentation` is a locked
passthrough with the usual metrics.

**Real-model E2E.** `tools/diagnostics/semantic_segmentation/vie_semantic_verify.cpp`
recognizes `testdata/onnx/deeplabv3_mobilenet_v3_large_520_opset18.onnx` (a
torchvision `COCO_WITH_VOC_LABELS_V1` DeepLabV3-MobileNetV3-Large exported at
opset 18 with a fixed `1x3x520x520` input, 168 nodes, SHA256 `ba4c4376…`),
generates and reloads the schema 1.0 manifest, then dumps the three raw
artifacts `reference_infer.py` compares against. CUDA runs the ONNX Runtime
CUDA EP with `cuda_cpu_ep_fallback=false` (Strict CUDA) and stages the
input via `CudaInputBuffer` — there is no silent fallback to CPU.

**ORT provider placement (Strict CUDA).** Re-running the tool with
`--profile-prefix <path>` and feeding the Chrome trace to
`vie_ort_profile_inspect` reports 348 node events across 2 executed
inferences, 174 unique nodes, **0 CPU-EP nodes, 0 unknown-provider nodes,
and 0 Memcpy events**; every executed node lands on the CUDA Execution
Provider. The 22,713,600-byte D2H reported by the tool is a separate
`Tensor::to(Cpu,0)` copy VIE performs before the CPU decoder, not an
ORT-side fallback.

Measured on `testdata/images/bus.ppm` (810x1080):

| Lane         | preprocess | execute | d2h | decode | total | d2h bytes |
|--------------|-----------:|--------:|----:|-------:|------:|----------:|
| CPU (ORT)    |   11.5 ms  | 44.4 ms |  –  | 43.0 ms| 98.9 ms|   0       |
| CUDA (ORT, strict) | 1.1 ms | 5.8 ms | ~5 ms | 51.3 ms | 58.3 ms | 22,713,600 |

Numeric comparison against the torchvision reference (same transform
contract, `F.interpolate bilinear align_corners=False` for both
preprocess and class-map upsample):

| Comparison           | preprocess MaxAbs | logits MaxAbs | logits RMSE | PixelAcc | MeanIoU |
|----------------------|------------------:|--------------:|------------:|---------:|--------:|
| CPU VIE vs reference |         5.6e-06   |     2.8e-05   |   4.3e-06   |  0.99785 | 0.9941  |
| CUDA VIE vs reference|         9.5e-07   |     1.6e-02   |   2.2e-03   |  0.99784 | 0.9941  |
| CPU vs CUDA (VIE)    |         –         |     1.6e-02   |   2.2e-03   |  0.99993 |   –     |

All three lanes agree on the class histogram (background ≈ 56.5 %, bus
≈ 31.5 %, pottedplant ≈ 11.9 %); no NaN/Inf, no out-of-range class ids,
oor `out_of_range_ids > 0` was observed. Disagreement pixels between VIE
and the reference are edge pixels where two logits nearly tie; the strict
`>` tie-break is what keeps the count deterministic across runs.

**Compatibility matrix (Phase 9.4).**

| Configuration | Recognizer | Decoder | Engine API | Real-model verify | CUDA E2E |
|---------------|-----------|---------|------------|-------------------|----------|
| Build A: Mock backend, CPU       | yes (synthetic ONNX graph) | yes | yes | n/a | n/a |
| Build B: ORT, CPU                | yes | yes | yes | PASS | n/a |
| Build C: ORT, CUDA (strict EP)   | yes | yes (CPU class + CUDA decoder) | yes | PASS | PASS (compact class map only) |
| Build D: ORT, CUDA + EP fallback | allowed but discouraged | same | same | same | same |

Tests: 19 new cases in `tests/test_semantic_segmentation.cpp` cover §12.1
decoder contract, §12.2 GeometryTransform inverse (landscape / portrait /
stretch / letterbox / odd padding / low-resolution logits), §12.3
recognizer + manifest (synthetic `seg_graph` positive and five negative
shapes), and §12.4 mock-engine end-to-end. Combined with the existing
suite the CPU-only build reports 294 / 294 passing.

**Scope and current limits.** Phase 9.4 delivers the first end-to-end
semantic-segmentation lane; the following are **not** supported and are
not silently proxied:

- Only the DeepLabV3-MobileNetV3-Large representative graph is proven.
  The recognizer may accept other graphs whose structure matches, but no
  other model has been verified.
- `SemanticSegmentationDecoder` has a CUDA counterpart as of Phase 13
  (`SemanticSegmentationCudaDecoder`: GPU ArgMax + GPU-side bilinear resize,
  only the class map is copied back). Batch decoding remains unavailable.
- `SemanticSegmentationResult` is one map per call. There is no
  `infer_semantic_segmentation_batch()` and no async submit path for this
  task in this phase; the `infer_batch()` entry point rejects
  `TaskType::SemanticSegmentation` explicitly.
- Only the `deeplabv3_semantic_seg_raw` protocol is accepted. `Softmax`ed
  probabilities, `ArgMax`ed class id maps, or non-NCHW outputs throw
  `Unsupported`; the recognizer does not fabricate a fallback protocol.

**Tools:** `vie_semantic_verify.cpp` (CMake target, produces
`<prefix>_input.f32 / _logits.f32 / _classids.i32 / manifest.json`),
`reference_infer.py` (torchvision reference + numeric deltas JSON report),
`visualize_result.py` (class-map PNG + overlay + histogram).

### Inference Configuration (Phase 12)

**Location**: `include/vie/config/inference_config.hpp`, `src/config/inference_config.cpp`

One typed policy object per task, resolved once into an immutable snapshot before any decoder runs. The resolver is independent of the decoders: every decoder receives the same `ResolvedInferenceConfig`-derived `DecoderConfig`, so no decoder carries its own defaults for request-scoped policy.

#### Layers and priority

| Layer | Carrier | Meaning |
|---|---|---|
| Per-request | `InferenceOptions` passed to `infer()`, `infer_batch()`, `infer_segmentation()`, `infer_classification()`, `infer_obb()`, `infer_pose()`, `Runtime::submit()` | explicit override for one call |
| Model | `ModelConfig::inference_options` captured at `load()` | defaults for one loaded model |
| Manifest | `nms.*`, `segmentation.*`, `decoder.class_selection` | model/user configuration that is actually usable |
| Framework | `kFramework*` constants in `inference_config.hpp` | last-resort fallback |

Priority: **request > model default > usable manifest value > framework default**. A parameter left unset at every layer takes the framework default; the effective value and its `ParameterSource` are both available on the resolved config.

#### Manifest usability

- `Certainty::Unresolved` and `Certainty::Inferred` values never participate: an inferred "typical YOLO threshold" is not a model fact and cannot become a default.
- A set value without a provenance entry counts as user-supplied (`ParameterSource::Manifest`).
- Framework defaults are never written back into the manifest and are never tagged as model observations.
- `decoder.class_selection` is a structural manifest fact: `all_above_threshold` resolves `multi_label=true`, `best_class` resolves `multi_label=false`.

#### Framework defaults

| Parameter | Default |
|---|---|
| `confidence_threshold` | `0.25` |
| `iou_threshold` | `0.45` |
| `max_detections` | `300` (upper bound `100000`) |
| `pre_nms_top_k` | `4096` (upper bound `4096`) |
| classification `top_k` | `min(5, num_classes)` |
| segmentation `retina_masks` / `binary_mask_gt_zero` / `max_mask_bytes` | no framework default; must come from the model, manifest, or request (`UnresolvedField` otherwise) |

#### Validation and task applicability

- Thresholds must be finite and in `[0, 1]`; `0` and `1` are legal boundaries. NaN, Inf and out-of-range values throw `InvalidArgument`.
- Counts must be `>= 1` and within their stated upper bounds; negatives, zero and overflow throw `InvalidArgument`.
- A non-empty block for a task other than the loaded model's task throws `InvalidArgument` instead of being ignored. Semantic segmentation accepts no policy block at all.
- RT-DETR rejects `iou_threshold`, `pre_nms_top_k`, `class_agnostic_nms` and `multi_label=true` with `Unsupported`; its structural `nms_enabled` is always `false`.
- Invalid values are rejected, never silently clamped to a default.

#### Per-task parameter matrix

| Task | Supported policy fields |
|---|---|
| Detection (YOLOv8, CPU/CUDA decoder) | `confidence_threshold`, `iou_threshold`, `max_detections`, `pre_nms_top_k`, `class_agnostic_nms`, `multi_label` |
| Detection (RT-DETR, CPU/CUDA decoder) | `confidence_threshold`, `max_detections`, `multi_label` (only `false`) |
| Instance segmentation | `confidence_threshold`, `iou_threshold`, `max_detections`, `pre_nms_top_k`, `class_agnostic_nms`, `retina_masks`, `binary_mask_gt_zero`, `max_mask_bytes` |
| Classification | `top_k` (bounded by `num_classes`; no confidence filter in the current decoder) |
| OBB | `confidence_threshold`, rotated-box `iou_threshold`, `max_detections`, `pre_nms_top_k`, `class_agnostic_nms` (class selection is `best_class`) |
| Pose | `confidence_threshold`, person-box `iou_threshold`, `max_detections`, `pre_nms_top_k`, `class_agnostic_nms` (no keypoint-score threshold exists) |
| Semantic segmentation | none (ArgMax only) |

#### Snapshot semantics and batch isolation

- Synchronous `infer*()` resolves the config at entry and runs that snapshot for the whole call.
- `Runtime::submit()` resolves **at submit time** and stores the snapshot in `InferenceRequest`; workers run the stored snapshot. Mutating the caller's `InferenceOptions` after `submit()` returns cannot affect the queued request. `ModelInstance` and its engine are never mutated by a request.
- The dynamic batch aggregator groups by `(ModelInstance identity, config.fingerprint())`. Requests with different effective configs form different groups even on the same model, so a batch never mixes policies; `Runtime::execute_batch()` re-checks the invariant defensively. Different-config requests are not merged to raise the batch hit rate.
- `fingerprint()` encodes task, protocol and every resolved policy value, so config equality is value equality, not object identity.

#### CPU/CUDA consistency

Both CPU and CUDA decoders consume the same resolved `DecoderConfig` built from the snapshot; neither keeps a private default set. The CPU/CUDA parity test feeds identical tensors and the same config to the CPU and CUDA decode paths and asserts equal counts/classes and confidence/coordinate equality within an explicit tolerance (confidence `1e-5`, coordinates `1e-4`).

### Classification and Semantic CUDA Decoders (Phase 13)

Phase 13 adds GPU post-processing for the two remaining single-output tasks. The
CPU decoder classes are untouched; the engine registers thin dispatchers that
route by `DeviceContext::device()`, so the public Engine/Runtime API, the
manifest schema and the Phase 12 config priority chain are unchanged.

**Fingerprint hardening (prerequisite audit).** `ResolvedInferenceConfig::fingerprint()`
is the dynamic-batch grouping key, so two different effective configs must never
serialize identically. The canonical writer now (a) formats every floating-point
value with `std::numeric_limits<double>::max_digits10` (round-trip exact), (b)
uses the classic locale explicitly, (c) folds `-0.0` to `0` and rejects NaN/Inf
at resolution time, (d) tags every optional field with an `unset,` marker so a
defaulted field cannot collide with an explicitly set one, and (e) serializes
independent blocks with `;` separators and an explicit `none` when a block is
absent. Tests cover float round-trip collisions, `-0.0`, optional presence,
field-by-field changes, locale independence and repeated-call stability.

**Classification CUDA Top-K** (`ClassificationCudaDecoder`, protocol names
`classification_logits_raw` / `classification_probabilities_raw`). The input
must be a rank-2 FP32 CUDA tensor (`[1, C]` or `[C, 1]`, `class_axis` 0/1) with
stride 1 along the class axis; the class count must equal
`ResolvedClassificationConfig::num_classes`, and `top_k` comes exclusively from
that resolved config (`resolved.classification().top_k`), never from a private
default. The kernel lane validates every logit on the device (`isfinite`), then
ranks with the same contract as the CPU decoder: descending score, ascending
class id on exact ties, no Softmax, negative logits preserved; `top_k` comes
from the Phase 12 resolver, which bounds it to `[1, num_classes]`, and an
out-of-range request is rejected rather than clamped silently. The
probabilities protocol additionally
checks `[0, 1]` and the FP64 sum within `1e-3`. Only the final `top_k`
`(class_id, score)` pairs are copied back (`top_k * 8` bytes); the full logits
never leave the device. Two control words (an 8-byte fault counter and an
8-byte probability sum) are read back during validation but are not counted in
`d2h_bytes`, matching the RT-DETR convention of reporting payload only.
Non-finite inputs report the lowest offending class id with the exact CPU
error text. CUDA-side failures never retry on the CPU.

**Semantic Segmentation CUDA ArgMax** (`SemanticSegmentationCudaDecoder`,
protocol `deeplabv3_semantic_seg_raw`). Input is a rank-4 FP32 CUDA tensor
`[1, C, H, W]` with `channel_axis == 1`; `C` must equal the manifest
`num_classes`. The lane mirrors the CPU math bit-for-bit: full-tensor
finiteness validation (reporting the first `[c, y, x]`), an optional bilinear
resize of the logits to the preprocessed grid using double-precision corners
stored back as FP32, then per original pixel `mx = ox * scale_x + origin_x`
bilinear sampling in double and a strict `>` argmax that keeps the smallest
class id. The `.cu` file is compiled with `--fmad=false` so the arithmetic
mirrors the CPU expression. Only the int32 class map is copied back
(`width * height * 4` bytes); the logits stay resident. The 8-byte fault index
(and, only on error, one 4-byte diagnostic value) is a control transfer and is
not counted in `d2h_bytes`.

**Kernel and stream usage.** Both decoders reuse the RT-DETR CUDA lane
conventions: legacy default stream (`nullptr`), which is the stream ONNX
Runtime's CUDA EP uses in this project (`has_user_compute_stream=0`,
`do_copy_in_default_stream=1`); `cudaSetDevice(device_id)` once at entry; a
`cudaStreamSynchronize` on that stream, never `cudaDeviceSynchronize`; RAII
`DeviceBuffer`/event pairs keep every temporary alive until its async copy has
completed; every launch is followed by `cudaGetLastError`; request/tensor
contracts (dtype, rank, device id, contiguity) are rejected before any device
work. A non-contiguous CUDA tensor is rejected with `Unsupported` (the existing
Tensor contract), and a CUDA decoder never touches `Tensor::data()`.

**Engine integration.** `infer_classification` and
`infer_semantic_segmentation` no longer perform a full `Tensor::to(Cpu)` before
decoding. On CUDA they hand the device tensor to the decoder, which performs
its own compact copy. `ClassificationResult` / `SemanticSegmentationResult`
still report `output_bytes` (full tensor bytes), and `d2h_bytes` now counts the
compact payload only. The result also carries `d2h_ms` (GPU-event time for the
compact copy); `decode_ms` in the timing struct is wall-clock on CUDA so launch
and synchronization overhead are never hidden. CPU timing keeps the existing
`d2h + validate + topk` composition.

**Measured payloads and timings.** RTX 4080 Laptop, Release build, CUDA 12.6
(`CMAKE_CUDA_ARCHITECTURES=89`), ONNX Runtime 1.20.1 CUDA EP, 3 warmups then 20
samples for the decoder microbenchmark:

| Decoder lane | CPU validate | CPU decode | CUDA validate | CUDA decode | D2H bytes |
|---|---:|---:|---:|---:|---:|
| Classification, 1000 classes, `top_k=5` | 0.0009 ms | 0.031 ms | 0.0085 ms | 0.095 ms + 0.060 ms D2H | **40** (was 4,000) |
| Semantic, 21 classes, 65x65 logits -> 160x120 map | 0.052 ms | 1.096 ms | 0.0095 ms | 0.088 ms + 0.071 ms D2H | **76,800** (was 354,900) |

Real models (`tests/test_cuda_decoder_e2e.cpp`): ResNet18 `[1,1000]` per-call
`decode_ms` 0.075 ms on CPU vs 0.373 ms wall-clock on CUDA (launch + sync
dominates a 4 KB reduction), while end-to-end `total_ms` is 10.42 ms vs 1.75 ms
because execution moves to the GPU. DeepLabV3-MobileNetV3-Large (520x520, 21
classes, `bus.ppm` 810x1080): class maps agree on 99.9864% of pixels (MeanIoU
0.999, 119 of 874,800 pixels differ at near-tie edges between EP kernels; the
decoder parity itself is exact when both paths consume the same logits).
Semantic `d2h_bytes` drops from 22,713,600 to **3,499,200** (84.6% less),
`decode_ms` 75.55 ms (CPU) vs 3.68 ms wall-clock (CUDA).

**Tests and builds.** `tests/test_classification_cuda_decoder.cpp` (8 cases: top_k
1/5/n, negative logits, ties, class axis 0, non-finite parity incl. message
equality, probabilities contract, invalid shapes/dtypes, dispatcher, timing) and
`tests/test_semantic_segmentation_cuda.cpp` (8 cases: identity/letterbox/resize
geometry, ties, non-finite coordinates, contracts, dispatcher, timing) run only
when `VIE_ENABLE_CUDA=ON`; `tests/test_cuda_decoder_e2e.cpp` adds the three real
model lanes. CUDA OFF builds never compile the CUDA branches.

**Scope.** At the end of Phase 13, Pose and OBB kept CPU decoders. Phase 14 adds the Pose
CUDA decoder described below; OBB keeps its CPU polygon NMS. There is no CUDA batch path
(the batch aggregator still rejects CUDA batch > 1), no model-specific logic, and no
implicit CPU fallback.

### Pose CUDA Decoder (Phase 14)

Phase 14 adds the Pose GPU post-processing lane. `PoseDispatchDecoder` routes by
`DeviceContext::device()`: CPU keeps `YoloV8PoseDecoder`, CUDA goes to
`PoseCudaDecoder`. The CUDA path never calls the CPU decoder, and a CUDA failure
never retries on the CPU. The public API, manifest schema 1.0 and the Phase 12
config chain are unchanged.

**Contract.** The decoded tensor must be rank-3 FP32, contiguous, device-matched
and batch 1 (`[1, C, P]` for `channel_axis=1`, `[1, P, C]` for `channel_axis=2`),
with `C == 4 + num_classes + keypoint_count * 3`. The first four channels are
already-decoded `cx, cy, w, h` in model-input space, followed by class scores and
`(x, y, confidence)` triplets. Batch > 1, non-FP32, non-contiguous tensors, wrong
channel counts and missing outputs/geometry are rejected before any kernel runs;
the non-contiguous case now reports `InvalidArgument`, matching the CPU decoder.
Explicit upper bounds (predictions ≤ 1e6, candidates ≤ 1e6, keypoints ≤ 10000,
keypoint buffer ≤ 256 MiB) prevent silent allocation overflow.

**Kernel pipeline.** `filter_decode_kernel` runs one thread per anchor and reads
fields in the exact CPU order: all class scores first (non-finite and probability
range faults are recorded with a 64-bit `atomicMin` key), then Select/AllAbove
filtering with the CPU tie-break (highest score, then smallest class id), then
`cx/cy/w/h` (non-positive size is a fault), then every keypoint. Anchors that
produce no candidate never touch their box/keypoint channels, so a NaN there is
ignored exactly as on the CPU. Candidates are sorted by
`score desc, anchor asc, class asc` (Thrust `sort`, deterministically) and
truncated to `pre_nms_top_k`. A parallel `map_kernel` performs the double-precision
inverse `GeometryTransform`, clamps boxes to `[0, W] x [0, H]`, drops degenerate
boxes, keeps keypoints unclamped with the 1e-3 `inside_image` slack and clamps
keypoint confidence; a single-thread `compact_kernel` preserves the CPU candidate
order. `nms_kernel` is class-aware exactly when the resolved config says so,
uses the CPU float IoU (`IoU >= iou_threshold` suppresses, union ≤ 0 or non-finite
returns 0), and stops at `max_detections`; when NMS is disabled the prefix of
`min(max_detections, candidates)` is kept. `pack_kernel` gathers the final compact
objects and keypoints; only that payload is copied back. Fault messages are
reconstructed from the GPU fault word and match the CPU text byte-for-byte.

**Config.** Both lanes consume the same Phase 12 `PoseDecodeConfig` produced by
`pose_decode_config_from_resolved()` (priority
RequestOverride > ModelConfig > Manifest > FrameworkDefault); the CUDA decoder
keeps no private defaults. `tests/test_pose_cuda_decoder.cpp::RunsTheResolvedPhase12Policy`
feeds one resolved config to both lanes and asserts identical NMS/selection
behavior; the existing `DecoderConfigPolicy.PoseResolvedConfigMapsEveryPolicyField`
and Runtime/fingerprint suites still pass.

**Stream and lifetime.** The lane follows the Phase 13 conventions: legacy default
stream (`nullptr`, the stream ORT CUDA EP uses here), `cudaSetDevice(device_id)` at
entry, `cudaStreamSynchronize` on that stream for the three control-word reads and
the final D2H, never `cudaDeviceSynchronize`; RAII `DeviceBuffer`/event pairs keep
temporaries alive; every launch checks `cudaGetLastError`; `Tensor::data()` is
never called on a CUDA tensor. `d2h_bytes` counts only the compact payload
(`kept * 28 + kept * keypoints * 16`); the two 8-byte fault words (filter and map)
and the three 4-byte counters are control transfers and are not counted, consistent
with the other CUDA decoders. `decode_ms` is wall-clock on CUDA (launch + sync included);
`validate_ms` covers filter + sort, `pose_decode_ms` covers map + compact, `nms_ms`
covers NMS/prefix, and `d2h_ms` covers pack + the two payload copies.

**Measured timings.** RTX 4080 Laptop, Release, CUDA 12.6, ORT 1.20.1 CUDA EP.
Synthetic decoder bench: 8400 anchors, 17 keypoints, 640x480 input, output tensor
1,881,600 bytes, 2 warmups + 5 samples, wall-clock includes launch and sync.

| Candidates | CPU decoder | CUDA wall | validate | pose_decode | NMS | D2H | `d2h_bytes` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.056 ms | 2.21 ms | 0.404 ms | 0.036 ms | 0.071 ms | 0.185 ms | 2,400 |
| 120 | 0.135 ms | 1.89 ms | 0.097 ms | 0.255 ms | 0.076 ms | 0.218 ms | 36,000 |
| 600 | 0.682 ms | 3.62 ms | 0.170 ms | 1.169 ms | 0.455 ms | 0.278 ms | 180,000 |

The CUDA lane is slower than the CPU decoder for these candidate counts because
launch and the control-word reads dominate; it pays off in the end-to-end path.

Real `yolov8n-pose.onnx` E2E (`tests/test_pose.cpp::StrictCudaMatchesCpu`,
640x640, 8400 anchors, 300 pre-NMS candidates, 20 kept, 17 keypoints, 3 warmups +
10 samples, strict CUDA EP with `cuda_cpu_ep_fallback=false`): CPU total
62.63 ms (preprocess 13.32, execute 42.67, decode 4.94 = validate 4.75 +
pose_decode 0.127 + NMS 0.060); CUDA total 10.22 ms (preprocess 0.85, execute 7.43,
decode wall 1.93 = validate 0.315 + pose_decode 0.625 + NMS 0.114 + D2H 0.169),
`d2h_bytes` 6,000 vs output 1,881,600. All 20 CPU detections share an anchor with
the CUDA result; raw EP logits differ by at most 0.6225 px (box), 6.96e-4 (score)
and 0.4023 (keypoint), which is why the E2E comparison uses explicit tolerances.

**Tests and builds.** `tests/test_pose_cuda_decoder.cpp` contains 9 GPU-only cases
(single/multiple/empty detections, axis 1 and 2, 17 keypoints, class policies,
ties and the exact IoU boundary, dropped anchors, CPU-identical fault messages,
tensor/request contracts, Phase 12 policy parity, timing record);
`tests/test_pose.cpp` adds the strict-CUDA real-model E2E. All five builds
(A: CUDA/ORT off, B: ORT CPU, C: OpenCV + ORT CPU, D: ORT CUDA, E: OpenCV +
CUDA) pass, ASan passes 372/372, and compute-sanitizer reports 0 errors for the
Pose CUDA cases. CUDA OFF builds never compile the `.cu` lane.

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
  // Phase 9.2.2
  BatchAggregationConfig batch_aggregation;
  // Phase 9.3
  ExecutionSchedulingConfig scheduling;
};
```

```cpp
struct BatchAggregationConfig {
  bool enabled = true;                             // Enable/disable aggregation
  std::size_t max_batch_size = 4;                  // Max images per batch
  std::chrono::milliseconds max_wait_time{5};      // Max wait before flushing
  std::size_t max_pending_requests = 256;          // Backpressure limit
};
```

```cpp
// Phase 9.3 — resource-level admission policy (NOT backend kernel capability).
struct ExecutionSchedulingConfig {
  std::size_t cuda_max_concurrent_per_device = 1;  // per CUDA device gate; 0 is ILLEGAL (throws)
  std::size_t cpu_max_concurrent = 0;              // 0 = unbounded (worker-governed)
  std::size_t lane_capacity = 0;                   // 0 = fall back to batch_execution_queue_capacity
};
```

**cuda_max_concurrent_per_device**: Max batches the scheduler runs concurrently on the *same* CUDA device. Default `1`. Must be ≥ 1 — `0` throws `VieError(InvalidArgument)` at construction (it does **not** mean unlimited). This is a conservative Runtime admission gate; it does not imply GPU kernel concurrency, multiple streams, or a session pool.

**cpu_max_concurrent**: Concurrency ceiling for the CPU lane. Default `0` = unbounded, preserving pre-Phase-9.3 CPU parallelism (worker count + per-instance mutex remain the only limits).

**lane_capacity**: Shared staging budget across all resource lanes (backpressure). `0` falls back to `batch_execution_queue_capacity`.

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
- **Phase 9.2.2.3**: `Runtime::submit()` throws `VieError(ErrorCode::Unsupported, ...)` synchronously, before any model lookup or queue push, if `image.device != DeviceType::Cpu` (see "Input Device Support" above). This rejection does not create a `std::future` and does not affect `ModelInstance::infer()`/`InferenceEngine::infer()` CUDA support.
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

- ~~**Automatic Batch Aggregation**~~: Implemented in Phase 9.2.2
- **Priority queues**: High-priority tasks preempt low-priority
- **Dynamic thread pool**: Adjust worker count based on load
- **Model sharing**: Share Backend/Decoder between instances (read-only)
- **Detailed profiling**: Per-layer timing, memory tracking
- **Async I/O**: Overlap data loading with inference
- **Auto-tuning batch size**: Dynamically adjust based on model capacity and load
