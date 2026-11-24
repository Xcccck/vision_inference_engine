# Phase 8.3 / 8.3.1 Runtime Production Readiness

## Executive Summary

VIE Runtime (Phase 8 → 8.3.1) is a multi-model execution layer that is safe
for concurrent use under the constraints documented below. Phase 8.3.1
corrected the benchmark methodology and constrained the technical claims to
what is actually supported by code and by experiment.

**Production readiness**: Runtime layer is ready for **multi-model concurrent
inference**. It is NOT a Session-pool or same-model high-concurrency
accelerator; those workloads must create multiple `ModelInstance` objects.

## 1. Runtime Current Capability

Implemented in Phase 8 / 8.1 / 8.2 / 8.3:

| Component | File | Responsibility |
|---|---|---|
| `ThreadPool` | `src/runtime/thread_pool.cpp` | Fixed worker count, task submission, `std::future<T>` return, graceful shutdown. Zero worker count is rejected via `std::invalid_argument`. |
| `ModelInstance` | `src/runtime/model_instance.cpp` | Wraps one `InferenceEngine`. Serializes `infer*` through a mutex. Emits metrics via RAII `InferenceGuard`. |
| `ModelManager` | `src/runtime/runtime.cpp` | Owns `map<string, shared_ptr<ModelInstance>>`. `load/unload/get/list`. Mutex-protected map. |
| `Runtime` | `src/runtime/runtime.cpp` | Composes `ThreadPool + ModelManager + MetricsCollector`. Provides `submit_*` helpers. |
| `MetricsCollector` | `src/runtime/runtime_metrics.cpp` | Per-model inference counters, timings, success rate. Bounded only by unique model IDs. |
| `BackendRegistry` | `src/backend/backend_registry.cpp` | Global factory registry. All methods under `std::mutex`. |

## 2. `ModelInstance` Concurrency Semantics

**Rule**: One `ModelInstance` executes one inference at a time.

```cpp
DetectionBatch ModelInstance::infer(const ImageInput& image, DetectionInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);   // serialize
  InferenceGuard guard(metrics_, id_);         // RAII metrics
  auto result = engine_.infer(image, timing);
  guard.set_success(true);
  return result;
}
```

**Why the mutex exists**:
- `InferenceEngine::backend_` holds a `Backend::timing_` structure that is
  mutated during every `execute()`. Concurrent calls would produce a data
  race on those `mutable` fields.
- ORT `Session::Run` itself may be thread-safe under certain EPs, but VIE's
  wrapper is not, because of the timing state above.
- The mutex therefore protects VIE-owned mutable state, not the Session.

**Implication**: for same-model high QPS, instantiate multiple
`ModelInstance` objects under distinct model IDs. `ModelManager` allows
duplicate model paths as long as IDs differ.

## 3. `ModelManager` unload Lifecycle Semantics

Semantics are defined by `shared_ptr` ownership, not by a state machine.

| API | Effect |
|---|---|
| `unload(id)` | Removes entry from `models_` map under `mutex_`. Returns bool (was it present). |
| Callers that already hold `shared_ptr<ModelInstance>` | Continue to work normally. Inference in flight is unaffected. |
| New calls to `get(id)` after `unload` | Return `nullptr`. |
| Underlying Backend / Session / CUDA resources | Released when the last `shared_ptr` is destroyed (may be later than `unload()` returned). |

`unload()` is NOT:
- A cancellation of in-flight inference.
- Immediate `cudaDeviceSynchronize` + memory free.

This is intentional and documented. Phase 8.2 tests covered:
- `get` then `unload`, use held `shared_ptr` — safe.
- `unload` then `load` with the same ID — works (no stale state).
- Concurrent `unload` with `submit_model` — safe, held `shared_ptr` survives.
- Multi-thread duplicate `unload(id)` calls — safe, only first returns true.

## 4. Backend Timing Thread Safety Constraint

Every `IInferenceBackend` implementation (MockBackend, OnnxRuntimeBackend,
OnnxRuntimeCudaBackend) exposes `last_*` timing fields via `mutable` members
updated inside `execute()`. These are the reason `ModelInstance::infer` cannot
drop its mutex.

If this constraint is lifted in a future phase, timing state must move to
per-request storage (e.g., passed through `DetectionInferTiming*`). Phase 8.3.1
does not implement that change.

## 5. CUDA Synchronization Strategy

The CUDA backend calls `cudaDeviceSynchronize()` at three points:

| Line | Context | Rationale |
|---:|---|---|
| 231 | `quiesce()` before Session release | Drain device work before releasing ORT resources. Not on hot path. |
| 488 | After `binding.SynchronizeInputs()` inside `execute()` | Ensures input tensors are fully written to CUDA memory before `Session::Run` launches kernels. Kept because ORT's internal stream semantics in `IoBinding::SynchronizeInputs()` in ORT 1.20.1 cannot be verified from VIE's position. |
| 498 | After `binding.SynchronizeOutputs()` inside `execute()` | Ensures GPU writes are visible to host before VIE accesses tensor data / decodes. Kept for the same reason. |

**Correctness**: current implementation is validated by the existing test
suite across all four build combinations (Section 9).

**Performance impact**: NOT quantified. No Nsight Systems or CUDA event
timeline was captured. The claim "CUDA is 13x faster than CPU despite device
sync" is real but does NOT prove device sync is free — it only shows it is
not the dominant cost for this workload.

**Cross-instance impact**: device-wide sync can serialize kernels from other
Sessions. This is a known theoretical ceiling, not measured.

**Phase 8.3.1 decision**: keep all three synchronizations. Any removal
requires a controlled A/B experiment and correctness test, which is out of
scope here.

## 6. `cuda_cpu_ep_fallback` Configuration

Added in Phase 8.3:

```cpp
struct ModelConfig {
  // ...
  bool cuda_cpu_ep_fallback = false;   // default: strict CUDA
};
```

**Default behavior unchanged**: `false` → ORT CUDA backend rejects graphs that
require CPU EP nodes, matching the original VIE strict-CUDA contract.

**Opt-in hybrid**: `true` → VIE sets ORT
`kOrtSessionOptionsDisableCPUEPFallback = "0"` before `prepare()`. ORT may
place unsupported nodes on CPU EP. Required by some ONNX models (YOLOv8n in
this project's test data).

**Observability**: VIE does not currently expose which nodes ORT placed on
CPU EP. `OnnxRuntimeCudaBackend::cpu_ep_fallback()` reports the configuration
only, not runtime placement. Marked as a Phase 8.3.1 limitation.

**Non-CUDA builds**: the field is present but ignored, guarded by
`#if defined(VIE_HAS_ONNXRUNTIME_CUDA)` in `inference_engine.cpp`.

**Test coverage**:
- `OnnxRuntimeCudaBackend` default: `EXPECT_FALSE(cpu_ep_fallback())`.
- After `set_cpu_ep_fallback(true)`: `EXPECT_TRUE(...)`.
- No end-to-end test through `ModelConfig::cuda_cpu_ep_fallback` (noted as
  a non-blocking gap).

## 7. ThreadSanitizer Execution Result

**Status**: Not Available in this container.

- Built `build-tsan/` with `-fsanitize=thread -fno-omit-frame-pointer -g -O1`
  under CUDA OFF + ORT OFF + Debug + tests ON.
- Compilation and manual linking succeeded.
- Executing any test binary (even `--gtest_list_tests`) fails with
  `FATAL: ThreadSanitizer: unexpected memory mapping 0x...`.
- Root cause: container kernel uses high-entropy ASLR (5-level page tables)
  that places `mmap` outside TSan's shadow-memory range.
- Container does not permit `setarch -R` (operation not permitted).

Phase 8.3.1 does NOT claim TSan pass. Concurrency evidence rests on:
- `test_runtime_audit` (14 tests) using TSan-unfriendly but deterministic
  thread interleavings.
- Manual code inspection of mutex coverage.
- Correct results across four build combinations.

## 8. Known Limitations and Operational Notes

1. **Same-model high QPS**: requires multiple `ModelInstance` objects, not
   multiple workers on one. The mutex in `ModelInstance::infer` enforces
   serialization.
2. **GPU kernel overlap**: not demonstrated. Observed throughput gains in
   Scenario C are attributable to Session-level concurrency; whether the
   kernels time-share the GPU is unknown.
3. **`MetricsCollector` memory growth**: per-model map entries are never
   removed. Long-running processes that load and unload many distinct model
   IDs will accumulate memory. Documented since Phase 8.1; not addressed.
4. **`BackendRegistry` lock granularity**: factory `create()` runs under the
   registry mutex. Acceptable because model load is a low-frequency operation.
5. **Per-stage timing not exposed by benchmark**: `DetectionInferTiming`
   exists but is not plumbed into benchmark output. Would require a schema
   change to `real_model_benchmark`.
6. **H2D once**: benchmark CUDA path transfers input once. Real production
   pipelines that stream new images would include H2D per request; benchmark
   numbers do NOT model this cost.
7. **Model set tested**: only `yolov8n.onnx` benchmarked with real runtime.
   Other task types (Seg, OBB, Pose, RT-DETR, Classification) have unit test
   coverage but were not benchmarked in Phase 8.3.

## 9. Regression Result (Phase 8.3.1)

Four build combinations, all rebuilt and rerun with the current tree:

| Combo | Build | Passed | Failed | Skipped | Notes |
|---|---|---:|---:|---:|---|
| A. CUDA OFF + ORT OFF | OK | 185 | 0 | 0 | ThreadPool + Runtime + Mock paths |
| B. CUDA OFF + ORT CPU | OK | 214 | 0 | 0 | Adds real ORT CPU tests |
| C. CUDA ON + ORT CPU | OK | 245 | 0 | 0 | Adds CUDA tensor plumbing |
| D. CUDA ON + ORT GPU | OK | 257 | 0 | 1 | GPU real-inference tests enabled |

Skip in D is `ClassificationEngine.MissingBackendIsExplicit`: the test asserts
a "no backend" failure path that is unreachable when ORT GPU is available.
Intentional pre-existing test.

## 10. Production Deployment Recommendations

- **Multi-model serving**: safe. Use `Runtime` with `thread_pool_size >= N`
  and `N` `ModelInstance`s. Verified in Scenario C (both backends).
- **Same-model concurrency**: instantiate multiple `ModelInstance` objects
  with distinct IDs pointing at the same model path.
- **Strict CUDA requirement**: leave `cuda_cpu_ep_fallback=false`.
- **Hybrid CUDA**: set `cuda_cpu_ep_fallback=true` and treat throughput as
  mixed CPU+CUDA path.
- **Monitoring**: `MetricsCollector` provides per-model counts and timing;
  it does not yet break down by EP.
- **Unload semantics**: `unload()` removes from the manager; resources
  release when the last `shared_ptr` dies. Callers that keep references
  continue to work.

## 11. Files Changed in Phase 8.3.1

- `tools/real_model_benchmark.cpp` — rewritten statistics, RAII CUDA buffer,
  identical warmup/request counts across scenarios, nearest-rank percentile,
  min/max reporting, consistency validation.
- `docs/research/PHASE_8_3_BENCHMARK_RESULTS.md` — replaced with corrected
  methodology and Phase 8.3.1 measured data.
- `docs/research/PHASE_8_3_PRODUCTION_READINESS.md` — this file.
- `docs/research/PHASE_8_3_1_FINAL_AUDIT.md` — new audit summary.

No production code paths (Runtime, Backend, Engine) were changed in Phase
8.3.1 beyond what was already in place at 276e25c.

## 12. Final Phase 8.3.1 Judgment

**Status**: **Conditionally Complete**.

Met acceptance criteria:
- [x] Avg/P50/P95 statistical definition unified.
- [x] Benchmark sample count = completed requests.
- [x] CPU/CUDA timing boundaries explicitly documented.
- [x] Performance ratio conclusions limited to specific configuration.
- [x] Multi-instance concurrency claims separated from GPU kernel overlap
  claims.
- [x] `cudaDeviceSynchronize` retained with explicit rationale and
  "performance impact not quantified" note.
- [x] `cuda_cpu_ep_fallback` default behavior unchanged, strict-CUDA still
  rejects.
- [x] Four build combinations all rebuilt and rerun.
- [x] Documentation aligned with code and experiment.

Not met (environment limitations, not code defects):
- [ ] TSan: cannot execute in this container, full failure log recorded.
- [ ] Nsight Systems GPU timeline: not available in environment.

These are explicitly marked "Not Available" rather than being presented as
pass.
