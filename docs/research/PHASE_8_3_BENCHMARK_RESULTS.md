# Phase 8.3 / 8.3.1 Real Model Benchmark Results

## 1. Executive Summary

Phase 8.3 executed real YOLOv8n benchmarks on CPU and CUDA.
Phase 8.3.1 fixed the statistics methodology and reran the benchmarks with corrected definitions.

**Key findings (Phase 8.3.1 corrected data, see Section 4 for details)**:
- CUDA throughput is approximately **13.0x** CPU throughput under the specific
  conditions measured (YOLOv8n, 640x640, input pre-allocated in GPU, H2D once,
  no per-request input transfer, includes preprocess/decode/NMS).
- Multi-instance concurrency improves throughput on both backends.
- Single-instance multi-thread remains serialized by the instance mutex.

**Scope limits**: these ratios apply only to the model, input size, and pipeline
described below. They do not generalize to other models, batch sizes, or a
deployment chain that includes per-request image H2D/D2H.

## 2. Statistics Definitions (Phase 8.3.1)

All metrics in this document follow the definitions implemented in
[`tools/real_model_benchmark.cpp`](../../tools/real_model_benchmark.cpp).

| Metric | Definition |
|---|---|
| `request_latency` | `steady_clock::now()` sample taken immediately before `model->infer(image)` and immediately after it returns or throws. Includes preprocessing, backend execute, decoder, NMS, and any mutex wait inside `ModelInstance::infer()`. |
| `sample_count` | Number of latency values pushed into the array used for stats. Failed requests are NOT added to samples. |
| `completed_requests` | Requests that returned without exception. Must equal `sample_count`. |
| `failed_requests` | Requests that threw. Counted separately, excluded from throughput. |
| `total_wall_time` | `steady_clock` measured before the first request and after the last `future.get()`. For concurrent scenarios this includes thread-pool task scheduling but excludes warmup. |
| `throughput` | `completed_requests / total_wall_time_seconds`. |
| `mean_latency` | `sum(samples) / sample_count`. |
| `p50_latency` / `p95_latency` | Nearest-rank percentile on the sorted sample vector: `rank = ceil(p/100 * N)`, value = `sorted[rank - 1]`. |
| `min_latency` / `max_latency` | Extremes of the same sample vector. |

Consistency checks enforced at runtime:
- Non-finite samples throw.
- Empty sample set throws (no fake zero metrics).
- `min <= p50 <= p95 <= max` asserted after computation.
- `min <= mean <= max` asserted.
- `sample_count == completed_requests` asserted in output.

## 3. Test Environment and Timing Boundaries

### 3.1 Environment

- **GPU**: NVIDIA RTX 4080 (`nvidia-smi` verified).
- **CPU**: container host, 32 hardware threads visible to build.
- **Model**: `testdata/onnx/yolov8n.onnx`, 640x640, RGB.
- **ONNX Runtime**: 1.20.1 (CPU and CUDA builds).
- **Backend flags**: `cuda_cpu_ep_fallback = true` in CUDA benchmarks.
  YOLOv8n has at least one node not implemented by CUDA EP (detected during
  Phase 8.3), so a strict-CUDA run fails `prepare()`. The fallback path is
  documented explicitly rather than hidden.

### 3.2 Input Path

Both CPU and CUDA paths allocate the 640x640x3 RGB buffer **once** before the
benchmark loop.

- CPU: host `std::vector<uint8_t>`, passed by pointer to `ImageInput.data`.
- CUDA: `cudaMalloc` + one `cudaMemcpy(H2D)` before warmup. The device pointer
  is reused for every request. `cudaFree` runs in `CudaImageBuffer` destructor.

### 3.3 What Each Latency Includes

| Stage | Included in per-request latency |
|---|---|
| Image preprocessing (letterbox, normalization, HWC -> NCHW float) | Yes |
| Backend execute (bind inputs, `session->Run`, bind outputs, D2H for CUDA output) | Yes |
| Decoder (YOLOv8 tensor decode) | Yes |
| NMS | Yes |
| `ModelInstance::mutex` wait | Yes |
| Thread pool task dispatch | Not measured per request; reflected in `total_wall_time` for concurrent scenarios |
| H2D of the input image | No — performed once before warmup (see 3.2) |
| Output tensor allocation in CUDA | Inside Backend::execute, so yes |
| InferenceGuard metrics bookkeeping | Yes |

The following separate phase timings are already tracked by Backend `timing`
structures (populated by `DetectionInferTiming`), but `real_model_benchmark`
currently does not record them into the output. Extending it would change the
benchmark path; they are marked **Not Available** in this Phase 8.3.1 report:

| Metric | Status |
|---|---|
| Preprocess-only latency | Not Available (would require `DetectionInferTiming` plumbing) |
| Backend execute-only latency | Not Available (same reason) |
| Postprocess / Decode-only latency | Not Available |
| H2D time per request | Not Applicable (input pre-transferred once) |
| D2H time per request | Included in `request_latency` via `SynchronizeOutputs` in Backend::execute |

## 4. Corrected Benchmark Data (Phase 8.3.1)

Common parameters across scenarios:
- Warmup: **10** iterations.
- Requests per worker thread: **25**.
- Total requests: `workers * 25`.
- Scenario A: 1 worker, 1 instance → 25 requests.
- Scenario B: 4 workers, 1 instance → 100 requests.
- Scenario C: 4 workers, 2 instances → 100 requests (2 workers per instance).

### 4.1 CPU EP (OnnxRuntime 1.20.1 CPU)

| Scenario | Workers | Instances | Requests | Throughput (req/s) | Mean (ms) | Min (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: 1 worker, 1 instance | 1 | 1 | 25 | **11.72** | 85.30 | 76.93 | 82.89 | 95.63 | 99.94 |
| B: 4 workers, 1 instance | 4 | 1 | 100 | **11.46** | 218.71 | 74.06 | 87.09 | 99.86 | 6639.15 |
| C: 4 workers, 2 instances | 4 | 2 | 100 | **17.62** | 195.96 | 92.12 | 114.57 | 452.78 | 2639.00 |

Notes:
- Scenario B throughput is essentially unchanged from A. This is consistent
  with `ModelInstance` serializing all `infer()` calls through a mutex; extra
  threads only add queueing time (visible in mean and max).
- Scenario C gains **+50.3%** over A (17.62 / 11.72). Two independent
  ModelInstances execute in parallel on two independent ORT Sessions.
- Very large Max values in B and C are the first thread to finish observing
  queue drain; the tail reflects scheduling under contention, not a data race.

### 4.2 CUDA EP (OnnxRuntime 1.20.1 CUDA, `cuda_cpu_ep_fallback=true`)

| Scenario | Workers | Instances | Requests | Throughput (req/s) | Mean (ms) | Min (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: 1 worker, 1 instance | 1 | 1 | 25 | **152.65** | 6.55 | 5.47 | 6.40 | 7.62 | 8.33 |
| B: 4 workers, 1 instance | 4 | 1 | 100 | **162.54** | 15.53 | 5.32 | 6.05 | 7.66 | 470.51 |
| C: 4 workers, 2 instances | 4 | 2 | 100 | **184.79** | 16.00 | 7.30 | 10.33 | 14.20 | 274.33 |

Notes:
- Scenario B is modestly above A (162.54 vs 152.65, +6.5%). The gain is
  plausibly from CPU-side overlap of preprocessing/decoding of one thread while
  another holds the mutex. It does NOT demonstrate GPU kernel concurrency
  inside one Session.
- Scenario C is **+21.1%** over A (184.79 vs 152.65). Two independent
  Sessions/Backends execute concurrently. Actual GPU kernel time overlap is
  NOT verified here — see Section 6.

### 4.3 Performance Ratio

Ratio of A_scenario_CUDA / A_scenario_CPU = 152.65 / 11.72 = **13.02x**.
Ratio of C_scenario_CUDA / C_scenario_CPU = 184.79 / 17.62 = **10.49x**.

The Phase 8.3 report claimed "11x"; recomputation from those raw numbers
(125.47 / 11.43) gives 10.98x, so "≈11x" was consistent with that dataset.
The Phase 8.3.1 dataset with corrected methodology gives 13.0x (single) and
10.5x (multi), because warmup is now 10 iterations and per-thread request
count is fixed. **Do not average across the two datasets.**

## 5. Concurrency Claims, Constrained to Evidence

The claims below are graded by evidence strength.

### 5.1 Worker Concurrency (Evidence: Benchmark + tests)

**Claim**: ThreadPool submits N tasks and executes them concurrently across N
worker threads.

- Verified by `test_runtime_audit` (ConcurrentSubmit, MassiveTaskSubmission).
- Verified indirectly by Scenario B/C (all futures eventually complete, wall
  time shrinks in C).
- **Strong evidence.**

### 5.2 Session-Level Concurrency (Different `ModelInstance`)

**Claim**: Two or more `ModelInstance` objects each own an independent ORT
`Session` and can execute `infer()` concurrently without shared mutable state.

- `ModelInstance` holds its own `InferenceEngine`, which owns one
  `Ort::Session`. No instance-level sharing.
- Scenario C observed throughput gains (+50% CPU, +21% CUDA) consistent with
  this claim.
- `BackendRegistry` is guarded by a mutex; registration and lookup do not
  serialize execution.
- **Strong evidence from throughput data + code inspection.**

### 5.3 Same-Instance Concurrency (Serialized)

**Claim**: Multiple threads calling `ModelInstance::infer()` on the same
instance serialize behind the `mutex_`.

- Verified by Scenario B: throughput essentially unchanged from A on CPU.
- Verified by `test_runtime_audit` and `test_runtime`.
- **Strong evidence.**

### 5.4 GPU Kernel Time Overlap

**Claim**: CUDA kernels from two independent ORT Sessions overlap in wall-clock
time on the GPU.

- **Not Verified.** Neither Nsight Systems nor a custom CUDA event timeline
  was captured. The observed throughput improvement in Scenario C is
  compatible with either:
  - True kernel overlap across sessions, OR
  - Sequential GPU kernel execution with CPU-side parallelism on preprocessing
    and decoding while the other session's `SynchronizeOutputs` waits.
- `cudaDeviceSynchronize()` calls in the CUDA backend are device-wide, which
  would in fact suppress overlap between instances if kernels were launched
  on separate streams. This strengthens the requirement to measure with a
  profiler before claiming GPU concurrency.
- **Do not claim "GPU kernel overlap is achieved"** from these numbers alone.

**Safe wording**: "Two `ModelInstance` objects can be driven concurrently
without data races or correctness failures. Whether their GPU kernels
actually overlap in time is unmeasured and unproven in Phase 8.3.1."

## 6. cudaDeviceSynchronize Audit (unchanged conclusion)

Locations in [`src/backend/onnxruntime_cuda_backend.cpp`](../../src/backend/onnxruntime_cuda_backend.cpp):

| Line | Function | Context | Status |
|---:|---|---|---|
| 231 | `quiesce()` | Called from `release_session()` before `session.reset()` | Kept: device-wide drain required before releasing ORT Session and any CUDA resources it may still hold. Not on hot path. |
| 488 | `execute()` | Immediately after `binding.SynchronizeInputs()` | Kept: see analysis below. |
| 498 | `execute()` | Immediately after `binding.SynchronizeOutputs()` | Kept: see analysis below. |

### 6.1 Why these two are retained

- ORT CUDA EP owns its compute stream internally; VIE never passes
  `has_user_compute_stream=1` or a `user_compute_stream`. The `Ort::IoBinding`
  calls `SynchronizeInputs/Outputs` only on the stream ORT chooses; the exact
  completion semantics of that synchronization in ORT 1.20.1 CUDA EP cannot be
  proven from VIE's position.
- Removing the extra `cudaDeviceSynchronize()` risks:
  - Reading output tensors before GPU writes complete (correctness bug).
  - Freeing ORT-held tensors while GPU still accesses them (illegal memory).
  - Data contamination across requests if the input buffer is reused.
- Phase 8.3.1 does NOT include a controlled A/B experiment (Option A: current;
  Option B: stream-only; Option C: `user_compute_stream`) because that would
  require changing the execution path and would produce only speculative
  evidence.

### 6.2 Performance impact

- Not quantified. Nsight Systems or an equivalent profiler is not available in
  the environment.
- Not observable as a bottleneck in the current data (CUDA Scenario A still
  exceeds CPU Scenario A by ~13x), but this is NOT a proof that device sync
  costs nothing.

### 6.3 Cross-instance impact

- `cudaDeviceSynchronize()` is device-wide. Two concurrent `ModelInstance`
  objects on the same device will serialize each other through this call even
  if their ORT streams are independent. This is a **theoretical penalty** but
  not the only reason Scenario C falls short of linear scaling.

**Recommended wording for architecture docs**:
> Current synchronization strategy is retained. Correctness is validated by
> existing real-model inference tests. The performance impact of the two
> device-wide syncs inside `execute()` has NOT been quantified by a controlled
> experiment, and should not be described as "no bottleneck" without data.

## 7. `cuda_cpu_ep_fallback` Configuration

Introduced in Phase 8.3 to make YOLOv8n runnable under CUDA EP without crashing
on unsupported ops.

| Aspect | Value |
|---|---|
| Struct field | `ModelConfig::cuda_cpu_ep_fallback` (bool) |
| Default | `false` — existing default behavior is NOT changed |
| Effect when `false` | CUDA backend keeps `kOrtSessionOptionsDisableCPUEPFallback = "1"`, strict CUDA mode. Any node CUDA cannot run fails `prepare()`. |
| Effect when `true` | CUDA backend sets `DisableCPUEPFallback = "0"`. ORT may schedule unsupported nodes on CPU EP. |
| Wiring point | `InferenceEngine::load()` — cast to `OnnxRuntimeCudaBackend`, call `set_cpu_ep_fallback(true)`, then `prepare()`. Guarded by `#if defined(VIE_HAS_ONNXRUNTIME_CUDA)`. |
| Test coverage | `test_onnxruntime_cuda_backend.cpp`: `EXPECT_FALSE(backend.cpu_ep_fallback())` (default); `backend.set_cpu_ep_fallback(true); EXPECT_TRUE(...)`. |
| Observability | VIE does not currently expose which nodes ORT placed on CPU EP. This is an ORT 1.20.1 limitation; no API is called to enumerate node→EP placement. |
| Non-CUDA builds | Field exists but is ignored (`#if defined(VIE_HAS_ONNXRUNTIME_CUDA)`). No behavior change for CUDA-OFF builds. |

**Answer to specific questions**:

1. **Default value?** `false` (strict CUDA).
2. **How to toggle?** Set `ModelConfig::cuda_cpu_ep_fallback = true` before
   `load()`. Backend-level toggle is `OnnxRuntimeCudaBackend::set_cpu_ep_fallback(bool)`.
3. **Does ORT place unsupported nodes on CPU EP when enabled?** Yes, that is
   the ORT-defined meaning of allowing CPU EP fallback.
4. **Strict CUDA still rejects?** Yes. When the flag is `false`, `prepare()`
   throws with "graph nodes assigned to the default CPU EP, but fallback to
   CPU EP has been explicitly disabled". This message was observed in Phase
   8.3 before enabling the flag for YOLOv8n.
5. **Impact on Backend timing / performance reports?** The `last_run_ms`
   timing includes any CPU-executed nodes; there is no breakdown by EP. Not
   verified whether CPU nodes inside a "CUDA" run affect measured
   `cudaDeviceSynchronize` behavior.
6. **Tests for enable/disable/default?** Backend-level tests exist. No
   dedicated `ModelConfig::cuda_cpu_ep_fallback` end-to-end test was added in
   Phase 8.3.1; noted as a non-blocking gap.
7. **Consistent behavior across builds?** Only CUDA-enabled builds honor the
   flag. CUDA-OFF builds ignore it silently; no misleading effect because the
   field is not present on the code path.

**Silent CPU fallback risk**: If `cuda_cpu_ep_fallback = true` is set and ORT
silently places many nodes on CPU EP, the benchmark label "CUDA" is still
technically correct (the backend runs through the CUDA Session), but measured
performance may not reflect a strict-CUDA graph. Users who need strict CUDA
should keep the default (`false`).

## 8. ThreadSanitizer

**Status**: **Not Available in this container.** Not marked as passing.

**Attempted**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DVIE_ENABLE_CUDA=OFF -DVIE_ENABLE_ONNXRUNTIME=OFF \
         -DVIE_BUILD_TESTS=ON \
         -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g -O1" \
         -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
```

Object files compiled cleanly. Linking via the make-generated post-build hook
was blocked by CMake's `gtest_discover_tests` invoking the executable to
enumerate tests, which triggered the failure. Manual linking with `-no-pie`
produced a runnable ELF, but running any gtest filter fails immediately:

```
FATAL: ThreadSanitizer: unexpected memory mapping 0x5c6a5e900000-0x5c6a5e923000
```

**Root cause**: TSan's shadow-memory layout conflicts with the container
kernel's mmap base. `randomize_va_space=2` and 5-level page tables push mmap
addresses above TSan's supported shadow range. The container runtime does not
permit `setarch -R` (operation not permitted), so userspace cannot disable ASLR.

**Conclusion**: TSan cannot be executed in this environment. This is an
environment limitation, NOT evidence that the code is race-free. Phase 8.3.1
does not claim TSan pass.

## 9. Regression: Four Build Combinations (Phase 8.3.1)

| Combo | CMake Flags | Build | Ran | Passed | Failed | Skipped | Skip Reason |
|---|---|---:|---:|---:|---:|---:|---|
| A. CUDA OFF + ORT OFF | `-DVIE_ENABLE_CUDA=OFF -DVIE_ENABLE_ONNXRUNTIME=OFF` | OK | 185 | 185 | 0 | 0 | — |
| B. CUDA OFF + ORT CPU | `-DVIE_ENABLE_CUDA=OFF -DVIE_ENABLE_ONNXRUNTIME=ON` | OK | 214 | 214 | 0 | 0 | — |
| C. CUDA ON + ORT CPU  | `-DVIE_ENABLE_CUDA=ON -DVIE_ENABLE_ONNXRUNTIME=ON` (no GPU root) | OK | 245 | 245 | 0 | 0 | — |
| D. CUDA ON + ORT GPU  | adds `-DVIE_ONNXRUNTIME_GPU_ROOT=/opt/onnxruntime-linux-x64-gpu-1.20.1` | OK | 258 | 257 | 0 | 1 | `ClassificationEngine.MissingBackendIsExplicit` |

`MissingBackendIsExplicit` is skipped when both CUDA and CPU ORT backends are
available — the test asserts a "no backend" error path that this build does
not exercise. This skip predates Phase 8.3 and is intentional in the test
definition.

## 10. Corrections vs Phase 8.3

| Item | Phase 8.3 (original) | Phase 8.3.1 (corrected) |
|---|---|---|
| Percentile algorithm | Linear interpolation | Nearest-rank |
| Sample size consistency check | None | Asserted |
| Failed request handling | Silent drop | Counted separately, excluded from throughput |
| Warmup across scenarios | Different (A=5, B=5, C=5) | Identical (10 for all) |
| Request count across scenarios | Different (A=50, B=100, C=100) | Uniform (workers × 25) |
| CUDA image buffer cleanup | Leaked `cudaMalloc` result | RAII `CudaImageBuffer` destructor |
| CUDA image H2D location | Not documented | Explicitly once-before-benchmark, documented |
| Min/Max reported | No | Yes |
| Concurrency evidence grading | Undifferentiated | 3-level: Worker / Session / Kernel overlap |
| GPU kernel overlap claim | "Different models run in parallel" (implied kernel overlap) | Only Session-level concurrency is claimed; kernel overlap explicitly marked **Not Verified** |
| `cudaDeviceSynchronize` performance impact | "Not a bottleneck" | "Not quantified by controlled experiment" |
| 11x CPU→CUDA ratio | Stated without scope | Scope explicitly limited to YOLOv8n/640×640/pipeline in this build |
| TSan | Not run, marked "environment limitation" | Not run, full error and commands recorded |

## 11. Production Deployment Notes (Corrected)

Given the evidence in this file:

1. **Runtime concurrency model** (ThreadPool + per-ModelInstance mutex) is
   appropriate for multi-model serving. Single-model high concurrency benefits
   from creating multiple `ModelInstance` objects, not from adding workers to
   one instance.
2. **`cudaDeviceSynchronize` calls** inside `execute()` are a serialization
   point across ModelInstances sharing the same CUDA device. Users measuring
   unexpected CUDA scaling ceilings should investigate this.
3. **`cuda_cpu_ep_fallback=true`** enables hybrid CPU+CUDA execution. It is
   required to run YOLOv8n on CUDA EP with the current ORT 1.20.1 CUDA kernel
   coverage. It should be flagged in operational metrics if strict-CUDA
   execution is a compliance requirement.
4. **GPU kernel overlap** between Sessions has NOT been demonstrated. Claims
   about "true CUDA concurrency" require profiler evidence.

## 12. Known Remaining Items (Non-Blocking)

- TSan cannot run in this container. Requires privileged environment or
  `vm.mmap_min_addr` reduction.
- No end-to-end test for `ModelConfig::cuda_cpu_ep_fallback` (backend-level
  tests exist).
- Per-stage timing (Preprocess / Backend::execute / Decode / NMS) is not
  broken out in the benchmark output even though `DetectionInferTiming` is
  plumbed through `ModelInstance::infer`.
- No Nsight Systems timeline captured to prove/disprove GPU kernel overlap.
- `MetricsCollector` unbounded per-model map growth remains a documented
  limitation from Phase 8.1.
