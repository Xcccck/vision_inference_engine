# Phase 8.3.1 Final Audit

## 1. Scope

Phase 8.3.1 is an audit and correction pass on Phase 8.3. It does not add
Runtime features, does not refactor Backends, does not implement Session Pool
or Dynamic Batch. Its goal is to make the Phase 8.3 benchmark data reproducible
and to align technical claims with actual evidence.

## 2. Git State

- Branch: `main`.
- Phase 8.3 local commits: `633b35b` (benchmark) and `276e25c` (conditional
  compile fix). At the start of Phase 8.3.1 the remote was at `633b35b`, i.e.
  `276e25c` was unpushed. Push was retried in this phase (see Section 10).
- `.git/hooks/commit-msg` strips Cursor `Co-authored-by` trailers. No
  `prepare-commit-msg` hook exists. Phase 8.3.1 commits contain no Cursor
  references.

## 3. Files Modified and Reason

| File | Change | Reason |
|---|---|---|
| `tools/real_model_benchmark.cpp` | Full rewrite | Fix statistics methodology (nearest-rank percentile, sample consistency validation, min/max reporting, failed-request tracking), unify warmup and per-thread request count across scenarios, RAII CUDA buffer cleanup. |
| `docs/research/PHASE_8_3_BENCHMARK_RESULTS.md` | Rewritten | Replace Phase 8.3 raw data with Phase 8.3.1 corrected methodology. Constrain concurrency claims to evidence. Add CPU/CUDA timing boundary section. Add percentile definition. |
| `docs/research/PHASE_8_3_PRODUCTION_READINESS.md` | Rewritten | Reflect corrected benchmark methodology, add explicit ModelInstance concurrency semantics, add fallback configuration section, document TSan environment failure. |
| `docs/research/PHASE_8_3_1_FINAL_AUDIT.md` | New | This file. |

No production source (Runtime / Backend / Engine / Decoder / Preprocess) was
changed in Phase 8.3.1.

## 4. Benchmark: Before vs After

### Before (Phase 8.3)
- Percentile: linear interpolation over sorted array.
- Warmup: 5 iterations for all scenarios.
- Request count: 50 (A) vs 100 (B/C). Not matched.
- CUDA memory: `cudaMalloc` result never freed.
- No sample-vs-completed consistency check.
- No min/max reported.
- No failed-request tracking.

### After (Phase 8.3.1)
- Percentile: nearest-rank, `rank = ceil(p/100 * N)`, value = `sorted[rank-1]`.
- Warmup: 10 iterations, same for all scenarios.
- Request count: `workers × 25` (A=25, B=100, C=100). Per-worker identical.
- CUDA memory: RAII `CudaImageBuffer` destructor calls `cudaFree`.
- `compute_stats` throws on empty sample set, non-finite samples, or
  `min > p50 > p95 > max` violations.
- Min/Max reported for every scenario.
- Failed and completed counts tracked separately; throughput excludes failed.
- Output prints `sample_count` alongside `completed_requests` and warns on
  mismatch.

### Numerical Differences

Because Phase 8.3.1 increased warmup (5→10) and reduced Scenario A's total
requests (50→25), the throughput numbers shift. The Phase 8.3 CUDA "11x over
CPU" claim does not directly carry over. Recomputed:

| Backend | Single-instance throughput (Phase 8.3) | Single-instance throughput (Phase 8.3.1) |
|---|---:|---:|
| CPU A | 11.43 req/s | 11.72 req/s |
| CUDA A | 125.47 req/s | 152.65 req/s |
| Ratio | 10.98x | 13.02x |

The 13.02x figure is the correct one going forward, with the scope limitation
documented in `PHASE_8_3_BENCHMARK_RESULTS.md` Section 4.3.

## 5. CUDA Synchronization Audit

Three `cudaDeviceSynchronize()` calls remain:

| Line | Function | Decision |
|---:|---|---|
| 231 | `quiesce()` | Keep. Drain required before Session release. Not on hot path. |
| 488 | `execute()` after `SynchronizeInputs()` | Keep. Cannot prove ORT's internal stream semantics from VIE's position. |
| 498 | `execute()` after `SynchronizeOutputs()` | Keep. Same reason. |

**Performance impact**: NOT quantified. Nsight Systems not available in
environment. The fact that CUDA is 13x faster than CPU under this benchmark
does NOT demonstrate that device sync costs nothing.

**Correctness**: current strategy is validated by existing real-model tests
across all four build combinations.

**Recommended wording**:
> Current synchronization strategy retained. Correctness verified by real
> model tests. Performance impact of the two execute() device-wide syncs has
> not been quantified by a controlled experiment.

## 6. ORT Concurrency Audit

| Question | Answer | Evidence |
|---|---|---|
| Does ORT `Session::Run` support concurrent calls? | Depends on EP; CUDA EP is documented as concurrency-capable for some graph configurations. VIE does not exercise this directly. | ORT documentation, not tested in this project. |
| Can VIE remove the `ModelInstance` mutex? | No. `Backend::timing_` is mutable state accessed by `execute()` under the current design. | Code inspection of all three Backend implementations. |
| Can different VIE `ModelInstance` objects execute concurrently? | Yes, verified in benchmark Scenario C on both CPU and CUDA. | Corrected benchmark data. |
| Are ORT internal streams accessible to VIE? | Not currently. `has_user_compute_stream=0` is hardcoded. | `src/backend/onnxruntime_cuda_backend.cpp:351`. |

Phase 8.3.1 does not change any of these. It only removes overreaching claims
about GPU kernel time overlap.

## 7. `cuda_cpu_ep_fallback` Audit

Answers to the questions asked by the Phase 8.3.1 specification:

1. **Default**: `false`.
2. **Toggle**: `ModelConfig::cuda_cpu_ep_fallback = true` before `load()`.
   Also `OnnxRuntimeCudaBackend::set_cpu_ep_fallback(bool)` at the Backend
   level.
3. **ORT behavior when enabled**: ORT is permitted to assign nodes that CUDA
   EP cannot execute to the default CPU EP.
4. **Strict CUDA still rejects**: yes, the `disable_cpu_ep_fallback=1` config
   entry causes `prepare()` to throw "graph nodes assigned to the default
   CPU EP, but fallback to CPU EP has been explicitly disabled". This was
   directly observed on YOLOv8n before the flag existed.
5. **Impact on Backend timing / performance reports**: `last_run_ms` includes
   any CPU EP work; no breakdown by EP is currently produced. Not a
   correctness issue, but a reporting limitation.
6. **Tests**: Backend-level default and enabled behavior are tested in
   `tests/test_onnxruntime_cuda_backend.cpp`. No end-to-end test through
   `ModelConfig::cuda_cpu_ep_fallback` was added; marked as a non-blocking
   gap.
7. **Cross-build behavior**: field is silently ignored when
   `VIE_HAS_ONNXRUNTIME_CUDA` is not defined. No misleading behavior in
   CUDA-OFF builds.

**Silent CPU fallback observability**: VIE cannot enumerate per-node EP
placement with the current ORT 1.20.1 API. If a deployment requires a strict
"100% CUDA graph" guarantee, use the default (`false`) and accept that some
models will fail `prepare()`.

## 8. ThreadSanitizer Result

**Status**: Not Available. Full failure reproduced and documented.

**Attempt commands** (recorded for reproducibility):
```
mkdir build-tsan && cd build-tsan
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DVIE_ENABLE_CUDA=OFF -DVIE_ENABLE_ONNXRUNTIME=OFF \
         -DVIE_BUILD_TESTS=ON \
         -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g -O1" \
         -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
make -j4 CMakeFiles/vie_tests.dir/tests/*.o  # object files OK
/usr/bin/c++ -fsanitize=thread -no-pie ... -o vie_tests_nopie  # link OK

./vie_tests_nopie --gtest_list_tests
# FATAL: ThreadSanitizer: unexpected memory mapping 0x5c6a5e900000-0x5c6a5e923000
```

**Root cause**: container kernel uses high-entropy ASLR (mmap bases above
TSan's shadow range). Container runtime blocks `setarch -R` and
`randomize_va_space` modification. Not a project code defect.

**Action**: do not claim TSan pass. Runtime concurrency is verified only via
the tests in `test_runtime_audit.cpp` and code inspection. This is a
documented residual risk.

## 9. Four Build Regression Results

| Combo | Build | Ran | Passed | Failed | Skipped | Notes |
|---|---|---:|---:|---:|---:|---|
| A. CUDA OFF + ORT OFF | OK | 185 | 185 | 0 | 0 | — |
| B. CUDA OFF + ORT CPU | OK | 214 | 214 | 0 | 0 | — |
| C. CUDA ON + ORT CPU  | OK | 245 | 245 | 0 | 0 | — |
| D. CUDA ON + ORT GPU  | OK | 258 | 257 | 0 | 1 | `ClassificationEngine.MissingBackendIsExplicit` intentionally skipped when both CPU and GPU ORT backends are present. |

Each build was rebuilt from the Phase 8.3.1 tree and executed fully. No tests
were removed or relaxed.

## 10. Remaining Issues

**Blocking**: none identified in Phase 8.3.1.

**Non-blocking**:
- TSan cannot execute in this container. Requires different environment.
- No Nsight Systems profiler; GPU kernel overlap remains unverified.
- No end-to-end test for `ModelConfig::cuda_cpu_ep_fallback`.
- `real_model_benchmark` does not yet break out Preprocess / Backend /
  Decode timings even though the underlying `DetectionInferTiming` fields
  exist.
- `MetricsCollector` unbounded per-model map (documented Phase 8.1).
- Push of `276e25c` to `origin/main` previously failed due to a transient
  TLS error; Phase 8.3.1 push must succeed and this commit is expected to
  reach the remote.

**Out of scope for Phase 8.3.1**:
- Session Pool.
- Dynamic Batch.
- New scheduling system.
- Removal of `cudaDeviceSynchronize` without controlled A/B evidence.

## 11. Phase 8.3.1 Final Judgment

**Conditionally Complete.**

All code and documentation corrections within the scope of this phase have
been applied. Statistical methodology is now unambiguous and reproducible.
Concurrency claims are limited to what the tests and benchmarks actually
show. The two items that could not be executed in this environment (TSan and
Nsight Systems profiling) are recorded as Not Available rather than claimed
as pass.

**Phase 9 readiness**: the Runtime layer is ready for a next phase if that
phase chooses to add Session Pool, per-stage timing extraction, or profiler
integration. This phase does not select Phase 9 scope.
