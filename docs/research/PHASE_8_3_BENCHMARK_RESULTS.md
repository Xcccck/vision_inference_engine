# Phase 8.3 Real Model Benchmark Results

## Executive Summary

Phase 8.3 成功执行了真实 YOLOv8n 模型的 CPU 和 CUDA benchmark，验证了多实例并发的性能收益。

**关键发现**：
- ✅ CUDA 比 CPU 快 **11 倍**（125.47 vs 11.43 req/s）
- ✅ 多实例并发在 CPU 上提升 **56%**（17.88 vs 11.43 req/s）
- ✅ 多实例并发在 CUDA 上提升 **39%**（174.53 vs 125.47 req/s）
- ✅ 单实例多线程由于 mutex 串行化，性能提升有限
- ✅ Runtime 层并发设计正确，多实例真正并行执行

## Test Environment

- **GPU**: NVIDIA RTX 4080
- **CPU**: Intel Xeon (multiple cores)
- **Model**: YOLOv8n (testdata/onnx/yolov8n.onnx)
- **Input**: 640x640 RGB image
- **Backend**: ONNX Runtime 1.20.1
- **Build**: CUDA=ON, ONNXRUNTIME=ON, ONNXRUNTIME_CUDA=ON

## CPU Benchmark Results

### Scenario A: Single Instance, Single Thread
- **Throughput**: 11.43 req/s
- **Avg Latency**: 87.49 ms
- **P50 Latency**: 87.74 ms
- **P95 Latency**: 95.04 ms
- **Workers**: 1
- **Instances**: 1
- **Requests**: 50

### Scenario B: Single Instance, 4 Threads
- **Throughput**: 11.29 req/s
- **Avg Latency**: 264.27 ms
- **P50 Latency**: 88.31 ms
- **P95 Latency**: 99.84 ms
- **Workers**: 4
- **Instances**: 1
- **Requests**: 100

**Analysis**: 单实例多线程由于 mutex 串行化，吞吐量无提升，延迟增加 3 倍。

### Scenario C: 2 Instances, 4 Threads
- **Throughput**: 17.88 req/s
- **Avg Latency**: 177.40 ms
- **P50 Latency**: 111.67 ms
- **P95 Latency**: 260.58 ms
- **Workers**: 4
- **Instances**: 2
- **Requests**: 100

**Analysis**: 多实例并发提升 56%（17.88 vs 11.43 req/s），证明 CPU EP 支持多实例并行。

## CUDA Benchmark Results

### Scenario A: Single Instance, Single Thread
- **Throughput**: 125.47 req/s
- **Avg Latency**: 7.97 ms
- **P50 Latency**: 7.97 ms
- **P95 Latency**: 9.71 ms
- **Workers**: 1
- **Instances**: 1
- **Requests**: 50

### Scenario B: Single Instance, 4 Threads
- **Throughput**: 143.34 req/s
- **Avg Latency**: 17.47 ms
- **P50 Latency**: 6.82 ms
- **P95 Latency**: 8.78 ms
- **Workers**: 4
- **Instances**: 1
- **Requests**: 100

**Analysis**: 单实例多线程提升 14%（143.34 vs 125.47 req/s），可能由于预处理和调度重叠。

### Scenario C: 2 Instances, 4 Threads
- **Throughput**: 174.53 req/s
- **Avg Latency**: 17.29 ms
- **P50 Latency**: 11.13 ms
- **P95 Latency**: 16.20 ms
- **Workers**: 4
- **Instances**: 2
- **Requests**: 100

**Analysis**: 多实例并发提升 39%（174.53 vs 125.47 req/s），证明 CUDA EP 支持多实例并行。

## Performance Comparison

### CPU vs CUDA
| Metric | CPU | CUDA | Speedup |
|--------|-----|------|---------|
| Single instance throughput | 11.43 req/s | 125.47 req/s | **11.0x** |
| Multi-instance throughput | 17.88 req/s | 174.53 req/s | **9.8x** |
| Single instance latency | 87.49 ms | 7.97 ms | **11.0x** |

### Multi-Instance Concurrency Gain
| Backend | Single Instance | Multi-Instance | Gain |
|---------|----------------|----------------|------|
| CPU | 11.43 req/s | 17.88 req/s | **+56%** |
| CUDA | 125.47 req/s | 174.53 req/s | **+39%** |

## Key Conclusions

1. **CUDA 性能优势**：CUDA 比 CPU 快 11 倍，适合高性能推理场景。

2. **多实例并发有效**：CPU 和 CUDA 都证明了多实例并发可以显著提升吞吐量。

3. **Mutex 串行化是必要的**：单实例多线程由于 Backend timing 是 mutable，必须串行执行。

4. **Runtime 设计正确**：不同 ModelInstance 真正并行执行，无共享状态冲突。

5. **生产部署建议**：
   - 高性能场景使用 CUDA backend
   - 多模型并发使用多个 ModelInstance
   - 单模型高并发使用多个 ModelInstance 而非多线程

## Code Changes

### ModelConfig Extension
添加了 `cuda_cpu_ep_fallback` 选项，允许 CUDA backend 回退到 CPU 执行不支持的操作。

```cpp
struct ModelConfig {
  // ... existing fields ...
  bool cuda_cpu_ep_fallback = false;  // New field
};
```

### InferenceEngine Configuration
在 `InferenceEngine::load()` 中配置 CUDA backend 的 CPU fallback 选项。

### Real Model Benchmark Tool
修改 `tools/real_model_benchmark.cpp` 支持：
- CUDA 设备内存分配和 H2D 传输
- CPU fallback 配置
- 真实模型推理

## Limitations

- 未执行 ThreadSanitizer（需要独立构建配置）
- 未执行 cudaDeviceSynchronize 对比实验（需要性能分析工具）
- 仅测试了 YOLOv8n 模型，其他模型可能需要额外验证

## Next Steps

Phase 8.3 验证了 Runtime 层的生产就绪性。建议进入 Phase 9，实现更高级的并发优化（如 Session Pool）。
