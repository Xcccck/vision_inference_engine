# Phase 8 架构审计报告

本文档记录 VIE 在 Phase 8 开始前的架构审计结果，重点分析生命周期、线程安全和资源所有权。

## 1. 当前架构概览

```text
InferenceEngine
    |
    +-- BackendRegistry (factories, no mutex)
    +-- DecoderRegistry   (factories, has mutex)
    +-- ModelManifest     (value, copied on load)
    +-- unique_ptr<IInferenceBackend>  (exclusive)
    +-- unique_ptr<IModelDecoder>      (exclusive)
    +-- DeviceContext     (value, shared CudaState)
```

## 2. Engine 生命周期

### 2.1 所有权

- Engine **独占** Backend 和 Decoder（`unique_ptr`）
- Engine **持有** Manifest 的值拷贝
- Engine **持有** DeviceContext 值（CUDA 时内含 `shared_ptr<CudaState>`）
- BackendRegistry 和 DecoderRegistry 是 Engine 成员，在构造时注册所有工厂

### 2.2 线程安全

Engine 头文件明确注释：

> One engine is not safe for concurrent load(), infer(), infer_segmentation(),
> infer_classification(), infer_obb(), or infer_pose() calls. Those calls share
> the session with no lock; overlapping them is a data race.

**结论：同一 Engine 实例不能并发调用 infer/load。不同 Engine 实例之间无共享状态，可以并发。**

### 2.3 生命周期

- `load()` 创建 Backend 和 Decoder，替换之前的（如果有）
- 析构时 `unique_ptr` 自动释放 Backend 和 Decoder
- 失败的 `load()` 保留之前的 session

## 3. Backend 生命周期

### 3.1 ORT CPU Backend

每个 `OnnxRuntimeBackend::Impl` 包含：
- `Ort::Env env` — ORT 环境
- `Ort::SessionOptions options` — 会话选项
- `std::unique_ptr<Ort::Session> session` — ORT 会话
- `OnnxRuntimeTiming timing` — **可变状态**，每次 execute() 重置

关键发现：
- `prepare()` 创建新的 ORT Session
- `execute()` 修改 `impl_->timing`，**不是线程安全的**
- Ort::Value 借用输入 Tensor 数据指针，在 execute() 范围内有效
- 输出 Tensor 被拷贝到 VIE 管理的存储

### 3.2 ORT CUDA Backend

每个 `OnnxRuntimeCudaBackend::Impl` 包含：
- 与 CPU Backend 相同的 ORT 资源
- `DeviceContext device` — CUDA 设备上下文
- `std::mutex log_mu` — 保护 session_log
- `bool cpu_ep_fallback` — 是否允许 CPU 回退

关键发现：
- `execute()` 创建局部 `Ort::IoBinding`，每次调用独立
- `execute()` 修改 `impl_->timing`，**不是线程安全的**
- 调用 `cudaDeviceSynchronize()` 确保输入/输出同步
- ORT 使用自己的 CUDA stream，不使用 DeviceContext 的 stream

### 3.3 Backend 并发结论

- **同一 Backend 实例**：不能并发 execute()（timing 竞争）
- **不同 Backend 实例**：可以并发（各自独立的 ORT Session）
- **ORT Session 本身**：ORT 文档表明同一 Session 可以并发 Run，但 VIE 的 Backend 包装了可变 timing 状态

## 4. Tensor 生命周期

### 4.1 存储所有权

```text
TensorStorage (shared_ptr)
    |
    +-- Tensor A (view)
    +-- Tensor B (view)
    +-- Tensor C (clone, new storage)
```

- `TensorStorage` 通过 `shared_ptr` 共享
- `Tensor` 是 view：shape + strides + byte_offset + `shared_ptr<TensorStorage>`
- `clone()` 分配新的独立 `TensorStorage`
- `to(device)` 在目标设备分配新存储并拷贝数据

### 4.2 线程安全

- `shared_ptr` 引用计数是原子的
- **数据读写不是线程安全的**：多个线程同时写同一 Storage 是 data race
- 不同 Storage 的并发读写是安全的

### 4.3 CUDA 内存

- `allocate_cuda()` 使用 `cudaMalloc`，同步 `cudaMemsetAsync` + `cudaStreamSynchronize`
- `release_cuda()` 使用 `cudaFree`
- `copy_device_memory()` 使用 `cudaMemcpyAsync` + `cudaStreamSynchronize`（如果有 CudaState）
- 所有 CUDA 操作在拷贝返回前已同步

## 5. DeviceContext

### 5.1 结构

```text
DeviceContext
    +-- DeviceType device (Cpu or Cuda)
    +-- int device_id
    +-- shared_ptr<CudaState> cuda (only for CUDA)
```

### 5.2 CudaState

```text
CudaState
    +-- int device_id
    +-- cudaStream_t stream
```

**关键发现：每次构造 CUDA DeviceContext 都会创建新的 cudaStream。**

- 两个 Engine 各自创建 CUDA DeviceContext → 两个独立的 CUDA stream
- `shared_ptr<CudaState>` 允许拷贝 DeviceContext 时共享同一 stream
- 析构时销毁 stream

### 5.3 并发含义

- 不同 Engine 的 CUDA 操作在不同 stream 上，可以并发
- 同一 Engine 的 CUDA 操作在同一 stream 上，按序执行
- `cudaDeviceSynchronize()` 等待**所有** stream，不仅仅是自己的

## 6. Decoder

### 6.1 接口

所有 decode 方法都是 `const`：
- `decode()` — 原始协议解码
- `decode_batch()` — Detection
- `decode_segmentation()` — Segmentation
- `decode_classification()` — Classification
- `decode_obb()` — OBB
- `decode_pose()` — Pose

### 6.2 线程安全

**Decoder 在 decode 调用中不修改自身状态。多个线程可以并发调用同一 Decoder 实例的 const 方法。**

### 6.3 实际使用

Engine 在 `load()` 时创建一个 Decoder，在每次 infer 时调用其 const 方法。由于方法是 const 的，理论上多个线程可以并发调用同一 Engine 的 Decoder，但 Engine 的 Backend 不是线程安全的，所以实际上不能并发使用同一 Engine。

## 7. Preprocess

### 7.1 结构

```text
ImagePreprocessor
    +-- PreprocessConfig config_ (set by configure())
    +-- bool configured_ (set by configure())
```

### 7.2 线程安全

- `configure()` 修改状态 — **不能与其他调用并发**
- `validate()` 和 `execute()` 是 `const` — **可以并发调用**

### 7.3 实际使用

Engine::preprocess() 每次创建局部 ImagePreprocessor，configure 后 execute。不存在共享状态问题。

## 8. Registry

### 8.1 BackendRegistry

- `std::map<BackendKind, Factory> factories_`
- **无 mutex**
- `register_factory()` 在 Engine 构造时调用
- `create()` 在 Engine::load() 时调用
- 如果多个线程同时创建 Engine，`register_factory` 和 `create` 可能竞争

### 8.2 DecoderRegistry

- `std::map<std::string, Factory> factories_`
- **有 `mutable std::mutex mutex_`**
- 线程安全

### 8.3 结论

BackendRegistry 需要增加 mutex 以与 DecoderRegistry 保持一致。

## 9. 当前限制总结

| 组件 | 并发安全 | 说明 |
|---|---|---|
| Engine | 否 | 同一实例不能并发 infer/load |
| Backend | 否 | 同一实例 timing 竞争 |
| Decoder | 是 | 所有 decode 方法 const |
| Preprocess | 是 | execute() const，每次创建局部实例 |
| Tensor 数据 | 否 | 共享 Storage 的并发写是 data race |
| Tensor Storage 生命周期 | 是 | shared_ptr 引用计数原子 |
| DeviceContext (CPU) | 是 | 只读数据 |
| DeviceContext (CUDA) | 部分 | 共享 CudaState 的 stream 可并发，但 cudaSetDevice 是进程级的 |
| BackendRegistry | 否 | 无 mutex |
| DecoderRegistry | 是 | 有 mutex |

## 10. Phase 8 设计决策

基于以上审计，Phase 8 的 Runtime 层应采取以下策略：

### 10.1 多模型并发

- 每个模型实例持有独立的 Engine（包含独立的 Backend 和 Decoder）
- 不同模型的并发通过不同 Engine 实现，天然安全
- Runtime 层的 Model Registry 使用 `shared_ptr<Engine>` 管理模型实例

### 10.2 同一模型并发

- 同一 Engine 实例不能并发 infer
- 方案 A：在 ModelInstance 内加 mutex，串行化同一模型的请求
- 方案 B：为每个并发请求创建独立 Engine（开销大，不推荐）
- **选择方案 A**：简单、安全、性能损失可接受（ORT Session 本身支持并发，但 Backend timing 不支持）

### 10.3 ThreadPool

- 实现轻量级 ThreadPool，固定 worker 数量
- 任务提交返回 `std::future`
- 异常通过 future 传递给调用者
- 队列无界，不拒绝任务

### 10.4 BackendRegistry 修复

- 增加 `mutable std::mutex` 以与 DecoderRegistry 一致

### 10.5 CUDA 并发

- 每个 Engine 创建自己的 DeviceContext（独立 CUDA stream）
- ORT CUDA Backend 使用自己的 stream，不使用 DeviceContext 的
- 不同 Engine 的 CUDA 推理在不同 ORT stream 上并发
- `cudaDeviceSynchronize()` 在 Backend execute() 内等待所有 stream

### 10.6 资源释放

- ModelInstance 使用 `shared_ptr` 管理
- unload 时从 Registry 移除入口，但已持有的引用可以继续执行
- Engine 析构时自动释放 Backend、Decoder 和 DeviceContext
- CUDA 资源在 Backend 析构和 DeviceContext 析构时同步并释放
