# Phase 9.2.1: 异步 Request Queue 与 Future 请求接口

## 1. 背景与动机

Phase 9.1 引入了 Batch 推理能力，但请求调度仍然是同步的：调用方直接驱动 `ModelInstance::infer()`，通过 `ThreadPool` 实现并发。这种模式存在以下局限：

1. **无背压机制**：调用方可以无限提交任务到 ThreadPool，无法限制排队请求数量。
2. **无请求级生命周期追踪**：无法度量从提交到完成的端到端延迟。
3. **无 Graceful Shutdown**：Runtime 析构时 ThreadPool 完成待处理任务，但没有独立的请求队列关闭语义。
4. **模型卸载安全性**：调用方通过 `shared_ptr<ModelInstance>` 持有模型引用，但如果通过 model_id 间接引用，模型可能在请求执行前被卸载。

Phase 9.2.1 引入独立的异步请求队列（RequestQueue）和 `std::future<DetectionResult>` 接口，为后续 Phase 9.2.2（Dynamic Batch Aggregator）奠定基础。

## 2. 设计目标

### 2.1 核心目标

- `runtime.submit(model_id, image)` 返回 `std::future<DetectionResult>`
- 线程安全、有界的 RequestQueue（独立于 ThreadPool 内部队列）
- 后台 Worker 线程从队列取请求，调用 `ModelInstance::infer()` 执行推理
- Promise/Future 结果返回，异常正确传播
- Graceful Shutdown：shutdown 后拒绝新请求，已接受请求继续执行完成
- 请求级 Metrics（submitted/accepted/rejected/completed/failed/queue_wait/execution/e2e）

### 2.2 非目标（Phase 9.2.2+）

- Dynamic Batch Aggregator（将多个单张请求合并为 Batch 推理）
- Session Pool
- 无锁队列
- 修改 ModelInstance 的 mutex 语义
- 重写 ThreadPool

## 3. 架构设计

### 3.1 组件关系

```
                      submit()                    RequestQueue              Worker Threads
                      ────────                    ────────────              ──────────────
 Caller Thread ──────────────────►  InferenceRequest ──► [bounded queue] ──►  worker_loop()
                                        │                      │                │
                                        │ promise              │ wait_pop       │ model->infer()
                                        ▼                      ▼                ▼
                                 std::future            mutex + cv        DetectionResult
                                        │                                   │
                                        └───────── set_value() ◄────────────┘
```

### 3.2 新增文件

| 文件 | 职责 |
|------|------|
| `include/vie/runtime/inference_request.hpp` | InferenceRequest 数据结构 |
| `include/vie/runtime/request_queue.hpp` | RequestQueue 接口 |
| `src/runtime/request_queue.cpp` | RequestQueue 实现 |
| `tests/test_async_runtime.cpp` | 18 个测试（8 RequestQueue + 10 AsyncRuntime） |

### 3.3 修改文件

| 文件 | 变更 |
|------|------|
| `include/vie/core/error.hpp` | 新增 4 个 ErrorCode |
| `include/vie/runtime/runtime_metrics.hpp` | 新增请求级 Metrics 字段 |
| `src/runtime/runtime_metrics.cpp` | 新增 Metrics 方法实现 |
| `include/vie/runtime/runtime_config.hpp` | 新增 queue_capacity, worker_count, defer_start |
| `include/vie/runtime/runtime.hpp` | 新增 submit/start/shutdown/worker_loop |
| `src/runtime/runtime.cpp` | 异步请求调度核心实现 |

## 4. 核心数据结构

### 4.1 InferenceRequest

```cpp
struct InferenceRequest {
  RequestId id = 0;                              // 原子递增的唯一标识
  std::string model_id;                          // 模型 ID（诊断用）
  ImageInput image;                              // 借用指针，调用方负责生命周期
  std::shared_ptr<ModelInstance> model;          // submit() 时解析，保持模型存活
  std::chrono::steady_clock::time_point submitted_at;
  std::promise<DetectionResult> promise;         // 结果通道
};
```

**关键设计决策**：`model` 字段在 `submit()` 时从 ModelManager 解析为 `shared_ptr`，而非在 Worker 执行时查找。这确保：
- 即使模型在排队期间被 `unload()`，Worker 仍持有有效引用
- `unload()` 语义为"从 Manager 移除，禁止新查找"，不影响已排队请求

### 4.2 RequestQueue

```cpp
class RequestQueue {
 public:
  explicit RequestQueue(std::size_t capacity);
  bool push(InferenceRequest request);     // 非阻塞，满或 shutdown 返回 false
  bool wait_pop(InferenceRequest& request); // 阻塞，空且 shutdown 返回 false
  void shutdown();                          // 设置标志，notify_all
  bool is_shutdown() const;
  std::size_t size() const;
  bool empty() const;
  std::size_t capacity() const noexcept;
};
```

**线程安全**：`std::mutex` + 两个 `std::condition_variable`（`not_empty_`, `not_full_`）。

**Graceful Shutdown 语义**：
- `shutdown()` 设置 `shutdown_ = true`，`notify_all` 唤醒所有等待线程
- `push()` 在 shutdown 后返回 false（拒绝新请求）
- `wait_pop()` 在 shutdown 且队列为空时返回 false；队列非空时继续返回 true（处理完已排队请求）

### 4.3 ErrorCode 扩展

```cpp
enum class ErrorCode {
  // ... existing codes ...
  RuntimeNotStarted,     // submit() 在 Runtime 未启动时调用
  RuntimeShuttingDown,   // 操作在 shutdown 过程中
  RequestQueueFull,      // 队列已满，拒绝新请求
  InvalidModelId,        // model_id 未加载
};
```

## 5. Runtime 生命周期

### 5.1 启动

```
Runtime(config)
  ├─ running_ = true
  ├─ request_queue_ = RequestQueue(config.request_queue_capacity)
  └─ if (!config.defer_start)
       └─ launch_workers()
            └─ 启动 config.inference_worker_count 个 worker 线程
```

`defer_start` 选项允许测试在填充队列后再启动 Worker。

### 5.2 请求提交

```
submit(model_id, image)
  ├─ 检查 running_ == true，否则抛 RuntimeNotStarted
  ├─ models_.get(model_id) 解析 shared_ptr，否则抛 InvalidModelId
  ├─ 构造 InferenceRequest（含 model shared_ptr）
  ├─ future = request.promise.get_future()
  ├─ metrics.record_request_submitted()
  ├─ request_queue_.push(request)
  │    └─ 失败时 record_request_rejected()，抛 RequestQueueFull
  ├─ metrics.record_request_accepted()
  └─ return future
```

### 5.3 Worker 执行

```
worker_loop()
  └─ while (true)
       ├─ request_queue_.wait_pop(request)
       │    └─ 返回 false 时 break（队列空且 shutdown）
       ├─ 记录 dequeue_time，计算 queue_wait_ms
       ├─ try:
       │    ├─ 检查 request.model != nullptr
       │    ├─ batch = request.model->infer(request.image)
       │    ├─ request.promise.set_value(batch.results[0])
       │    └─ success = true
       ├─ catch:
       │    └─ request.promise.set_exception(current_exception())
       ├─ 记录 execution_ms, e2e_ms
       └─ metrics.record_request_completed(...)
```

### 5.4 关闭

```
shutdown()
  ├─ running_ = false
  ├─ request_queue_.shutdown()    // 拒绝新 push，唤醒等待的 worker
  └─ for worker in inference_workers_:
       └─ worker.join()           // 等待 worker 处理完剩余请求后退出
```

**Graceful 语义**：shutdown 后 Worker 继续处理队列中已排队的请求，直到队列为空才退出。

### 5.5 析构

```
~Runtime()
  └─ shutdown()    // 确保 graceful 关闭
```

## 6. 请求级 Metrics

### 6.1 新增字段

```cpp
struct RuntimeMetrics {
  // Phase 9.2.1
  std::size_t submitted_requests = 0;     // 提交总数
  std::size_t accepted_requests = 0;      // 入队总数
  std::size_t rejected_requests = 0;      // 被拒绝总数（队列满/shutdown）
  std::size_t completed_requests = 0;     // 成功完成总数
  std::size_t failed_requests = 0;        // 执行失败总数
  double total_queue_wait_time_ms = 0.0;  // 排队等待总时间
  double total_execution_time_ms = 0.0;   // 推理执行总时间
  double total_e2e_time_ms = 0.0;         // 端到端总时间（wait + execution）
  std::size_t current_queue_size = 0;     // 当前队列大小
  std::size_t max_queue_size = 0;         // 历史最大队列大小
};
```

### 6.2 时间度量

- **queue_wait_time** = `dequeue_time - submitted_at`（从提交到 Worker 取出的等待时间）
- **execution_time** = `execution_end - execution_start`（推理执行耗时）
- **e2e_time** = `execution_end - submitted_at`（端到端总耗时）

所有计时使用 `std::chrono::steady_clock`，不受系统时钟调整影响。

## 7. 错误传播

### 7.1 提交阶段（同步）

| 条件 | ErrorCode | 行为 |
|------|-----------|------|
| Runtime 未启动/shutdown | `RuntimeNotStarted` | 同步抛 VieError |
| model_id 不存在 | `InvalidModelId` | 同步抛 VieError |
| 队列已满 | `RequestQueueFull` | 同步抛 VieError |

### 7.2 执行阶段（异步）

| 条件 | 行为 |
|------|------|
| 模型在排队期间被卸载 | 不会发生（submit 时已持有 shared_ptr） |
| 推理异常 | `promise.set_exception()`，通过 `future.get()` 抛出 |
| Promise 已满足/broken | `set_exception()` 的 catch 块静默忽略 |

## 8. 测试覆盖

### 8.1 RequestQueue 单元测试（8 个）

| 测试 | 验证内容 |
|------|----------|
| Constructs | 初始状态：capacity、size、empty、is_shutdown |
| RejectsZeroCapacity | capacity=0 抛 std::invalid_argument |
| PushAndWaitPop | 基本 FIFO 入队出队 |
| PushRejectsWhenFull | 有界队列满时 push 返回 false |
| ShutdownRejectsPush | shutdown 后 push 返回 false |
| ShutdownWakesWaitingPop | shutdown 唤醒阻塞的 wait_pop 线程 |
| GracefulShutdownProcessesQueuedRequests | shutdown 后仍可 pop 已排队请求 |
| MultiThreadedPushPop | 多线程并发 push/pop 正确性 |

### 8.2 AsyncRuntimeTest 集成测试（10 个）

| 测试 | 验证内容 |
|------|----------|
| SubmitsAndReceivesResult | 基本 submit → future.get() 流程 |
| MultipleRequestsComplete | 多个请求全部完成，Metrics 正确 |
| InvalidModelIdThrows | 无效 model_id 同步抛异常 |
| QueueFullRejectsRequest | 队列满时同步抛 RequestQueueFull |
| ShutdownRejectsNewRequests | shutdown 后 submit 抛 RuntimeNotStarted |
| GracefulShutdownCompletesQueuedRequests | shutdown 不丢弃已排队请求 |
| ExceptionPropagatesThroughFuture | 无效 model 异常通过 future 传播 |
| ConcurrentSubmitsFromMultipleThreads | 4 线程 × 3 请求并发正确性 |
| ModelUnloadDuringRequest | 模型卸载不影响已排队请求（shared_ptr 保护） |
| MetricsRecorded | 请求级 Metrics 字段正确记录 |

## 9. 四构建回归结果

| Build | CUDA | ORT | Tests | Result |
|-------|------|-----|-------|--------|
| A | OFF | OFF | 201 | ✓ All passed |
| B | OFF | ON (CPU) | 245 | ✓ All passed |
| C | ON | OFF | 278 | ✓ All passed |
| D | ON | ON (GPU) | 290 | ✓ All passed |

注：AsyncRuntimeTest 集成测试需要 ONNX Runtime（Build B/C/D），RequestQueue 单元测试在所有构建中运行。

## 10. 配置参数

```cpp
struct RuntimeConfig {
  // ... existing fields ...
  std::size_t request_queue_capacity = 1024;  // 请求队列最大容量
  std::size_t inference_worker_count = 2;     // 推理 Worker 线程数
  bool defer_start = false;                   // 延迟启动 Worker（测试用）
};
```

**request_queue_capacity**：队列满时新请求被同步拒绝（抛 `RequestQueueFull`）。默认 1024。

**inference_worker_count**：Worker 线程数。每个 Worker 从同一队列取请求并串行执行推理。默认 2。多个 Worker 可以并行推理不同的 ModelInstance（同一 ModelInstance 仍被其 mutex 串行化）。

## 11. 与 ThreadPool 的关系

Phase 9.2.1 新增的异步请求通道与现有 ThreadPool 并存：

- **ThreadPool**：通用任务提交（`submit(callable)`），用于 preprocess、自定义任务等
- **RequestQueue + Workers**：专用的推理请求调度通道，支持请求级 Metrics、背压、Graceful Shutdown

后续 Phase 9.2.2 的 Dynamic Batch Aggregator 将基于 RequestQueue 构建，不影响 ThreadPool 的使用方式。
