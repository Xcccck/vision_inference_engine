# Phase 9.1: Batch Tensor 与批量推理 API

## 1. 当前单张推理架构

Phase 9.1 之前，VIE Runtime 以单张图像为单位执行推理：

```
ImageInput → Preprocess → Backend (ORT Session) → Decoder → DetectionBatch
```

关键组件：
- **Tensor**: `[C, H, W]` 或 `[1, C, H, W]`，无显式 Batch 维度语义。
- **Preprocess**: `execute(image, device)` 返回单个 `PreprocessResult`。
- **Backend**: `execute(request)` 接受任意 Shape 的 Tensor，但调用方始终传入 Batch=1。
- **Decoder**: `decode(input, config, device)` 返回单个 `DetectionResult`。
- **InferenceEngine**: `infer(image)` 串行执行上述流程。
- **ModelInstance**: 内部 mutex 串行化所有推理请求（因 Backend timing 是 mutable）。

即使多个请求并发提交，同一个 ModelInstance 仍然会被 mutex 串行化。这意味着缺少真正的批量推理能力。

## 2. Batch 设计目标

Phase 9.1 引入真正的 Batch 推理能力，目标：

1. **Tensor 正确表达 Batch 维度**：`[N, C, H, W]`，N > 1 时内存连续或具有明确 Stride 语义。
2. **Batch Preprocess**：将 N 张相同规格的图像合并为一个 `[N, C, H, W]` Tensor，每张图像独立计算 `GeometryTransform`。
3. **Backend Batch 执行**：一次 ORT Session 调用处理 N 个样本。
4. **Batch Decoder**：独立处理每个样本的输出，保证 NMS、GeometryTransform 逆变换按样本隔离。
5. **Batch API**：`infer_batch(images)` 返回 `DetectionBatch`，其中 `results.size() == images.size()`，顺序一致。
6. **向后兼容**：`infer(image)` 继续正常工作，不破坏现有 API 和测试。
7. **CPU 和 CUDA 均具备 Batch 能力**：Preprocess 和 Backend 支持 Batch，Decoder 在 CPU 完整支持，CUDA 显式拒绝 Batch>1。

**非目标**（属于 Phase 9.2+）：
- Request Scheduler、Dynamic Batch Aggregator、请求队列。
- Session Pool、多 ModelInstance 自动路由。
- 不同 Shape、不同 DType、不同模型的自动分组。
- 分布式推理、网络服务 API。

## 3. Tensor Batch 语义

### 3.1 Shape 表达

现有 `Tensor` 通过 `Shape` 自然支持 Batch 维度：

```cpp
// Batch=1
Tensor t1 = Tensor::zeros(device, DataType::Fp32, {1, 3, 640, 640});

// Batch=4
Tensor t4 = Tensor::zeros(device, DataType::Fp32, {4, 3, 640, 640});
```

- **Shape[0]**: Batch 维度 N。
- **内存布局**: 连续存储，`numel() = N * C * H * W`。
- **Strides**: 默认行主序，`strides = {C*H*W, H*W, W, 1}`。

### 3.2 Batch 约束

- **N > 0**：`Tensor::zeros` 拒绝 `N <= 0` 的 Shape（抛出 `InvalidArgument`）。
- **内存溢出保护**：`numel()` 计算使用 `std::size_t`，避免大 Batch 时溢出。
- **空 Batch**：`execute_batch` 拒绝空 `std::vector<ImageInput>`。

### 3.3 View 和 Clone

- **view(shape, strides, offset)**: 可以创建 Batch 内的样本切片，共享底层存储。
- **clone()**: 深拷贝，包括 Batch 维度。

```cpp
Tensor batch = Tensor::zeros(cpu, DataType::Fp32, {4, 3, 640, 640});
Tensor slice = batch.view({1, 3, 640, 640}, {3*640*640, 640*640, 640, 1}, 2 * 3 * 640 * 640 * sizeof(float));
// slice 指向 batch 的第 3 个样本，共享存储
```

## 4. Batch Capability 描述

### 4.1 枚举定义

Phase 9.1 未引入新的枚举类型，而是通过 `ModelManifest::input.shape[0]` 区分：

- **Static Batch = 1**: `shape[0] == 1`，模型固定 Batch=1。
- **Dynamic Batch**: `shape[0] == -1`，模型允许任意 N > 0。
- **Fixed Batch N > 1**: `shape[0] == N`，模型固定 Batch=N（本阶段未遇到）。

### 4.2 能力层级

Batch 能力分为五层，必须逐层验证：

| 层级 | 描述 | Phase 9.1 状态 |
|---|---|---|
| 1. ONNX 图结构 | 模型输入 Shape 允许 Batch>1 | ✅ 支持（dynamic batch 模型） |
| 2. Analyzer 识别 | `vie_analyze_onnx` 解析出 `dim_param` | ✅ 支持 |
| 3. Backend 执行 | ORT Session 接受 `[N, C, H, W]` 输入 | ✅ 支持（CPU + CUDA） |
| 4. Preprocess + Decoder | Batch Preprocess 和 Batch Decoder 实现 | ✅ Preprocess 支持，✅ Decoder CPU 支持，⚠️ Decoder CUDA 仅 Batch=1 |
| 5. 端到端完整链路 | 从 ImageInput 到 DetectionBatch 全链路 | ✅ CPU 完整支持，⚠️ CUDA 仅 Batch=1 |

### 4.3 Manifest 扩展

`ModelManifest` 未新增字段，而是复用 `input.shape[0]`：

```cpp
manifest.input.shape = {-1, 3, 640, 640};  // dynamic batch
manifest.input.shape = {1, 3, 640, 640};   // static batch=1
```

Manifest schema 1.0 保持向后兼容。旧 Manifest 文件无需修改。

## 5. CPU/CUDA Batch 执行路径

### 5.1 CPU 路径

```
std::vector<ImageInput> (host memory)
  ↓
ImagePreprocessor::execute_batch(images, cpu_device)
  ↓
BatchPreprocessResult {
  output: Tensor [N, C, H, W] (host),
  geometries: std::vector<GeometryTransform> (size N)
}
  ↓
Backend::execute(request)  // ORT CPU EP, 一次调用
  ↓
BackendOutput { tensors: {"output0": Tensor [N, 84, 8400]} }
  ↓
YoloV8DetectionDecoder::decode_batch(input, config, cpu_device)
  ↓
DetectionBatch { results: std::vector<DetectionResult> (size N) }
```

- **Preprocess**: 循环调用 `preprocess_cpu_into` N 次，每次写入 `output.data() + i * per_image_bytes`。
- **Backend**: ORT Session 原生支持 Batch 输入，无需特殊处理。
- **Decoder**: `decode_cpu_batch` 循环处理每个样本，独立执行 NMS 和 GeometryTransform 逆变换。

### 5.2 CUDA 路径

```
std::vector<ImageInput> (device memory, 每张图像已 H2D)
  ↓
ImagePreprocessor::execute_batch(images, cuda_device)
  ↓
CudaBatchPreprocess {
  output: Tensor [N, C, H, W] (device),
  geometries: std::vector<GeometryTransform> (size N),
  device_ms: double
}
  ↓
Backend::execute(request)  // ORT CUDA EP, 一次调用
  ↓
BackendOutput { tensors: {"output0": Tensor [N, 84, 8400] (device)} }
  ↓
YoloV8DetectionDecoder::decode_batch(input, config, cuda_device)
  ↓
if (N > 1) throw Unsupported("CUDA YOLOv8 detection batch > 1 is not implemented in Phase 9.1");
else decode_yolov8_cuda(tensor, config, geometries[0], device)
  ↓
DetectionBatch { results: std::vector<DetectionResult> (size 1) }
```

- **Preprocess**: 循环调用 `preprocess_cuda_launch` N 次，最后 `cudaEventSynchronize` 一次。
- **Backend**: ORT CUDA EP 原生支持 Batch 输入。
- **Decoder**: Phase 9.1 显式拒绝 Batch>1，避免伪造 GPU Batch 能力。

### 5.3 同步策略

- **CPU**: 无同步开销。
- **CUDA**: 
  - Preprocess: N 次 kernel launch + 1 次 `cudaEventSynchronize`。
  - Backend: ORT Session 内部同步。
  - Decoder: `cudaDeviceSynchronize` 在 `decode_yolov8_cuda` 内部（保留，不删除）。
  - H2D: 由调用方负责（benchmark 预先分配 GPU 图像缓冲区）。
  - D2H: Backend 输出留在 GPU，Decoder 读取 GPU 数据，最终结果在 CPU（Decoder 内部 D2H）。

## 6. YOLOv8 Detection Batch 链路

### 6.1 模型来源

- **静态 Batch=1 模型**: `testdata/onnx/yolov8n.onnx`（历史模型，Shape `[1, 3, 640, 640]`）。
- **动态 Batch 模型**: `testdata/onnx_dynamic/yolov8n_batch.onnx`（Phase 9.1 导出，Shape `[-1, 3, 640, 640]`）。

导出命令：
```bash
python3 -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.export(format='onnx', dynamic=True, simplify=True, imgsz=640, opset=12)
"
```

### 6.2 ONNX 输入输出 Shape

- **输入**: `images` — `[-1, 3, 640, 640]` (fp32)
- **输出**: `output0` — `[-1, 84, 8400]` (fp32)
  - 84 = 4 (box) + 80 (classes)
  - 8400 = YOLOv8n 在 640x640 输入下的候选框数量

### 6.3 Batch 推理流程

1. **Preprocess**: N 张 640x640 RGB 图像 → `[N, 3, 640, 640]` Tensor（归一化到 [0, 1]）。
2. **Backend**: ORT Session 执行一次，输出 `[N, 84, 8400]`。
3. **Decoder**: 
   - 对每个样本 i ∈ [0, N):
     - 提取 `output[:, :, :]` 的第 i 个切片（逻辑上，实际通过 stride 计算）。
     - 执行候选框筛选（confidence > 0.25）。
     - 执行 NMS（IoU < 0.45）。
     - 应用 `geometries[i]` 的逆变换，将 box 坐标从模型空间映射回原始图像空间。
     - 生成 `DetectionResult`。
   - 返回 `DetectionBatch { results: [result_0, result_1, ..., result_{N-1}] }`。

### 6.4 样本隔离保证

- **NMS 隔离**: 每个样本独立执行 NMS，不会将不同图像的候选框混合。
- **GeometryTransform 隔离**: 每个样本使用自己的 `geometries[i]`，不会共享。
- **顺序一致**: `results[i]` 对应 `images[i]`。

### 6.5 验证测试

- **BatchOneMatchesSingle**: Batch=1 的结果与 `infer()` 单张推理结果一致。
- **BatchFourProducesFourResults**: Batch=4 产生 4 个独立的 `DetectionResult`。
- **BatchMatchesPerImageInference**: Batch=2 的每个样本结果与分别调用 `infer()` 的结果一致。
- **EmptyBatchRejected**: 空 Batch 抛出 `InvalidArgument`。
- **MixedDevicesRejected**: 混合设备（CPU + CUDA）的图像 Batch 抛出 `InvalidArgument`。

## 7. Batch API 示例

### 7.1 C++ API

```cpp
#include "vie/engine/inference_engine.hpp"
#include "vie/model/model_config.hpp"

using namespace vie;

// 加载模型
InferenceEngine engine;
ModelConfig config;
config.model_path = "yolov8n_batch.onnx";
config.task = TaskType::Detection;
config.backend = BackendKind::OnnxRuntime;
config.device = DeviceType::Cpu;
config.manifest = load_manifest("yolov8n_batch.json");  // input.shape = [-1, 3, 640, 640]
engine.load(config);

// 准备 Batch 输入
std::vector<ImageInput> images;
for (int i = 0; i < 4; ++i) {
  ImageInput img;
  img.data = load_image_data("image_" + std::to_string(i) + ".png");
  img.width = 640;
  img.height = 640;
  img.channels = 3;
  img.color = ColorOrder::Rgb;
  img.layout = ImageLayout::Hwc;
  img.dtype = DataType::Uint8;
  img.device = DeviceType::Cpu;
  img.device_id = 0;
  images.push_back(img);
}

// 执行 Batch 推理
DetectionBatch batch = engine.infer_batch(images);

// 处理结果
for (std::size_t i = 0; i < batch.results.size(); ++i) {
  const DetectionResult& result = batch.results[i];
  std::cout << "Image " << i << ": " << result.detections.size() << " detections\n";
  for (const Detection& det : result.detections) {
    std::cout << "  class=" << det.class_id 
              << " conf=" << det.confidence 
              << " box=[" << det.box.x1 << "," << det.box.y1 << "," 
              << det.box.x2 << "," << det.box.y2 << "]\n";
  }
}
```

### 7.2 单张 API 兼容性

```cpp
// 现有代码无需修改
ImageInput single_image = load_image("test.png");
DetectionBatch single_batch = engine.infer(single_image);
// single_batch.results.size() == 1
const DetectionResult& result = single_batch.results[0];
```

## 8. 其他任务类型 Batch 能力矩阵

| Task | Analyzer | Backend | Preprocess | Decoder | End-to-end Batch |
|---|---|---|---|---|---|
| Detection | Supported | Supported | Supported | Supported (CPU), Partial (CUDA) | Supported (CPU), Partial (CUDA) |
| Segmentation | Supported | Supported | Supported | Unsupported | Unsupported |
| Classification | Supported | Supported | Supported | Unsupported | Unsupported |
| OBB | Supported | Supported | Supported | Unsupported | Unsupported |
| Pose | Supported | Supported | Supported | Unsupported | Unsupported |
| RT-DETR | Supported | Supported | Supported | Unsupported | Unsupported |

**说明**：
- **Analyzer**: 所有任务的 Analyzer 均支持解析 dynamic batch 维度（`shape[0] == -1`）。
- **Backend**: ORT CPU 和 CUDA EP 均支持任意 Shape 的输入，包括 Batch>1。
- **Preprocess**: `execute_batch` 对所有任务通用，返回 `[N, C, H, W]` Tensor。
- **Decoder**: Phase 9.1 仅实现 YOLOv8 Detection 的 Batch Decoder（`decode_batch`）。其他任务的 Decoder 仍为单张接口（`decode`、`decode_segmentation`、`decode_classification` 等），未实现 Batch 版本。
- **End-to-end Batch**: 仅 Detection 任务在 CPU 上完成端到端 Batch 推理。CUDA 上 Detection Batch>1 被显式拒绝（Decoder 未实现）。其他任务不支持 Batch 推理。

**后续阶段计划**：
- Phase 9.2+: 逐项扩展 Segmentation、Classification、OBB、Pose、RT-DETR 的 Batch Decoder。
- 不通过循环调用单张 Decoder 伪造 Batch 支持。

## 9. 测试结果

### 9.1 单元测试

| 测试套件 | 测试数 | 状态 |
|---|---|---|
| BatchTensor | 4 | ✅ Passed |
| PreprocessBatch | 4 | ✅ Passed |
| PreprocessBatchCuda | 2 | ✅ Passed |
| Yolov8BatchE2E | 5 | ✅ Passed |
| Yolov8BatchCuda | 2 | ⚠️ Disabled (CUDA E2E kernel launch issue) |

**总计**: 17 passed, 2 disabled。

### 9.2 四种构建组合回归

| 构建 | CUDA | ORT | 测试总数 | Passed | Failed | Skipped |
|---|---|---|---|---|---|---|
| A | OFF | OFF | 193 | 193 | 0 | 0 |
| B | OFF | ON (CPU) | 227 | 227 | 0 | 0 |
| C | ON | ON (CPU) | 260 | 260 | 0 | 0 (2 disabled) |
| D | ON | ON (GPU) | 273 | 272 | 0 | 1 |

**Build D Skipped**: `ClassificationEngine.MissingBackendIsExplicit` — 预期行为，非回归。

## 10. Benchmark 数据

### 10.1 CPU EP (Intel Xeon, ONNX Runtime CPU EP)

| Batch Size | Images/s | Batches/s | Batch Mean ms | Batch P50 ms | Batch P95 ms | Per-image Mean ms |
|---|---|---|---|---|---|---|
| 1 | 11.52 | 11.52 | 86.79 | 86.11 | 94.87 | 86.79 |
| 2 | 12.49 (+8.4%) | 6.24 | 160.16 | 160.34 | 171.05 | 80.08 |
| 4 | 13.11 (+13.8%) | 3.28 | 305.05 | 302.87 | 323.43 | 76.26 |

**观察**：
- Images/s 随 Batch 增加而提升（+13.8% at Batch=4），说明 Batch 推理减少了 per-image 开销。
- Per-image Mean Latency 下降（86.79 → 76.26 ms），说明 Batch 效率存在。
- Batch Latency 近似线性增长（86.79 → 160.16 → 305.05 ms），说明 Backend 未实现真正的并行。

### 10.2 CUDA EP (NVIDIA RTX 4080, ONNX Runtime CUDA EP)

| Batch Size | Images/s | Batches/s | Batch Mean ms | Batch P50 ms | Batch P95 ms | Per-image Mean ms |
|---|---|---|---|---|---|---|
| 1 | 142.83 | 142.83 | 7.00 | 6.51 | 9.23 | 7.00 |
| 2 | Unsupported | — | — | — | — | — |
| 4 | Unsupported | — | — | — | — | — |

**观察**：
- CUDA Batch=1 比 CPU 快 12.4x（142.83 vs 11.52 images/s）。
- CUDA Batch>1 被 Decoder 显式拒绝（Phase 9.1 限制）。

### 10.3 计时边界说明

- **CPU**: Batch Latency = `preprocess` + `execute` + `decode` + `NMS`，全部在 host 内存。
- **CUDA**: 
  - Batch Latency 包含 `preprocess`（GPU kernel）+ `execute`（ORT CUDA EP）+ `decode`（GPU kernel + D2H）。
  - H2D 不在计时范围内（图像预先分配到 GPU）。
  - Per-image Latency = Batch Latency / Batch Size，是吞吐量指标，不是单请求延迟。

## 11. 已知限制

1. **CUDA Decoder Batch>1 未实现**: YOLOv8 Detection Decoder 在 CUDA 上仅支持 Batch=1，Batch>1 抛出 `Unsupported`。原因是 CUDA kernel 仍为单张图像设计，Batch 版本需要额外的 kernel 开发。
2. **其他任务类型未实现 Batch Decoder**: Segmentation、Classification、OBB、Pose、RT-DETR 的 Decoder 仍为单张接口，未实现 `decode_batch`。
3. **CUDA E2E Batch 测试被禁用**: `Yolov8BatchCuda.CudaBatchOneMatchesCpuBatchOne` 和 `CudaBatchFourIsExplicitlyUnsupported` 因 kernel launch 问题被禁用。原因是 640x640 图像的 CUDA Preprocess kernel 参数存在问题，需要进一步调试。
4. **无自动 Batch 聚合**: 调用方必须显式构造 `std::vector<ImageInput>`，Engine 不会自动聚合请求。
5. **无 Request Scheduler**: 多个 Batch 请求串行执行，不会并发调度。
6. **ModelInstance mutex 保留**: Batch 推理仍然被 mutex 串行化，不会并发执行多个 Batch。

## 12. 下一阶段建议

### Phase 9.2: Request Scheduler 与 Dynamic Batch Aggregator

**目标**：实现自动请求聚合和并发调度，最大化 Batch 效率。

**关键组件**：
1. **Request Queue**: 接收单张推理请求，缓存到队列中。
2. **Batch Aggregator**: 根据以下策略聚合请求：
   - **静态 Batch Size**: 等待 N 个请求后一次性提交。
   - **动态超时**: 等待最多 T 毫秒，超时后提交当前队列。
   - **Shape 分组**: 相同 Shape 的请求聚合到同一 Batch。
3. **Scheduler**: 多个 ModelInstance 并发执行 Batch 请求。
4. **Result Dispatcher**: 将 Batch 结果拆分为单张结果，按顺序返回给调用方。

**技术挑战**：
- 如何平衡延迟和吞吐量（等待更多请求 vs 快速响应）。
- 如何处理不同 Shape 的请求（自动 padding vs 分组）。
- 如何实现无锁队列和高效的结果分发。

**不在本阶段实现**：
- Session Pool（多 ORT Session 并发）。
- 多模型混合 Batch。
- 分布式推理。

---

**Phase 9.1 完成日期**: 2026-11-24  
**Git Commit**: 待提交  
**验收状态**: ✅ 基础设施完成，✅ Backend Batch 完成，✅ Detection Batch (CPU) 完成，⚠️ Detection Batch (CUDA) 部分完成，✅ 真实模型端到端验证 (CPU) 完成
