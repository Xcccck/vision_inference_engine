# 模型兼容性

本文档记录 VIE 当前支持的六类视觉任务、已验证模型、对应协议和执行支持情况。

## 1. 支持的任务

| 任务 | TaskType | 说明 |
|---|---|---|
| Detection | `detection` | YOLOv8 目标检测 |
| Segmentation | `segmentation` | YOLOv8 实例分割 |
| OBB | `obb` | YOLOv8 旋转目标检测 |
| Pose | `pose` | YOLOv8 人体关键点估计 |
| RT-DETR | `detection` | RT-DETR 目标检测 |
| Classification | `classification` | 单标签图像分类 |

## 2. 已验证模型

### 2.1 Detection

| 模型 | 协议 | 识别器 | Opset | Input Shape | Output Shape |
|---|---|---|---|---|---|
| YOLOv8n | `yolov8_detect_raw` | `yolov8_detection` | 20 | [1,3,640,640] | [1,84,8400] |

### 2.2 Segmentation

| 模型 | 协议 | 识别器 | Opset | Input Shape | Output Shapes |
|---|---|---|---|---|---|
| YOLOv8n-seg | `yolov8_seg_raw` | `yolov8_segmentation` | 17 | [1,3,640,640] | [1,117,8400], [1,32,160,160] |

### 2.3 OBB

| 模型 | 协议 | 识别器 | Opset | Input Shape | Output Shape |
|---|---|---|---|---|---|
| YOLOv8n-obb | `yolov8_obb_raw` | `yolov8_obb` | 17 | [1,3,1024,1024] | [1,20,8400] |

### 2.4 Pose

| 模型 | 协议 | 识别器 | Opset | Input Shape | Output Shape |
|---|---|---|---|---|---|
| YOLOv8n-pose | `yolov8_pose_raw` | `yolov8_pose` | 17 | [1,3,640,640] | [1,56,8400] |

`yolov8_pose_raw` 的输入契约（CPU 与 CUDA Decoder 共用）：

- rank-3 FP32、连续内存、batch 必须为 1；`channel_axis=1` 为 `[1, C, P]`，`channel_axis=2` 为 `[1, P, C]`。
- `C == 4 + num_classes + keypoint_count * 3`；`keypoint_count` 来自 Manifest，每关键点固定为 `(x, y, confidence)`。
- 前 4 个通道是已解码的 `cx, cy, w, h`（模型输入空间），后面是类别分数，再后面是关键点；`w`/`h` 必须为正。
- CPU Decoder 与 CUDA Decoder 都用 `GeometryTransform` 把人体框与关键点从模型输入空间逆映射回原图，再按 Manifest 的 NMS 策略过滤。
- 不支持的输入：batch > 1、非 FP32、非连续 Tensor、通道数与 Manifest 不一致、模型内置 NMS。

### 2.5 RT-DETR

| 模型 | 协议 | 识别器 | Opset | Input Shape | Output Shapes |
|---|---|---|---|---|---|
| RT-DETR-L | `rtdetr_detection_raw` | `rtdetr_detection` | 17 | [1,3,640,640] | [1,300,4], [1,300] |

### 2.6 Classification

| 模型 | 协议 | 识别器 | Opset | Input Shape | Output Shape |
|---|---|---|---|---|---|
| ResNet18 | `classification_logits_raw` | `classification_logits` | 17 | [1,3,224,224] | [1,1000] |
| ResNet18 + Softmax | `classification_probabilities_raw` | `classification_logits` | 17 | [1,3,224,224] | [1,1000] |
| MobileNetV2 | `classification_logits_raw` | `classification_logits` | 17 | [1,3,224,224] | [1,1000] |
| MobileNetV3-Small | `classification_logits_raw` | `classification_logits` | 17 | [1,3,224,224] | [1,1000] |
| EfficientNet-B0 | `classification_logits_raw` | `classification_logits` | 17 | [1,3,224,224] | [1,1000] |

## 3. Batch 支持

| 任务 | Batch 1 | Batch N | 说明 |
|---|---|---|---|
| Detection | 支持 | 不支持 | Decoder 仅实现 batch 1 |
| Segmentation | 支持 | 不支持 | Decoder 仅实现 batch 1 |
| OBB | 支持 | 不支持 | Decoder 仅实现 batch 1 |
| Pose | 支持 | 不支持 | Decoder 仅实现 batch 1 |
| RT-DETR | 支持 | 不支持 | Decoder 仅实现 batch 1 |
| Classification | 支持 | 不支持 | Decoder 仅实现 batch 1 |

当前所有 Decoder 仅支持 batch 1。ONNX Analyzer 可以解析动态 batch 维度，但 Engine 不执行 batch N 推理。

## 4. Dynamic Input 支持

| 任务 | 动态 H/W | 动态 Batch | 说明 |
|---|---|---|---|
| Detection | 不支持 | 不支持 | Preprocess 需要固定输入尺寸 |
| Segmentation | 不支持 | 不支持 | Preprocess 需要固定输入尺寸 |
| OBB | 不支持 | 不支持 | Preprocess 需要固定输入尺寸 |
| Pose | 不支持 | 不支持 | Preprocess 需要固定输入尺寸 |
| RT-DETR | 不支持 | 不支持 | Preprocess 需要固定输入尺寸 |
| Classification | 不支持 | 不支持 | Preprocess 需要固定输入尺寸 |

ONNX Analyzer 可以解析动态维度（symbolic dimensions），但当前 Preprocess 和 Backend 要求固定输入尺寸。如果模型使用动态输入，Manifest Validator 会报告 unresolved critical 错误。

## 5. CPU EP 支持

所有六类任务都有 CPU Decoder。使用 ORT CPU Backend 时，整个推理链路在 CPU 上执行。

| 任务 | CPU EP | 说明 |
|---|---|---|
| Detection | 支持 | YoloV8DetectionDecoder |
| Segmentation | 支持 | YoloV8SegmentationDecoder |
| OBB | 支持 | YoloV8ObbDecoder |
| Pose | 支持 | YoloV8PoseDecoder |
| RT-DETR | 支持 | RtdetrDetectionDecoder |
| Classification | 支持 | ClassificationDecoder / ClassificationProbabilitiesDecoder |

## 6. 严格 CUDA EP 支持

CUDA EP 使用 ORT CUDA Backend 执行模型前向。Detection / Instance Segmentation /
RT-DETR / Classification / Semantic Segmentation / Pose / OBB 在 CUDA 上有真正的后处理
（Phase 13 补齐分类 Top-K 与语义分割 ArgMax，Phase 14 补齐 Pose 关键点解码与人体框 NMS，
Phase 15 补齐 OBB 旋转框解码、Polygon IoU 与 Rotated NMS）。

| 任务 | CUDA EP | 说明 |
|---|---|---|
| Detection | 支持 | 前向 CUDA + CUDA Decoder |
| Segmentation | 支持 | 前向 CUDA + CUDA Decoder |
| OBB | 支持 | 前向 CUDA + CUDA Decoder，GPU 完成过滤、排序、逆映射、Polygon IoU 与 Rotated NMS，仅回传保留框 |
| Pose | 支持 | 前向 CUDA + CUDA Decoder，GPU 完成筛选、关键点解码与人体框 NMS，仅回传保留结果 |
| RT-DETR | 支持 | 前向 CUDA + CUDA Decoder |
| Classification | 支持 | 前向 CUDA + CUDA Top-K，仅回传 `top_k * 8` 字节 |
| Semantic Segmentation | 支持 | 前向 CUDA + CUDA ArgMax，仅回传 `width * height * 4` 字节的类别图 |

Pose 的 `d2h_bytes` 只统计最终保留结果的紧凑 payload：每个保留人体 28 字节（框、分数、
类别、anchor），每个关键点 16 字节（x、y、置信度、是否在图像内）。OBB 每个保留框 32 字节
（框、分数、类别、anchor）。控制字与错误诊断数据（fault、候选计数、IoU 调用数）不计入，
与其它 CUDA Decoder 的口径一致。

CUDA EP 禁止 CPU fallback。如果 CUDA 初始化失败，Engine 返回错误，不自动切换到 CPU。

## 7. 已知不支持的模型结构

1. **多输入模型**：当前 Recognizer 仅接受单输入模型。
2. **多输出模型（非标准）**：当前 Recognizer 仅接受特定输出结构。
3. **带内置 NMS 的模型**：YOLOv8 导出的带 NMS 的模型不被支持。Recognizer 会检测到 `NonMaxSuppression` 节点并拒绝识别。
4. **INT8 量化模型**：当前 Backend 仅支持 FP32 输入。
5. **FP16 模型**：当前 Backend 仅支持 FP32。
6. **非 YOLOv8/RT-DETR/Classification 的检测模型**：当前没有通用的检测 Decoder。

## 8. 常见错误

### 8.1 Analyzer 失败

```
error: cannot open file: model.onnx
```

模型文件不存在或无法读取。

### 8.2 Recognizer Unknown

```
Diagnostic: unknown
```

没有 Recognizer 接受该模型图结构。可能原因：
- 模型不是 YOLOv8、RT-DETR 或常见分类网络
- 模型图结构不符合任何已注册 Recognizer 的证据规则

### 8.3 Recognizer Ambiguous

```
Diagnostic: ambiguous
```

多个 Recognizer 同时接受该模型。这通常意味着模型结构不典型，需要人工检查。

### 8.4 Manifest 验证失败

```
Unresolved Critical:
  - input.color: input color is unresolved
  - input.normalization: input normalization is unresolved
  - input.resize: input resize is unresolved
```

生成的 Manifest 缺少关键语义字段。这些字段无法从 ONNX 图结构推断，需要用户手动提供。

### 8.5 Backend 不支持

```
error: no backend is registered for model format: onnx
```

当前构建未启用 ONNX Runtime。需要使用 `-DVIE_ENABLE_ONNXRUNTIME=ON` 重新构建。

### 8.6 CUDA 不可用

```
error: CUDA backend is not available in this build
```

当前构建未启用 CUDA。需要使用 `-DVIE_ENABLE_CUDA=ON` 重新构建，并确保 CUDA Toolkit 已安装。

## 9. `vie_model_inspect` 使用方法

### 9.1 基础检查

```bash
vie_model_inspect model.onnx
```

输出人类可读的检查报告，包括 Analyzer、Recognizer、Manifest 和 Execution 状态。

### 9.2 JSON 报告

```bash
vie_model_inspect model.onnx --json
vie_model_inspect model.onnx --report report.json
```

输出 JSON 格式的检查报告，方便其他工具解析。

### 9.3 验证用户 Manifest

```bash
vie_model_inspect model.onnx --manifest model.json
```

加载用户提供的 Manifest JSON 并验证其有效性。

### 9.4 测试 CPU 执行

```bash
vie_model_inspect model.onnx --cpu
```

使用 CPU EP 执行一次推理（需要完整的 Manifest，包括 color、normalization、resize）。

### 9.5 测试 CUDA 执行

```bash
vie_model_inspect model.onnx --cuda
```

使用 CUDA EP 执行一次推理（需要完整的 Manifest 和 CUDA 构建）。

## 10. 新模型适配流程

1. **导出 ONNX**：使用框架原生工具导出模型为 ONNX 格式。
2. **运行 `vie_model_inspect`**：检查 Analyzer 和 Recognizer 是否能正确解析。
3. **检查 Manifest**：确认生成的 Manifest 是否包含所有必要字段。
4. **补充语义字段**：如果 color、normalization、resize 为 unresolved，手动填写。
5. **测试执行**：使用 `--cpu` 和 `--cuda` 测试推理是否正常。
6. **编写 Recognizer**（可选）：如果模型不被任何 Recognizer 识别，需要编写新的 Recognizer。
7. **编写 Decoder**（可选）：如果模型的输出协议不被任何 Decoder 支持，需要编写新的 Decoder。
