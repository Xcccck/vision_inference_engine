# RT-DETR ONNX 结构与输出协议调研

本文件只覆盖这一份实际导出的模型。它不代表其他 RT-DETR 实现、其他权重或其他导出选项。VIE 当前没有接入 RT-DETR 推理。

结论：**PASS WITH CHANGES**。现有 Analyzer、Backend、`DetectionBatch`、`GeometryTransform` 和 `IModelDecoder::decode_batch` 可以复用。要接入这一份模型，需要新增识别器和后处理，并让检测配置接受归一化 `cxcywh`。Manifest schema 1.0 不必为这一份导出增加字段。

## 1. 模型来源与版本

| 项 | 值 |
|---|---|
| 权重 | Ultralytics assets v8.4.0 `rtdetr-l.pt` |
| 权重 URL | `https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt` |
| 权重 SHA256 | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` |
| 权重大小 | 66511432 字节 |
| ONNX | `testdata/onnx/rtdetr-l.onnx` |
| ONNX SHA256 | `91e7cb0358575e37d7d241db616af07fcce4924ca0b3598ac7e35f2ea8ff719f` |
| ONNX 大小 | 128797889 字节 |
| Ultralytics | 8.4.0 |
| PyTorch | 2.14.1+cu130，导出设备 CPU |
| ONNX | 1.23.2 |
| Opset | 17 |
| 导出参数 | `imgsz=640`，`batch=1`，`dynamic=False`，`simplify=False`，`opset=17`，`nms=False`，`device=cpu` |

仓库里原来没有 RT-DETR 文件。导出脚本是 `tools/export_rtdetr_l.py`。Ultralytics 对 `RTDETRDecoder` 会强制 `nms=False`，导出日志也写明输出 shape 为 `(1, 300, 84)`。ONNX metadata 中的 `args` 与上表一致，另外还有 `task=detect`、`stride=32`、`imgsz=[640, 640]`、`batch=1` 和 80 个 COCO 类名。这些 metadata 不是图结构证明。

导出第一次失败是因为系统 SciPy 与 NumPy 2.2.6 不兼容，`RTDETRDecoder.forward` 会导入 `scipy.optimize`。安装 SciPy 1.15.3 后导出成功。这只影响调研用的 Python，没有链进 VIE。

## 2. ONNX 基础信息

现有 `vie_analyze_onnx` 退出码 0，warnings 为空。

| 项 | Analyzer 结果 |
|---|---|
| IR | 8 |
| producer | pytorch 2.14.1 |
| graph | `main_graph` |
| opset | `ai.onnx` 17，只有这一个 domain |
| runtime input | 1 个，`images`，float，declared `[1, 3, 640, 640]` |
| graph output | 1 个，`output0`，float |
| output declared shape | 三个符号维：`Concatoutput0_dim_0/1/2` |
| value_info | 无 |
| initializer | 446，全部 raw，无 external data |
| sparse initializer | 0 |
| function | 0 |
| node | 2835。算子计数之和等于 ONNX 的 node 数 |

算子计数里和后处理有关的事实：

| 算子 | 数量 |
|---|---|
| Conv | 110 |
| MatMul | 111 |
| Gemm | 7 |
| Add | 167 |
| Mul | 95 |
| Sigmoid | 32 |
| Softmax | 13 |
| LayerNormalization | 21 |
| Reshape | 152 |
| Transpose | 104 |
| Gather | 123 |
| TopK | 1 |
| Concat | 169 |
| GridSample | 18 |
| NonMaxSuppression | 0 |

Analyzer 能读完整份 protobuf，也能列出全部节点和算子。它没有把最终输出 shape 推导出来。`propagated_shapes` 只有 6 个 Identity 权重 shape，没有 `output0`，也没有 shape conflict warning。Manifest 生成器因此把输出写成 `[-1, -1, -1]`。Validator 对动态输出要求具体 profile，所以这份自动生成的 Manifest 不能直接加载。

静态 `[1, 300, 84]` 来自两处独立运行结果：Ultralytics 导出日志，以及下面的 PyTorch 和 VIE ONNX Runtime 执行。它不在 ONNX 的 output type 里。

## 3. 输出节点

以下是图中直接连到 `output0` 的节点，不是根据名字猜测的。

```text
output0
  Concat axis=-1
    Squeeze(decoder box)
      Sigmoid
        Add
          dec_bbox_head.5 最后一层 Linear，bias shape [4]
          Log(Div(Clip, Clip))
    Sigmoid
      Squeeze(decoder class)
        dec_score_head.5 Linear，bias shape [80]，权重 [256, 80]
```

图里唯一的 TopK 是 `/model.28/TopK`。`k` 是常量 `[300]`，axis 1，largest 1，sorted 1。它的输入是 `ReduceMax`，位于 decoder head 之前，对应源码里的 encoder query selection。它不是对最终 300 个 query 再做一次筛选。

分类分支到达 `output0` 之前的最后一个激活是 Sigmoid。框分支到达 `output0` 之前的最后一个激活也是 Sigmoid。这条末端路径上没有 Softmax，也没有 NonMaxSuppression。图中另有 13 个 Softmax，位于注意力路径，不直接充当类别归一化。

## 4. 输出语义

证据分成四档。

### 图和 initializer 可以直接看到

- 一个 float 输出，末端是 axis 为 -1 的 Concat。
- 框侧最后一层 bias 长度是 4，类别侧 bias 长度是 80。
- 两侧在 Concat 之前都经过 Sigmoid。
- 没有 NonMaxSuppression。
- 图内有一次 `k=300` 的 TopK，位置在最终 Concat 之前。

### 一次真实执行确认

`bus.ppm` 为 P6，810×1080。PyTorch 与 ORT 的原始输出都是 float `[1, 300, 84]`。

| 切片 | min | max | mean |
|---|---|---|---|
| 前 4 维，框 | 0.007047 | 0.992436 | 0.264604 |
| 后 80 维，分数 | 0.000168 | 0.955342 | 0.004022 |

框和分数都落在 `(0, 1)`，没有负值，也没有大于 1 的值。对已经是概率的分数再做一次 Sigmoid，最大绝对变化是 0.499874。因此运行时不能再激活一次。

### 源码和官方后处理确认

Ultralytics 8.4.0 `RTDETRDecoder.forward` 在推理时执行：

```text
y = concat(dec_bboxes, sigmoid(dec_scores))
```

`dec_bboxes` 已是 `sigmoid(delta + inverse_sigmoid(reference))`，单位是模型输入上的归一化 cxcywh。`dec_scores` 在进 ONNX 之前是 logits，ONNX 里对应的 Sigmoid 在 Concat 之前。80 维里没有额外的 no-object 类。每个 query 取 80 个分数的最大值和对应类别，一对 query 只产生一个类别。

`RTDETRPredictor.postprocess`：

1. 把 cxcywh 转成 xyxy。
2. 按置信度阈值过滤。预测默认阈值是 0.25。
3. 按分数排序，截断到 `max_det=300`。
4. 用原图宽高分别乘 x 和 y。
5. 不执行 NMS，不去掉某个背景类，也不再做 TopK。图内的 TopK 已经选出 300 个 query。

`pre_transform` 使用 `LetterBox(scale_fill=True)`，也就是拉伸到 640×640，不保留长宽比，不补边。随后 BGR 转 RGB，除以 255。这和 YOLOv8 的 letterbox 不同。

### 这一张图上的重复框

置信度大于 0.25 的 9 个框，两两最大 IoU 是 0.153。按官方 0.45 的 IoU 阈值，NMS 不会删掉其中任何一个。这只说明这张图没有需要被 NMS 去掉的重复框。官方代码本身不调用 NMS。

## 5. PyTorch 与 ONNX 数值

同一张 `bus.ppm`，同一份 Ultralytics 预处理 tensor。

| 对比 | 结果 |
|---|---|
| PyTorch 原始输出 vs VIE `OnnxRuntimeBackend` CPU | 最大绝对差 `1.679e-5`，平均 `4.755e-8`。`allclose(rtol=1e-4, atol=1e-4)` 成立，`1e-5` 不成立 |
| 按官方规则得到的 9 个框 | 类别完全一致。框的最大绝对差 0.000122 像素，置信度最大绝对差 `5.66e-7` |
| 手写后处理 vs `model.predict` | 9 个框，类别一致。框的最大绝对差 0.000183 像素，置信度最大绝对差 `1.13e-6` |

ORT 使用 `/opt/onnxruntime-linux-x64-1.20.1` 的 CPU EP。请求的输出 profile 是 `[1, 300, 84]`，实际返回也是这个 shape。Backend 没有因为 ONNX type 里的符号维而失败。

### 预处理差异

官方拉伸使用 OpenCV `INTER_LINEAR`。VIE 已有 `ResizeMode::Stretch`、bilinear、RGB、`scale=1/255`、mean 0、std 1。对这张图，VIE 输出与官方输入的最大绝对差是 0.002937，平均 0.001003。1228800 个元素里没有任何一个差超过 `1/255`，有 140995 个差超过 `0.5/255`。两者不是逐值相同。

用 VIE stretch 再跑同一个 ONNX，置信度大于 0.25 仍然是同样的 9 个类别。按类别和 IoU 配对后，角点最大差 0.409 像素，最小 IoU 0.992983。原始 `[1, 300, 84]` 不能逐元素比较，最大差 0.972。原因是不足 1 个灰度级的采样差改变了图内 TopK 选出的 query，最终保留框仍然对应同一组目标。

VIE letterbox 对这张图是 scale 0.592593、左侧 padding 80。这是 YOLOv8 路径使用的几何。用它跑 RT-DETR 后只剩 7 个框，少了 class 10 和一个 class 27。官方还原直接乘原图宽高，只在拉伸、无 padding 时成立。`GeometryTransform` 的公式是 `(x_model - pad) / scale`。把归一化坐标先乘 640，再交给这个公式，在 pad 为 0 时等于乘回原图宽高。letterbox 不能这样替代。

## 6. 和现有 YOLOv8 后处理的差异

| 项 | 这份 RT-DETR | 现有 YOLOv8 Detection |
|---|---|---|
| 输出 | `[1, 300, 84]`，query 在中间、通道在最后 | `[1, 84, 8400]`，通道在中间 |
| 框 | 归一化 cxcywh，图内已经 Sigmoid | 模型输入像素 cxcywh |
| 分数 | 80 维 Sigmoid 概率，无 objectness | 80 维 Sigmoid 概率，无 objectness |
| 筛选 | 图内 TopK 到 300，运行时只做阈值、排序和 `max_det` | 阈值、TopK 和 NMS |
| NMS | 这份导出没有，官方后处理也不做 | 默认需要 |
| 几何 | 拉伸，无 padding | letterbox |
| 每 query 类别 | 只取最大类 | `best_class` 或 `all_above_threshold` |

现有 `read_channel` 在 `channel_axis != 1` 时按 `[prediction, channel]` 读取，内存布局可以覆盖 `[300, 84]`。现有 `cxcywh_to_xyxy`、阈值、最大类、`nms_enabled=false` 时的截断，以及 `map_box_to_original` 都能用。缺的一步是：在映射前把归一化坐标乘上模型输入的宽和高。现有 `decoded_cxcywh` 把坐标当成模型输入像素，直接套用会把框缩小约 640 倍。

## 7. 模块兼容性

| 模块 | 可以复用 | 需要修改 | 原因 |
|---|---|---|---|
| ONNX Analyzer | 解析可以 | shape propagation 以后才需要补 | protobuf、节点和算子完整。最终输出 shape 停在符号维 |
| Recognizer | 注册机制可以 | 需要新识别器 | 当前结果是 `unknown`，候选为空。不能把含 Transformer 的图都当成 RT-DETR |
| Manifest schema 1.0 | 可以 | 这一份模型不需要新字段 | 协议名、`cxcywh_normalized`、概率分数、类别数、NMS 开关都已有位置 |
| Validator | 可以 | 不改 | 动态输出继续视为不可直接运行。具体 shape 由识别证明或用户 profile 填入 |
| `decoder_config_from_manifest` | 大部分可以 | 需要接受归一化框 | 现在要求 `box_decode=decoded_cxcywh` 且 `box_format` 必须等于 `cxcywh`，归一化格式会被拒绝 |
| `IModelDecoder` | 可以 | 不改接口 | `decode_batch` 已经返回 `DetectionBatch` |
| `DetectionBatch` / `Detection` | 可以 | 不改 | 框、类别、置信度足够 |
| `GeometryTransform` | 可以 | 不改 | Stretch 的 scale 和 pad 正确。Decoder 必须先把归一化框变成模型输入像素 |
| Preprocess | 可以 | 不改算法 | 使用已有 Stretch。不要为了这张图改 bilinear 实现。Letterbox 会改变检测结果 |
| `InferenceEngine` | 调用链可以 | 只注册新 Decoder | `infer()` 继续走 `decode_batch`。不要加新的任务类型 |
| CPU Backend | 已验证可以 | 不改 | 同一 tensor 上与 PyTorch 对齐 |
| CUDA Backend | 接口可以 | 本次未跑 | Backend 只接收命名 Tensor，不含 YOLO 语义。这份图能否被 CUDA EP 完整执行尚未验证 |

## 8. Manifest

这一份导出可以用现有字段描述：

| 字段 | 建议值 | 依据 |
|---|---|---|
| `task` | detection | 图只有检测输出。metadata `task=detect` 只作旁证 |
| `model_format` | onnx | 文件解析 |
| `num_classes` | 80 | `dec_score_head.5.bias` 的长度 |
| `outputs[0].name` | `output0` | 图 |
| `outputs[0].shape` | `[1, 300, 84]` | 执行结果。type proto 本身仍是符号维，自动生成不能假装已经观察到 |
| `decoder.protocol` | 新协议名，例如 `rtdetr_ultralytics_raw` | 不能复用 `yolov8_detect_raw` |
| `decoder.box_format` | `cxcywh_normalized` | 末端 Sigmoid 后的 cxcywh，范围在 0 到 1 |
| `decoder.box_decode` | 需要与归一化格式兼容的现有或扩展值 | 当前构建函数不接受 `cxcywh_normalized` |
| `decoder.box_channels` | 4 | bias 长度 |
| `decoder.channel_axis` | 2 | `[batch, query, channel]` |
| `decoder.scores_are_probabilities` | true | Concat 前已经 Sigmoid |
| `decoder.class_selection` | `best_class` | 官方每个 query 只取最大类 |
| `nms.enabled` | false | 图中没有 NMS，官方后处理也不做 |
| `nms.confidence_threshold` / `max_detections` | 运行配置 | 官方默认 0.25 和 300，不是模型固有值 |

不需要新的 no-object 字段：类别维就是 80，没有第 81 个背景 logit。query 数可以由具体输出 shape 表示，不必另加 `num_queries`。如果以后遇到带背景类或输出 shape 无法唯一确定的 DETR，再考虑最小字段，不要现在加。

类名可以在图上的类别数等于 metadata `names` 数量时填入。数量不一致就保持未解析。文件名不参与识别。

## 9. 识别规则

只读 `ModelAnalysisReport`。建议同时满足才接受，任一结构对不上就返回 unknown：

1. 恰好一个 float runtime 输入，rank 4，通道轴是静态 3。
2. 恰好一个 float 输出，rank 3。
3. 输出生产者是 Concat，两个输入分别来自框 Sigmoid 和类别 Sigmoid。
4. 框分支能追到长度为 4 的线性层，并且在 Sigmoid 前有残差式 refinement。
5. 类别分支能追到长度为 `C` 的线性层，`C` 不是由文件名或 metadata 决定。
6. 图中没有 NonMaxSuppression。
7. 存在且仅存在一个用于 query selection 的 TopK。`k` 可以解释 query 数，但只有能证明它决定输出的 query 轴时才写入 shape。
8. 不把 GridSample、LayerNormalization 或 Softmax 的存在本身当作 RT-DETR。

metadata 的 `task`、`names`、`imgsz` 只能在图结构已经成立后补充类名或检查一致性。

## 10. CPU Decoder

新的 `RtdetrDecoder` 实现 `decode_batch`，返回现有 `DetectionBatch`。对这一份模型，步骤是：

1. 按名称取 `output0`。不需要再做 Sigmoid 或 Softmax。
2. 每个 query 读取 4 个归一化 cxcywh 和 80 个概率。
3. 乘上模型输入宽高，转成 xyxy。
4. 取最大类，按 `confidence_threshold` 过滤。
5. 按分数排序，截断到 `max_detections`。图内已经 TopK，运行时不再做 8400 规模的 TopK。
6. `nms_enabled` 保持 false。不要调用 YOLO 的 greedy NMS，除非以后的导出真的把 NMS 留在图外并在协议里声明。
7. 调用现有 `map_box_to_original`。

`cxcywh_to_xyxy`、`map_box_to_original` 和 `DetectionBatch` 可以直接调用。不要调用 `decode_yolov8_cpu_predictions`，因为它把 `decoded_cxcywh` 定义为模型输入像素。

## 11. CUDA Decoder

本次没有跑 CUDA EP，也没有写 CUDA kernel。

`output0` 是 `300 * 84 = 100800` 个 FP32 元素，`100800 * 4 = 403200` 字节。这里原先写成 `300 * 84 * 4 = 100800` 字节，把元素数当成了字节数。第一版可以在 CUDA EP 返回设备 Tensor 后，把这一份输出 D2H，再走同一个 CPU Decoder。这和 YOLO 避免搬运整份 `[1, 84, 8400]` 的原因不同，这一份输出本身就是筛选后的 query。403200 字节仍然很小。

现有 YOLO CUDA kernel 假定通道在前、坐标是像素，并且包含 NMS。不能为了复用那些 kernel 把 RT-DETR 重排成 YOLOv8 布局。如果以后要减少这次 D2H，只需要一个很小的 query kernel 加现有的设备排序。300 个元素不足以成为第一版接入的阻塞项。

CUDA EP 是否接受这张含 GridSample 的图，标记为**待验证**。

## 12. 最小接入

不改这些接口和实现：

- `IModelDecoder` 的虚函数
- `DetectionBatch`、`Detection`、`BoundingBox`
- `GeometryTransform` 和 Preprocess 算法
- Tensor、DeviceContext
- ONNX Runtime CPU Backend 的执行策略
- YOLOv8 Detection 和 Segmentation Decoder
- Manifest schema 1.0 的 JSON 结构

以后的代码改动限制在：

1. 新增只读 `ModelAnalysisReport` 的 RT-DETR 识别器，并在两个 YOLO 识别器都不匹配时才参与。两个识别器同时匹配则保持 Ambiguous。
2. 新增一个 Decoder 协议。CPU 上完成上面的 7 步。
3. `InferenceEngine` 构造函数注册这个工厂。
4. `decoder_config_from_manifest` 接受 `cxcywh_normalized`，并告诉 Decoder 先乘模型输入尺寸。不要把归一化值塞进现有的像素 `decoded_cxcywh`。
5. 识别器只有在图证明 query 轴和 `4+C` 之后，才把输出 shape 写成具体值。证明不了就保持动态，加载继续失败。

## 13. 限制

- 结论只针对 Ultralytics 8.4.0、opset 17、静态 640、`nms=False`、未 simplify 的 `rtdetr-l`。
- ResNet 版 RT-DETR、动态 shape、End-to-End NMS 导出和其他实现没有测试。
- Analyzer 的 shape propagation 不能从这张图推出 `[1, 300, 84]`。
- VIE Stretch 与 OpenCV `INTER_LINEAR` 相差小于一个 8 位灰度级，但不是逐值相同。
- 只在 `bus.ppm` 上做了数值对比。
- CUDA EP 执行这张 ONNX 尚未验证。
- 官方后处理没有 NMS。这不能推广成所有 RT-DETR 图都不需要 NMS。

## 14. 实验命令

```text
python3 tools/export_rtdetr_l.py
sha256sum testdata/onnx/rtdetr-l.onnx
build/vie_analyze_onnx testdata/onnx/rtdetr-l.onnx
build/vie_recognize_onnx testdata/onnx/rtdetr-l.onnx
```

Analyzer 退出码 0，2835 个节点，无 warning，输出 declared shape 为三个符号维。Recognizer 打印 `recognition: unknown`。

PyTorch 与 ORT 对比使用 `/tmp/rtdetr_ort_probe.cpp`，链接 `build-ort/libvie.a` 和 ORT 1.20.1 CPU。同一官方输入上原始输出最大差 `1.679e-5`。VIE Stretch 的几何是 `scale_x=0.790123`、`scale_y=0.592593`、pad 0。配对后的 9 个框最小 IoU 0.992983。

## 15. 结论

现有架构能够扩展到这份非 YOLO 检测模型。Analyzer 和 CPU Backend 已经能处理这张 ONNX。识别器正确地拒绝了它，而不是把它认成 YOLOv8。接入时新增的是协议和后处理，不是另一套检测结果类型。

判定为 **PASS WITH CHANGES**。阻塞项不存在。未完成的是归一化框在现有检测配置入口被拒绝，以及输出 shape 仍是符号维。这两处都是局部扩展，不需要重做 Engine、Backend 或 Manifest schema。

## 16. Phase 5.2 实现记录

本节是接入记录。第 14 节里 `vie_recognize_onnx` 打印 `recognition: unknown` 是接入前的结果。接入后同一命令对这份 ONNX 打印 `recognition: recognized`，候选是 `rtdetr_detection`。

识别器只读 `ModelAnalysisReport`。它要求同时成立：唯一 float 输入的声明轴 1 为 3；唯一 float 输出的 rank 为 3，且由 `axis=-1` 的两输入 Concat 产生；图中没有 `NonMaxSuppression`；一个输入经 Sigmoid 接到 `Add(width-4 linear, Log(Div(...)))`；另一个输入经 Sigmoid 接到宽度为 C 的 linear；全图只有一个 `ai.onnx` TopK，其 `axis=1`，k 是 Constant 的标量 int64，运行时输入能到达它的数据输入，且它的某个输出能到达两条 linear 的数据输入。C 和 k 都从图上读取。这份模型得到 C=80、k=300。单独的 TopK(k=300) 或单独的 `[1,300,84]` 会返回 Unknown。

Manifest 仍是 schema 1.0。为了不把符号输出写成静态 shape，增加了可选字段 `decoder.num_queries`。未设置时 JSON 不出现这个键，已有 YOLOv8 Manifest 不变。`box_decode` 增加闭集取值 `decoded_normalized_cxcywh`，表示图内已经完成的归一化 cxcywh，不是模型输入像素。`decoder_config_from_manifest` 没有接受这个取值，所以现有 YOLOv8 解码入口不变。

这份模型生成的 Manifest：

- task `detection`，family `rtdetr`，protocol `rtdetr_detection_raw`
- 输入 `images`，`[1,3,640,640]`，fp32，layout `nchw`，resize `stretch`
- 输出 `output0`，角色 `predictions`，shape `[-1,-1,-1]`，provenance 是 observed 的声明 shape
- `num_classes=80`，`num_queries=300`，都是 inferred
- box `cxcywh_normalized` / `decoded_normalized_cxcywh`，4 通道，channel axis 2
- 分类分数是概率，NMS `enabled=false`，`pre_nms_top_k` 保持未设置
- 颜色、归一化和插值保持 unresolved

`rtdetr_detection_raw` 没有注册 Decoder。未改过的 Manifest 在 `load` 时因动态输出和未决预处理返回 `UnresolvedField`。测试把这些字段补成具体值后，有 ONNX Runtime 的构建返回 `Unsupported`，消息包含未注册的协议名。没有 ONNX Runtime 的构建在 Decoder 检查之前就因没有后端而 `Unsupported`。两种情况都没有创建可执行的 RT-DETR Decoder。

四种构建在接入后的测试数是 85、102、123、132，全部通过。相对接入前的 80、97、118、127，各增加 5 个 RT-DETR 测试，没有删除或跳过已有测试。

仍未实现 RT-DETR Decoder。CUDA EP 能否执行这张含 GridSample 的图仍然待验证。颜色和归一化没有写入 Manifest。输出的 batch 轴没有被 shape propagation 证明，所以没有写成静态 1。

## 17. Phase 5.3 实现记录

CPU Decoder 已接入。协议 `rtdetr_detection_raw` 注册在现有 `DecoderRegistry`，`InferenceEngine::infer()` 按 protocol 选择它，没有 `infer_rtdetr()`，也没有按 family 字符串分支。

自动生成的 Manifest 没有改。`output0` 仍是 observed 的 `[-1,-1,-1]`，颜色、归一化和插值仍是 unresolved。测试另做一份运行时 Manifest：这些字段和具体输出 shape `[1,300,84]` 使用 `user` provenance，证据写明这是调用方根据官方预处理和 ORT 实测提供的，不是 ONNX 声明 shape。现有 schema 用 certainty 区分这两份 shape，没有新增字段。

Decoder 从 Manifest 读取 query 数、类别数、输出名和坐标语义，不写死 300 或 80。每个 query 取最大有限概率，不执行 Sigmoid、Softmax、objectness 或 background。`cx,w` 乘模型输入宽度，`cy,h` 乘模型输入高度，再经 `cxcywh_to_xyxy` 和 `GeometryTransform`。NMS 关闭时不要求 IoU，也不读取 `pre_nms_top_k`。请求打开 NMS 会返回 Unsupported。`decoded_normalized_cxcywh` 不会进入 YOLOv8 的 DFL 分支。

`bus.ppm` 上，同一份 PyTorch 原始输出经本 Decoder 与 Ultralytics `model.predict` 对比：9 个框，类别一致，最小 IoU 0.999998，最大角点误差 0.000122 像素，confidence 差为 0。这是 Decoder 误差。

VIE Stretch 相对官方预处理的最大绝对差是 0.002937，平均 0.001003，没有元素超过 1/255。`scale_x=0.790123`，`scale_y=0.592593`，pad 为 0。公共 Preprocess 没有为这个差值改动。

完整 CPU `infer` 与官方 9 个框对比：类别和数量一致，最小 IoU 0.992981，最大角点误差 0.409 像素，最大 confidence 差 0.00704。这个差值来自预处理采样，不是第二次 Sigmoid。

Release、CPU 设备、100 次稳定运行：preprocess mean/p50/p95/min/max 30.71/29.31/40.19/25.71/41.43 ms；execute 290.39/289.66/318.63/263.39/324.65；decode 0.055/0.051/0.070/0.043/0.093；端到端 321.17/320.51/354.64/291.44/365.69。首次 load 280.60 ms，首次 execute 257.91 ms。没有 H2D、D2H 或额外同步。

四种构建 92、111、132、141 全部通过。YOLOv8 Detection 和 Segmentation 测试仍在其中。CUDA Decoder 和这张图的 CUDA EP 仍未实现。

## 18. Phase 5.4.1 严格 CUDA EP 验证

判定：**PASS WITH CHANGES**。`testdata/onnx/rtdetr-l.onnx` 能在现有 `OnnxRuntimeCudaBackend` 上、关闭 CPU EP fallback 后完整执行。917 个实际执行的 kernel 全部落在 CUDA EP。原始输出不是逐元素接近 CPU EP，低分 query 的最大绝对差是 0.906。置信度不低于 0.25 的 9 个检测框类别一致，最小 IoU 0.998558。本阶段没有改 Backend、Decoder、Recognizer、Manifest、Tensor、DeviceContext 或 Preprocess，也没有实现 CUDA Decoder。验证程序在仓库外，链接的是已有 Release `build-ort-gpu/libvie.a`。

环境：RTX 4080 Laptop，驱动 566.36，12282 MiB，计算能力 8.9。`cudaRuntimeGetVersion` 为 12060，`nvcc` 为 12.6.85，`cudaDriverGetVersion` 为 12070。cuDNN 头文件 9.6.0。ONNX Runtime GPU 包 1.20.1，库文件 `libonnxruntime.so.1.20.1`。CPU EP 对比使用的是这个 GPU 包里的 CPU EP，不是 Phase 5.3 的 CPU 包计时。模型 opset 17，输入 `[1,3,640,640]`，运行输出 `[1,300,84]` fp32。输入是 `bus.ppm` 经现有 Stretch、双线性、RGB、除以 255 得到的张量。CUDA 侧是这份张量的显式 H2D，回拷最大绝对差为 0。

### Session 与执行

`set_cpu_ep_fallback(false)`。Session 日志里的配置是 `session.disable_cpu_ep_fallback: 1`。Session 创建成功，load 336.33 ms（计时用的 severity 2）。模型执行成功。输出在 `cuda:0`，shape `[1,300,84]`，dtype fp32。`cudaPointerGetAttributes` 的类型是 device memory。返回生命周期是 `Copied`。Session 释放后，输出张量第一个元素仍是 0.912687，差为 0。

输入必须已经是这块 GPU 上的连续 CUDA Tensor，Backend 借用它的 device 指针创建 ORT Tensor，不接受 CPU 输入，因此没有隐式 H2D。输出先由 ORT 分配在 CUDA 上，再 D2D 进新的 VIE CUDA Tensor，然后释放 ORT 的 value。比较用的 D2H 是验证程序在 Backend 之外做的。Backend 内部没有把模型输出转到 Host。`SynchronizeInputs`、`SynchronizeOutputs` 和 `cudaDeviceSynchronize` 是默认流等待。Backend 没有使用 `DeviceContext` 的 stream，`do_copy_in_default_stream` 为 1。

现有 Backend 没有查询每个节点 Provider 的接口。severity 1 的 Session 日志只有 67129 字节，低于 2 MiB 上限，但里面没有逐节点分配。唯一出现 `CPUExecutionProvider` 的一行是 `TransposeOptimizer_CPUExecutionProvider modified: 0`，这是图变换器的名字，不是节点被放到 CPU。`MemcpyTransformer modified: 0`。日志不能单独证明整图在 CUDA 上执行。

节点分配来自另一个 ORT Session。它使用相同的严格选项：禁止 CPU fallback、只注册 CUDA EP、`ORT_ENABLE_ALL`、intra 4、inter 1、device 0、`do_copy_in_default_stream=1`，输入绑定同一块 VIE CUDA 内存，并打开 Profiling 和 optimized model 导出。Chrome trace 有 917 个 `*_kernel_time` 事件，`args.provider` 全部是 `CUDAExecutionProvider`。另外 1834 个事件是 `fence_before` 和 `fence_after`，没有 provider 字段。优化后的图有 917 个节点，与 kernel 事件数量一致，warnings 为 0。

GridSample 在优化图里仍是 18 个，18 个 kernel 都在 CUDA EP。原图 21 个 LayerNormalization：优化后剩 1 个 LayerNormalization，另外 20 个融合成 SkipLayerNormalization。这 1 个和 20 个都在 CUDA EP。没有 `MemcpyFromHost`、`MemcpyToHost`、`NonMaxSuppression`，也没有 CPU EP kernel。

917 个 kernel 的算子数量：

| 算子 | 数量 | 算子 | 数量 |
|---|---:|---|---:|
| Reshape | 141 | Add | 127 |
| Conv | 110 | MatMul | 83 |
| Relu | 79 | Transpose | 73 |
| Unsqueeze | 38 | FusedMatMul | 26 |
| Mul | 25 | QuickGelu | 24 |
| Concat | 21 | SkipLayerNormalization | 20 |
| Squeeze | 20 | Split | 18 |
| GridSample | 18 | Clip | 18 |
| Softmax | 13 | Div | 12 |
| Sub | 12 | Sigmoid | 8 |
| Gemm | 7 | ReduceSum | 6 |
| Log | 6 | Pad | 2 |
| Resize | 2 | Gather | 2 |
| MaxPool | 1 | BiasGelu | 1 |
| LayerNormalization | 1 | ReduceMax | 1 |
| TopK | 1 | Flatten | 1 |

### 数值

两边都是 `[1,300,84]` fp32，NaN 和 Inf 都是 0。框的范围 CPU 是 0.007030 到 0.992442，CUDA 是 0.007027 到 0.992442。分类概率 CPU 是 0.000168 到 0.955230，CUDA 是 0.000168 到 0.955240。

按 query 下标对齐：最大绝对差 0.905968，平均绝对差 0.002993，中位数 9.30e-6，P95 0.004606，P99 0.04498。84.75% 的元素小于 1e-3，97.37% 小于 1e-2。以 `max(|cpu|, 1e-6)` 为分母的最大相对差是 90.63，这个值被接近 0 的分数放大。只看 `|cpu| > 0.01` 的 2581 个元素，最大相对差 36.74，平均 0.304。68 个 query 的最大类别不一致；其中 66 个两侧最大分数都低于 0.25。

这 68 个不一致不是后处理造成的。输入回拷差为 0。Session 释放前后的拷贝差为 0。两边用同一个未修改的 CPU `RtdetrDecoder` 解码。差值来自 CUDA EP 与 CPU EP 的算子数值，并经过 TopK 变成 query 顺序差异。

下标 21 和 22 发生了相邻交换。CPU[22] 与 CUDA[21] 是同一个框：类别 27，分数 0.266430 对 0.266086，归一化框 L1 为 3.90e-5。CPU[21] 与 CUDA[22] 是另一个低于阈值的框：类别 0，分数约 0.145。按同一下标比较时，这一对的框 L1 是 0.362，类别看起来不一致。按框匹配后，它是同一次检测。

CPU 上分数不低于 0.25 的 9 个 query，各自都能在 CUDA 输出里找到框 L1 不超过 1.09e-4 的对应 query，类别一致，分数差最大 3.44e-4。其中 8 个下标相同，剩下的就是上面的 22 到 21。最大绝对差 0.906 落在两侧分数都低于 0.25 的 query 上。75 个 query 的框 L1 大于 0.1，也都在这个低分区域。

同一 Decoder、同一 `GeometryTransform`、置信度 0.25、最多 300、不开启 NMS：两边都是 9 个框，类别 5、0、0、0、0、10、9、27、27，没有类别不一致。最小 IoU 0.998558，最大角点差 0.054 像素，最大 confidence 差 0.000344。检测框一致不能代替上面的原始输出差。

### 性能

Release，intra 4，inter 1，`ORT_ENABLE_ALL`，同一输入。首次执行不计入 100 次。百分位与 Phase 5.3 相同：排序后 P50 取下标 `n/2`，P95 取 `floor(0.95*n)`。单位是毫秒。

| 项 | mean | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| CPU EP execute | 494.48 | 489.89 | 550.31 | 445.33 | 573.11 |
| CUDA Backend 端到端 | 19.54 | 17.56 | 26.34 | 15.24 | 55.53 |
| CUDA ORT Run | 19.20 | 17.23 | 25.80 | 14.93 | 55.23 |
| CUDA 输出同步 | 0.0205 | 0.0225 | 0.0261 | 0.0103 | 0.0343 |
| CUDA 输入同步 | 0.0329 | 0.0287 | 0.0680 | 0.0096 | 0.1416 |
| CUDA D2D | 0.0216 | 0.0210 | 0.0271 | 0.0157 | 0.0536 |
| 显式 H2D | 0.443 | 0.427 | 0.553 | 0.370 | 0.781 |
| 显式 D2H | 0.113 | 0.076 | 0.440 | 0.050 | 0.604 |

Session load：CPU EP 371.75 ms，计时用的 CUDA Session 336.33 ms。首次 CPU execute 506.79 ms。首次 CUDA 端到端 578.77 ms，其中 ORT Run 577.17 ms，之后稳定在约 19 ms；首次包含 CUDA 启动，不是 fallback。H2D 和 D2H 在 Backend 之外，因为调用方交出的是 CUDA Tensor，Backend 交回的也是 CUDA Tensor。端到端减去 Run、两次同步和 D2D 后大约 0.26 ms，是绑定和输出分配，100 次循环没有单独计时。模型计算是 ORT Run。传输和同步相对它大约小两个数量级。

### 未完成与后续条件

没有缺失的 CUDA kernel，也没有因为动态 shape 或算子属性导致的分配失败。不需要为了让这张图跑起来而改 ONNX 或 ORT 环境。原始输出的低分 query 偏差仍然存在，开发 CUDA Decoder 时要单独验收，不能只看最终 9 个框。

仍未做的是 RT-DETR CUDA Decoder。自动生成的 Manifest 仍然不能直接加载。颜色、归一化和插值仍由调用方给出。公共双线性与 cv2 的采样差没有在本阶段重测。Backend 仍然不能直接列出节点 Provider，本记录的分配证据来自同选项的 Profiling Session。本阶段没有改产品源码，因此没有重跑四套 ctest。

具备开发 CUDA Decoder 的条件：严格 CUDA EP 能执行整张图，输出已经是生命周期独立的 VIE CUDA Tensor。数值偏差是可控的 EP 差异，不是执行失败。本阶段在此停止。

## 19. Phase 5.4.3 Decoder 契约收尾

Phase 5.4.2 已经接入 RT-DETR CUDA Decoder。本节只记录契约收尾，没有新模型、没有 Kernel 优化，也没有改 ORT Backend、Recognizer 或 Manifest schema。

`Detection.query_index` 是模型输出里的 Query 下标，不是排序后的数组位置。CPU Decoder 和 CUDA Decoder 都写这个下标。同一份 Tensor 上两者一致。YOLOv8 Detection 仍写 -1，排序时间和 D2H 字节数也仍是 0。

非有限值的位置用一个整数记录：`(query << 3) | field`。`field` 占低 3 位，顺序与 CPU 相同：cx、cy、width、height、class score。一次 `atomicMin` 写入整个整数。不会用两次 `atomicMin` 分别记 Query 和字段，因此不会把较早 Query 配上较晚 Query 的字段。Host 解码后返回 `InvalidArgument`。消息形如 `class score is not finite at query 17`。256 个 Query、多个 block 同时出现 NaN 或 Inf 时，重复 32 次，位置与 CPU Decoder 相同。同一 Query 的多个字段取检查顺序里的第一个。

`sort_ms` 和 `d2h_bytes` 留在 `DetectionResult` 上。它们描述这一张图的一次解码，和原有的 `gpu_execute_ms`、`d2h_ms` 同一范围，不挂在单个 `Detection` 上。CPU 和 YOLOv8 保持 0。RT-DETR CUDA 的 `d2h_bytes` 只计留下的检测，每条 28 字节。12 字节计数器（fault、passed、kept）是控制信息，不计入。`[1,300,84]` FP32 是 100800 个元素、403200 字节，仍留在 GPU。第 11 节原先把 `300 * 84 * 4` 写成 100800 字节，那个等式把元素数当成了字节数，已经改正。

四种构建 94、113、144、154 全部通过。相对 Phase 5.4.2 的 92、111、140、150，每个构建增加 2 个 CPU 契约测试；两个 CUDA 构建再增加 Query 索引对照和并发错误位置测试。YOLOv8 Detection、YOLOv8 Segmentation 和 RT-DETR CPU Decoder 仍在其中。CUDA OFF 的 `vie_tests` 不链接 `libcudart`；ORT CPU 构建只链接 CPU 包的 `libonnxruntime.so.1`。`compute-sanitizer --tool memcheck` 跑了 `RtdetrCudaDecoder.NonFiniteLocationIsOneEncodingAndStaysPut`，`ERROR SUMMARY: 0 errors`。100 次性能循环和 ORT 端到端没有放进 memcheck。

仍未改的限制：300 个 Query 时 CUDA Decoder 端到端仍慢于 CPU Decoder；Decoder 仍使用 legacy 默认 stream，并依赖 Backend 返回前已经同步；只支持 batch 1、FP32、连续 Tensor；非有限的 class score 不指出是哪一个类别；自动生成的 Manifest 仍然不能直接加载。本阶段在此停止。
