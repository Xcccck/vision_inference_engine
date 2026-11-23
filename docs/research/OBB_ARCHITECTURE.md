# OBB 架构

本文记录 Phase 7.1 的任务语义、YOLOv8-OBB 协议和实测结果。OBB 是任务类型。YOLOv8-OBB 是一种模型结构。`yolov8_obb_raw` 是这种结构导出后的输出协议。`ObbObject` 是 VIE 返回的任务结果。四者不能互相替代。

## 1. 接入前的状态

`TaskType::Obb` 已经存在，字符串是 `obb`。当时没有 OBB 结果、旋转 IoU、旋转 NMS、角度约定、Decoder、Recognizer，也没有旋转框的坐标映射。

检测 NMS 是水平框的 `greedy_nms`。`map_box_to_original` 把框转成 xyxy 再裁到图像内。各向异性缩放下，旋转矩形的四条边不再保持垂直，那条路径不能复用。可以复用的只有与几何无关的排序：分数降序，分数相同则按下标升序。

`GeometryTransform` 的逆映射是 `x_original = (x_model - origin_x) / scale_x`，`origin = pad - crop`。Letterbox 取整后 `scale_x` 和 `scale_y` 可以差大约 `6e-4`。Stretch 的两个比例可以完全不同。把 width、height 分别除以两个比例并保留 angle，不能表示被拉成平行四边形的旋转矩形。

Manifest schema 1.0 已经能带可选对象，分类用的就是这种方式。识别器只看图结构。Analyzer 原先只留下整数常量；角度仿射 `(sigmoid - 0.25) * pi` 需要标量浮点，因此 Analyzer 增加了单元素 FP32/FP64 常量记录，不复制权重。

## 2. 模型来源

权重来自 Ultralytics 8.4.0 的官方发布：

`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-obb.pt`

| 项 | 值 |
|---|---|
| Ultralytics | 8.4.0 |
| torch | 2.14.1+cu130 |
| onnx | 1.23.2 |
| 导出 | `imgsz=1024, batch=1, dynamic=False, simplify=False, opset=17, nms=False, device=cpu` |
| pt 大小 | 6,567,590 字节 |
| pt SHA256 | `fa6e4cd2691f132875c143135affaa66b5d89394ebb1d07d19770a9b6382c1b8` |
| onnx 大小 | 12,632,341 字节 |
| onnx SHA256 | `cf9ab5cc1493cce32b141a1f8f9b3a1212469b2aa7f2b4df74780a4c8e49910f` |

权重和 ONNX 都在 `/tmp/vie_phase71`，不进入 Git。导出脚本是 `tools/export_yolov8n_obb.py`。它不导入 VIE。

`yolov8-obb.yaml` 的默认类别数是 80。这份 DOTA 检查点把它改成了 15。输出通道数是 20，即 `4 + 15 + 1`，与检查点一致。

## 3. ONNX 输入输出

IR 8，opset 17，domain `ai.onnx`。

| 端点 | 名称 | dtype | 声明 shape |
|---|---|---|---|
| 输入 | `images` | FP32 | `[1, 3, 1024, 1024]` |
| 输出 | `output0` | FP32 | `[1, 20, 21504]` |

最终节点是 axis 1 的 Concat，三个输入分别是框、类别 Sigmoid、角度。图中有 1 个 Cos、1 个 Sin、1 个 Softmax、73 个 Conv，没有 `NonMaxSuppression`。

21504 个锚点来自 1024 输入上的三个步长：`128^2 + 64^2 + 32^2`。

## 4. `yolov8_obb_raw`

图内已经完成 DFL、`dist2rbox` 和类别 Sigmoid。Decoder 不再做第二次 DFL 或 Sigmoid。

通道顺序，axis 1：

`cx, cy, width, height, class probabilities..., angle`

坐标是模型输入像素，不是归一化坐标，也不需要再乘 stride。没有 objectness。角度是最后一维，计算是 `(sigmoid(logits) - 0.25) * pi`。常量在图里是标量 0.25 和标量 `pi`。`OBB26` 直接输出未经过 Sigmoid 的角度，本识别器不接受。

batch 必须是 1。dtype 必须是连续的 CPU FP32。类别概率允许 `[-1e-4, 1+1e-4]` 的数值溢出，然后夹到 `[0, 1]`。NaN 和 Inf 拒绝。越过置信度的框如果宽或高不是正数，或者角度超出 `[-pi/4, 3pi/4]` 超过 `1e-3`，整次解码失败。低于阈值的锚点直接跳过，避免背景锚点中断整张图。

## 5. 角度

单位是弧度。宽边方向是 `(cos θ, sin θ)`，高边方向是 `(-sin θ, cos θ)`，与 Ultralytics 8.4.0 的 `xywhr2xyxyxyxy` 相同。图像坐标 +x 向右、+y 向下，所以正角度在画面上是顺时针。

模型输出范围是 `[-pi/4, 3pi/4]`。结果保留解码得到的 `(width, height, angle)`，不改写成另一种等价对。`(w, h, θ)` 与 `(h, w, θ + pi/2)` 的四个顶点相同。

## 6. 旋转 IoU

内部表示是 `cx, cy, width, height, angle`。四个顶点由这个表示算出，再排成逆时针。交集用凸多边形的 Sutherland–Hodgman 裁剪，面积用鞋带公式。IoU 是交集面积除以并集面积。交集被限制在两个框各自的面积之内，比值限制在 `[0, 1]`。

这不是水平外接矩形的 IoU，也不是 Ultralytics NMS 使用的 ProbIoU。非法输入（非有限值、非正宽高、非正面积）抛出 `InvalidArgument`。

## 7. 旋转 NMS

按类抑制。排序是分数降序，分数相同则 `anchor_index` 升序，再按 `class_id` 升序。置信度过滤之后取 `pre_nms_top_k`（1 到 4096），映射到原图并丢掉越界框，然后做 NMS。IoU 大于等于阈值才抑制。`iou_threshold` 必须在 `[0, 1]`，因此阈值 0 会抑制后面的每一个框。`max_detections` 至少为 1。非有限分数拒绝。

这些阈值来自当次 `ObbRuntimeConfig`，不写回 Manifest。

## 8. 几何映射

相对差 `|scale_x - scale_y| / max(|scale_x|, |scale_y|) <= 1e-3` 时视为各向同性，包括 letterbox 的取整误差。中心按各自的逆比例映射，宽和高除以两个比例的平均值，角度不变。

超过这个差时，四个顶点做逆映射，再拟合这个平行四边形的最小面积、边对齐矩形，并把角度折进 `[-pi/4, 3pi/4)`。面积相同则取绝对值更小的角度，再取更小的角度。拟合结果是矩形，不是平行四边形本身。

任一原始顶点或拟合后的顶点离开原图超过 `1e-3` 像素，这个框被丢掉。不裁剪，也不按可见碎片重拟合。裁剪窗口外但仍然落在原图内的框保留。

## 9. 识别器

识别器 id 是 `yolov8_obb`，架构 `yolov8_obb`，家族 `yolov8`，协议 `yolov8_obb_raw`。必须同时满足：

- 一个 FP32 输入，rank 4，axis 1 是 3。
- 一个 FP32 输出，rank 3，batch 静态为 1。
- 没有 `NonMaxSuppression`。
- 最终 Concat 恰好三条分支。
- 一条分支化简为 `(sigmoid - 0.25) * pi`，容差 `1e-4`，并且这个 Sigmoid 前面是三个输出通道为 1 的卷积。
- 另一条是三个卷积的 Sigmoid，共享的输出通道数是类别数。
- 框分支在卷积这个特征边界之上能到达 Cos 或 Sin，并且只有一个 DFL 卷积：Softmax 后接权重形状 `[1, reg_max, 1, 1]`。Softmax 的输入来自三个 `4 * reg_max` 通道的卷积。
- 输出的静态轴等于 `4 + 类别数 + 1`，并且这个轴是 1 或 2。

卷积是特征边界。Cos 和 Sin 必须出现在这条边界之上的解码里。继续走进骨干网络会把主干里的算子误当成旋转解码，所以回溯在 Conv 处停止。只靠输出维度、`(sigmoid - 0.5) * pi`、两条角度仿射，或者多一个输出，都返回 Unknown。与 Detection、Segmentation、RT-DETR 或 Classification 同时命中时是 Ambiguous。

真实的 `yolov8n-obb.onnx` 被识别为 15 类、channel axis 1。仓库里的 `testdata/onnx/yolov8n.onnx` 仍然是 `yolov8_detection`。

生成的 Manifest 不填写颜色、resize、归一化和 NMS。它 schema 合法，但还不能 `load`。预处理由调用者用 `user` provenance 补上。

## 10. Manifest

schema 仍是 1.0。可选对象 `obb` 的字段是 `output_name`、`channel_axis`（1 或 2）、`angle_unit`（`radian`）、`angle_range`（`[-pi/4,3pi/4]`）、`score_type`（`probabilities`）。框格式标记是 `cxcywh_angle`。置信度、IoU、Top-K 和是否按类抑制不属于协议。非 OBB 任务带上这些字段是 schema 错误。已有的 Detection、Segmentation、RT-DETR 和 Classification 文件不出现这个对象，含义不变。

## 11. CPU 与 CUDA 路径

CPU：CPU Tensor，ORT CPU EP，CPU Tensor，CPU Decoder，旋转 NMS。

严格 CUDA：CUDA Tensor，ORT CUDA EP，CUDA Tensor，显式把名为 `output0` 的整份输出拷回 CPU，然后是同一个 Decoder。没有静默 CPU fallback，没有 CUDA 旋转 NMS，也没有 CUDA OBB Decoder。`d2h_bytes` 只计这次显式拷贝，本次是 1,720,320 字节。ORT 内部搬移不计入。结果里的执行设备是 CPU，因为 NMS 在 CPU 上。缺少 CUDA Backend 时返回 Unsupported。

## 12. 独立参考

参考不是 Ultralytics 的 `non_max_suppression`。那条路径在 `rotated=True` 时使用 ProbIoU，和多边形 IoU 不必得到同一组框。

同一张 1024×1024 的确定性 RGB 图案分别送进：

- Ultralytics 8.4.0 的 PyTorch 前向，保存 Sigmoid 之后的原始张量。
- `tools/export_yolov8n_obb.py` 里单独写的多边形 IoU NMS。
- VIE CPU EP 的 `infer_obb`。

图案不是 DOTA 图片。最高类别概率是 `2.865880960598588e-4`。这次比较证明协议和数值，不证明检测精度。

CPU EP 的原始张量与 PyTorch 的最大绝对差小于测试门槛 `1e-3`。置信度 0、`pre_nms_top_k` 300、IoU 0.45、最多 20 个框时，NMS 后的 20 个框在类别、锚点、分数（`1e-4`）、中心和宽高（`1e-2` 像素）、角度（`1e-3` 弧度）上与独立 Python 参考一致。过滤后、映射前有 300 个候选；200 个留在原图内；NMS 留下 20 个。

## 13. CPU 与 CUDA 的一致性

同一图案上，CPU EP 与严格 CUDA EP 的原始张量：

| 通道 | 最大绝对差 |
|---|---|
| cx, cy, width, height | 0.634567 像素 |
| 15 个类别概率 | 1.19209e-6 |
| angle | 0.00279343 弧度 |

按锚点对齐后，两边 NMS 留下的 20 个框是同一组锚点。同一锚点的中心和宽高差不超过 0.05 像素，角度差不超过 0.005 弧度，分数差不超过 `1e-4`。位置顺序不要求相同，因为这些分数都在 `3e-4` 附近，Execution Provider 的误差会改变排序。CUDA 路径在映射后剩 199 个候选，CPU 路径剩 200 个；多出的那一个是数值误差把顶点推出原图后被丢掉的。最终 20 个框不受这个差别影响。

## 14. 性能

输入 1024×1024，warmup 3 次，采样 10 次。百分位是升序后的 `floor(fraction * (n - 1))`。单位是毫秒。

CPU EP（CPU 包 `/opt/onnxruntime-linux-x64-1.20.1`）：

| 阶段 | 平均 | P50 | P95 |
|---|---|---|---|
| preprocess | 59.9719 | 59.1782 | 62.347 |
| execute | 61.826 | 61.7089 | 65.0305 |
| d2h | 0 | 0 | 0 |
| validate | 4.93288 | 4.05999 | 6.71533 |
| rotated NMS | 4.3929 | 4.08548 | 5.5605 |
| total | 135.581 | 135.382 | 141.408 |

`d2h_bytes` 是 0。候选 200，NMS 后 20。输出 1,720,320 字节。

严格 CUDA EP（GPU 包 `/opt/onnxruntime-linux-x64-gpu-1.20.1`，禁止 CPU fallback）：

| 阶段 | 平均 | P50 | P95 |
|---|---|---|---|
| preprocess | 0.871016 | 0.787542 | 1.06981 |
| execute | 6.07057 | 6.11255 | 6.5164 |
| d2h | 0.282822 | 0.268577 | 0.33483 |
| validate | 1.37077 | 1.33899 | 1.38412 |
| rotated NMS | 0.339805 | 0.341425 | 0.360318 |
| total | 9.83453 | 9.79741 | 10.3617 |

`d2h_bytes` 是 1,720,320。

在大约 200 个候选上，旋转 NMS 不是瓶颈。CPU 上它大约 4 毫秒，执行大约 61 毫秒；CUDA 上它大约 0.34 毫秒，执行大约 6.1 毫秒。候选数提到 `pre_nms_top_k` 的上限 4096 时，多边形 IoU 会按候选对增长。那时候再单独做 CUDA NMS，不放进本阶段。

## 15. 已知限制

- batch 只支持 1，输出 rank 必须是 3。
- 类别数来自图，不支持动态类别数。
- 没有 CUDA OBB Decoder，也没有 CUDA 旋转 NMS。
- 不实现 ProbIoU，因此不要求与 Ultralytics NMS 的结果集相同。
- 不接受 OBB26 那种未做 Sigmoid 的角度。
- 各向异性缩放的结果是平行四边形的最小面积矩形。
- 顶点离开原图的框被丢掉，不裁剪成可见部分。
- 识别器不猜测 letterbox 或 ImageNet 归一化。
- IoU 阈值 0 会抑制后续所有框，因为比较是大于等于。
- 分数接近时，CPU 与 CUDA 的排序可以不同。
- 这次真实模型比较使用的是固定图案，不是 DOTA 标注，不能当成精度结果。
