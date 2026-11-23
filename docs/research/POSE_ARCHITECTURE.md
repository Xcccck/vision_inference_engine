# Pose 架构

本文记录 Phase 7.2 的任务语义、YOLOv8-Pose 协议和实测结果。Pose 是任务类型。YOLOv8-Pose 是一种模型结构。`yolov8_pose_raw` 是这种结构导出后的输出协议。`PoseObject` 是 VIE 返回的任务结果。四者不能互相替代。

这次真实模型比较使用固定图案，验证的是协议和推理链路，不是人体检测精度。没有使用带标注的数据集。

## 1. 接入前的状态

`TaskType::Pose` 已经存在，字符串是 `pose`。当时没有关键点、Pose 结果、Pose Decoder、Pose Recognizer，也没有把关键点映回原图的接口。`infer()` 只服务检测。

人体框可以复用检测的 `map_box_to_original` 和 `greedy_nms`。关键点不能从框的中心和宽高反推。OBB 在各向异性缩放下用平均比例拟合旋转矩形，那条规则只属于角度几何，不能用在点上。

`GeometryTransform` 的逆映射是 `x_original = (x_model - origin_x) / scale_x`，`origin = pad - crop`。`scale_x` 和 `scale_y` 可以不同。Center Crop 和 ShortestSide Resize + Center Crop 把窗口记在 `crop_x`、`crop_y`，pad 保持 0。

Manifest schema 1.0 已经能带可选对象。识别器只看图结构。

## 2. 模型来源

权重来自 Ultralytics 8.4.0 的官方发布：

`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt`

| 项 | 值 |
|---|---|
| Ultralytics | 8.4.0 |
| torch | 2.14.1+cu130 |
| onnx | 1.23.2 |
| 导出 | `imgsz=640, batch=1, dynamic=False, simplify=False, opset=17, nms=False, device=cpu` |
| pt 大小 | 6,832,633 字节 |
| pt SHA256 | `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212` |
| onnx 大小 | 13,484,600 字节 |
| onnx SHA256 | `f65cc23a7ac97f1b216336cea51700ac88be5475db1d36f6fc3ab29a2fddb3d3` |

权重和 ONNX 都在 `/tmp/vie_phase72`，不进入 Git。导出脚本是 `tools/export_yolov8n_pose.py`。它不导入 VIE。

检查点的 `nc` 是 1，`kpt_shape` 是 `[17, 3]`。公共数据结构不把 17 写死，关键点个数来自 Manifest，并必须和输出通道数一致。

## 3. ONNX 输入输出

IR 8，opset 17，domain `ai.onnx`。图有 313 个节点。

| 端点 | 名称 | dtype | 声明 shape |
|---|---|---|---|
| 输入 | `images` | FP32 | `[1, 3, 640, 640]` |
| 输出 | `output0` | FP32 | `[1, 56, 8400]` |

56 = `4 + 1 + 17 * 3`。8400 个锚点来自 640 输入上的三个步长：`80^2 + 40^2 + 20^2`。

最终节点是 axis 1 的 Concat，三个输入分别是框、类别 Sigmoid、关键点 Reshape：

`Concat ['/model.22/Mul_2_output_0', '/model.22/Sigmoid_output_0', '/model.22/Reshape_10_output_0'] ['output0']`

没有 `NonMaxSuppression`。检测识别器要求最终 Concat 恰好两个输入，所以这张图不会被认成 Detection。仓库里的 `testdata/onnx/yolov8n.onnx` 仍然是 `yolov8_detection`。

## 4. `yolov8_pose_raw`

对照 Ultralytics 8.4.0 `Pose.kpts_decode`：关键点先 reshape 成 `[B, K, 3, anchors]`，xy 是 `(raw * 2 + (anchors - 0.5)) * strides`，第三通道再做 Sigmoid。`Pose._inference` 把已经解码的 `cxcywh`、类别 Sigmoid 和这组关键点拼在一起。图内已经完成这些计算。Decoder 不再做第二次 Sigmoid，也不再乘 stride。

通道顺序，axis 1：

`cx, cy, width, height, class probabilities..., (x, y, confidence) * keypoint_count`

坐标是模型输入像素，不是归一化坐标。没有 objectness。类别分数已经是 Sigmoid 之后的概率。每个关键点三个字段，第三字段也是 Sigmoid 之后的概率。Ultralytics 把这一通道叫做 visible。它不是 COCO 标注里的 0/1/2，也不能当成人体检测分数。公共字段名是 `confidence`，语义就是这个概率。

batch 必须是 1。dtype 必须是连续的 CPU FP32。通道数必须等于 `4 + num_classes + keypoint_count * 3`，`keypoint_dimensions` 必须是 3。概率允许 `[-1e-4, 1+1e-4]` 的数值溢出。NaN 和 Inf 拒绝。低于置信度的锚点直接跳过，避免背景锚点里的坏关键点中断整张图。留下来的人体如果框、关键点或分数非有限，或者宽高不是正数，整次解码失败。关键点分数低不会删掉人体。

## 5. 解码顺序

1. 校验输出名、shape、dtype、batch 和关键点个数。
2. 按置信度过滤。比较是大于等于。
3. 分数降序，分数相同则 `anchor_index` 升序，再按 `class_id` 升序。
4. 取 `pre_nms_top_k`（1 到 4096）。
5. 人体框走 `map_box_to_original`。裁剪后面积为 0 的框丢掉。公开的 `cx, cy, width, height` 是裁剪后的框。
6. 每个关键点单独做逆映射，不裁到图像边缘。
7. 对人体框做 `greedy_nms`。IoU 大于等于阈值才抑制。是否按类抑制来自当次 runtime，不看关键点距离。

这些阈值来自当次 `PoseRuntimeConfig`，不写回 Manifest。`confidence` 和 `iou` 必须在 `[0, 1]`，`max_detections` 至少为 1。

## 6. 几何映射

关键点：

`x_original = (x_model - origin_x) / scale_x`

`y_original = (y_model - origin_y) / scale_y`

`origin = pad - crop`。x 和 y 各自用自己的比例，不用平均值。Stretch、Letterbox、Resize、Center Crop、ShortestSide Resize + Center Crop 都走这一条。Letterbox 取整后两个比例可以不同。

`inside_image` 在逆映射点落在原图内时为 true，边界各放宽 `1e-3` 像素。点可以留在原图外，坐标保持逆映射结果。裁剪窗口外的点同样保留，只是 `inside_image` 为 false。

人体框仍然使用检测那条会裁到图像内的映射。框被裁掉不改变关键点的独立映射；框本身面积为 0 时，整个人连同关键点一起丢掉。

## 7. 识别器

识别器 id 是 `yolov8_pose`，架构 `yolov8_pose`，家族 `yolov8`，协议 `yolov8_pose_raw`。必须同时满足：

- 一个 FP32 输入，rank 4，axis 1 是 3。
- 一个 FP32 输出，rank 3，batch 静态为 1。
- 没有 `NonMaxSuppression`。
- 最终 Concat 恰好三个输入：类别 Sigmoid、DFL 框、关键点。
- 框是 stride 乘法，一边是常量，另一边是 Div 和 Sub 的 Concat。DFL 是一个 Softmax 后接 `[1, reg_max, 1, 1]` 卷积，并来自三个 `4 * reg_max` 通道的卷积。8.4.0 导出的 stride 是 Constant 节点，不是 initializer。
- 类别 Sigmoid 前面是三个卷积，通道数是类别数。
- 关键点分支是同一份 reshape 上的两段 Slice 或 Split：一段乘标量 2，一段做 Sigmoid，再 Concat。三个关键点卷积的通道数不小于 3 且能被 3 整除。
- 唯一的静态通道轴等于 `4 + 类别数 + 关键点通道数`，并且这个轴是 1 或 2。

只靠输出维度、关键点个数、文件名、节点名或输入尺寸不够。缺少乘 2、多一个输出、通道数对不上，或者图里有 NMS，都返回 Unknown。Pose26 那种额外 sigma 或 RealNVP 不满足这条关键点分支，拒绝而不是套进当前协议。与 Detection、Segmentation、RT-DETR、OBB 或 Classification 同时命中时是 Ambiguous。

真实的 `yolov8n-pose.onnx` 被识别为 1 类、17 个关键点、每个 3 维、channel axis 1、layout `xy_confidence`。

生成的 Manifest 不填写颜色、resize、归一化、插值和 NMS。`box_decode` 标成 `decoded_cxcywh`，因为图里已经是 stride 乘法之后的框。它 schema 合法，但还不能 `load`。预处理由调用者用 `user` provenance 补上。

## 8. Manifest

schema 仍是 1.0。可选对象 `pose` 的字段是 `output_name`、`channel_axis`（1 或 2）、`keypoint_count`（至少 1）、`keypoint_dimensions`（必须是 3）、`keypoint_layout`（`xy_confidence`）、`score_type`（`probabilities`）。协议 `yolov8_pose_raw` 要求任务是 pose，反过来也一样。非 Pose 任务带上这些字段是 schema 错误。已有的 Detection、Segmentation、RT-DETR、Classification 和 OBB 文件不出现这个对象，含义不变。

具体输出的 rank 3 通道长度必须等于 `4 + num_classes + keypoint_count * keypoint_dimensions`。框契约是 `box_format=cxcywh`、`box_channels=4`、`box_decode=decoded_cxcywh`、`scores_are_probabilities=true`。

## 9. CPU 与 CUDA 路径

CPU：CPU Tensor，ORT CPU EP，CPU Tensor，CPU Decoder，人体框 NMS。

严格 CUDA：CUDA Tensor，ORT CUDA EP，CUDA Tensor，显式把名为 `output0` 的整份输出拷回 CPU，然后是同一个 Decoder。没有静默 CPU fallback，没有 CUDA Pose Decoder，也没有 CUDA NMS。`d2h_bytes` 只计这次显式拷贝，本次是 1,881,600 字节。ORT 内部搬移不计入。结果里的执行设备是 CPU，因为解码和 NMS 在 CPU 上。缺少 CUDA Backend 时返回 Unsupported。

`infer()` 遇到 Pose 返回 Unsupported。`infer_pose()` 要求任务是 Pose 且协议是 `yolov8_pose_raw`，否则 Unsupported，不会悄悄走检测 Decoder。

## 10. 独立参考

参考不是 Ultralytics 的完整 `predict`。同一张 640×640 的确定性 RGB 图案 `(index * 17 + 3) % 251` 分别送进：

- Ultralytics 8.4.0 的 PyTorch 前向。`net.export = True`，所以走导出时的解码，而不是训练头。
- `tools/export_yolov8n_pose.py` 里单独写的人体框 NMS。它按人体框做类相关抑制，不调用 `ultralytics.utils.nms`。
- VIE CPU EP 的 `infer_pose`。

图案不是 COCO 图片。置信度 0、`pre_nms_top_k` 300、IoU 0.45、最多 20 个人时，独立 Python 留下 20 个人。原图和网络输入同为 640×640，几何是恒等 stretch。

CPU EP 的原始张量与 PyTorch 的最大绝对差是 0.000732422，低于测试门槛 `1e-3`。输出 1,881,600 字节。NMS 后的 20 个人在类别、锚点、`inside_image`、分数（`1e-4`）、框（`1e-2` 像素）和关键点坐标（`1e-2` 像素）上与独立 Python 参考一致。

## 11. CPU 与 CUDA 的一致性

同一图案上，CPU EP 与严格 CUDA EP 的原始张量最大绝对差：

| 通道 | 最大绝对差 |
|---|---|
| 全张量 | 0.622498 |
| cx, cy, width, height | 0.622498 像素 |
| 类别概率 | 0.000695616 |
| 关键点 x、y、confidence | 0.402252 |

按锚点对齐后，两边 NMS 留下的 20 个人是同一组锚点。同一锚点的框差不超过 0.75 像素，关键点坐标差不超过 0.5 像素，人体分数和关键点概率差不超过 `1e-3`。位置顺序不要求相同。`d2h_bytes` 等于 `output_bytes`，都是 1,881,600。

## 12. 性能

输入 640×640，warmup 3 次，采样 10 次。百分位是升序后的 `floor(fraction * (n - 1))`。单位是毫秒。候选 300，NMS 后 20。这些数字只描述这一台机器和这一份模型。

CPU EP（CPU 包 `/opt/onnxruntime-linux-x64-1.20.1`，`build-ort`）：

| 阶段 | 平均 | P50 | P95 |
|---|---|---|---|
| preprocess | 27.3606 | 27.5371 | 28.6995 |
| execute | 28.2807 | 28.0802 | 29.2356 |
| d2h | 0 | 0 | 0 |
| validate | 9.09164 | 8.32427 | 11.6361 |
| pose decode | 0.264756 | 0.257909 | 0.325694 |
| NMS | 0.13192 | 0.124732 | 0.155582 |
| total | 71.5915 | 70.9281 | 75.157 |

`d2h_bytes` 是 0。validate、关键点映射和 NMS 是分开计时的。

严格 CUDA EP（GPU 包 `/opt/onnxruntime-linux-x64-gpu-1.20.1`，禁止 CPU fallback）：

| 阶段 | 平均 | P50 | P95 |
|---|---|---|---|
| preprocess | 0.631059 | 0.598069 | 0.66018 |
| execute | 4.86301 | 4.44959 | 5.74282 |
| d2h | 0.343428 | 0.291052 | 0.382338 |
| validate | 1.94941 | 1.93943 | 2.03296 |
| pose decode | 0.0556313 | 0.05431 | 0.057108 |
| NMS | 0.0302596 | 0.030169 | 0.03056 |
| total | 8.62583 | 8.06068 | 9.82329 |

`d2h_bytes` 是 1,881,600。

## 13. 已知限制

- batch 只支持 1，输出 rank 必须是 3。
- 关键点维度只接受 3，layout 只接受 `xy_confidence`。
- 没有 CUDA Pose Decoder，也没有 CUDA NMS。
- 不接受 Pose26，也不按文件名适配其他姿态模型。
- 关键点第三通道不映射成 COCO visibility。
- 越界关键点保留坐标，只把 `inside_image` 标成 false。
- 人体框仍会裁到原图内；裁完没有面积就丢掉整个人。
- 识别器不猜测颜色、resize 或归一化。
- IoU 阈值 0 会抑制后续所有框，因为比较是大于等于。
- 分数接近时，CPU 与 CUDA 的排序可以不同。
- 这次比较用的是固定图案，不是 COCO 标注，不能当成姿态精度。
