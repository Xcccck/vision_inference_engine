# VIE 设计与实现细节

> 本文从 README 拆出，保留按 Phase 记录的实现语义与边界约束（provenance 规则、解码协议、设备行为等）。
> 使用者入口见 [../README.md](../README.md)；能力矩阵见 [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md)；运行时架构见 [RUNTIME_ARCHITECTURE.md](RUNTIME_ARCHITECTURE.md)。

## 概述

命名空间：`vie`。这是一个通用视觉推理框架，当前接通了 ONNX 上的 YOLOv8 Detection、YOLOv8 Segmentation、YOLOv8-OBB、Ultralytics RT-DETR-L 的 Detection，以及单标签 Classification。它不会把任意 ONNX 模型认成这些家族。TensorRT 还没有实现。YOLOv8-Pose 走 `infer_pose`，协议是 `yolov8_pose_raw`。

Phase 1 做了这些事：

- CPU `TensorStorage` / `Tensor` 视图，64 字节对齐，`clone()` 不共享存储
- Host `DeviceContext`。未编译 CUDA 时请求 CUDA 会失败；编译 CUDA 后，上下文持有真实 device id 和 stream
- Manifest schema `1.0`，每个字段带 `observed` / `metadata` / `inferred` / `user` / `unresolved`
- 结构校验和关键字段未决列表是两份分开的结果
- `IInferenceBackend` 返回具名张量；`IModelDecoder` 按输出协议读取这些张量
- Mock backend、mock decoder，以及注册表
- `InferenceEngine::load` 在关键字段未决、或请求了本机构建没有的后端时失败

Mock 协议是管线测试替身。它从名为 manifest 第一个输出的 fp32 张量读取 6 个数：`x1, y1, x2, y2, score, class_id`。它不是 YOLO 解码器。

Phase 2 增加 `OnnxAnalyzer`。它只读 ONNX ModelProto 的结构：原始声明的 shape、文件里的 value_info、以及单独计算的 propagated shape 互不覆盖。动态维度保留 symbolic 名称。算子按 domain 和 op type 一起计数。多输入、多输出、initializer、external data 和不同 opset domain 都记在 `ModelAnalysisReport` 里。JSON 和文本只是这份结构的输出。

`ModelRecognizerRegistry` 只在某个识别器接受整张计算图时给出候选。证据条数相同则是 `ambiguous`，没有候选则是 `unknown`。已注册的是 YOLOv8 Detection（单输出，`yolov8_detect_raw`）、YOLOv8 Segmentation（预测加 prototype，`yolov8_seg_raw`）和 Ultralytics RT-DETR Detection（`rtdetr_detection`，协议 `rtdetr_detection_raw`）。三者按图结构互斥，不靠文件名。RT-DETR 识别要求最终 Concat 同时接上 sigmoid 后的归一化框分支、sigmoid 后的分类分支，以及喂给这两条线性层的唯一 TopK；单独的 TopK 或单独的 `[1,300,84]` 不够。Query 数来自图内 TopK 的 k，不是 `pre_nms_top_k`。声明的输出 shape 保持符号维度，不改写成运行时的 `[1,300,84]`。这份导出的 resize 记为 `stretch`，NMS 记为关闭。颜色和归一化仍然未决。识别结果本身不能 `load`：输出 shape 仍是符号维。调用方可以另交一份运行时 Manifest，用 `user` provenance 写明 ORT 实测的 `[1, num_queries, 4+num_classes]`、RGB、`/255` 和双线性插值；这些值不回写成图的声明 shape。`rtdetr_detection_raw` 已注册，走 `infer()`，不跑 NMS，也不做第二次 Sigmoid。CPU 设备用 CPU Decoder；CUDA 设备用 CUDA Decoder，失败不会退回 CPU。未编译 CUDA 时，CUDA 设备返回 Unsupported。YOLOv8 的颜色、归一化、resize、NMS 和 mask 运行策略保持未决。`testdata/onnx/yolov8n.onnx` 的 SHA256 是 `65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5`。`testdata/onnx/yolov8n-seg.onnx` 的 SHA256 是 `430dd97586f37e9b87940419eb54ccf7a8e6460c8218785510a714ad46c8607e`。`testdata/onnx/rtdetr-l.onnx` 的 SHA256 是 `91e7cb0358575e37d7d241db616af07fcce4924ca0b3598ac7e35f2ea8ff719f`。

```bash
/home/workspace/vision_inference_engine/build/vie_analyze_onnx testdata/onnx/fixed_shapes.onnx
```

## 模型检查工具

`vie_model_inspect` 检查一个 ONNX 模型能否被 VIE 正确解析、识别和运行。

```bash
# 基础检查：分析、识别、生成 Manifest、验证
vie_model_inspect model.onnx

# JSON 报告
vie_model_inspect model.onnx --json --report report.json

# 验证用户提供的 Manifest
vie_model_inspect model.onnx --manifest model.json

# 测试 CPU EP 执行（需要完整 Manifest）
vie_model_inspect model.onnx --cpu

# 测试 CUDA EP 执行（需要 CUDA 构建和完整 Manifest）
vie_model_inspect model.onnx --cuda
```

诊断状态说明：

- `analyzer_failed`：ONNX 文件无法解析
- `recognition_failed`：没有 Recognizer 接受该模型
- `manifest_invalid`：Manifest 缺少关键语义字段（color、normalization、resize）
- `recognized`：模型被识别，但 Manifest 不完整
- `cpu_tested` / `cuda_tested`：对应 EP 执行成功

详见 [docs/MODEL_COMPATIBILITY.md](docs/MODEL_COMPATIBILITY.md)。

## 构建与测试

```bash
cmake -S /home/workspace/vision_inference_engine -B /home/workspace/vision_inference_engine/build -DVIE_BUILD_TESTS=ON
cmake --build /home/workspace/vision_inference_engine/build -j
ctest --test-dir /home/workspace/vision_inference_engine/build --output-on-failure
```

依赖：CMake ≥ 3.16、C++17、nlohmann/json ≥ 3.2、GoogleTest（仅测试）。

`VIE_ENABLE_CUDA`、`VIE_ENABLE_TENSORRT`、`VIE_ENABLE_ONNXRUNTIME`、`VIE_ENABLE_OPENCV` 默认关闭。`VIE_ONNXRUNTIME_ROOT` 是 CPU 包。`VIE_ONNXRUNTIME_GPU_ROOT` 只在 CUDA 和 ONNX Runtime 都打开、并且路径非空时使用；这时 CUDA Backend 才会编译。同一个进程只链接一份 `libonnxruntime.so.1`：指定了 GPU 包时，CPU Backend 也链到这份库，但仍然只注册 CPU Execution Provider。

CUDA Backend 默认关闭 CPU EP fallback（`session.disable_cpu_ep_fallback=1`）。Session 能创建，表示 CUDA EP 接受了整张图。`set_cpu_ep_fallback(true)` 允许 ORT 把不支持的节点放到 CPU，这是混合执行，不是纯 GPU 执行。当前这份 YOLOv8n 在严格模式下无法创建 Session，因为 ORT 1.20.1 会把部分节点分给 CPU EP；要跑它必须显式打开 fallback。输入必须已经是连续的 CUDA Tensor，绑定借用它的设备指针。输出由 ORT 分配在 CUDA 上，再做一次 D2D 复制到 VIE 的 Storage。ORT 使用自己的 stream，返回前会同步。TensorRT 和 OpenCV 仍然没有实现。

图像预处理是 `ImagePreprocessor`。`ImageInput` 只借用调用方的内存，并写明设备、布局、颜色、dtype 和行距。CPU 与 CUDA 使用同一采样公式：`source = (index + 0.5) * (source_size / resized_size) - 0.5`。最近邻是 `floor(source + 0.5)`，双线性复制边界。Letterbox 先取 `min(target/source)`，内容尺寸用 `round` 后再夹到目标内，余数用整数除法分到两侧。`shortest_side` 把短边钉成 `short_side`，长边按一半远离 0 取整，再做中心裁剪；奇数余数分给右和下。`scale_x` 和 `scale_y` 是取整后的比例，可以不相等。几何信息在 `GeometryTransform` 里，不放进 Tensor。

Manifest 的 `input.color` 是模型输入颜色，不是源图像颜色。`interpolation` 和 `padding_value` 可以写入 manifest；缺了它们时，`preprocess()` 返回 `UnresolvedField`，已有的 Tensor `execute()` 不受影响。`resize=shortest_side` 还要 `short_side`、`crop_width`、`crop_height`，模型输入 shape 是裁剪后的尺寸。`normalization=explicit` 使用调用者给出的 `scale`、`mean`、`std`。旧词 `crop` 仍不执行。识别器不会因为模型是 YOLOv8 或 ResNet 就填上 RGB、letterbox、短边或 mean/std。CUDA 预处理只在 `VIE_ENABLE_CUDA` 时编译；输入必须已经在同一块 GPU 上，返回前 `cudaDeviceSynchronize`。

## 已接通的路径

- 模型格式：ONNX。Mock 只用于管线测试。
- 任务：YOLOv8 Detection 和 Ultralytics RT-DETR Detection（都走 `infer`），YOLOv8 Segmentation（`infer_segmentation`），语义分割（`infer_semantic_segmentation`），YOLOv8-OBB（`infer_obb`），YOLOv8-Pose（`infer_pose`），以及单标签分类（`infer_classification`）。Decoder 按 Manifest 里的 protocol 选择，不按模型家族分支。RT-DETR、分类和语义分割在 CPU 和 CUDA 上各有一个 Decoder，按设备选择。OBB 的旋转 NMS 和 Pose 的人体框 NMS 只在 CPU 上。
- Backend：构建打开 ONNX Runtime 时有 CPU EP；同时打开 CUDA 和 GPU 包时另有 CUDA EP。CUDA EP 默认禁止 CPU fallback。
- 预处理和解码：CPU 始终可用；CUDA 只在 `VIE_ENABLE_CUDA` 时编译。

## 还没有实现

- NMS 阈值的自动确定
- TensorRT / PyTorch 解析
- 图像文件解码
- 运行时 Softmax 和多标签。模型自己输出的概率走 `classification_probabilities_raw`，解码时不再做 Softmax
- resize 模式 `crop`
- 多人跟踪、ReID、动作识别，以及 CUDA 上的 Pose Decoder 和 CUDA NMS
- 并发推理。`load`、`infer`、`infer_segmentation`、`infer_classification`、`infer_obb` 和 `infer_pose` 共用同一个 session，没有锁。重叠调用是数据竞争，库不会把它们串成队列。

`device()` 只描述最近一次成功的 `load`。第一次 `load` 失败时 `loaded()` 仍为 false，`device()` 仍是默认的 CPU 上下文，这不是一次 CPU 推理。再次 `load` 失败不会拆掉已经成功的 session。

`InferenceEngine::infer(ImageInput)` 按 Manifest 的 decoder protocol 选择解码器。`yolov8_detect_raw` 的图内已经做完 DFL、dist2bbox 和 Sigmoid，输出是模型输入像素下的 cxcywh 加类别概率；Decoder 不再做一次 DFL。`rtdetr_detection_raw` 的输出是归一化 cxcywh 加已经做过 Sigmoid 的类别概率；Decoder 先按模型输入宽、高分别还原到像素，再走 `GeometryTransform`，不执行 NMS，也不要求 IoU。`Detection.query_index` 是模型输出里的 Query 下标，不是排序后的位置。RT-DETR 的 CPU Decoder 和 CUDA Decoder 都写这个下标；YOLOv8 仍保持 -1。任一 Query 出现 NaN 或 Inf 时，两个 Decoder 都返回 `InvalidArgument`，并指出最早的 Query 和该 Query 内按 cx、cy、width、height、class score 顺序遇到的第一个字段。CUDA 用一次 `atomicMin` 写入 `(query << 3) | field`，不会把一个 Query 的下标和另一个 Query 的字段拼在一起。

当前 RT-DETR-L 的输出是 `[1,300,84]` FP32，共 100800 个元素、403200 字节。CUDA Decoder 不把这份 Tensor 拷回 Host。`DetectionResult.d2h_bytes` 只计留下的检测：每条 28 字节（`x1,y1,x2,y2,confidence,class_id,query_index`）。另外一次 12 字节的计数器读取（fault、passed、kept）是控制信息，不计入 `d2h_bytes`。`sort_ms` 和 `d2h_bytes` 是这一张图的 Decoder 统计，不是单个 `Detection` 的字段；CPU 路径两者都是 0，YOLOv8 也不写它们。CUDA Decoder 在 GPU 上完成 argmax、过滤、坐标变换和排序，只把留下的框拷回 Host；它使用默认 stream，不是 `DeviceContext` 的 stream。300 个 query 时，kernel 启动和 Thrust 排序的开销大于计算本身，端到端比 CPU Decoder 慢。NMS 阈值、Top-K 和单类/多类选择写在该次加载的 Manifest 里，`infer` 不会改这份 Manifest。`execute()` 仍然只跑 Tensor。当前这份 `yolov8n.onnx` 在严格 CUDA EP 下无法创建 Session；失败不会改走 CPU。

Manifest 描述模型固有语义和预处理。分割的置信度、IoU、NMS、`retina_masks`、二值规则和 `max_mask_bytes` 只来自当次 `SegmentationRuntimeConfig`，不写回 Manifest，也不读取 Manifest 里的同名运行参数。`max_mask_bytes` 只限制返回的 mask 字节数，不是整次推理的显存峰值。

```cpp
vie::ModelConfig detection;
detection.manifest = detection_manifest;  // 识别结果，另填预处理和 NMS
detection.model_path = "testdata/onnx/yolov8n.onnx";
detection.device = vie::DeviceType::Cpu;
vie::InferenceEngine detector;
detector.load(detection);
vie::DetectionBatch detected = detector.infer(image);

vie::SegmentationRuntimeConfig runtime;
runtime.confidence_threshold = 0.25f;
runtime.iou_threshold = 0.45f;
runtime.class_selection = vie::ClassSelection::BestClass;
runtime.class_aware_nms = true;
runtime.nms_enabled = true;
runtime.pre_nms_top_k = 300;
runtime.max_detections = 20;
runtime.retina_masks = true;
runtime.binary_mask_gt_zero = true;
runtime.max_mask_bytes = 64ull << 20;
vie::ModelConfig segmentation;
segmentation.manifest = segmentation_manifest;  // 识别结果，只另填预处理
segmentation.model_path = "testdata/onnx/yolov8n-seg.onnx";
segmentation.device = vie::DeviceType::Cpu;
vie::InferenceEngine segmenter;
segmenter.load(segmentation);
vie::SegmentationBatch segmented = segmenter.infer_segmentation(image, runtime);
```

检测模型调用 `infer_segmentation`，或分割模型调用 `infer`，都返回 `Unsupported`。分类模型调用 `infer` 同样返回 `Unsupported`。CUDA 推理要求 `ImageInput` 已经在同一块 GPU 上。

`classification_logits_raw` 解码 rank-2 的 FP32 logits。`classification_probabilities_raw` 解码模型已经输出的 rank-2 FP32 概率，要求每个值在 `[0,1]` 且概率和与 1 的差不超过 `1e-3`，不会再次 Softmax，也不会把向量重新归一化。两种协议都按分数从高到低取 Top-K，分数相同则按 `class_id` 从小到大。`score_type` 必须和协议一致。CUDA EP 的输出留在 CUDA 上，`infer_classification()` 在 GPU 上完成校验和 Top-K，只把最终的 `top_k` 个 `(class_id, score)` 拷回 CPU（`top_k * 8` 字节），结果里的执行设备是 Cuda。

```cpp
vie::ClassificationRuntimeConfig labels;
labels.top_k = 5;
vie::ModelConfig classification;
classification.manifest = classification_manifest;  // 识别器可填 task、protocol 和 classification；颜色、resize、short_side、crop、mean/std 仍由调用者填写
classification.model_path = "model.onnx";
classification.device = vie::DeviceType::Cpu;
vie::InferenceEngine classifier;
classifier.load(classification);
vie::ClassificationBatch ranked = classifier.infer_classification(image, labels);
```

`yolov8_seg_raw` 有 CPU 和 CUDA 解码。它按名称读取 `output0` 和 prototype，复用检测的置信度过滤、坐标映射和 NMS，并保留 prediction index 以收集 mask 系数。重建是 `M = a · P`，不再做 Sigmoid；二值规则必须显式写成 `M > 0`。`retina_masks=false` 时 mask 是模型输入尺寸，`true` 时是原图尺寸。CUDA 路径把 mask 留在 CUDA Tensor 上，失败时不会退回 CPU。Kernel 使用默认 stream，不是 `DeviceContext` 自己的 stream。缺 `retina_masks`、NMS 或 mask 内存预算时返回 `UnresolvedField`。`InferenceEngine::infer_segmentation()` 走这条路径；`infer()` 遇到分割 Manifest 返回 `Unsupported`。TensorRT 还没有实现。

`yolov8_obb_raw` 读取已经解码好的 `cx, cy, width, height, class probabilities, angle`。角度是弧度，范围 `[-pi/4, 3pi/4]`，宽边方向是 `(cos, sin)`。`infer_obb()` 用调用者给出的置信度、IoU 和 Top-K 做旋转矩形的多边形 NMS，再把框映射回原图。各向同性缩放保持角度；各向异性缩放先反映射四个顶点，再拟合最小面积矩形。顶点离开原图的框会被丢掉，不会裁成可见碎片。CUDA EP 把整份输出拷回 CPU 后使用同一个 Decoder，这次拷贝计入 `d2h_bytes`。`infer()` 遇到 OBB Manifest 返回 `Unsupported`。

```cpp
vie::ObbRuntimeConfig obb_runtime;
obb_runtime.confidence_threshold = 0.25f;
obb_runtime.iou_threshold = 0.45f;
obb_runtime.class_selection = vie::ClassSelection::BestClass;
obb_runtime.class_aware = true;
obb_runtime.nms_enabled = true;
obb_runtime.pre_nms_top_k = 1000;
obb_runtime.max_detections = 300;
vie::ModelConfig obb;
obb.manifest = obb_manifest;  // 识别器可填 task、protocol 和 obb；颜色、resize、归一化仍由调用者填写
obb.model_path = "yolov8n-obb.onnx";
obb.device = vie::DeviceType::Cpu;
vie::InferenceEngine oriented;
oriented.load(obb);
vie::ObbBatch found = oriented.infer_obb(image, obb_runtime);
```

`yolov8_pose_raw` 读取已经解码好的 `cx, cy, width, height, class probabilities`，后面是若干组 `x, y, confidence`。关键点个数来自 Manifest，不写死成 17。`confidence` 是图内 Sigmoid 之后的概率，Ultralytics 把这一通道叫做 visible；它不是 COCO 的 0/1/2 可见性，也不是人体检测分数。低关键点分数不会删掉人体。`infer_pose()` 用调用者给出的置信度、IoU 和 Top-K 对人体框做 NMS，框和关键点分别映射回原图。关键点使用各自的 `scale_x`、`scale_y` 和 crop/pad 原点，不从框中心反推，也不裁到图像边缘；落在原图外时 `inside_image` 为 false。CUDA EP 把整份输出拷回 CPU 后使用同一个 Decoder，这次拷贝计入 `d2h_bytes`。`infer()` 遇到 Pose Manifest 返回 `Unsupported`。颜色、resize 和归一化仍由调用者填写。

```cpp
vie::PoseRuntimeConfig pose_runtime;
pose_runtime.confidence_threshold = 0.25f;
pose_runtime.iou_threshold = 0.45f;
pose_runtime.class_selection = vie::ClassSelection::BestClass;
pose_runtime.class_aware = true;
pose_runtime.nms_enabled = true;
pose_runtime.pre_nms_top_k = 1000;
pose_runtime.max_detections = 300;
vie::ModelConfig pose;
pose.manifest = pose_manifest;  // 识别器可填 task、protocol 和 pose；颜色、resize、归一化仍由调用者填写
pose.model_path = "yolov8n-pose.onnx";
pose.device = vie::DeviceType::Cpu;
vie::InferenceEngine posed;
posed.load(pose);
vie::PoseBatch people = posed.infer_pose(image, pose_runtime);
```

分类识别接受全局池化后的单个线性头。输出前没有 Softmax 时协议是 `classification_logits_raw`；同一条路径上、axis 与类别轴一致的 Softmax 使用 `classification_probabilities_raw`。它不填写颜色、resize、归一化或类别名，也不把 1000 维当成 ImageNet。Sigmoid 和单独的 Softmax 不会被认成单标签分类。

## 运行时与并发推理

Phase 8 引入了 Runtime 层，支持多模型管理和并发推理。

### 核心组件

- **ThreadPool**：固定大小线程池，支持异步任务提交和 `std::future` 返回值
- **ModelInstance**：线程安全的模型实例包装器，自动记录推理指标
- **ModelManager**：多模型生命周期管理（加载/卸载/查询）
- **Runtime**：顶层协调器，集成线程池、模型管理和指标收集
- **MetricsCollector**：线程安全的性能指标收集器

### 并发模型

**不同模型**：多个 `ModelInstance` 可以完全并行推理，无同步开销。每个实例拥有独立的 `InferenceEngine`、Backend 和 Decoder。

**同一模型**：对同一 `ModelInstance` 的并发调用会被内部 mutex 串行化。这是因为 Backend 的 `execute()` 方法会修改 timing 状态（非线程安全）。

**建议**：
- 需要并行执行同一模型时，加载多个实例（不同 ID）
- 线程池大小设置为并发模型数，而非总模型数
- CUDA 模型天然支持 GPU 共享（通过 CUDA streams）

### 使用示例

```cpp
// 配置并创建 Runtime
vie::RuntimeConfig config;
config.thread_pool_size = 4;
config.enable_profiling = true;
vie::Runtime runtime(config);

// 加载多个模型
auto detector = runtime.models().load("det", det_config);
auto segmenter = runtime.models().load("seg", seg_config);

// 并发提交推理任务
auto task1 = runtime.thread_pool().submit([detector, image1] {
  return detector->infer(image1);
});
auto task2 = runtime.thread_pool().submit([segmenter, image2, seg_runtime] {
  return segmenter->infer_segmentation(image2, seg_runtime);
});

// 获取结果
auto result1 = task1.get();
auto result2 = task2.get();

// 查看指标
auto metrics = runtime.snapshot_metrics();
std::cout << "Total inferences: " << metrics.total_inferences << "\n";
std::cout << "Total time: " << metrics.total_inference_time_ms << " ms\n";

// 卸载模型
runtime.models().unload("det");
runtime.models().unload("seg");
```

详见 [docs/RUNTIME_ARCHITECTURE.md](docs/RUNTIME_ARCHITECTURE.md) 和 [examples/multi_model_runtime.cpp](examples/multi_model_runtime.cpp)。

