# Vision Inference Engine (VIE)

> C++17 ONNX 视觉推理框架：按计算图结构识别模型，用 Manifest 显式记录模型语义，CPU/CUDA 双路径预处理与解码，自带多模型异步运行时。

`namespace vie` · v0.1.0 · Linux（Ubuntu 22.04 验证）

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue) ![CMake](https://img.shields.io/badge/CMake-%E2%89%A53.16-064F8C) ![ONNX%20Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20.1-red) ![Status](https://img.shields.io/badge/status-pre--release-orange)

**VIE 定位**：ONNX 推理框架，不是模型格式转换器。模型格式只支持 ONNX；`.pt` 转换依赖外部导出脚本，TensorRT 尚未实现。

---

## 目录

- [核心特性](#核心特性)
- [支持的任务与模型](#支持的任务与模型)
- [开发环境](#开发环境)
- [构建](#构建)
- [快速开始](#快速开始)
- [命令行工具](#命令行工具)
- [测试](#测试)
- [部署与集成](#部署与集成)
- [已知限制](#已知限制)
- [文档](#文档)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 核心特性

- **靠图结构识别，不靠文件名**。7 个 Recognizer 只在整个计算图满足证据时给出候选；证据条数相同为 `ambiguous`，无候选为 `unknown`，不会把任意 ONNX 认成 YOLO。
- **Manifest 1.0 + provenance**。每个字段标注 `observed` / `metadata` / `inferred` / `user` / `unresolved`；颜色、归一化、resize 这类图上读不出来的语义保持未决，`load()` 直接返回 `UnresolvedField` 而不是猜一个默认值。
- **任务覆盖**：目标检测、实例分割、语义分割、旋转框、人体关键点、单标签分类。
- **CPU 与 CUDA 预处理一致**：共用同一采样公式 `source = (index + 0.5) * (source_size / resized_size) - 0.5`，支持 `letterbox` 与 `shortest_side`，几何信息留在 `GeometryTransform` 里而不是塞进 Tensor。
- **严格 CUDA**：默认 `disable_cpu_ep_fallback=1`，CUDA 失败不回退 CPU；Detection / Instance Segmentation / RT-DETR 有真正的 CUDA Decoder，只把保留结果拷回 Host。
- **运行时层**：多模型生命周期管理、线程池、`submit()` → `std::future`、动态 batch 聚合、设备感知执行调度（饱和 GPU 不会阻塞 CPU 队列）、CPU→CUDA 输入 staging、指标采集。
- **零拷贝输入**：`ImageInput` 只借用调用方内存，写明设备、布局、颜色、dtype 和行距。
- **诊断可交付**：模型检查、provider placement 审计、端到端部署校验都有独立工具和 Python 报告。

## 支持的任务与模型

| 任务 | 入口 API | 已验证模型 | 解码执行 |
|---|---|---|---|
| Detection | `infer()`，`infer_batch()`（仅 CPU 上的 `yolov8_detect_raw`） | YOLOv8n、RT-DETR-L | CPU kernel + CUDA kernel |
| Instance Segmentation | `infer_segmentation()` | YOLOv8n-seg | CPU kernel + CUDA kernel |
| Semantic Segmentation | `infer_semantic_segmentation()` | DeepLabV3-MobileNetV3-Large（torchvision 权重） | CPU（CUDA 仅前向） |
| Oriented Box | `infer_obb()` | YOLOv8n-obb | CPU（旋转多边形 NMS） |
| Pose | `infer_pose()` | YOLOv8n-pose | CPU（关键点个数来自 Manifest） |
| Classification | `infer_classification()` | ResNet18 / MobileNetV2 / MobileNetV3-Small / EfficientNet-B0（随机初始化导出） | CPU Top-K |

字段级细节、错误码和新模型适配流程见 [docs/MODEL_COMPATIBILITY.md](docs/MODEL_COMPATIBILITY.md)。

## 开发环境

| 组件 | 是否必需 | 验证过的版本 |
|---|---|---|
| Linux + GCC | 必需 | Ubuntu 22.04、GCC 11.4（C++17，`-Wall -Wextra -Wpedantic -Werror`） |
| CMake | 必需 | ≥ 3.16（本机 3.22.1） |
| nlohmann-json | 必需 | ≥ 3.2（本机 3.10.5） |
| GoogleTest | 构建测试时必需 | 本机 1.11.0（apt 包） |
| ONNX Runtime | 跑真实模型时必需 | 1.20.1，CPU 包与 GPU 包分别放在 `/opt/onnxruntime-linux-x64-1.20.1`、`/opt/onnxruntime-linux-x64-gpu-1.20.1` |
| CUDA Toolkit | 可选 | 12.6，默认 `CMAKE_CUDA_ARCHITECTURES=89` |
| Python | 可选 | 3.10 + `torch` / `torchvision` / `onnx` / `ultralytics` / `opencv-python`，只在跑 `tools/*.py` 导出与校验脚本时需要 |

`testdata/` 里的测试资产按来源分三类：

| 资产 | 是否随仓库分发 | 原因 |
|---|---|---|
| `fixed_shapes.onnx`、`dynamic_multi_io.onnx`、`external_and_subgraph.onnx`、`weights.bin`、`bus.ppm` | 是 | 合成小模型与测试图，无许可约束 |
| `deeplabv3_mobilenet_v3_large_520_opset18.onnx`（44MB） | 是，走 Git LFS | torchvision 官方权重，BSD-3 允许再分发 |
| `yolov8n.onnx`、`yolov8n-seg.onnx`、`yolov8n_batch.onnx`、`rtdetr-l.onnx`（共约 165MB） | **否** | 由 Ultralytics 预训练权重导出，受 AGPL-3.0 约束 |

缺这四个模型时，CMake 会在配置阶段列出缺失项，真实模型用例会报文件不存在。用记录好的导出参数重建（需 `ultralytics` / `torch` / `onnx`，`.pt` 落在仓库外的临时目录）：

```bash
python3 tools/export_test_models.py --list    # 先看缺哪几个
python3 tools/export_test_models.py          # 只补齐缺失的 fixture
```

脚本永远先校验输入名、shape、opset（Recognizer 依赖这些结构证据），SHA256 不一致只给警告——不同 ultralytics/torch build 的序列化结果不同。

## 构建

```bash
# 1) 核心库：不需要 ONNX Runtime / CUDA，只跑 mock backend 的测试
cmake -S . -B build && cmake --build build -j

# 2) ONNX Runtime CPU EP
cmake -S . -B build-ort -DVIE_ENABLE_ONNXRUNTIME=ON \
      -DVIE_ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.20.1
cmake --build build-ort -j

# 3) ONNX Runtime + 严格 CUDA EP（GPU 包路径非空时才会编译 CUDA Backend）
cmake -S . -B build-ort-gpu -DVIE_ENABLE_ONNXRUNTIME=ON -DVIE_ENABLE_CUDA=ON \
      -DVIE_ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.20.1 \
      -DVIE_ONNXRUNTIME_GPU_ROOT=/opt/onnxruntime-linux-x64-gpu-1.20.1 \
      -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-ort-gpu -j

# 4) 可选：cv::Mat → ImageInput 适配（可在上面任意构建上追加开关）
cmake -S . -B build-opencv -DVIE_ENABLE_ONNXRUNTIME=ON \
      -DVIE_ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.20.1 \
      -DVIE_ENABLE_OPENCV=ON
cmake --build build-opencv -j
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `VIE_ENABLE_ONNXRUNTIME` | `OFF` | 打开后才有真实 ONNX Backend，否则只有 Mock |
| `VIE_ENABLE_CUDA` | `OFF` | 找不到 `CUDAToolkit` 时自动回落 `OFF` |
| `VIE_ONNXRUNTIME_ROOT` | `""` | CPU 包前缀；未设置时回退环境变量 `ONNXRUNTIME_ROOT` |
| `VIE_ONNXRUNTIME_GPU_ROOT` | `""` | 仅 `CUDA` 与 `ONNXRUNTIME` 同时打开且路径非空时生效 |
| `VIE_BUILD_TESTS` | `ON` | 需要 GoogleTest；上面的命令都默认打开 |
| `VIE_ENABLE_TENSORRT` | `OFF` | 占位选项，开启也会被强制回 `OFF` |
| `VIE_ENABLE_OPENCV` | `OFF` | 打开后编译可选的 `cv::Mat` → `ImageInput` 适配；核心库仍不含 OpenCV 头 |

同一个进程只链接一份 `libonnxruntime.so.1`：指定 GPU 包时 CPU Backend 也用它，但仍然只注册 CPU Execution Provider。

## 快速开始

```cpp
#include <vie/vie.hpp>

// 1. 分析 + 识别 + 生成 Manifest，全部来自图结构
const auto report = vie::OnnxAnalyzer{}.analyze_file("testdata/onnx/yolov8n.onnx");
const auto recognition = vie::default_recognizer_registry().recognize(report);
vie::ModelManifest manifest =
    vie::ManifestGenerator{}.generate(report, recognition, "testdata/onnx/yolov8n.onnx");

// 2. 补齐图上读不出来的语义，否则 load() 报 UnresolvedField
manifest.input.color = "rgb";
manifest.input.normalization = "scale_0_1";
manifest.input.resize = "letterbox";
manifest.input.interpolation = "bilinear";
manifest.input.padding_value = 114.0f;

// 3. 加载并推理。像素由调用方提供，VIE 不解码图像文件
vie::ModelConfig config;
config.manifest = manifest;
config.model_path = "testdata/onnx/yolov8n.onnx";
config.device = vie::DeviceType::Cpu;

vie::InferenceEngine engine;
engine.load(config);
const vie::DetectionBatch batch = engine.infer(image);  // image 是 vie::ImageInput
for (const vie::Detection& det : batch.results.front().detections) {
  std::cout << det.class_id << ": " << det.confidence << "\n";
}
```

现成的 JSON Manifest 例子：[tools/diagnostics/model_deployment/yolov8n.manifest.json](tools/diagnostics/model_deployment/yolov8n.manifest.json)，用 `ModelManifest::load_file()` 读入。

已经用 OpenCV 或其他工具解码好图像时，可以直接接入（需 `-DVIE_ENABLE_OPENCV=ON`）：

```cpp
#include <vie/opencv/image_input.hpp>   // 不在 vie/vie.hpp 里，OpenCV 是可选依赖
#include <opencv2/core.hpp>

cv::Mat bgr = ...;  // 调用方自己 imread / 采集；VIE 不做 JPG/PNG 解码
const vie::ImageInput view = vie::from_cv_mat(bgr);  // 默认 BGR；ROI 用真实 step 作 stride
const vie::DetectionBatch out = engine.infer(view);

// 异步提交：submit() 内部已深拷贝 CPU 像素，返回后即可释放 bgr
std::future<vie::DetectionResult> fut = runtime.submit("det", vie::from_cv_mat(bgr));
```

适配只接受 `CV_8UC1`（Gray）、`CV_8UC3`（默认 BGR，可显式传 `ColorOrder::Rgb`）、`CV_32FC1`、`CV_32FC3`；`CV_8UC4`/16 位等直接抛 `Unsupported`，不静默丢 alpha 或换深度。同步 `infer()` 是借用语义，`cv::Mat` 必须活到调用返回。

多模型 + 异步提交：

```cpp
#include <vie/vie.hpp>
#include <vie/runtime/runtime.hpp>   // Runtime 不在聚合头文件里

vie::RuntimeConfig rc;
rc.thread_pool_size = 4;
rc.enable_profiling = true;
vie::Runtime runtime(rc);

runtime.models().load("det", det_config);
std::future<vie::DetectionResult> fut = runtime.submit("det", image);  // 异步路径只接受 CPU 输入
auto result = fut.get();
std::cout << result.detections.size() << " boxes, "
          << runtime.snapshot_metrics().total_inferences << " inferences\n";
```

同一模型的并发调用会被内部 mutex 串行；要真正并行同一模型，请用不同 ID 加载多个实例。架构细节见 [docs/RUNTIME_ARCHITECTURE.md](docs/RUNTIME_ARCHITECTURE.md)。

## 命令行工具

| 工具 | 用途 |
|---|---|
| `vie_analyze_onnx <model.onnx>` | ONNX 结构报告：声明 shape、value_info、propagated shape 互不覆盖，算子按 domain + op type 计数 |
| `vie_recognize_onnx <model.onnx>` | 识别候选与 `recognized` / `ambiguous` / `unknown` 判定 |
| `vie_model_inspect <model.onnx> [--json --report F --manifest M --cpu --cuda]` | 一站式诊断：分析 → 识别 → Manifest → 真实 EP 执行 |
| `vie_ort_profile_inspect` | 解析 ORT profiling trace，审计节点是否真的落在 CUDA EP |
| `vie_deploy_verify --model M --manifest J --image PPM` | 单张真实图像跑通完整链路，输出 session / 检测 / 延迟 / profile JSON |
| `vie_semantic_verify --model M --image PPM` | 语义分割端到端校验，配合 `reference_infer.py` 做数值对照 |
| `runtime_benchmark` / `real_model_benchmark` / `batch_benchmark` / `batch_aggregator_benchmark` / `multi_model_concurrency_benchmark` | 吞吐与并发基准 |

## 测试

```bash
ctest --test-dir build-ort --output-on-failure          # CPU + ORT：366 个用例
ctest --test-dir build-ort-gpu --output-on-failure      # CUDA + ORT：432 个用例
```

缺 fixture 时真实模型用例走 `VIE_REQUIRE_MODEL` 跳过而不是失败，所以新 clone 直接 `ctest` 也是全绿 —— 但那意味着约 73 个用例根本没跑。要验证真实模型链路，先执行 `python3 tools/export_test_models.py`，跳过数应归零。

40 个测试源文件覆盖 Tensor / Manifest / Analyzer / Recognizer / Decoder / Preprocess / Runtime / OpenCV 适配。sanitizer 构建：

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DVIE_ENABLE_ONNXRUNTIME=ON \
      -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -g -O1"
```

## 部署与集成

**源码集成**（推荐，仓库暂未提供 `install()` / CMake package config，`find_package(vie)` 不可用）：

```cmake
add_subdirectory(third_party/vision_inference_engine)
target_link_libraries(your_app PRIVATE vie::vie)   # include 目录与 VIE_HAS_* 编译定义随 INTERFACE 传播
```

也可以直接链 `build*/libvie.a` 并加上 `include/`。运行期需要让动态加载器找到 ONNX Runtime 与 CUDA 库：仓库内工具已写入 `BUILD_RPATH`，你自己的可执行文件需要设置 `INSTALL_RPATH` 或 `LD_LIBRARY_PATH`。

**CUDA 上线检查清单**：

1. `vie_model_inspect model.onnx --manifest model.json --cuda` 确认能建 Session；
2. `vie_ort_profile_inspect` 确认没有节点掉回 CPU。已知 `yolov8n.onnx`（opset 20）在 ORT 1.20.1 CUDA EP 下缺 Resize v19 kernel，严格模式建 Session 失败，用 `tools/diagnostics/model_deployment/convert_onnx_opset.py` 降到 opset 18，或显式 `cuda_cpu_ep_fallback=true` 接受混合执行；
3. `python3 tools/diagnostics/model_deployment/validate_model_deployment.py --model m.onnx --target-device cuda` 出完整部署报告。

**输入约束**：FP32、固定输入尺寸、单输入、batch = 1（仅 CPU 上的 YOLOv8 检测 `infer_batch()` 支持多帧）；模型内不能自带 `NonMaxSuppression`；CUDA 推理时 `ImageInput` 必须已在同一块 GPU 上（异步路径除外，它自带 CPU→CUDA staging）。

## 已知限制

- 不做图像文件解码（imread / JPG / PNG）；OpenCV 仅是可选的 `cv::Mat` → `ImageInput` 适配层（`VIE_ENABLE_OPENCV=ON`），解码与采集仍由调用方完成。
- TensorRT / PyTorch 解析未实现。
- NMS 阈值、Top-K、单类/多类选择不自动确定，全部来自 Manifest 或当次 runtime config。
- 单个 `InferenceEngine` 非线程安全；重叠调用是数据竞争，库不会自动排队。
- 异步 `submit()` 只接受 CPU 输入；batch > 1 仅 CPU 上的 `yolov8_detect_raw` 支持，其余任务与 CUDA 返回 `Unsupported`。
- Classification / OBB / Pose / Semantic Segmentation 的解码在 CPU，CUDA 只加速前向并计入 `d2h_bytes`。
- 多人跟踪、ReID、动作识别未实现。

## 文档

- [docs/MODEL_COMPATIBILITY.md](docs/MODEL_COMPATIBILITY.md) — 任务/模型/协议矩阵、错误码、新模型适配流程
- [docs/RUNTIME_ARCHITECTURE.md](docs/RUNTIME_ARCHITECTURE.md) — 运行时、队列、聚合与调度设计
- [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) — 各 Phase 的实现语义与边界约束（原 README 明细）
- [docs/research/](docs/research/) — 架构审计、基准结果与单项调研
- [examples/](examples/)、[tools/](tools/) — 可运行示例与诊断工具源码

## 致谢

VIE 建立在以下开源项目之上，模型侧结论也以它们的实现为对照基准：

- [ONNX Runtime](https://github.com/microsoft/onnxruntime)（MIT）— 唯一的真实推理后端，CPU / CUDA Execution Provider。
- [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）— YOLOv8 Detection/Segment/OBB/Pose 与 RT-DETR 的导出脚本和图结构参考实现。它们的权重不再进入本仓库，见[开发环境](#开发环境)。
- [pytorch/vision](https://github.com/pytorch/vision)（BSD-3-Clause）— DeepLabV3-MobileNetV3-Large 官方预训练权重与分类网络结构。
- [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)（Apache-2.0）— RT-DETR 原始实现。
- [nlohmann/json](https://github.com/nlohmann/json)（MIT）— Manifest 与报告序列化，构建期必需依赖。
- [GoogleTest](https://github.com/google/googletest)（BSD-3-Clause）— 全部单元测试与基准。

## 许可证

代码采[MIT](LICENSE)。测试资产按来源分开：合成 ONNX 模型随仓库分发；DeepLabV3 权重是 torchvision BSD-3；Ultralytics 系（YOLOv8n / -seg / batch、RT-DETR-L）不随仓库分发，`tools/export_test_models.py` 重建出的文件仍受 Ultralytics 权重的 **AGPL-3.0** 约束，商用或对外提供服务前请自行确认用途。
