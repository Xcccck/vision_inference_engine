#include "onnx_test_writer.hpp"
#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)
#include <algorithm>
#include <chrono>
#endif
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include <cuda_runtime.h>

#include <iostream>
#endif

namespace {

OnnxTestWriter dual_identity() {
  OnnxTestWriter graph;
  graph.message(1, node_message("Identity", {"images"}, {"output0"}));
  graph.message(1, node_message("Identity", {"images"}, {"output1"}));
  graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(640),
                                                               static_dim(640)})));
  graph.message(12, value_info("output0", tensor_type(1, true, {static_dim(1), static_dim(116), static_dim(8400)})));
  graph.message(12, value_info("output1", tensor_type(1, true, {static_dim(1), static_dim(32), static_dim(160),
                                                               static_dim(160)})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.text(2, "yolov8n-seg.onnx");
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

#if defined(VIE_HAS_ONNXRUNTIME)
void mark_user(vie::ModelManifest& manifest, const char* field) {
  manifest.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {"integration test"}};
}

void fill_preprocess(vie::ModelManifest& manifest) {
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  const char* fields[] = {"input.color", "input.normalization", "input.resize", "input.interpolation",
                          "input.padding_value"};
  for (const char* field : fields) mark_user(manifest, field);
}

void fill_detection_policy(vie::ModelManifest& manifest, float confidence) {
  manifest.decoder.class_selection = "best_class";
  manifest.nms.enabled = true;
  manifest.nms.confidence_threshold = confidence;
  manifest.nms.iou_threshold = 0.45f;
  manifest.nms.max_detections = 20;
  manifest.nms.pre_nms_top_k = 300;
  manifest.nms.class_aware = true;
  const char* fields[] = {"decoder.class_selection", "nms.enabled", "nms.confidence_threshold",
                          "nms.iou_threshold", "nms.max_detections", "nms.pre_nms_top_k", "nms.class_aware"};
  for (const char* field : fields) mark_user(manifest, field);
}

vie::ModelManifest recognized(const std::string& path) {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  return vie::ManifestGenerator{}.generate(report, result, path);
}

vie::ImageInput load_bus(std::vector<std::uint8_t>& pixels) {
  std::ifstream image_file(vie_fixture("testdata/images/bus.ppm"), std::ios::binary);
  std::string magic;
  int width = 0;
  int height = 0;
  int max_value = 0;
  image_file >> magic >> width >> height >> max_value;
  image_file.get();
  EXPECT_EQ(magic, "P6");
  pixels.resize(static_cast<std::size_t>(width * height * 3));
  image_file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  EXPECT_TRUE(image_file);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

vie::SegmentationRuntimeConfig runtime_at(float confidence) {
  vie::SegmentationRuntimeConfig config;
  config.confidence_threshold = confidence;
  config.iou_threshold = 0.45f;
  config.class_selection = vie::ClassSelection::BestClass;
  config.class_aware_nms = true;
  config.pre_nms_top_k = 300;
  config.max_detections = 20;
  config.nms_enabled = true;
  config.retina_masks = true;
  config.binary_mask_gt_zero = true;
  config.max_mask_bytes = 64ull << 20;
  return config;
}

vie::ModelConfig detection_config(float confidence) {
  vie::ModelConfig config;
  config.model_path = vie_fixture("testdata/onnx/yolov8n.onnx");
  config.manifest = recognized(config.model_path);
  fill_preprocess(*config.manifest);
  fill_detection_policy(*config.manifest, confidence);
  config.device = vie::DeviceType::Cpu;
  return config;
}

vie::ModelConfig segmentation_config() {
  vie::ModelConfig config;
  config.model_path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  config.manifest = recognized(config.model_path);
  fill_preprocess(*config.manifest);
  config.device = vie::DeviceType::Cpu;
  return config;
}
#endif

}  // namespace

TEST(Integration, RegistryKeepsDetectionAndRejectsSameShapeIdentity) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx", "testdata/onnx/yolov8n.onnx");
  const vie::RecognitionResult detection =
      vie::default_recognizer_registry().recognize(vie::OnnxAnalyzer{}.analyze_file(
          vie_fixture("testdata/onnx/yolov8n.onnx")));
  EXPECT_EQ(detection.status, vie::RecognitionStatus::Recognized);
  ASSERT_NE(detection.selected(), nullptr);
  EXPECT_EQ(detection.selected()->decoder.protocol, "yolov8_detect_raw");

  const vie::RecognitionResult segmentation =
      vie::default_recognizer_registry().recognize(vie::OnnxAnalyzer{}.analyze_file(
          vie_fixture("testdata/onnx/yolov8n-seg.onnx")));
  EXPECT_EQ(segmentation.status, vie::RecognitionStatus::Recognized);
  ASSERT_NE(segmentation.selected(), nullptr);
  EXPECT_EQ(segmentation.selected()->decoder.protocol, "yolov8_seg_raw");

  const vie::RecognitionResult identity = vie::default_recognizer_registry().recognize(
      vie::OnnxAnalyzer{}.analyze_bytes(dual_identity().str(), "yolov8n-seg.onnx", "."));
  EXPECT_EQ(identity.status, vie::RecognitionStatus::Unknown);

  const vie::ModelManifest detection_manifest =
      vie::ManifestGenerator{}.generate(vie::OnnxAnalyzer{}.analyze_file(vie_fixture("testdata/onnx/yolov8n.onnx")),
                                        detection, "det.onnx");
  const std::string detection_json = detection_manifest.to_json_string();
  EXPECT_EQ(detection_json.find("\"segmentation\""), std::string::npos);
  const vie::ModelManifest detection_loaded = vie::ModelManifest::from_json_string(detection_json);
  EXPECT_EQ(detection_loaded.task, vie::TaskType::Detection);
  EXPECT_EQ(detection_loaded.decoder.protocol, "yolov8_detect_raw");
  EXPECT_FALSE(detection_loaded.segmentation.mask_channels.has_value());

  const vie::ModelManifest segmentation_manifest = vie::ManifestGenerator{}.generate(
      vie::OnnxAnalyzer{}.analyze_file(vie_fixture("testdata/onnx/yolov8n-seg.onnx")), segmentation, "seg.onnx");
  const vie::ModelManifest segmentation_loaded =
      vie::ModelManifest::from_json_string(segmentation_manifest.to_json_string());
  EXPECT_EQ(segmentation_loaded.task, vie::TaskType::Segmentation);
  EXPECT_EQ(segmentation_loaded.decoder.protocol, "yolov8_seg_raw");
  EXPECT_EQ(segmentation_loaded.outputs.size(), 2u);
  EXPECT_EQ(*segmentation_loaded.segmentation.mask_channels, 32);
  EXPECT_EQ(segmentation_loaded.segmentation.prototype_name, "output1");
  EXPECT_EQ(segmentation_loaded.segmentation.mask_coefficient_activation, "none");
  EXPECT_EQ(segmentation_loaded.segmentation.class_names.size(), 80u);
  EXPECT_FALSE(segmentation_loaded.segmentation.retina_masks.has_value());
}

TEST(Integration, FailedLoadDoesNotInventADeviceOrReplaceASession) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
  vie::InferenceEngine engine;
  EXPECT_FALSE(engine.loaded());
  EXPECT_EQ(engine.device().device(), vie::DeviceType::Cpu);

  vie::ModelConfig onnx;
  onnx.manifest = vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  onnx.manifest->model_format = vie::ModelFormat::Onnx;
  onnx.manifest->model_path = vie_fixture("testdata/onnx/yolov8n.onnx");
  onnx.model_path = onnx.manifest->model_path;
  onnx.device = vie::DeviceType::Cpu;
  onnx.manifest->provenance["model_format"] = vie::FieldProvenance{vie::Certainty::User, {"test"}};
  if (!engine.backends().contains(vie::BackendKind::OnnxRuntime)) {
    VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(onnx));
    EXPECT_FALSE(engine.loaded());
    EXPECT_EQ(engine.device().device(), vie::DeviceType::Cpu);
  }

  vie::ModelConfig cuda_request = onnx;
  cuda_request.device = vie::DeviceType::Cuda;
  if (!engine.backends().contains(vie::BackendKind::OnnxRuntimeCuda)) {
    VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(cuda_request));
    EXPECT_EQ(engine.device().device(), vie::DeviceType::Cpu);
  }

  vie::ModelConfig mock;
  mock.manifest = vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  engine.load(mock);
  EXPECT_TRUE(engine.loaded());
  EXPECT_EQ(engine.manifest().decoder.protocol, "mock");
  if (engine.backends().contains(vie::BackendKind::OnnxRuntimeCuda)) {
    VIE_EXPECT_ERROR(vie::ErrorCode::Internal, engine.load(cuda_request));
  } else {
    VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(cuda_request));
  }
  EXPECT_TRUE(engine.loaded());
  EXPECT_EQ(engine.manifest().decoder.protocol, "mock");
  EXPECT_EQ(engine.device().device(), vie::DeviceType::Cpu);

  vie::ModelConfig mock_cuda = mock;
  mock_cuda.device = vie::DeviceType::Cuda;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(mock_cuda));
  EXPECT_EQ(engine.manifest().model_format, vie::ModelFormat::Mock);
}

#if defined(VIE_HAS_ONNXRUNTIME)
TEST(Integration, CrossTasksStayOnTheirOwnProtocols) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx", "testdata/onnx/yolov8n.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);
  vie::InferenceEngine detector;
  detector.load(detection_config(0.25f));
  vie::InferenceEngine segmenter;
  segmenter.load(segmentation_config());
  EXPECT_EQ(detector.manifest().decoder.protocol, "yolov8_detect_raw");
  EXPECT_EQ(segmenter.manifest().decoder.protocol, "yolov8_seg_raw");
  EXPECT_EQ(detector.device().device(), vie::DeviceType::Cpu);
  EXPECT_EQ(segmenter.device().device(), vie::DeviceType::Cpu);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, detector.infer_segmentation(image, runtime_at(0.25f)));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, segmenter.infer(image));
  EXPECT_EQ(detector.manifest().decoder.protocol, "yolov8_detect_raw");
  EXPECT_EQ(segmenter.manifest().decoder.protocol, "yolov8_seg_raw");

  vie::ModelConfig forced = segmentation_config();
  forced.task = vie::TaskType::Detection;
  vie::InferenceEngine relabeled;
  relabeled.load(forced);
  EXPECT_EQ(relabeled.manifest().task, vie::TaskType::Detection);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, relabeled.infer_segmentation(image, runtime_at(0.25f)));
}

TEST(Integration, RuntimeConfigDoesNotLeakOrReadTheManifest) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx", "testdata/onnx/yolov8n.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);
  vie::InferenceEngine engine;
  vie::ModelConfig config = segmentation_config();
  config.manifest->nms.confidence_threshold = 0.01f;
  config.manifest->nms.iou_threshold = 0.10f;
  config.manifest->nms.enabled = true;
  config.manifest->segmentation.retina_masks = false;
  mark_user(*config.manifest, "nms.confidence_threshold");
  mark_user(*config.manifest, "nms.iou_threshold");
  mark_user(*config.manifest, "nms.enabled");
  mark_user(*config.manifest, "segmentation.retina_masks");
  engine.load(config);

  const vie::SegmentationBatch open = engine.infer_segmentation(image, runtime_at(0.25f));
  const vie::SegmentationBatch closed = engine.infer_segmentation(image, runtime_at(0.95f));
  const vie::SegmentationBatch open_again = engine.infer_segmentation(image, runtime_at(0.25f));
  ASSERT_FALSE(open.instances.empty());
  EXPECT_TRUE(closed.instances.empty());
  EXPECT_EQ(open_again.instances.size(), open.instances.size());
  EXPECT_FLOAT_EQ(*engine.manifest().nms.confidence_threshold, 0.01f);
  EXPECT_FLOAT_EQ(*engine.manifest().nms.iou_threshold, 0.10f);
  EXPECT_FALSE(*engine.manifest().segmentation.retina_masks);
  EXPECT_TRUE(engine.manifest().decoder.class_selection.empty());

  vie::SegmentationRuntimeConfig missing = runtime_at(0.25f);
  missing.iou_threshold.reset();
  try {
    engine.infer_segmentation(image, missing);
    FAIL();
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::UnresolvedField);
    EXPECT_NE(std::string(error.what()).find("iou_threshold"), std::string::npos);
  }
  vie::SegmentationRuntimeConfig illegal = runtime_at(0.25f);
  illegal.pre_nms_top_k = 5000;
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, engine.infer_segmentation(image, illegal));
  vie::SegmentationRuntimeConfig budget = runtime_at(0.25f);
  budget.max_mask_bytes = 1;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_segmentation(image, budget));

  vie::InferenceEngine loose;
  loose.load(detection_config(0.25f));
  vie::InferenceEngine strict;
  strict.load(detection_config(0.95f));
  const std::size_t loose_count = loose.infer(image).results[0].detections.size();
  const std::size_t strict_count = strict.infer(image).results[0].detections.size();
  EXPECT_GT(loose_count, strict_count);
  EXPECT_FLOAT_EQ(*loose.manifest().nms.confidence_threshold, 0.25f);
  EXPECT_EQ(loose.infer(image).results[0].detections.size(), loose_count);
}

TEST(Integration, ResultsOutliveTheEngineAndTheBackendTensors) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx", "testdata/onnx/yolov8n.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);
  vie::DetectionBatch detected;
  vie::SegmentationBatch segmented;
  {
    vie::InferenceEngine detector;
    detector.load(detection_config(0.25f));
    vie::InferenceEngine segmenter;
    segmenter.load(segmentation_config());
    detected = detector.infer(image);
    segmented = segmenter.infer_segmentation(image, runtime_at(0.25f));
    vie::BackendRequest request;
    request.inputs.emplace("images", segmenter.preprocess(image).output);
    vie::BackendOutput temporary = segmenter.execute(request);
    temporary.tensors.clear();
  }
  ASSERT_FALSE(detected.results.empty());
  ASSERT_FALSE(detected.results[0].detections.empty());
  EXPECT_GE(detected.results[0].detections[0].confidence, 0.25f);
  ASSERT_FALSE(segmented.instances.empty());
  vie::Mask copy = segmented.instances[0].mask;
  EXPECT_TRUE(copy.tensor().shares_storage(segmented.instances[0].mask.tensor()));
  vie::Mask moved = std::move(copy);
  EXPECT_TRUE(moved.tensor().shares_storage(segmented.instances[0].mask.tensor()));
  const std::uint8_t* bytes = moved.binary_data();
  ASSERT_NE(bytes, nullptr);
  EXPECT_TRUE(bytes[0] == 0 || bytes[0] == 1);
}

#if defined(NDEBUG) && !defined(VIE_HAS_ONNXRUNTIME_CUDA)
TEST(IntegrationBenchmark, CpuDetectionAndSegmentation) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx", "testdata/onnx/yolov8n.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);
  auto bench = [&](const char* label, auto&& run) {
    const auto load_start = std::chrono::steady_clock::now();
    run(true);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    vie::DetectionInferTiming first_detection{};
    vie::SegmentationInferTiming first_segmentation{};
    run(false, &first_detection, &first_segmentation);
    constexpr int kRuns = 100;
    std::vector<double> preprocess;
    std::vector<double> execute;
    std::vector<double> decode;
    std::vector<double> total;
    preprocess.reserve(kRuns);
    for (int i = 0; i < kRuns; ++i) {
      vie::DetectionInferTiming detection{};
      vie::SegmentationInferTiming segmentation{};
      run(false, &detection, &segmentation);
      const bool seg = std::string(label).find("seg") != std::string::npos;
      preprocess.push_back(seg ? segmentation.preprocess_ms : detection.preprocess_ms);
      execute.push_back(seg ? segmentation.execute_ms : detection.execute_ms);
      decode.push_back(seg ? segmentation.decode_ms : detection.decode_ms);
      total.push_back(seg ? segmentation.total_ms : detection.total_ms);
    }
    auto stats = [](std::vector<double> values) {
      std::sort(values.begin(), values.end());
      double sum = 0;
      for (const double value : values) sum += value;
      const std::size_t p95 = std::min(values.size() - 1, static_cast<std::size_t>(0.95 * values.size()));
      return std::make_tuple(sum / static_cast<double>(values.size()), values[values.size() / 2], values[p95],
                             values.front(), values.back());
    };
    const auto prep = stats(preprocess);
    const auto ran = stats(execute);
    const auto decoded = stats(decode);
    const auto all = stats(total);
    const bool seg = std::string(label).find("seg") != std::string::npos;
    std::ofstream out(std::string("/tmp/vie_phase433_step7_") + label + ".txt");
    out << label << " image=810x1080 model=" << (seg ? "yolov8n-seg.onnx" : "yolov8n.onnx")
        << " ep=cpu includes_h2d=0 includes_sync=0 includes_allocation=1 same_ep_outputs=1 runs=100\n";
    out << "load_ms=" << load_ms << "\n";
    out << "first preprocess=" << (seg ? first_segmentation.preprocess_ms : first_detection.preprocess_ms)
        << " execute=" << (seg ? first_segmentation.execute_ms : first_detection.execute_ms)
        << " decode=" << (seg ? first_segmentation.decode_ms : first_detection.decode_ms)
        << " total=" << (seg ? first_segmentation.total_ms : first_detection.total_ms) << "\n";
    out << "steady preprocess mean/p50/p95/min/max " << std::get<0>(prep) << " " << std::get<1>(prep) << " "
        << std::get<2>(prep) << " " << std::get<3>(prep) << " " << std::get<4>(prep) << "\n";
    out << "steady execute mean/p50/p95/min/max " << std::get<0>(ran) << " " << std::get<1>(ran) << " "
        << std::get<2>(ran) << " " << std::get<3>(ran) << " " << std::get<4>(ran) << "\n";
    out << "steady decode mean/p50/p95/min/max " << std::get<0>(decoded) << " " << std::get<1>(decoded) << " "
        << std::get<2>(decoded) << " " << std::get<3>(decoded) << " " << std::get<4>(decoded) << "\n";
    out << "steady e2e mean/p50/p95/min/max " << std::get<0>(all) << " " << std::get<1>(all) << " "
        << std::get<2>(all) << " " << std::get<3>(all) << " " << std::get<4>(all) << "\n";
    (void)first_detection;
    (void)first_segmentation;
  };

  vie::InferenceEngine detector;
  const vie::ModelConfig det = detection_config(0.25f);
  bench("cpu_det", [&](bool loading, vie::DetectionInferTiming* detection = nullptr,
                       vie::SegmentationInferTiming* = nullptr) {
    if (loading) {
      detector.load(det);
      return;
    }
    const vie::DetectionBatch batch = detector.infer(image, detection);
    ASSERT_FALSE(batch.results[0].detections.empty());
  });

  vie::InferenceEngine segmenter;
  const vie::ModelConfig seg = segmentation_config();
  bench("cpu_seg", [&](bool loading, vie::DetectionInferTiming* = nullptr,
                       vie::SegmentationInferTiming* segmentation = nullptr) {
    if (loading) {
      segmenter.load(seg);
      return;
    }
    const vie::SegmentationBatch batch = segmenter.infer_segmentation(image, runtime_at(0.25f), segmentation);
    ASSERT_EQ(batch.instances.size(), 6u);
    EXPECT_EQ(batch.instances[0].mask.width(), image.width);
  });
  {
    const vie::DetectionBatch batch = detector.infer(image);
    std::ofstream note("/tmp/vie_phase433_step7_cpu_det.txt", std::ios::app);
    note << "instances=" << batch.results[0].detections.size() << " mask=none\n";
  }
  {
    const vie::SegmentationBatch batch = segmenter.infer_segmentation(image, runtime_at(0.25f));
    std::ofstream note("/tmp/vie_phase433_step7_cpu_seg.txt", std::ios::app);
    note << "instances=" << batch.instances.size() << " mask=" << batch.instances[0].mask.width() << "x"
         << batch.instances[0].mask.height() << "\n";
  }
}
#endif
#endif

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
TEST(Integration, CudaBackendSelectionAndMaskLifetime) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx", "testdata/onnx/yolov8n.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput cpu_image = load_bus(pixels);
  vie::ModelConfig detection = detection_config(0.25f);
  detection.device = vie::DeviceType::Cuda;
  detection.backend = vie::BackendKind::OnnxRuntimeCuda;
  vie::InferenceEngine detector;
  try {
    detector.load(detection);
    FAIL() << "strict CUDA EP accepted yolov8n.onnx";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::Internal) << error.what();
    EXPECT_NE(std::string(error.what()).find("CPU EP"), std::string::npos) << error.what();
  }
  EXPECT_FALSE(detector.loaded());
  EXPECT_EQ(detector.device().device(), vie::DeviceType::Cpu);

  std::size_t free_before = 0;
  std::size_t total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);
  std::uint8_t* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_pixels), pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput device_image = cpu_image;
  device_image.data = device_pixels;
  device_image.device = vie::DeviceType::Cuda;
  device_image.device_id = 0;

  vie::SegmentationBatch kept;
  {
    vie::InferenceEngine segmenter;
    vie::ModelConfig segmentation = segmentation_config();
    segmentation.device = vie::DeviceType::Cuda;
    segmentation.backend = vie::BackendKind::OnnxRuntimeCuda;
    segmenter.load(segmentation);
    EXPECT_EQ(segmenter.device().device(), vie::DeviceType::Cuda);
    VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                     segmenter.infer_segmentation(cpu_image, runtime_at(0.25f)));
    kept = segmenter.infer_segmentation(device_image, runtime_at(0.25f));
    EXPECT_TRUE(segmenter.infer_segmentation(device_image, runtime_at(0.95f)).instances.empty());
  }
  cudaFree(device_pixels);
  ASSERT_FALSE(kept.instances.empty());
  EXPECT_EQ(kept.instances[0].mask.device(), vie::DeviceType::Cuda);
  const vie::Mask host = kept.instances[0].mask.to_cpu();
  EXPECT_TRUE(host.binary_data()[0] == 0 || host.binary_data()[0] == 1);

  device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_pixels), pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  device_image.data = device_pixels;
  vie::InferenceEngine segmenter;
  vie::ModelConfig segmentation = segmentation_config();
  segmentation.device = vie::DeviceType::Cuda;
  segmentation.backend = vie::BackendKind::OnnxRuntimeCuda;
  segmenter.load(segmentation);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::size_t free_after_load = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_after_load, &total), cudaSuccess);
  std::size_t free_while_held = 0;
  {
    vie::SegmentationBatch held = segmenter.infer_segmentation(device_image, runtime_at(0.25f));
    ASSERT_EQ(held.instances.size(), kept.instances.size());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemGetInfo(&free_while_held, &total), cudaSuccess);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::size_t free_after_release = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_after_release, &total), cudaSuccess);
  std::size_t free_after_10 = free_after_release;
  std::size_t free_after_100 = 0;
  for (int i = 0; i < 100; ++i) {
    {
      vie::SegmentationBatch batch = segmenter.infer_segmentation(device_image, runtime_at(0.25f));
      ASSERT_EQ(batch.instances.size(), kept.instances.size());
    }
    if (i == 9 || i == 99) {
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ASSERT_EQ(cudaMemGetInfo(i == 9 ? &free_after_10 : &free_after_100, &total), cudaSuccess);
    }
  }
  cudaFree(device_pixels);
  const auto drop = [](std::size_t before, std::size_t after) {
    return before > after ? before - after : 0;
  };
  std::ofstream out("/tmp/vie_phase433_step7_vram.txt");
  out << "free_before=" << free_before << " free_after_load=" << free_after_load
      << " free_while_result_held=" << free_while_held << " free_after_result_released=" << free_after_release
      << " free_after_10_released=" << free_after_10 << " free_after_100_released=" << free_after_100
      << " first_inference_cache=" << drop(free_after_load, free_after_release)
      << " growth_10_to_100=" << drop(free_after_10, free_after_100)
      << " kept_masks_resident=1\n";
  EXPECT_LT(drop(free_after_10, free_after_100), 8ull << 20);
}

#if defined(NDEBUG)
TEST(IntegrationBenchmark, CudaSegmentation) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput cpu_image = load_bus(pixels);
  std::uint8_t* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_pixels), pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput image = cpu_image;
  image.data = device_pixels;
  image.device = vie::DeviceType::Cuda;
  image.device_id = 0;
  vie::InferenceEngine engine;
  vie::ModelConfig config = segmentation_config();
  config.device = vie::DeviceType::Cuda;
  config.backend = vie::BackendKind::OnnxRuntimeCuda;
  const auto load_start = std::chrono::steady_clock::now();
  engine.load(config);
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
  vie::SegmentationInferTiming first;
  const vie::SegmentationBatch first_batch = engine.infer_segmentation(image, runtime_at(0.25f), &first);
  ASSERT_EQ(first_batch.instances.size(), 6u);
  constexpr int kRuns = 100;
  std::vector<double> preprocess(kRuns);
  std::vector<double> execute(kRuns);
  std::vector<double> decode(kRuns);
  std::vector<double> total(kRuns);
  for (int i = 0; i < kRuns; ++i) {
    vie::SegmentationInferTiming timing;
    const vie::SegmentationBatch batch = engine.infer_segmentation(image, runtime_at(0.25f), &timing);
    ASSERT_EQ(batch.instances.size(), 6u);
    preprocess[static_cast<std::size_t>(i)] = timing.preprocess_ms;
    execute[static_cast<std::size_t>(i)] = timing.execute_ms;
    decode[static_cast<std::size_t>(i)] = timing.decode_ms;
    total[static_cast<std::size_t>(i)] = timing.total_ms;
  }
  cudaFree(device_pixels);
  auto write = [](std::ostream& out, const char* name, std::vector<double> values) {
    std::sort(values.begin(), values.end());
    double sum = 0;
    for (const double value : values) sum += value;
    const std::size_t p95 = std::min(values.size() - 1, static_cast<std::size_t>(0.95 * values.size()));
    out << name << " mean/p50/p95/min/max " << sum / static_cast<double>(values.size()) << " "
        << values[values.size() / 2] << " " << values[p95] << " " << values.front() << " " << values.back()
        << "\n";
  };
  std::ofstream out("/tmp/vie_phase433_step7_cuda_seg.txt");
  out << "cuda_seg image=810x1080 ep=cuda_strict instances=6 mask=810x1080 includes_h2d=0 includes_mask_d2h=0 "
         "includes_sync=1 includes_allocation=1 same_ep_outputs=0 runs=100\n";
  out << "load_ms=" << load_ms << "\n";
  out << "first preprocess=" << first.preprocess_ms << " execute=" << first.execute_ms
      << " decode=" << first.decode_ms << " total=" << first.total_ms << "\n";
  write(out, "steady preprocess", preprocess);
  write(out, "steady execute", execute);
  write(out, "steady decode", decode);
  write(out, "steady e2e", total);
}
#endif
#endif
