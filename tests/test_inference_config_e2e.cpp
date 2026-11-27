// Phase 12: real-model end-to-end tests for the unified inference config.
//
// Every test here runs the real ONNX model through the real ONNX Runtime
// backend, preprocessor and decoder; nothing is mocked. The available fixtures
// cover detection (yolov8n), instance segmentation (yolov8n-seg), RT-DETR
// (rtdetr-l) and semantic segmentation (deeplabv3). This repository has no
// exportable classification, OBB, or pose fixture, so those three tasks are
// covered by the resolver / decoder / engine tests and are explicitly reported
// as not end-to-end verified.

#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/config/inference_config.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/runtime/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)

namespace {

using vie::InferenceOptions;
using vie::ModelManifest;
using vie::ParameterSource;
using vie::ResolvedInferenceConfig;
using vie::TaskType;

void mark_user(ModelManifest& manifest, const char* field, const char* evidence) {
  manifest.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {evidence}};
}

ModelManifest recognized(const std::string& path) {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  return vie::ManifestGenerator{}.generate(report, result, path);
}

// Fills the preprocess contract the graph cannot prove. resize is "letterbox"
// for the YOLO exports and "stretch" for RT-DETR / DeepLabV3.
void fill_preprocess(ModelManifest& manifest, const char* resize) {
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = resize;
  manifest.input.interpolation = "bilinear";
  if (std::string(resize) == "letterbox") {
    manifest.input.padding_value = 114.f;
    mark_user(manifest, "input.padding_value", "Ultralytics letterbox fill, supplied by the caller");
  }
  mark_user(manifest, "input.color", "BGR->RGB conversion, supplied by the caller");
  mark_user(manifest, "input.normalization", "divide by 255, supplied by the caller");
  mark_user(manifest, "input.resize", "export contract, supplied by the caller");
  mark_user(manifest, "input.interpolation", "bilinear resize, supplied by the caller");
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
  pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
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

vie::ModelConfig detection_config() {
  const std::string path = vie_fixture("testdata/onnx/yolov8n.onnx");
  vie::ModelConfig config;
  config.model_path = path;
  config.manifest = recognized(path);
  fill_preprocess(*config.manifest, "letterbox");
  config.device = vie::DeviceType::Cpu;
  return config;
}

vie::ModelConfig segmentation_config() {
  const std::string path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  vie::ModelConfig config;
  config.model_path = path;
  config.manifest = recognized(path);
  fill_preprocess(*config.manifest, "letterbox");
  // The real model needs a mask output shape and budget; the graph cannot
  // prove either, so they are model-level defaults here.
  config.inference_options.segmentation.retina_masks = true;
  config.inference_options.segmentation.binary_mask_gt_zero = true;
  config.inference_options.segmentation.max_mask_bytes = 128ull << 20;
  config.device = vie::DeviceType::Cpu;
  return config;
}

vie::ModelConfig rt_detr_config() {
  const std::string path = vie_fixture("testdata/onnx/rtdetr-l.onnx");
  vie::ModelConfig config;
  config.model_path = path;
  config.manifest = recognized(path);
  fill_preprocess(*config.manifest, "stretch");
  ModelManifest& manifest = *config.manifest;
  // The generated manifest keeps the graph's symbolic output shape; complete it
  // with the runtime-observed profile, exactly like the RT-DETR E2E test does.
  manifest.outputs[0].shape = {1, static_cast<std::int64_t>(*manifest.decoder.num_queries),
                               static_cast<std::int64_t>(4 + *manifest.num_classes)};
  mark_user(manifest, "outputs.0.shape", "runtime profile observed from execution");
  manifest.decoder.class_selection = "best_class";
  mark_user(manifest, "decoder.class_selection", "official postprocess keeps the best class per query");
  // Manifest-level policy: these are caller values, not model observations.
  manifest.nms.confidence_threshold = 0.25f;
  mark_user(manifest, "nms.confidence_threshold", "caller threshold, Ultralytics predict default");
  manifest.nms.max_detections = 300;
  mark_user(manifest, "nms.max_detections", "caller cap matching Ultralytics max_det");
  config.device = vie::DeviceType::Cpu;
  return config;
}

vie::ModelConfig semantic_config() {
  const std::string path = vie_fixture("testdata/onnx/deeplabv3_mobilenet_v3_large_520_opset18.onnx");
  vie::ModelConfig config;
  config.model_path = path;
  config.manifest = recognized(path);
  fill_preprocess(*config.manifest, "stretch");
  config.device = vie::DeviceType::Cpu;
  return config;
}

}  // namespace

TEST(InferenceConfigE2E, RealYolov8DetectionConfigLayers) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);

  vie::InferenceEngine engine;
  engine.load(detection_config());

  // No nms block in the generated manifest: every YOLO post-processing knob
  // comes from the framework defaults, with the source recorded.
  const ResolvedInferenceConfig defaults = engine.resolve_inference_config();
  EXPECT_EQ(defaults.task, TaskType::Detection);
  EXPECT_EQ(defaults.protocol, "yolov8_detect_raw");
  const vie::ResolvedDetectionConfig& detection = defaults.require_detection();
  EXPECT_FLOAT_EQ(detection.confidence_threshold.value, 0.25f);
  EXPECT_EQ(detection.confidence_threshold.source, ParameterSource::FrameworkDefault);
  EXPECT_EQ(detection.max_detections.value, 300);
  EXPECT_EQ(detection.max_detections.source, ParameterSource::FrameworkDefault);

  const vie::DetectionBatch baseline = engine.infer(image);
  const vie::DetectionBatch empty_options = engine.infer(image, InferenceOptions{});
  ASSERT_EQ(baseline.results.size(), 1u);
  ASSERT_FALSE(baseline.results[0].detections.empty());
  EXPECT_EQ(empty_options.results[0].detections.size(), baseline.results[0].detections.size());
  for (const vie::Detection& kept : baseline.results[0].detections) {
    EXPECT_GE(kept.confidence, 0.25f);
  }
  const std::size_t baseline_count = baseline.results[0].detections.size();

  // A stricter request threshold removes every detection.
  InferenceOptions strict;
  strict.detection.confidence_threshold = 0.95f;
  EXPECT_TRUE(engine.infer(image, strict).results[0].detections.empty());

  // A request-level max_detections cap is honored after NMS.
  InferenceOptions capped;
  capped.detection.max_detections = 2;
  const vie::DetectionBatch limited = engine.infer(image, capped);
  EXPECT_EQ(limited.results[0].detections.size(), std::min<std::size_t>(2, baseline_count));

  // A lower request threshold cannot remove detections that already survived
  // the default threshold (NMS processes the higher scores first).
  InferenceOptions loose;
  loose.detection.confidence_threshold = 0.10f;
  EXPECT_GE(engine.infer(image, loose).results[0].detections.size(), baseline_count);

  // Manifest-provided policy is used when no request overrides it, and the
  // request still wins over the manifest.
  vie::ModelConfig from_manifest = detection_config();
  from_manifest.manifest->nms.enabled = true;
  from_manifest.manifest->nms.confidence_threshold = 0.5f;
  from_manifest.manifest->nms.iou_threshold = 0.45f;
  from_manifest.manifest->nms.max_detections = 300;
  from_manifest.manifest->nms.pre_nms_top_k = 4096;
  from_manifest.manifest->nms.class_aware = true;
  for (const char* field : {"nms.enabled", "nms.confidence_threshold", "nms.iou_threshold",
                            "nms.max_detections", "nms.pre_nms_top_k", "nms.class_aware"}) {
    mark_user(*from_manifest.manifest, field, "explicit test configuration");
  }
  vie::InferenceEngine manifest_engine;
  manifest_engine.load(from_manifest);
  const ResolvedInferenceConfig manifest_resolved = manifest_engine.resolve_inference_config();
  EXPECT_FLOAT_EQ(manifest_resolved.require_detection().confidence_threshold.value, 0.5f);
  EXPECT_EQ(manifest_resolved.require_detection().confidence_threshold.source, ParameterSource::Manifest);
  const vie::DetectionBatch manifest_batch = manifest_engine.infer(image);
  EXPECT_LE(manifest_batch.results[0].detections.size(), baseline_count);
  for (const vie::Detection& kept : manifest_batch.results[0].detections) {
    EXPECT_GE(kept.confidence, 0.5f);
  }
  InferenceOptions override_manifest;
  override_manifest.detection.confidence_threshold = 0.95f;
  EXPECT_TRUE(manifest_engine.infer(image, override_manifest).results[0].detections.empty());

  // Invalid values are rejected before the engine touches the backend.
  InferenceOptions illegal;
  illegal.detection.iou_threshold = 1.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer(image, illegal));
}

TEST(InferenceConfigE2E, RuntimeRealDetectionSnapshot) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);

  vie::RuntimeConfig config;
  config.request_queue_capacity = 16;
  config.inference_worker_count = 1;
  config.batch_aggregation.max_batch_size = 1;
  config.batch_aggregation.max_wait_time = std::chrono::milliseconds(0);
  vie::Runtime runtime(config);
  runtime.models().load("det", detection_config());

  const vie::DetectionResult defaults = runtime.submit("det", image).get();
  ASSERT_FALSE(defaults.detections.empty());

  InferenceOptions strict;
  strict.detection.confidence_threshold = 0.95f;
  EXPECT_TRUE(runtime.submit("det", image, strict).get().detections.empty());
}

TEST(InferenceConfigE2E, RealYolov8SegmentationConfigLayers) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);

  vie::InferenceEngine engine;
  engine.load(segmentation_config());

  // Confidence comes from the framework default; the mask policy comes from
  // the model-level defaults and keeps its own provenance.
  const ResolvedInferenceConfig resolved = engine.resolve_inference_config();
  EXPECT_EQ(resolved.task, TaskType::Segmentation);
  const vie::ResolvedSegmentationConfig& segmentation = resolved.require_segmentation();
  EXPECT_FLOAT_EQ(segmentation.confidence_threshold.value, 0.25f);
  EXPECT_EQ(segmentation.confidence_threshold.source, ParameterSource::FrameworkDefault);
  ASSERT_TRUE(segmentation.retina_masks.has_value());
  EXPECT_TRUE(segmentation.retina_masks->value);
  EXPECT_EQ(segmentation.retina_masks->source, ParameterSource::ModelDefault);
  ASSERT_TRUE(segmentation.max_mask_bytes.has_value());
  EXPECT_EQ(segmentation.max_mask_bytes->source, ParameterSource::ModelDefault);

  const vie::SegmentationBatch baseline = engine.infer_segmentation(image, InferenceOptions{});
  ASSERT_FALSE(baseline.instances.empty());
  const std::size_t baseline_count = baseline.instances.size();
  for (const vie::InstanceSegmentation& instance : baseline.instances) {
    EXPECT_GE(instance.detection.confidence, 0.25f);
    // retina_masks=true restores the original image size.
    EXPECT_EQ(instance.mask.width(), image.width);
    EXPECT_EQ(instance.mask.height(), image.height);
  }

  // A stricter request threshold removes every instance.
  InferenceOptions strict;
  strict.segmentation.confidence_threshold = 0.95f;
  EXPECT_TRUE(engine.infer_segmentation(image, strict).instances.empty());

  // Request-level max_detections caps the instance count.
  InferenceOptions capped;
  capped.segmentation.max_detections = 1;
  EXPECT_EQ(engine.infer_segmentation(image, capped).instances.size(), std::min<std::size_t>(1, baseline_count));

  // Request-level retina_masks=false switches the mask grid to the model input.
  InferenceOptions model_grid;
  model_grid.segmentation.retina_masks = false;
  const vie::SegmentationBatch non_retina = engine.infer_segmentation(image, model_grid);
  ASSERT_FALSE(non_retina.instances.empty());
  EXPECT_EQ(non_retina.instances[0].mask.width(), 640);
  EXPECT_EQ(non_retina.instances[0].mask.height(), 640);
  // binary_mask_gt_zero=true keeps the masks binary.
  const std::uint8_t* mask_bytes = non_retina.instances[0].mask.binary_data();
  ASSERT_NE(mask_bytes, nullptr);
  ASSERT_GT(non_retina.instances[0].mask.width() * non_retina.instances[0].mask.height(), 0);
  EXPECT_TRUE(mask_bytes[0] == 0 || mask_bytes[0] == 1);

  // Detection knobs cannot be smuggled into a segmentation request.
  InferenceOptions wrong;
  wrong.detection.confidence_threshold = 0.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_segmentation(image, wrong));
}

TEST(InferenceConfigE2E, RealRtDetrKeepsItsOwnSemantics) {
  VIE_REQUIRE_MODEL("testdata/onnx/rtdetr-l.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);

  vie::InferenceEngine engine;
  engine.load(rt_detr_config());

  const ResolvedInferenceConfig resolved = engine.resolve_inference_config();
  EXPECT_EQ(resolved.protocol, "rtdetr_detection_raw");
  EXPECT_FLOAT_EQ(resolved.require_detection().confidence_threshold.value, 0.25f);
  EXPECT_EQ(resolved.require_detection().confidence_threshold.source, ParameterSource::Manifest);
  EXPECT_FALSE(resolved.require_detection().nms_enabled);

  const vie::DetectionBatch baseline = engine.infer(image);
  ASSERT_FALSE(baseline.results[0].detections.empty());
  EXPECT_FALSE(baseline.results[0].nms_applied);
  const std::size_t baseline_count = baseline.results[0].detections.size();
  for (const vie::Detection& kept : baseline.results[0].detections) {
    EXPECT_GE(kept.confidence, 0.25f);
  }

  // RT-DETR keeps its own Top-K / max detection cap; the request cannot inject
  // YOLO NMS knobs.
  InferenceOptions capped;
  capped.detection.max_detections = 3;
  EXPECT_EQ(engine.infer(image, capped).results[0].detections.size(), std::min<std::size_t>(3, baseline_count));

  InferenceOptions high;
  high.detection.confidence_threshold = 0.999f;
  EXPECT_TRUE(engine.infer(image, high).results[0].detections.empty());

  InferenceOptions iou;
  iou.detection.iou_threshold = 0.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image, iou));
  InferenceOptions pre_nms;
  pre_nms.detection.pre_nms_top_k = 100;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image, pre_nms));
  InferenceOptions agnostic;
  agnostic.detection.class_agnostic_nms = true;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image, agnostic));
  InferenceOptions multi_label;
  multi_label.detection.multi_label = true;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image, multi_label));
}

TEST(InferenceConfigE2E, RealSemanticSegmentationHasNoDetectionKnobs) {
  VIE_REQUIRE_MODEL("testdata/onnx/deeplabv3_mobilenet_v3_large_520_opset18.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);

  vie::InferenceEngine engine;
  engine.load(semantic_config());

  const ResolvedInferenceConfig resolved = engine.resolve_inference_config();
  EXPECT_EQ(resolved.task, TaskType::SemanticSegmentation);
  EXPECT_FALSE(resolved.detection.has_value());
  EXPECT_FALSE(resolved.segmentation.has_value());
  EXPECT_FALSE(resolved.obb.has_value());
  EXPECT_FALSE(resolved.pose.has_value());
  EXPECT_FALSE(resolved.classification.has_value());

  const vie::SemanticSegmentationResult result = engine.infer_semantic_segmentation(image);
  EXPECT_EQ(result.width, image.width);
  EXPECT_EQ(result.height, image.height);
  EXPECT_EQ(result.num_classes, 21);
  ASSERT_EQ(result.class_ids.size(), static_cast<std::size_t>(image.width) *
                                          static_cast<std::size_t>(image.height));
  for (const std::int32_t id : result.class_ids) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, 21);
  }

  // A detection request cannot resolve against a real semantic-segmentation
  // engine: its block is rejected instead of being ignored.
  InferenceOptions detection_block;
  detection_block.detection.confidence_threshold = 0.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer(image, detection_block));
}

#endif  // VIE_HAS_ONNXRUNTIME