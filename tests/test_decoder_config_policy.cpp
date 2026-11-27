// Phase 12: decoder configuration policy tests.
//
// Covers:
// - ResolvedInferenceConfig -> DecoderConfig mapping for YOLOv8 and RT-DETR
// - the real YOLOv8 CPU decoder honoring confidence / IoU / max-detections /
//   pre-NMS top-k / class-agnostic NMS / multi-label from resolved configs
// - CPU and CUDA decoders receiving the same DecoderConfig and producing the
//   same detections within an explicit tolerance
// - engine-level option plumbing for detection, instance segmentation,
//   classification, OBB, and pose through the real decoders with the mock
//   backend (synthetic model output, never a mock decoder)

#include "test_util.hpp"

#include "vie/config/inference_config.hpp"
#include "vie/decoder/yolov8_detect_decoder.hpp"
#include "vie/decoder/yolov8_obb_decoder.hpp"
#include "vie/engine/inference_engine.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using vie::InferenceOptions;
using vie::ModelManifest;
using vie::ParameterSource;
using vie::ResolvedInferenceConfig;
using vie::TaskType;

namespace {

ModelManifest detection_manifest(const std::string& protocol = "yolov8_detect_raw",
                                 int num_classes = 2) {
  ModelManifest manifest;
  manifest.model_name = "policy_det";
  manifest.task = TaskType::Detection;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = num_classes;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  manifest.decoder.protocol = protocol;
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.class_selection = "best_class";
  manifest.decoder.scores_are_probabilities = true;
  if (protocol == "rtdetr_detection_raw") {
    manifest.decoder.box_format = "cxcywh_normalized";
    manifest.decoder.box_decode = "decoded_normalized_cxcywh";
    manifest.decoder.channel_axis = 2;
    manifest.decoder.num_queries = 300;
  }
  vie::OutputDesc prediction;
  prediction.name = "predictions";
  prediction.shape = protocol == "rtdetr_detection_raw" ? std::vector<std::int64_t>{1, 300, 4 + num_classes}
                                                        : std::vector<std::int64_t>{1, 4 + num_classes, 1};
  prediction.dtype = vie::DataType::Fp32;
  prediction.semantic = "predictions";
  manifest.outputs = {prediction};
  manifest.ensure_provenance();
  return manifest;
}

ModelManifest segmentation_manifest() {
  ModelManifest manifest;
  manifest.model_name = "policy_seg";
  manifest.task = TaskType::Segmentation;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 80;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  manifest.input.interpolation = "bilinear";
  manifest.decoder.protocol = "yolov8_seg_raw";
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.scores_are_probabilities = true;
  vie::OutputDesc prediction;
  prediction.name = "output0";
  prediction.shape = {1, 116, 4};
  prediction.dtype = vie::DataType::Fp32;
  prediction.semantic = "cxcywh_class_probabilities_mask_coefficients";
  vie::OutputDesc prototype;
  prototype.name = "output1";
  prototype.shape = {1, 32, 2, 2};
  prototype.dtype = vie::DataType::Fp32;
  prototype.layout = vie::Layout::Nchw;
  prototype.semantic = "mask_prototypes";
  manifest.outputs = {prediction, prototype};
  manifest.segmentation.mask_channels = 32;
  manifest.segmentation.prototype_name = "output1";
  manifest.segmentation.prototype_layout = vie::Layout::Nchw;
  manifest.segmentation.prototype_channel_axis = 1;
  manifest.segmentation.prototype_height = 2;
  manifest.segmentation.prototype_width = 2;
  manifest.segmentation.mask_coefficient_activation = "none";
  manifest.segmentation.prototype_activation = "model_output_features";
  manifest.segmentation.has_objectness = false;
  manifest.ensure_provenance();
  return manifest;
}

ModelManifest classification_manifest(int classes) {
  ModelManifest manifest;
  manifest.model_name = "policy_cls";
  manifest.task = TaskType::Classification;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = classes;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  manifest.input.interpolation = "bilinear";
  manifest.decoder.protocol = "classification_logits_raw";
  vie::OutputDesc logits;
  logits.name = "logits";
  logits.shape = {1, classes};
  logits.dtype = vie::DataType::Fp32;
  manifest.outputs = {logits};
  manifest.classification.output_name = "logits";
  manifest.classification.class_axis = 1;
  manifest.classification.score_type = "logits";
  manifest.ensure_provenance();
  return manifest;
}

ModelManifest obb_manifest(int num_classes = 2) {
  ModelManifest manifest;
  manifest.model_name = "policy_obb";
  manifest.task = TaskType::Obb;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = num_classes;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  manifest.decoder.protocol = "yolov8_obb_raw";
  manifest.decoder.box_format = "cxcywh_angle";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.scores_are_probabilities = true;
  vie::OutputDesc output;
  output.name = "output0";
  output.shape = {1, 4 + num_classes + 1, 1};
  output.dtype = vie::DataType::Fp32;
  manifest.outputs = {output};
  manifest.obb.output_name = "output0";
  manifest.obb.channel_axis = 1;
  manifest.obb.angle_unit = "radian";
  manifest.obb.angle_range = "[-pi/4,3pi/4]";
  manifest.obb.score_type = "probabilities";
  manifest.ensure_provenance();
  return manifest;
}

ModelManifest pose_manifest(int keypoints) {
  ModelManifest manifest;
  manifest.model_name = "policy_pose";
  manifest.task = TaskType::Pose;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 1;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  manifest.decoder.protocol = "yolov8_pose_raw";
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.scores_are_probabilities = true;
  vie::OutputDesc output;
  output.name = "output0";
  output.shape = {1, 4 + 1 + keypoints * 3, 1};
  output.dtype = vie::DataType::Fp32;
  manifest.outputs = {output};
  manifest.pose.output_name = "output0";
  manifest.pose.channel_axis = 1;
  manifest.pose.keypoint_count = keypoints;
  manifest.pose.keypoint_dimensions = 3;
  manifest.pose.keypoint_layout = "xy_confidence";
  manifest.pose.score_type = "probabilities";
  manifest.ensure_provenance();
  return manifest;
}

vie::ModelConfig model_config(ModelManifest manifest, const InferenceOptions& defaults = {}) {
  vie::ModelConfig config;
  config.manifest = std::move(manifest);
  config.backend = vie::BackendKind::Auto;
  config.device = vie::DeviceType::Cpu;
  config.inference_options = defaults;
  return config;
}

vie::ImageInput tiny_image(std::vector<std::uint8_t>& pixels) {
  pixels.assign(8 * 8 * 3, 9);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = 8;
  image.height = 8;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

vie::GeometryTransform identity(int width, int height) {
  vie::GeometryTransform geometry;
  geometry.original_width = width;
  geometry.original_height = height;
  geometry.resized_width = width;
  geometry.resized_height = height;
  geometry.output_width = width;
  geometry.output_height = height;
  geometry.scale_x = 1;
  geometry.scale_y = 1;
  geometry.uniform_scale = 1;
  return geometry;
}

// Four anchors, two classes, channel-major [1, 6, 4]:
//   A: (10,10,4,4) class0 0.9, class1 0.7
//   B: (10.5,10,4,4) class0 0.8, class1 0.1  (same class as A, IoU ~0.78)
//   C: (10.5,10,4,4) class0 0.1, class1 0.85 (other class, overlaps A)
//   D: (40,40,4,4) class0 0.6, class1 0.2   (isolated)
std::vector<float> four_anchor_values() {
  constexpr int kAnchors = 4;
  std::vector<float> values(6 * kAnchors, 0.f);
  const float cx[kAnchors] = {10.f, 10.5f, 10.5f, 40.f};
  const float cy[kAnchors] = {10.f, 10.f, 10.f, 40.f};
  const float w[kAnchors] = {4.f, 4.f, 4.f, 4.f};
  const float h[kAnchors] = {4.f, 4.f, 4.f, 4.f};
  const float c0[kAnchors] = {0.9f, 0.8f, 0.1f, 0.6f};
  const float c1[kAnchors] = {0.7f, 0.1f, 0.85f, 0.2f};
  for (int i = 0; i < kAnchors; ++i) {
    values[0 * kAnchors + i] = cx[i];
    values[1 * kAnchors + i] = cy[i];
    values[2 * kAnchors + i] = w[i];
    values[3 * kAnchors + i] = h[i];
    values[4 * kAnchors + i] = c0[i];
    values[5 * kAnchors + i] = c1[i];
  }
  return values;
}

ResolvedInferenceConfig resolve(ModelManifest& manifest, const InferenceOptions& request = {}) {
  return vie::resolve_inference_config(manifest.task, manifest, InferenceOptions{}, request);
}

vie::DetectionBatch decode_cpu(const std::vector<float>& values, const vie::DecoderConfig& config,
                               int anchors = 4) {
  const vie::Tensor tensor = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 6, anchors},
                                                    values.data(), values.size() * sizeof(float));
  vie::DecoderInput input;
  input.outputs.emplace("predictions", tensor);
  input.geometry = identity(40, 40);
  return vie::YoloV8DetectionDecoder{}.decode_batch(input, config, vie::DeviceContext{});
}

}  // namespace

TEST(DecoderConfigPolicy, YoloResolvedConfigMapsEveryPolicyField) {
  ModelManifest manifest = detection_manifest();
  manifest.nms.enabled = true;
  manifest.ensure_provenance();

  InferenceOptions request;
  request.detection.confidence_threshold = 0.33f;
  request.detection.iou_threshold = 0.55f;
  request.detection.max_detections = 7;
  request.detection.pre_nms_top_k = 11;
  request.detection.class_agnostic_nms = true;
  request.detection.multi_label = true;
  const ResolvedInferenceConfig resolved = resolve(manifest, request);
  const vie::DecoderConfig config = vie::decoder_config_from_resolved(manifest, resolved);

  EXPECT_EQ(config.output_name, "predictions");
  EXPECT_EQ(config.channel_axis, 1);
  EXPECT_EQ(config.box_channels, 4);
  EXPECT_EQ(config.num_classes, 2);
  EXPECT_FLOAT_EQ(*config.confidence_threshold, 0.33f);
  EXPECT_FLOAT_EQ(*config.iou_threshold, 0.55f);
  EXPECT_EQ(*config.max_detections, 7);
  EXPECT_EQ(*config.pre_nms_top_k, 11);
  EXPECT_FALSE(*config.class_aware_nms);
  EXPECT_TRUE(*config.nms_enabled);
  EXPECT_EQ(*config.class_selection, vie::ClassSelection::AllAboveThreshold);

  // An explicit nms.enabled=false reaches the decoder as-is; the policy values
  // are still validated by the resolver.
  ModelManifest no_nms = detection_manifest();
  no_nms.nms.enabled = false;
  no_nms.ensure_provenance();
  const vie::DecoderConfig without_nms = vie::decoder_config_from_resolved(no_nms, resolve(no_nms));
  EXPECT_FALSE(*without_nms.nms_enabled);
}

TEST(DecoderConfigPolicy, YoloResolvedConfigRejectsForeignProtocols) {
  ModelManifest detection = detection_manifest();
  ModelManifest other = detection_manifest();
  other.decoder.protocol = "mock";
  other.ensure_provenance();
  const ResolvedInferenceConfig resolved = resolve(detection);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::decoder_config_from_resolved(other, resolved));

  ModelManifest segmentation = segmentation_manifest();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::decoder_config_from_resolved(segmentation, resolved));
}

TEST(DecoderConfigPolicy, YoloDecoderHonorsConfidenceAndMaxDetections) {
  ModelManifest manifest = detection_manifest();
  const std::vector<float> values = four_anchor_values();

  EXPECT_EQ(decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest)))
                .results[0].detections.size(),
            3u);

  InferenceOptions high;
  high.detection.confidence_threshold = 0.95f;
  EXPECT_TRUE(decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest, high)))
                  .results[0].detections.empty());

  InferenceOptions low;
  low.detection.confidence_threshold = 0.f;
  const vie::DetectionBatch everything =
      decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest, low)));
  ASSERT_FALSE(everything.results[0].detections.empty());
  EXPECT_FLOAT_EQ(everything.results[0].detections.front().confidence, 0.9f);

  InferenceOptions keep_two;
  keep_two.detection.max_detections = 2;
  const vie::DetectionBatch limited =
      decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest, keep_two)));
  ASSERT_EQ(limited.results[0].detections.size(), 2u);
  EXPECT_FLOAT_EQ(limited.results[0].detections[0].confidence, 0.9f);
  EXPECT_FLOAT_EQ(limited.results[0].detections[1].confidence, 0.85f);
}

TEST(DecoderConfigPolicy, YoloDecoderHonorsIouPreNmsAndNmsPolicy) {
  ModelManifest manifest = detection_manifest();
  const std::vector<float> values = four_anchor_values();

  InferenceOptions loose;
  loose.detection.iou_threshold = 0.8f;
  const vie::DetectionBatch all =
      decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest, loose)));
  EXPECT_EQ(all.results[0].detections.size(), 4u);

  InferenceOptions strict;
  strict.detection.iou_threshold = 0.6f;
  const vie::DetectionBatch suppressed =
      decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest, strict)));
  EXPECT_EQ(suppressed.results[0].detections.size(), 3u);

  InferenceOptions agnostic;
  agnostic.detection.iou_threshold = 0.6f;
  agnostic.detection.class_agnostic_nms = true;
  const vie::DetectionBatch across_classes =
      decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest, agnostic)));
  EXPECT_EQ(across_classes.results[0].detections.size(), 2u);

  InferenceOptions top_two;
  top_two.detection.iou_threshold = 0.8f;
  top_two.detection.pre_nms_top_k = 2;
  const vie::DetectionBatch pre_nms =
      decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest, top_two)));
  EXPECT_EQ(pre_nms.results[0].detections.size(), 2u);
  EXPECT_EQ(pre_nms.results[0].candidates_before_nms, 2);
}

TEST(DecoderConfigPolicy, YoloDecoderHonorsMultiLabel) {
  ModelManifest manifest = detection_manifest();
  constexpr int kAnchors = 1;
  std::vector<float> values(6 * kAnchors, 0.f);
  values[0] = 10.f;  // cx
  values[1] = 10.f;  // cy
  values[2] = 4.f;   // w
  values[3] = 4.f;   // h
  values[4] = 0.9f;  // class 0
  values[5] = 0.8f;  // class 1

  const vie::DetectionBatch best =
      decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest)), 1);
  ASSERT_EQ(best.results[0].detections.size(), 1u);
  EXPECT_EQ(best.results[0].detections[0].class_id, 0);

  InferenceOptions multi;
  multi.detection.multi_label = true;
  const vie::DetectionBatch all =
      decode_cpu(values, vie::decoder_config_from_resolved(manifest, resolve(manifest, multi)), 1);
  ASSERT_EQ(all.results[0].detections.size(), 2u);
  EXPECT_EQ(all.results[0].detections[0].class_id, 0);
  EXPECT_EQ(all.results[0].detections[1].class_id, 1);
}

TEST(DecoderConfigPolicy, RtDetrResolvedConfigKeepsItsOwnSemantics) {
  ModelManifest manifest = detection_manifest("rtdetr_detection_raw", 3);
  InferenceOptions request;
  request.detection.confidence_threshold = 0.4f;
  request.detection.max_detections = 20;
  request.detection.multi_label = false;
  const ResolvedInferenceConfig resolved = resolve(manifest, request);
  const vie::DecoderConfig config = vie::decoder_config_from_resolved(manifest, resolved);

  EXPECT_EQ(config.output_name, "predictions");
  EXPECT_EQ(config.num_queries, 300);
  EXPECT_FLOAT_EQ(*config.confidence_threshold, 0.4f);
  EXPECT_EQ(*config.max_detections, 20);
  EXPECT_FALSE(*config.nms_enabled);
  EXPECT_EQ(*config.class_selection, vie::ClassSelection::BestClass);
  // RT-DETR never receives YOLO NMS knobs.
  EXPECT_FALSE(config.iou_threshold.has_value());
  EXPECT_FALSE(config.class_aware_nms.has_value());
  EXPECT_FALSE(config.pre_nms_top_k.has_value());
}

#if defined(VIE_HAS_CUDA)
TEST(DecoderConfigPolicy, CpuAndCudaHonorTheSameResolvedConfig) {
  ModelManifest manifest = detection_manifest();
  const std::vector<float> values = four_anchor_values();
  InferenceOptions request;
  request.detection.confidence_threshold = 0.5f;
  request.detection.iou_threshold = 0.6f;
  request.detection.max_detections = 300;
  const vie::DecoderConfig config =
      vie::decoder_config_from_resolved(manifest, resolve(manifest, request));

  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  const vie::Tensor host = vie::Tensor::from_host(cpu, vie::DataType::Fp32, {1, 6, 4}, values.data(),
                                                  values.size() * sizeof(float));
  const vie::Tensor device = host.to(gpu);
  vie::DecoderInput input;
  input.geometry = identity(40, 40);
  input.outputs.emplace("predictions", host);
  const vie::DetectionBatch cpu_batch = vie::YoloV8DetectionDecoder{}.decode_batch(input, config, cpu);
  input.outputs.clear();
  input.outputs.emplace("predictions", device);
  const vie::DetectionBatch cuda_batch = vie::YoloV8DetectionDecoder{}.decode_batch(input, config, gpu);

  ASSERT_EQ(cpu_batch.results[0].detections.size(), cuda_batch.results[0].detections.size());
  ASSERT_EQ(cpu_batch.results[0].detections.size(), 3u);
  EXPECT_EQ(cuda_batch.results[0].execution_device, vie::DeviceType::Cuda);
  for (std::size_t i = 0; i < cpu_batch.results[0].detections.size(); ++i) {
    const vie::Detection& expected = cpu_batch.results[0].detections[i];
    const vie::Detection& actual = cuda_batch.results[0].detections[i];
    EXPECT_EQ(actual.class_id, expected.class_id);
    EXPECT_NEAR(actual.confidence, expected.confidence, 1e-5f);
    EXPECT_NEAR(actual.box.x1, expected.box.x1, 1e-4f);
    EXPECT_NEAR(actual.box.y1, expected.box.y1, 1e-4f);
    EXPECT_NEAR(actual.box.x2, expected.box.x2, 1e-4f);
    EXPECT_NEAR(actual.box.y2, expected.box.y2, 1e-4f);
  }

  // The same request-level override is applied on both devices.
  InferenceOptions high;
  high.detection.confidence_threshold = 0.95f;
  const vie::DecoderConfig strict =
      vie::decoder_config_from_resolved(manifest, resolve(manifest, high));
  input.outputs.clear();
  input.outputs.emplace("predictions", host);
  EXPECT_TRUE(vie::YoloV8DetectionDecoder{}.decode_batch(input, strict, cpu)
                  .results[0].detections.empty());
  input.outputs.clear();
  input.outputs.emplace("predictions", device);
  EXPECT_TRUE(vie::YoloV8DetectionDecoder{}.decode_batch(input, strict, gpu)
                  .results[0].detections.empty());
}
#endif  // VIE_HAS_CUDA

TEST(DecoderConfigPolicy, EngineDetectionAppliesOptionsAndKeepsLegacyCall) {
  vie::InferenceEngine engine;
  engine.load(model_config(detection_manifest()));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  const vie::DetectionBatch legacy = engine.infer(image);
  const vie::DetectionBatch defaults = engine.infer(image, InferenceOptions{});
  ASSERT_EQ(legacy.results.size(), 1u);
  ASSERT_EQ(defaults.results.size(), 1u);
  ASSERT_EQ(legacy.results[0].detections.size(), defaults.results[0].detections.size());
  ASSERT_EQ(legacy.results[0].detections.size(), 1u);
  EXPECT_FLOAT_EQ(legacy.results[0].detections[0].confidence, 0.9f);

  InferenceOptions high;
  high.detection.confidence_threshold = 0.95f;
  EXPECT_TRUE(engine.infer(image, high).results[0].detections.empty());

  // A resolved config resolved for a different protocol is rejected.
  vie::InferenceEngine rtdetr_engine;
  rtdetr_engine.load(model_config(detection_manifest("rtdetr_detection_raw")));
  const ResolvedInferenceConfig foreign = rtdetr_engine.resolve_inference_config();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_resolved(image, foreign));
}

TEST(DecoderConfigPolicy, EngineSegmentationOptionsNeedMaskPolicyFromAConfigLayer) {
  // The mock output has all-zero mask coefficients, so no instance survives the
  // mask threshold; the engine-level effect is covered by the real-model E2E
  // test. Here we check the option plumbing contract.
  vie::InferenceEngine without_policy;
  without_policy.load(model_config(segmentation_manifest()));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   without_policy.infer_segmentation(image, InferenceOptions{}));

  InferenceOptions defaults;
  defaults.segmentation.retina_masks = true;
  defaults.segmentation.binary_mask_gt_zero = true;
  defaults.segmentation.max_mask_bytes = 64ull << 20;
  vie::InferenceEngine engine;
  engine.load(model_config(segmentation_manifest(), defaults));
  const vie::ResolvedInferenceConfig resolved = engine.resolve_inference_config();
  EXPECT_TRUE(resolved.require_segmentation().retina_masks.has_value());
  EXPECT_EQ(resolved.require_segmentation().retina_masks->source, ParameterSource::ModelDefault);

  InferenceOptions wrong;
  wrong.detection.confidence_threshold = 0.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_segmentation(image, wrong));

  InferenceOptions high;
  high.segmentation.confidence_threshold = 0.95f;
  EXPECT_TRUE(engine.infer_segmentation(image, high).instances.empty());
}

TEST(DecoderConfigPolicy, EngineClassificationAppliesTopK) {
  vie::InferenceEngine engine;
  engine.load(model_config(classification_manifest(6)));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  InferenceOptions one;
  one.classification.top_k = 1;
  const vie::ClassificationBatch top1 = engine.infer_classification(image, one);
  ASSERT_EQ(top1.results.size(), 1u);
  ASSERT_EQ(top1.results[0].labels.size(), 1u);
  EXPECT_EQ(top1.results[0].labels[0].class_id, 3);

  InferenceOptions three;
  three.classification.top_k = 3;
  const vie::ClassificationBatch top3 = engine.infer_classification(image, three);
  ASSERT_EQ(top3.results[0].labels.size(), 3u);
  EXPECT_EQ(top3.results[0].labels[0].class_id, 3);
  EXPECT_EQ(top3.results[0].labels[1].class_id, 2);
  EXPECT_EQ(top3.results[0].labels[2].class_id, 1);

  // Default top_k is min(5, num_classes).
  const vie::ClassificationBatch defaults = engine.infer_classification(image, InferenceOptions{});
  EXPECT_EQ(defaults.results[0].labels.size(), 5u);

  // A detection block cannot be smuggled into a classification call.
  InferenceOptions wrong;
  wrong.detection.confidence_threshold = 0.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_classification(image, wrong));
}

TEST(DecoderConfigPolicy, ObbResolvedConfigMapsAndHonorsRotatedIou) {
  ModelManifest manifest = obb_manifest();
  InferenceOptions request;
  request.obb.confidence_threshold = 0.5f;
  request.obb.iou_threshold = 0.6f;
  request.obb.max_detections = 300;
  request.obb.pre_nms_top_k = 4096;
  request.obb.class_agnostic_nms = false;
  const ResolvedInferenceConfig resolved = resolve(manifest, request);
  const vie::ObbDecodeConfig config = vie::obb_decode_config_from_resolved(manifest, resolved);
  EXPECT_EQ(config.output_name, "output0");
  EXPECT_EQ(config.num_classes, 2);
  EXPECT_EQ(*config.class_selection, vie::ClassSelection::BestClass);
  EXPECT_FLOAT_EQ(*config.confidence_threshold, 0.5f);
  EXPECT_FLOAT_EQ(*config.iou_threshold, 0.6f);
  EXPECT_EQ(*config.max_detections, 300);
  EXPECT_EQ(*config.pre_nms_top_k, 4096);
  EXPECT_TRUE(*config.class_aware);
  EXPECT_TRUE(*config.nms_enabled);

  // Two same-class squares whose rotated IoU is about 0.78, plus a far box.
  constexpr int kAnchors = 3;
  std::vector<float> values(7 * kAnchors, 0.f);
  const float cx[kAnchors] = {10.f, 10.5f, 20.f};
  const float cy[kAnchors] = {10.f, 10.f, 20.f};
  const float score[kAnchors] = {0.9f, 0.8f, 0.6f};
  for (int i = 0; i < kAnchors; ++i) {
    values[0 * kAnchors + i] = cx[i];
    values[1 * kAnchors + i] = cy[i];
    values[2 * kAnchors + i] = 4.f;
    values[3 * kAnchors + i] = 4.f;
    values[4 * kAnchors + i] = score[i];
    values[6 * kAnchors + i] = 0.f;  // angle
  }

  const auto decode = [&values](const vie::ObbDecodeConfig& policy) {
    const vie::Tensor tensor = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 7, kAnchors},
                                                      values.data(), values.size() * sizeof(float));
    vie::DecoderInput input;
    input.outputs.emplace("output0", tensor);
    input.geometry = identity(40, 40);
    return vie::YoloV8ObbDecoder{}.decode_obb(input, policy, vie::DeviceContext{});
  };

  const vie::ObbBatch suppressed = decode(config);
  ASSERT_EQ(suppressed.results.size(), 1u);
  EXPECT_EQ(suppressed.results[0].objects.size(), 2u);

  InferenceOptions loose;
  loose.obb.iou_threshold = 0.9f;
  const vie::ObbBatch kept =
      decode(vie::obb_decode_config_from_resolved(manifest, resolve(manifest, loose)));
  EXPECT_EQ(kept.results[0].objects.size(), 3u);

  InferenceOptions one;
  one.obb.max_detections = 1;
  EXPECT_EQ(decode(vie::obb_decode_config_from_resolved(manifest, resolve(manifest, one)))
                .results[0].objects.size(),
            1u);

  InferenceOptions pre_nms;
  pre_nms.obb.pre_nms_top_k = 1;
  const vie::ObbBatch limited =
      decode(vie::obb_decode_config_from_resolved(manifest, resolve(manifest, pre_nms)));
  EXPECT_EQ(limited.results[0].objects.size(), 1u);
  EXPECT_EQ(limited.results[0].candidates_before_nms, 1);

  // A resolved config for another task is rejected.
  ModelManifest detection = detection_manifest();
  const ResolvedInferenceConfig foreign = resolve(detection);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::obb_decode_config_from_resolved(manifest, foreign));
}

TEST(DecoderConfigPolicy, PoseResolvedConfigMapsEveryPolicyField) {
  ModelManifest manifest = pose_manifest(2);
  InferenceOptions request;
  request.pose.confidence_threshold = 0.45f;
  request.pose.iou_threshold = 0.6f;
  request.pose.max_detections = 9;
  request.pose.pre_nms_top_k = 30;
  request.pose.class_agnostic_nms = true;
  const ResolvedInferenceConfig resolved = resolve(manifest, request);
  const vie::PoseDecodeConfig config = vie::pose_decode_config_from_resolved(manifest, resolved);
  EXPECT_EQ(config.output_name, "output0");
  EXPECT_EQ(config.num_classes, 1);
  EXPECT_EQ(config.keypoint_count, 2);
  EXPECT_EQ(config.keypoint_dimensions, 3);
  EXPECT_EQ(*config.class_selection, vie::ClassSelection::BestClass);
  EXPECT_FLOAT_EQ(*config.confidence_threshold, 0.45f);
  EXPECT_FLOAT_EQ(*config.iou_threshold, 0.6f);
  EXPECT_EQ(*config.max_detections, 9);
  EXPECT_EQ(*config.pre_nms_top_k, 30);
  EXPECT_FALSE(*config.class_aware);
  EXPECT_TRUE(*config.nms_enabled);
}

TEST(DecoderConfigPolicy, EngineObbRejectsMismatchedBlocks) {
  vie::InferenceEngine engine;
  engine.load(model_config(obb_manifest()));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  InferenceOptions wrong;
  wrong.detection.confidence_threshold = 0.95f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_obb(image, wrong));

  InferenceOptions wrong_task;
  wrong_task.pose.confidence_threshold = 0.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_obb(image, wrong_task));
}

TEST(DecoderConfigPolicy, EnginePoseAppliesOptions) {
  vie::InferenceEngine engine;
  engine.load(model_config(pose_manifest(2)));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  const vie::PoseBatch kept = engine.infer_pose(image, InferenceOptions{});
  ASSERT_EQ(kept.results.size(), 1u);
  ASSERT_EQ(kept.results[0].objects.size(), 1u);
  EXPECT_EQ(kept.results[0].objects[0].class_id, 0);
  ASSERT_EQ(kept.results[0].objects[0].keypoints.size(), 2u);

  InferenceOptions high;
  high.pose.confidence_threshold = 0.95f;
  EXPECT_TRUE(engine.infer_pose(image, high).results[0].objects.empty());

  InferenceOptions wrong;
  wrong.obb.confidence_threshold = 0.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_pose(image, wrong));
}