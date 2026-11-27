// Phase 12: unified inference configuration resolution tests.
//
// Covers:
// - framework defaults and their ParameterSource
// - manifest / model-default / request-override precedence
// - Unresolved and Inferred manifest values never become defaults
// - numeric validation, boundaries, non-finite input
// - task isolation (a non-matching block is rejected, never ignored)
// - RT-DETR rejecting knobs its post-processing cannot honor
// - classification top-k bounded by num_classes
// - fingerprint stability and separation
// - engine-level model defaults captured from ModelConfig

#include "test_util.hpp"

#include "vie/config/inference_config.hpp"
#include "vie/engine/inference_engine.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <string>

using vie::InferenceOptions;
using vie::ModelManifest;
using vie::ParameterSource;
using vie::ResolvedInferenceConfig;
using vie::TaskType;

namespace {

ModelManifest detection_manifest(const std::string& protocol = "yolov8_detect_raw") {
  ModelManifest manifest;
  manifest.model_name = "cfg_det";
  manifest.task = TaskType::Detection;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 2;
  manifest.decoder.protocol = protocol;
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.class_selection = "best_class";
  manifest.decoder.scores_are_probabilities = true;
  return manifest;
}

ModelManifest detection_manifest_with_nms() {
  ModelManifest manifest = detection_manifest();
  manifest.nms.enabled = true;
  manifest.nms.confidence_threshold = 0.6f;
  manifest.nms.iou_threshold = 0.7f;
  manifest.nms.max_detections = 42;
  manifest.nms.pre_nms_top_k = 123;
  manifest.nms.class_aware = false;
  manifest.decoder.class_selection = "all_above_threshold";
  manifest.ensure_provenance();
  return manifest;
}

ModelManifest classification_manifest(int classes) {
  ModelManifest manifest;
  manifest.model_name = "cfg_cls";
  manifest.task = TaskType::Classification;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = classes;
  manifest.decoder.protocol = "classification_logits_raw";
  manifest.classification.output_name = "logits";
  manifest.classification.class_axis = 1;
  manifest.classification.score_type = "logits";
  manifest.ensure_provenance();
  return manifest;
}

ModelManifest semantic_manifest() {
  ModelManifest manifest;
  manifest.model_name = "cfg_sem";
  manifest.task = TaskType::SemanticSegmentation;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 21;
  manifest.decoder.protocol = "deeplabv3_semantic_seg_raw";
  manifest.semantic_segmentation.output_name = "logits";
  manifest.semantic_segmentation.channel_axis = 1;
  manifest.semantic_segmentation.logits_semantics = "per_pixel_class_logits";
  manifest.ensure_provenance();
  return manifest;
}

ModelManifest segmentation_manifest() {
  ModelManifest manifest;
  manifest.model_name = "cfg_seg";
  manifest.task = TaskType::Segmentation;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 2;
  manifest.decoder.protocol = "yolov8_seg_raw";
  return manifest;
}

ModelManifest obb_manifest() {
  ModelManifest manifest;
  manifest.model_name = "cfg_obb";
  manifest.task = TaskType::Obb;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 2;
  manifest.decoder.protocol = "yolov8_obb_raw";
  manifest.ensure_provenance();
  return manifest;
}

ModelManifest pose_manifest() {
  ModelManifest manifest;
  manifest.model_name = "cfg_pose";
  manifest.task = TaskType::Pose;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 1;
  manifest.decoder.protocol = "yolov8_pose_raw";
  manifest.ensure_provenance();
  return manifest;
}

ResolvedInferenceConfig resolve(const ModelManifest& manifest, const InferenceOptions& model = {},
                                const InferenceOptions& request = {}) {
  return vie::resolve_inference_config(manifest.task, manifest, model, request);
}

std::string thrown_message(const std::function<void()>& body) {
  try {
    body();
  } catch (const vie::VieError& error) {
    return error.what();
  }
  ADD_FAILURE() << "expected VieError";
  return {};
}

}  // namespace

TEST(InferenceConfig, FrameworkDefaultsAndSources) {
  const ResolvedInferenceConfig resolved = resolve(detection_manifest());
  ASSERT_TRUE(resolved.detection.has_value());
  EXPECT_FALSE(resolved.segmentation.has_value());
  EXPECT_FALSE(resolved.obb.has_value());
  EXPECT_FALSE(resolved.pose.has_value());
  EXPECT_FALSE(resolved.classification.has_value());
  EXPECT_EQ(resolved.task, TaskType::Detection);
  EXPECT_EQ(resolved.protocol, "yolov8_detect_raw");

  const vie::ResolvedDetectionConfig& detection = resolved.require_detection();
  EXPECT_FLOAT_EQ(detection.confidence_threshold.value, vie::kFrameworkConfidenceThreshold);
  EXPECT_FLOAT_EQ(detection.iou_threshold.value, vie::kFrameworkIouThreshold);
  EXPECT_EQ(detection.max_detections.value, vie::kFrameworkMaxDetections);
  EXPECT_EQ(detection.pre_nms_top_k.value, vie::kFrameworkPreNmsTopK);
  EXPECT_FALSE(detection.class_agnostic_nms.value);
  EXPECT_FALSE(detection.multi_label.value);
  EXPECT_TRUE(detection.nms_enabled);
  for (const ParameterSource source :
       {detection.confidence_threshold.source, detection.iou_threshold.source,
        detection.max_detections.source, detection.pre_nms_top_k.source,
        detection.class_agnostic_nms.source}) {
    EXPECT_EQ(source, ParameterSource::FrameworkDefault);
  }
  // class_selection=best_class is a manifest-proved structural fact, so the
  // multi-label policy is known even without an NMS block.
  EXPECT_FALSE(detection.multi_label.value);
  EXPECT_EQ(detection.multi_label.source, ParameterSource::Manifest);
  EXPECT_TRUE(static_cast<bool>(detection.nms_enabled));
}

TEST(InferenceConfig, ManifestValuesAreUsedAndTagged) {
  const ResolvedInferenceConfig resolved = resolve(detection_manifest_with_nms());
  const vie::ResolvedDetectionConfig& detection = resolved.require_detection();
  EXPECT_FLOAT_EQ(detection.confidence_threshold.value, 0.6f);
  EXPECT_FLOAT_EQ(detection.iou_threshold.value, 0.7f);
  EXPECT_EQ(detection.max_detections.value, 42);
  EXPECT_EQ(detection.pre_nms_top_k.value, 123);
  // nms.class_aware=false in the manifest means class-agnostic NMS.
  EXPECT_TRUE(detection.class_agnostic_nms.value);
  // decoder.class_selection=all_above_threshold is the manifest-proved multi-label policy.
  EXPECT_TRUE(detection.multi_label.value);
  for (const ParameterSource source :
       {detection.confidence_threshold.source, detection.iou_threshold.source,
        detection.max_detections.source, detection.pre_nms_top_k.source,
        detection.class_agnostic_nms.source, detection.multi_label.source}) {
    EXPECT_EQ(source, ParameterSource::Manifest);
  }
}

TEST(InferenceConfig, UnresolvedAndInferredManifestValuesNeverParticipate) {
  ModelManifest manifest = detection_manifest_with_nms();
  manifest.provenance["nms.confidence_threshold"] =
      vie::FieldProvenance{vie::Certainty::Unresolved, {"not observed"}};
  manifest.provenance["nms.iou_threshold"] =
      vie::FieldProvenance{vie::Certainty::Inferred, {"typical yolov8 value"}};
  manifest.provenance["nms.max_detections"] =
      vie::FieldProvenance{vie::Certainty::Inferred, {"guessed from family"}};

  const ResolvedInferenceConfig resolved = resolve(manifest);
  const vie::ResolvedDetectionConfig& detection = resolved.require_detection();
  EXPECT_FLOAT_EQ(detection.confidence_threshold.value, vie::kFrameworkConfidenceThreshold);
  EXPECT_EQ(detection.confidence_threshold.source, ParameterSource::FrameworkDefault);
  EXPECT_FLOAT_EQ(detection.iou_threshold.value, vie::kFrameworkIouThreshold);
  EXPECT_EQ(detection.iou_threshold.source, ParameterSource::FrameworkDefault);
  EXPECT_EQ(detection.max_detections.value, vie::kFrameworkMaxDetections);
  EXPECT_EQ(detection.max_detections.source, ParameterSource::FrameworkDefault);
  // A value without a provenance entry counts as user-supplied.
  EXPECT_EQ(detection.pre_nms_top_k.value, 123);
  EXPECT_EQ(detection.pre_nms_top_k.source, ParameterSource::Manifest);
}

TEST(InferenceConfig, RequestBeatsModelDefaultBeatsManifest) {
  ModelManifest manifest = detection_manifest();
  manifest.nms.confidence_threshold = 0.1f;
  manifest.nms.iou_threshold = 0.7f;
  manifest.ensure_provenance();

  InferenceOptions model;
  model.detection.confidence_threshold = 0.2f;
  model.detection.max_detections = 50;

  InferenceOptions request;
  request.detection.confidence_threshold = 0.3f;

  const ResolvedInferenceConfig resolved = resolve(manifest, model, request);
  const vie::ResolvedDetectionConfig& detection = resolved.require_detection();
  EXPECT_FLOAT_EQ(detection.confidence_threshold.value, 0.3f);
  EXPECT_EQ(detection.confidence_threshold.source, ParameterSource::RequestOverride);
  EXPECT_FLOAT_EQ(detection.iou_threshold.value, 0.7f);
  EXPECT_EQ(detection.iou_threshold.source, ParameterSource::Manifest);
  EXPECT_EQ(detection.max_detections.value, 50);
  EXPECT_EQ(detection.max_detections.source, ParameterSource::ModelDefault);

  const ResolvedInferenceConfig without_request = resolve(manifest, model);
  EXPECT_FLOAT_EQ(without_request.require_detection().confidence_threshold.value, 0.2f);
  EXPECT_EQ(without_request.require_detection().confidence_threshold.source,
            ParameterSource::ModelDefault);
}

TEST(InferenceConfig, NonMatchingTaskBlocksAreRejected) {
  InferenceOptions detection_block;
  detection_block.detection.confidence_threshold = 0.4f;

  const std::string classification_error =
      thrown_message([&] { resolve(classification_manifest(3), {}, detection_block); });
  EXPECT_NE(classification_error.find("detection"), std::string::npos);
  EXPECT_NE(classification_error.find("classification"), std::string::npos);

  InferenceOptions classification_block;
  classification_block.classification.top_k = 2;
  const std::string detection_error =
      thrown_message([&] { resolve(detection_manifest(), {}, classification_block); });
  EXPECT_NE(detection_error.find("classification"), std::string::npos);

  // A wrong block in the model-default layer is rejected just like a request block.
  const std::string model_error =
      thrown_message([&] { resolve(semantic_manifest(), detection_block, {}); });
  EXPECT_NE(model_error.find("detection"), std::string::npos);
  EXPECT_NE(model_error.find("semantic_segmentation"), std::string::npos);

  // Semantic segmentation has no configurable post-processing parameter.
  const ResolvedInferenceConfig semantic = resolve(semantic_manifest());
  EXPECT_EQ(semantic.task, TaskType::SemanticSegmentation);
  EXPECT_FALSE(semantic.detection.has_value());
  EXPECT_FALSE(semantic.segmentation.has_value());
  EXPECT_FALSE(semantic.classification.has_value());
  EXPECT_EQ(semantic.fingerprint(), "semantic_segmentation|deeplabv3_semantic_seg_raw|none");
}

TEST(InferenceConfig, RtDetrRejectsKnobsItCannotHonor) {
  const ModelManifest manifest = detection_manifest("rtdetr_detection_raw");

  const auto rejects = [&manifest](const char* field, const InferenceOptions& options) {
    const std::string message = thrown_message([&] { resolve(manifest, {}, options); });
    EXPECT_NE(message.find(field), std::string::npos) << message;
  };

  InferenceOptions iou;
  iou.detection.iou_threshold = 0.5f;
  rejects("iou_threshold", iou);

  InferenceOptions pre_nms;
  pre_nms.detection.pre_nms_top_k = 100;
  rejects("pre_nms_top_k", pre_nms);

  InferenceOptions agnostic;
  agnostic.detection.class_agnostic_nms = true;
  rejects("class_agnostic_nms", agnostic);

  InferenceOptions multi_label;
  multi_label.detection.multi_label = true;
  rejects("multi_label", multi_label);

  // multi_label=false is the RT-DETR semantic (one class per query) and is accepted.
  InferenceOptions best_class;
  best_class.detection.multi_label = false;
  best_class.detection.confidence_threshold = 0.4f;
  best_class.detection.max_detections = 100;
  const ResolvedInferenceConfig resolved = resolve(manifest, {}, best_class);
  EXPECT_FLOAT_EQ(resolved.require_detection().confidence_threshold.value, 0.4f);
  EXPECT_EQ(resolved.require_detection().confidence_threshold.source,
            ParameterSource::RequestOverride);
  EXPECT_FALSE(resolved.require_detection().multi_label.value);
  EXPECT_FALSE(resolved.require_detection().nms_enabled);
}

TEST(InferenceConfig, ThresholdValidationRejectsNonFiniteAndOutOfRange) {
  const ModelManifest manifest = detection_manifest();
  const auto check = [&manifest](std::optional<float> confidence, const char* field) {
    InferenceOptions options;
    options.detection.confidence_threshold = confidence;
    const std::string message = thrown_message([&] { resolve(manifest, {}, options); });
    EXPECT_NE(message.find(field), std::string::npos) << message;
  };
  check(std::nanf(""), "confidence_threshold");
  check(std::numeric_limits<float>::infinity(), "confidence_threshold");
  check(-std::numeric_limits<float>::infinity(), "confidence_threshold");
  check(-0.01f, "confidence_threshold");
  check(1.01f, "confidence_threshold");

  InferenceOptions iou;
  iou.detection.iou_threshold = 1.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, resolve(manifest, {}, iou));

  // Inclusive boundaries are legal.
  InferenceOptions zero;
  zero.detection.confidence_threshold = 0.f;
  zero.detection.iou_threshold = 0.f;
  const ResolvedInferenceConfig low = resolve(manifest, {}, zero);
  EXPECT_FLOAT_EQ(low.require_detection().confidence_threshold.value, 0.f);
  EXPECT_FLOAT_EQ(low.require_detection().iou_threshold.value, 0.f);

  InferenceOptions one;
  one.detection.confidence_threshold = 1.f;
  one.detection.iou_threshold = 1.f;
  const ResolvedInferenceConfig high = resolve(manifest, {}, one);
  EXPECT_FLOAT_EQ(high.require_detection().confidence_threshold.value, 1.f);
  EXPECT_FLOAT_EQ(high.require_detection().iou_threshold.value, 1.f);
}

TEST(InferenceConfig, CountValidationAndBoundaries) {
  const ModelManifest manifest = detection_manifest();
  const auto rejects = [&manifest](const char* field, const InferenceOptions& options) {
    const std::string message = thrown_message([&] { resolve(manifest, {}, options); });
    EXPECT_NE(message.find(field), std::string::npos) << message;
  };

  for (const int value : {0, -1, vie::kMaxDetectionsLimit + 1}) {
    InferenceOptions options;
    options.detection.max_detections = value;
    rejects("max_detections", options);
  }
  for (const int value : {0, -5, 4097}) {
    InferenceOptions options;
    options.detection.pre_nms_top_k = value;
    rejects("pre_nms_top_k", options);
  }

  InferenceOptions upper;
  upper.detection.max_detections = vie::kMaxDetectionsLimit;
  upper.detection.pre_nms_top_k = 4096;
  const ResolvedInferenceConfig resolved = resolve(manifest, {}, upper);
  EXPECT_EQ(resolved.require_detection().max_detections.value, vie::kMaxDetectionsLimit);
  EXPECT_EQ(resolved.require_detection().pre_nms_top_k.value, 4096);

  InferenceOptions mask_budget;
  mask_budget.segmentation.max_mask_bytes = 0;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, resolve(segmentation_manifest(), {}, mask_budget));
}

TEST(InferenceConfig, EffectiveManifestValuesAreRevalidated) {
  ModelManifest high = detection_manifest();
  high.nms.confidence_threshold = 1.5f;
  high.ensure_provenance();
  const std::string confidence =
      thrown_message([&] { resolve(high); });
  EXPECT_NE(confidence.find("confidence_threshold"), std::string::npos) << confidence;

  ModelManifest nan_iou = detection_manifest();
  nan_iou.nms.iou_threshold = std::nanf("");
  nan_iou.ensure_provenance();
  const std::string iou = thrown_message([&] { resolve(nan_iou); });
  EXPECT_NE(iou.find("iou_threshold"), std::string::npos) << iou;

  ModelManifest zero_max = detection_manifest();
  zero_max.nms.max_detections = 0;
  zero_max.ensure_provenance();
  const std::string max_detections = thrown_message([&] { resolve(zero_max); });
  EXPECT_NE(max_detections.find("max_detections"), std::string::npos) << max_detections;
}

TEST(InferenceConfig, ClassificationTopKIsBoundedByClassCount) {
  const ResolvedInferenceConfig three = resolve(classification_manifest(3));
  EXPECT_EQ(three.require_classification().top_k.value, 3);
  EXPECT_EQ(three.require_classification().top_k.source, ParameterSource::FrameworkDefault);

  const ResolvedInferenceConfig many = resolve(classification_manifest(1000));
  EXPECT_EQ(many.require_classification().top_k.value, vie::kFrameworkClassificationTopK);

  ModelManifest unknown_classes = classification_manifest(3);
  unknown_classes.num_classes.reset();
  const ResolvedInferenceConfig unset = resolve(unknown_classes);
  EXPECT_EQ(unset.require_classification().top_k.value, vie::kFrameworkClassificationTopK);

  InferenceOptions request;
  request.classification.top_k = 2;
  const ResolvedInferenceConfig requested = resolve(classification_manifest(3), {}, request);
  EXPECT_EQ(requested.require_classification().top_k.value, 2);
  EXPECT_EQ(requested.require_classification().top_k.source, ParameterSource::RequestOverride);

  InferenceOptions too_large;
  too_large.classification.top_k = 4;
  const std::string message =
      thrown_message([&] { resolve(classification_manifest(3), {}, too_large); });
  EXPECT_NE(message.find("top_k"), std::string::npos) << message;
  EXPECT_NE(message.find("num_classes"), std::string::npos) << message;

  InferenceOptions zero;
  zero.classification.top_k = 0;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, resolve(classification_manifest(3), {}, zero));

  InferenceOptions model_default;
  model_default.classification.top_k = 1;
  const ResolvedInferenceConfig from_model = resolve(classification_manifest(3), model_default);
  EXPECT_EQ(from_model.require_classification().top_k.value, 1);
  EXPECT_EQ(from_model.require_classification().top_k.source, ParameterSource::ModelDefault);
}

TEST(InferenceConfig, SegmentationMaskPolicyIsOptionalAndTracked) {
  ModelManifest manifest = segmentation_manifest();
  manifest.segmentation.retina_masks = true;
  manifest.segmentation.binary_mask_gt_zero = false;
  manifest.segmentation.max_mask_bytes = 1024;
  manifest.ensure_provenance();

  const ResolvedInferenceConfig resolved = resolve(manifest);
  const vie::ResolvedSegmentationConfig& segmentation = resolved.require_segmentation();
  ASSERT_TRUE(segmentation.retina_masks.has_value());
  EXPECT_TRUE(segmentation.retina_masks->value);
  EXPECT_EQ(segmentation.retina_masks->source, ParameterSource::Manifest);
  ASSERT_TRUE(segmentation.binary_mask_gt_zero.has_value());
  EXPECT_FALSE(segmentation.binary_mask_gt_zero->value);
  ASSERT_TRUE(segmentation.max_mask_bytes.has_value());
  EXPECT_EQ(segmentation.max_mask_bytes->value, 1024u);

  InferenceOptions request;
  request.segmentation.retina_masks = false;
  const ResolvedInferenceConfig overridden = resolve(manifest, {}, request);
  EXPECT_FALSE(overridden.require_segmentation().retina_masks->value);
  EXPECT_EQ(overridden.require_segmentation().retina_masks->source,
            ParameterSource::RequestOverride);

  // Unset mask policy stays unset: no framework default is invented for it.
  const ResolvedInferenceConfig empty = resolve(segmentation_manifest());
  EXPECT_FALSE(empty.require_segmentation().retina_masks.has_value());
  EXPECT_FALSE(empty.require_segmentation().binary_mask_gt_zero.has_value());
  EXPECT_FALSE(empty.require_segmentation().max_mask_bytes.has_value());
  EXPECT_FLOAT_EQ(empty.require_segmentation().confidence_threshold.value,
                  vie::kFrameworkConfidenceThreshold);
}

TEST(InferenceConfig, ObbAndPoseUseTheirOwnBlocks) {
  InferenceOptions obb_request;
  obb_request.obb.confidence_threshold = 0.35f;
  obb_request.obb.iou_threshold = 0.25f;
  const ResolvedInferenceConfig obb = resolve(obb_manifest(), {}, obb_request);
  ASSERT_TRUE(obb.obb.has_value());
  EXPECT_FALSE(obb.detection.has_value());
  EXPECT_FLOAT_EQ(obb.require_obb().iou_threshold.value, 0.25f);
  EXPECT_EQ(obb.require_obb().iou_threshold.source, ParameterSource::RequestOverride);

  InferenceOptions pose_request;
  pose_request.pose.confidence_threshold = 0.4f;
  pose_request.pose.max_detections = 12;
  const ResolvedInferenceConfig pose = resolve(pose_manifest(), {}, pose_request);
  ASSERT_TRUE(pose.pose.has_value());
  EXPECT_FALSE(pose.detection.has_value());
  EXPECT_EQ(pose.require_pose().max_detections.value, 12);

  // A pose block cannot be applied to an OBB model.
  const std::string message = thrown_message([&] { resolve(obb_manifest(), {}, pose_request); });
  EXPECT_NE(message.find("pose"), std::string::npos) << message;
}

TEST(InferenceConfig, ManifestTaskMustMatchTheResolvedTask) {
  const ModelManifest classification = classification_manifest(2);
  const std::string message = thrown_message(
      [&] { vie::resolve_inference_config(TaskType::Detection, classification, {}, {}); });
  EXPECT_NE(message.find("task"), std::string::npos) << message;

  ModelManifest unknown = detection_manifest();
  unknown.task = TaskType::Unknown;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::resolve_inference_config(TaskType::Detection, unknown, {}, {}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::resolve_inference_config(TaskType::Unknown, unknown, {}, {}));
}

TEST(InferenceConfig, FingerprintSeparatesEffectiveConfigs) {
  const ModelManifest manifest = detection_manifest();
  InferenceOptions first;
  first.detection.confidence_threshold = 0.5f;
  InferenceOptions same = first;
  InferenceOptions lower = first;
  lower.detection.confidence_threshold = 0.4f;
  InferenceOptions agnostic = first;
  agnostic.detection.class_agnostic_nms = true;

  const ResolvedInferenceConfig a = resolve(manifest, {}, first);
  const ResolvedInferenceConfig b = resolve(manifest, {}, same);
  const ResolvedInferenceConfig c = resolve(manifest, {}, lower);
  const ResolvedInferenceConfig d = resolve(manifest, {}, agnostic);
  EXPECT_EQ(a.fingerprint(), b.fingerprint());
  EXPECT_NE(a.fingerprint(), c.fingerprint());
  EXPECT_NE(a.fingerprint(), d.fingerprint());

  // The same numeric policy on a different task is still a different config.
  InferenceOptions obb_options;
  obb_options.obb.confidence_threshold = 0.5f;
  const ResolvedInferenceConfig e = resolve(obb_manifest(), {}, obb_options);
  EXPECT_NE(a.fingerprint(), e.fingerprint());

  // The protocol participates, so two graphs never share a batch key.
  const ResolvedInferenceConfig rtdetr = resolve(detection_manifest("rtdetr_detection_raw"));
  EXPECT_NE(a.fingerprint(), rtdetr.fingerprint());
}

TEST(InferenceConfig, ParameterSourceNamesAreStable) {
  EXPECT_EQ(std::string(vie::to_string(ParameterSource::FrameworkDefault)), "framework_default");
  EXPECT_EQ(std::string(vie::to_string(ParameterSource::Manifest)), "manifest");
  EXPECT_EQ(std::string(vie::to_string(ParameterSource::ModelDefault)), "model_default");
  EXPECT_EQ(std::string(vie::to_string(ParameterSource::RequestOverride)), "request_override");
}

TEST(InferenceConfig, EmptyOptionBlocksAreDetected) {
  const InferenceOptions empty;
  EXPECT_TRUE(empty.empty());
  InferenceOptions detection;
  detection.detection.confidence_threshold = 0.5f;
  EXPECT_FALSE(detection.empty());
  EXPECT_FALSE(detection.detection.empty());
  EXPECT_TRUE(detection.classification.empty());
}

TEST(InferenceConfigEngine, ModelDefaultsAreCapturedAndRequestOverridesThem) {
  vie::ModelConfig config;
  config.manifest = ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  config.inference_options.detection.confidence_threshold = 0.4f;
  config.inference_options.detection.max_detections = 7;

  vie::InferenceEngine engine;
  engine.load(config);
  ASSERT_TRUE(engine.loaded());
  EXPECT_FLOAT_EQ(engine.model_inference_options().detection.confidence_threshold.value(), 0.4f);
  EXPECT_EQ(engine.model_inference_options().detection.max_detections.value(), 7);

  const ResolvedInferenceConfig defaults = engine.resolve_inference_config();
  EXPECT_FLOAT_EQ(defaults.require_detection().confidence_threshold.value, 0.4f);
  EXPECT_EQ(defaults.require_detection().confidence_threshold.source, ParameterSource::ModelDefault);
  EXPECT_EQ(defaults.require_detection().max_detections.value, 7);
  EXPECT_FLOAT_EQ(defaults.require_detection().iou_threshold.value, vie::kFrameworkIouThreshold);

  InferenceOptions request;
  request.detection.confidence_threshold = 0.6f;
  const ResolvedInferenceConfig overridden = engine.resolve_inference_config(request);
  EXPECT_FLOAT_EQ(overridden.require_detection().confidence_threshold.value, 0.6f);
  EXPECT_EQ(overridden.require_detection().confidence_threshold.source,
            ParameterSource::RequestOverride);
  EXPECT_EQ(overridden.require_detection().max_detections.value, 7);
  EXPECT_EQ(overridden.require_detection().max_detections.source, ParameterSource::ModelDefault);
}

TEST(InferenceConfigEngine, LoadRejectsInvalidModelDefaultsAndUnloadedResolveThrows) {
  vie::InferenceEngine unloaded;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, unloaded.resolve_inference_config());

  vie::ModelConfig invalid;
  invalid.manifest = ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  invalid.inference_options.detection.confidence_threshold = 1.5f;
  vie::InferenceEngine engine;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.load(invalid));
  EXPECT_FALSE(engine.loaded());

  vie::ModelConfig wrong_block;
  wrong_block.manifest = ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  wrong_block.inference_options.classification.top_k = 2;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.load(wrong_block));
  EXPECT_FALSE(engine.loaded());
}