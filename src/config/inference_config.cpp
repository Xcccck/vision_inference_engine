#include "vie/config/inference_config.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <type_traits>

namespace vie {
namespace {

constexpr int kMinCount = 1;
constexpr int kMaxPreNmsTopK = 4096;

void require_unit(float value, const char* name) {
  if (!std::isfinite(value) || value < 0.f || value > 1.f) {
    throw VieError(ErrorCode::InvalidArgument,
                   std::string(name) + " must be finite and within the inclusive range [0, 1]");
  }
}

void require_count(int value, int low, int high, const char* name) {
  if (value < low || value > high) {
    throw VieError(ErrorCode::InvalidArgument,
                   std::string(name) + " must be within " + std::to_string(low) + ".." +
                       std::to_string(high));
  }
}

void require_no_other_blocks(TaskType task, const InferenceOptions& options) {
  const auto reject = [task](const char* block) {
    throw VieError(ErrorCode::InvalidArgument,
                   std::string(block) + " options cannot be applied to task " + to_string(task) +
                       "; only the matching task block is accepted");
  };
  if (task != TaskType::Detection && !options.detection.empty()) reject("detection");
  if (task != TaskType::Segmentation && !options.segmentation.empty()) reject("segmentation");
  if (task != TaskType::Obb && !options.obb.empty()) reject("obb");
  if (task != TaskType::Pose && !options.pose.empty()) reject("pose");
  if (task != TaskType::Classification && !options.classification.empty()) reject("classification");
}

void validate_detection_block(const DetectionOptions& options, const ModelManifest* manifest) {
  if (options.confidence_threshold.has_value()) {
    require_unit(*options.confidence_threshold, "detection.confidence_threshold");
  }
  if (options.max_detections.has_value()) {
    require_count(*options.max_detections, kMinCount, kMaxDetectionsLimit, "detection.max_detections");
  }
  if (options.pre_nms_top_k.has_value()) {
    require_count(*options.pre_nms_top_k, kMinCount, kMaxPreNmsTopK, "detection.pre_nms_top_k");
  }
  const bool rtdetr = manifest != nullptr && manifest->decoder.protocol == "rtdetr_detection_raw";
  if (rtdetr) {
    // RT-DETR selects one class per query and runs TopK in the graph. It has no
    // NMS and no runtime pre-NMS candidate stage, so those knobs are rejected
    // rather than silently ignored.
    if (options.iou_threshold.has_value()) {
      throw VieError(ErrorCode::Unsupported,
                     "rtdetr_detection_raw does not run NMS and does not accept detection.iou_threshold");
    }
    if (options.pre_nms_top_k.has_value()) {
      throw VieError(ErrorCode::Unsupported,
                     "rtdetr_detection_raw selects queries in the graph and does not accept "
                     "detection.pre_nms_top_k");
    }
    if (options.class_agnostic_nms.has_value()) {
      throw VieError(ErrorCode::Unsupported,
                     "rtdetr_detection_raw does not run NMS and does not accept "
                     "detection.class_agnostic_nms");
    }
    if (options.multi_label.has_value() && *options.multi_label) {
      throw VieError(ErrorCode::Unsupported,
                     "rtdetr_detection_raw keeps one class per query and does not accept "
                     "detection.multi_label=true");
    }
    return;
  }
  if (options.iou_threshold.has_value()) {
    require_unit(*options.iou_threshold, "detection.iou_threshold");
  }
}

void validate_segmentation_block(const SegmentationOptions& options) {
  if (options.confidence_threshold.has_value()) {
    require_unit(*options.confidence_threshold, "segmentation.confidence_threshold");
  }
  if (options.iou_threshold.has_value()) {
    require_unit(*options.iou_threshold, "segmentation.iou_threshold");
  }
  if (options.max_detections.has_value()) {
    require_count(*options.max_detections, kMinCount, kMaxDetectionsLimit,
                  "segmentation.max_detections");
  }
  if (options.pre_nms_top_k.has_value()) {
    require_count(*options.pre_nms_top_k, kMinCount, kMaxPreNmsTopK, "segmentation.pre_nms_top_k");
  }
  if (options.max_mask_bytes.has_value() && *options.max_mask_bytes == 0) {
    throw VieError(ErrorCode::InvalidArgument, "segmentation.max_mask_bytes must be >= 1");
  }
}

void validate_obb_block(const ObbOptions& options) {
  if (options.confidence_threshold.has_value()) {
    require_unit(*options.confidence_threshold, "obb.confidence_threshold");
  }
  if (options.iou_threshold.has_value()) {
    require_unit(*options.iou_threshold, "obb.iou_threshold");
  }
  if (options.max_detections.has_value()) {
    require_count(*options.max_detections, kMinCount, kMaxDetectionsLimit, "obb.max_detections");
  }
  if (options.pre_nms_top_k.has_value()) {
    require_count(*options.pre_nms_top_k, kMinCount, kMaxPreNmsTopK, "obb.pre_nms_top_k");
  }
}

void validate_pose_block(const PoseOptions& options) {
  if (options.confidence_threshold.has_value()) {
    require_unit(*options.confidence_threshold, "pose.confidence_threshold");
  }
  if (options.iou_threshold.has_value()) {
    require_unit(*options.iou_threshold, "pose.iou_threshold");
  }
  if (options.max_detections.has_value()) {
    require_count(*options.max_detections, kMinCount, kMaxDetectionsLimit, "pose.max_detections");
  }
  if (options.pre_nms_top_k.has_value()) {
    require_count(*options.pre_nms_top_k, kMinCount, kMaxPreNmsTopK, "pose.pre_nms_top_k");
  }
}

void validate_classification_block(const ClassificationOptions& options, const ModelManifest* manifest) {
  if (!options.top_k.has_value()) return;
  require_count(*options.top_k, kMinCount, 1000000, "classification.top_k");
  if (manifest != nullptr && manifest->num_classes.has_value() &&
      *options.top_k > *manifest->num_classes) {
    throw VieError(ErrorCode::InvalidArgument,
                   "classification.top_k must be <= num_classes (" +
                       std::to_string(*manifest->num_classes) + ")");
  }
}

template <typename T>
std::optional<T> usable_manifest_value(const ModelManifest& manifest, const char* path,
                                       const std::optional<T>& value) {
  if (!value.has_value()) return std::nullopt;
  const FieldProvenance* provenance = manifest.find_provenance(path);
  if (provenance != nullptr && (provenance->certainty == Certainty::Unresolved ||
                                provenance->certainty == Certainty::Inferred)) {
    return std::nullopt;
  }
  return value;
}

template <typename T>
ResolvedParameter<T> choose(const std::optional<T>& request, const std::optional<T>& model,
                            const std::optional<T>& manifest, T fallback) {
  if (request.has_value()) return ResolvedParameter<T>{*request, ParameterSource::RequestOverride};
  if (model.has_value()) return ResolvedParameter<T>{*model, ParameterSource::ModelDefault};
  if (manifest.has_value()) return ResolvedParameter<T>{*manifest, ParameterSource::Manifest};
  return ResolvedParameter<T>{fallback, ParameterSource::FrameworkDefault};
}

template <typename T>
std::optional<ResolvedParameter<T>> choose_optional(const std::optional<T>& request,
                                                    const std::optional<T>& model,
                                                    const std::optional<T>& manifest) {
  if (request.has_value()) return ResolvedParameter<T>{*request, ParameterSource::RequestOverride};
  if (model.has_value()) return ResolvedParameter<T>{*model, ParameterSource::ModelDefault};
  if (manifest.has_value()) return ResolvedParameter<T>{*manifest, ParameterSource::Manifest};
  return std::nullopt;
}

const std::optional<bool> manifest_multi_label(const ModelManifest& manifest) {
  if (manifest.decoder.class_selection == "all_above_threshold") return true;
  if (manifest.decoder.class_selection == "best_class") return false;
  return std::nullopt;
}

ResolvedDetectionConfig resolve_detection(const ModelManifest& manifest,
                                          const InferenceOptions& model_defaults,
                                          const InferenceOptions& request) {
  const DetectionOptions& model = model_defaults.detection;
  const DetectionOptions& want = request.detection;
  ResolvedDetectionConfig resolved;
  resolved.confidence_threshold =
      choose(want.confidence_threshold, model.confidence_threshold,
             usable_manifest_value(manifest, "nms.confidence_threshold",
                                   manifest.nms.confidence_threshold),
             kFrameworkConfidenceThreshold);
  resolved.iou_threshold = choose(
      want.iou_threshold, model.iou_threshold,
      usable_manifest_value(manifest, "nms.iou_threshold", manifest.nms.iou_threshold),
      kFrameworkIouThreshold);
  resolved.max_detections = choose(
      want.max_detections, model.max_detections,
      usable_manifest_value(manifest, "nms.max_detections", manifest.nms.max_detections),
      kFrameworkMaxDetections);
  resolved.pre_nms_top_k = choose(
      want.pre_nms_top_k, model.pre_nms_top_k,
      usable_manifest_value(manifest, "nms.pre_nms_top_k", manifest.nms.pre_nms_top_k),
      kFrameworkPreNmsTopK);
  std::optional<bool> manifest_class_agnostic;
  if (const std::optional<bool> class_aware = usable_manifest_value(
          manifest, "nms.class_aware", manifest.nms.class_aware);
      class_aware.has_value()) {
    manifest_class_agnostic = !*class_aware;
  }
  resolved.class_agnostic_nms =
      choose(want.class_agnostic_nms, model.class_agnostic_nms, manifest_class_agnostic, false);
  resolved.multi_label =
      choose(want.multi_label, model.multi_label, manifest_multi_label(manifest), false);
  resolved.nms_enabled =
      manifest.nms.enabled.value_or(manifest.decoder.protocol != "rtdetr_detection_raw");
  return resolved;
}

ResolvedSegmentationConfig resolve_segmentation(const ModelManifest& manifest,
                                                const InferenceOptions& model_defaults,
                                                const InferenceOptions& request) {
  const SegmentationOptions& model = model_defaults.segmentation;
  const SegmentationOptions& want = request.segmentation;
  ResolvedSegmentationConfig resolved;
  resolved.confidence_threshold =
      choose(want.confidence_threshold, model.confidence_threshold,
             usable_manifest_value(manifest, "nms.confidence_threshold",
                                   manifest.nms.confidence_threshold),
             kFrameworkConfidenceThreshold);
  resolved.iou_threshold = choose(
      want.iou_threshold, model.iou_threshold,
      usable_manifest_value(manifest, "nms.iou_threshold", manifest.nms.iou_threshold),
      kFrameworkIouThreshold);
  resolved.max_detections = choose(
      want.max_detections, model.max_detections,
      usable_manifest_value(manifest, "nms.max_detections", manifest.nms.max_detections),
      kFrameworkMaxDetections);
  resolved.pre_nms_top_k = choose(
      want.pre_nms_top_k, model.pre_nms_top_k,
      usable_manifest_value(manifest, "nms.pre_nms_top_k", manifest.nms.pre_nms_top_k),
      kFrameworkPreNmsTopK);
  std::optional<bool> manifest_class_agnostic;
  if (const std::optional<bool> class_aware = usable_manifest_value(
          manifest, "nms.class_aware", manifest.nms.class_aware);
      class_aware.has_value()) {
    manifest_class_agnostic = !*class_aware;
  }
  resolved.class_agnostic_nms =
      choose(want.class_agnostic_nms, model.class_agnostic_nms, manifest_class_agnostic, false);
  resolved.retina_masks =
      choose_optional(want.retina_masks, model.retina_masks,
                      usable_manifest_value(manifest, "segmentation.retina_masks",
                                            manifest.segmentation.retina_masks));
  resolved.binary_mask_gt_zero =
      choose_optional(want.binary_mask_gt_zero, model.binary_mask_gt_zero,
                      usable_manifest_value(manifest, "segmentation.binary_mask_gt_zero",
                                            manifest.segmentation.binary_mask_gt_zero));
  resolved.max_mask_bytes =
      choose_optional(want.max_mask_bytes, model.max_mask_bytes,
                      usable_manifest_value(manifest, "segmentation.max_mask_bytes",
                                            manifest.segmentation.max_mask_bytes));
  resolved.nms_enabled = manifest.nms.enabled.value_or(true);
  return resolved;
}

ResolvedObbConfig resolve_obb(const ModelManifest& manifest, const InferenceOptions& model_defaults,
                              const InferenceOptions& request) {
  const ObbOptions& model = model_defaults.obb;
  const ObbOptions& want = request.obb;
  ResolvedObbConfig resolved;
  resolved.confidence_threshold =
      choose(want.confidence_threshold, model.confidence_threshold,
             usable_manifest_value(manifest, "nms.confidence_threshold",
                                   manifest.nms.confidence_threshold),
             kFrameworkConfidenceThreshold);
  resolved.iou_threshold = choose(
      want.iou_threshold, model.iou_threshold,
      usable_manifest_value(manifest, "nms.iou_threshold", manifest.nms.iou_threshold),
      kFrameworkIouThreshold);
  resolved.max_detections = choose(
      want.max_detections, model.max_detections,
      usable_manifest_value(manifest, "nms.max_detections", manifest.nms.max_detections),
      kFrameworkMaxDetections);
  resolved.pre_nms_top_k = choose(
      want.pre_nms_top_k, model.pre_nms_top_k,
      usable_manifest_value(manifest, "nms.pre_nms_top_k", manifest.nms.pre_nms_top_k),
      kFrameworkPreNmsTopK);
  std::optional<bool> manifest_class_agnostic;
  if (const std::optional<bool> class_aware = usable_manifest_value(
          manifest, "nms.class_aware", manifest.nms.class_aware);
      class_aware.has_value()) {
    manifest_class_agnostic = !*class_aware;
  }
  resolved.class_agnostic_nms =
      choose(want.class_agnostic_nms, model.class_agnostic_nms, manifest_class_agnostic, false);
  resolved.nms_enabled = manifest.nms.enabled.value_or(true);
  return resolved;
}

ResolvedPoseConfig resolve_pose(const ModelManifest& manifest, const InferenceOptions& model_defaults,
                                const InferenceOptions& request) {
  const PoseOptions& model = model_defaults.pose;
  const PoseOptions& want = request.pose;
  ResolvedPoseConfig resolved;
  resolved.confidence_threshold =
      choose(want.confidence_threshold, model.confidence_threshold,
             usable_manifest_value(manifest, "nms.confidence_threshold",
                                   manifest.nms.confidence_threshold),
             kFrameworkConfidenceThreshold);
  resolved.iou_threshold = choose(
      want.iou_threshold, model.iou_threshold,
      usable_manifest_value(manifest, "nms.iou_threshold", manifest.nms.iou_threshold),
      kFrameworkIouThreshold);
  resolved.max_detections = choose(
      want.max_detections, model.max_detections,
      usable_manifest_value(manifest, "nms.max_detections", manifest.nms.max_detections),
      kFrameworkMaxDetections);
  resolved.pre_nms_top_k = choose(
      want.pre_nms_top_k, model.pre_nms_top_k,
      usable_manifest_value(manifest, "nms.pre_nms_top_k", manifest.nms.pre_nms_top_k),
      kFrameworkPreNmsTopK);
  std::optional<bool> manifest_class_agnostic;
  if (const std::optional<bool> class_aware = usable_manifest_value(
          manifest, "nms.class_aware", manifest.nms.class_aware);
      class_aware.has_value()) {
    manifest_class_agnostic = !*class_aware;
  }
  resolved.class_agnostic_nms =
      choose(want.class_agnostic_nms, model.class_agnostic_nms, manifest_class_agnostic, false);
  resolved.nms_enabled = manifest.nms.enabled.value_or(true);
  return resolved;
}

ResolvedClassificationConfig resolve_classification(const ModelManifest& manifest,
                                                    const InferenceOptions& model_defaults,
                                                    const InferenceOptions& request) {
  int fallback = kFrameworkClassificationTopK;
  if (manifest.num_classes.has_value() && *manifest.num_classes > 0) {
    fallback = std::min(fallback, *manifest.num_classes);
  }
  ResolvedClassificationConfig resolved;
  // There is no manifest classification.top_k field. The manifest can supply
  // num_classes (which bounds the default) but never the value itself.
  resolved.top_k = choose(request.classification.top_k, model_defaults.classification.top_k,
                          std::optional<int>{}, fallback);
  return resolved;
}

// Canonical, locale-independent field serialization. Streams are imbued with
// the classic locale and use float max_digits10, so a digit can never depend on
// the process locale and two distinct effective floats can never collapse into
// the same text. -0.0 is folded into +0.0 because every decoder comparison
// treats them identically: they are the same effective configuration and must
// produce the same batch key.
template <typename T>
void append_value(std::ostringstream& out, const ResolvedParameter<T>& parameter) {
  if constexpr (std::is_floating_point_v<T>) {
    const T value = parameter.value == T{0} ? T{0} : parameter.value;
    out << value << ',';
  } else {
    out << parameter.value << ',';
  }
}

// An absent optional is not the same configuration as an explicit false/0: the
// mask fields have no framework default, so the absent form is rejected later
// while the explicit form runs. The presence marker keeps those two apart.
template <typename T>
void append_optional(std::ostringstream& out,
                     const std::optional<ResolvedParameter<T>>& parameter) {
  if (!parameter.has_value()) {
    out << "unset,";
    return;
  }
  append_value(out, *parameter);
}

}  // namespace

const char* to_string(ParameterSource source) {
  switch (source) {
    case ParameterSource::FrameworkDefault:
      return "framework_default";
    case ParameterSource::Manifest:
      return "manifest";
    case ParameterSource::ModelDefault:
      return "model_default";
    case ParameterSource::RequestOverride:
      return "request_override";
  }
  return "framework_default";
}

bool DetectionOptions::empty() const noexcept {
  return !confidence_threshold.has_value() && !iou_threshold.has_value() &&
         !max_detections.has_value() && !pre_nms_top_k.has_value() &&
         !class_agnostic_nms.has_value() && !multi_label.has_value();
}

bool SegmentationOptions::empty() const noexcept {
  return !confidence_threshold.has_value() && !iou_threshold.has_value() &&
         !max_detections.has_value() && !pre_nms_top_k.has_value() &&
         !class_agnostic_nms.has_value() && !retina_masks.has_value() &&
         !binary_mask_gt_zero.has_value() && !max_mask_bytes.has_value();
}

bool ObbOptions::empty() const noexcept {
  return !confidence_threshold.has_value() && !iou_threshold.has_value() &&
         !max_detections.has_value() && !pre_nms_top_k.has_value() &&
         !class_agnostic_nms.has_value();
}

bool PoseOptions::empty() const noexcept {
  return !confidence_threshold.has_value() && !iou_threshold.has_value() &&
         !max_detections.has_value() && !pre_nms_top_k.has_value() &&
         !class_agnostic_nms.has_value();
}

bool ClassificationOptions::empty() const noexcept { return !top_k.has_value(); }

bool InferenceOptions::empty() const noexcept {
  return detection.empty() && segmentation.empty() && obb.empty() && pose.empty() &&
         classification.empty();
}

const ResolvedDetectionConfig& ResolvedInferenceConfig::require_detection() const {
  if (!detection.has_value()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "resolved inference config does not contain a detection block");
  }
  return *detection;
}

const ResolvedSegmentationConfig& ResolvedInferenceConfig::require_segmentation() const {
  if (!segmentation.has_value()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "resolved inference config does not contain a segmentation block");
  }
  return *segmentation;
}

const ResolvedObbConfig& ResolvedInferenceConfig::require_obb() const {
  if (!obb.has_value()) {
    throw VieError(ErrorCode::InvalidArgument, "resolved inference config does not contain an obb block");
  }
  return *obb;
}

const ResolvedPoseConfig& ResolvedInferenceConfig::require_pose() const {
  if (!pose.has_value()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "resolved inference config does not contain a pose block");
  }
  return *pose;
}

const ResolvedClassificationConfig& ResolvedInferenceConfig::require_classification() const {
  if (!classification.has_value()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "resolved inference config does not contain a classification block");
  }
  return *classification;
}

std::string ResolvedInferenceConfig::fingerprint() const {
  std::ostringstream out;
  // Classical locale and float max_digits10 (9) significant digits: a float
  // round-trips through this text, so two distinct effective values always
  // serialize differently, independent of the process locale.
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<float>::max_digits10);
  out << to_string(task) << '|' << protocol << '|';
  // Each present block is serialized; the resolver sets exactly one, but a
  // hand-built config with an extra block must not collide with the same
  // values without it.
  if (detection.has_value()) {
    out << "det:";
    append_value(out, detection->confidence_threshold);
    append_value(out, detection->iou_threshold);
    append_value(out, detection->max_detections);
    append_value(out, detection->pre_nms_top_k);
    append_value(out, detection->class_agnostic_nms);
    append_value(out, detection->multi_label);
    out << detection->nms_enabled << ';';
  }
  if (segmentation.has_value()) {
    out << "seg:";
    append_value(out, segmentation->confidence_threshold);
    append_value(out, segmentation->iou_threshold);
    append_value(out, segmentation->max_detections);
    append_value(out, segmentation->pre_nms_top_k);
    append_value(out, segmentation->class_agnostic_nms);
    append_optional(out, segmentation->retina_masks);
    append_optional(out, segmentation->binary_mask_gt_zero);
    append_optional(out, segmentation->max_mask_bytes);
    out << segmentation->nms_enabled << ';';
  }
  if (obb.has_value()) {
    out << "obb:";
    append_value(out, obb->confidence_threshold);
    append_value(out, obb->iou_threshold);
    append_value(out, obb->max_detections);
    append_value(out, obb->pre_nms_top_k);
    append_value(out, obb->class_agnostic_nms);
    out << obb->nms_enabled << ';';
  }
  if (pose.has_value()) {
    out << "pose:";
    append_value(out, pose->confidence_threshold);
    append_value(out, pose->iou_threshold);
    append_value(out, pose->max_detections);
    append_value(out, pose->pre_nms_top_k);
    append_value(out, pose->class_agnostic_nms);
    out << pose->nms_enabled << ';';
  }
  if (classification.has_value()) {
    out << "cls:";
    append_value(out, classification->top_k);
  }
  if (!detection.has_value() && !segmentation.has_value() && !obb.has_value() &&
      !pose.has_value() && !classification.has_value()) {
    out << "none";
  }
  return out.str();
}

void validate_inference_options(TaskType task, const InferenceOptions& options,
                                const ModelManifest* manifest) {
  if (task == TaskType::Unknown) {
    if (!options.empty()) {
      throw VieError(ErrorCode::InvalidArgument,
                     "inference options cannot be validated for an unknown task");
    }
    return;
  }
  require_no_other_blocks(task, options);
  switch (task) {
    case TaskType::Detection:
      validate_detection_block(options.detection, manifest);
      break;
    case TaskType::Segmentation:
      validate_segmentation_block(options.segmentation);
      break;
    case TaskType::Obb:
      validate_obb_block(options.obb);
      break;
    case TaskType::Pose:
      validate_pose_block(options.pose);
      break;
    case TaskType::Classification:
      validate_classification_block(options.classification, manifest);
      break;
    case TaskType::SemanticSegmentation:
      // No configurable post-processing parameter exists. require_no_other_blocks
      // already rejected every block, so an empty option set is the only input.
      break;
    case TaskType::Unknown:
      break;
  }
}

ResolvedInferenceConfig resolve_inference_config(TaskType task, const ModelManifest& manifest,
                                                const InferenceOptions& model_defaults,
                                                const InferenceOptions& request) {
  if (manifest.task != task) {
    throw VieError(ErrorCode::InvalidArgument,
                   "manifest task does not match the task being resolved");
  }
  validate_inference_options(task, model_defaults, &manifest);
  validate_inference_options(task, request, &manifest);

  ResolvedInferenceConfig resolved;
  resolved.task = task;
  resolved.protocol = manifest.decoder.protocol;
  switch (task) {
    case TaskType::Detection:
      resolved.detection = resolve_detection(manifest, model_defaults, request);
      break;
    case TaskType::Segmentation:
      resolved.segmentation = resolve_segmentation(manifest, model_defaults, request);
      break;
    case TaskType::Obb:
      resolved.obb = resolve_obb(manifest, model_defaults, request);
      break;
    case TaskType::Pose:
      resolved.pose = resolve_pose(manifest, model_defaults, request);
      break;
    case TaskType::Classification:
      resolved.classification = resolve_classification(manifest, model_defaults, request);
      break;
    case TaskType::SemanticSegmentation:
      break;
    case TaskType::Unknown:
      throw VieError(ErrorCode::InvalidArgument, "cannot resolve inference options for an unknown task");
  }
  // Every effective value is checked again, whichever layer supplied it. A
  // manifest value can pass schema validation and still break the decoder
  // contract (for example max_detections above the supported bound).
  if (resolved.detection.has_value()) {
    const ResolvedDetectionConfig& block = *resolved.detection;
    require_unit(block.confidence_threshold.value, "confidence_threshold");
    require_unit(block.iou_threshold.value, "iou_threshold");
    require_count(block.max_detections.value, kMinCount, kMaxDetectionsLimit, "max_detections");
    require_count(block.pre_nms_top_k.value, kMinCount, kMaxPreNmsTopK, "pre_nms_top_k");
  } else if (resolved.segmentation.has_value()) {
    const ResolvedSegmentationConfig& block = *resolved.segmentation;
    require_unit(block.confidence_threshold.value, "confidence_threshold");
    require_unit(block.iou_threshold.value, "iou_threshold");
    require_count(block.max_detections.value, kMinCount, kMaxDetectionsLimit, "max_detections");
    require_count(block.pre_nms_top_k.value, kMinCount, kMaxPreNmsTopK, "pre_nms_top_k");
  } else if (resolved.obb.has_value()) {
    const ResolvedObbConfig& block = *resolved.obb;
    require_unit(block.confidence_threshold.value, "confidence_threshold");
    require_unit(block.iou_threshold.value, "iou_threshold");
    require_count(block.max_detections.value, kMinCount, kMaxDetectionsLimit, "max_detections");
    require_count(block.pre_nms_top_k.value, kMinCount, kMaxPreNmsTopK, "pre_nms_top_k");
  } else if (resolved.pose.has_value()) {
    const ResolvedPoseConfig& block = *resolved.pose;
    require_unit(block.confidence_threshold.value, "confidence_threshold");
    require_unit(block.iou_threshold.value, "iou_threshold");
    require_count(block.max_detections.value, kMinCount, kMaxDetectionsLimit, "max_detections");
    require_count(block.pre_nms_top_k.value, kMinCount, kMaxPreNmsTopK, "pre_nms_top_k");
  } else if (resolved.classification.has_value()) {
    require_count(resolved.classification->top_k.value, kMinCount, 1000000, "top_k");
  }
  return resolved;
}

}  // namespace vie