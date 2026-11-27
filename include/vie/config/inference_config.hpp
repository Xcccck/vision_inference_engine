#pragma once

#include "vie/core/types.hpp"
#include "vie/model/manifest.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace vie {

// Where the effective value of one inference parameter came from. The order is
// also the resolution order used by resolve_inference_config(): a request
// overrides a model default, which overrides a usable manifest value, which
// overrides the framework default.
enum class ParameterSource {
  FrameworkDefault,
  Manifest,
  ModelDefault,
  RequestOverride,
};

const char* to_string(ParameterSource source);

// One effective parameter plus its provenance. The value is always validated
// before it reaches a decoder.
template <typename T>
struct ResolvedParameter {
  T value{};
  ParameterSource source = ParameterSource::FrameworkDefault;
};

// Detection post-processing policy. Every field is optional at the call site:
// an unset field inherits the next lower layer. The decoder still owns the
// structural output contract (box decode, class selection support, NMS
// support); these fields only change policy.
struct DetectionOptions {
  std::optional<float> confidence_threshold;
  std::optional<float> iou_threshold;
  std::optional<int> max_detections;
  std::optional<int> pre_nms_top_k;
  std::optional<bool> class_agnostic_nms;
  // YOLOv8 only: false keeps one class per candidate (best_class), true expands
  // every class above the threshold (all_above_threshold). RT-DETR rejects true.
  std::optional<bool> multi_label;

  bool empty() const noexcept;
};

// Instance-segmentation policy. The mask shape/budget fields are not detection
// parameters: they describe the mask output and are only read by yolov8_seg_raw.
struct SegmentationOptions {
  std::optional<float> confidence_threshold;
  std::optional<float> iou_threshold;
  std::optional<int> max_detections;
  std::optional<int> pre_nms_top_k;
  std::optional<bool> class_agnostic_nms;
  std::optional<bool> retina_masks;
  std::optional<bool> binary_mask_gt_zero;
  std::optional<std::size_t> max_mask_bytes;

  bool empty() const noexcept;
};

// Oriented-box policy. iou_threshold is the rotated-box IoU used by rotated
// NMS, never an axis-aligned IoU.
struct ObbOptions {
  std::optional<float> confidence_threshold;
  std::optional<float> iou_threshold;
  std::optional<int> max_detections;
  std::optional<int> pre_nms_top_k;
  std::optional<bool> class_agnostic_nms;

  bool empty() const noexcept;
};

// Pose policy. iou_threshold applies to the person-box NMS. The current decoder
// has no keypoint-score filter, so no keypoint threshold is exposed.
struct PoseOptions {
  std::optional<float> confidence_threshold;
  std::optional<float> iou_threshold;
  std::optional<int> max_detections;
  std::optional<int> pre_nms_top_k;
  std::optional<bool> class_agnostic_nms;

  bool empty() const noexcept;
};

// Classification policy. The current decoder has no confidence filter and
// returns only the top-k labels, so top_k is the only supported parameter.
struct ClassificationOptions {
  std::optional<int> top_k;

  bool empty() const noexcept;
};

// Per-call inference policy. Only the block matching the loaded task is read;
// a non-empty block for another task is rejected instead of being ignored.
// Semantic segmentation has no configurable post-processing parameter.
struct InferenceOptions {
  DetectionOptions detection;
  SegmentationOptions segmentation;
  ObbOptions obb;
  PoseOptions pose;
  ClassificationOptions classification;

  bool empty() const noexcept;
};

struct ResolvedDetectionConfig {
  ResolvedParameter<float> confidence_threshold;
  ResolvedParameter<float> iou_threshold;
  ResolvedParameter<int> max_detections;
  ResolvedParameter<int> pre_nms_top_k;
  ResolvedParameter<bool> class_agnostic_nms;
  ResolvedParameter<bool> multi_label;
  // Structural: the graph's decoder contract, not a user policy knob.
  bool nms_enabled = true;
};

struct ResolvedSegmentationConfig {
  ResolvedParameter<float> confidence_threshold;
  ResolvedParameter<float> iou_threshold;
  ResolvedParameter<int> max_detections;
  ResolvedParameter<int> pre_nms_top_k;
  ResolvedParameter<bool> class_agnostic_nms;
  std::optional<ResolvedParameter<bool>> retina_masks;
  std::optional<ResolvedParameter<bool>> binary_mask_gt_zero;
  std::optional<ResolvedParameter<std::size_t>> max_mask_bytes;
  bool nms_enabled = true;
};

struct ResolvedObbConfig {
  ResolvedParameter<float> confidence_threshold;
  ResolvedParameter<float> iou_threshold;
  ResolvedParameter<int> max_detections;
  ResolvedParameter<int> pre_nms_top_k;
  ResolvedParameter<bool> class_agnostic_nms;
  bool nms_enabled = true;
};

struct ResolvedPoseConfig {
  ResolvedParameter<float> confidence_threshold;
  ResolvedParameter<float> iou_threshold;
  ResolvedParameter<int> max_detections;
  ResolvedParameter<int> pre_nms_top_k;
  ResolvedParameter<bool> class_agnostic_nms;
  bool nms_enabled = true;
};

struct ResolvedClassificationConfig {
  ResolvedParameter<int> top_k;
};

// Task-tagged effective configuration. Exactly one optional block is set for a
// known task. The fingerprint covers every effective value and is used by the
// async batch aggregator to keep different policies in different batches.
struct ResolvedInferenceConfig {
  TaskType task = TaskType::Unknown;
  std::string protocol;
  std::optional<ResolvedDetectionConfig> detection;
  std::optional<ResolvedSegmentationConfig> segmentation;
  std::optional<ResolvedObbConfig> obb;
  std::optional<ResolvedPoseConfig> pose;
  std::optional<ResolvedClassificationConfig> classification;

  const ResolvedDetectionConfig& require_detection() const;
  const ResolvedSegmentationConfig& require_segmentation() const;
  const ResolvedObbConfig& require_obb() const;
  const ResolvedPoseConfig& require_pose() const;
  const ResolvedClassificationConfig& require_classification() const;

  std::string fingerprint() const;
};

// Framework defaults. These are VIE policy defaults, not values observed in a
// model. They are used only when no request, model, or usable manifest value is
// present.
inline constexpr float kFrameworkConfidenceThreshold = 0.25f;
inline constexpr float kFrameworkIouThreshold = 0.45f;
inline constexpr int kFrameworkMaxDetections = 300;
inline constexpr int kFrameworkPreNmsTopK = 4096;
inline constexpr int kFrameworkClassificationTopK = 5;
inline constexpr int kMaxDetectionsLimit = 100000;

// Validates one layer on its own. manifest may be null when only the option
// shapes can be checked. Throws InvalidArgument for a non-finite threshold, an
// out-of-range count, or a block that does not belong to task. RT-DETR rejects
// NMS/IoU/top-k/multi-label options it cannot honor.
void validate_inference_options(TaskType task, const InferenceOptions& options,
                                const ModelManifest* manifest = nullptr);

// Resolves request > model default > usable manifest value > framework default.
// A manifest value is usable only when its provenance is Observed, Metadata, or
// User (or when it has no provenance entry and is therefore treated as User).
// Unresolved and Inferred manifest values never participate in resolution.
// Throws InvalidArgument/Unsupported for invalid or task-inapplicable values.
ResolvedInferenceConfig resolve_inference_config(TaskType task, const ModelManifest& manifest,
                                                const InferenceOptions& model_defaults,
                                                const InferenceOptions& request);

}  // namespace vie