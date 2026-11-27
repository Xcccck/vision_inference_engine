#pragma once

#include "vie/decoder/detection.hpp"

#include <vector>

namespace vie {

// One keypoint in original-image pixels.
//
// confidence is the model's third keypoint channel after the graph Sigmoid.
// Ultralytics names that channel "visible". It is a probability in [0, 1],
// not a COCO visibility label (0, 1, or 2) and not the person detection score.
// A low value does not remove the person.
//
// inside_image is true when the inverse-mapped point lies inside the original
// image, inclusive, with 1e-3 px of slack. x and y are not clipped to the
// border, including when the point falls outside a center crop.
struct Keypoint {
  float x = 0.f;
  float y = 0.f;
  float confidence = 0.f;
  bool inside_image = false;
};

// One person. cx, cy, width, and height are the axis-aligned box after the
// same per-axis map and clip used by detection. They are not a second
// coordinate convention. keypoints has one entry per configured keypoint;
// the count is not fixed at 17.
struct PoseObject {
  int class_id = -1;
  float score = 0.f;
  float cx = 0.f;
  float cy = 0.f;
  float width = 0.f;
  float height = 0.f;
  std::vector<Keypoint> keypoints;
  // Anchor index in the model output, before the score sort.
  int anchor_index = -1;
};

struct PoseResult {
  std::vector<PoseObject> objects;
  int image_width = 0;
  int image_height = 0;
  int num_classes = 0;
  // The device that ran pose decode and NMS. CPU builds and CPU requests use
  // DeviceType::Cpu; a CUDA request uses DeviceType::Cuda and only the kept
  // objects are copied back.
  DeviceType execution_device = DeviceType::Cpu;
  int execution_device_id = 0;
  std::string protocol;
  std::size_t output_bytes = 0;
  std::size_t d2h_bytes = 0;
  double preprocess_ms = 0;
  double execute_ms = 0;
  double d2h_ms = 0;
  double validate_ms = 0;
  double pose_decode_ms = 0;
  double nms_ms = 0;
  double decode_ms = 0;
  int candidates_after_filter = 0;
  int candidates_before_nms = 0;
};

struct PoseBatch {
  std::vector<PoseResult> results;
};

// Caller-supplied policy plus the structural fields copied from the manifest
// for this call. None of the thresholds are written back to the manifest.
struct PoseDecodeConfig {
  std::string output_name;
  int channel_axis = -1;
  int num_classes = 0;
  int keypoint_count = 0;
  int keypoint_dimensions = 0;
  std::optional<ClassSelection> class_selection;
  std::optional<float> confidence_threshold;
  std::optional<bool> nms_enabled;
  std::optional<bool> class_aware;
  std::optional<float> iou_threshold;
  std::optional<int> max_detections;
  std::optional<int> pre_nms_top_k;
};

struct PoseCandidate {
  BoundingBox box;
  float score = 0.f;
  int class_id = -1;
  int anchor_index = 0;
  std::vector<Keypoint> keypoints;
};

void validate_pose_decode_config(const PoseDecodeConfig& config);

// Phase 12: build the decoder policy from the effective config resolved for
// this request. This is the only place that maps ResolvedPoseConfig onto
// PoseDecodeConfig; the CPU and CUDA decoders share this result.
PoseDecodeConfig pose_decode_config_from_resolved(const ModelManifest& manifest,
                                                 const ResolvedInferenceConfig& resolved);

// Inverse of GeometryTransform for one point. scale_x and scale_y are applied
// separately, including when they differ. The point is not clipped.
void map_keypoint_to_original(float model_x, float model_y, const GeometryTransform& geometry, float& original_x,
                              float& original_y, bool& inside_image);

}  // namespace vie
