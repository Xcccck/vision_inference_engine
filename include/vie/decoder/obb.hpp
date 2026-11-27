#pragma once

#include "vie/decoder/detection.hpp"

#include <vector>

namespace vie {

// One oriented box. cx, cy, width, and height are pixels. angle is radians.
// The width edge is (cos(angle), sin(angle)) and the height edge is
// (-sin(angle), cos(angle)), the same rotation used by Ultralytics
// xywhr2xyxyxyxy. In an image where +x points right and +y points down, a
// positive angle turns the width edge clockwise on the screen.
//
// YOLOv8-OBB writes angle in [-pi/4, 3pi/4]. (width, height, angle) and
// (height, width, angle + pi/2) are the same corner set. Results keep the
// decoded pair; they are not rewritten to a single canonical pair.
struct OrientedBox {
  float cx = 0.f;
  float cy = 0.f;
  float width = 0.f;
  float height = 0.f;
  float angle = 0.f;
};

struct ObbObject {
  int class_id = -1;
  float score = 0.f;
  OrientedBox box;
  // Anchor index in the model output, before the score sort.
  int anchor_index = -1;
};

struct ObbResult {
  std::vector<ObbObject> objects;
  int image_width = 0;
  int image_height = 0;
  int num_classes = 0;
  // The device that ran field decode, the map, Polygon IoU, and rotated NMS.
  // A CPU request uses DeviceType::Cpu; a CUDA request uses DeviceType::Cuda
  // and only the kept objects are copied back.
  DeviceType execution_device = DeviceType::Cpu;
  int execution_device_id = 0;
  std::string protocol;
  std::size_t output_bytes = 0;
  std::size_t d2h_bytes = 0;
  double preprocess_ms = 0;
  double execute_ms = 0;
  double d2h_ms = 0;
  double validate_ms = 0;
  // CUDA only: the deterministic score sort and the corner map in front of
  // NMS. The CPU decoder keeps both inside its decode_ms component sum and
  // leaves these zero.
  double sort_ms = 0;
  double map_ms = 0;
  double nms_ms = 0;
  double decode_ms = 0;
  int candidates_after_filter = 0;
  int candidates_before_nms = 0;
  // CUDA only: Polygon IoU evaluations the greedy NMS performed, which is the
  // number of candidate pairs that passed the class filter. The CPU decoder
  // leaves it zero.
  long long iou_calls = 0;
};

struct ObbBatch {
  std::vector<ObbResult> results;
};

// Caller-supplied policy. None of these values is read back from the manifest.
struct ObbDecodeConfig {
  std::string output_name;
  int channel_axis = -1;
  int num_classes = 0;
  std::optional<ClassSelection> class_selection;
  std::optional<float> confidence_threshold;
  std::optional<bool> nms_enabled;
  std::optional<bool> class_aware;
  std::optional<float> iou_threshold;
  std::optional<int> max_detections;
  std::optional<int> pre_nms_top_k;
};

struct ObbCandidate {
  OrientedBox box;
  float score = 0.f;
  int class_id = -1;
  int anchor_index = 0;
};

void validate_obb_decode_config(const ObbDecodeConfig& config);

// Phase 12: build the decoder policy from the effective config resolved for
// this request. This is the only place that maps ResolvedObbConfig onto
// ObbDecodeConfig, so a CPU and a future CUDA OBB decoder receive the same
// semantics. Rejects a config resolved for a different task or protocol.
ObbDecodeConfig obb_decode_config_from_resolved(const ModelManifest& manifest,
                                               const ResolvedInferenceConfig& resolved);

// Polygon intersection of the two rotated rectangles. Rejects non-finite
// values and non-positive width or height. The value is the true intersection
// area over the union area, not an axis-aligned approximation and not ProbIoU.
float rotated_intersection_over_union(const OrientedBox& left, const OrientedBox& right);

// sorted is score descending, then anchor_index ascending, then class_id
// ascending. IoU >= threshold suppresses. Returns positions into that list.
std::vector<int> rotated_nms(const std::vector<ObbCandidate>& sorted, float iou_threshold, bool class_aware,
                             int max_detections);

// Maps a model-input OBB through GeometryTransform into original-image pixels.
// Isotropic scale (relative |scale_x-scale_y| <= 1e-3, including letterbox
// rounding) keeps the angle and divides width and height by the mean scale.
// A larger gap, including stretch, inverse-maps the four corners and fits the
// minimum-area rectangle of that parallelogram. A result whose corners leave
// the original image is dropped. Partial boxes are not clipped or refit to
// the visible fragment.
bool map_obb_to_original(const OrientedBox& model_box, const GeometryTransform& geometry, OrientedBox& original);

}  // namespace vie
