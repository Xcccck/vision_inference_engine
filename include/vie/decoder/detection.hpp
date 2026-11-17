#pragma once

#include "vie/core/device_context.hpp"
#include "vie/core/tensor.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vie {

// Original-image pixels. x1, y1 is the top-left corner and x2, y2 is the
// bottom-right corner, after one GeometryTransform mapping and clipping to
// [0, width] x [0, height].
struct BoundingBox {
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;
};

struct Detection {
  BoundingBox box;
  float confidence = 0.f;
  int class_id = -1;
  // Index of this box in the model output, before confidence sorting. RT-DETR
  // CPU and CUDA both write that query index. It is not the position of this
  // Detection inside DetectionResult::detections. YOLOv8 and the mock decoder
  // are not query decoders and leave it at -1.
  int query_index = -1;
};

// One image inside a DetectionBatch. detections live in host memory. A CUDA
// decoder copies only the kept boxes back before filling this struct.
//
// The four timing fields below belong to this image's decode, not to any
// Detection. They are zero unless that decoder measured them. YOLOv8 does not
// write sort_ms or d2h_bytes; its sort stays inside its own timing.
//
// RT-DETR CUDA, one decode_batch call, legacy default stream:
//   gpu_execute_ms  decode kernel only
//   sort_ms         Thrust sort of the kept candidates; 0 when fewer than two
//                   candidates remain, because that sort is skipped
//   d2h_ms          kept-detection payload copy only
//   d2h_bytes       that payload, copied * 28 bytes
//                   (x1, y1, x2, y2, confidence, class_id, query_index)
// The 12-byte counter read (fault, passed, kept) is control metadata. It is
// not part of d2h_bytes. Neither field is the raw tensor. [1, 300, 84] fp32
// is 100800 elements and 403200 bytes, and that tensor stays on the device.
struct DetectionResult {
  std::vector<Detection> detections;
  int original_width = 0;
  int original_height = 0;
  DeviceType execution_device = DeviceType::Cpu;
  int execution_device_id = 0;
  std::string protocol;
  bool nms_applied = false;
  bool coordinates_mapped = false;
  int candidates_after_filter = 0;
  int candidates_before_nms = 0;
  double sort_ms = 0;
  std::size_t d2h_bytes = 0;
  double gpu_execute_ms = 0;
  double d2h_ms = 0;
};

struct DetectionBatch {
  std::vector<DetectionResult> results;
};

enum class BoxDecode { DecodedCxcywh, Dfl, DecodedNormalizedCxcywh };

enum class ClassSelection { BestClass, AllAboveThreshold };

// One feature level. Cells are row-major, x fastest. The anchor is the cell
// center (x + 0.5, y + 0.5), and dist2bbox multiplies by stride afterwards.
// Levels are concatenated in the order given. The recognizer does not prove
// this order for an exported tensor, so it has to be supplied explicitly.
struct FeatureLevel {
  int stride = 0;
  int grid_width = 0;
  int grid_height = 0;
};

struct DecoderConfig {
  std::string output_name;
  int channel_axis = -1;
  int box_channels = 0;
  int num_classes = 0;
  BoxDecode box_decode = BoxDecode::DecodedCxcywh;
  int reg_max = 0;
  std::vector<FeatureLevel> levels;
  std::optional<bool> scores_are_probabilities;
  std::optional<ClassSelection> class_selection;
  std::optional<float> confidence_threshold;
  std::optional<bool> nms_enabled;
  std::optional<bool> class_aware_nms;
  std::optional<float> iou_threshold;
  std::optional<int> max_detections;
  std::optional<int> pre_nms_top_k;
  // Query axis length for a decoder that reads it from the manifest. YOLOv8
  // leaves this unset and does not consult it.
  std::optional<int> num_queries;
  // Zero means the tensor has no mask coefficients. Segmentation sets this
  // explicitly. Detection configs leave it at zero, so their channel check is
  // unchanged.
  int mask_channels = 0;
  std::string prototype_name;
  // 1 is NCHW [1, C, H, W]. Unset stays unresolved for segmentation.
  std::optional<int> prototype_channel_axis;
  // Unset stays unresolved. This decoder does not pick a default.
  std::optional<bool> retina_masks;
  // True requests the M > 0 binary test. Unset stays unresolved.
  std::optional<bool> binary_mask_gt_zero;
  std::optional<std::size_t> max_mask_bytes;
};

struct DecoderInput {
  std::map<std::string, Tensor> outputs;
  std::optional<GeometryTransform> geometry;
};

struct NmsCandidate {
  BoundingBox box;
  float score = 0.f;
  int class_id = -1;
  int index = 0;
};

void validate_decoder_config(const DecoderConfig& config);

// Stable softmax expectation. logits length is reg_max. Throws on a non-finite
// logit or reg_max < 2.
float dfl_expectation(const float* logits, int reg_max);

// l, t, r, b are bin expectations at one anchor. The result is xyxy in model
// input pixels.
void dist2bbox(float left, float top, float right, float bottom, float anchor_x, float anchor_y,
               float stride, float& x1, float& y1, float& x2, float& y2);

void cxcywh_to_xyxy(float cx, float cy, float width, float height, float& x1, float& y1, float& x2,
                    float& y2);

// One mapping. Uses scale_x and scale_y, never uniform_scale. Clips to the
// original image. Returns false when the clipped box has no area.
bool map_box_to_original(BoundingBox& box, const GeometryTransform& geometry);

// candidates must already be sorted by score descending, then index ascending.
// IoU >= threshold suppresses. Returns kept positions into that sorted list.
std::vector<int> greedy_nms(const std::vector<NmsCandidate>& sorted, float iou_threshold,
                            bool class_aware, int max_detections);

DecoderConfig decoder_config_from_manifest(const ModelManifest& manifest);

}  // namespace vie
