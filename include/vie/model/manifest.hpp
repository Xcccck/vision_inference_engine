#pragma once

#include "vie/core/types.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vie {

struct InputDesc {
  std::string name;
  std::vector<std::int64_t> shape;
  DataType dtype = DataType::Unknown;
  Layout layout = Layout::Unknown;
  std::string color;
  std::string normalization;
  std::string resize;
  // Optional. Empty means the interpolation was not supplied. Image preprocess
  // refuses to run until it is set; tensor execute does not read it.
  std::string interpolation;
  // Optional letterbox border sample, in the same numeric space as the image
  // before normalization. Unset does not block tensor execute.
  std::optional<float> padding_value;
  // Shortest-side resize followed by center crop. Unset unless resize is
  // "shortest_side". The model input shape is the crop, not the short side.
  std::optional<int> short_side;
  std::optional<int> crop_width;
  std::optional<int> crop_height;
  // Used only when normalization is "explicit": y = (x * scale - mean) / std.
  // Named normalization tokens do not read these fields.
  std::optional<float> scale;
  std::vector<float> mean;
  std::vector<float> std;
};

struct OutputDesc {
  std::string name;
  std::vector<std::int64_t> shape;
  DataType dtype = DataType::Unknown;
  Layout layout = Layout::Unknown;
  std::string semantic;
};

struct DecoderDesc {
  std::string protocol;
  std::string box_format;
  // "decoded_cxcywh" means the graph already produced center/size pixels.
  // "decoded_normalized_cxcywh" means those center/size values are in the
  // unit square, not model-input pixels. "dfl" means the box channels are
  // still a distribution. Empty is unresolved.
  std::string box_decode;
  std::optional<bool> scores_are_probabilities;
  std::optional<int> box_channels;
  std::optional<int> channel_axis;
  std::optional<int> reg_max;
  // Query count selected inside the graph. This is not pre_nms_top_k.
  // Unset for heads that do not prove a query axis. Omitted from JSON when unset.
  std::optional<int> num_queries;
  // "best_class" or "all_above_threshold". Empty is unresolved.
  std::string class_selection;
};

struct SegmentationDesc {
  std::optional<int> mask_channels;
  std::string prototype_name;
  Layout prototype_layout = Layout::Unknown;
  std::optional<int> prototype_channel_axis;
  std::optional<int> prototype_height;
  std::optional<int> prototype_width;
  std::string mask_coefficient_activation;
  std::string prototype_activation;
  std::optional<bool> has_objectness;
  std::vector<std::string> class_names;
  std::optional<bool> retina_masks;
  std::optional<bool> binary_mask_gt_zero;
  std::optional<std::size_t> max_mask_bytes;
};

// YOLOv8-Pose output contract. Runtime thresholds stay on the call, not here.
// keypoint_count comes from the proved keypoint convs, not from a filename.
// keypoint_layout "xy_confidence" means groups of x, y, and the third-channel
// sigmoid. That sigmoid is not a COCO visibility label.
struct PoseDesc {
  std::string output_name;
  std::optional<int> channel_axis;
  std::optional<int> keypoint_count;
  std::optional<int> keypoint_dimensions;
  std::string keypoint_layout;
  std::string score_type;
};

// YOLOv8-OBB output contract. Runtime thresholds stay on the call, not here.
// angle_range is the image of a proven affine, not a guess from the channel count.
struct ObbDesc {
  std::string output_name;
  std::optional<int> channel_axis;
  std::string angle_unit;
  std::string angle_range;
  std::string score_type;
};

// Optional single-label classification head. Absent fields stay empty and are
// not inferred from an output shape. Omitted from JSON when every field is unset
// and the task is not classification.
struct ClassificationDesc {
  std::string output_name;
  std::optional<int> class_axis;
  // "logits" or "probabilities". Empty is unresolved. The first decoder only
  // executes logits.
  std::string score_type;
  std::vector<std::string> class_names;
};

struct NmsDesc {
  std::optional<bool> enabled;
  std::optional<float> confidence_threshold;
  std::optional<float> iou_threshold;
  std::optional<int> max_detections;
  std::optional<int> pre_nms_top_k;
  std::optional<bool> class_aware;
};

// Schema 1.0 description of a model. Absent semantic values stay empty or
// nullopt and are recorded as Unresolved. Parsing does not fill them in.
class ModelManifest {
 public:
  std::string schema_version = "1.0";
  std::string model_name;
  std::string model_family;
  std::string model_version;
  TaskType task = TaskType::Unknown;
  ModelFormat model_format = ModelFormat::Unspecified;
  std::string model_path;
  std::optional<int> num_classes;
  InputDesc input;
  std::vector<OutputDesc> outputs;
  DecoderDesc decoder;
  NmsDesc nms;
  SegmentationDesc segmentation;
  ClassificationDesc classification;
  ObbDesc obb;
  PoseDesc pose;
  std::map<std::string, FieldProvenance> provenance;

  static ModelManifest from_json_string(std::string_view json_text);
  static ModelManifest load_file(const std::string& path);

  std::string to_json_string(int indent = 2) const;
  void save_file(const std::string& path) const;

  // Fills provenance entries that the caller left absent, using the sentinel
  // state of each field. Existing entries are kept.
  void ensure_provenance();

  const FieldProvenance* find_provenance(std::string_view path) const;
};

struct FieldState {
  bool set = false;
  FieldProvenance provenance;
};

std::vector<std::string> manifest_field_paths(const ModelManifest& manifest);
bool is_manifest_field_path(const ModelManifest& manifest, std::string_view path);
bool field_is_set(const ModelManifest& manifest, std::string_view path);
FieldState inspect_field(const ModelManifest& manifest, std::string_view path);

}  // namespace vie
