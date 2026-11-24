#include "vie/model/manifest_validator.hpp"

#include <cmath>
#include <set>
#include <sstream>

namespace vie {
namespace {

void add_shape_issues(const std::string& path, const std::vector<std::int64_t>& shape,
                      std::vector<SchemaIssue>& issues) {
  for (const std::int64_t dim : shape) {
    if (dim == 0 || dim < -1) {
      issues.push_back({path, "dimension must be positive or -1 when dynamic"});
      return;
    }
  }
}

bool shape_ready(const std::vector<std::int64_t>& shape, const std::string& path,
                 std::vector<SchemaIssue>& unresolved) {
  if (shape.empty()) {
    unresolved.push_back({path, "shape is unresolved"});
    return false;
  }
  // Phase 9.1: allow a single dynamic dimension at axis 0 (the batch axis) so
  // a dynamic-batch ONNX can carry a Manifest with input.shape=[-1, C, H, W]
  // and output.shape=[-1, ...]. Any other dynamic or non-positive axis stays
  // unresolved, because the engine does not yet support dynamic spatial or
  // channel axes.
  const bool batch_dynamic = !shape.empty() && shape[0] == -1 && shape.size() >= 3;
  if (batch_dynamic) {
    for (std::size_t i = 1; i < shape.size(); ++i) {
      if (shape[i] <= 0) {
        unresolved.push_back({path, "only the batch axis may be dynamic"});
        return false;
      }
    }
    return true;
  }
  if (!is_concrete_shape(shape)) {
    unresolved.push_back({path, "dynamic dimension requires a concrete profile"});
    return false;
  }
  return true;
}

}  // namespace

std::string format_issues(const std::vector<SchemaIssue>& issues) {
  std::ostringstream stream;
  for (std::size_t i = 0; i < issues.size(); ++i) {
    if (i != 0) stream << "; ";
    stream << issues[i].path << ": " << issues[i].message;
  }
  return stream.str();
}

ManifestValidation ManifestValidator::validate(const ModelManifest& manifest) const {
  ManifestValidation report;
  if (manifest.schema_version != "1.0") {
    report.schema_issues.push_back(
        {"schema_version", "unsupported schema_version '" + manifest.schema_version + "'"});
  }

  for (const auto& entry : manifest.provenance) {
    if (!is_manifest_field_path(manifest, entry.first)) {
      report.schema_issues.push_back({entry.first, "unknown provenance path"});
    }
  }
  for (const std::string& path : manifest_field_paths(manifest)) {
    const FieldState state = inspect_field(manifest, path);
    if (state.set && state.provenance.certainty == Certainty::Unresolved) {
      report.schema_issues.push_back({path, "value is set but provenance is unresolved"});
    }
    if (!state.set && state.provenance.certainty != Certainty::Unresolved) {
      report.schema_issues.push_back(
          {path, std::string("value is unset but provenance is ") + to_string(state.provenance.certainty)});
    }
  }

  add_shape_issues("input.shape", manifest.input.shape, report.schema_issues);
  for (std::size_t i = 0; i < manifest.outputs.size(); ++i) {
    add_shape_issues("outputs." + std::to_string(i) + ".shape", manifest.outputs[i].shape,
                     report.schema_issues);
  }

  if (manifest.num_classes.has_value() && *manifest.num_classes < 1) {
    report.schema_issues.push_back({"num_classes", "must be >= 1"});
  }
  if (!manifest.input.color.empty() && !is_known_color(manifest.input.color)) {
    report.schema_issues.push_back({"input.color", "unsupported color"});
  }
  if (!manifest.input.normalization.empty() &&
      !is_known_normalization(manifest.input.normalization)) {
    report.schema_issues.push_back({"input.normalization", "unsupported normalization"});
  }
  if (!manifest.input.resize.empty() && !is_known_resize(manifest.input.resize)) {
    report.schema_issues.push_back({"input.resize", "unsupported resize"});
  }
  if (!manifest.input.interpolation.empty() && manifest.input.interpolation != "nearest" &&
      manifest.input.interpolation != "bilinear") {
    report.schema_issues.push_back({"input.interpolation", "unsupported interpolation"});
  }
  if (manifest.input.padding_value.has_value() && !std::isfinite(*manifest.input.padding_value)) {
    report.schema_issues.push_back({"input.padding_value", "must be finite"});
  }
  const bool shortest = manifest.input.resize == "shortest_side";
  const bool crop_fields = manifest.input.short_side.has_value() || manifest.input.crop_width.has_value() ||
                           manifest.input.crop_height.has_value();
  if (crop_fields && !shortest) {
    report.schema_issues.push_back(
        {"input.short_side", "short_side, crop_width, and crop_height require resize shortest_side"});
  }
  if (shortest) {
    if (manifest.input.short_side.has_value() && *manifest.input.short_side < 1) {
      report.schema_issues.push_back({"input.short_side", "must be positive"});
    }
    if (manifest.input.crop_width.has_value() && *manifest.input.crop_width < 1) {
      report.schema_issues.push_back({"input.crop_width", "must be positive"});
    }
    if (manifest.input.crop_height.has_value() && *manifest.input.crop_height < 1) {
      report.schema_issues.push_back({"input.crop_height", "must be positive"});
    }
    if (manifest.input.short_side.has_value() && manifest.input.crop_width.has_value() &&
        manifest.input.crop_height.has_value() &&
        (*manifest.input.crop_width > *manifest.input.short_side ||
         *manifest.input.crop_height > *manifest.input.short_side)) {
      report.schema_issues.push_back(
          {"input.crop_width", "center crop exceeds short_side and cannot fit the resized image"});
    }
    if (manifest.input.crop_width.has_value() && manifest.input.crop_height.has_value() &&
        is_concrete_shape(manifest.input.shape) && manifest.input.shape.size() == 4) {
      const std::int64_t height = manifest.input.layout == Layout::Nhwc ? manifest.input.shape[1]
                                                                        : manifest.input.shape[2];
      const std::int64_t width = manifest.input.layout == Layout::Nhwc ? manifest.input.shape[2]
                                                                       : manifest.input.shape[3];
      if (height != *manifest.input.crop_height || width != *manifest.input.crop_width) {
        report.schema_issues.push_back({"input.shape", "does not match the center crop"});
      }
    }
  }
  const bool explicit_norm = manifest.input.normalization == "explicit";
  const bool norm_fields =
      manifest.input.scale.has_value() || !manifest.input.mean.empty() || !manifest.input.std.empty();
  if (norm_fields && !explicit_norm) {
    report.schema_issues.push_back(
        {"input.scale", "scale, mean, and std require normalization explicit"});
  }
  if (explicit_norm && manifest.input.scale.has_value() && !std::isfinite(*manifest.input.scale)) {
    report.schema_issues.push_back({"input.scale", "must be finite"});
  }
  if (explicit_norm && !manifest.input.mean.empty() && manifest.input.mean.size() != 3) {
    report.schema_issues.push_back({"input.mean", "must have 3 values"});
  }
  if (explicit_norm && !manifest.input.std.empty() && manifest.input.std.size() != 3) {
    report.schema_issues.push_back({"input.std", "must have 3 values"});
  }
  if (explicit_norm) {
    for (float value : manifest.input.mean) {
      if (!std::isfinite(value)) report.schema_issues.push_back({"input.mean", "must be finite"});
    }
    for (float value : manifest.input.std) {
      if (!std::isfinite(value)) {
        report.schema_issues.push_back({"input.std", "must be finite"});
      } else if (value == 0.f) {
        report.schema_issues.push_back({"input.std", "must be non-zero"});
      }
    }
  }
  if (!manifest.decoder.box_format.empty() && !is_known_box_format(manifest.decoder.box_format)) {
    report.schema_issues.push_back({"decoder.box_format", "unsupported box format"});
  }
  if (!manifest.decoder.box_decode.empty() && manifest.decoder.box_decode != "decoded_cxcywh" &&
      manifest.decoder.box_decode != "decoded_normalized_cxcywh" && manifest.decoder.box_decode != "dfl") {
    report.schema_issues.push_back({"decoder.box_decode", "unsupported box decode"});
  }
  if (!manifest.decoder.class_selection.empty() && manifest.decoder.class_selection != "best_class" &&
      manifest.decoder.class_selection != "all_above_threshold") {
    report.schema_issues.push_back({"decoder.class_selection", "unsupported class selection"});
  }
  if (manifest.decoder.channel_axis.has_value() && *manifest.decoder.channel_axis != 1 &&
      *manifest.decoder.channel_axis != 2) {
    report.schema_issues.push_back({"decoder.channel_axis", "must be 1 or 2"});
  }
  if (manifest.nms.pre_nms_top_k.has_value() &&
      (*manifest.nms.pre_nms_top_k < 1 || *manifest.nms.pre_nms_top_k > 4096)) {
    report.schema_issues.push_back({"nms.pre_nms_top_k", "must be in 1..4096"});
  }
  if (manifest.nms.max_detections.has_value() && *manifest.nms.max_detections < 1) {
    report.schema_issues.push_back({"nms.max_detections", "must be >= 1"});
  }

  auto check_unit = [&](const char* path, const std::optional<float>& value) {
    if (!value.has_value()) return;
    if (!std::isfinite(*value) || *value < 0.f || *value > 1.f) {
      report.schema_issues.push_back({path, "must be in the inclusive range [0, 1]"});
    }
  };
  check_unit("nms.confidence_threshold", manifest.nms.confidence_threshold);
  check_unit("nms.iou_threshold", manifest.nms.iou_threshold);

  std::set<std::string> output_names;
  for (std::size_t i = 0; i < manifest.outputs.size(); ++i) {
    const std::string& name = manifest.outputs[i].name;
    if (name.empty()) continue;
    if (!output_names.insert(name).second) {
      report.schema_issues.push_back(
          {"outputs." + std::to_string(i) + ".name", "duplicate output name '" + name + "'"});
    }
  }

  auto require = [&](const std::string& path, bool ready, const std::string& message) {
    if (!ready) report.unresolved_critical.push_back({path, message});
  };

  require("model_name", !manifest.model_name.empty(), "model name is required");
  require("task", manifest.task != TaskType::Unknown, "task is unresolved");
  require("model_format", manifest.model_format != ModelFormat::Unspecified,
          "model format is unresolved");
  if (task_needs_num_classes(manifest.task)) {
    require("num_classes", manifest.num_classes.has_value(), "num_classes is unresolved");
  }
  require("input.name", !manifest.input.name.empty(), "input name is unresolved");
  require("input.dtype", manifest.input.dtype != DataType::Unknown, "input dtype is unresolved");
  require("input.layout", manifest.input.layout != Layout::Unknown, "input layout is unresolved");
  require("input.color", !manifest.input.color.empty(), "input color is unresolved");
  require("input.normalization", !manifest.input.normalization.empty(),
          "input normalization is unresolved");
  require("input.resize", !manifest.input.resize.empty(), "input resize is unresolved");
  if (manifest.input.resize == "shortest_side") {
    require("input.short_side", manifest.input.short_side.has_value(), "short_side is unresolved");
    require("input.crop_width", manifest.input.crop_width.has_value(), "crop width is unresolved");
    require("input.crop_height", manifest.input.crop_height.has_value(), "crop height is unresolved");
  }
  if (manifest.input.normalization == "explicit") {
    require("input.scale", manifest.input.scale.has_value(), "normalize scale is unresolved");
    require("input.mean", manifest.input.mean.size() == 3, "normalize mean is unresolved");
    require("input.std", manifest.input.std.size() == 3, "normalize std is unresolved");
  }
  shape_ready(manifest.input.shape, "input.shape", report.unresolved_critical);

  if (manifest.outputs.empty()) {
    report.unresolved_critical.push_back({"outputs", "at least one output is required"});
  }
  for (std::size_t i = 0; i < manifest.outputs.size(); ++i) {
    const OutputDesc& output = manifest.outputs[i];
    const std::string prefix = "outputs." + std::to_string(i);
    require(prefix + ".name", !output.name.empty(), "output name is unresolved");
    require(prefix + ".dtype", output.dtype != DataType::Unknown, "output dtype is unresolved");
    shape_ready(output.shape, prefix + ".shape", report.unresolved_critical);
  }

  require("decoder.protocol", !manifest.decoder.protocol.empty(), "decoder protocol is unresolved");
  if (task_needs_box_format(manifest.task)) {
    require("decoder.box_format", !manifest.decoder.box_format.empty(),
            "box format is unresolved");
  }
  if (manifest.nms.enabled.has_value() && *manifest.nms.enabled) {
    require("nms.confidence_threshold", manifest.nms.confidence_threshold.has_value(),
            "NMS confidence threshold is unresolved");
    require("nms.iou_threshold", manifest.nms.iou_threshold.has_value(),
            "NMS IoU threshold is unresolved");
  }
  if (manifest.task == TaskType::Segmentation) {
    require("segmentation.mask_channels", manifest.segmentation.mask_channels.has_value(),
            "mask channels are unresolved");
    require("segmentation.prototype_name", !manifest.segmentation.prototype_name.empty(),
            "prototype output name is unresolved");
    require("segmentation.prototype_layout", manifest.segmentation.prototype_layout != Layout::Unknown,
            "prototype layout is unresolved");
    require("segmentation.mask_coefficient_activation",
            !manifest.segmentation.mask_coefficient_activation.empty(),
            "mask coefficient activation is unresolved");
    require("segmentation.prototype_activation", !manifest.segmentation.prototype_activation.empty(),
            "prototype activation is unresolved");
    require("segmentation.has_objectness", manifest.segmentation.has_objectness.has_value(),
            "objectness is unresolved");
    // retina_masks, binary_mask_gt_zero, and max_mask_bytes stay out of the
    // load check. infer_segmentation() reads them from SegmentationRuntimeConfig.
    if (manifest.outputs.size() < 2) {
      report.unresolved_critical.push_back(
          {"outputs", "segmentation requires a prediction output and a prototype output"});
    }
  }
  if (manifest.segmentation.mask_channels.has_value() && *manifest.segmentation.mask_channels < 1) {
    report.schema_issues.push_back({"segmentation.mask_channels", "must be >= 1"});
  }
  if (manifest.segmentation.prototype_channel_axis.has_value() &&
      *manifest.segmentation.prototype_channel_axis != 1) {
    report.schema_issues.push_back({"segmentation.prototype_channel_axis", "must be 1"});
  }
  if (!manifest.segmentation.mask_coefficient_activation.empty() &&
      manifest.segmentation.mask_coefficient_activation != "none") {
    report.schema_issues.push_back(
        {"segmentation.mask_coefficient_activation", "unsupported activation"});
  }
  if (!manifest.segmentation.prototype_activation.empty() &&
      manifest.segmentation.prototype_activation != "model_output_features") {
    report.schema_issues.push_back({"segmentation.prototype_activation", "unsupported activation"});
  }
  if (!manifest.segmentation.class_names.empty() && manifest.num_classes.has_value() &&
      static_cast<int>(manifest.segmentation.class_names.size()) != *manifest.num_classes) {
    report.schema_issues.push_back(
        {"segmentation.class_names", "count does not match num_classes"});
  }
  const bool classification_protocol =
      manifest.decoder.protocol == "classification_logits_raw" ||
      manifest.decoder.protocol == "classification_probabilities_raw";
  if (classification_protocol) {
    require("classification.output_name", !manifest.classification.output_name.empty(),
            "classification output name is unresolved");
    require("classification.class_axis", manifest.classification.class_axis.has_value(),
            "classification class axis is unresolved");
    require("classification.score_type", !manifest.classification.score_type.empty(),
            "classification score type is unresolved");
  }
  if (manifest.classification.score_type == "logits" &&
      manifest.decoder.protocol != "classification_logits_raw") {
    report.schema_issues.push_back(
        {"classification.score_type",
         "logits requires decoder protocol classification_logits_raw"});
  }
  if (manifest.classification.score_type == "probabilities" &&
      manifest.decoder.protocol != "classification_probabilities_raw") {
    report.schema_issues.push_back(
        {"classification.score_type",
         "probabilities requires decoder protocol classification_probabilities_raw"});
  }
  if (manifest.classification.class_axis.has_value() && *manifest.classification.class_axis < 0) {
    report.schema_issues.push_back({"classification.class_axis", "must be >= 0"});
  }
  if (!manifest.classification.score_type.empty() &&
      manifest.classification.score_type != "logits" &&
      manifest.classification.score_type != "probabilities") {
    report.schema_issues.push_back({"classification.score_type", "unsupported score type"});
  }
  if (!manifest.classification.class_names.empty() && manifest.num_classes.has_value() &&
      static_cast<int>(manifest.classification.class_names.size()) != *manifest.num_classes) {
    report.schema_issues.push_back(
        {"classification.class_names", "count does not match num_classes"});
  }
  if (manifest.task == TaskType::Obb && manifest.decoder.protocol != "yolov8_obb_raw") {
    report.schema_issues.push_back({"decoder.protocol", "task obb requires yolov8_obb_raw"});
  }
  if (manifest.decoder.protocol == "yolov8_obb_raw" && manifest.task != TaskType::Obb) {
    report.schema_issues.push_back({"task", "yolov8_obb_raw requires task obb"});
  }
  if (manifest.decoder.protocol == "yolov8_obb_raw") {
    require("obb.output_name", !manifest.obb.output_name.empty(), "obb output name is unresolved");
    require("obb.channel_axis", manifest.obb.channel_axis.has_value(), "obb channel axis is unresolved");
    require("obb.angle_unit", manifest.obb.angle_unit == "radian", "obb angle unit is unresolved");
    require("obb.angle_range", manifest.obb.angle_range == "[-pi/4,3pi/4]", "obb angle range is unresolved");
    require("obb.score_type", manifest.obb.score_type == "probabilities", "obb score type is unresolved");
    require("decoder.box_format", manifest.decoder.box_format == "cxcywh_angle",
            "obb box format must be cxcywh_angle");
    require("decoder.box_channels", manifest.decoder.box_channels == 4, "obb box channels must be 4");
    require("decoder.scores_are_probabilities", manifest.decoder.scores_are_probabilities == true,
            "obb class scores must already be probabilities");
    if (manifest.obb.channel_axis.has_value() && manifest.decoder.channel_axis.has_value() &&
        *manifest.obb.channel_axis != *manifest.decoder.channel_axis) {
      report.schema_issues.push_back({"obb.channel_axis", "does not match decoder.channel_axis"});
    }
  }
  if (!manifest.obb.angle_unit.empty() && manifest.obb.angle_unit != "radian") {
    report.schema_issues.push_back({"obb.angle_unit", "unsupported angle unit"});
  }
  if (!manifest.obb.angle_range.empty() && manifest.obb.angle_range != "[-pi/4,3pi/4]") {
    report.schema_issues.push_back({"obb.angle_range", "unsupported angle range"});
  }
  if (!manifest.obb.score_type.empty() && manifest.obb.score_type != "probabilities") {
    report.schema_issues.push_back({"obb.score_type", "unsupported score type"});
  }
  if (manifest.task != TaskType::Obb && manifest.decoder.protocol != "yolov8_obb_raw" &&
      (!manifest.obb.output_name.empty() || manifest.obb.channel_axis.has_value() ||
       !manifest.obb.angle_unit.empty() || !manifest.obb.angle_range.empty() || !manifest.obb.score_type.empty())) {
    report.schema_issues.push_back({"obb", "obb fields require task obb"});
  }
  if (manifest.task == TaskType::Pose && manifest.decoder.protocol != "yolov8_pose_raw") {
    report.schema_issues.push_back({"decoder.protocol", "task pose requires yolov8_pose_raw"});
  }
  if (manifest.decoder.protocol == "yolov8_pose_raw" && manifest.task != TaskType::Pose) {
    report.schema_issues.push_back({"task", "yolov8_pose_raw requires task pose"});
  }
  if (manifest.decoder.protocol == "yolov8_pose_raw") {
    require("pose.output_name", !manifest.pose.output_name.empty(), "pose output name is unresolved");
    require("pose.channel_axis", manifest.pose.channel_axis.has_value(), "pose channel axis is unresolved");
    require("pose.keypoint_count", manifest.pose.keypoint_count.has_value() && *manifest.pose.keypoint_count >= 1,
            "pose keypoint count is unresolved");
    require("pose.keypoint_dimensions", manifest.pose.keypoint_dimensions == 3,
            "pose keypoint dimensions are unresolved");
    require("pose.keypoint_layout", manifest.pose.keypoint_layout == "xy_confidence",
            "pose keypoint layout is unresolved");
    require("pose.score_type", manifest.pose.score_type == "probabilities", "pose score type is unresolved");
    require("decoder.box_format", manifest.decoder.box_format == "cxcywh", "pose box format must be cxcywh");
    require("decoder.box_channels", manifest.decoder.box_channels == 4, "pose box channels must be 4");
    require("decoder.box_decode", manifest.decoder.box_decode == "decoded_cxcywh",
            "pose boxes must already be decoded cxcywh pixels");
    require("decoder.scores_are_probabilities", manifest.decoder.scores_are_probabilities == true,
            "pose class scores must already be probabilities");
    if (manifest.pose.channel_axis.has_value() && manifest.decoder.channel_axis.has_value() &&
        *manifest.pose.channel_axis != *manifest.decoder.channel_axis) {
      report.schema_issues.push_back({"pose.channel_axis", "does not match decoder.channel_axis"});
    }
  }
  if (manifest.pose.keypoint_count.has_value() && *manifest.pose.keypoint_count < 1) {
    report.schema_issues.push_back({"pose.keypoint_count", "must be >= 1"});
  }
  if (manifest.pose.keypoint_dimensions.has_value() && *manifest.pose.keypoint_dimensions != 3) {
    report.schema_issues.push_back({"pose.keypoint_dimensions", "must be 3"});
  }
  if (!manifest.pose.keypoint_layout.empty() && manifest.pose.keypoint_layout != "xy_confidence") {
    report.schema_issues.push_back({"pose.keypoint_layout", "unsupported keypoint layout"});
  }
  if (!manifest.pose.score_type.empty() && manifest.pose.score_type != "probabilities") {
    report.schema_issues.push_back({"pose.score_type", "unsupported score type"});
  }
  if (manifest.task != TaskType::Pose && manifest.decoder.protocol != "yolov8_pose_raw" &&
      (!manifest.pose.output_name.empty() || manifest.pose.channel_axis.has_value() ||
       manifest.pose.keypoint_count.has_value() || manifest.pose.keypoint_dimensions.has_value() ||
       !manifest.pose.keypoint_layout.empty() || !manifest.pose.score_type.empty())) {
    report.schema_issues.push_back({"pose", "pose fields require task pose"});
  }
  if (!manifest.pose.output_name.empty()) {
    const OutputDesc* named = nullptr;
    for (const OutputDesc& output : manifest.outputs) {
      if (output.name == manifest.pose.output_name) named = &output;
    }
    if (named == nullptr) {
      report.schema_issues.push_back({"pose.output_name", "does not match an output"});
    } else if (manifest.pose.channel_axis.has_value() && manifest.pose.keypoint_count.has_value() &&
               manifest.pose.keypoint_dimensions.has_value() && !named->shape.empty() &&
               is_concrete_shape(named->shape) && manifest.num_classes.has_value()) {
      const int axis = *manifest.pose.channel_axis;
      const std::int64_t expect = static_cast<std::int64_t>(4 + *manifest.num_classes +
                                                            *manifest.pose.keypoint_count *
                                                                *manifest.pose.keypoint_dimensions);
      if (named->shape.size() != 3 || axis < 0 || axis >= static_cast<int>(named->shape.size())) {
        report.schema_issues.push_back({"pose.channel_axis", "is outside rank 3"});
      } else if (named->shape[static_cast<std::size_t>(axis)] != expect) {
        report.schema_issues.push_back(
            {"pose.channel_axis", "length is not 4 + num_classes + keypoint_count * keypoint_dimensions"});
      }
    }
  }
  if (!manifest.obb.output_name.empty()) {
    const OutputDesc* named = nullptr;
    for (const OutputDesc& output : manifest.outputs) {
      if (output.name == manifest.obb.output_name) named = &output;
    }
    if (named == nullptr) {
      report.schema_issues.push_back({"obb.output_name", "does not match an output"});
    } else if (manifest.obb.channel_axis.has_value() && !named->shape.empty() && is_concrete_shape(named->shape) &&
               manifest.num_classes.has_value()) {
      const int axis = *manifest.obb.channel_axis;
      if (named->shape.size() != 3 || axis < 0 || axis >= static_cast<int>(named->shape.size())) {
        report.schema_issues.push_back({"obb.channel_axis", "is outside rank 3"});
      } else if (named->shape[static_cast<std::size_t>(axis)] !=
                 static_cast<std::int64_t>(4 + *manifest.num_classes + 1)) {
        report.schema_issues.push_back(
            {"obb.channel_axis", "length is not 4 + num_classes + 1"});
      }
    }
  }
  if (!manifest.classification.output_name.empty()) {
    const OutputDesc* named = nullptr;
    for (const OutputDesc& output : manifest.outputs) {
      if (output.name == manifest.classification.output_name) named = &output;
    }
    if (named == nullptr) {
      report.schema_issues.push_back(
          {"classification.output_name", "does not match an output"});
    } else if (manifest.classification.class_axis.has_value() && !named->shape.empty() &&
               is_concrete_shape(named->shape)) {
      const int axis = *manifest.classification.class_axis;
      if (named->shape.size() != 2) {
        report.schema_issues.push_back(
            {"classification.output_name", "classification output must be rank 2"});
      } else if (axis > 1) {
        report.schema_issues.push_back({"classification.class_axis", "is outside rank 2"});
      } else if (manifest.num_classes.has_value() &&
                 named->shape[static_cast<std::size_t>(axis)] != *manifest.num_classes) {
        report.schema_issues.push_back(
            {"classification.class_axis", "length does not match num_classes"});
      } else if (named->shape[static_cast<std::size_t>(1 - axis)] != 1) {
        report.schema_issues.push_back(
            {"classification.class_axis", "the other dimension must be batch size 1"});
      }
    }
  }
  return report;
}

}  // namespace vie
