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
  return report;
}

}  // namespace vie
