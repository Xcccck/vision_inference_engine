#include "vie/model/manifest.hpp"

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <utility>

namespace vie {
namespace {

using json = nlohmann::json;

[[noreturn]] void schema_error(const std::string& message) {
  throw VieError(ErrorCode::SchemaError, message);
}

bool parse_output_field(std::string_view path, std::size_t& index, std::string& field) {
  constexpr std::string_view kPrefix = "outputs.";
  if (path.size() < kPrefix.size() || path.substr(0, kPrefix.size()) != kPrefix) return false;
  const std::string_view rest = path.substr(kPrefix.size());
  const std::size_t dot = rest.find('.');
  if (dot == std::string_view::npos || dot == 0 || dot + 1 >= rest.size()) return false;
  const std::string_view index_text = rest.substr(0, dot);
  for (const char ch : index_text) {
    if (ch < '0' || ch > '9') return false;
  }
  try {
    index = static_cast<std::size_t>(std::stoul(std::string(index_text)));
  } catch (const std::exception&) {
    return false;
  }
  field = std::string(rest.substr(dot + 1));
  return true;
}

bool is_output_field(std::string_view field) {
  return field == "name" || field == "shape" || field == "dtype" || field == "layout" ||
         field == "semantic";
}

bool is_coefficient_activation(std::string_view text) { return text == "none"; }

bool is_prototype_activation(std::string_view text) { return text == "model_output_features"; }

bool is_classification_score_type(std::string_view text) {
  return text == "logits" || text == "probabilities";
}

bool classification_active(const ModelManifest& manifest) {
  const ClassificationDesc& classification = manifest.classification;
  return manifest.task == TaskType::Classification || !classification.output_name.empty() ||
         classification.class_axis.has_value() || !classification.score_type.empty() ||
         !classification.class_names.empty();
}

bool is_obb_angle_unit(std::string_view text) { return text == "radian"; }

bool is_obb_angle_range(std::string_view text) { return text == "[-pi/4,3pi/4]"; }

bool is_obb_score_type(std::string_view text) { return text == "probabilities"; }

bool is_pose_layout(std::string_view text) { return text == "xy_confidence"; }

bool is_pose_score_type(std::string_view text) { return text == "probabilities"; }

bool pose_active(const ModelManifest& manifest) {
  const PoseDesc& pose = manifest.pose;
  return manifest.task == TaskType::Pose || !pose.output_name.empty() || pose.channel_axis.has_value() ||
         pose.keypoint_count.has_value() || pose.keypoint_dimensions.has_value() || !pose.keypoint_layout.empty() ||
         !pose.score_type.empty();
}

bool obb_active(const ModelManifest& manifest) {
  const ObbDesc& obb = manifest.obb;
  return manifest.task == TaskType::Obb || !obb.output_name.empty() || obb.channel_axis.has_value() ||
         !obb.angle_unit.empty() || !obb.angle_range.empty() || !obb.score_type.empty();
}

bool segmentation_active(const ModelManifest& manifest) {
  const SegmentationDesc& segmentation = manifest.segmentation;
  return manifest.task == TaskType::Segmentation || segmentation.mask_channels.has_value() ||
         !segmentation.prototype_name.empty() || segmentation.prototype_layout != Layout::Unknown ||
         segmentation.prototype_channel_axis.has_value() || segmentation.prototype_height.has_value() ||
         segmentation.prototype_width.has_value() || !segmentation.mask_coefficient_activation.empty() ||
         !segmentation.prototype_activation.empty() || segmentation.has_objectness.has_value() ||
         !segmentation.class_names.empty() || segmentation.retina_masks.has_value() ||
         segmentation.binary_mask_gt_zero.has_value() || segmentation.max_mask_bytes.has_value();
}

const OutputDesc* output_at(const ModelManifest& manifest, std::size_t index) {
  if (index >= manifest.outputs.size()) return nullptr;
  return &manifest.outputs[index];
}

bool output_field_set(const OutputDesc& output, std::string_view field) {
  if (field == "name" || field == "semantic") {
    const std::string& value = field == "name" ? output.name : output.semantic;
    return !value.empty();
  }
  if (field == "shape") return !output.shape.empty();
  if (field == "dtype") return output.dtype != DataType::Unknown;
  if (field == "layout") return output.layout != Layout::Unknown;
  return false;
}

json string_or_null(const std::string& value) {
  if (value.empty()) return nullptr;
  return value;
}

json shape_or_null(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) return nullptr;
  json array = json::array();
  for (const std::int64_t dim : shape) array.push_back(dim);
  return array;
}

json enum_or_null(const char* text, bool unset) {
  if (unset) return nullptr;
  return text;
}

std::string require_text(const json& value, const std::string& path) {
  if (!value.is_string()) schema_error(path + " must be a string");
  return value.get<std::string>();
}

std::string read_free_string(const json& object, const char* key, bool reject_empty) {
  if (!object.contains(key) || object.at(key).is_null()) return "";
  const std::string path = key;
  const std::string value = require_text(object.at(key), path);
  if (value.empty()) {
    if (reject_empty) schema_error(std::string(key) + " is empty");
    return "";
  }
  return value;
}

std::string read_vocab(const json& object, const char* key, bool (*allowed)(std::string_view),
                       const char* allowed_text) {
  if (!object.contains(key) || object.at(key).is_null()) return "";
  const std::string value = require_text(object.at(key), key);
  if (value.empty()) schema_error(std::string(key) + " is empty");
  if (!allowed(value)) {
    schema_error(std::string(key) + " has unsupported value '" + value +
                 "'. Expected one of: " + allowed_text);
  }
  return value;
}

std::vector<std::int64_t> read_shape(const json& value, const std::string& path) {
  if (!value.is_array() || value.empty()) schema_error(path + " must be a non-empty array");
  std::vector<std::int64_t> shape;
  shape.reserve(value.size());
  for (const auto& dim : value) {
    if (!dim.is_number_integer()) schema_error(path + " dimensions must be integers");
    const std::int64_t number = dim.get<std::int64_t>();
    if (number == 0 || number < -1) schema_error(path + " has an invalid dimension");
    shape.push_back(number);
  }
  return shape;
}

template <typename Enum>
Enum read_enum(const json& object, const char* key, Enum (*parse)(std::string_view), Enum unset) {
  if (!object.contains(key) || object.at(key).is_null()) return unset;
  return parse(require_text(object.at(key), key));
}

void read_input(const json& value, InputDesc& input) {
  if (!value.is_object()) schema_error("input must be an object");
  input.name = read_free_string(value, "name", true);
  if (value.contains("shape") && !value.at("shape").is_null()) {
    input.shape = read_shape(value.at("shape"), "input.shape");
  }
  input.dtype = read_enum(value, "dtype", data_type_from_string, DataType::Unknown);
  input.layout = read_enum(value, "layout", layout_from_string, Layout::Unknown);
  input.color = read_vocab(value, "color", is_known_color, "rgb, bgr");
  input.normalization = read_vocab(value, "normalization", is_known_normalization,
                                   "scale_0_1, scale_1_255, imagenet, none, explicit");
  input.resize = read_vocab(value, "resize", is_known_resize, "letterbox, stretch, crop, shortest_side");
  input.interpolation =
      read_vocab(value, "interpolation",
                 [](std::string_view text) { return text == "nearest" || text == "bilinear"; },
                 "nearest, bilinear");
  if (value.contains("padding_value") && !value.at("padding_value").is_null()) {
    if (!value.at("padding_value").is_number()) schema_error("padding_value must be a number");
    const float number = value.at("padding_value").get<float>();
    if (!std::isfinite(number)) schema_error("padding_value must be finite");
    input.padding_value = number;
  }
  auto read_positive_int = [&](const char* key, std::optional<int>& slot) {
    if (!value.contains(key) || value.at(key).is_null()) return;
    if (!value.at(key).is_number_integer()) schema_error(std::string(key) + " must be an integer");
    const int number = value.at(key).get<int>();
    if (number < 1) schema_error(std::string(key) + " must be positive");
    slot = number;
  };
  read_positive_int("short_side", input.short_side);
  read_positive_int("crop_width", input.crop_width);
  read_positive_int("crop_height", input.crop_height);
  if (value.contains("scale") && !value.at("scale").is_null()) {
    if (!value.at("scale").is_number()) schema_error("scale must be a number");
    const float number = value.at("scale").get<float>();
    if (!std::isfinite(number)) schema_error("scale must be finite");
    input.scale = number;
  }
  auto read_channels = [&](const char* key, std::vector<float>& slot) {
    if (!value.contains(key) || value.at(key).is_null()) return;
    if (!value.at(key).is_array()) schema_error(std::string(key) + " must be an array");
    for (const auto& item : value.at(key)) {
      if (!item.is_number()) schema_error(std::string(key) + " entries must be numbers");
      const float number = item.get<float>();
      if (!std::isfinite(number)) schema_error(std::string(key) + " entries must be finite");
      slot.push_back(number);
    }
  };
  read_channels("mean", input.mean);
  read_channels("std", input.std);
}

OutputDesc read_output(const json& value, std::size_t index) {
  const std::string path = "outputs." + std::to_string(index);
  if (!value.is_object()) schema_error(path + " must be an object");
  OutputDesc output;
  output.name = read_free_string(value, "name", true);
  if (value.contains("shape") && !value.at("shape").is_null()) {
    output.shape = read_shape(value.at("shape"), path + ".shape");
  }
  output.dtype = read_enum(value, "dtype", data_type_from_string, DataType::Unknown);
  output.layout = read_enum(value, "layout", layout_from_string, Layout::Unknown);
  output.semantic = read_free_string(value, "semantic", false);
  return output;
}

void read_decoder(const json& value, DecoderDesc& decoder) {
  if (!value.is_object()) schema_error("decoder must be an object");
  decoder.protocol = read_free_string(value, "protocol", true);
  decoder.box_format =
      read_vocab(value, "box_format", is_known_box_format,
                 "xyxy, cxcywh, xywh, xyxy_normalized, cxcywh_normalized");
  decoder.box_decode = read_vocab(
      value, "box_decode",
      [](std::string_view text) {
        return text == "decoded_cxcywh" || text == "decoded_normalized_cxcywh" || text == "dfl";
      },
      "decoded_cxcywh, decoded_normalized_cxcywh, dfl");
  decoder.class_selection = read_vocab(
      value, "class_selection",
      [](std::string_view text) { return text == "best_class" || text == "all_above_threshold"; },
      "best_class, all_above_threshold");
  if (value.contains("scores_are_probabilities") && !value.at("scores_are_probabilities").is_null()) {
    if (!value.at("scores_are_probabilities").is_boolean()) {
      schema_error("scores_are_probabilities must be a boolean");
    }
    decoder.scores_are_probabilities = value.at("scores_are_probabilities").get<bool>();
  }
  auto read_positive = [&](const char* key, std::optional<int>& slot) {
    if (!value.contains(key) || value.at(key).is_null()) return;
    if (!value.at(key).is_number_integer()) schema_error(std::string(key) + " must be an integer");
    const int number = value.at(key).get<int>();
    if (number < 1) schema_error(std::string(key) + " must be >= 1");
    slot = number;
  };
  read_positive("box_channels", decoder.box_channels);
  read_positive("reg_max", decoder.reg_max);
  read_positive("num_queries", decoder.num_queries);
  if (value.contains("channel_axis") && !value.at("channel_axis").is_null()) {
    if (!value.at("channel_axis").is_number_integer()) schema_error("channel_axis must be an integer");
    const int axis = value.at("channel_axis").get<int>();
    if (axis != 1 && axis != 2) schema_error("channel_axis must be 1 or 2");
    decoder.channel_axis = axis;
  }
}

void read_nms(const json& value, NmsDesc& nms) {
  if (!value.is_object()) schema_error("nms must be an object");
  if (value.contains("enabled") && !value.at("enabled").is_null()) {
    if (!value.at("enabled").is_boolean()) schema_error("nms.enabled must be a boolean");
    nms.enabled = value.at("enabled").get<bool>();
  }
  auto read_unit = [&](const char* key, std::optional<float>& slot) {
    if (!value.contains(key) || value.at(key).is_null()) return;
    if (!value.at(key).is_number()) schema_error(std::string(key) + " must be a number");
    const float number = value.at(key).get<float>();
    if (!(number >= 0.f && number <= 1.f)) {
      schema_error(std::string(key) + " must be in the inclusive range [0, 1]");
    }
    slot = number;
  };
  read_unit("confidence_threshold", nms.confidence_threshold);
  read_unit("iou_threshold", nms.iou_threshold);
  auto read_positive = [&](const char* key, std::optional<int>& slot, int upper) {
    if (!value.contains(key) || value.at(key).is_null()) return;
    if (!value.at(key).is_number_integer()) schema_error(std::string(key) + " must be an integer");
    const int number = value.at(key).get<int>();
    if (number < 1 || (upper > 0 && number > upper)) {
      schema_error(std::string(key) + " is outside its allowed range");
    }
    slot = number;
  };
  read_positive("max_detections", nms.max_detections, 0);
  read_positive("pre_nms_top_k", nms.pre_nms_top_k, 4096);
  if (value.contains("class_aware") && !value.at("class_aware").is_null()) {
    if (!value.at("class_aware").is_boolean()) schema_error("class_aware must be a boolean");
    nms.class_aware = value.at("class_aware").get<bool>();
  }
}

void read_provenance(const json& value, ModelManifest& manifest) {
  if (!value.is_object()) schema_error("provenance must be an object");
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (!it.value().is_object()) schema_error("provenance." + it.key() + " must be an object");
    if (!it.value().contains("certainty") || !it.value().at("certainty").is_string()) {
      schema_error("provenance." + it.key() + ".certainty must be a string");
    }
    FieldProvenance record;
    record.certainty = certainty_from_string(it.value().at("certainty").get<std::string>());
    if (it.value().contains("evidence")) {
      if (!it.value().at("evidence").is_array()) {
        schema_error("provenance." + it.key() + ".evidence must be an array");
      }
      for (const auto& item : it.value().at("evidence")) {
        if (!item.is_string()) schema_error("provenance evidence entries must be strings");
        record.evidence.push_back(item.get<std::string>());
      }
    }
    manifest.provenance[it.key()] = std::move(record);
  }
}

json decoder_json(const DecoderDesc& decoder) {
  json document = json{{"protocol", string_or_null(decoder.protocol)},
                       {"box_format", string_or_null(decoder.box_format)},
                       {"box_decode", string_or_null(decoder.box_decode)},
                       {"class_selection", string_or_null(decoder.class_selection)},
                       {"scores_are_probabilities", decoder.scores_are_probabilities
                                                        ? json(*decoder.scores_are_probabilities)
                                                        : json(nullptr)},
                       {"box_channels", decoder.box_channels ? json(*decoder.box_channels) : json(nullptr)},
                       {"channel_axis", decoder.channel_axis ? json(*decoder.channel_axis) : json(nullptr)},
                       {"reg_max", decoder.reg_max ? json(*decoder.reg_max) : json(nullptr)}};
  if (decoder.num_queries.has_value()) document["num_queries"] = *decoder.num_queries;
  return document;
}

json provenance_json(const FieldProvenance& record) {
  json evidence = json::array();
  for (const auto& item : record.evidence) evidence.push_back(item);
  return json{{"certainty", to_string(record.certainty)}, {"evidence", std::move(evidence)}};
}

json manifest_json(const ModelManifest& manifest) {
  json outputs = json::array();
  for (const OutputDesc& output : manifest.outputs) {
    outputs.push_back(json{{"name", string_or_null(output.name)},
                           {"shape", shape_or_null(output.shape)},
                           {"dtype", enum_or_null(to_string(output.dtype),
                                                  output.dtype == DataType::Unknown)},
                           {"layout", enum_or_null(to_string(output.layout),
                                                   output.layout == Layout::Unknown)},
                           {"semantic", string_or_null(output.semantic)}});
  }
  json provenance = json::object();
  for (const std::string& path : manifest_field_paths(manifest)) {
    provenance[path] = provenance_json(manifest.provenance.at(path));
  }
  json document = json{
      {"schema_version", manifest.schema_version},
      {"model_name", string_or_null(manifest.model_name)},
      {"model_family", string_or_null(manifest.model_family)},
      {"model_version", string_or_null(manifest.model_version)},
      {"task", enum_or_null(to_string(manifest.task), manifest.task == TaskType::Unknown)},
      {"model_format", enum_or_null(to_string(manifest.model_format),
                                    manifest.model_format == ModelFormat::Unspecified)},
      {"model_path", string_or_null(manifest.model_path)},
      {"num_classes", manifest.num_classes ? json(*manifest.num_classes) : json(nullptr)},
      {"input",
       json{{"name", string_or_null(manifest.input.name)},
            {"shape", shape_or_null(manifest.input.shape)},
            {"dtype", enum_or_null(to_string(manifest.input.dtype),
                                   manifest.input.dtype == DataType::Unknown)},
            {"layout", enum_or_null(to_string(manifest.input.layout),
                                    manifest.input.layout == Layout::Unknown)},
            {"color", string_or_null(manifest.input.color)},
            {"normalization", string_or_null(manifest.input.normalization)},
            {"resize", string_or_null(manifest.input.resize)},
            {"interpolation", string_or_null(manifest.input.interpolation)},
            {"padding_value", manifest.input.padding_value ? json(*manifest.input.padding_value)
                                                          : json(nullptr)}}},
      {"outputs", std::move(outputs)},
      {"decoder", decoder_json(manifest.decoder)},
      {"nms", json{{"enabled", manifest.nms.enabled ? json(*manifest.nms.enabled) : json(nullptr)},
                   {"confidence_threshold", manifest.nms.confidence_threshold
                                                ? json(*manifest.nms.confidence_threshold)
                                                : json(nullptr)},
                   {"iou_threshold", manifest.nms.iou_threshold ? json(*manifest.nms.iou_threshold)
                                                                : json(nullptr)},
                   {"max_detections", manifest.nms.max_detections ? json(*manifest.nms.max_detections)
                                                                  : json(nullptr)},
                   {"pre_nms_top_k", manifest.nms.pre_nms_top_k ? json(*manifest.nms.pre_nms_top_k) : json(nullptr)},
                   {"class_aware", manifest.nms.class_aware ? json(*manifest.nms.class_aware) : json(nullptr)}}},
      {"provenance", std::move(provenance)}};
  if (manifest.input.short_side.has_value()) document["input"]["short_side"] = *manifest.input.short_side;
  if (manifest.input.crop_width.has_value()) document["input"]["crop_width"] = *manifest.input.crop_width;
  if (manifest.input.crop_height.has_value()) document["input"]["crop_height"] = *manifest.input.crop_height;
  if (manifest.input.scale.has_value()) document["input"]["scale"] = *manifest.input.scale;
  if (!manifest.input.mean.empty()) document["input"]["mean"] = manifest.input.mean;
  if (!manifest.input.std.empty()) document["input"]["std"] = manifest.input.std;
  if (segmentation_active(manifest)) {
    json names = json::array();
    for (const std::string& name : manifest.segmentation.class_names) names.push_back(name);
    document["segmentation"] = json{
        {"mask_channels", manifest.segmentation.mask_channels ? json(*manifest.segmentation.mask_channels)
                                                              : json(nullptr)},
        {"prototype_name", string_or_null(manifest.segmentation.prototype_name)},
        {"prototype_layout", enum_or_null(to_string(manifest.segmentation.prototype_layout),
                                          manifest.segmentation.prototype_layout == Layout::Unknown)},
        {"prototype_channel_axis", manifest.segmentation.prototype_channel_axis
                                       ? json(*manifest.segmentation.prototype_channel_axis)
                                       : json(nullptr)},
        {"prototype_height", manifest.segmentation.prototype_height
                                 ? json(*manifest.segmentation.prototype_height)
                                 : json(nullptr)},
        {"prototype_width", manifest.segmentation.prototype_width
                                ? json(*manifest.segmentation.prototype_width)
                                : json(nullptr)},
        {"mask_coefficient_activation", string_or_null(manifest.segmentation.mask_coefficient_activation)},
        {"prototype_activation", string_or_null(manifest.segmentation.prototype_activation)},
        {"has_objectness", manifest.segmentation.has_objectness ? json(*manifest.segmentation.has_objectness)
                                                                : json(nullptr)},
        {"class_names", manifest.segmentation.class_names.empty() ? json(nullptr) : json(std::move(names))},
        {"retina_masks", manifest.segmentation.retina_masks ? json(*manifest.segmentation.retina_masks)
                                                            : json(nullptr)},
        {"binary_mask_gt_zero", manifest.segmentation.binary_mask_gt_zero
                                    ? json(*manifest.segmentation.binary_mask_gt_zero)
                                    : json(nullptr)},
        {"max_mask_bytes", manifest.segmentation.max_mask_bytes
                               ? json(*manifest.segmentation.max_mask_bytes)
                               : json(nullptr)}};
  }
  if (classification_active(manifest)) {
    json names = json::array();
    for (const std::string& name : manifest.classification.class_names) names.push_back(name);
    document["classification"] = json{
        {"output_name", string_or_null(manifest.classification.output_name)},
        {"class_axis", manifest.classification.class_axis ? json(*manifest.classification.class_axis)
                                                          : json(nullptr)},
        {"score_type", string_or_null(manifest.classification.score_type)},
        {"class_names",
         manifest.classification.class_names.empty() ? json(nullptr) : json(std::move(names))}};
  }
  if (pose_active(manifest)) {
    document["pose"] = json{
        {"output_name", string_or_null(manifest.pose.output_name)},
        {"channel_axis", manifest.pose.channel_axis ? json(*manifest.pose.channel_axis) : json(nullptr)},
        {"keypoint_count", manifest.pose.keypoint_count ? json(*manifest.pose.keypoint_count) : json(nullptr)},
        {"keypoint_dimensions",
         manifest.pose.keypoint_dimensions ? json(*manifest.pose.keypoint_dimensions) : json(nullptr)},
        {"keypoint_layout", string_or_null(manifest.pose.keypoint_layout)},
        {"score_type", string_or_null(manifest.pose.score_type)}};
  }
  if (obb_active(manifest)) {
    document["obb"] = json{
        {"output_name", string_or_null(manifest.obb.output_name)},
        {"channel_axis", manifest.obb.channel_axis ? json(*manifest.obb.channel_axis) : json(nullptr)},
        {"angle_unit", string_or_null(manifest.obb.angle_unit)},
        {"angle_range", string_or_null(manifest.obb.angle_range)},
        {"score_type", string_or_null(manifest.obb.score_type)}};
  }
  return document;
}

void read_segmentation(const json& value, SegmentationDesc& segmentation) {
  if (!value.is_object()) schema_error("segmentation must be an object");
  auto read_positive = [&](const char* key, std::optional<int>& slot) {
    if (!value.contains(key) || value.at(key).is_null()) return;
    if (!value.at(key).is_number_integer()) schema_error(std::string(key) + " must be an integer");
    const int number = value.at(key).get<int>();
    if (number < 1) schema_error(std::string(key) + " must be >= 1");
    slot = number;
  };
  read_positive("mask_channels", segmentation.mask_channels);
  segmentation.prototype_name = read_free_string(value, "prototype_name", false);
  segmentation.prototype_layout =
      read_enum(value, "prototype_layout", layout_from_string, Layout::Unknown);
  read_positive("prototype_channel_axis", segmentation.prototype_channel_axis);
  if (segmentation.prototype_channel_axis.has_value() && *segmentation.prototype_channel_axis != 1) {
    schema_error("prototype_channel_axis must be 1");
  }
  read_positive("prototype_height", segmentation.prototype_height);
  read_positive("prototype_width", segmentation.prototype_width);
  segmentation.mask_coefficient_activation =
      read_vocab(value, "mask_coefficient_activation", is_coefficient_activation, "none");
  segmentation.prototype_activation =
      read_vocab(value, "prototype_activation", is_prototype_activation, "model_output_features");
  auto read_bool = [&](const char* key, std::optional<bool>& slot) {
    if (!value.contains(key) || value.at(key).is_null()) return;
    if (!value.at(key).is_boolean()) schema_error(std::string(key) + " must be a boolean");
    slot = value.at(key).get<bool>();
  };
  read_bool("has_objectness", segmentation.has_objectness);
  read_bool("retina_masks", segmentation.retina_masks);
  read_bool("binary_mask_gt_zero", segmentation.binary_mask_gt_zero);
  if (value.contains("class_names") && !value.at("class_names").is_null()) {
    if (!value.at("class_names").is_array()) schema_error("class_names must be an array");
    for (const auto& item : value.at("class_names")) {
      if (!item.is_string() || item.get<std::string>().empty()) {
        schema_error("class_names entries must be non-empty strings");
      }
      segmentation.class_names.push_back(item.get<std::string>());
    }
  }
  if (value.contains("max_mask_bytes") && !value.at("max_mask_bytes").is_null()) {
    if (!value.at("max_mask_bytes").is_number_unsigned() && !value.at("max_mask_bytes").is_number_integer()) {
      schema_error("max_mask_bytes must be an integer");
    }
    const auto bytes = value.at("max_mask_bytes").get<std::uint64_t>();
    if (bytes < 1) schema_error("max_mask_bytes must be >= 1");
    segmentation.max_mask_bytes = static_cast<std::size_t>(bytes);
  }
}

void read_pose(const json& value, PoseDesc& pose) {
  if (!value.is_object()) schema_error("pose must be an object");
  pose.output_name = read_free_string(value, "output_name", false);
  if (value.contains("channel_axis") && !value.at("channel_axis").is_null()) {
    if (!value.at("channel_axis").is_number_integer()) schema_error("pose.channel_axis must be an integer");
    const int axis = value.at("channel_axis").get<int>();
    if (axis != 1 && axis != 2) schema_error("pose.channel_axis must be 1 or 2");
    pose.channel_axis = axis;
  }
  if (value.contains("keypoint_count") && !value.at("keypoint_count").is_null()) {
    if (!value.at("keypoint_count").is_number_integer()) schema_error("pose.keypoint_count must be an integer");
    const int count = value.at("keypoint_count").get<int>();
    if (count < 1) schema_error("pose.keypoint_count must be >= 1");
    pose.keypoint_count = count;
  }
  if (value.contains("keypoint_dimensions") && !value.at("keypoint_dimensions").is_null()) {
    if (!value.at("keypoint_dimensions").is_number_integer()) {
      schema_error("pose.keypoint_dimensions must be an integer");
    }
    const int dimensions = value.at("keypoint_dimensions").get<int>();
    if (dimensions != 3) schema_error("pose.keypoint_dimensions must be 3");
    pose.keypoint_dimensions = dimensions;
  }
  pose.keypoint_layout = read_vocab(value, "keypoint_layout", is_pose_layout, "xy_confidence");
  pose.score_type = read_vocab(value, "score_type", is_pose_score_type, "probabilities");
}

void read_obb(const json& value, ObbDesc& obb) {
  if (!value.is_object()) schema_error("obb must be an object");
  obb.output_name = read_free_string(value, "output_name", false);
  if (value.contains("channel_axis") && !value.at("channel_axis").is_null()) {
    if (!value.at("channel_axis").is_number_integer()) schema_error("obb.channel_axis must be an integer");
    const int axis = value.at("channel_axis").get<int>();
    if (axis != 1 && axis != 2) schema_error("obb.channel_axis must be 1 or 2");
    obb.channel_axis = axis;
  }
  obb.angle_unit = read_vocab(value, "angle_unit", is_obb_angle_unit, "radian");
  obb.angle_range = read_vocab(value, "angle_range", is_obb_angle_range, "[-pi/4,3pi/4]");
  obb.score_type = read_vocab(value, "score_type", is_obb_score_type, "probabilities");
}

void read_classification(const json& value, ClassificationDesc& classification) {
  if (!value.is_object()) schema_error("classification must be an object");
  classification.output_name = read_free_string(value, "output_name", false);
  if (value.contains("class_axis") && !value.at("class_axis").is_null()) {
    if (!value.at("class_axis").is_number_integer()) schema_error("class_axis must be an integer");
    const int axis = value.at("class_axis").get<int>();
    if (axis < 0) schema_error("class_axis must be >= 0");
    classification.class_axis = axis;
  }
  classification.score_type =
      read_vocab(value, "score_type", is_classification_score_type, "logits, probabilities");
  if (value.contains("class_names") && !value.at("class_names").is_null()) {
    if (!value.at("class_names").is_array()) schema_error("class_names must be an array");
    for (const auto& item : value.at("class_names")) {
      if (!item.is_string() || item.get<std::string>().empty()) {
        schema_error("class_names entries must be non-empty strings");
      }
      classification.class_names.push_back(item.get<std::string>());
    }
  }
}

}  // namespace

std::vector<std::string> manifest_field_paths(const ModelManifest& manifest) {
  std::vector<std::string> paths = {
      "model_name",
      "model_family",
      "model_version",
      "task",
      "model_format",
      "model_path",
      "num_classes",
      "input.name",
      "input.shape",
      "input.dtype",
      "input.layout",
      "input.color",
      "input.normalization",
      "input.resize",
      "input.interpolation",
      "input.padding_value",
      "decoder.protocol",
      "decoder.box_format",
      "decoder.box_decode",
      "decoder.class_selection",
      "decoder.scores_are_probabilities",
      "decoder.box_channels",
      "decoder.channel_axis",
      "decoder.reg_max",
      "nms.enabled",
      "nms.confidence_threshold",
      "nms.iou_threshold",
      "nms.max_detections",
      "nms.pre_nms_top_k",
      "nms.class_aware",
  };
  if (manifest.input.short_side.has_value()) paths.push_back("input.short_side");
  if (manifest.input.crop_width.has_value()) paths.push_back("input.crop_width");
  if (manifest.input.crop_height.has_value()) paths.push_back("input.crop_height");
  if (manifest.input.scale.has_value()) paths.push_back("input.scale");
  if (!manifest.input.mean.empty()) paths.push_back("input.mean");
  if (!manifest.input.std.empty()) paths.push_back("input.std");
  if (manifest.decoder.num_queries.has_value()) paths.push_back("decoder.num_queries");
  if (segmentation_active(manifest)) {
    paths.insert(paths.end(),
                 {"segmentation.mask_channels", "segmentation.prototype_name",
                  "segmentation.prototype_layout", "segmentation.prototype_channel_axis",
                  "segmentation.prototype_height", "segmentation.prototype_width",
                  "segmentation.mask_coefficient_activation", "segmentation.prototype_activation",
                  "segmentation.has_objectness", "segmentation.class_names",
                  "segmentation.retina_masks", "segmentation.binary_mask_gt_zero",
                  "segmentation.max_mask_bytes"});
  }
  if (classification_active(manifest)) {
    paths.insert(paths.end(), {"classification.output_name", "classification.class_axis",
                               "classification.score_type", "classification.class_names"});
  }
  if (obb_active(manifest)) {
    paths.insert(paths.end(), {"obb.output_name", "obb.channel_axis", "obb.angle_unit", "obb.angle_range",
                               "obb.score_type"});
  }
  if (pose_active(manifest)) {
    paths.insert(paths.end(), {"pose.output_name", "pose.channel_axis", "pose.keypoint_count",
                               "pose.keypoint_dimensions", "pose.keypoint_layout", "pose.score_type"});
  }
  for (std::size_t i = 0; i < manifest.outputs.size(); ++i) {
    const std::string prefix = "outputs." + std::to_string(i) + ".";
    paths.push_back(prefix + "name");
    paths.push_back(prefix + "shape");
    paths.push_back(prefix + "dtype");
    paths.push_back(prefix + "layout");
    paths.push_back(prefix + "semantic");
  }
  return paths;
}

bool is_manifest_field_path(const ModelManifest& manifest, std::string_view path) {
  for (const std::string& known : manifest_field_paths(manifest)) {
    if (known == path) return true;
  }
  return false;
}

bool field_is_set(const ModelManifest& manifest, std::string_view path) {
  if (path == "model_name") return !manifest.model_name.empty();
  if (path == "model_family") return !manifest.model_family.empty();
  if (path == "model_version") return !manifest.model_version.empty();
  if (path == "task") return manifest.task != TaskType::Unknown;
  if (path == "model_format") return manifest.model_format != ModelFormat::Unspecified;
  if (path == "model_path") return !manifest.model_path.empty();
  if (path == "num_classes") return manifest.num_classes.has_value();
  if (path == "input.name") return !manifest.input.name.empty();
  if (path == "input.shape") return !manifest.input.shape.empty();
  if (path == "input.dtype") return manifest.input.dtype != DataType::Unknown;
  if (path == "input.layout") return manifest.input.layout != Layout::Unknown;
  if (path == "input.color") return !manifest.input.color.empty();
  if (path == "input.normalization") return !manifest.input.normalization.empty();
  if (path == "input.resize") return !manifest.input.resize.empty();
  if (path == "input.interpolation") return !manifest.input.interpolation.empty();
  if (path == "input.padding_value") return manifest.input.padding_value.has_value();
  if (path == "input.short_side") return manifest.input.short_side.has_value();
  if (path == "input.crop_width") return manifest.input.crop_width.has_value();
  if (path == "input.crop_height") return manifest.input.crop_height.has_value();
  if (path == "input.scale") return manifest.input.scale.has_value();
  if (path == "input.mean") return !manifest.input.mean.empty();
  if (path == "input.std") return !manifest.input.std.empty();
  if (path == "decoder.protocol") return !manifest.decoder.protocol.empty();
  if (path == "decoder.box_format") return !manifest.decoder.box_format.empty();
  if (path == "decoder.box_decode") return !manifest.decoder.box_decode.empty();
  if (path == "decoder.class_selection") return !manifest.decoder.class_selection.empty();
  if (path == "decoder.scores_are_probabilities") return manifest.decoder.scores_are_probabilities.has_value();
  if (path == "decoder.box_channels") return manifest.decoder.box_channels.has_value();
  if (path == "decoder.channel_axis") return manifest.decoder.channel_axis.has_value();
  if (path == "decoder.reg_max") return manifest.decoder.reg_max.has_value();
  if (path == "decoder.num_queries") return manifest.decoder.num_queries.has_value();
  if (path == "nms.enabled") return manifest.nms.enabled.has_value();
  if (path == "nms.confidence_threshold") return manifest.nms.confidence_threshold.has_value();
  if (path == "nms.iou_threshold") return manifest.nms.iou_threshold.has_value();
  if (path == "nms.max_detections") return manifest.nms.max_detections.has_value();
  if (path == "nms.pre_nms_top_k") return manifest.nms.pre_nms_top_k.has_value();
  if (path == "nms.class_aware") return manifest.nms.class_aware.has_value();
  if (path == "segmentation.mask_channels") return manifest.segmentation.mask_channels.has_value();
  if (path == "segmentation.prototype_name") return !manifest.segmentation.prototype_name.empty();
  if (path == "segmentation.prototype_layout") {
    return manifest.segmentation.prototype_layout != Layout::Unknown;
  }
  if (path == "segmentation.prototype_channel_axis") {
    return manifest.segmentation.prototype_channel_axis.has_value();
  }
  if (path == "segmentation.prototype_height") return manifest.segmentation.prototype_height.has_value();
  if (path == "segmentation.prototype_width") return manifest.segmentation.prototype_width.has_value();
  if (path == "segmentation.mask_coefficient_activation") {
    return !manifest.segmentation.mask_coefficient_activation.empty();
  }
  if (path == "segmentation.prototype_activation") {
    return !manifest.segmentation.prototype_activation.empty();
  }
  if (path == "segmentation.has_objectness") return manifest.segmentation.has_objectness.has_value();
  if (path == "segmentation.class_names") return !manifest.segmentation.class_names.empty();
  if (path == "segmentation.retina_masks") return manifest.segmentation.retina_masks.has_value();
  if (path == "segmentation.binary_mask_gt_zero") {
    return manifest.segmentation.binary_mask_gt_zero.has_value();
  }
  if (path == "segmentation.max_mask_bytes") return manifest.segmentation.max_mask_bytes.has_value();
  if (path == "classification.output_name") return !manifest.classification.output_name.empty();
  if (path == "classification.class_axis") return manifest.classification.class_axis.has_value();
  if (path == "classification.score_type") return !manifest.classification.score_type.empty();
  if (path == "classification.class_names") return !manifest.classification.class_names.empty();
  if (path == "pose.output_name") return !manifest.pose.output_name.empty();
  if (path == "pose.channel_axis") return manifest.pose.channel_axis.has_value();
  if (path == "pose.keypoint_count") return manifest.pose.keypoint_count.has_value();
  if (path == "pose.keypoint_dimensions") return manifest.pose.keypoint_dimensions.has_value();
  if (path == "pose.keypoint_layout") return !manifest.pose.keypoint_layout.empty();
  if (path == "pose.score_type") return !manifest.pose.score_type.empty();
  if (path == "obb.output_name") return !manifest.obb.output_name.empty();
  if (path == "obb.channel_axis") return manifest.obb.channel_axis.has_value();
  if (path == "obb.angle_unit") return !manifest.obb.angle_unit.empty();
  if (path == "obb.angle_range") return !manifest.obb.angle_range.empty();
  if (path == "obb.score_type") return !manifest.obb.score_type.empty();

  std::size_t index = 0;
  std::string field;
  if (!parse_output_field(path, index, field) || !is_output_field(field)) return false;
  const OutputDesc* output = output_at(manifest, index);
  if (output == nullptr) return false;
  return output_field_set(*output, field);
}

FieldState inspect_field(const ModelManifest& manifest, std::string_view path) {
  FieldState state;
  state.set = field_is_set(manifest, path);
  const auto found = manifest.provenance.find(std::string(path));
  if (found != manifest.provenance.end()) {
    state.provenance = found->second;
    return state;
  }
  if (state.set) {
    state.provenance.certainty = Certainty::User;
    state.provenance.evidence.push_back("manifest value without provenance");
  }
  return state;
}

void ModelManifest::ensure_provenance() {
  for (const std::string& path : manifest_field_paths(*this)) {
    if (provenance.find(path) == provenance.end()) {
      provenance.emplace(path, inspect_field(*this, path).provenance);
    }
  }
}

const FieldProvenance* ModelManifest::find_provenance(std::string_view path) const {
  const auto found = provenance.find(std::string(path));
  if (found == provenance.end()) return nullptr;
  return &found->second;
}

ModelManifest ModelManifest::from_json_string(std::string_view json_text) {
  json document;
  try {
    document = json::parse(json_text.begin(), json_text.end());
  } catch (const VieError&) {
    throw;
  } catch (const json::exception& error) {
    schema_error(std::string("manifest JSON parse error: ") + error.what());
  }

  try {
    if (!document.is_object()) schema_error("manifest must be a JSON object");
    if (!document.contains("schema_version") || !document.at("schema_version").is_string()) {
      schema_error("schema_version is required");
    }
    ModelManifest manifest;
    manifest.schema_version = document.at("schema_version").get<std::string>();
    if (manifest.schema_version != "1.0") {
      schema_error("unsupported manifest schema_version '" + manifest.schema_version + "'");
    }
    manifest.model_name = read_free_string(document, "model_name", true);
    manifest.model_family = read_free_string(document, "model_family", false);
    manifest.model_version = read_free_string(document, "model_version", false);
    manifest.task = read_enum(document, "task", task_type_from_string, TaskType::Unknown);
    manifest.model_format =
        read_enum(document, "model_format", model_format_from_string, ModelFormat::Unspecified);
    manifest.model_path = read_free_string(document, "model_path", false);
    if (document.contains("num_classes") && !document.at("num_classes").is_null()) {
      if (!document.at("num_classes").is_number_integer()) {
        schema_error("num_classes must be an integer");
      }
      const int classes = document.at("num_classes").get<int>();
      if (classes < 1) schema_error("num_classes must be >= 1");
      manifest.num_classes = classes;
    }
    if (document.contains("input") && !document.at("input").is_null()) {
      read_input(document.at("input"), manifest.input);
    }
    if (document.contains("outputs") && !document.at("outputs").is_null()) {
      if (!document.at("outputs").is_array()) schema_error("outputs must be an array");
      std::size_t index = 0;
      for (const auto& item : document.at("outputs")) {
        manifest.outputs.push_back(read_output(item, index));
        ++index;
      }
    }
    if (document.contains("decoder") && !document.at("decoder").is_null()) {
      read_decoder(document.at("decoder"), manifest.decoder);
    }
    if (document.contains("nms") && !document.at("nms").is_null()) {
      read_nms(document.at("nms"), manifest.nms);
    }
    if (document.contains("segmentation") && !document.at("segmentation").is_null()) {
      read_segmentation(document.at("segmentation"), manifest.segmentation);
    }
    if (document.contains("classification") && !document.at("classification").is_null()) {
      read_classification(document.at("classification"), manifest.classification);
    }
    if (document.contains("obb") && !document.at("obb").is_null()) {
      read_obb(document.at("obb"), manifest.obb);
    }
    if (document.contains("pose") && !document.at("pose").is_null()) {
      read_pose(document.at("pose"), manifest.pose);
    }
    if (document.contains("provenance") && !document.at("provenance").is_null()) {
      read_provenance(document.at("provenance"), manifest);
    }
    for (const auto& entry : manifest.provenance) {
      if (!is_manifest_field_path(manifest, entry.first)) {
        schema_error("unknown provenance path: " + entry.first);
      }
      const bool set = field_is_set(manifest, entry.first);
      if (set && entry.second.certainty == Certainty::Unresolved) {
        schema_error(entry.first + " is set but provenance is unresolved");
      }
      if (!set && entry.second.certainty != Certainty::Unresolved) {
        schema_error(entry.first + " is unset but provenance is " +
                     to_string(entry.second.certainty));
      }
    }
    manifest.ensure_provenance();
    return manifest;
  } catch (const VieError&) {
    throw;
  } catch (const json::exception& error) {
    schema_error(std::string("manifest JSON type error: ") + error.what());
  }
}

ModelManifest ModelManifest::load_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw VieError(ErrorCode::InvalidArgument, "cannot open manifest: " + path);
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  return from_json_string(buffer.str());
}

std::string ModelManifest::to_json_string(int indent) const {
  ModelManifest copy = *this;
  copy.ensure_provenance();
  const json document = manifest_json(copy);
  if (indent < 0) return document.dump();
  return document.dump(indent);
}

void ModelManifest::save_file(const std::string& path) const {
  std::ofstream output(path);
  if (!output) {
    throw VieError(ErrorCode::InvalidArgument, "cannot write manifest: " + path);
  }
  output << to_json_string(2) << '\n';
  if (!output) {
    throw VieError(ErrorCode::InvalidArgument, "failed while writing manifest: " + path);
  }
}

}  // namespace vie
