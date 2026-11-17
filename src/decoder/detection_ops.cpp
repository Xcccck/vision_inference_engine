#include "vie/decoder/detection.hpp"

#include "vie/decoder/rtdetr_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vie {
namespace {

void require(bool ready, const std::string& message) {
  if (!ready) throw VieError(ErrorCode::UnresolvedField, message);
}

float clampf(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

}  // namespace

void validate_decoder_config(const DecoderConfig& config) {
  if (config.output_name.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder output_name is empty");
  }
  if (config.channel_axis != 1 && config.channel_axis != 2) {
    throw VieError(ErrorCode::InvalidArgument, "decoder channel_axis must be 1 or 2");
  }
  if (config.num_classes < 1) {
    throw VieError(ErrorCode::InvalidArgument, "decoder num_classes must be >= 1");
  }
  if (!config.scores_are_probabilities.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "decoder.scores_are_probabilities is unresolved");
  }
  if (!config.class_selection.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "decoder.class_selection is unresolved");
  }
  if (!config.confidence_threshold.has_value() || !std::isfinite(*config.confidence_threshold) ||
      *config.confidence_threshold < 0.f || *config.confidence_threshold > 1.f) {
    throw VieError(ErrorCode::UnresolvedField,
                   "nms.confidence_threshold is unresolved or outside [0, 1]");
  }
  if (!config.nms_enabled.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "nms.enabled is unresolved");
  }
  if (!config.max_detections.has_value() || *config.max_detections < 1) {
    throw VieError(ErrorCode::UnresolvedField, "nms.max_detections is unresolved");
  }
  if (!config.pre_nms_top_k.has_value() || *config.pre_nms_top_k < 1 || *config.pre_nms_top_k > 4096) {
    throw VieError(ErrorCode::UnresolvedField, "nms.pre_nms_top_k is unresolved or outside 1..4096");
  }
  if (*config.nms_enabled) {
    if (!config.class_aware_nms.has_value()) {
      throw VieError(ErrorCode::UnresolvedField, "nms.class_aware is unresolved");
    }
    if (!config.iou_threshold.has_value() || !std::isfinite(*config.iou_threshold) ||
        *config.iou_threshold < 0.f || *config.iou_threshold > 1.f) {
      throw VieError(ErrorCode::UnresolvedField, "nms.iou_threshold is unresolved or outside [0, 1]");
    }
  }
  if (config.box_decode == BoxDecode::DecodedCxcywh) {
    if (config.box_channels != 4) {
      throw VieError(ErrorCode::InvalidArgument, "decoded cxcywh requires 4 box channels");
    }
    return;
  }
  if (config.box_decode == BoxDecode::DecodedNormalizedCxcywh) {
    throw VieError(ErrorCode::InvalidArgument,
                   "decoded_normalized_cxcywh is not a YOLOv8 box decode");
  }
  if (config.reg_max < 2) {
    throw VieError(ErrorCode::UnresolvedField, "decoder.reg_max is unresolved");
  }
  if (config.box_channels != 4 * config.reg_max) {
    throw VieError(ErrorCode::InvalidArgument, "DFL box channels must equal 4 * reg_max");
  }
  if (config.levels.empty()) {
    throw VieError(ErrorCode::UnresolvedField, "decoder feature levels are unresolved");
  }
  for (const FeatureLevel& level : config.levels) {
    if (level.stride <= 0 || level.grid_width <= 0 || level.grid_height <= 0) {
      throw VieError(ErrorCode::InvalidArgument, "feature level stride and grid must be positive");
    }
  }
}

float dfl_expectation(const float* logits, int reg_max) {
  if (logits == nullptr || reg_max < 2) {
    throw VieError(ErrorCode::InvalidArgument, "DFL requires reg_max >= 2");
  }
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < reg_max; ++i) {
    if (!std::isfinite(logits[i])) {
      throw VieError(ErrorCode::InvalidArgument, "DFL logit is not finite");
    }
    max_logit = std::max(max_logit, logits[i]);
  }
  float sum = 0.f;
  for (int i = 0; i < reg_max; ++i) sum += std::exp(logits[i] - max_logit);
  if (!(sum > 0.f) || !std::isfinite(sum)) {
    throw VieError(ErrorCode::InvalidArgument, "DFL softmax is not finite");
  }
  float expectation = 0.f;
  for (int i = 0; i < reg_max; ++i) {
    expectation += static_cast<float>(i) * (std::exp(logits[i] - max_logit) / sum);
  }
  return expectation;
}

void dist2bbox(float left, float top, float right, float bottom, float anchor_x, float anchor_y,
               float stride, float& x1, float& y1, float& x2, float& y2) {
  x1 = (anchor_x - left) * stride;
  y1 = (anchor_y - top) * stride;
  x2 = (anchor_x + right) * stride;
  y2 = (anchor_y + bottom) * stride;
}

void cxcywh_to_xyxy(float cx, float cy, float width, float height, float& x1, float& y1, float& x2,
                    float& y2) {
  x1 = cx - width * 0.5f;
  y1 = cy - height * 0.5f;
  x2 = cx + width * 0.5f;
  y2 = cy + height * 0.5f;
}

bool map_box_to_original(BoundingBox& box, const GeometryTransform& geometry) {
  if (geometry.original_width <= 0 || geometry.original_height <= 0 || geometry.scale_x == 0.0 ||
      geometry.scale_y == 0.0) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale or original size is invalid");
  }
  const auto map_x = [&](float value) {
    return static_cast<float>((static_cast<double>(value) - static_cast<double>(geometry.pad_left)) /
                              geometry.scale_x);
  };
  const auto map_y = [&](float value) {
    return static_cast<float>((static_cast<double>(value) - static_cast<double>(geometry.pad_top)) /
                              geometry.scale_y);
  };
  box.x1 = map_x(box.x1);
  box.x2 = map_x(box.x2);
  box.y1 = map_y(box.y1);
  box.y2 = map_y(box.y2);
  const float width = static_cast<float>(geometry.original_width);
  const float height = static_cast<float>(geometry.original_height);
  box.x1 = clampf(box.x1, 0.f, width);
  box.x2 = clampf(box.x2, 0.f, width);
  box.y1 = clampf(box.y1, 0.f, height);
  box.y2 = clampf(box.y2, 0.f, height);
  return box.x2 > box.x1 && box.y2 > box.y1;
}

float intersection_over_union(const BoundingBox& a, const BoundingBox& b) {
  const float left = std::max(a.x1, b.x1);
  const float top = std::max(a.y1, b.y1);
  const float right = std::min(a.x2, b.x2);
  const float bottom = std::min(a.y2, b.y2);
  const float width = right - left;
  const float height = bottom - top;
  if (width <= 0.f || height <= 0.f) return 0.f;
  const float intersection = width * height;
  const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  const float union_area = area_a + area_b - intersection;
  if (!(union_area > 0.f) || !std::isfinite(union_area)) return 0.f;
  return intersection / union_area;
}

std::vector<int> greedy_nms(const std::vector<NmsCandidate>& sorted, float iou_threshold, bool class_aware,
                            int max_detections) {
  std::vector<char> suppressed(sorted.size(), 0);
  std::vector<int> kept;
  kept.reserve(static_cast<std::size_t>(std::min(max_detections, static_cast<int>(sorted.size()))));
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (suppressed[i] != 0) continue;
    kept.push_back(static_cast<int>(i));
    if (static_cast<int>(kept.size()) == max_detections) break;
    for (std::size_t j = i + 1; j < sorted.size(); ++j) {
      if (suppressed[j] != 0) continue;
      if (class_aware && sorted[j].class_id != sorted[i].class_id) continue;
      if (intersection_over_union(sorted[i].box, sorted[j].box) >= iou_threshold) suppressed[j] = 1;
    }
  }
  return kept;
}

DecoderConfig rtdetr_decoder_config(const ModelManifest& manifest) {
  require(manifest.decoder.box_decode == "decoded_normalized_cxcywh",
          "decoder.box_decode is not decoded_normalized_cxcywh");
  require(manifest.decoder.box_format == "cxcywh_normalized",
          "decoder.box_format is not cxcywh_normalized");
  require(manifest.decoder.scores_are_probabilities.has_value(),
          "decoder.scores_are_probabilities is unresolved");
  require(manifest.decoder.box_channels.has_value(), "decoder.box_channels is unresolved");
  require(manifest.decoder.channel_axis.has_value(), "decoder.channel_axis is unresolved");
  require(manifest.decoder.num_queries.has_value(), "decoder.num_queries is unresolved");
  require(manifest.num_classes.has_value(), "num_classes is unresolved");
  const OutputDesc* prediction = nullptr;
  for (const OutputDesc& output : manifest.outputs) {
    if (output.semantic == "predictions") prediction = &output;
  }
  require(prediction != nullptr && !prediction->name.empty(), "outputs semantic predictions is unresolved");
  DecoderConfig config;
  config.output_name = prediction->name;
  config.channel_axis = *manifest.decoder.channel_axis;
  config.box_channels = *manifest.decoder.box_channels;
  config.num_classes = *manifest.num_classes;
  config.num_queries = manifest.decoder.num_queries;
  config.box_decode = BoxDecode::DecodedNormalizedCxcywh;
  config.scores_are_probabilities = manifest.decoder.scores_are_probabilities;
  if (manifest.decoder.class_selection.empty()) {
    throw VieError(ErrorCode::UnresolvedField, "decoder.class_selection is unresolved");
  }
  if (manifest.decoder.class_selection == "best_class") {
    config.class_selection = ClassSelection::BestClass;
  } else if (manifest.decoder.class_selection == "all_above_threshold") {
    config.class_selection = ClassSelection::AllAboveThreshold;
  } else {
    throw VieError(ErrorCode::InvalidArgument, "decoder.class_selection is not a known value");
  }
  config.confidence_threshold = manifest.nms.confidence_threshold;
  config.nms_enabled = manifest.nms.enabled;
  config.max_detections = manifest.nms.max_detections;
  validate_rtdetr_decoder_config(config);
  return config;
}

DecoderConfig decoder_config_from_manifest(const ModelManifest& manifest) {
  if (manifest.decoder.protocol == "rtdetr_detection_raw") return rtdetr_decoder_config(manifest);
  require(!manifest.decoder.box_decode.empty(), "decoder.box_decode is unresolved");
  require(manifest.decoder.scores_are_probabilities.has_value(),
          "decoder.scores_are_probabilities is unresolved");
  require(!manifest.decoder.class_selection.empty(), "decoder.class_selection is unresolved");
  require(manifest.decoder.box_channels.has_value(), "decoder.box_channels is unresolved");
  require(manifest.decoder.channel_axis.has_value(), "decoder.channel_axis is unresolved");
  require(manifest.num_classes.has_value(), "num_classes is unresolved");
  require(!manifest.outputs.empty() && !manifest.outputs.front().name.empty(),
          "outputs.0.name is unresolved");
  DecoderConfig config;
  config.output_name = manifest.outputs.front().name;
  config.channel_axis = *manifest.decoder.channel_axis;
  config.box_channels = *manifest.decoder.box_channels;
  config.num_classes = *manifest.num_classes;
  config.scores_are_probabilities = manifest.decoder.scores_are_probabilities;
  if (manifest.decoder.box_decode == "decoded_cxcywh") {
    config.box_decode = BoxDecode::DecodedCxcywh;
  } else if (manifest.decoder.box_decode == "dfl") {
    config.box_decode = BoxDecode::Dfl;
    require(manifest.decoder.reg_max.has_value(), "decoder.reg_max is unresolved");
    config.reg_max = *manifest.decoder.reg_max;
    throw VieError(ErrorCode::UnresolvedField,
                   "decoder feature levels are unresolved; pass DecoderConfig explicitly");
  } else {
    throw VieError(ErrorCode::InvalidArgument, "decoder.box_decode is not decoded_cxcywh or dfl");
  }
  if (manifest.decoder.class_selection == "best_class") {
    config.class_selection = ClassSelection::BestClass;
  } else if (manifest.decoder.class_selection == "all_above_threshold") {
    config.class_selection = ClassSelection::AllAboveThreshold;
  } else {
    throw VieError(ErrorCode::InvalidArgument, "decoder.class_selection is not a known value");
  }
  config.confidence_threshold = manifest.nms.confidence_threshold;
  config.nms_enabled = manifest.nms.enabled;
  config.class_aware_nms = manifest.nms.class_aware;
  config.iou_threshold = manifest.nms.iou_threshold;
  config.max_detections = manifest.nms.max_detections;
  config.pre_nms_top_k = manifest.nms.pre_nms_top_k;
  if (config.box_decode == BoxDecode::DecodedCxcywh && manifest.decoder.box_format != "cxcywh") {
    throw VieError(ErrorCode::UnresolvedField, "decoder.box_format is not cxcywh for decoded boxes");
  }
  validate_decoder_config(config);
  return config;
}

}  // namespace vie
