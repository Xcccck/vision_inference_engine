#include "vie/decoder/yolov8_pose_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

namespace vie {
namespace {

constexpr float kScoreSlack = 1e-4f;
constexpr float kInsideSlack = 1e-3f;

double milliseconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

float read_channel(const float* data, int channel_axis, int channels, int predictions, int channel, int prediction) {
  const std::size_t index = channel_axis == 1
                                ? static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) +
                                      static_cast<std::size_t>(prediction)
                                : static_cast<std::size_t>(prediction) * static_cast<std::size_t>(channels) +
                                      static_cast<std::size_t>(channel);
  const float value = data[index];
  if (!std::isfinite(value)) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_pose_raw prediction " + std::to_string(prediction) + " channel " +
                       std::to_string(channel) + " is not finite");
  }
  return value;
}

float probability(float value, int prediction, int channel) {
  if (value < -kScoreSlack || value > 1.f + kScoreSlack) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_pose_raw probability at prediction " + std::to_string(prediction) + " channel " +
                       std::to_string(channel) + " is outside [0, 1]");
  }
  if (value < 0.f) return 0.f;
  if (value > 1.f) return 1.f;
  return value;
}

}  // namespace

void validate_pose_decode_config(const PoseDecodeConfig& config) {
  if (config.output_name.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "pose output_name is empty");
  }
  if (config.channel_axis != 1 && config.channel_axis != 2) {
    throw VieError(ErrorCode::InvalidArgument, "pose channel_axis must be 1 or 2");
  }
  if (config.num_classes < 1) {
    throw VieError(ErrorCode::InvalidArgument, "pose num_classes must be >= 1");
  }
  if (config.keypoint_count < 1) {
    throw VieError(ErrorCode::InvalidArgument, "pose keypoint_count must be >= 1");
  }
  if (config.keypoint_dimensions != 3) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_pose_raw keypoint_dimensions must be 3");
  }
  if (!config.class_selection.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "pose class_selection is unresolved");
  }
  if (!config.confidence_threshold.has_value() || !std::isfinite(*config.confidence_threshold) ||
      *config.confidence_threshold < 0.f || *config.confidence_threshold > 1.f) {
    throw VieError(ErrorCode::UnresolvedField, "pose confidence_threshold is unresolved or outside [0, 1]");
  }
  if (!config.nms_enabled.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "pose nms_enabled is unresolved");
  }
  if (!config.max_detections.has_value() || *config.max_detections < 1) {
    throw VieError(ErrorCode::UnresolvedField, "pose max_detections is unresolved");
  }
  if (!config.pre_nms_top_k.has_value() || *config.pre_nms_top_k < 1 || *config.pre_nms_top_k > 4096) {
    throw VieError(ErrorCode::UnresolvedField, "pose pre_nms_top_k is unresolved or outside 1..4096");
  }
  if (*config.nms_enabled) {
    if (!config.class_aware.has_value()) {
      throw VieError(ErrorCode::UnresolvedField, "pose class_aware is unresolved");
    }
    if (!config.iou_threshold.has_value() || !std::isfinite(*config.iou_threshold) || *config.iou_threshold < 0.f ||
        *config.iou_threshold > 1.f) {
      throw VieError(ErrorCode::UnresolvedField, "pose iou_threshold is unresolved or outside [0, 1]");
    }
  }
}

void map_keypoint_to_original(float model_x, float model_y, const GeometryTransform& geometry, float& original_x,
                              float& original_y, bool& inside_image) {
  if (!std::isfinite(model_x) || !std::isfinite(model_y)) {
    throw VieError(ErrorCode::InvalidArgument, "keypoint model coordinate is not finite");
  }
  if (geometry.original_width <= 0 || geometry.original_height <= 0 || geometry.scale_x == 0.0 ||
      geometry.scale_y == 0.0 || !std::isfinite(geometry.scale_x) || !std::isfinite(geometry.scale_y)) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale or original size is invalid");
  }
  const double x = (static_cast<double>(model_x) - static_cast<double>(geometry_model_origin_x(geometry))) /
                   geometry.scale_x;
  const double y = (static_cast<double>(model_y) - static_cast<double>(geometry_model_origin_y(geometry))) /
                   geometry.scale_y;
  if (!std::isfinite(x) || !std::isfinite(y)) {
    throw VieError(ErrorCode::InvalidArgument, "keypoint inverse map is not finite");
  }
  original_x = static_cast<float>(x);
  original_y = static_cast<float>(y);
  const double width = static_cast<double>(geometry.original_width);
  const double height = static_cast<double>(geometry.original_height);
  inside_image = x >= -static_cast<double>(kInsideSlack) && x <= width + static_cast<double>(kInsideSlack) &&
                 y >= -static_cast<double>(kInsideSlack) && y <= height + static_cast<double>(kInsideSlack);
}

std::string YoloV8PoseDecoder::protocol() const { return "yolov8_pose_raw"; }

RawPredictions YoloV8PoseDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_pose_raw decodes through decode_pose");
}

PoseBatch YoloV8PoseDecoder::decode_pose(const DecoderInput& input, const PoseDecodeConfig& config,
                                         const DeviceContext& device) const {
  validate_pose_decode_config(config);
  if (device.device() != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "yolov8_pose_raw pose decode runs on CPU; copy the tensor first");
  }
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "pose geometry is unresolved");
  }
  const auto found = input.outputs.find(config.output_name);
  if (found == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "pose output '" + config.output_name + "' is missing");
  }
  const Tensor& tensor = found->second;
  if (!tensor.is_cpu() || tensor.dtype() != DataType::Fp32 || !tensor.contiguous()) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_pose_raw expects a contiguous CPU fp32 tensor");
  }
  if (tensor.shape().size() != 3 || tensor.shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_pose_raw expects shape [1, channels, anchors] or [1, anchors, channels]");
  }
  const int channels = static_cast<int>(tensor.shape()[static_cast<std::size_t>(config.channel_axis)]);
  const int prediction_axis = config.channel_axis == 1 ? 2 : 1;
  const int predictions = static_cast<int>(tensor.shape()[static_cast<std::size_t>(prediction_axis)]);
  const int keypoint_channels = config.keypoint_count * config.keypoint_dimensions;
  const int expected = 4 + config.num_classes + keypoint_channels;
  if (channels != expected || predictions < 0) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_pose_raw channels must be 4 + num_classes + keypoint_count * 3");
  }
  const float* data = tensor.data_as<float>();
  if (predictions > 0 && data == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_pose_raw tensor is empty");
  }

  const auto validate_start = std::chrono::steady_clock::now();
  const float threshold = *config.confidence_threshold;
  const int keypoint_base = 4 + config.num_classes;
  struct KeptPerson {
    float cx = 0.f;
    float cy = 0.f;
    float width = 0.f;
    float height = 0.f;
    float score = 0.f;
    int class_id = -1;
    int anchor_index = 0;
    std::vector<Keypoint> model_keypoints;
  };
  std::vector<KeptPerson> candidates;
  for (int prediction = 0; prediction < predictions; ++prediction) {
    std::vector<std::pair<int, float>> hits;
    if (*config.class_selection == ClassSelection::BestClass) {
      int best_class = 0;
      float best_score = -1.f;
      for (int cls = 0; cls < config.num_classes; ++cls) {
        const float score = probability(
            read_channel(data, config.channel_axis, channels, predictions, 4 + cls, prediction), prediction, 4 + cls);
        if (score > best_score || (score == best_score && cls < best_class)) {
          best_score = score;
          best_class = cls;
        }
      }
      if (best_score >= threshold) hits.emplace_back(best_class, best_score);
    } else {
      for (int cls = 0; cls < config.num_classes; ++cls) {
        const float score = probability(
            read_channel(data, config.channel_axis, channels, predictions, 4 + cls, prediction), prediction, 4 + cls);
        if (score >= threshold) hits.emplace_back(cls, score);
      }
    }
    if (hits.empty()) continue;
    const float cx = read_channel(data, config.channel_axis, channels, predictions, 0, prediction);
    const float cy = read_channel(data, config.channel_axis, channels, predictions, 1, prediction);
    const float width = read_channel(data, config.channel_axis, channels, predictions, 2, prediction);
    const float height = read_channel(data, config.channel_axis, channels, predictions, 3, prediction);
    if (!(width > 0.f) || !(height > 0.f)) {
      throw VieError(ErrorCode::InvalidArgument,
                     "yolov8_pose_raw kept anchor " + std::to_string(prediction) + " has a non-positive box size");
    }
    std::vector<Keypoint> model_keypoints;
    model_keypoints.reserve(static_cast<std::size_t>(config.keypoint_count));
    for (int keypoint = 0; keypoint < config.keypoint_count; ++keypoint) {
      const int base = keypoint_base + keypoint * config.keypoint_dimensions;
      Keypoint point;
      point.x = read_channel(data, config.channel_axis, channels, predictions, base, prediction);
      point.y = read_channel(data, config.channel_axis, channels, predictions, base + 1, prediction);
      point.confidence = probability(
          read_channel(data, config.channel_axis, channels, predictions, base + 2, prediction), prediction, base + 2);
      model_keypoints.push_back(point);
    }
    for (const auto& hit : hits) {
      KeptPerson person;
      person.cx = cx;
      person.cy = cy;
      person.width = width;
      person.height = height;
      person.score = hit.second;
      person.class_id = hit.first;
      person.anchor_index = prediction;
      person.model_keypoints = model_keypoints;
      candidates.push_back(std::move(person));
    }
  }
  const double validate_ms = milliseconds_since(validate_start);
  const int after_filter = static_cast<int>(candidates.size());
  std::sort(candidates.begin(), candidates.end(), [](const KeptPerson& left, const KeptPerson& right) {
    if (left.score != right.score) return left.score > right.score;
    if (left.anchor_index != right.anchor_index) return left.anchor_index < right.anchor_index;
    return left.class_id < right.class_id;
  });
  if (static_cast<int>(candidates.size()) > *config.pre_nms_top_k) {
    candidates.resize(static_cast<std::size_t>(*config.pre_nms_top_k));
  }

  const auto decode_start = std::chrono::steady_clock::now();
  std::vector<PoseCandidate> mapped;
  mapped.reserve(candidates.size());
  for (const KeptPerson& candidate : candidates) {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
    cxcywh_to_xyxy(candidate.cx, candidate.cy, candidate.width, candidate.height, x1, y1, x2, y2);
    BoundingBox box{x1, y1, x2, y2};
    if (!map_box_to_original(box, *input.geometry)) continue;
    PoseCandidate kept;
    kept.box = box;
    kept.score = candidate.score;
    kept.class_id = candidate.class_id;
    kept.anchor_index = candidate.anchor_index;
    kept.keypoints.reserve(candidate.model_keypoints.size());
    for (const Keypoint& model_point : candidate.model_keypoints) {
      Keypoint point;
      point.confidence = model_point.confidence;
      map_keypoint_to_original(model_point.x, model_point.y, *input.geometry, point.x, point.y, point.inside_image);
      kept.keypoints.push_back(point);
    }
    mapped.push_back(std::move(kept));
  }
  const double pose_decode_ms = milliseconds_since(decode_start);
  const int before_nms = static_cast<int>(mapped.size());

  std::vector<NmsCandidate> nms_input;
  nms_input.reserve(mapped.size());
  for (const PoseCandidate& candidate : mapped) {
    nms_input.push_back(NmsCandidate{candidate.box, candidate.score, candidate.class_id, candidate.anchor_index});
  }
  const auto nms_start = std::chrono::steady_clock::now();
  std::vector<int> kept;
  if (*config.nms_enabled) {
    kept = greedy_nms(nms_input, *config.iou_threshold, *config.class_aware, *config.max_detections);
  } else {
    const int limit = std::min(*config.max_detections, before_nms);
    kept.resize(static_cast<std::size_t>(limit));
    for (int index = 0; index < limit; ++index) kept[static_cast<std::size_t>(index)] = index;
  }
  const double nms_ms = milliseconds_since(nms_start);

  PoseResult result;
  result.image_width = input.geometry->original_width;
  result.image_height = input.geometry->original_height;
  result.num_classes = config.num_classes;
  result.execution_device = DeviceType::Cpu;
  result.execution_device_id = device.device_id();
  result.protocol = protocol();
  result.output_bytes = tensor.nbytes();
  result.validate_ms = validate_ms;
  result.pose_decode_ms = pose_decode_ms;
  result.nms_ms = nms_ms;
  result.decode_ms = validate_ms + pose_decode_ms + nms_ms;
  result.candidates_after_filter = after_filter;
  result.candidates_before_nms = before_nms;
  result.objects.reserve(kept.size());
  for (const int index : kept) {
    const PoseCandidate& candidate = mapped[static_cast<std::size_t>(index)];
    PoseObject object;
    object.class_id = candidate.class_id;
    object.score = candidate.score;
    object.cx = (candidate.box.x1 + candidate.box.x2) * 0.5f;
    object.cy = (candidate.box.y1 + candidate.box.y2) * 0.5f;
    object.width = candidate.box.x2 - candidate.box.x1;
    object.height = candidate.box.y2 - candidate.box.y1;
    object.keypoints = candidate.keypoints;
    object.anchor_index = candidate.anchor_index;
    result.objects.push_back(std::move(object));
  }
  PoseBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
}

}  // namespace vie
