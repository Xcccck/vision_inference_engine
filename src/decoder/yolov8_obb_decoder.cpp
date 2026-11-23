#include "vie/decoder/yolov8_obb_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

namespace vie {
namespace {

constexpr float kAngleMin = -0.7853981633974483f;
constexpr float kAngleMax = 2.356194490192345f;
constexpr float kAngleSlack = 1e-3f;
constexpr float kScoreSlack = 1e-4f;

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
                   "yolov8_obb_raw prediction " + std::to_string(prediction) + " channel " +
                       std::to_string(channel) + " is not finite");
  }
  return value;
}

float probability(float value, int prediction, int channel) {
  if (value < -kScoreSlack || value > 1.f + kScoreSlack) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_obb_raw class score at prediction " + std::to_string(prediction) + " channel " +
                       std::to_string(channel) + " is outside [0, 1]");
  }
  if (value < 0.f) return 0.f;
  if (value > 1.f) return 1.f;
  return value;
}

}  // namespace

std::string YoloV8ObbDecoder::protocol() const { return "yolov8_obb_raw"; }

RawPredictions YoloV8ObbDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_obb_raw decodes through decode_obb");
}

ObbBatch YoloV8ObbDecoder::decode_obb(const DecoderInput& input, const ObbDecodeConfig& config,
                                      const DeviceContext& device) const {
  validate_obb_decode_config(config);
  if (device.device() != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "yolov8_obb_raw rotated NMS runs on CPU; copy the tensor first");
  }
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "obb geometry is unresolved");
  }
  const auto found = input.outputs.find(config.output_name);
  if (found == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "obb output '" + config.output_name + "' is missing");
  }
  const Tensor& tensor = found->second;
  if (!tensor.is_cpu() || tensor.dtype() != DataType::Fp32 || !tensor.contiguous()) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_obb_raw expects a contiguous CPU fp32 tensor");
  }
  if (tensor.shape().size() != 3 || tensor.shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_obb_raw expects shape [1, channels, anchors] or [1, anchors, channels]");
  }
  const int channels = static_cast<int>(tensor.shape()[static_cast<std::size_t>(config.channel_axis)]);
  const int prediction_axis = config.channel_axis == 1 ? 2 : 1;
  const int predictions = static_cast<int>(tensor.shape()[static_cast<std::size_t>(prediction_axis)]);
  const int expected = 4 + config.num_classes + 1;
  if (channels != expected || predictions < 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_obb_raw channels must be 4 + num_classes + 1 and the anchor count must be positive");
  }
  const float* data = tensor.data_as<float>();
  if (data == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_obb_raw tensor is empty");
  }

  const auto validate_start = std::chrono::steady_clock::now();
  const float threshold = *config.confidence_threshold;
  const int angle_channel = 4 + config.num_classes;
  std::vector<ObbCandidate> candidates;
  for (int prediction = 0; prediction < predictions; ++prediction) {
    const float cx = read_channel(data, config.channel_axis, channels, predictions, 0, prediction);
    const float cy = read_channel(data, config.channel_axis, channels, predictions, 1, prediction);
    const float width = read_channel(data, config.channel_axis, channels, predictions, 2, prediction);
    const float height = read_channel(data, config.channel_axis, channels, predictions, 3, prediction);
    const float angle = read_channel(data, config.channel_axis, channels, predictions, angle_channel, prediction);
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
      if (best_score < threshold) continue;
      if (!(width > 0.f) || !(height > 0.f) || angle < kAngleMin - kAngleSlack || angle > kAngleMax + kAngleSlack) {
        throw VieError(ErrorCode::InvalidArgument,
                       "yolov8_obb_raw kept anchor " + std::to_string(prediction) +
                           " has a non-positive size or an angle outside [-pi/4, 3pi/4]");
      }
      candidates.push_back(ObbCandidate{OrientedBox{cx, cy, width, height, angle}, best_score, best_class, prediction});
    } else {
      for (int cls = 0; cls < config.num_classes; ++cls) {
        const float score = probability(
            read_channel(data, config.channel_axis, channels, predictions, 4 + cls, prediction), prediction, 4 + cls);
        if (score < threshold) continue;
        if (!(width > 0.f) || !(height > 0.f) || angle < kAngleMin - kAngleSlack || angle > kAngleMax + kAngleSlack) {
          throw VieError(ErrorCode::InvalidArgument,
                         "yolov8_obb_raw kept anchor " + std::to_string(prediction) +
                             " has a non-positive size or an angle outside [-pi/4, 3pi/4]");
        }
        candidates.push_back(ObbCandidate{OrientedBox{cx, cy, width, height, angle}, score, cls, prediction});
      }
    }
  }
  const double validate_ms = milliseconds_since(validate_start);
  const int after_filter = static_cast<int>(candidates.size());
  std::sort(candidates.begin(), candidates.end(), [](const ObbCandidate& left, const ObbCandidate& right) {
    if (left.score != right.score) return left.score > right.score;
    if (left.anchor_index != right.anchor_index) return left.anchor_index < right.anchor_index;
    return left.class_id < right.class_id;
  });
  if (static_cast<int>(candidates.size()) > *config.pre_nms_top_k) {
    candidates.resize(static_cast<std::size_t>(*config.pre_nms_top_k));
  }

  std::vector<ObbCandidate> mapped;
  mapped.reserve(candidates.size());
  for (const ObbCandidate& candidate : candidates) {
    OrientedBox original;
    if (!map_obb_to_original(candidate.box, *input.geometry, original)) continue;
    ObbCandidate kept = candidate;
    kept.box = original;
    mapped.push_back(kept);
  }
  const int before_nms = static_cast<int>(mapped.size());
  const auto nms_start = std::chrono::steady_clock::now();
  std::vector<int> kept;
  if (*config.nms_enabled) {
    kept = rotated_nms(mapped, *config.iou_threshold, *config.class_aware, *config.max_detections);
  } else {
    const int limit = std::min(*config.max_detections, before_nms);
    kept.resize(static_cast<std::size_t>(limit));
    for (int index = 0; index < limit; ++index) kept[static_cast<std::size_t>(index)] = index;
  }
  const double nms_ms = milliseconds_since(nms_start);

  ObbResult result;
  result.image_width = input.geometry->original_width;
  result.image_height = input.geometry->original_height;
  result.num_classes = config.num_classes;
  result.execution_device = DeviceType::Cpu;
  result.execution_device_id = device.device_id();
  result.protocol = protocol();
  result.output_bytes = tensor.nbytes();
  result.validate_ms = validate_ms;
  result.nms_ms = nms_ms;
  result.decode_ms = validate_ms + nms_ms;
  result.candidates_after_filter = after_filter;
  result.candidates_before_nms = before_nms;
  result.objects.reserve(kept.size());
  for (const int index : kept) {
    const ObbCandidate& candidate = mapped[static_cast<std::size_t>(index)];
    result.objects.push_back(ObbObject{candidate.class_id, candidate.score, candidate.box, candidate.anchor_index});
  }
  ObbBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
}

}  // namespace vie
