#include "yolov8_cpu_decode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace vie {
namespace {

float sigmoid(float value) {
  if (value >= 0.f) {
    const float exponent = std::exp(-value);
    return 1.f / (1.f + exponent);
  }
  const float exponent = std::exp(value);
  return exponent / (1.f + exponent);
}

float activate_score(float value, bool probabilities) {
  if (!std::isfinite(value)) {
    throw VieError(ErrorCode::InvalidArgument, "class score is not finite");
  }
  if (probabilities) {
    if (value < -1e-4f || value > 1.f + 1e-4f) {
      throw VieError(ErrorCode::InvalidArgument,
                     "class score is outside [0, 1] but scores are marked as probabilities");
    }
    if (value < 0.f) return 0.f;
    if (value > 1.f) return 1.f;
    return value;
  }
  const float activated = sigmoid(value);
  if (!std::isfinite(activated)) {
    throw VieError(ErrorCode::InvalidArgument, "sigmoid class score is not finite");
  }
  return activated;
}

float read_channel(const float* data, int channel_axis, int channels, int predictions, int channel,
                   int prediction) {
  const std::size_t index = channel_axis == 1
                                ? static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) +
                                      static_cast<std::size_t>(prediction)
                                : static_cast<std::size_t>(prediction) * static_cast<std::size_t>(channels) +
                                      static_cast<std::size_t>(channel);
  const float value = data[index];
  if (!std::isfinite(value)) {
    throw VieError(ErrorCode::InvalidArgument, "prediction tensor contains a non-finite value");
  }
  return value;
}

bool level_anchor(const std::vector<FeatureLevel>& levels, int prediction, float& anchor_x, float& anchor_y,
                  float& stride) {
  int offset = 0;
  for (const FeatureLevel& level : levels) {
    const int cells = level.grid_width * level.grid_height;
    if (prediction < offset + cells) {
      const int local = prediction - offset;
      const int x = local % level.grid_width;
      const int y = local / level.grid_width;
      anchor_x = static_cast<float>(x) + 0.5f;
      anchor_y = static_cast<float>(y) + 0.5f;
      stride = static_cast<float>(level.stride);
      return true;
    }
    offset += cells;
  }
  return false;
}

double milliseconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct Working {
  BoundingBox model_box;
  BoundingBox box;
  float score = 0.f;
  int class_id = -1;
  int sort_index = 0;
  int prediction_index = -1;
};

}  // namespace

CpuDetectionPass decode_yolov8_cpu_predictions(const float* data, int channel_axis, int channels, int predictions,
                                              const DecoderConfig& config, const GeometryTransform& geometry,
                                              bool record_times) {
  if (data == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "prediction tensor is empty");
  }
  if (channels != config.box_channels + config.num_classes + config.mask_channels) {
    throw VieError(ErrorCode::InvalidArgument, "prediction channels do not match box channels plus classes");
  }
  if (config.box_decode == BoxDecode::Dfl) {
    int cells = 0;
    for (const FeatureLevel& level : config.levels) cells += level.grid_width * level.grid_height;
    if (cells != predictions) {
      throw VieError(ErrorCode::InvalidArgument, "feature levels do not cover the prediction count");
    }
  }
  const bool probabilities = *config.scores_are_probabilities;
  const float threshold = *config.confidence_threshold;
  std::vector<Working> candidates;
  double box_ms = 0;
  double filter_ms = 0;
  for (int prediction = 0; prediction < predictions; ++prediction) {
    const auto box_start = record_times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    BoundingBox box;
    bool keep_box = true;
    if (config.box_decode == BoxDecode::DecodedCxcywh) {
      const float cx = read_channel(data, channel_axis, channels, predictions, 0, prediction);
      const float cy = read_channel(data, channel_axis, channels, predictions, 1, prediction);
      const float width = read_channel(data, channel_axis, channels, predictions, 2, prediction);
      const float height = read_channel(data, channel_axis, channels, predictions, 3, prediction);
      if (width <= 0.f || height <= 0.f) keep_box = false;
      else cxcywh_to_xyxy(cx, cy, width, height, box.x1, box.y1, box.x2, box.y2);
    } else {
      float sides[4];
      for (int side = 0; side < 4; ++side) {
        std::vector<float> logits(static_cast<std::size_t>(config.reg_max));
        for (int bin = 0; bin < config.reg_max; ++bin) {
          logits[static_cast<std::size_t>(bin)] = read_channel(
              data, channel_axis, channels, predictions, side * config.reg_max + bin, prediction);
        }
        sides[side] = dfl_expectation(logits.data(), config.reg_max);
      }
      float anchor_x = 0.f;
      float anchor_y = 0.f;
      float stride = 0.f;
      if (!level_anchor(config.levels, prediction, anchor_x, anchor_y, stride)) {
        throw VieError(ErrorCode::InvalidArgument, "prediction is outside the feature levels");
      }
      dist2bbox(sides[0], sides[1], sides[2], sides[3], anchor_x, anchor_y, stride, box.x1, box.y1, box.x2,
                box.y2);
      if (!(box.x2 > box.x1) || !(box.y2 > box.y1)) keep_box = false;
    }
    if (record_times) box_ms += milliseconds_since(box_start);
    if (!keep_box) continue;
    const auto filter_start = record_times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (*config.class_selection == ClassSelection::BestClass) {
      int best_class = 0;
      float best_score = -1.f;
      for (int cls = 0; cls < config.num_classes; ++cls) {
        const float score = activate_score(
            read_channel(data, channel_axis, channels, predictions, config.box_channels + cls, prediction),
            probabilities);
        if (score > best_score || (score == best_score && cls < best_class)) {
          best_score = score;
          best_class = cls;
        }
      }
      if (best_score >= threshold) {
        candidates.push_back(Working{box, box, best_score, best_class, prediction, prediction});
      }
    } else {
      for (int cls = 0; cls < config.num_classes; ++cls) {
        const float score = activate_score(
            read_channel(data, channel_axis, channels, predictions, config.box_channels + cls, prediction),
            probabilities);
        if (score >= threshold) {
          candidates.push_back(Working{box, box, score, cls, prediction * config.num_classes + cls, prediction});
        }
      }
    }
    if (record_times) filter_ms += milliseconds_since(filter_start);
  }
  const int after_filter = static_cast<int>(candidates.size());
  const auto nms_start = record_times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  std::sort(candidates.begin(), candidates.end(), [](const Working& a, const Working& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.sort_index < b.sort_index;
  });
  if (static_cast<int>(candidates.size()) > *config.pre_nms_top_k) {
    candidates.resize(static_cast<std::size_t>(*config.pre_nms_top_k));
  }
  std::vector<Working> mapped;
  mapped.reserve(candidates.size());
  for (Working candidate : candidates) {
    if (!map_box_to_original(candidate.box, geometry)) continue;
    mapped.push_back(candidate);
  }
  const int before_nms = static_cast<int>(mapped.size());
  std::vector<NmsCandidate> nms_input;
  nms_input.reserve(mapped.size());
  for (const Working& candidate : mapped) {
    nms_input.push_back(NmsCandidate{candidate.box, candidate.score, candidate.class_id, candidate.sort_index});
  }
  std::vector<int> kept;
  if (*config.nms_enabled) {
    kept = greedy_nms(nms_input, *config.iou_threshold, *config.class_aware_nms, *config.max_detections);
  } else {
    const int limit = std::min(*config.max_detections, before_nms);
    kept.resize(static_cast<std::size_t>(limit));
    for (int i = 0; i < limit; ++i) kept[static_cast<std::size_t>(i)] = i;
  }
  CpuDetectionPass pass;
  pass.candidates_after_filter = after_filter;
  pass.candidates_before_nms = before_nms;
  pass.nms_applied = *config.nms_enabled;
  pass.box_ms = box_ms;
  pass.filter_ms = filter_ms;
  pass.nms_ms = record_times ? milliseconds_since(nms_start) : 0;
  pass.kept.reserve(kept.size());
  for (const int index : kept) {
    const Working& candidate = mapped[static_cast<std::size_t>(index)];
    KeptPrediction item;
    item.model_box = candidate.model_box;
    item.original_box = candidate.box;
    item.score = candidate.score;
    item.class_id = candidate.class_id;
    item.prediction_index = candidate.prediction_index;
    pass.kept.push_back(item);
  }
  return pass;
}

}  // namespace vie
