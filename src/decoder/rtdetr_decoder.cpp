#include "vie/decoder/rtdetr_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace vie {
namespace {

struct KeptQuery {
  BoundingBox box;
  float score = 0.f;
  int class_id = -1;
  int query = 0;
};

void require_finite(float value, const char* field, int query) {
  if (!std::isfinite(value)) {
    throw VieError(ErrorCode::InvalidArgument, rtdetr_nonfinite_message(field, query));
  }
}

}  // namespace

void validate_rtdetr_decoder_config(const DecoderConfig& config) {
  if (config.output_name.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder output_name is empty");
  }
  if (config.box_decode != BoxDecode::DecodedNormalizedCxcywh) {
    throw VieError(ErrorCode::InvalidArgument,
                   "rtdetr_detection_raw requires decoded_normalized_cxcywh");
  }
  if (config.channel_axis != 2) {
    throw VieError(ErrorCode::InvalidArgument, "rtdetr_detection_raw requires channel axis 2");
  }
  if (config.box_channels != 4) {
    throw VieError(ErrorCode::InvalidArgument, "rtdetr_detection_raw requires 4 box channels");
  }
  if (config.num_classes < 1) {
    throw VieError(ErrorCode::InvalidArgument, "decoder num_classes must be >= 1");
  }
  if (!config.num_queries.has_value() || *config.num_queries < 1) {
    throw VieError(ErrorCode::UnresolvedField, "decoder.num_queries is unresolved");
  }
  if (!config.scores_are_probabilities.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "decoder.scores_are_probabilities is unresolved");
  }
  if (!*config.scores_are_probabilities) {
    throw VieError(ErrorCode::Unsupported,
                   "rtdetr_detection_raw does not apply Sigmoid to class scores");
  }
  if (!config.class_selection.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "decoder.class_selection is unresolved");
  }
  if (*config.class_selection != ClassSelection::BestClass) {
    throw VieError(ErrorCode::Unsupported,
                   "rtdetr_detection_raw keeps one class per query and does not expand every class");
  }
  if (!config.confidence_threshold.has_value() || !std::isfinite(*config.confidence_threshold) ||
      *config.confidence_threshold < 0.f || *config.confidence_threshold > 1.f) {
    throw VieError(ErrorCode::UnresolvedField,
                   "nms.confidence_threshold is unresolved or outside [0, 1]");
  }
  if (!config.nms_enabled.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "nms.enabled is unresolved");
  }
  if (*config.nms_enabled) {
    throw VieError(ErrorCode::Unsupported, "rtdetr_detection_raw does not run NMS");
  }
  if (!config.max_detections.has_value() || *config.max_detections < 1) {
    throw VieError(ErrorCode::UnresolvedField, "nms.max_detections is unresolved");
  }
}

std::string RtdetrDecoder::protocol() const { return "rtdetr_detection_raw"; }

RawPredictions RtdetrDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "rtdetr_detection_raw decodes through decode_batch");
}

DetectionBatch RtdetrDecoder::decode_batch(const DecoderInput& input, const DecoderConfig& config,
                                          const DeviceContext& device) const {
  validate_rtdetr_decoder_config(config);
  if (device.device() != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "rtdetr_detection_raw has no CUDA decoder");
  }
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField,
                   "preprocess geometry is unresolved; normalized coordinates are not original-image pixels");
  }
  const GeometryTransform& geometry = *input.geometry;
  if (geometry.output_width <= 0 || geometry.output_height <= 0) {
    throw VieError(ErrorCode::InvalidArgument, "model input width and height must be positive");
  }
  const auto found = input.outputs.find(config.output_name);
  if (found == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder is missing output tensor '" + config.output_name + "'");
  }
  const Tensor& tensor = found->second;
  if (tensor.dtype() != DataType::Fp32 || !tensor.is_cpu()) {
    throw VieError(ErrorCode::InvalidArgument, "rtdetr_detection_raw requires a CPU fp32 output");
  }
  if (!tensor.contiguous()) {
    throw VieError(ErrorCode::Unsupported, "rtdetr_detection_raw does not read a non-contiguous tensor");
  }
  if (tensor.rank() != 3 || tensor.shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument, "rtdetr_detection_raw expects a rank-3 batch of 1");
  }
  const int queries = static_cast<int>(tensor.shape()[1]);
  const int channels = static_cast<int>(tensor.shape()[2]);
  if (queries != *config.num_queries || channels != config.box_channels + config.num_classes) {
    throw VieError(ErrorCode::InvalidArgument,
                   "rtdetr_detection_raw output shape does not match num_queries and 4 + num_classes");
  }

  const float* data = tensor.data_as<float>();
  const float model_width = static_cast<float>(geometry.output_width);
  const float model_height = static_cast<float>(geometry.output_height);
  const float threshold = *config.confidence_threshold;
  std::vector<KeptQuery> kept;
  kept.reserve(static_cast<std::size_t>(queries));
  int passed_threshold = 0;
  for (int query = 0; query < queries; ++query) {
    const float* row = data + static_cast<std::size_t>(query) * static_cast<std::size_t>(channels);
    require_finite(row[0], "normalized cx", query);
    require_finite(row[1], "normalized cy", query);
    require_finite(row[2], "normalized width", query);
    require_finite(row[3], "normalized height", query);
    int best_class = 0;
    float best_score = row[4];
    require_finite(best_score, "class score", query);
    for (int cls = 1; cls < config.num_classes; ++cls) {
      const float score = row[4 + cls];
      require_finite(score, "class score", query);
      if (score > best_score || (score == best_score && cls < best_class)) {
        best_score = score;
        best_class = cls;
      }
    }
    if (best_score < threshold) continue;
    ++passed_threshold;
    const float cx = row[0] * model_width;
    const float cy = row[1] * model_height;
    const float width = row[2] * model_width;
    const float height = row[3] * model_height;
    BoundingBox box;
    cxcywh_to_xyxy(cx, cy, width, height, box.x1, box.y1, box.x2, box.y2);
    if (!map_box_to_original(box, geometry)) continue;
    kept.push_back(KeptQuery{box, best_score, best_class, query});
  }

  std::sort(kept.begin(), kept.end(), [](const KeptQuery& left, const KeptQuery& right) {
    if (left.score != right.score) return left.score > right.score;
    return left.query < right.query;
  });
  const int before_limit = static_cast<int>(kept.size());
  if (before_limit > *config.max_detections) {
    kept.resize(static_cast<std::size_t>(*config.max_detections));
  }

  DetectionResult result;
  result.original_width = geometry.original_width;
  result.original_height = geometry.original_height;
  result.execution_device = DeviceType::Cpu;
  result.execution_device_id = device.device_id();
  result.protocol = protocol();
  result.nms_applied = false;
  result.coordinates_mapped = true;
  result.candidates_after_filter = passed_threshold;
  result.candidates_before_nms = before_limit;
  result.detections.reserve(kept.size());
  for (const KeptQuery& query : kept) {
    // query.query was stored before the sort. It is the model output index,
    // not this detection's position in the sorted vector.
    result.detections.push_back(Detection{query.box, query.score, query.class_id, query.query});
  }
  DetectionBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
}

}  // namespace vie
