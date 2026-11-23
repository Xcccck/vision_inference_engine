#include "vie/decoder/classification.hpp"

#include "classification_topk.hpp"
#include "vie/core/error.hpp"

#include <chrono>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace vie {
namespace {

const Tensor* find_output(const DecoderInput& input, const std::string& name) {
  const auto found = input.outputs.find(name);
  if (found == input.outputs.end()) return nullptr;
  return &found->second;
}

std::string named(const std::string& output_name, const std::string& detail) {
  return "classification output '" + output_name + "' " + detail;
}

}  // namespace

std::string ClassificationProbabilitiesDecoder::protocol() const {
  return "classification_probabilities_raw";
}

RawPredictions ClassificationProbabilitiesDecoder::decode(const std::map<std::string, Tensor>&,
                                                         const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported,
                 "classification_probabilities_raw decodes through decode_classification()");
}

ClassificationBatch ClassificationProbabilitiesDecoder::decode_classification(
    const DecoderInput& input, const ClassificationDecodeRequest& request,
    const DeviceContext& device) const {
  if (device.device() != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "classification_probabilities_raw runs on CPU");
  }
  if (request.score_type.empty()) {
    throw VieError(ErrorCode::UnresolvedField, "classification.score_type is unresolved");
  }
  if (request.score_type != "probabilities") {
    throw VieError(ErrorCode::Unsupported,
                   "classification_probabilities_raw accepts score_type probabilities only and does "
                   "not treat logits as probabilities");
  }
  if (request.output_name.empty()) {
    throw VieError(ErrorCode::UnresolvedField, "classification.output_name is unresolved");
  }
  if (request.class_axis < 0) {
    throw VieError(ErrorCode::UnresolvedField, "classification.class_axis is unresolved");
  }
  if (request.num_classes < 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(request.output_name, "num_classes must be >= 1"));
  }
  if (request.top_k < 1) {
    throw VieError(ErrorCode::InvalidArgument, named(request.output_name, "top_k must be >= 1"));
  }
  if (request.top_k > request.num_classes) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(request.output_name, "top_k exceeds num_classes"));
  }
  if (!request.class_names.empty() &&
      static_cast<int>(request.class_names.size()) != request.num_classes) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(request.output_name, "class_names count does not match num_classes"));
  }

  const Tensor* tensor = find_output(input, request.output_name);
  if (tensor == nullptr) {
    throw VieError(ErrorCode::InvalidArgument,
                   "classification output '" + request.output_name + "' is missing");
  }
  if (tensor->device() != DeviceType::Cpu || tensor->is_cuda()) {
    throw VieError(ErrorCode::Unsupported,
                   "classification_probabilities_raw does not read a CUDA tensor");
  }
  if (tensor->dtype() != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument, named(request.output_name, "must be fp32"));
  }
  if (tensor->shape().size() != 2) {
    throw VieError(ErrorCode::InvalidArgument, named(request.output_name, "must be rank 2"));
  }
  if (tensor->numel() == 0) {
    throw VieError(ErrorCode::InvalidArgument, named(request.output_name, "is empty"));
  }
  if (!tensor->contiguous()) {
    throw VieError(ErrorCode::Unsupported,
                   "classification_probabilities_raw does not read a non-contiguous tensor");
  }
  if (request.class_axis > 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(request.output_name, "class_axis is outside rank 2"));
  }

  const std::vector<std::int64_t>& shape = tensor->shape();
  const std::size_t axis = static_cast<std::size_t>(request.class_axis);
  if (shape[axis] != request.num_classes) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(request.output_name, "class axis length does not match num_classes"));
  }
  if (shape[1 - axis] != 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(request.output_name, "batch dimension must be 1"));
  }

  const auto validate_start = std::chrono::steady_clock::now();
  const float* packed = tensor->data_as<float>();
  const std::int64_t stride = tensor->strides()[axis];
  std::vector<float> scores(static_cast<std::size_t>(request.num_classes));
  double sum = 0.0;
  for (int class_id = 0; class_id < request.num_classes; ++class_id) {
    const float score = packed[static_cast<std::size_t>(class_id) * static_cast<std::size_t>(stride)];
    if (!std::isfinite(score)) {
      const char* reason = std::isnan(score) ? "NaN" : (score > 0.f ? "positive infinity" : "negative infinity");
      throw VieError(ErrorCode::InvalidArgument,
                     "classification output '" + request.output_name + "' class_id " +
                         std::to_string(class_id) + " probability is " + reason);
    }
    if (score < 0.f || score > 1.f) {
      const char* reason = score < 0.f ? "below 0" : "above 1";
      throw VieError(ErrorCode::InvalidArgument,
                     "classification output '" + request.output_name + "' class_id " +
                         std::to_string(class_id) + " probability is " + reason);
    }
    scores[static_cast<std::size_t>(class_id)] = score;
    sum += static_cast<double>(score);
  }
  const double delta = std::fabs(sum - 1.0);
  if (delta > kClassificationProbabilitySumTolerance) {
    std::ostringstream message;
    message << "classification output '" << request.output_name
            << "' probability sum " << sum << " differs from 1 by " << delta
            << ", which is outside tolerance " << kClassificationProbabilitySumTolerance;
    throw VieError(ErrorCode::InvalidArgument, message.str());
  }
  const double validate_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - validate_start)
                                 .count();

  const auto topk_start = std::chrono::steady_clock::now();
  const std::vector<RankedScore> ranked = rank_class_scores(scores, request.top_k);
  const double topk_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - topk_start).count();

  ClassificationResult result;
  result.output_name = request.output_name;
  result.protocol = protocol();
  result.score_type = request.score_type;
  result.num_classes = request.num_classes;
  result.top_k = request.top_k;
  result.execution_device = DeviceType::Cpu;
  result.execution_device_id = device.device_id();
  result.output_bytes = tensor->nbytes();
  result.d2h_bytes = 0;
  result.validate_ms = validate_ms;
  result.topk_ms = topk_ms;
  result.labels.reserve(static_cast<std::size_t>(request.top_k));
  for (const RankedScore& ranked_score : ranked) {
    Classification label;
    label.class_id = ranked_score.class_id;
    label.score = ranked_score.score;
    if (!request.class_names.empty()) {
      if (ranked_score.class_id < 0 ||
          ranked_score.class_id >= static_cast<int>(request.class_names.size())) {
        throw VieError(ErrorCode::InvalidArgument,
                       "classification output '" + request.output_name + "' class_id " +
                           std::to_string(ranked_score.class_id) + " is outside class_names");
      }
      label.class_name = request.class_names[static_cast<std::size_t>(ranked_score.class_id)];
    }
    result.labels.push_back(std::move(label));
  }

  ClassificationBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
}

}  // namespace vie
