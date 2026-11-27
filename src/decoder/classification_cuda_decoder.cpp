#include "vie/decoder/classification_cuda_decoder.hpp"

#include "vie/decoder/classification.hpp"

#if defined(VIE_HAS_CUDA)
#include "classification_cuda.hpp"
#endif

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace vie {
namespace {

std::string protocol_name(bool probabilities) {
  return probabilities ? "classification_probabilities_raw" : "classification_logits_raw";
}

#if defined(VIE_HAS_CUDA)

const Tensor* find_output(const DecoderInput& input, const std::string& name) {
  const auto found = input.outputs.find(name);
  if (found == input.outputs.end()) return nullptr;
  return &found->second;
}

std::string named(const std::string& output_name, const std::string& detail) {
  return "classification output '" + output_name + "' " + detail;
}

// Mirrors the request checks of the matching CPU decoder. The protocol-specific
// score_type wording is shared between the two modes.
void validate_request(const ClassificationDecodeRequest& request, bool probabilities) {
  if (request.score_type.empty()) {
    throw VieError(ErrorCode::UnresolvedField, "classification.score_type is unresolved");
  }
  if (probabilities) {
    if (request.score_type != "probabilities") {
      throw VieError(ErrorCode::Unsupported,
                     "classification_probabilities_raw accepts score_type probabilities only and does "
                     "not treat logits as probabilities");
    }
  } else if (request.score_type != "logits") {
    throw VieError(ErrorCode::Unsupported,
                   "classification_logits_raw accepts score_type logits only and does not run Softmax");
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
}

#endif  // VIE_HAS_CUDA

}  // namespace

ClassificationCudaDecoder::ClassificationCudaDecoder(bool probabilities_mode)
    : probabilities_mode_(probabilities_mode) {}

std::string ClassificationCudaDecoder::protocol() const { return protocol_name(probabilities_mode_); }

RawPredictions ClassificationCudaDecoder::decode(const std::map<std::string, Tensor>&,
                                                 const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, protocol() + " decodes through decode_classification()");
}

ClassificationBatch ClassificationCudaDecoder::decode_classification(
    const DecoderInput& input, const ClassificationDecodeRequest& request,
    const DeviceContext& device) const {
#if !defined(VIE_HAS_CUDA)
  (void)input;
  (void)request;
  (void)device;
  throw VieError(ErrorCode::Unsupported,
                 protocol() + " CUDA decoder is not available in this build");
#else
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported,
                   protocol() + " CUDA decoder does not run on a CPU device");
  }
  validate_request(request, probabilities_mode_);

  const Tensor* tensor = find_output(input, request.output_name);
  if (tensor == nullptr) {
    throw VieError(ErrorCode::InvalidArgument,
                   "classification output '" + request.output_name + "' is missing");
  }
  if (!tensor->is_cuda()) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(request.output_name, "must be a CUDA tensor for the CUDA decoder"));
  }
  if (tensor->device_id() != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(request.output_name, "device does not match the decoder device"));
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
                   protocol() + " does not read a non-contiguous tensor");
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

  const std::int64_t stride = tensor->strides()[axis];
  const ClassificationCudaLaunch launch = launch_classification_cuda(
      reinterpret_cast<const float*>(tensor->device_data()), request.num_classes, stride,
      request.top_k, probabilities_mode_, request.output_name, device.device_id());

  ClassificationResult result;
  result.output_name = request.output_name;
  result.protocol = protocol();
  result.score_type = request.score_type;
  result.num_classes = request.num_classes;
  result.top_k = request.top_k;
  result.execution_device = DeviceType::Cuda;
  result.execution_device_id = device.device_id();
  result.output_bytes = tensor->nbytes();
  result.d2h_bytes = launch.d2h_bytes;
  result.validate_ms = launch.validate_ms;
  result.topk_ms = launch.sort_ms;
  result.d2h_ms = launch.d2h_ms;
  result.labels.reserve(static_cast<std::size_t>(request.top_k));
  for (const RankedScore& ranked_score : launch.ranked) {
    Classification label;
    label.class_id = ranked_score.class_id;
    label.score = ranked_score.score;
    if (!request.class_names.empty()) {
      if (ranked_score.class_id < 0 ||
          ranked_score.class_id >= static_cast<int>(request.class_names.size())) {
        throw VieError(ErrorCode::InvalidArgument,
                       named(request.output_name, "class_id " +
                                                      std::to_string(ranked_score.class_id) +
                                                      " is outside class_names"));
      }
      label.class_name = request.class_names[static_cast<std::size_t>(ranked_score.class_id)];
    }
    result.labels.push_back(std::move(label));
  }

  ClassificationBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
#endif
}

ClassificationDispatchDecoder::ClassificationDispatchDecoder(bool probabilities_mode)
    : probabilities_mode_(probabilities_mode) {}

std::string ClassificationDispatchDecoder::protocol() const {
  return protocol_name(probabilities_mode_);
}

RawPredictions ClassificationDispatchDecoder::decode(const std::map<std::string, Tensor>&,
                                                     const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, protocol() + " decodes through decode_classification()");
}

ClassificationBatch ClassificationDispatchDecoder::decode_classification(
    const DecoderInput& input, const ClassificationDecodeRequest& request,
    const DeviceContext& device) const {
  if (device.device() == DeviceType::Cpu) {
    if (probabilities_mode_) {
      return ClassificationProbabilitiesDecoder{}.decode_classification(input, request, device);
    }
    return ClassificationDecoder{}.decode_classification(input, request, device);
  }
  if (device.device() == DeviceType::Cuda) {
    return ClassificationCudaDecoder{probabilities_mode_}.decode_classification(input, request, device);
  }
  throw VieError(ErrorCode::InvalidArgument, "decoder device must be cpu or cuda");
}

}  // namespace vie