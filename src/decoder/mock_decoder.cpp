#include "vie/decoder/mock_decoder.hpp"

#include <cmath>

namespace vie {

RawPredictions MockDecoder::decode(const std::map<std::string, Tensor>& outputs,
                                   const DecoderContext& context) const {
  if (context.protocol != "mock") {
    throw VieError(ErrorCode::InvalidArgument, "MockDecoder only accepts protocol 'mock'");
  }
  if (context.outputs.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "mock decoder context has no outputs");
  }
  const TensorSpec& spec = context.outputs.front();
  const auto found = outputs.find(spec.name);
  if (found == outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "mock decoder is missing output tensor '" + spec.name + "'");
  }
  const Tensor& tensor = found->second;
  if (tensor.device() != DeviceType::Cpu || !tensor.contiguous()) {
    throw VieError(ErrorCode::InvalidArgument, "mock decoder requires a contiguous CPU tensor");
  }
  if (tensor.dtype() != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument, "mock decoder requires an fp32 output");
  }

  RawPredictions predictions;
  predictions.protocol = context.protocol;
  predictions.box_format = context.box_format;
  if (tensor.numel() < 6) return predictions;

  const float* values = tensor.data_as<float>();
  for (int index = 0; index < 6; ++index) {
    if (!std::isfinite(values[index])) {
      throw VieError(ErrorCode::InvalidArgument, "mock decoder output contains a non-finite value");
    }
  }
  const float rounded = std::round(values[5]);
  if (std::fabs(values[5] - rounded) > 1e-4f) {
    throw VieError(ErrorCode::InvalidArgument, "mock decoder class id is not an integer value");
  }
  RawDetection detection;
  detection.box.x1 = values[0];
  detection.box.y1 = values[1];
  detection.box.x2 = values[2];
  detection.box.y2 = values[3];
  detection.score = values[4];
  detection.class_id = static_cast<int>(rounded);
  if (context.num_classes.has_value() &&
      (detection.class_id < 0 || detection.class_id >= *context.num_classes)) {
    throw VieError(ErrorCode::InvalidArgument, "class id is outside num_classes");
  }
  predictions.detections.push_back(detection);
  return predictions;
}

}  // namespace vie
