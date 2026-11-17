#include "vie/decoder/yolov8_detect_decoder.hpp"

#include "yolov8_cpu_decode.hpp"

#if defined(VIE_HAS_CUDA)
#include "yolov8_cuda.hpp"
#endif

namespace vie {
namespace {

DetectionResult decode_cpu(const Tensor& tensor, const DecoderConfig& config, const GeometryTransform& geometry) {
  if (tensor.rank() != 3 || tensor.shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 detection expects a rank-3 batch of 1");
  }
  const int channels = static_cast<int>(tensor.shape()[static_cast<std::size_t>(config.channel_axis)]);
  const int other_axis = config.channel_axis == 1 ? 2 : 1;
  const int predictions = static_cast<int>(tensor.shape()[static_cast<std::size_t>(other_axis)]);
  const CpuDetectionPass pass = decode_yolov8_cpu_predictions(tensor.data_as<float>(), config.channel_axis, channels,
                                                             predictions, config, geometry, false);
  DetectionResult result;
  result.original_width = geometry.original_width;
  result.original_height = geometry.original_height;
  result.execution_device = DeviceType::Cpu;
  result.execution_device_id = 0;
  result.protocol = "yolov8_detect_raw";
  result.nms_applied = pass.nms_applied;
  result.coordinates_mapped = true;
  result.candidates_after_filter = pass.candidates_after_filter;
  result.candidates_before_nms = pass.candidates_before_nms;
  result.detections.reserve(pass.kept.size());
  for (const KeptPrediction& candidate : pass.kept) {
    result.detections.push_back(Detection{candidate.original_box, candidate.score, candidate.class_id});
  }
  return result;
}

}  // namespace

std::string YoloV8DetectionDecoder::protocol() const { return "yolov8_detect_raw"; }

RawPredictions YoloV8DetectionDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_detect_raw decodes through decode_batch");
}

DetectionBatch YoloV8DetectionDecoder::decode_batch(const DecoderInput& input, const DecoderConfig& config,
                                                   const DeviceContext& device) const {
  validate_decoder_config(config);
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField,
                   "preprocess geometry is unresolved; model-input coordinates are not used as original-image coordinates");
  }
  const auto found = input.outputs.find(config.output_name);
  if (found == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder is missing output tensor '" + config.output_name + "'");
  }
  const Tensor& tensor = found->second;
  if (tensor.dtype() != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 detection requires an fp32 output");
  }
  if (!tensor.contiguous()) {
    throw VieError(ErrorCode::Unsupported, "YOLOv8 detection does not read a non-contiguous tensor");
  }
  if (tensor.device() != device.device() || tensor.device_id() != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument, "prediction tensor device does not match the decoder device");
  }
  DetectionResult result;
  if (device.device() == DeviceType::Cpu) {
    if (!tensor.is_cpu()) {
      throw VieError(ErrorCode::InvalidArgument, "CPU decoder cannot read a CUDA tensor");
    }
    result = decode_cpu(tensor, config, *input.geometry);
  } else if (device.device() == DeviceType::Cuda) {
#if defined(VIE_HAS_CUDA)
    result = decode_yolov8_cuda(tensor, config, *input.geometry, device);
#else
    throw VieError(ErrorCode::Unsupported, "CUDA YOLOv8 decoder is not available in this build");
#endif
  } else {
    throw VieError(ErrorCode::InvalidArgument, "decoder device must be cpu or cuda");
  }
  DetectionBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
}

}  // namespace vie
