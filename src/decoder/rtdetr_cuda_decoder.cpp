#include "vie/decoder/rtdetr_cuda_decoder.hpp"

#include "vie/decoder/rtdetr_decoder.hpp"

#if defined(VIE_HAS_CUDA)
#include "rtdetr_cuda.hpp"
#endif

namespace vie {
namespace {

#if defined(VIE_HAS_CUDA)

const Tensor& require_output(const DecoderInput& input, const DecoderConfig& config) {
  const auto found = input.outputs.find(config.output_name);
  if (found == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder is missing output tensor '" + config.output_name + "'");
  }
  return found->second;
}

void require_geometry(const DecoderInput& input) {
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField,
                   "preprocess geometry is unresolved; normalized coordinates are not original-image pixels");
  }
  const GeometryTransform& geometry = *input.geometry;
  if (geometry.output_width <= 0 || geometry.output_height <= 0) {
    throw VieError(ErrorCode::InvalidArgument, "model input width and height must be positive");
  }
  if (geometry.original_width <= 0 || geometry.original_height <= 0 || geometry.scale_x == 0.0 ||
      geometry.scale_y == 0.0) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale or original size is invalid");
  }
}

#endif

}  // namespace

std::string RtdetrCudaDecoder::protocol() const { return "rtdetr_detection_raw"; }

RawPredictions RtdetrCudaDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "rtdetr_detection_raw decodes through decode_batch");
}

DetectionBatch RtdetrCudaDecoder::decode_batch(const DecoderInput& input, const DecoderConfig& config,
                                              const DeviceContext& device) const {
  validate_rtdetr_decoder_config(config);
#if !defined(VIE_HAS_CUDA)
  (void)input;
  (void)device;
  throw VieError(ErrorCode::Unsupported, "CUDA RT-DETR decoder is not available in this build");
#else
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported, "rtdetr CUDA decoder does not run on a CPU device");
  }
  require_geometry(input);
  const Tensor& tensor = require_output(input, config);
  if (tensor.dtype() != DataType::Fp32 || !tensor.is_cuda()) {
    throw VieError(ErrorCode::InvalidArgument, "rtdetr CUDA decoder requires a CUDA fp32 output");
  }
  if (tensor.device_id() != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument, "prediction tensor device does not match the decoder device");
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
  const RtdetrCudaLaunch launch =
      launch_rtdetr_cuda(reinterpret_cast<const float*>(tensor.device_data()), queries, config.num_classes,
                         *config.confidence_threshold, *config.max_detections, device.device_id(),
                         *input.geometry);
  DetectionResult result;
  result.detections = launch.detections;
  result.original_width = input.geometry->original_width;
  result.original_height = input.geometry->original_height;
  result.execution_device = DeviceType::Cuda;
  result.execution_device_id = device.device_id();
  result.protocol = protocol();
  result.nms_applied = false;
  result.coordinates_mapped = true;
  result.candidates_after_filter = launch.candidates_after_filter;
  result.candidates_before_nms = launch.candidates_before_nms;
  result.sort_ms = launch.sort_ms;
  result.d2h_bytes = launch.d2h_bytes;
  result.gpu_execute_ms = launch.kernel_ms;
  result.d2h_ms = launch.d2h_ms;
  DetectionBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
#endif
}

std::string RtdetrDetectionDecoder::protocol() const { return "rtdetr_detection_raw"; }

RawPredictions RtdetrDetectionDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "rtdetr_detection_raw decodes through decode_batch");
}

DetectionBatch RtdetrDetectionDecoder::decode_batch(const DecoderInput& input, const DecoderConfig& config,
                                                   const DeviceContext& device) const {
  if (device.device() == DeviceType::Cpu) {
    return RtdetrDecoder{}.decode_batch(input, config, device);
  }
  if (device.device() == DeviceType::Cuda) {
    return RtdetrCudaDecoder{}.decode_batch(input, config, device);
  }
  throw VieError(ErrorCode::InvalidArgument, "decoder device must be cpu or cuda");
}

}  // namespace vie
