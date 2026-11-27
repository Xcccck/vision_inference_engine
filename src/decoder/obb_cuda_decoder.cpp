#include "vie/decoder/obb_cuda_decoder.hpp"

#include "vie/decoder/yolov8_obb_decoder.hpp"

#if defined(VIE_HAS_CUDA)
#include "obb_cuda.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vie {
namespace {

constexpr char kProtocol[] = "yolov8_obb_raw";

#if defined(VIE_HAS_CUDA)

const Tensor* find_output(const DecoderInput& input, const std::string& name) {
  const auto found = input.outputs.find(name);
  if (found == input.outputs.end()) return nullptr;
  return &found->second;
}

std::string named(const std::string& output_name, const std::string& detail) {
  return "obb output '" + output_name + "' " + detail;
}

#endif  // VIE_HAS_CUDA

}  // namespace

std::string ObbCudaDecoder::protocol() const { return kProtocol; }

RawPredictions ObbCudaDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_obb_raw decodes through decode_obb()");
}

ObbBatch ObbCudaDecoder::decode_obb(const DecoderInput& input, const ObbDecodeConfig& config,
                                    const DeviceContext& device) const {
#if !defined(VIE_HAS_CUDA)
  (void)input;
  (void)config;
  (void)device;
  throw VieError(ErrorCode::Unsupported, "yolov8_obb_raw CUDA decoder is not available in this build");
#else
  validate_obb_decode_config(config);
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported, "yolov8_obb_raw CUDA decoder does not run on a CPU device");
  }
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "obb geometry is unresolved");
  }
  const Tensor* tensor = find_output(input, config.output_name);
  if (tensor == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "obb output '" + config.output_name + "' is missing");
  }
  if (!tensor->is_cuda()) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(config.output_name, "must be a CUDA tensor for the CUDA decoder"));
  }
  if (tensor->device_id() != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument,
                   named(config.output_name, "device does not match the decoder device"));
  }
  if (tensor->dtype() != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument, named(config.output_name, "must be fp32"));
  }
  if (!tensor->contiguous()) {
    throw VieError(ErrorCode::InvalidArgument, named(config.output_name, "must be contiguous"));
  }
  if (tensor->shape().size() != 3 || tensor->shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_obb_raw expects shape [1, channels, anchors] or [1, anchors, channels]");
  }
  const std::vector<std::int64_t>& shape = tensor->shape();
  const std::int64_t channels = shape[static_cast<std::size_t>(config.channel_axis)];
  const std::int64_t predictions = shape[config.channel_axis == 1 ? 2 : 1];
  const std::int64_t expected = 4 + static_cast<std::int64_t>(config.num_classes) + 1;
  if (channels != expected || predictions < 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_obb_raw channels must be 4 + num_classes + 1 and the anchor count must be positive");
  }
  if (tensor->device_data() == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_obb_raw tensor is empty");
  }

  const ObbCudaLaunch launch = launch_obb_cuda(*tensor, config, *input.geometry, device.device_id());
  ObbResult result;
  result.image_width = input.geometry->original_width;
  result.image_height = input.geometry->original_height;
  result.num_classes = config.num_classes;
  result.execution_device = DeviceType::Cuda;
  result.execution_device_id = device.device_id();
  result.protocol = kProtocol;
  result.output_bytes = tensor->nbytes();
  result.d2h_bytes = launch.d2h_bytes;
  result.validate_ms = launch.validate_ms;
  result.sort_ms = launch.sort_ms;
  result.map_ms = launch.map_ms;
  result.nms_ms = launch.nms_ms;
  result.d2h_ms = launch.d2h_ms;
  // The engine replaces this with the decoder wall clock on CUDA so launch and
  // stream-sync cost stays visible; the component sum is kept for direct calls.
  result.decode_ms = launch.validate_ms + launch.sort_ms + launch.map_ms + launch.nms_ms;
  result.candidates_after_filter = launch.candidates_after_filter;
  result.candidates_before_nms = launch.candidates_before_nms;
  result.iou_calls = launch.iou_calls;
  result.objects = launch.objects;

  ObbBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
#endif
}

std::string ObbDispatchDecoder::protocol() const { return kProtocol; }

RawPredictions ObbDispatchDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_obb_raw decodes through decode_obb()");
}

ObbBatch ObbDispatchDecoder::decode_obb(const DecoderInput& input, const ObbDecodeConfig& config,
                                        const DeviceContext& device) const {
  if (device.device() == DeviceType::Cpu) {
    return YoloV8ObbDecoder{}.decode_obb(input, config, device);
  }
  if (device.device() == DeviceType::Cuda) {
    return ObbCudaDecoder{}.decode_obb(input, config, device);
  }
  throw VieError(ErrorCode::InvalidArgument, "decoder device must be cpu or cuda");
}

}  // namespace vie