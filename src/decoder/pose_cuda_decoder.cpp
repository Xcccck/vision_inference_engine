#include "vie/decoder/pose_cuda_decoder.hpp"

#include "vie/decoder/yolov8_pose_decoder.hpp"

#if defined(VIE_HAS_CUDA)
#include "pose_cuda.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vie {
namespace {

constexpr char kProtocol[] = "yolov8_pose_raw";

#if defined(VIE_HAS_CUDA)

const Tensor* find_output(const DecoderInput& input, const std::string& name) {
  const auto found = input.outputs.find(name);
  if (found == input.outputs.end()) return nullptr;
  return &found->second;
}

std::string named(const std::string& output_name, const std::string& detail) {
  return "pose output '" + output_name + "' " + detail;
}

#endif  // VIE_HAS_CUDA

}  // namespace

std::string PoseCudaDecoder::protocol() const { return kProtocol; }

RawPredictions PoseCudaDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_pose_raw decodes through decode_pose()");
}

PoseBatch PoseCudaDecoder::decode_pose(const DecoderInput& input, const PoseDecodeConfig& config,
                                       const DeviceContext& device) const {
#if !defined(VIE_HAS_CUDA)
  (void)input;
  (void)config;
  (void)device;
  throw VieError(ErrorCode::Unsupported, "yolov8_pose_raw CUDA decoder is not available in this build");
#else
  validate_pose_decode_config(config);
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported, "yolov8_pose_raw CUDA decoder does not run on a CPU device");
  }
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "pose geometry is unresolved");
  }
  const Tensor* tensor = find_output(input, config.output_name);
  if (tensor == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "pose output '" + config.output_name + "' is missing");
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
                   "yolov8_pose_raw expects shape [1, channels, anchors] or [1, anchors, channels]");
  }
  const std::vector<std::int64_t>& shape = tensor->shape();
  const std::int64_t channels = shape[static_cast<std::size_t>(config.channel_axis)];
  const std::int64_t predictions = shape[config.channel_axis == 1 ? 2 : 1];
  const std::int64_t expected =
      4 + static_cast<std::int64_t>(config.num_classes) +
      static_cast<std::int64_t>(config.keypoint_count) * static_cast<std::int64_t>(config.keypoint_dimensions);
  if (channels != expected) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_pose_raw channels must be 4 + num_classes + keypoint_count * 3");
  }
  if (predictions < 0) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_pose_raw anchor count is negative");
  }
  if (predictions > 0 && tensor->device_data() == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_pose_raw tensor is empty");
  }

  const PoseCudaLaunch launch =
      launch_pose_cuda(*tensor, config, *input.geometry, device.device_id());
  PoseResult result;
  result.image_width = input.geometry->original_width;
  result.image_height = input.geometry->original_height;
  result.num_classes = config.num_classes;
  result.execution_device = DeviceType::Cuda;
  result.execution_device_id = device.device_id();
  result.protocol = kProtocol;
  result.output_bytes = tensor->nbytes();
  result.d2h_bytes = launch.d2h_bytes;
  result.validate_ms = launch.validate_ms;
  result.pose_decode_ms = launch.pose_decode_ms;
  result.nms_ms = launch.nms_ms;
  result.d2h_ms = launch.d2h_ms;
  // The engine replaces this with the decoder wall clock on CUDA so launch and
  // stream-sync cost stays visible; the component sum is kept for direct calls.
  result.decode_ms = launch.validate_ms + launch.pose_decode_ms + launch.nms_ms;
  result.candidates_after_filter = launch.candidates_after_filter;
  result.candidates_before_nms = launch.candidates_before_nms;
  result.objects = launch.objects;

  PoseBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
#endif
}

std::string PoseDispatchDecoder::protocol() const { return kProtocol; }

RawPredictions PoseDispatchDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_pose_raw decodes through decode_pose()");
}

PoseBatch PoseDispatchDecoder::decode_pose(const DecoderInput& input, const PoseDecodeConfig& config,
                                           const DeviceContext& device) const {
  if (device.device() == DeviceType::Cpu) {
    return YoloV8PoseDecoder{}.decode_pose(input, config, device);
  }
  if (device.device() == DeviceType::Cuda) {
    return PoseCudaDecoder{}.decode_pose(input, config, device);
  }
  throw VieError(ErrorCode::InvalidArgument, "decoder device must be cpu or cuda");
}

}  // namespace vie