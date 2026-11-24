#include "vie/decoder/yolov8_detect_decoder.hpp"

#include "yolov8_cpu_decode.hpp"

#include <vector>

#if defined(VIE_HAS_CUDA)
#include "yolov8_cuda.hpp"
#endif

namespace vie {
namespace {

// Phase 9.1: one per-image decode. `data` points to the start of the batch
// sample inside a contiguous [N, C, P] tensor; `channels` and `predictions`
// describe the per-sample slice. The caller advances `data` between samples
// so nothing crosses batch boundaries during filtering, sorting, or NMS.
DetectionResult decode_one_cpu_sample(const float* data, int channel_axis, int channels,
                                     int predictions, const DecoderConfig& config,
                                     const GeometryTransform& geometry) {
  const CpuDetectionPass pass =
      decode_yolov8_cpu_predictions(data, channel_axis, channels, predictions, config, geometry, false);
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

// Phase 9.1: Batch tensor is rank-3 [N, C, P] (or [N, P, C] when the channel
// axis is 2). Sample i occupies elements [i * C * P, (i + 1) * C * P) in
// row-major memory, which is exactly what the current single-image path
// already assumes for N=1. Iterate per sample so filtering, sort, top-k,
// GeometryTransform mapping, and NMS remain isolated.
DetectionBatch decode_cpu_batch(const Tensor& tensor, const DecoderConfig& config,
                                const std::vector<GeometryTransform>& geometries) {
  if (tensor.rank() != 3) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 detection expects a rank-3 tensor [N, C, P]");
  }
  const std::int64_t batch = tensor.shape()[0];
  if (batch <= 0) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 detection requires a positive batch dimension");
  }
  if (static_cast<std::size_t>(batch) != geometries.size()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "YOLOv8 detection batch mismatch: tensor batch " + std::to_string(batch) +
                       " but " + std::to_string(geometries.size()) + " GeometryTransforms supplied");
  }
  const int channels = static_cast<int>(tensor.shape()[static_cast<std::size_t>(config.channel_axis)]);
  const int other_axis = config.channel_axis == 1 ? 2 : 1;
  const int predictions = static_cast<int>(tensor.shape()[static_cast<std::size_t>(other_axis)]);
  const std::size_t per_sample = static_cast<std::size_t>(channels) * static_cast<std::size_t>(predictions);
  if (per_sample == 0) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 detection has zero-length per-sample output");
  }
  const float* base = tensor.data_as<float>();
  DetectionBatch batch_out;
  batch_out.results.reserve(static_cast<std::size_t>(batch));
  for (std::int64_t i = 0; i < batch; ++i) {
    batch_out.results.push_back(decode_one_cpu_sample(base + i * static_cast<std::ptrdiff_t>(per_sample),
                                                     config.channel_axis, channels, predictions, config,
                                                     geometries[static_cast<std::size_t>(i)]));
  }
  return batch_out;
}

}  // namespace

std::string YoloV8DetectionDecoder::protocol() const { return "yolov8_detect_raw"; }

RawPredictions YoloV8DetectionDecoder::decode(const std::map<std::string, Tensor>&, const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_detect_raw decodes through decode_batch");
}

DetectionBatch YoloV8DetectionDecoder::decode_batch(const DecoderInput& input, const DecoderConfig& config,
                                                   const DeviceContext& device) const {
  validate_decoder_config(config);
  // Resolve the per-sample GeometryTransform list. The batch field takes
  // precedence; when only the single-image field is set, wrap it as size 1
  // so the tensor batch check below is identical for both callers.
  std::vector<GeometryTransform> geometries = input.geometries;
  if (geometries.empty()) {
    if (!input.geometry.has_value()) {
      throw VieError(ErrorCode::UnresolvedField,
                     "preprocess geometry is unresolved; model-input coordinates are not used as "
                     "original-image coordinates");
    }
    geometries.push_back(*input.geometry);
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
  if (device.device() == DeviceType::Cpu) {
    if (!tensor.is_cpu()) {
      throw VieError(ErrorCode::InvalidArgument, "CPU decoder cannot read a CUDA tensor");
    }
    return decode_cpu_batch(tensor, config, geometries);
  }
  if (device.device() == DeviceType::Cuda) {
#if defined(VIE_HAS_CUDA)
    // Phase 9.1: the CUDA kernel still decodes one image per launch. Batch>1
    // through CUDA is not implemented in this phase, so reject rather than
    // silently looping over decode_yolov8_cuda and mis-reporting it as a
    // true GPU batch.
    if (tensor.rank() == 3 && tensor.shape()[0] > 1) {
      throw VieError(ErrorCode::Unsupported,
                     "CUDA YOLOv8 detection batch > 1 is not implemented in Phase 9.1");
    }
    DetectionResult result = decode_yolov8_cuda(tensor, config, geometries.front(), device);
    DetectionBatch batch_out;
    batch_out.results.push_back(std::move(result));
    return batch_out;
#else
    throw VieError(ErrorCode::Unsupported, "CUDA YOLOv8 decoder is not available in this build");
#endif
  }
  throw VieError(ErrorCode::InvalidArgument, "decoder device must be cpu or cuda");
}

}  // namespace vie
