#include "vie/decoder/semantic_segmentation_cuda_decoder.hpp"

#include "vie/decoder/semantic_segmentation.hpp"

#if defined(VIE_HAS_CUDA)
#include "semantic_segmentation_cuda.hpp"
#endif

#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace vie {
namespace {

#if defined(VIE_HAS_CUDA)

const Tensor* find_output(const DecoderInput& input, const std::string& name) {
  const auto found = input.outputs.find(name);
  if (found == input.outputs.end()) return nullptr;
  return &found->second;
}

// Same request checks and wording as SemanticSegmentationDecoder. Geometry is
// validated by the CPU decoder before touching the tensor; the CUDA decoder
// keeps that order so both paths fail on the same defect first.
void validate_request(const SemanticSegmentationDecodeRequest& request, const DecoderInput& input) {
  if (request.output_name.empty()) {
    throw VieError(ErrorCode::UnresolvedField, "semantic_segmentation.output_name is unresolved");
  }
  if (request.logits_semantics.empty()) {
    throw VieError(ErrorCode::UnresolvedField, "semantic_segmentation.logits_semantics is unresolved");
  }
  if (request.logits_semantics != "per_pixel_class_logits") {
    throw VieError(ErrorCode::Unsupported,
                   "deeplabv3_semantic_seg_raw only supports logits_semantics per_pixel_class_logits");
  }
  if (!request.output_interpolation.empty() && request.output_interpolation != "bilinear") {
    throw VieError(ErrorCode::Unsupported, "semantic segmentation logits resize must be bilinear");
  }
  if (!request.class_id_interpolation.empty() && request.class_id_interpolation != "nearest") {
    throw VieError(ErrorCode::Unsupported, "semantic segmentation class-id resize must be nearest");
  }
  if (request.channel_axis != 1) {
    throw VieError(ErrorCode::Unsupported, "semantic segmentation expects channel_axis 1 (NCHW)");
  }
  if (request.num_classes < 2) {
    throw VieError(ErrorCode::InvalidArgument, "semantic segmentation num_classes must be >= 2");
  }
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation requires a GeometryTransform to restore original coordinates");
  }
  const GeometryTransform& geometry = *input.geometry;
  if (geometry.original_width < 1 || geometry.original_height < 1) {
    throw VieError(ErrorCode::InvalidArgument, "geometry original size must be positive");
  }
  if (geometry.output_width < 1 || geometry.output_height < 1) {
    throw VieError(ErrorCode::InvalidArgument, "geometry model-input size must be positive");
  }
  if (geometry.scale_x <= 0.0 || geometry.scale_y <= 0.0) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale must be positive");
  }
}

#endif  // VIE_HAS_CUDA

}  // namespace

std::string SemanticSegmentationCudaDecoder::protocol() const { return "deeplabv3_semantic_seg_raw"; }

RawPredictions SemanticSegmentationCudaDecoder::decode(const std::map<std::string, Tensor>&,
                                                       const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported,
                 "deeplabv3_semantic_seg_raw decodes through decode_semantic_segmentation()");
}

SemanticSegmentationBatch SemanticSegmentationCudaDecoder::decode_semantic_segmentation(
    const DecoderInput& input, const SemanticSegmentationDecodeRequest& request,
    const DeviceContext& device) const {
#if !defined(VIE_HAS_CUDA)
  (void)input;
  (void)request;
  (void)device;
  throw VieError(ErrorCode::Unsupported,
                 "deeplabv3_semantic_seg_raw CUDA decoder is not available in this build");
#else
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported,
                   "deeplabv3_semantic_seg_raw CUDA decoder does not run on a CPU device");
  }
  validate_request(request, input);
  const GeometryTransform& geometry = *input.geometry;

  const Tensor* tensor = find_output(input, request.output_name);
  if (tensor == nullptr) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation output '" + request.output_name + "' is missing");
  }
  if (!tensor->is_cuda()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation output '" + request.output_name +
                       "' must be a CUDA tensor for the CUDA decoder");
  }
  if (tensor->device_id() != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation output '" + request.output_name +
                       "' device does not match the decoder device");
  }
  if (tensor->dtype() != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation output '" + request.output_name + "' must be fp32");
  }
  if (!tensor->contiguous()) {
    throw VieError(ErrorCode::Unsupported,
                   "deeplabv3_semantic_seg_raw does not read a non-contiguous tensor");
  }
  const std::vector<std::int64_t>& shape = tensor->shape();
  if (shape.size() != 4) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation output '" + request.output_name +
                       "' must be rank 4 [N, C, H, W]");
  }
  if (shape[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation batch dimension must be 1 in this phase");
  }
  if (shape[1] != request.num_classes) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation class axis does not match num_classes");
  }
  if (shape[2] < 1 || shape[3] < 1) {
    throw VieError(ErrorCode::InvalidArgument, "semantic segmentation logits spatial size must be positive");
  }
  const std::size_t out_count = static_cast<std::size_t>(geometry.original_width) *
                                static_cast<std::size_t>(geometry.original_height);
  if (geometry.original_width > 0 &&
      out_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw VieError(ErrorCode::InvalidArgument, "semantic segmentation output size overflows");
  }

  const int channels = request.num_classes;
  const int logits_height = static_cast<int>(shape[2]);
  const int logits_width = static_cast<int>(shape[3]);
  const SemanticSegmentationCudaLaunch launch = launch_semantic_segmentation_cuda(
      reinterpret_cast<const float*>(tensor->device_data()), channels, logits_height, logits_width,
      geometry, request.output_name, device.device_id());

  SemanticSegmentationResult result;
  result.protocol = protocol();
  result.output_name = request.output_name;
  result.num_classes = channels;
  result.width = geometry.original_width;
  result.height = geometry.original_height;
  result.original_width = geometry.original_width;
  result.original_height = geometry.original_height;
  result.logits_width = logits_width;
  result.logits_height = logits_height;
  result.execution_device = DeviceType::Cuda;
  result.execution_device_id = device.device_id();
  result.output_bytes = tensor->nbytes();
  result.d2h_bytes = launch.d2h_bytes;
  result.class_ids = launch.class_ids;
  result.validate_ms = launch.validate_ms;
  result.argmax_ms = launch.argmax_ms;
  result.d2h_ms = launch.d2h_ms;

  SemanticSegmentationBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
#endif
}

std::string SemanticSegmentationDispatchDecoder::protocol() const {
  return "deeplabv3_semantic_seg_raw";
}

RawPredictions SemanticSegmentationDispatchDecoder::decode(const std::map<std::string, Tensor>&,
                                                           const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported,
                 "deeplabv3_semantic_seg_raw decodes through decode_semantic_segmentation()");
}

SemanticSegmentationBatch SemanticSegmentationDispatchDecoder::decode_semantic_segmentation(
    const DecoderInput& input, const SemanticSegmentationDecodeRequest& request,
    const DeviceContext& device) const {
  if (device.device() == DeviceType::Cpu) {
    return SemanticSegmentationDecoder{}.decode_semantic_segmentation(input, request, device);
  }
  if (device.device() == DeviceType::Cuda) {
    return SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(input, request, device);
  }
  throw VieError(ErrorCode::InvalidArgument, "decoder device must be cpu or cuda");
}

}  // namespace vie