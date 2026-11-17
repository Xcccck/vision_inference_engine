#include "vie/decoder/yolov8_seg_decoder.hpp"

#include "yolov8_cpu_decode.hpp"

#if defined(VIE_HAS_CUDA)
#include "yolov8_seg_cuda.hpp"
#endif

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace vie {
namespace {

void require_set(bool ready, const std::string& message) {
  if (!ready) throw VieError(ErrorCode::UnresolvedField, message);
}

std::size_t checked_mask_bytes(std::size_t instances, int width, int height) {
  if (width < 1 || height < 1) {
    throw VieError(ErrorCode::InvalidArgument, "mask output size is invalid");
  }
  const std::size_t wide = static_cast<std::size_t>(width);
  const std::size_t high = static_cast<std::size_t>(height);
  if (wide > std::numeric_limits<std::size_t>::max() / high) {
    throw VieError(ErrorCode::InvalidArgument, "mask width times height overflows");
  }
  const std::size_t pixels = wide * high;
  if (instances > std::numeric_limits<std::size_t>::max() / pixels) {
    throw VieError(ErrorCode::InvalidArgument, "mask allocation size overflows");
  }
  return instances * pixels;
}

float read_prediction(const float* data, int channel_axis, int channels, int predictions, int channel,
                      int prediction) {
  const std::size_t index = channel_axis == 1
                                ? static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) +
                                      static_cast<std::size_t>(prediction)
                                : static_cast<std::size_t>(prediction) * static_cast<std::size_t>(channels) +
                                      static_cast<std::size_t>(channel);
  const float value = data[index];
  if (!std::isfinite(value)) {
    throw VieError(ErrorCode::InvalidArgument, "mask coefficient is not finite");
  }
  return value;
}

float sample_bilinear(const float* source, int width, int height, double x, double y) {
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const float wx = static_cast<float>(x - static_cast<double>(x0));
  const float wy = static_cast<float>(y - static_cast<double>(y0));
  const auto at = [&](int px, int py) {
    if (px < 0) px = 0;
    if (py < 0) py = 0;
    if (px >= width) px = width - 1;
    if (py >= height) py = height - 1;
    return source[static_cast<std::size_t>(py) * static_cast<std::size_t>(width) + static_cast<std::size_t>(px)];
  };
  const float top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * wx;
  const float bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * wx;
  return top + (bottom - top) * wy;
}

bool covers_pixel(float x1, float y1, float x2, float y2, int x, int y) {
  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);
  return fx >= x1 && fx < x2 && fy >= y1 && fy < y2;
}

void zero_outside(std::vector<float>& image, int width, int height, const BoundingBox& box) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (!covers_pixel(box.x1, box.y1, box.x2, box.y2, x, y)) {
        image[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] = 0.f;
      }
    }
  }
}

}  // namespace

std::string YoloV8SegmentationDecoder::protocol() const { return "yolov8_seg_raw"; }

RawPredictions YoloV8SegmentationDecoder::decode(const std::map<std::string, Tensor>&,
                                                 const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw decodes through decode_segmentation");
}

DetectionBatch YoloV8SegmentationDecoder::decode_batch(const DecoderInput&, const DecoderConfig&,
                                                      const DeviceContext&) const {
  throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw does not produce DetectionBatch");
}

SegmentationBatch YoloV8SegmentationDecoder::decode_segmentation(const DecoderInput& input,
                                                                const DecoderConfig& config,
                                                                const DeviceContext& device) const {
  return decode_timed(input, config, device, nullptr);
}

SegmentationBatch YoloV8SegmentationDecoder::decode_timed(const DecoderInput& input, const DecoderConfig& config,
                                                         const DeviceContext& device,
                                                         SegmentationTiming* timing) const {
  const auto total_start = std::chrono::steady_clock::now();
  if (device.device() == DeviceType::Cuda) {
#if defined(VIE_HAS_CUDA)
    return decode_cuda_timed(input, config, device, nullptr);
#else
    throw VieError(ErrorCode::Unsupported, "CUDA YOLOv8 segmentation decoder is not available");
#endif
  }
  if (device.device() != DeviceType::Cpu) {
    throw VieError(ErrorCode::InvalidArgument, "segmentation decoder device must be cpu or cuda");
  }
  validate_decoder_config(config);
  require_set(!config.prototype_name.empty(), "decoder.prototype_name is unresolved");
  require_set(config.mask_channels >= 1, "decoder.mask_channels is unresolved");
  require_set(config.prototype_channel_axis.has_value(), "decoder.prototype_channel_axis is unresolved");
  require_set(config.retina_masks.has_value(), "decoder.retina_masks is unresolved");
  require_set(config.binary_mask_gt_zero.has_value(), "decoder.binary_mask_gt_zero is unresolved");
  require_set(config.max_mask_bytes.has_value(), "decoder.max_mask_bytes is unresolved");
  if (config.box_decode != BoxDecode::DecodedCxcywh || config.box_channels != 4) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_seg_raw requires decoded cxcywh with 4 box channels");
  }
  if (!*config.scores_are_probabilities) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_seg_raw class scores are already probabilities; refusing a second sigmoid");
  }
  if (*config.prototype_channel_axis != 1) {
    throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw only reads an NCHW prototype tensor");
  }
  if (!*config.binary_mask_gt_zero) {
    throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw only emits a binary mask from the M > 0 test");
  }
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField,
                   "preprocess geometry is unresolved; model-input coordinates are not used as original-image coordinates");
  }
  const GeometryTransform& geometry = *input.geometry;
  if (geometry.output_width < 1 || geometry.output_height < 1 || geometry.original_width < 1 ||
      geometry.original_height < 1 || geometry.scale_x == 0.0 || geometry.scale_y == 0.0) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale or image size is invalid");
  }
  const auto predictions = input.outputs.find(config.output_name);
  if (predictions == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder is missing output tensor '" + config.output_name + "'");
  }
  const auto prototypes = input.outputs.find(config.prototype_name);
  if (prototypes == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "decoder is missing prototype tensor '" + config.prototype_name + "'");
  }
  const Tensor& prediction_tensor = predictions->second;
  const Tensor& prototype_tensor = prototypes->second;
  if (prediction_tensor.dtype() != DataType::Fp32 || prototype_tensor.dtype() != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 segmentation requires fp32 tensors");
  }
  if (!prediction_tensor.contiguous() || !prototype_tensor.contiguous()) {
    throw VieError(ErrorCode::Unsupported, "YOLOv8 segmentation does not read a non-contiguous tensor");
  }
  if (!prediction_tensor.is_cpu() || !prototype_tensor.is_cpu() || prediction_tensor.device_id() != device.device_id() ||
      prototype_tensor.device_id() != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument, "CPU segmentation decoder cannot read a CUDA tensor");
  }
  if (prediction_tensor.rank() != 3 || prediction_tensor.shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 segmentation expects a rank-3 prediction batch of 1");
  }
  if (prototype_tensor.rank() != 4 || prototype_tensor.shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 segmentation expects a rank-4 prototype batch of 1");
  }
  const int channels = static_cast<int>(prediction_tensor.shape()[static_cast<std::size_t>(config.channel_axis)]);
  const int prediction_axis = config.channel_axis == 1 ? 2 : 1;
  const int prediction_count = static_cast<int>(prediction_tensor.shape()[static_cast<std::size_t>(prediction_axis)]);
  if (channels != 4 + config.num_classes + config.mask_channels) {
    throw VieError(ErrorCode::InvalidArgument,
                   "prediction channels do not match 4 + classes + mask channels");
  }
  if (prototype_tensor.shape()[1] != config.mask_channels) {
    throw VieError(ErrorCode::InvalidArgument, "prototype channels do not match mask channels");
  }
  const int proto_h = static_cast<int>(prototype_tensor.shape()[2]);
  const int proto_w = static_cast<int>(prototype_tensor.shape()[3]);
  if (proto_h < 1 || proto_w < 1) {
    throw VieError(ErrorCode::InvalidArgument, "prototype spatial size is invalid");
  }
  const bool retina = *config.retina_masks;
  const int mask_w = retina ? geometry.original_width : geometry.output_width;
  const int mask_h = retina ? geometry.original_height : geometry.output_height;

  CpuDetectionPass detected = decode_yolov8_cpu_predictions(
      prediction_tensor.data_as<float>(), config.channel_axis, channels, prediction_count, config, geometry,
      timing != nullptr);
  const std::size_t needed = checked_mask_bytes(detected.kept.size(), mask_w, mask_h);
  if (needed > *config.max_mask_bytes) {
    throw VieError(ErrorCode::InvalidArgument, "mask byte budget exceeded: need " + std::to_string(needed) +
                                                   ", budget " + std::to_string(*config.max_mask_bytes));
  }

  const float* proto = prototype_tensor.data_as<float>();
  const std::size_t proto_pixels = static_cast<std::size_t>(proto_h) * static_cast<std::size_t>(proto_w);
  if (!detected.kept.empty()) {
    for (std::size_t i = 0; i < proto_pixels * static_cast<std::size_t>(config.mask_channels); ++i) {
      if (!std::isfinite(proto[i])) {
        throw VieError(ErrorCode::InvalidArgument, "prototype tensor contains a non-finite value");
      }
    }
  }
  std::vector<float> logits;
  std::vector<float> scaled;
  std::vector<std::uint8_t> binary;
  std::vector<float> coefficients;
  if (!detected.kept.empty()) {
    logits.resize(proto_pixels);
    scaled.resize(static_cast<std::size_t>(mask_w) * static_cast<std::size_t>(mask_h));
    binary.resize(scaled.size());
    coefficients.resize(static_cast<std::size_t>(config.mask_channels));
  }
  const double proto_per_model_x = static_cast<double>(proto_w) / static_cast<double>(geometry.output_width);
  const double proto_per_model_y = static_cast<double>(proto_h) / static_cast<double>(geometry.output_height);
  double gather_ms = 0;
  double reconstruct_ms = 0;
  double crop_ms = 0;
  double resize_ms = 0;
  SegmentationBatch batch;
  batch.original_width = geometry.original_width;
  batch.original_height = geometry.original_height;
  batch.execution_device = DeviceType::Cpu;
  batch.execution_device_id = device.device_id();
  batch.protocol = "yolov8_seg_raw";
  batch.instances.reserve(detected.kept.size());
  for (const KeptPrediction& kept : detected.kept) {
    const auto gather_start = std::chrono::steady_clock::now();
    for (int channel = 0; channel < config.mask_channels; ++channel) {
      coefficients[static_cast<std::size_t>(channel)] = read_prediction(
          prediction_tensor.data_as<float>(), config.channel_axis, channels, prediction_count,
          config.box_channels + config.num_classes + channel, kept.prediction_index);
    }
    gather_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gather_start).count();

    const auto reconstruct_start = std::chrono::steady_clock::now();
    for (int y = 0; y < proto_h; ++y) {
      for (int x = 0; x < proto_w; ++x) {
        float sum = 0.f;
        const std::size_t spatial = static_cast<std::size_t>(y) * static_cast<std::size_t>(proto_w) +
                                    static_cast<std::size_t>(x);
        for (int channel = 0; channel < config.mask_channels; ++channel) {
          const float value = proto[static_cast<std::size_t>(channel) * proto_pixels + spatial];
          sum += coefficients[static_cast<std::size_t>(channel)] * value;
        }
        if (!std::isfinite(sum)) {
          throw VieError(ErrorCode::InvalidArgument, "mask reconstruction is not finite");
        }
        logits[spatial] = sum;
      }
    }
    reconstruct_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - reconstruct_start).count();

    BoundingBox proto_box;
    proto_box.x1 = static_cast<float>(static_cast<double>(kept.model_box.x1) * proto_per_model_x);
    proto_box.y1 = static_cast<float>(static_cast<double>(kept.model_box.y1) * proto_per_model_y);
    proto_box.x2 = static_cast<float>(static_cast<double>(kept.model_box.x2) * proto_per_model_x);
    proto_box.y2 = static_cast<float>(static_cast<double>(kept.model_box.y2) * proto_per_model_y);
    if (!retina) {
      const auto crop_start = std::chrono::steady_clock::now();
      zero_outside(logits, proto_w, proto_h, proto_box);
      crop_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - crop_start).count();
      const auto resize_start = std::chrono::steady_clock::now();
      for (int y = 0; y < mask_h; ++y) {
        const double src_y = (static_cast<double>(y) + 0.5) * proto_per_model_y - 0.5;
        for (int x = 0; x < mask_w; ++x) {
          const double src_x = (static_cast<double>(x) + 0.5) * proto_per_model_x - 0.5;
          const float value = sample_bilinear(logits.data(), proto_w, proto_h, src_x, src_y);
          if (!std::isfinite(value)) {
            throw VieError(ErrorCode::InvalidArgument, "mask resize is not finite");
          }
          scaled[static_cast<std::size_t>(y) * static_cast<std::size_t>(mask_w) + static_cast<std::size_t>(x)] = value;
        }
      }
      resize_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - resize_start).count();
    } else {
      const auto resize_start = std::chrono::steady_clock::now();
      for (int y = 0; y < mask_h; ++y) {
        const double src_y = (static_cast<double>(geometry.pad_top) +
                              (static_cast<double>(y) + 0.5) * geometry.scale_y) *
                                 proto_per_model_y -
                             0.5;
        for (int x = 0; x < mask_w; ++x) {
          const double src_x = (static_cast<double>(geometry.pad_left) +
                                (static_cast<double>(x) + 0.5) * geometry.scale_x) *
                                   proto_per_model_x -
                               0.5;
          const float value = sample_bilinear(logits.data(), proto_w, proto_h, src_x, src_y);
          if (!std::isfinite(value)) {
            throw VieError(ErrorCode::InvalidArgument, "mask resize is not finite");
          }
          scaled[static_cast<std::size_t>(y) * static_cast<std::size_t>(mask_w) + static_cast<std::size_t>(x)] = value;
        }
      }
      resize_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - resize_start).count();
      const auto crop_start = std::chrono::steady_clock::now();
      zero_outside(scaled, mask_w, mask_h, kept.original_box);
      crop_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - crop_start).count();
    }
    for (std::size_t i = 0; i < binary.size(); ++i) binary[i] = scaled[i] > 0.f ? 1 : 0;
    InstanceSegmentation instance;
    instance.detection.box = kept.original_box;
    instance.detection.confidence = kept.score;
    instance.detection.class_id = kept.class_id;
    instance.mask = Mask::binary(DeviceContext{}, mask_w, mask_h, binary.data(), binary.size());
    batch.instances.push_back(std::move(instance));
  }
  if (timing != nullptr) {
    timing->instances = static_cast<int>(batch.instances.size());
    timing->mask_width = detected.kept.empty() ? 0 : mask_w;
    timing->mask_height = detected.kept.empty() ? 0 : mask_h;
    timing->detection_ms = detected.box_ms;
    timing->filter_ms = detected.filter_ms;
    timing->nms_ms = detected.nms_ms;
    timing->gather_ms = gather_ms;
    timing->reconstruct_ms = reconstruct_ms;
    timing->crop_ms = crop_ms;
    timing->resize_ms = resize_ms;
    timing->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
    timing->includes_allocation = true;
  }
  return batch;
}

SegmentationBatch YoloV8SegmentationDecoder::decode_cuda_timed(const DecoderInput& input, const DecoderConfig& config,
                                                             const DeviceContext& device,
                                                             CudaSegmentationTiming* timing) const {
#if defined(VIE_HAS_CUDA)
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::InvalidArgument, "CUDA segmentation decoder requires a CUDA device");
  }
  return decode_yolov8_seg_cuda(input, config, device, timing);
#else
  (void)input;
  (void)config;
  (void)device;
  (void)timing;
  throw VieError(ErrorCode::Unsupported, "CUDA YOLOv8 segmentation decoder is not available");
#endif
}

}  // namespace vie
