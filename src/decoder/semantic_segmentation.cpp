#include "vie/decoder/semantic_segmentation.hpp"

#include "vie/core/error.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace vie {
namespace {

const Tensor* find_output(const DecoderInput& input, const std::string& name) {
  const auto found = input.outputs.find(name);
  if (found == input.outputs.end()) return nullptr;
  return &found->second;
}

// Bilinear sample of a single-channel [height, width] plane laid out row-major,
// at continuous coordinate (x, y). Coordinates are clamped to the plane, so the
// caller never reads out of bounds. This is the same half-pixel convention the
// preprocessor documents: sampling a plane at index i maps to source (i+0.5)*
// (src/dst)-0.5, done by the caller before reaching this function.
inline float sample_plane(const float* plane, int height, int width, double x, double y) {
  if (x < 0) x = 0;
  if (x > width - 1) x = width - 1;
  if (y < 0) y = 0;
  if (y > height - 1) y = height - 1;
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = x0 + 1 < width ? x0 + 1 : x0;
  const int y1 = y0 + 1 < height ? y0 + 1 : y0;
  const double fx = x - x0;
  const double fy = y - y0;
  const float v00 = plane[static_cast<std::size_t>(y0) * width + x0];
  const float v10 = plane[static_cast<std::size_t>(y0) * width + x1];
  const float v01 = plane[static_cast<std::size_t>(y1) * width + x0];
  const float v11 = plane[static_cast<std::size_t>(y1) * width + x1];
  const double top = v00 * (1.0 - fx) + v10 * fx;
  const double bottom = v01 * (1.0 - fx) + v11 * fx;
  return static_cast<float>(top * (1.0 - fy) + bottom * fy);
}

// Bilinear-resize a [channels, src_h, src_w] logits volume to
// [channels, dst_h, dst_w] using the half-pixel convention. Continuous values
// (logits) only. Returns a row-major contiguous buffer.
std::vector<float> resize_logits(const float* src, int channels, int src_h, int src_w, int dst_h,
                                 int dst_w) {
  std::vector<float> dst(static_cast<std::size_t>(channels) * dst_h * dst_w);
  const double sx = src_w > 0 ? static_cast<double>(src_w) / dst_w : 1.0;
  const double sy = src_h > 0 ? static_cast<double>(src_h) / dst_h : 1.0;
  for (int c = 0; c < channels; ++c) {
    const float* plane = src + static_cast<std::size_t>(c) * src_h * src_w;
    float* out = dst.data() + static_cast<std::size_t>(c) * dst_h * dst_w;
    for (int y = 0; y < dst_h; ++y) {
      const double source_y = (y + 0.5) * sy - 0.5;
      for (int x = 0; x < dst_w; ++x) {
        const double source_x = (x + 0.5) * sx - 0.5;
        out[static_cast<std::size_t>(y) * dst_w + x] = sample_plane(plane, src_h, src_w, source_x, source_y);
      }
    }
  }
  return dst;
}

}  // namespace

std::string SemanticSegmentationDecoder::protocol() const { return "deeplabv3_semantic_seg_raw"; }

RawPredictions SemanticSegmentationDecoder::decode(const std::map<std::string, Tensor>&,
                                                   const DecoderContext&) const {
  throw VieError(ErrorCode::Unsupported,
                 "deeplabv3_semantic_seg_raw decodes through decode_semantic_segmentation()");
}

SemanticSegmentationBatch SemanticSegmentationDecoder::decode_semantic_segmentation(
    const DecoderInput& input, const SemanticSegmentationDecodeRequest& request,
    const DeviceContext& device) const {
  if (device.device() != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "deeplabv3_semantic_seg_raw runs on CPU");
  }
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

  const Tensor* tensor = find_output(input, request.output_name);
  if (tensor == nullptr) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation output '" + request.output_name + "' is missing");
  }
  if (tensor->device() != DeviceType::Cpu || tensor->is_cuda()) {
    throw VieError(ErrorCode::Unsupported,
                   "deeplabv3_semantic_seg_raw does not read a CUDA tensor (engine must D2H first)");
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
                   "semantic segmentation output '" + request.output_name + "' must be rank 4 [N, C, H, W]");
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

  const int channels = request.num_classes;
  const int logits_height = static_cast<int>(shape[2]);
  const int logits_width = static_cast<int>(shape[3]);
  const std::size_t numel = tensor->numel();

  // Validate every logit is finite before any arithmetic. Report the first
  // offending element with its (class, y, x) coordinates. Never coerce NaN.
  const auto validate_start = std::chrono::steady_clock::now();
  const float* raw = tensor->data_as<float>();
  for (std::size_t i = 0; i < numel; ++i) {
    if (!std::isfinite(raw[i])) {
      const std::size_t plane = static_cast<std::size_t>(logits_height) * logits_width;
      const int c = static_cast<int>(i / plane);
      const std::size_t rem = i % plane;
      const int y = static_cast<int>(rem / logits_width);
      const int x = static_cast<int>(rem % logits_width);
      const char* reason = std::isnan(raw[i]) ? "NaN" : "infinite";
      throw VieError(ErrorCode::InvalidArgument, "semantic segmentation logit at [c=" + std::to_string(c) +
                                                     ", y=" + std::to_string(y) +
                                                     ", x=" + std::to_string(x) + "] is " + reason);
    }
  }
  const double validate_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - validate_start).count();

  const auto argmax_start = std::chrono::steady_clock::now();

  // Step A: bring the logits onto the model-input grid (Bilinear) when the model
  // produced a lower-resolution map. Identity when the resolutions already match.
  const int grid_h = geometry.output_height;
  const int grid_w = geometry.output_width;
  std::vector<float> resized;
  const float* grid = raw;
  int grid_channels = channels;
  if (grid_h != logits_height || grid_w != logits_width) {
    resized = resize_logits(raw, channels, logits_height, logits_width, grid_h, grid_w);
    grid = resized.data();
  }

  // Step B: for every original pixel, map to model-input coordinates (which
  // removes letterbox padding via origin_* and stretch via scale_*), sample each
  // channel bilinearly, then ArgMax. class ids are produced per original pixel,
  // so no interpolation is ever applied to integer ids (nearest semantics).
  const std::size_t out_count =
      static_cast<std::size_t>(geometry.original_width) * static_cast<std::size_t>(geometry.original_height);
  if (geometry.original_width > 0 &&
      out_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw VieError(ErrorCode::InvalidArgument, "semantic segmentation output size overflows");
  }
  const int origin_x = geometry_model_origin_x(geometry);
  const int origin_y = geometry_model_origin_y(geometry);

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
  result.execution_device = DeviceType::Cpu;
  result.execution_device_id = device.device_id();
  result.output_bytes = tensor->nbytes();
  result.d2h_bytes = 0;
  result.class_ids.assign(out_count, 0);

  for (int oy = 0; oy < geometry.original_height; ++oy) {
    const double my = oy * geometry.scale_y + origin_y;
    for (int ox = 0; ox < geometry.original_width; ++ox) {
      const double mx = ox * geometry.scale_x + origin_x;
      int best_class = 0;
      float best_value = -std::numeric_limits<float>::infinity();
      for (int c = 0; c < grid_channels; ++c) {
        const float* plane = grid + static_cast<std::size_t>(c) * grid_h * grid_w;
        const float value = sample_plane(plane, grid_h, grid_w, mx, my);
        // Strict '>' keeps the first (smallest) class index on a tie, giving a
        // deterministic result for equal maxima.
        if (value > best_value) {
          best_value = value;
          best_class = c;
        }
      }
      if (best_class < 0 || best_class >= channels) {
        throw VieError(ErrorCode::InvalidArgument, "semantic segmentation produced an out-of-range class id");
      }
      result.class_ids[static_cast<std::size_t>(oy) * geometry.original_width + ox] =
          static_cast<std::int32_t>(best_class);
    }
  }
  const double argmax_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - argmax_start).count();
  result.validate_ms = validate_ms;
  result.argmax_ms = argmax_ms;

  SemanticSegmentationBatch batch;
  batch.results.push_back(std::move(result));
  return batch;
}

}  // namespace vie
