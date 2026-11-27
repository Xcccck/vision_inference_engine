#pragma once

#include "vie/config/inference_config.hpp"
#include "vie/decoder/detection.hpp"
#include "vie/preprocess/image_input.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace vie {

// Forward declaration - full definition in model_instance.hpp
class ModelInstance;

// Unique request identifier
using RequestId = std::uint64_t;

// One asynchronous inference request. Holds the image data, a pre-resolved
// shared_ptr to the model instance (keeps model alive during queueing and
// execution), and a promise for the result. Move-only.
//
// Phase 9.2.2.2: For CPU Borrowed images, pixel data is deep-copied into
// owned_pixels at submit time. image.data is then redirected to point to
// owned_pixels.data(), ensuring the caller can safely release their buffer
// immediately after submit() returns without causing Use-After-Free.
//
// Phase 9.2.2.3: CUDA device images (device != Cpu) are rejected at
// Runtime::submit() before reaching InferenceRequest. The async pipeline
// cannot guarantee CUDA buffer lifetime; use InferenceEngine::infer() for
// synchronous CUDA inference instead.
struct InferenceRequest {
  RequestId id = 0;
  std::string model_id;
  ImageInput image;
  std::vector<std::uint8_t> owned_pixels;  // Deep copy for CPU Borrowed images
  std::shared_ptr<ModelInstance> model;    // Resolved at submit time
  // Phase 12: effective inference policy, resolved and frozen at submit time.
  // Empty (task Unknown) for requests built outside Runtime::submit(), which
  // keep the legacy engine path. config_key is config.fingerprint() and is the
  // second grouping key used by the batch aggregator so requests with different
  // policies never share a batch.
  ResolvedInferenceConfig config;
  std::string config_key;
  std::chrono::steady_clock::time_point submitted_at;
  std::promise<DetectionResult> promise;

  InferenceRequest() = default;
  InferenceRequest(InferenceRequest&&) = default;
  InferenceRequest& operator=(InferenceRequest&&) = default;

  InferenceRequest(const InferenceRequest&) = delete;
  InferenceRequest& operator=(const InferenceRequest&) = delete;

  // Deep-copies CPU pixel data so the request is self-contained.
  // After this call, image.data points to owned_pixels and the caller may
  // safely release the original buffer.
  // CUDA inputs are rejected at Runtime::submit() before reaching this
  // method, so this function is always called with device == Cpu.
  //
  // Only the packed width*channels*element bytes of each row are read, never
  // the source row padding. A strided source (a cv::Mat ROI, or any caller
  // supplied row_stride_bytes larger than one packed row) is therefore
  // repacked into a contiguous buffer: the last read ends at the last valid
  // pixel even when the ROI's last row ends exactly on the parent's
  // allocation boundary. CHW planes are packed independently, matching the
  // documented "row stride inside one plane, planes contiguous" projection.
  //
  // Non-positive geometry is left untouched so the established worker-side
  // rejection (staging/preprocess throw through the future before reading a
  // pixel) is preserved. Throws ErrorCode::InvalidArgument for an unsized
  // dtype, a stride smaller than one packed row, a misaligned stride or a
  // size overflow.
  void adopt_cpu_pixels() {
    if (image.device != DeviceType::Cpu || image.data == nullptr) {
      return;  // Defensive: CUDA images should not reach here via submit().
    }
    if (!owned_pixels.empty() && image.data == static_cast<const void*>(owned_pixels.data())) {
      return;  // Already adopted: the pixels are packed and request-owned.
    }
    if (image.width <= 0 || image.height <= 0 || image.channels <= 0) {
      // Keep the established submit() contract: a geometry error surfaces from
      // CUDA staging or preprocess inside the worker (through the future)
      // before any pixel is read. Copying is meaningless, so leave the image
      // untouched rather than rejecting it at submit time.
      return;
    }
    if (image.layout != ImageLayout::Hwc && image.layout != ImageLayout::Chw) {
      throw VieError(ErrorCode::InvalidArgument,
                     "async image copy requires an HWC or CHW image layout");
    }
    const std::size_t element = dtype_size(image.dtype);
    const bool hwc = image.layout == ImageLayout::Hwc;
    const std::size_t packed_row =
        mul_checked(mul_checked(static_cast<std::size_t>(image.width),
                                hwc ? static_cast<std::size_t>(image.channels) : 1u,
                                "width*channels"),
                    element, "row bytes");
    const std::size_t source_row =
        image.row_stride_bytes != 0 ? image.row_stride_bytes : packed_row;
    if (source_row < packed_row || source_row % element != 0) {
      throw VieError(ErrorCode::InvalidArgument,
                     "image row_stride_bytes is smaller than one packed row or is misaligned");
    }
    const std::size_t planes = hwc ? 1u : static_cast<std::size_t>(image.channels);
    const std::size_t packed_plane =
        mul_checked(packed_row, static_cast<std::size_t>(image.height), "packed row*height");
    const std::size_t source_plane =
        mul_checked(source_row, static_cast<std::size_t>(image.height), "row stride*height");
    const std::size_t total_bytes = mul_checked(packed_plane, planes, "plane bytes*plane count");

    owned_pixels.resize(total_bytes);
    const auto* source = static_cast<const std::uint8_t*>(image.data);
    std::uint8_t* destination = owned_pixels.data();
    if (source_row == packed_row) {
      // Already packed: this single copy cannot run past the last valid pixel.
      std::memcpy(destination, source, total_bytes);
    } else {
      // Copy each row's valid bytes only. The final read ends at
      // source + (planes-1)*source_plane + (height-1)*source_row + packed_row,
      // which is the image's last valid pixel for any ROI.
      for (std::size_t plane = 0; plane < planes; ++plane) {
        const std::uint8_t* source_plane_base = source + plane * source_plane;
        std::uint8_t* destination_plane_base = destination + plane * packed_plane;
        for (int row = 0; row < image.height; ++row) {
          std::memcpy(destination_plane_base + static_cast<std::size_t>(row) * packed_row,
                      source_plane_base + static_cast<std::size_t>(row) * source_row, packed_row);
        }
      }
    }
    // Redirect image.data to the owned, tightly packed copy.
    image.data = owned_pixels.data();
    image.row_stride_bytes = packed_row;
  }

 private:
  static std::size_t mul_checked(std::size_t lhs, std::size_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
      throw VieError(ErrorCode::InvalidArgument,
                     std::string("async image copy overflow computing ") + what);
    }
    return lhs * rhs;
  }
};

}  // namespace vie
