#pragma once

#include "vie/decoder/detection.hpp"
#include "vie/preprocess/image_input.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
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
  std::chrono::steady_clock::time_point submitted_at;
  std::promise<DetectionResult> promise;

  InferenceRequest() = default;
  InferenceRequest(InferenceRequest&&) = default;
  InferenceRequest& operator=(InferenceRequest&&) = default;

  InferenceRequest(const InferenceRequest&) = delete;
  InferenceRequest& operator=(const InferenceRequest&) = delete;

  // Deep-copies CPU pixel data so the request is self-contained.
  // After this call, image.data points to owned_pixels and the caller
  // may safely release the original buffer.
  // CUDA inputs are rejected at Runtime::submit() before reaching this
  // method, so this function is always called with device == Cpu.
  void adopt_cpu_pixels() {
    if (image.device != DeviceType::Cpu || image.data == nullptr) {
      return;  // Defensive: CUDA images should not reach here via submit().
    }
    const std::size_t row_bytes =
        (image.row_stride_bytes != 0)
            ? image.row_stride_bytes
            : static_cast<std::size_t>(image.width) * image.channels * dtype_bytes(image.dtype);
    const std::size_t total_bytes = row_bytes * static_cast<std::size_t>(image.height);
    owned_pixels.resize(total_bytes);
    std::memcpy(owned_pixels.data(), image.data, total_bytes);
    // Redirect image.data to our owned copy
    image.data = owned_pixels.data();
    image.row_stride_bytes = row_bytes;
  }

 private:
  static std::size_t dtype_bytes(DataType dt) {
    switch (dt) {
      case DataType::Uint8:
      case DataType::Int8:
        return 1;
      case DataType::Fp16:
        return 2;
      case DataType::Fp32:
      case DataType::Int32:
        return 4;
      case DataType::Int64:
        return 8;
      default:
        return 1;
    }
  }
};

}  // namespace vie
