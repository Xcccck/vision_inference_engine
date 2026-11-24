#pragma once

#include "vie/decoder/detection.hpp"
#include "vie/preprocess/image_input.hpp"

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>

namespace vie {

// Forward declaration - full definition in model_instance.hpp
class ModelInstance;

// Unique request identifier
using RequestId = std::uint64_t;

// One asynchronous inference request. Holds the image data, a pre-resolved
// shared_ptr to the model instance (keeps model alive during queueing and
// execution), and a promise for the result. Move-only.
struct InferenceRequest {
  RequestId id = 0;
  std::string model_id;
  ImageInput image;
  std::shared_ptr<ModelInstance> model;  // Resolved at submit time
  std::chrono::steady_clock::time_point submitted_at;
  std::promise<DetectionResult> promise;

  InferenceRequest() = default;
  InferenceRequest(InferenceRequest&&) = default;
  InferenceRequest& operator=(InferenceRequest&&) = default;

  InferenceRequest(const InferenceRequest&) = delete;
  InferenceRequest& operator=(const InferenceRequest&) = delete;
};

}  // namespace vie
