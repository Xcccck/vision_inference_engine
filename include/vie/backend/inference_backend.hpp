#pragma once

#include "vie/core/tensor.hpp"

#include <map>
#include <string>
#include <vector>

namespace vie {

// input is the optional concrete profile for one tensor. A backend that
// reads its own session uses that session as the source of truth, and
// enforces this profile only when the name is set and the shape is concrete.
struct BackendConfig {
  std::string model_path;
  ModelFormat model_format = ModelFormat::Unspecified;
  DeviceType device = DeviceType::Cpu;
  int device_id = 0;
  TensorSpec input;
  std::vector<TensorSpec> outputs;
};

struct BackendRequest {
  std::map<std::string, Tensor> inputs;
};

struct NamedTensor {
  Tensor tensor;
  TensorLifetime lifetime = TensorLifetime::Copied;
};

struct BackendOutput {
  std::map<std::string, NamedTensor> tensors;
};

struct BackendInfo {
  std::string name;
  BackendKind kind = BackendKind::Mock;
  DeviceType device = DeviceType::Cpu;
  bool supports_cuda = false;
};

// Executes a prepared model. The result is named tensors, not detections.
class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  virtual void prepare(const BackendConfig& config) = 0;
  virtual BackendOutput execute(const BackendRequest& request) = 0;
  virtual BackendInfo info() const = 0;
};

}  // namespace vie
