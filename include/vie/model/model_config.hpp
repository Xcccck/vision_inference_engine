#pragma once

#include "vie/model/manifest.hpp"

#include <optional>
#include <string>

namespace vie {

// Runtime request. task == Unknown means "do not override the manifest".
// backend == Auto selects from the manifest format and the requested device.
// ONNX on CUDA uses the CUDA backend when this build has one. A failed CUDA
// load leaves the requested device unchanged and does not switch to CPU.
struct ModelConfig {
  std::string model_path;
  TaskType task = TaskType::Unknown;
  BackendKind backend = BackendKind::Auto;
  DeviceType device = DeviceType::Cpu;
  int device_id = 0;
  std::optional<ModelManifest> manifest;
};

}  // namespace vie
