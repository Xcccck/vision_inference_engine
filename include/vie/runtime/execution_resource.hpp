#pragma once

#include "vie/core/device_context.hpp"
#include "vie/core/types.hpp"

#include <string>

namespace vie {

// Phase 9.3: Identifies the real execution resource a ModelInstance occupies.
//
// This is intentionally NOT the ImageInput device. Per the Phase 9.3 audit,
// InferenceEngine commits a single DeviceContext at load() time and every
// stage of inference (preprocess, backend execute, decode) uses that same
// context. resolve_backend() guarantees OnnxRuntimeCuda <=> DeviceType::Cuda
// and OnnxRuntime => DeviceType::Cpu, and the mock backend never runs on
// CUDA. Therefore, for a *loaded* engine, device() is the authoritative
// execution resource of the whole inference request.
//
// Unknown is a real state (e.g. an engine that has never loaded) and must
// never be silently treated as CPU by the scheduler.
enum class ExecutionResourceType {
  Cpu,
  Cuda,
  Unknown,
};

struct ExecutionResource {
  ExecutionResourceType type = ExecutionResourceType::Unknown;
  int device_id = 0;

  bool operator==(const ExecutionResource& other) const noexcept {
    return type == other.type && device_id == other.device_id;
  }
  bool operator!=(const ExecutionResource& other) const noexcept {
    return !(*this == other);
  }

  bool valid() const noexcept { return type != ExecutionResourceType::Unknown; }

  // Stable, human-readable key used for diagnostics and metrics ("cpu",
  // "cuda:0"). Unknown renders as "unknown".
  std::string key() const {
    switch (type) {
      case ExecutionResourceType::Cpu:
        return "cpu";
      case ExecutionResourceType::Cuda:
        return "cuda:" + std::to_string(device_id);
      case ExecutionResourceType::Unknown:
        break;
    }
    return "unknown";
  }
};

inline const char* to_string(ExecutionResourceType type) {
  switch (type) {
    case ExecutionResourceType::Cpu:
      return "cpu";
    case ExecutionResourceType::Cuda:
      return "cuda";
    case ExecutionResourceType::Unknown:
      break;
  }
  return "unknown";
}

// Maps a resolved (DeviceType, device_id) pair to an ExecutionResource. This
// core mapping is independent of DeviceContext construction so it stays
// testable in CUDA-OFF builds, where DeviceContext(Cuda, ...) throws.
inline ExecutionResource execution_resource_from(DeviceType device, int device_id) noexcept {
  switch (device) {
    case DeviceType::Cpu:
      return ExecutionResource{ExecutionResourceType::Cpu, 0};
    case DeviceType::Cuda:
      return ExecutionResource{ExecutionResourceType::Cuda, device_id};
  }
  return ExecutionResource{ExecutionResourceType::Unknown, 0};
}

// Maps a committed engine DeviceContext to an ExecutionResource. The caller
// must only use the result for a loaded engine; a never-loaded engine reports
// the default CPU context and would be misclassified, which is why
// ModelInstance::execution_resource() guards on loaded() and returns Unknown
// otherwise.
inline ExecutionResource execution_resource_from_device(const DeviceContext& device) noexcept {
  return execution_resource_from(device.device(), device.device_id());
}

}  // namespace vie
