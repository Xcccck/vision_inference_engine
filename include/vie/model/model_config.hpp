#pragma once

#include "vie/config/inference_config.hpp"
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
  // Allow CUDA backend to fall back to CPU for unsupported ops (CUDA only)
  bool cuda_cpu_ep_fallback = false;
  // Phase 9.3.2.1: Opt-in ONNX Runtime profiling for EP-placement diagnostics.
  // Both default-off, so the normal production path is byte-for-byte unchanged.
  // When enabled, the backend calls Ort::SessionOptions::EnableProfiling on the
  // *same* session the engine runs (never a separate profiling session); the
  // prefix names the output Chrome-trace JSON. Call InferenceEngine /
  // ModelInstance::end_profiling() after inference to flush and obtain the path.
  bool ort_enable_profiling = false;
  std::string ort_profile_prefix;
  // Phase 12: model-level inference policy defaults. They sit between a usable
  // manifest value and a per-request override. Empty means "inherit". A block
  // that does not belong to the model's task is rejected at load().
  InferenceOptions inference_options;
};

}  // namespace vie
