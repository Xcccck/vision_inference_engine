#pragma once

#include "vie/backend/onnxruntime_backend.hpp"

#include <memory>
#include <string>

namespace vie {

// Times are wall-clock milliseconds and include the synchronization that makes
// the corresponding bytes visible. last_run_ms is only the ORT Run call.
// last_sync_ms is the explicit wait after Run. A GPU execution figure is
// last_run_ms + last_sync_ms, not last_run_ms by itself.
struct OnnxRuntimeCudaTiming {
  double load_ms = 0;
  double last_bind_ms = 0;
  double last_input_sync_ms = 0;
  double last_run_ms = 0;
  double last_sync_ms = 0;
  double last_output_alloc_ms = 0;
  double last_output_copy_ms = 0;
};

// ONNX Runtime CUDA backend. Inputs must already be contiguous CUDA tensors on
// the configured device. Ort::Value borrows that storage for the duration of
// execute() and does not free it. Outputs are copied with cudaMemcpy into a
// new VIE CUDA tensor before the ORT value is released. ORT keeps its own
// compute stream; this backend does not inject DeviceContext's stream.
// cpu_ep_fallback is off by default, so a node CUDA cannot run fails prepare().
// Turning it on allows ORT's mixed CPU fallback and must not be reported as a
// pure GPU execution. Not thread-safe. Options apply on the next prepare().
class OnnxRuntimeCudaBackend final : public IInferenceBackend {
 public:
  OnnxRuntimeCudaBackend();
  ~OnnxRuntimeCudaBackend() override;

  OnnxRuntimeCudaBackend(const OnnxRuntimeCudaBackend&) = delete;
  OnnxRuntimeCudaBackend& operator=(const OnnxRuntimeCudaBackend&) = delete;

  void set_intra_op_threads(int threads);
  void set_inter_op_threads(int threads);
  void set_graph_optimization(OnnxRuntimeOptimization level);
  void set_log_severity(int severity);
  void set_cpu_ep_fallback(bool enabled);

  void prepare(const BackendConfig& config) override;
  BackendOutput execute(const BackendRequest& request) override;
  BackendInfo info() const override;

  const OnnxRuntimeCudaTiming& timing() const;
  bool cpu_ep_fallback() const;
  std::string session_log() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vie
