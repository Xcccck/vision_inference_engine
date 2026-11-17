#pragma once

#include "vie/backend/inference_backend.hpp"

#include <memory>

namespace vie {

// Applied on the next prepare(). These names are VIE's, not ORT's.
enum class OnnxRuntimeOptimization { Disabled, Basic, Extended, All };

struct OnnxRuntimeTiming {
  double load_ms = 0;
  double last_execute_ms = 0;
  double last_input_copy_ms = 0;
  double last_output_copy_ms = 0;
  bool input_was_copied = false;
};

// CPU ONNX Runtime backend. Outputs are copied into VIE-owned storage before
// the runtime releases them. Non-contiguous inputs are packed into a
// contiguous temporary for that call. Not thread-safe. Thread count,
// optimization, and log level apply on the next prepare().
class OnnxRuntimeBackend final : public IInferenceBackend {
 public:
  OnnxRuntimeBackend();
  ~OnnxRuntimeBackend() override;

  OnnxRuntimeBackend(const OnnxRuntimeBackend&) = delete;
  OnnxRuntimeBackend& operator=(const OnnxRuntimeBackend&) = delete;

  void set_intra_op_threads(int threads);
  void set_inter_op_threads(int threads);
  void set_graph_optimization(OnnxRuntimeOptimization level);
  void set_log_severity(int severity);

  void prepare(const BackendConfig& config) override;
  BackendOutput execute(const BackendRequest& request) override;
  BackendInfo info() const override;

  const OnnxRuntimeTiming& timing() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vie
