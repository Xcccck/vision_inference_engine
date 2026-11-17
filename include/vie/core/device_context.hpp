#pragma once

#include "vie/core/types.hpp"

#include <memory>

namespace vie {

struct CudaState;

// CPU context, or one CUDA device plus a stream owned for the context lifetime.
// Copies share that stream. synchronize() is a no-op on CPU. On CUDA it waits
// until work already issued on this device has finished. Tensor copies in this
// phase are synchronous, so they have finished before the copy call returns.
class DeviceContext {
 public:
  DeviceContext();
  DeviceContext(DeviceType device, int device_id);

  DeviceType device() const noexcept { return device_; }
  int device_id() const noexcept { return device_id_; }
  const std::shared_ptr<CudaState>& execution_state() const noexcept { return cuda_; }

  void synchronize() const;

 private:
  DeviceType device_ = DeviceType::Cpu;
  int device_id_ = 0;
  std::shared_ptr<CudaState> cuda_;
};

}  // namespace vie
