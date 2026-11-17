#include "vie/core/device_context.hpp"

#include "cuda_memory.hpp"

namespace vie {

DeviceContext::DeviceContext() = default;

DeviceContext::DeviceContext(DeviceType device, int device_id)
    : device_(device), device_id_(device_id) {
  if (device_id_ < 0) {
    throw VieError(ErrorCode::InvalidArgument, "device id must be >= 0");
  }
  if (device_ == DeviceType::Cuda) {
#if !defined(VIE_HAS_CUDA)
    throw VieError(ErrorCode::Unsupported, "CUDA device context is not available in this build");
#else
    cuda_ = make_cuda_state(device_id_);
#endif
  }
}

void DeviceContext::synchronize() const {
  if (device_ != DeviceType::Cuda) return;
  if (cuda_ == nullptr) {
    throw VieError(ErrorCode::Unsupported, "CUDA synchronize is not available in this build");
  }
  synchronize_cuda_state(*cuda_);
}

}  // namespace vie
