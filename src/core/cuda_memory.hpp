#pragma once

#include "vie/core/device_context.hpp"

#include <cstddef>
#include <memory>

namespace vie {

std::shared_ptr<CudaState> make_cuda_state(int device_id);
void synchronize_cuda_state(const CudaState& state);

// Synchronous byte copy. A CUDA state, when provided, is synchronized before
// the function returns. Different CUDA device ids are rejected.
void copy_device_memory(CudaState* state, DeviceType src_device, int src_id, const std::byte* src,
                        DeviceType dst_device, int dst_id, std::byte* dst, std::size_t bytes);

}  // namespace vie
