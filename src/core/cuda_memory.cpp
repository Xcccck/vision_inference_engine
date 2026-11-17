#include "cuda_memory.hpp"

#include "vie/core/tensor_storage.hpp"

#include <string>

#if defined(VIE_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace vie {
namespace {

std::string device_text(DeviceType device, int device_id) {
  return std::string(to_string(device)) + ":" + std::to_string(device_id);
}

#if defined(VIE_HAS_CUDA)

void check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  throw VieError(ErrorCode::Internal, std::string(operation) + " failed: " + cudaGetErrorString(status));
}

#endif

}  // namespace

#if defined(VIE_HAS_CUDA)

struct CudaState {
  int device_id = 0;
  cudaStream_t stream = nullptr;

  ~CudaState() {
    if (stream == nullptr) return;
    cudaSetDevice(device_id);
    cudaStreamDestroy(stream);
    stream = nullptr;
  }
};

namespace {

void release_cuda(std::byte* data, int device_id, std::size_t) noexcept {
  if (data == nullptr) return;
  cudaSetDevice(device_id);
  cudaFree(data);
}

}  // namespace

std::shared_ptr<CudaState> make_cuda_state(int device_id) {
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (device_id < 0 || device_id >= count) {
    throw VieError(ErrorCode::InvalidArgument,
                   "CUDA device id " + std::to_string(device_id) + " is invalid; device count is " +
                       std::to_string(count));
  }
  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  try {
    auto state = std::make_shared<CudaState>();
    state->device_id = device_id;
    state->stream = stream;
    return state;
  } catch (...) {
    cudaSetDevice(device_id);
    cudaStreamDestroy(stream);
    throw;
  }
}

void synchronize_cuda_state(const CudaState& state) {
  check_cuda(cudaSetDevice(state.device_id), "cudaSetDevice");
  check_cuda(cudaStreamSynchronize(state.stream), "cudaStreamSynchronize");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

std::shared_ptr<TensorStorage> TensorStorage::allocate_cuda(int device_id, std::size_t bytes,
                                                            CudaState* state) {
  if (device_id < 0) {
    throw VieError(ErrorCode::InvalidArgument, "device id must be >= 0");
  }
  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
  std::byte* data = nullptr;
  if (bytes == 0) {
    return std::shared_ptr<TensorStorage>(
        new TensorStorage(nullptr, 0, DeviceType::Cuda, device_id, release_cuda));
  }
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "cudaMalloc");
  const cudaStream_t stream = state == nullptr ? nullptr : state->stream;
  const cudaError_t cleared =
      state == nullptr ? cudaMemset(data, 0, bytes) : cudaMemsetAsync(data, 0, bytes, stream);
  if (cleared != cudaSuccess) {
    cudaFree(data);
    check_cuda(cleared, "cudaMemset");
  }
  if (state != nullptr) {
    const cudaError_t synced = cudaStreamSynchronize(stream);
    if (synced != cudaSuccess) {
      cudaFree(data);
      check_cuda(synced, "cudaStreamSynchronize");
    }
  }
  try {
    return std::shared_ptr<TensorStorage>(
        new TensorStorage(data, bytes, DeviceType::Cuda, device_id, release_cuda));
  } catch (...) {
    cudaFree(data);
    throw;
  }
}

void copy_device_memory(CudaState* state, DeviceType src_device, int src_id, const std::byte* src,
                        DeviceType dst_device, int dst_id, std::byte* dst, std::size_t bytes) {
  if (bytes == 0) return;
  if (src == nullptr || dst == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "copy from " + device_text(src_device, src_id) +
                                                   " to " + device_text(dst_device, dst_id) +
                                                   " failed: null buffer");
  }
  if (src_device == DeviceType::Cuda && dst_device == DeviceType::Cuda && src_id != dst_id) {
    throw VieError(ErrorCode::Unsupported, "copy from " + device_text(src_device, src_id) + " to " +
                                               device_text(dst_device, dst_id) +
                                               " is not supported");
  }
  const int device_id = src_device == DeviceType::Cuda ? src_id : dst_id;
  cudaMemcpyKind kind = cudaMemcpyHostToHost;
  if (src_device == DeviceType::Cpu && dst_device == DeviceType::Cuda) {
    kind = cudaMemcpyHostToDevice;
  } else if (src_device == DeviceType::Cuda && dst_device == DeviceType::Cpu) {
    kind = cudaMemcpyDeviceToHost;
  } else if (src_device == DeviceType::Cuda && dst_device == DeviceType::Cuda) {
    kind = cudaMemcpyDeviceToDevice;
  } else {
    throw VieError(ErrorCode::InvalidArgument, "copy from " + device_text(src_device, src_id) +
                                                   " to " + device_text(dst_device, dst_id) +
                                                   " failed: host copies do not use the CUDA runtime");
  }
  try {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
    if (state == nullptr) {
      check_cuda(cudaMemcpy(dst, src, bytes, kind), "cudaMemcpy");
      return;
    }
    check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, state->stream), "cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(state->stream), "cudaStreamSynchronize");
  } catch (const VieError& error) {
    throw VieError(error.code(), "copy from " + device_text(src_device, src_id) + " to " +
                                     device_text(dst_device, dst_id) + " failed: " + error.what());
  }
}

#else

struct CudaState {};

std::shared_ptr<CudaState> make_cuda_state(int) {
  throw VieError(ErrorCode::Unsupported, "CUDA device context is not available in this build");
}

void synchronize_cuda_state(const CudaState&) {
  throw VieError(ErrorCode::Unsupported, "CUDA synchronize is not available in this build");
}

std::shared_ptr<TensorStorage> TensorStorage::allocate_cuda(int, std::size_t, CudaState*) {
  throw VieError(ErrorCode::Unsupported, "CUDA tensor allocation is not available in this build");
}

void copy_device_memory(CudaState*, DeviceType src_device, int src_id, const std::byte*,
                        DeviceType dst_device, int dst_id, std::byte*, std::size_t) {
  throw VieError(ErrorCode::Unsupported, "copy from " + device_text(src_device, src_id) + " to " +
                                             device_text(dst_device, dst_id) +
                                             " is not available in this build");
}

#endif

}  // namespace vie
