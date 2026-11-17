#pragma once

#include "vie/core/types.hpp"

#include <cstddef>
#include <memory>

namespace vie {

struct CudaState;

// Owning byte buffer. Tensor views share one storage through shared_ptr.
// Host storage is 64-byte aligned and released with the aligned operator delete.
// CUDA storage is a cudaMalloc allocation released with cudaFree.
// The raw address stays inside the class. Tensor::data() exposes it only for CPU,
// and Tensor::device_data() exposes it only for CUDA.
class TensorStorage {
 public:
  static constexpr std::size_t kAlignment = 64;
  using ReleaseFn = void (*)(std::byte* data, int device_id, std::size_t bytes) noexcept;

  static std::shared_ptr<TensorStorage> allocate_host(std::size_t bytes, int device_id = 0);
  static std::shared_ptr<TensorStorage> allocate_cuda(int device_id, std::size_t bytes,
                                                      CudaState* state);

  TensorStorage(const TensorStorage&) = delete;
  TensorStorage& operator=(const TensorStorage&) = delete;

  std::size_t size_bytes() const noexcept { return size_bytes_; }
  DeviceType device() const noexcept { return device_; }
  int device_id() const noexcept { return device_id_; }

  ~TensorStorage();

 private:
  friend class Tensor;

  std::byte* data() noexcept { return data_; }
  const std::byte* data() const noexcept { return data_; }

  TensorStorage(std::byte* data, std::size_t bytes, DeviceType device, int device_id,
                ReleaseFn release);

  std::byte* data_ = nullptr;
  std::size_t size_bytes_ = 0;
  DeviceType device_ = DeviceType::Cpu;
  int device_id_ = 0;
  ReleaseFn release_ = nullptr;
};

}  // namespace vie
