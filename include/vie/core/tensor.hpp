#pragma once

#include "vie/core/device_context.hpp"
#include "vie/core/tensor_storage.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace vie {

// Non-owning view of a TensorStorage: shape, strides, and a byte offset.
// Copies share storage. clone() allocates a new contiguous buffer on the same
// device. data() is a host pointer and rejects CUDA tensors. device_data() is
// the CUDA address and rejects CPU tensors.
class Tensor {
 public:
  Tensor() = default;

  static Tensor zeros(const DeviceContext& device, DataType dtype,
                       std::vector<std::int64_t> shape);
  static Tensor from_host(const DeviceContext& device, DataType dtype,
                           std::vector<std::int64_t> shape, const void* src,
                           std::size_t src_bytes);

  DataType dtype() const noexcept { return dtype_; }
  const std::vector<std::int64_t>& shape() const noexcept { return shape_; }
  const std::vector<std::int64_t>& strides() const noexcept { return strides_; }
  DeviceType device() const noexcept { return device_; }
  int device_id() const noexcept { return device_id_; }
  std::size_t byte_offset() const noexcept { return byte_offset_; }
  Layout layout() const noexcept { return layout_; }
  void set_layout(Layout layout) noexcept { layout_ = layout; }

  std::size_t rank() const noexcept { return shape_.size(); }
  std::size_t numel() const;
  std::size_t nbytes() const;
  std::size_t storage_bytes() const;
  bool empty() const noexcept { return storage_ == nullptr; }
  bool contiguous() const;
  bool is_cpu() const noexcept { return device_ == DeviceType::Cpu; }
  bool is_cuda() const noexcept { return device_ == DeviceType::Cuda; }

  std::byte* data();
  const std::byte* data() const;
  std::byte* device_data();
  const std::byte* device_data() const;

  // Contiguous copy on destination. Same device is clone(). Different CUDA
  // device ids are rejected.
  Tensor to(const DeviceContext& destination) const;
  // Overwrite this contiguous tensor from a contiguous source of the same
  // shape and dtype. Returns after the bytes are visible.
  void copy_from(const Tensor& source);

  // Strict row-major view. Strides are element strides, not byte strides.
  Tensor view(std::vector<std::int64_t> shape, std::vector<std::int64_t> strides,
              std::size_t byte_offset) const;

  Tensor clone() const;
  bool shares_storage(const Tensor& other) const noexcept;

  template <typename T>
  T* mutable_data();

  template <typename T>
  const T* data_as() const;

 private:
  Tensor(std::shared_ptr<TensorStorage> storage, DataType dtype,
         std::vector<std::int64_t> shape, std::vector<std::int64_t> strides,
         std::size_t byte_offset, DeviceType device, int device_id);

  void require_bound() const;
  std::size_t span_bytes() const;

  std::shared_ptr<TensorStorage> storage_;
  DataType dtype_ = DataType::Unknown;
  std::vector<std::int64_t> shape_;
  std::vector<std::int64_t> strides_;
  std::size_t byte_offset_ = 0;
  DeviceType device_ = DeviceType::Cpu;
  int device_id_ = 0;
  Layout layout_ = Layout::Unknown;
};

template <typename T>
DataType dtype_of();

template <>
inline DataType dtype_of<float>() {
  return DataType::Fp32;
}
template <>
inline DataType dtype_of<std::int32_t>() {
  return DataType::Int32;
}
template <>
inline DataType dtype_of<std::int8_t>() {
  return DataType::Int8;
}
template <>
inline DataType dtype_of<std::uint8_t>() {
  return DataType::Uint8;
}
template <>
inline DataType dtype_of<std::int64_t>() {
  return DataType::Int64;
}

template <typename T>
T* Tensor::mutable_data() {
  require_bound();
  if (device_ != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "typed access is implemented for CPU tensors only");
  }
  if (!contiguous()) {
    throw VieError(ErrorCode::InvalidArgument, "typed access requires a contiguous tensor");
  }
  if (dtype_ != dtype_of<T>()) {
    throw VieError(ErrorCode::InvalidArgument, "tensor dtype does not match the requested C++ type");
  }
  return reinterpret_cast<T*>(data());
}

template <typename T>
const T* Tensor::data_as() const {
  return const_cast<Tensor*>(this)->mutable_data<T>();
}

}  // namespace vie
