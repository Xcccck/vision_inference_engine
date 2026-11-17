#include "vie/core/tensor.hpp"

#include "cuda_memory.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace vie {
namespace {

std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    throw VieError(ErrorCode::InvalidArgument, "tensor byte extent overflows size_t");
  }
  return lhs + rhs;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw VieError(ErrorCode::InvalidArgument, "tensor byte extent overflows size_t");
  }
  return lhs * rhs;
}

std::string device_text(DeviceType device, int device_id) {
  return std::string(to_string(device)) + ":" + std::to_string(device_id);
}

void gather_strided(const std::byte* source, const std::vector<std::int64_t>& shape,
                    const std::vector<std::int64_t>& strides, std::size_t elem, std::byte* dest) {
  std::vector<std::int64_t> index(shape.size(), 0);
  std::size_t written = 0;
  const std::size_t total = checked_numel(shape);
  while (written < total) {
    std::size_t source_index = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      source_index = checked_add(source_index, checked_mul(static_cast<std::size_t>(index[axis]),
                                                           static_cast<std::size_t>(strides[axis])));
    }
    std::memcpy(dest + written * elem, source + source_index * elem, elem);
    ++written;
    if (written == total) break;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
      ++index[axis];
      if (index[axis] < shape[axis]) break;
      index[axis] = 0;
    }
  }
}

}  // namespace

Tensor::Tensor(std::shared_ptr<TensorStorage> storage, DataType dtype,
               std::vector<std::int64_t> shape, std::vector<std::int64_t> strides,
               std::size_t byte_offset, DeviceType device, int device_id)
    : storage_(std::move(storage)),
      dtype_(dtype),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      byte_offset_(byte_offset),
      device_(device),
      device_id_(device_id) {}

Tensor Tensor::zeros(const DeviceContext& device, DataType dtype, std::vector<std::int64_t> shape) {
  if (dtype == DataType::Unknown) {
    throw VieError(ErrorCode::InvalidArgument, "cannot allocate a tensor with an unknown dtype");
  }
  if (!is_concrete_shape(shape)) {
    throw VieError(ErrorCode::InvalidArgument, "tensor shape must be concrete and non-empty");
  }
  const auto strides = row_major_strides(shape);
  const std::size_t bytes = checked_mul(checked_numel(shape), dtype_size(dtype));
  if (device.device() == DeviceType::Cpu) {
    auto storage = TensorStorage::allocate_host(bytes, device.device_id());
    return Tensor(std::move(storage), dtype, std::move(shape), strides, 0, DeviceType::Cpu,
                  device.device_id());
  }
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported, "tensor allocation supports CPU and CUDA only");
  }
  if (device.execution_state() == nullptr) {
    throw VieError(ErrorCode::Unsupported, "CUDA tensor allocation is not available in this build");
  }
  auto storage = TensorStorage::allocate_cuda(device.device_id(), bytes, device.execution_state().get());
  return Tensor(std::move(storage), dtype, std::move(shape), strides, 0, DeviceType::Cuda,
                device.device_id());
}

Tensor Tensor::from_host(const DeviceContext& device, DataType dtype, std::vector<std::int64_t> shape,
                         const void* src, std::size_t src_bytes) {
  Tensor tensor = zeros(device, dtype, std::move(shape));
  if (src == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "host source pointer is null");
  }
  if (src_bytes < tensor.nbytes()) {
    throw VieError(ErrorCode::InvalidArgument, "host source is smaller than the tensor");
  }
  if (tensor.is_cpu()) {
    std::memcpy(tensor.data(), src, tensor.nbytes());
    return tensor;
  }
  copy_device_memory(device.execution_state().get(), DeviceType::Cpu, 0,
                     static_cast<const std::byte*>(src), DeviceType::Cuda, tensor.device_id(),
                     tensor.device_data(), tensor.nbytes());
  return tensor;
}

void Tensor::require_bound() const {
  if (storage_ == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "tensor has no storage");
  }
}

std::size_t Tensor::numel() const {
  if (storage_ == nullptr) return 0;
  return checked_numel(shape_);
}

std::size_t Tensor::nbytes() const {
  if (storage_ == nullptr) return 0;
  return checked_mul(numel(), dtype_size(dtype_));
}

std::size_t Tensor::storage_bytes() const {
  if (storage_ == nullptr) return 0;
  return storage_->size_bytes();
}

bool Tensor::contiguous() const {
  if (storage_ == nullptr) return true;
  if (shape_.size() != strides_.size()) return false;
  std::int64_t expected = 1;
  for (std::size_t i = shape_.size(); i-- > 0;) {
    if (strides_[i] != expected) return false;
    if (shape_[i] != 0 && expected > std::numeric_limits<std::int64_t>::max() / shape_[i]) {
      return false;
    }
    expected *= shape_[i];
  }
  return true;
}

std::size_t Tensor::span_bytes() const {
  require_bound();
  if (dtype_ == DataType::Unknown) {
    throw VieError(ErrorCode::InvalidArgument, "tensor dtype is unknown");
  }
  if (shape_.size() != strides_.size()) {
    throw VieError(ErrorCode::InvalidArgument, "shape and strides have different ranks");
  }
  const std::size_t elem = dtype_size(dtype_);
  if (elem == 0 || byte_offset_ % elem != 0) {
    throw VieError(ErrorCode::InvalidArgument, "byte offset is not aligned to the dtype");
  }
  std::size_t max_index = 0;
  for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
    if (shape_[axis] <= 0) {
      throw VieError(ErrorCode::InvalidArgument, "view shape must be concrete");
    }
    if (strides_[axis] < 0) {
      throw VieError(ErrorCode::InvalidArgument, "negative strides are not supported");
    }
    const std::size_t span = static_cast<std::size_t>(shape_[axis] - 1);
    const std::size_t stride = static_cast<std::size_t>(strides_[axis]);
    max_index = checked_add(max_index, checked_mul(span, stride));
  }
  const std::size_t bytes = checked_mul(checked_add(max_index, 1), elem);
  return checked_add(byte_offset_, bytes);
}

std::byte* Tensor::data() {
  require_bound();
  if (device_ != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported,
                   "data() returns a host pointer and cannot be used on a " +
                       device_text(device_, device_id_) + " tensor");
  }
  return storage_->data() + byte_offset_;
}

const std::byte* Tensor::data() const { return const_cast<Tensor*>(this)->data(); }

std::byte* Tensor::device_data() {
  require_bound();
  if (device_ != DeviceType::Cuda) {
    throw VieError(ErrorCode::InvalidArgument,
                   "device_data() requires a CUDA tensor, got " + device_text(device_, device_id_));
  }
  const auto address = reinterpret_cast<std::uintptr_t>(storage_->data()) + byte_offset_;
  return reinterpret_cast<std::byte*>(address);
}

const std::byte* Tensor::device_data() const { return const_cast<Tensor*>(this)->device_data(); }

Tensor Tensor::view(std::vector<std::int64_t> shape, std::vector<std::int64_t> strides,
                    std::size_t byte_offset) const {
  require_bound();
  Tensor viewed(storage_, dtype_, std::move(shape), std::move(strides), byte_offset, device_,
                device_id_);
  viewed.layout_ = layout_;
  const std::size_t extent = viewed.span_bytes();
  if (extent > storage_->size_bytes()) {
    throw VieError(ErrorCode::InvalidArgument, "tensor view extends past its storage");
  }
  return viewed;
}

Tensor Tensor::clone() const {
  require_bound();
  const std::size_t elem = dtype_size(dtype_);
  const auto strides = row_major_strides(shape_);
  std::shared_ptr<TensorStorage> storage;
  if (device_ == DeviceType::Cpu) {
    storage = TensorStorage::allocate_host(nbytes(), device_id_);
  } else if (device_ == DeviceType::Cuda) {
    storage = TensorStorage::allocate_cuda(device_id_, nbytes(), nullptr);
  } else {
    throw VieError(ErrorCode::Unsupported, "clone supports CPU and CUDA tensors only");
  }
  Tensor copy(std::move(storage), dtype_, shape_, strides, 0, device_, device_id_);
  copy.layout_ = layout_;
  if (device_ == DeviceType::Cpu) {
    if (contiguous()) {
      std::memcpy(copy.data(), data(), nbytes());
      return copy;
    }
    gather_strided(data(), shape_, strides_, elem, copy.data());
    return copy;
  }

  if (contiguous()) {
    copy_device_memory(nullptr, DeviceType::Cuda, device_id_, device_data(), DeviceType::Cuda,
                       device_id_, copy.device_data(), nbytes());
    return copy;
  }
  const std::size_t region = span_bytes() - byte_offset_;
  std::vector<std::byte> staged(region);
  copy_device_memory(nullptr, DeviceType::Cuda, device_id_, device_data(), DeviceType::Cpu, 0,
                     staged.data(), region);
  std::vector<std::byte> packed(nbytes());
  gather_strided(staged.data(), shape_, strides_, elem, packed.data());
  copy_device_memory(nullptr, DeviceType::Cpu, 0, packed.data(), DeviceType::Cuda, device_id_,
                     copy.device_data(), nbytes());
  return copy;
}

Tensor Tensor::to(const DeviceContext& destination) const {
  require_bound();
  if (destination.device() == device_ && destination.device_id() == device_id_) return clone();
  if (device_ == DeviceType::Cuda && destination.device() == DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported, "copy from " + device_text(device_, device_id_) +
                                               " to " +
                                               device_text(destination.device(), destination.device_id()) +
                                               " is not supported");
  }
  Tensor result = zeros(destination, dtype_, shape_);
  result.layout_ = layout_;
  if (contiguous()) {
    result.copy_from(*this);
  } else {
    result.copy_from(clone());
  }
  return result;
}

void Tensor::copy_from(const Tensor& source) {
  require_bound();
  source.require_bound();
  if (dtype_ != source.dtype_) {
    throw VieError(ErrorCode::InvalidArgument, "copy from " + device_text(source.device_, source.device_id_) +
                                                   " to " + device_text(device_, device_id_) +
                                                   " failed: dtype mismatch");
  }
  if (shape_ != source.shape_) {
    throw VieError(ErrorCode::InvalidArgument, "copy from " + device_text(source.device_, source.device_id_) +
                                                   " to " + device_text(device_, device_id_) +
                                                   " failed: shape mismatch");
  }
  if (!contiguous() || !source.contiguous()) {
    throw VieError(ErrorCode::InvalidArgument, "copy_from requires contiguous tensors");
  }
  auto fits = [](const Tensor& tensor) {
    return tensor.byte_offset_ <= tensor.storage_->size_bytes() &&
           tensor.nbytes() <= tensor.storage_->size_bytes() - tensor.byte_offset_;
  };
  if (!fits(*this) || !fits(source)) {
    throw VieError(ErrorCode::InvalidArgument, "copy from " + device_text(source.device_, source.device_id_) +
                                                   " to " + device_text(device_, device_id_) +
                                                   " failed: copy exceeds storage");
  }
  const std::byte* src = source.is_cpu() ? source.data() : source.device_data();
  std::byte* dst = is_cpu() ? data() : device_data();
  if (is_cpu() && source.is_cpu()) {
    std::memcpy(dst, src, nbytes());
    return;
  }
  copy_device_memory(nullptr, source.device_, source.device_id_, src, device_, device_id_, dst,
                     nbytes());
}

bool Tensor::shares_storage(const Tensor& other) const noexcept {
  return storage_ != nullptr && storage_ == other.storage_;
}

}  // namespace vie
