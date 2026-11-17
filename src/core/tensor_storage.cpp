#include "vie/core/tensor_storage.hpp"

#include <cstring>
#include <new>

namespace vie {
namespace {

void release_host(std::byte* data, int, std::size_t) noexcept {
  if (data != nullptr) {
    ::operator delete(data, std::align_val_t{TensorStorage::kAlignment});
  }
}

}  // namespace

TensorStorage::TensorStorage(std::byte* data, std::size_t bytes, DeviceType device, int device_id,
                             ReleaseFn release)
    : data_(data), size_bytes_(bytes), device_(device), device_id_(device_id), release_(release) {}

TensorStorage::~TensorStorage() {
  if (release_ != nullptr) release_(data_, device_id_, size_bytes_);
  data_ = nullptr;
  release_ = nullptr;
}

std::shared_ptr<TensorStorage> TensorStorage::allocate_host(std::size_t bytes, int device_id) {
  if (device_id < 0) {
    throw VieError(ErrorCode::InvalidArgument, "device id must be >= 0");
  }
  std::byte* data = nullptr;
  if (bytes > 0) {
    data = static_cast<std::byte*>(::operator new(bytes, std::align_val_t{kAlignment}));
    std::memset(data, 0, bytes);
  }
  try {
    return std::shared_ptr<TensorStorage>(
        new TensorStorage(data, bytes, DeviceType::Cpu, device_id, release_host));
  } catch (...) {
    release_host(data, device_id, bytes);
    throw;
  }
}

}  // namespace vie
