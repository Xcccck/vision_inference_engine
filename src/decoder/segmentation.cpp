#include "vie/decoder/segmentation.hpp"

#include <cmath>
#include <limits>

namespace vie {
namespace {

std::size_t pixel_count(int width, int height) {
  if (width < 1 || height < 1) {
    throw VieError(ErrorCode::InvalidArgument, "mask width and height must be positive");
  }
  const std::size_t wide = static_cast<std::size_t>(width);
  const std::size_t high = static_cast<std::size_t>(height);
  if (wide > std::numeric_limits<std::size_t>::max() / high) {
    throw VieError(ErrorCode::InvalidArgument, "mask width times height overflows");
  }
  return wide * high;
}

void require_binary_pixels(const std::uint8_t* pixels, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (pixels[i] > 1) {
      throw VieError(ErrorCode::InvalidArgument, "binary mask values must be 0 or 1");
    }
  }
}

void require_probability_pixels(const float* pixels, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(pixels[i]) || pixels[i] < 0.f || pixels[i] > 1.f) {
      throw VieError(ErrorCode::InvalidArgument,
                     "probability mask values must be finite and inside [0, 1]; logits are not probabilities");
    }
  }
}

}  // namespace

Mask::Mask(MaskKind kind, Tensor tensor) : kind_(kind), tensor_(std::move(tensor)) {}

Mask Mask::binary(const DeviceContext& device, int width, int height) {
  pixel_count(width, height);
  return Mask(MaskKind::Binary,
              Tensor::zeros(device, DataType::Uint8, {static_cast<std::int64_t>(height),
                                                      static_cast<std::int64_t>(width)}));
}

Mask Mask::binary(const DeviceContext& device, int width, int height, const std::uint8_t* pixels,
                  std::size_t count) {
  if (count != pixel_count(width, height)) {
    throw VieError(ErrorCode::InvalidArgument, "mask pixel count does not match width times height");
  }
  if (pixels == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "binary mask pixels are null");
  }
  require_binary_pixels(pixels, count);
  return Mask(MaskKind::Binary,
              Tensor::from_host(device, DataType::Uint8,
                                {static_cast<std::int64_t>(height), static_cast<std::int64_t>(width)}, pixels,
                                count));
}

Mask Mask::probability(const DeviceContext& device, int width, int height) {
  pixel_count(width, height);
  return Mask(MaskKind::Probability,
              Tensor::zeros(device, DataType::Fp32, {static_cast<std::int64_t>(height),
                                                     static_cast<std::int64_t>(width)}));
}

Mask Mask::probability(const DeviceContext& device, int width, int height, const float* pixels,
                       std::size_t count) {
  if (count != pixel_count(width, height)) {
    throw VieError(ErrorCode::InvalidArgument, "mask pixel count does not match width times height");
  }
  if (pixels == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "probability mask pixels are null");
  }
  require_probability_pixels(pixels, count);
  return Mask(MaskKind::Probability,
              Tensor::from_host(device, DataType::Fp32,
                                {static_cast<std::int64_t>(height), static_cast<std::int64_t>(width)}, pixels,
                                count * sizeof(float)));
}

Mask Mask::from_tensor(MaskKind kind, Tensor tensor) {
  if (tensor.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "mask tensor has no storage");
  }
  if (tensor.rank() != 2) {
    throw VieError(ErrorCode::InvalidArgument, "mask shape must be height by width");
  }
  if (!tensor.contiguous()) {
    throw VieError(ErrorCode::Unsupported, "mask does not read a non-contiguous tensor");
  }
  const std::int64_t height = tensor.shape()[0];
  const std::int64_t width = tensor.shape()[1];
  if (height < 1 || width < 1 || height > std::numeric_limits<int>::max() ||
      width > std::numeric_limits<int>::max()) {
    throw VieError(ErrorCode::InvalidArgument, "mask width and height must be positive");
  }
  const std::size_t count = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  if (kind == MaskKind::Binary) {
    if (tensor.dtype() != DataType::Uint8) {
      throw VieError(ErrorCode::InvalidArgument, "binary mask requires uint8");
    }
    if (tensor.is_cpu()) require_binary_pixels(tensor.data_as<std::uint8_t>(), count);
  } else if (kind == MaskKind::Probability) {
    if (tensor.dtype() != DataType::Fp32) {
      throw VieError(ErrorCode::InvalidArgument, "probability mask requires fp32");
    }
    if (tensor.is_cpu()) require_probability_pixels(tensor.data_as<float>(), count);
  } else {
    throw VieError(ErrorCode::InvalidArgument, "mask kind is not binary or probability");
  }
  if (tensor.device() != DeviceType::Cpu && tensor.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported, "mask storage supports CPU and CUDA only");
  }
  return Mask(kind, std::move(tensor));
}

int Mask::width() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  return static_cast<int>(tensor_.shape()[1]);
}

int Mask::height() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  return static_cast<int>(tensor_.shape()[0]);
}

MaskKind Mask::kind() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  return kind_;
}

MaskLayout Mask::layout() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  return MaskLayout::Hw;
}

DataType Mask::dtype() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  return tensor_.dtype();
}

DeviceType Mask::device() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  return tensor_.device();
}

int Mask::device_id() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  return tensor_.device_id();
}

const std::vector<std::int64_t>& Mask::shape() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  return tensor_.shape();
}

const Tensor& Mask::tensor() const { return tensor_; }

const std::uint8_t* Mask::binary_data() const {
  if (kind() != MaskKind::Binary) {
    throw VieError(ErrorCode::InvalidArgument, "binary_data requires a binary mask");
  }
  return tensor_.data_as<std::uint8_t>();
}

const float* Mask::probability_data() const {
  if (kind() != MaskKind::Probability) {
    throw VieError(ErrorCode::InvalidArgument, "probability_data requires a probability mask");
  }
  return tensor_.data_as<float>();
}

Mask Mask::to_cpu() const {
  if (tensor_.empty()) throw VieError(ErrorCode::InvalidArgument, "mask has no storage");
  if (tensor_.is_cpu()) return Mask(kind_, tensor_.clone());
  return Mask::from_tensor(kind_, tensor_.to(DeviceContext{}));
}

}  // namespace vie
