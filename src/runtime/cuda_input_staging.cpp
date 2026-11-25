#include "vie/runtime/cuda_input_staging.hpp"

#include "vie/core/error.hpp"
#include "vie/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace vie {
namespace {

std::size_t mul_checked(std::size_t lhs, std::size_t rhs, const char* what) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw VieError(ErrorCode::InvalidArgument,
                   std::string("cuda input staging overflow computing ") + what);
  }
  return lhs * rhs;
}

}  // namespace

std::size_t cuda_stage_byte_count(const ImageInput& image) {
  if (image.data == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "cuda input staging requires a non-null image buffer");
  }
  if (image.width <= 0 || image.height <= 0 || image.channels <= 0) {
    throw VieError(ErrorCode::InvalidArgument,
                   "cuda input staging requires positive width, height and channels");
  }
  const std::size_t elem = dtype_size(image.dtype);
  if (elem == 0) {
    throw VieError(ErrorCode::InvalidArgument, "cuda input staging image dtype has no size");
  }
  // A row is either an explicit stride or a tightly packed width*channels*elem.
  const std::size_t tight_row =
      mul_checked(mul_checked(static_cast<std::size_t>(image.width),
                              static_cast<std::size_t>(image.channels), "width*channels"),
                  elem, "channels*dtype_size");
  std::size_t row_bytes = image.row_stride_bytes != 0 ? image.row_stride_bytes : tight_row;
  if (row_bytes < tight_row) {
    throw VieError(ErrorCode::InvalidArgument,
                   "cuda input staging row stride is smaller than one packed row");
  }
  return mul_checked(row_bytes, static_cast<std::size_t>(image.height), "row_bytes*height");
}

CudaInputBuffer CudaInputBuffer::stage(const ImageInput& cpu_image, const DeviceContext& device) {
  // Only ever stages a Runtime-owned CPU image onto a CUDA execution device.
  if (cpu_image.device != DeviceType::Cpu) {
    throw VieError(ErrorCode::InvalidArgument,
                   "cuda input staging source must be a CPU image");
  }
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::InvalidArgument,
                   "cuda input staging target execution device is not CUDA");
  }

  // Validates dimensions/stride and computes the exact transfer span, reusing
  // the same row layout the request deep copy produced (see InferenceRequest).
  const std::size_t total_bytes = cuda_stage_byte_count(cpu_image);
  const std::size_t elem = dtype_size(cpu_image.dtype);
  const std::size_t tight_row = static_cast<std::size_t>(cpu_image.width) *
                                static_cast<std::size_t>(cpu_image.channels) * elem;
  const std::size_t row_bytes =
      cpu_image.row_stride_bytes != 0 ? cpu_image.row_stride_bytes : tight_row;

  // Tensor::from_host allocates a cudaMalloc buffer, issues a stream-ordered
  // H2D on the device's stream, and synchronizes before returning, so the bytes
  // are on device (and the CPU source is safe to release/reuse) once it returns.
  // The returned Tensor owns the device memory via shared_ptr<TensorStorage>.
  Tensor buffer = Tensor::from_host(device, DataType::Uint8,
                                    {static_cast<std::int64_t>(cpu_image.height),
                                     static_cast<std::int64_t>(row_bytes)},
                                    cpu_image.data, total_bytes);

  ImageInput staged = cpu_image;
  staged.data = buffer.device_data();
  staged.device = DeviceType::Cuda;
  staged.device_id = device.device_id();
  staged.row_stride_bytes = row_bytes;

  return CudaInputBuffer(std::move(buffer), staged);
}

}  // namespace vie
