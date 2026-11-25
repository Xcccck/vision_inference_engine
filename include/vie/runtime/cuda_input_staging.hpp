#pragma once

#include "vie/core/device_context.hpp"
#include "vie/core/tensor.hpp"
#include "vie/preprocess/image_input.hpp"

#include <cstddef>

namespace vie {

// Computes how many bytes a CPU ImageInput occupies on a tightly-packed,
// per-row-strided layout, validating the descriptor first. CUDA-free and
// allocation-free so the size/overflow rules are unit-testable in CPU-only
// builds (Build A/B). Throws VieError(InvalidArgument) on a malformed image
// (null data, non-positive dimensions, unsupported dtype, or an integer
// overflow while computing the byte span). This is the single source of truth
// for the H2D transfer size used by CudaInputBuffer::stage().
std::size_t cuda_stage_byte_count(const ImageInput& image);

// Phase 9.3.2: RAII owner of a Runtime-staged CPU->CUDA input pixel buffer for
// exactly one asynchronous execution. It never wraps a caller-supplied device
// pointer; it only ever takes a Runtime-OWNED CPU image (deep-copied at submit
// time) and uploads it to the engine's committed CUDA device. The device memory
// is a TensorStorage cudaMalloc allocation released with cudaFree when this
// object (and the Tensor it holds) is destroyed, so the buffer lives at least as
// long as the staging -> preprocess -> backend -> decoder -> D2H call it feeds
// and is freed on every path, including exceptions. Move-only; the raw address
// is never handed to callers as an owning handle.
//
// The public header deliberately pulls in no CUDA headers: it is built on
// Tensor / DeviceContext / ImageInput only. In a CPU-only build the engine can
// never report a CUDA device, so stage() is unreachable and nothing here needs
// the CUDA runtime to compile or link.
class CudaInputBuffer {
 public:
  CudaInputBuffer() = default;

  CudaInputBuffer(const CudaInputBuffer&) = delete;
  CudaInputBuffer& operator=(const CudaInputBuffer&) = delete;
  CudaInputBuffer(CudaInputBuffer&&) noexcept = default;
  CudaInputBuffer& operator=(CudaInputBuffer&&) noexcept = default;

  // Uploads a Runtime-owned CPU image to `device` (which must be a CUDA device)
  // using a synchronous, stream-ordered H2D. On success the returned buffer's
  // image() has device == Cuda and a valid on-device data pointer with the same
  // width/height/channels/pixel-format/row-stride as the source. Throws
  // InvalidArgument if `cpu_image` is not a CPU image, or Unsupported if the
  // build/device cannot allocate CUDA memory. Any partial allocation is freed
  // before the exception propagates.
  static CudaInputBuffer stage(const ImageInput& cpu_image, const DeviceContext& device);

  // The staged CUDA image view. Only valid while this buffer is alive and
  // valid(); do not copy the raw pointer past this object's lifetime.
  const ImageInput& image() const noexcept { return image_; }
  bool valid() const noexcept { return !tensor_.empty(); }
  // Bytes of device memory held for this input (0 when empty).
  std::size_t device_bytes() const noexcept { return tensor_.storage_bytes(); }

 private:
  CudaInputBuffer(Tensor tensor, ImageInput image)
      : tensor_(std::move(tensor)), image_(image) {}

  Tensor tensor_;  // owns the cudaMalloc'd pixel buffer (cudaFree on dtor)
  ImageInput image_;
};

}  // namespace vie
