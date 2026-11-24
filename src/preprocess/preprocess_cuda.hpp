#pragma once

#include "vie/preprocess/preprocess.hpp"

#include <cstddef>

namespace vie {

PreprocessResult preprocess_cuda(const PreprocessConfig& config, const ImageInput& image,
                                 const DeviceContext& device, const GeometryTransform& geometry);

// Phase 9.1: launches the preprocess kernel once, writing one image into a
// caller-owned device buffer laid out as [1, 3, H, W] (NCHW) or [1, H, W, 3]
// (NHWC). Does not synchronize; the caller owns timing and sync. Used by the
// batch path to launch N kernels back-to-back and synchronize once.
void preprocess_cuda_launch(const PreprocessConfig& config, const ImageInput& image,
                            const GeometryTransform& geometry, std::byte* device_output_base);

// Phase 9.1: allocates a single [N, ...] CUDA Tensor and launches one
// preprocess kernel per image, synchronizing exactly once at the end.
// `images` must be non-empty; every image must already live on the target
// device (H2D cost stays outside this function). Returns the batch Tensor,
// per-image GeometryTransform list (in input order), and the total kernel
// elapsed_ms measured between the first launch and the final sync.
struct CudaBatchPreprocess {
  Tensor output;
  std::vector<GeometryTransform> geometries;
  double device_ms = 0;
};
CudaBatchPreprocess preprocess_cuda_batch(const PreprocessConfig& config,
                                          const std::vector<ImageInput>& images,
                                          const DeviceContext& device);

}  // namespace vie
