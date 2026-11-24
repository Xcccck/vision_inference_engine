#pragma once

#include "vie/preprocess/preprocess.hpp"

namespace vie {

PreprocessResult preprocess_cpu(const PreprocessConfig& config, const ImageInput& image,
                                const DeviceContext& device, const GeometryTransform& geometry);

// Phase 9.1: writes one preprocessed image into a caller-owned buffer laid
// out as [1, 3, H, W] (NCHW) or [1, H, W, 3] (NHWC), selected by
// config.output_layout. Used by the batch path to fill one contiguous slice
// of a larger [N, ...] Tensor without allocating a per-image intermediate.
void preprocess_cpu_into(const PreprocessConfig& config, const ImageInput& image,
                         const GeometryTransform& geometry, void* output_base);

}  // namespace vie
