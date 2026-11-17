#pragma once

#include "vie/preprocess/preprocess.hpp"

namespace vie {

PreprocessResult preprocess_cuda(const PreprocessConfig& config, const ImageInput& image,
                                 const DeviceContext& device, const GeometryTransform& geometry);

}  // namespace vie
