#pragma once

#include "vie/preprocess/preprocess.hpp"

namespace vie {

PreprocessResult preprocess_cpu(const PreprocessConfig& config, const ImageInput& image,
                                const DeviceContext& device, const GeometryTransform& geometry);

}  // namespace vie
