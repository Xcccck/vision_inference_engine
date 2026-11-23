#pragma once

#include "vie/decoder/model_decoder.hpp"
#include "vie/decoder/obb.hpp"

namespace vie {

// Protocol yolov8_obb_raw. The exported graph already applied DFL, dist2rbox,
// the class Sigmoid, and angle = (sigmoid - 0.25) * pi. This decoder reads
// cx, cy, width, height, class probabilities, and that angle. It does not run
// DFL, Sigmoid, or the angle affine again.
class YoloV8ObbDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs, const DecoderContext& context) const override;
  ObbBatch decode_obb(const DecoderInput& input, const ObbDecodeConfig& config,
                      const DeviceContext& device) const override;
};

}  // namespace vie
