#pragma once

#include "vie/decoder/model_decoder.hpp"

namespace vie {

// Protocol yolov8_detect_raw. The recognizer's evidence says the exported
// tensor is already cxcywh in model-input pixels followed by sigmoid class
// scores. DFL runs only when DecoderConfig::box_decode is Dfl and the caller
// supplies reg_max plus feature levels. This decoder does not run the model.
class YoloV8DetectionDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  DetectionBatch decode_batch(const DecoderInput& input, const DecoderConfig& config,
                              const DeviceContext& device) const override;
};

}  // namespace vie
