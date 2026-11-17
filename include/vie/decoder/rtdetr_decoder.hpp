#pragma once

#include "vie/decoder/model_decoder.hpp"

#include <string>

namespace vie {

// Shared by the CPU and CUDA decoders so a non-finite value names the same
// query and the same field on both devices. field is one of the CPU check
// names: "normalized cx", "normalized cy", "normalized width",
// "normalized height", or "class score".
inline std::string rtdetr_nonfinite_message(const char* field, int query) {
  return std::string(field) + " is not finite at query " + std::to_string(query);
}

// Protocol rtdetr_detection_raw. The tensor is [batch, query, 4 + class]
// normalized cxcywh followed by probabilities. This decoder does not run
// Sigmoid, Softmax, objectness, a second query selection, or NMS. It stays
// on CPU. The CUDA implementation is RtdetrCudaDecoder.
class RtdetrDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  DetectionBatch decode_batch(const DecoderInput& input, const DecoderConfig& config,
                              const DeviceContext& device) const override;
};

// Checks the RT-DETR fields only. It does not require reg_max, pre_nms_top_k,
// or an IoU threshold. YOLOv8 validation is unchanged.
void validate_rtdetr_decoder_config(const DecoderConfig& config);

}  // namespace vie
