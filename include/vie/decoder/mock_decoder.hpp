#pragma once

#include "vie/decoder/model_decoder.hpp"

namespace vie {

// Test-double protocol "mock". Reads six fp32 values from the first declared
// output, by name: x1, y1, x2, y2, score, class_id. Shorter tensors decode
// to an empty detection list.
class MockDecoder : public IModelDecoder {
 public:
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
};

}  // namespace vie
