#pragma once

#include "vie/recognizer/model_recognizer.hpp"

namespace vie {

// Recognizes an exported YOLOv8 segmentation graph: one decoded prediction
// tensor (cxcywh, sigmoid class scores, raw mask coefficients) and one
// prototype feature tensor. A matching shape without that graph is declined.
class YoloV8SegRecognizer : public IModelRecognizer {
 public:
  std::string id() const override;
  std::optional<RecognitionCandidate> consider(const ModelAnalysisReport& report) const override;
};

}  // namespace vie
