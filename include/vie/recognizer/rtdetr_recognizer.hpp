#pragma once

#include "vie/recognizer/model_recognizer.hpp"

namespace vie {

// Ultralytics RT-DETR detection head without in-graph NMS. The output is one
// rank-3 tensor: a final Concat of a sigmoid-refined normalized box and a
// sigmoid class vector. Query selection is the single TopK that feeds both
// linears. The output protocol is rtdetr_detection_raw, and no decoder is
// registered for it in this phase.
class RtdetrRecognizer : public IModelRecognizer {
 public:
  std::string id() const override;
  std::optional<RecognitionCandidate> consider(const ModelAnalysisReport& report) const override;
};

}  // namespace vie
