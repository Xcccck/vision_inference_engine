#pragma once

#include "vie/recognizer/model_recognizer.hpp"

namespace vie {

// Ultralytics YOLOv8-OBB, one decoded output. The graph must prove DFL,
// dist2rbox (Cos or Sin on the box path), a class Sigmoid, and the angle
// affine (sigmoid - 1/4) * pi. A rank-3 tensor of the right width is not enough.
class YoloV8ObbRecognizer : public IModelRecognizer {
 public:
  std::string id() const override;
  std::optional<RecognitionCandidate> consider(const ModelAnalysisReport& report) const override;
};

}  // namespace vie
