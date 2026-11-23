#pragma once

#include "vie/recognizer/model_recognizer.hpp"

namespace vie {

// Ultralytics YOLOv8-Pose, one decoded output and no in-graph NMS.
// The graph must prove a decoded person box, a class Sigmoid, and a keypoint
// branch whose xy path is multiplied by 2 and whose third field is a Sigmoid
// of a Slice or Split. Channel count alone is not enough, and a detection
// head is not accepted.
class YoloV8PoseRecognizer : public IModelRecognizer {
 public:
  std::string id() const override;
  std::optional<RecognitionCandidate> consider(const ModelAnalysisReport& report) const override;
};

}  // namespace vie
