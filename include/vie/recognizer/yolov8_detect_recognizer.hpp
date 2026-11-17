#pragma once

#include "vie/recognizer/model_recognizer.hpp"

namespace vie {

// Ultralytics YOLOv8 detection head without in-graph NMS:
// three scale heads, a DFL softmax/conv, dist2bbox to cxcywh, then concat
// with sigmoid class scores. The output protocol is yolov8_detect_raw.
class YoloV8DetectRecognizer : public IModelRecognizer {
 public:
  std::string id() const override;
  std::optional<RecognitionCandidate> consider(const ModelAnalysisReport& report) const override;
};

}  // namespace vie
