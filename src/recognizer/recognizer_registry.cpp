#include "vie/recognizer/recognizer_registry.hpp"

#include "vie/recognizer/classification_recognizer.hpp"
#include "vie/recognizer/rtdetr_recognizer.hpp"
#include "vie/recognizer/yolov8_detect_recognizer.hpp"
#include "vie/recognizer/yolov8_obb_recognizer.hpp"
#include "vie/recognizer/yolov8_seg_recognizer.hpp"

#include <algorithm>
#include <utility>

namespace vie {
namespace {

int evidence_count(const RecognitionCandidate& candidate) {
  return static_cast<int>(candidate.evidence.size());
}

}  // namespace

void ModelRecognizerRegistry::add(std::unique_ptr<IModelRecognizer> recognizer) {
  recognizers_.push_back(std::move(recognizer));
}

RecognitionResult ModelRecognizerRegistry::recognize(const ModelAnalysisReport& report) const {
  std::vector<RecognitionCandidate> matched;
  for (const std::unique_ptr<IModelRecognizer>& recognizer : recognizers_) {
    if (recognizer == nullptr) continue;
    if (std::optional<RecognitionCandidate> candidate = recognizer->consider(report)) {
      matched.push_back(std::move(*candidate));
    }
  }
  RecognitionResult result;
  if (matched.empty()) {
    result.status = RecognitionStatus::Unknown;
    result.explanation = "no recognizer accepted this graph";
    return result;
  }
  std::stable_sort(matched.begin(), matched.end(),
                   [](const RecognitionCandidate& left, const RecognitionCandidate& right) {
                     return evidence_count(left) > evidence_count(right);
                   });
  const int best = evidence_count(matched.front());
  int tied = 0;
  for (const RecognitionCandidate& candidate : matched) {
    if (evidence_count(candidate) == best) ++tied;
  }
  bool saw_detection = false;
  bool saw_segmentation = false;
  bool saw_rtdetr = false;
  bool saw_classification = false;
  bool saw_obb = false;
  for (const RecognitionCandidate& candidate : matched) {
    if (candidate.architecture == "yolov8_detection") saw_detection = true;
    if (candidate.architecture == "yolov8_segmentation") saw_segmentation = true;
    if (candidate.architecture == "rtdetr_detection") saw_rtdetr = true;
    if (candidate.architecture == "single_label_classification") saw_classification = true;
    if (candidate.architecture == "yolov8_obb") saw_obb = true;
  }
  result.candidates = std::move(matched);
  if (saw_detection && saw_segmentation) {
    result.status = RecognitionStatus::Ambiguous;
    result.explanation =
        "YOLOv8 detection and segmentation recognizers both accepted this graph";
    return result;
  }
  if (saw_rtdetr && (saw_detection || saw_segmentation)) {
    result.status = RecognitionStatus::Ambiguous;
    result.explanation = "RT-DETR and YOLOv8 recognizers both accepted this graph";
    return result;
  }
  if (saw_classification && (saw_detection || saw_segmentation || saw_rtdetr || saw_obb)) {
    result.status = RecognitionStatus::Ambiguous;
    result.explanation = "classification and another task recognizer both accepted this graph";
    return result;
  }
  if (saw_obb && (saw_detection || saw_segmentation || saw_rtdetr)) {
    result.status = RecognitionStatus::Ambiguous;
    result.explanation = "YOLOv8 OBB and another box recognizer both accepted this graph";
    return result;
  }
  if (tied > 1) {
    result.status = RecognitionStatus::Ambiguous;
    result.explanation =
        "more than one candidate has the same structural evidence count, so none is selected";
    return result;
  }
  result.status = RecognitionStatus::Recognized;
  result.explanation = "one candidate has strictly more structural evidence than the others";
  return result;
}

ModelRecognizerRegistry default_recognizer_registry() {
  ModelRecognizerRegistry registry;
  registry.add(std::make_unique<YoloV8DetectRecognizer>());
  registry.add(std::make_unique<YoloV8ObbRecognizer>());
  registry.add(std::make_unique<YoloV8SegRecognizer>());
  registry.add(std::make_unique<RtdetrRecognizer>());
  registry.add(std::make_unique<ClassificationRecognizer>());
  return registry;
}

}  // namespace vie
