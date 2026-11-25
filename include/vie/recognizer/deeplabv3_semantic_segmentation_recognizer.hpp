#pragma once

#include "vie/recognizer/model_recognizer.hpp"

namespace vie {

// Accepts a dense semantic segmentation graph: one fp32 NCHW image input
// [N, 3, H, W] and one fp32 rank-4 output [N, C, H_out, W_out] whose channel
// axis 1 equals the output channels of a terminal 1x1 Conv, with only Resize /
// passthrough ops between that Conv and the graph output. Softmax, ArgMax, or a
// global-pool classification head in that path disqualify the graph, so raw
// per-pixel class logits are required. Instance segmentation (two outputs) and
// single-label classification (rank-2 output) are declined.
class Deeplabv3SemanticSegmentationRecognizer : public IModelRecognizer {
 public:
  std::string id() const override;
  std::optional<RecognitionCandidate> consider(const ModelAnalysisReport& report) const override;
};

}  // namespace vie
