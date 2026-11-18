#pragma once

#include "vie/recognizer/model_recognizer.hpp"

namespace vie {

// Accepts one float input and one rank-2 float output when that output is the
// logits of a Gemm or MatMul whose data input comes from a global pool.
// Embedding, regression, softmax, sigmoid, and extra outputs are declined.
class ClassificationRecognizer : public IModelRecognizer {
 public:
  std::string id() const override;
  std::optional<RecognitionCandidate> consider(const ModelAnalysisReport& report) const override;
};

}  // namespace vie
