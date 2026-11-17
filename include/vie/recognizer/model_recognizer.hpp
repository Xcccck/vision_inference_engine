#pragma once

#include "vie/analyzer/analysis_report.hpp"
#include "vie/recognizer/recognition.hpp"

#include <optional>
#include <string>

namespace vie {

// A recognizer either declines or returns one explained candidate.
// Filename and a bare tensor shape are not sufficient grounds to accept.
class IModelRecognizer {
 public:
  virtual ~IModelRecognizer() = default;
  virtual std::string id() const = 0;
  virtual std::optional<RecognitionCandidate> consider(const ModelAnalysisReport& report) const = 0;
};

}  // namespace vie
