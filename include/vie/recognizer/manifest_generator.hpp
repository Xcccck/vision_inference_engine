#pragma once

#include "vie/analyzer/analysis_report.hpp"
#include "vie/model/manifest.hpp"
#include "vie/recognizer/recognition.hpp"

#include <string>

namespace vie {

// Builds a schema 1.0 manifest from an analysis report and a recognition
// result. Inferred fields keep Inferred provenance. Color, normalization,
// and resize are left unresolved.
class ManifestGenerator {
 public:
  ModelManifest generate(const ModelAnalysisReport& report, const RecognitionResult& recognition,
                         const std::string& model_path) const;
};

}  // namespace vie
