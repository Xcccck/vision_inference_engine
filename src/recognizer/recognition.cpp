#include "vie/recognizer/recognition.hpp"

#include <sstream>

namespace vie {

const RecognitionCandidate* RecognitionResult::selected() const {
  if (status != RecognitionStatus::Recognized || candidates.empty()) return nullptr;
  return &candidates.front();
}

const char* to_string(RecognitionStatus status) {
  switch (status) {
    case RecognitionStatus::Recognized:
      return "recognized";
    case RecognitionStatus::Unknown:
      return "unknown";
    case RecognitionStatus::Ambiguous:
      return "ambiguous";
  }
  return "unknown";
}

std::string recognition_to_text(const RecognitionResult& result) {
  std::ostringstream out;
  out << "recognition: " << to_string(result.status) << '\n';
  out << result.explanation << '\n';
  if (result.candidates.empty()) {
    out << "candidates: (none)\n";
    return out.str();
  }
  out << "candidates:\n";
  for (const RecognitionCandidate& candidate : result.candidates) {
    out << "  " << candidate.architecture << "  recognizer " << candidate.recognizer_id
        << "  evidence " << candidate.evidence.size() << '\n';
    out << "  decoder " << candidate.decoder.protocol;
    if (!candidate.decoder.output_name.empty()) out << "  output " << candidate.decoder.output_name;
    if (!candidate.decoder.box_format.empty()) out << "  box " << candidate.decoder.box_format;
    if (candidate.decoder.num_classes.has_value()) {
      out << "  classes " << *candidate.decoder.num_classes;
    }
    out << '\n';
    for (const RecognitionEvidence& evidence : candidate.evidence) {
      out << "    " << evidence.code << ": " << evidence.detail << '\n';
    }
  }
  return out.str();
}

}  // namespace vie
