#pragma once

#include <optional>
#include <string>
#include <vector>

namespace vie {

// Structural recognition states. There is no numeric confidence: a tie is
// Ambiguous, and a graph that matches no recognizer is Unknown.
enum class RecognitionStatus {
  Recognized,
  Unknown,
  Ambiguous,
};

struct RecognitionEvidence {
  std::string code;
  std::string detail;
};

// The decoder contract a recognizer is willing to name. It is not a runtime
// decoder and does not execute the model.
struct DecoderCandidate {
  std::string protocol;
  std::string output_name;
  std::string box_format;
  std::string channel_order;
  int channel_axis = -1;
  int box_channels = 0;
  std::optional<int> num_classes;
  bool class_scores_are_probabilities = false;
  std::optional<int> reg_max;
  std::optional<int> num_queries;
  std::string prototype_name;
  std::optional<int> mask_channels;
  std::optional<int> prototype_channel_axis;
  std::optional<int> prototype_height;
  std::optional<int> prototype_width;
  std::string mask_coefficient_activation;
  std::string prototype_activation;
  std::optional<bool> has_objectness;
  // Classification only. Unset for detection and segmentation candidates.
  std::optional<int> class_axis;
  std::string score_type;
};

struct RecognitionCandidate {
  std::string recognizer_id;
  std::string architecture;
  std::string family;
  std::vector<RecognitionEvidence> evidence;
  DecoderCandidate decoder;
};

struct RecognitionResult {
  RecognitionStatus status = RecognitionStatus::Unknown;
  std::vector<RecognitionCandidate> candidates;
  std::string explanation;

  const RecognitionCandidate* selected() const;
};

const char* to_string(RecognitionStatus status);
std::string recognition_to_text(const RecognitionResult& result);

}  // namespace vie
