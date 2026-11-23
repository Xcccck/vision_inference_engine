#pragma once

#include <algorithm>
#include <vector>

namespace vie {

struct RankedScore {
  int class_id = 0;
  float score = 0.f;
};

// Sort only. The caller has already checked the numeric meaning of each score.
inline std::vector<RankedScore> rank_class_scores(const std::vector<float>& scores, int top_k) {
  std::vector<int> order(scores.size());
  for (int class_id = 0; class_id < static_cast<int>(scores.size()); ++class_id) {
    order[static_cast<std::size_t>(class_id)] = class_id;
  }
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    const float left_score = scores[static_cast<std::size_t>(left)];
    const float right_score = scores[static_cast<std::size_t>(right)];
    if (left_score != right_score) return left_score > right_score;
    return left < right;
  });
  std::vector<RankedScore> ranked;
  ranked.reserve(static_cast<std::size_t>(top_k));
  for (int rank = 0; rank < top_k; ++rank) {
    const int class_id = order[static_cast<std::size_t>(rank)];
    ranked.push_back(RankedScore{class_id, scores[static_cast<std::size_t>(class_id)]});
  }
  return ranked;
}

}  // namespace vie
