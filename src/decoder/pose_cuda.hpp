#pragma once

#include "vie/decoder/pose.hpp"

#include <cstddef>
#include <vector>

namespace vie {

// Result of one GPU pose pass. The caller has already checked the tensor
// contract (device, dtype, shape, axis, channel count). objects holds exactly
// the NMS survivors in the same order and with the same field semantics as the
// CPU decoder. validate_ms covers candidate filtering, box/keypoint decode,
// and the score sort; pose_decode_ms covers the inverse map and the
// pre-NMS top-k cut; nms_ms covers greedy NMS (or the prefix cut when NMS is
// off); d2h_ms covers the compact gather plus copy; d2h_bytes is exactly the
// copied object and keypoint payload. Fault counters, the raw candidate count,
// the mapped count, and the kept count are control transfers and are not
// counted in d2h_bytes.
struct PoseCudaLaunch {
  std::vector<PoseObject> objects;
  int candidates_after_filter = 0;
  int candidates_before_nms = 0;
  double validate_ms = 0;
  double pose_decode_ms = 0;
  double nms_ms = 0;
  double d2h_ms = 0;
  std::size_t d2h_bytes = 0;
};

// Reads the same channel layout as the CPU decoder: cx, cy, width, height,
// num_classes scores, then keypoint_count groups of (x, y, confidence).
// Throws InvalidArgument with the CPU decoder's wording for a non-finite
// value, an out-of-range probability, a non-positive kept box, or an
// invalid geometry, before any result is built.
PoseCudaLaunch launch_pose_cuda(const Tensor& tensor, const PoseDecodeConfig& config,
                                const GeometryTransform& geometry, int device_id);

}  // namespace vie