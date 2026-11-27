#pragma once

#include "vie/decoder/obb.hpp"

#include <cstddef>
#include <vector>

namespace vie {

// Result of one GPU OBB pass. The caller has already checked the tensor
// contract (device, dtype, shape, axis, channel count). objects holds exactly
// the NMS survivors in the same order and with the same field semantics as the
// CPU decoder.
//
// validate_ms covers the candidate validation kernel only, matching the CPU
// validate_ms; sort_ms covers the deterministic score sort; map_ms covers the
// corner transform, the minimum-area refit, and the drop compaction; nms_ms
// covers greedy rotated NMS (or the prefix cut when NMS is off) including every
// Polygon IoU evaluation; d2h_ms covers the pack and copy; d2h_bytes is exactly
// the copied object payload (32 bytes per kept object). The fault word, the
// filter count, the mapped count, the kept count, and the IoU call count are
// control transfers and are not counted in d2h_bytes.
struct ObbCudaLaunch {
  std::vector<ObbObject> objects;
  int candidates_after_filter = 0;
  int candidates_before_nms = 0;
  long long iou_calls = 0;
  double validate_ms = 0;
  double sort_ms = 0;
  double map_ms = 0;
  double nms_ms = 0;
  double d2h_ms = 0;
  std::size_t d2h_bytes = 0;
};

// Reads the same channel layout as the CPU decoder: cx, cy, width, height,
// num_classes scores, then the angle in the last channel. Throws
// InvalidArgument with the CPU decoder's wording for a non-finite value, an
// out-of-range probability, a kept box with a non-positive size or an angle
// outside [-pi/4, 3pi/4], an invalid geometry, or a failed minimum-area fit,
// before any result is built.
ObbCudaLaunch launch_obb_cuda(const Tensor& tensor, const ObbDecodeConfig& config,
                              const GeometryTransform& geometry, int device_id);

// Phase 15 test probe: the same device Polygon IoU the NMS kernel uses, one
// value per pair. Both vectors must have the same length. A value the CPU
// rotated_intersection_over_union rejects throws the same error message; the
// failing pair is named by its index so a parity test can report it.
struct ObbCudaIouBatch {
  std::vector<float> values;
};

ObbCudaIouBatch launch_obb_iou_cuda(const std::vector<OrientedBox>& left,
                                    const std::vector<OrientedBox>& right, int device_id);

}  // namespace vie