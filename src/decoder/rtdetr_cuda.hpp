#pragma once

#include "vie/decoder/detection.hpp"

#include <vector>

namespace vie {

struct RtdetrCudaLaunch {
  std::vector<Detection> detections;
  int candidates_after_filter = 0;
  int candidates_before_nms = 0;
  double kernel_ms = 0;
  double sort_ms = 0;
  double d2h_ms = 0;
  std::size_t d2h_bytes = 0;
};

// data is a contiguous CUDA fp32 tensor of [1, queries, 4 + classes]. The
// caller has already validated the config, geometry, and tensor.
RtdetrCudaLaunch launch_rtdetr_cuda(const float* data, int queries, int num_classes, float threshold,
                                    int max_detections, int device_id, const GeometryTransform& geometry);

}  // namespace vie
