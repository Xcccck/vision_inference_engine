#pragma once

#include "vie/preprocess/preprocess.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vie {

// Result of one GPU ArgMax pass. The caller has already checked the tensor
// contract and the request fields. class_ids holds original_width *
// original_height values in row-major order. validate_ms covers the finite
// check of the raw logits; argmax_ms covers the optional bilinear resize plus
// the per-pixel coordinate recovery and ArgMax; d2h_ms is the compact class
// map copy and d2h_bytes is its size.
struct SemanticSegmentationCudaLaunch {
  std::vector<std::int32_t> class_ids;
  double validate_ms = 0;
  double argmax_ms = 0;
  double d2h_ms = 0;
  std::size_t d2h_bytes = 0;
};

// logits is a contiguous [channels, logits_height, logits_width] FP32 plane on
// the CUDA device named by device_id. Every value must be finite; the first
// offending element in linear order is reported with the same wording as the
// CPU decoder. class ids are produced for the ORIGINAL geometry size.
SemanticSegmentationCudaLaunch launch_semantic_segmentation_cuda(
    const float* logits, int channels, int logits_height, int logits_width,
    const GeometryTransform& geometry, const std::string& output_name, int device_id);

}  // namespace vie