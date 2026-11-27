#pragma once

#include "classification_topk.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace vie {

// Result of one GPU top-k pass. The caller has already checked the tensor
// contract (device, dtype, shape, axis). ranked holds exactly top_k entries in
// the same order the CPU decoder produces. validate_ms covers the finite /
// probability-range check and the class-order sum; sort_ms is only the sort;
// d2h_ms is the explicit copy of the top-k pairs and d2h_bytes is its size.
struct ClassificationCudaLaunch {
  std::vector<RankedScore> ranked;
  double validate_ms = 0;
  double sort_ms = 0;
  double d2h_ms = 0;
  std::size_t d2h_bytes = 0;
};

// Reads data[i * stride] for class id i. probabilities=true additionally
// requires every value in [0, 1] and a class-order sum within
// kClassificationProbabilitySumTolerance of 1. Throws InvalidArgument with the
// same wording as the CPU decoders, before any sorting that could compare NaN.
ClassificationCudaLaunch launch_classification_cuda(const float* data, int num_classes,
                                                    long long stride, int top_k, bool probabilities,
                                                    const std::string& output_name, int device_id);

}  // namespace vie