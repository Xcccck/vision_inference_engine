#pragma once

#include "vie/decoder/detection.hpp"

#include <vector>

namespace vie {

struct CudaKeptPrediction {
  BoundingBox model_box;
  BoundingBox original_box;
  float score = 0.f;
  int class_id = -1;
  int prediction_index = -1;
};

struct CudaDetectionPass {
  std::vector<CudaKeptPrediction> kept;
  int candidates_after_filter = 0;
  int candidates_before_nms = 0;
  bool nms_applied = false;
  double gpu_execute_ms = 0;
  double d2h_ms = 0;
  double sort_ms = 0;
  double nms_ms = 0;
};

// mask_channels is 0 for yolov8_detect_raw. Segmentation passes the extra
// coefficient channels so they are not read as classes. The kept index is the
// sort key; prediction_index is the row in the prediction tensor.
CudaDetectionPass decode_yolov8_cuda_pass(const Tensor& tensor, const DecoderConfig& config,
                                          const GeometryTransform& geometry, const DeviceContext& device,
                                          int mask_channels);

DetectionResult decode_yolov8_cuda(const Tensor& tensor, const DecoderConfig& config,
                                   const GeometryTransform& geometry, const DeviceContext& device);

}  // namespace vie
