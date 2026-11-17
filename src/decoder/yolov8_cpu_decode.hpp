#pragma once

#include "vie/decoder/detection.hpp"

namespace vie {

// One NMS survivor. original_box is in original-image pixels. model_box is the
// xyxy box in model-input pixels before that mapping. prediction_index is the
// column in the exported tensor, not the sort key.
struct KeptPrediction {
  BoundingBox model_box;
  BoundingBox original_box;
  float score = 0.f;
  int class_id = -1;
  int prediction_index = -1;
};

struct CpuDetectionPass {
  std::vector<KeptPrediction> kept;
  int candidates_after_filter = 0;
  int candidates_before_nms = 0;
  bool nms_applied = false;
  double box_ms = 0;
  double filter_ms = 0;
  double nms_ms = 0;
};

// Same filter, sort, top-k, map-then-NMS path as the detection decoder.
// mask_channels are present in the tensor but are not read here.
CpuDetectionPass decode_yolov8_cpu_predictions(const float* data, int channel_axis, int channels,
                                              int predictions, const DecoderConfig& config,
                                              const GeometryTransform& geometry, bool record_times);

}  // namespace vie
