#pragma once

#include "vie/decoder/model_decoder.hpp"

namespace vie {

// Wall times for one CPU segmentation decode. Each stage includes the
// allocations made inside that stage. total_ms covers the whole call.
struct SegmentationTiming {
  int instances = 0;
  int mask_width = 0;
  int mask_height = 0;
  double detection_ms = 0;
  double filter_ms = 0;
  double nms_ms = 0;
  double gather_ms = 0;
  double reconstruct_ms = 0;
  double crop_ms = 0;
  double resize_ms = 0;
  double total_ms = 0;
  bool includes_allocation = true;
};

// Device times for one CUDA segmentation decode. Filtering is inside the
// detection emit kernel, so filter_ms stays 0 and detection_ms is that kernel.
// Kernels run on the default stream. d2h_ms is the kept-candidate copy only.
// Masks stay on the device. Times include the launches and the final stream
// sync. mask_alloc_ms is the output Tensor allocation.
struct CudaSegmentationTiming {
  int instances = 0;
  int mask_width = 0;
  int mask_height = 0;
  int prototype_channels = 0;
  int prototype_width = 0;
  int prototype_height = 0;
  double detection_ms = 0;
  double filter_ms = 0;
  double sort_ms = 0;
  double nms_ms = 0;
  double gather_ms = 0;
  double reconstruct_ms = 0;
  double crop_ms = 0;
  double resize_ms = 0;
  double gpu_ms = 0;
  double wall_ms = 0;
  double mask_alloc_ms = 0;
  double d2h_ms = 0;
  bool includes_allocation = true;
  bool includes_sync = true;
  bool masks_copied_to_host = false;
};

// Protocol yolov8_seg_raw. The exported detection tensor is already decoded
// cxcywh plus sigmoid class scores plus raw mask coefficients. Prototypes are
// SiLU features. This decoder does not run DFL or another sigmoid. Mask crop
// and resize are not in the graph, so retina_masks and the binary test must
// be set on DecoderConfig. A CUDA device runs the CUDA path and does not fall
// back to the CPU decoder.
class YoloV8SegmentationDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  DetectionBatch decode_batch(const DecoderInput& input, const DecoderConfig& config,
                              const DeviceContext& device) const override;
  SegmentationBatch decode_segmentation(const DecoderInput& input, const DecoderConfig& config,
                                        const DeviceContext& device) const override;
  SegmentationBatch decode_timed(const DecoderInput& input, const DecoderConfig& config,
                                 const DeviceContext& device, SegmentationTiming* timing) const;
  SegmentationBatch decode_cuda_timed(const DecoderInput& input, const DecoderConfig& config,
                                      const DeviceContext& device, CudaSegmentationTiming* timing) const;
};

}  // namespace vie
