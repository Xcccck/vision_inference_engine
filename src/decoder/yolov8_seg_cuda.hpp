#pragma once

#include "vie/decoder/yolov8_seg_decoder.hpp"

namespace vie {

// Default-stream CUDA path. It does not call the CPU decoder and does not copy
// output0 or output1 to the host. Only the NMS survivors are copied back, and
// the binary masks stay in CUDA storage.
SegmentationBatch decode_yolov8_seg_cuda(const DecoderInput& input, const DecoderConfig& config,
                                         const DeviceContext& device, CudaSegmentationTiming* timing);

}  // namespace vie
