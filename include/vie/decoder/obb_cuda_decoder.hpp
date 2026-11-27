#pragma once

#include "vie/decoder/model_decoder.hpp"
#include "vie/decoder/obb.hpp"

namespace vie {

// Protocol yolov8_obb_raw on CUDA. The rank-3 fp32 tensor stays on the device:
// candidate validation, class selection, the score sort, the pre-NMS top-k cut,
// the GeometryTransform map (including the minimum-area refit), Polygon IoU,
// and greedy rotated NMS all run in kernels. Only the kept ObbObjects are
// copied D2H. It never calls Tensor::data() or Tensor::to(), and a failure is
// not retried on the CPU decoder.
//
// Work is issued on the legacy default stream (the null stream), which is the
// ORT CUDA EP stream: the backend sets has_user_compute_stream to 0 and
// synchronizes the device before execute() returns. The stream is synchronized
// before a fault flag or counter is read and before the host result is built.
// It does not call cudaDeviceSynchronize().
//
// Semantics match YoloV8ObbDecoder exactly: the box channels are cx, cy,
// width, height and the last channel is the angle in radians in
// [-pi/4, 3pi/4]; the class probabilities are checked against
// [-1e-4, 1 + 1e-4] and clamped; BestClass keeps the highest score with the
// smaller class id breaking ties; candidates are ordered by descending score,
// then ascending anchor, then ascending class id; the map is the same
// double-precision corner transform and minimum-area fit; rotated IoU is the
// true convex-polygon intersection over union with the CPU clipping algorithm;
// NMS is greedy with IoU >= threshold suppressing. A non-finite or
// out-of-range value the CPU decoder would have read fails the whole call with
// the same message.
class ObbCudaDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  ObbBatch decode_obb(const DecoderInput& input, const ObbDecodeConfig& config,
                      const DeviceContext& device) const override;
};

// Registry object for yolov8_obb_raw. CPU goes to YoloV8ObbDecoder, CUDA goes
// to ObbCudaDecoder. InferenceEngine still selects by protocol only.
class ObbDispatchDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  ObbBatch decode_obb(const DecoderInput& input, const ObbDecodeConfig& config,
                      const DeviceContext& device) const override;
};

}  // namespace vie