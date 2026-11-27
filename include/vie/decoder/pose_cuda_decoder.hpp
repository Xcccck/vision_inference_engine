#pragma once

#include "vie/decoder/model_decoder.hpp"
#include "vie/decoder/pose.hpp"

namespace vie {

// Protocol yolov8_pose_raw on CUDA. The rank-3 fp32 tensor stays on the
// device: class filtering, box and keypoint decode, the score sort, the
// pre-NMS top-k cut, the GeometryTransform inverse map, and greedy NMS all run
// in kernels, and only the kept PoseObjects (box, score, and keypoints) are
// copied D2H. It never calls Tensor::data() or Tensor::to(), and a failure is
// not retried on the CPU decoder.
//
// Work is issued on the legacy default stream (the null stream), which is the
// ORT CUDA EP stream: the backend sets has_user_compute_stream to 0 and
// synchronizes the device before execute() returns. The stream is synchronized
// before a fault flag or counter is read and before the host result is built.
// It does not call cudaDeviceSynchronize().
//
// Semantics match YoloV8PoseDecoder exactly: probabilities are checked against
// [-1e-4, 1 + 1e-4] and clamped, the best class is the highest score with the
// smaller class id breaking ties, candidates are ordered by descending score
// then ascending anchor then ascending class id, boxes and keypoints use the
// same double-precision inverse map, keypoints are never clipped, and NMS is
// greedy with IoU >= threshold suppressing. A non-finite or out-of-range value
// in an anchor that the threshold keeps fails the whole call with the same
// message the CPU decoder uses; values in a dropped anchor are not read.
class PoseCudaDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  PoseBatch decode_pose(const DecoderInput& input, const PoseDecodeConfig& config,
                        const DeviceContext& device) const override;
};

// Registry object for yolov8_pose_raw. CPU goes to YoloV8PoseDecoder, CUDA
// goes to PoseCudaDecoder. InferenceEngine still selects by protocol only.
class PoseDispatchDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  PoseBatch decode_pose(const DecoderInput& input, const PoseDecodeConfig& config,
                        const DeviceContext& device) const override;
};

}  // namespace vie