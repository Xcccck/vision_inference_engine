#pragma once

#include "vie/decoder/model_decoder.hpp"
#include "vie/decoder/pose.hpp"

namespace vie {

// yolov8_pose_raw reads one already-decoded tensor:
//   cx, cy, width, height, class probabilities, then keypoint_count groups of
//   (x, y, sigmoid confidence).
// The graph has already applied DFL, dist2bbox, the stride scale, and the
// keypoint (x * 2 + anchor - 0.5) * stride decode. This decoder does not
// repeat those ops and does not apply another Sigmoid. NMS uses the person
// box. A low keypoint confidence does not drop the person.
class YoloV8PoseDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs, const DecoderContext& context) const override;
  PoseBatch decode_pose(const DecoderInput& input, const PoseDecodeConfig& config,
                        const DeviceContext& device) const override;
};

}  // namespace vie
