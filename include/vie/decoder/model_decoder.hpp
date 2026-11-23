#pragma once

#include "vie/core/tensor.hpp"
#include "vie/decoder/obb.hpp"
#include "vie/decoder/pose.hpp"
#include "vie/decoder/segmentation.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vie {

struct ClassificationBatch;
struct ClassificationDecodeRequest;

// Boxes are in the model's input coordinate space. The decoder does not run
// NMS and does not map coordinates back to the original image.
struct RawBox {
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;
};

struct RawDetection {
  int class_id = -1;
  float score = 0.f;
  RawBox box;
};

struct RawPredictions {
  std::string protocol;
  std::string box_format;
  std::vector<RawDetection> detections;
};

struct DecoderContext {
  std::string protocol;
  std::string box_format;
  std::optional<int> num_classes;
  std::vector<TensorSpec> outputs;
};

class IModelDecoder {
 public:
  virtual ~IModelDecoder() = default;

  virtual std::string protocol() const { return {}; }

  // Existing raw path used by the mock decoder. It does not apply NMS and
  // does not map coordinates.
  virtual RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                                const DecoderContext& context) const = 0;

  // Detection path. The default refuses so a decoder that only implements
  // decode() cannot be mistaken for a detection decoder.
  virtual DetectionBatch decode_batch(const DecoderInput& input, const DecoderConfig& config,
                                      const DeviceContext& device) const;

  // Segmentation path. The default refuses so a detection or mock decoder
  // cannot be mistaken for a segmentation decoder.
  virtual SegmentationBatch decode_segmentation(const DecoderInput& input, const DecoderConfig& config,
                                                const DeviceContext& device) const;

  // Classification path. The default refuses so a detection or segmentation
  // decoder cannot be mistaken for a classification decoder.
  virtual ClassificationBatch decode_classification(const DecoderInput& input,
                                                    const ClassificationDecodeRequest& request,
                                                    const DeviceContext& device) const;

  // OBB path. The default refuses so a detection decoder cannot be mistaken
  // for an oriented-box decoder. Rotated NMS in this phase is CPU-only.
  virtual ObbBatch decode_obb(const DecoderInput& input, const ObbDecodeConfig& config,
                              const DeviceContext& device) const;

  // Pose path. The default refuses so a detection decoder cannot be mistaken
  // for a keypoint decoder. Pose NMS in this phase is CPU-only.
  virtual PoseBatch decode_pose(const DecoderInput& input, const PoseDecodeConfig& config,
                                const DeviceContext& device) const;
};

}  // namespace vie
