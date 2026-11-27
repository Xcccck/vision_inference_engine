#pragma once

#include "vie/core/device_context.hpp"
#include "vie/core/tensor.hpp"
#include "vie/decoder/model_decoder.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vie {

// Dense per-pixel prediction in ORIGINAL image coordinates. class_ids is
// row-major and sized width*height, every value in [0, num_classes). This type
// is intentionally separate from the instance-segmentation result (SegmentationBatch
// / Mask): a semantic map has no boxes, no per-instance masks, and no objectness.
struct SemanticSegmentationResult {
  int width = 0;    // original image width
  int height = 0;   // original image height
  int num_classes = 0;
  std::vector<std::int32_t> class_ids;  // row-major, size == width * height

  // Diagnostics and coordinate provenance.
  std::string protocol;
  std::string output_name;
  int logits_width = 0;    // raw model output spatial width
  int logits_height = 0;   // raw model output spatial height
  int original_width = 0;  // the image this map was produced for
  int original_height = 0;
  DeviceType execution_device = DeviceType::Cpu;
  int execution_device_id = 0;
  // Bytes of the raw logits tensor and the explicit D2H copy the decoder
  // performs. On CPU d2h_bytes is 0 (the engine copies the whole tensor before
  // calling the decoder); on CUDA it is only the compact class map. ORT's own
  // transfers are excluded.
  std::size_t output_bytes = 0;
  std::size_t d2h_bytes = 0;
  // Decoder time. On CPU validate_ms is the finite-value and shape check and
  // argmax_ms is the coordinate recovery plus per-pixel ArgMax; d2h_ms is 0.
  // On CUDA all three are GPU event times: the logits validation kernel, the
  // resize/ArgMax kernels, and the compact class-map copy. Host launch and
  // stream-sync cost is outside these fields and is covered by the engine wall
  // clock.
  double validate_ms = 0;
  double argmax_ms = 0;
  double d2h_ms = 0;
};

// One result per image. Batch=1 in this phase; a batch is a future extension.
struct SemanticSegmentationBatch {
  std::vector<SemanticSegmentationResult> results;
};

// Fields the semantic decoder reads. Runtime thresholds do not exist here: the
// output is a full class map, not a thresholded object list.
struct SemanticSegmentationDecodeRequest {
  std::string output_name;
  int num_classes = 0;
  int channel_axis = 1;             // NCHW class axis
  std::string logits_semantics;     // "per_pixel_class_logits"
  std::string output_interpolation; // "bilinear" (continuous logits resize)
  std::string class_id_interpolation; // "nearest" (integer class ids)
};

// Protocol deeplabv3_semantic_seg_raw. Reads a Float32 [1, C, H_out, W_out]
// logits tensor, brings it onto the model-input grid (Bilinear when the model
// output is lower resolution), inverse-maps every original pixel through the
// GeometryTransform (removing letterbox padding and stretch scaling), and runs
// ArgMax across the class axis. It never runs Softmax. A CUDA output tensor is
// refused here; the engine performs an explicit D2H before calling decode.
class SemanticSegmentationDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  SemanticSegmentationBatch decode_semantic_segmentation(
      const DecoderInput& input, const SemanticSegmentationDecodeRequest& request,
      const DeviceContext& device) const override;
};

}  // namespace vie
