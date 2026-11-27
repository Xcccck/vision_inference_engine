#pragma once

#include "vie/decoder/semantic_segmentation.hpp"

namespace vie {

// Protocol deeplabv3_semantic_seg_raw on CUDA. The fp32 [1, C, H, W] logits
// stay on the device: the finite-value check, the bilinear resize onto the
// model-input grid, the inverse coordinate recovery, and the per-pixel ArgMax
// all run in kernels, and only the int32 class map of the ORIGINAL image is
// copied D2H. It never calls Tensor::data() or Tensor::to(), and a failure is
// not retried on the CPU decoder.
//
// Work is issued on the legacy default stream (the null stream), which is the
// ORT CUDA EP stream: the backend sets has_user_compute_stream to 0 and
// synchronizes the device before execute() returns (see RtdetrCudaDecoder).
// The stream is synchronized before the fault index and the class map are
// read. It does not call cudaDeviceSynchronize().
//
// Semantics match the CPU decoder exactly: sampling uses the same half-pixel
// bilinear formula in FP64 with a float intermediate plane, and the strict '>'
// comparison keeps the smallest class id on a tie, so identical logits produce
// identical class ids.
class SemanticSegmentationCudaDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  SemanticSegmentationBatch decode_semantic_segmentation(
      const DecoderInput& input, const SemanticSegmentationDecodeRequest& request,
      const DeviceContext& device) const override;
};

// Registry object for deeplabv3_semantic_seg_raw. CPU goes to the CPU decoder,
// CUDA goes to SemanticSegmentationCudaDecoder. InferenceEngine still selects
// by protocol only, exactly like RtdetrDetectionDecoder.
class SemanticSegmentationDispatchDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  SemanticSegmentationBatch decode_semantic_segmentation(
      const DecoderInput& input, const SemanticSegmentationDecodeRequest& request,
      const DeviceContext& device) const override;
};

}  // namespace vie