#pragma once

#include "vie/decoder/classification.hpp"

namespace vie {

// Protocols classification_logits_raw and classification_probabilities_raw on
// CUDA. The preset rank-2 fp32 tensor stays on the device: validation, the
// probability sum, the sort, and the top-k cut all run in kernels, and only
// the final top_k (class_id, score) pairs are copied D2H. It never calls
// Tensor::data() or Tensor::to(), and a failure is not retried on the CPU
// decoder.
//
// Work is issued on the legacy default stream (the null stream), which is the
// ORT CUDA EP stream: the backend sets has_user_compute_stream to 0 and
// synchronizes the device before execute() returns (see RtdetrCudaDecoder).
// The stream is synchronized before the fault flag is read and before the
// host result is built. It does not call cudaDeviceSynchronize().
//
// Semantics match the CPU decoders exactly: no Softmax, descending score
// order, ties resolved by the smaller class id, top_k in [1, num_classes],
// and a non-finite or out-of-range value anywhere in the tensor is rejected
// with the same first offending class id.
class ClassificationCudaDecoder : public IModelDecoder {
 public:
  explicit ClassificationCudaDecoder(bool probabilities_mode);

  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  ClassificationBatch decode_classification(const DecoderInput& input,
                                            const ClassificationDecodeRequest& request,
                                            const DeviceContext& device) const override;

 private:
  bool probabilities_mode_ = false;
};

// Registry object for both classification protocols. CPU goes to the matching
// CPU decoder, CUDA goes to ClassificationCudaDecoder. InferenceEngine still
// selects by protocol only, exactly like RtdetrDetectionDecoder.
class ClassificationDispatchDecoder : public IModelDecoder {
 public:
  explicit ClassificationDispatchDecoder(bool probabilities_mode);

  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  ClassificationBatch decode_classification(const DecoderInput& input,
                                            const ClassificationDecodeRequest& request,
                                            const DeviceContext& device) const override;

 private:
  bool probabilities_mode_ = false;
};

}  // namespace vie