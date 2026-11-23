#pragma once

#include "vie/core/device_context.hpp"
#include "vie/core/tensor.hpp"
#include "vie/decoder/model_decoder.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace vie {

// One class from a rank-2 logits vector. class_id is the index on the class
// axis. class_name is empty when the manifest has no names. There is no box.
struct Classification {
  int class_id = -1;
  float score = 0.f;
  std::string class_name;
};

// Top-K for one image. The model output tensor is not stored here.
// output_bytes is the logits tensor size. d2h_bytes is the explicit host copy
// of that tensor and stays 0 when the tensor was already on CPU. ORT's own
// transfers are not included. execution_device is the device that ran Top-K.
struct ClassificationResult {
  std::vector<Classification> labels;
  std::string output_name;
  std::string protocol;
  std::string score_type;
  int num_classes = 0;
  int top_k = 0;
  DeviceType execution_device = DeviceType::Cpu;
  int execution_device_id = 0;
  std::size_t output_bytes = 0;
  std::size_t d2h_bytes = 0;
  // CPU time inside the decoder. validate_ms is the numeric check. topk_ms is
  // only the sort. The engine adds the explicit D2H separately.
  double validate_ms = 0;
  double topk_ms = 0;
};

// One result per image. The first decoder fills a single result for batch 1.
struct ClassificationBatch {
  std::vector<ClassificationResult> results;
};

// Fields the logits decoder reads. top_k comes from the call, not the manifest.
struct ClassificationDecodeRequest {
  std::string output_name;
  int class_axis = -1;
  int num_classes = 0;
  int top_k = 0;
  std::string score_type;
  std::vector<std::string> class_names;
};

// Fixed protocol tolerance for a single-label probability vector. The sum is
// accumulated in FP64. Values outside [0, 1], or a sum farther than this from
// 1, are rejected and are not renormalized.
inline constexpr double kClassificationProbabilitySumTolerance = 1e-3;

// Protocol classification_logits_raw. Rank-2 FP32 logits, batch 1. It does not
// run Softmax. A CUDA tensor is refused; the engine copies the vector to CPU
// before calling this decoder.
class ClassificationDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  ClassificationBatch decode_classification(const DecoderInput& input,
                                            const ClassificationDecodeRequest& request,
                                            const DeviceContext& device) const override;
};

// Protocol classification_probabilities_raw. Rank-2 FP32 probabilities that
// the model already produced. It does not run Softmax, Sigmoid, or any
// renormalization. A CUDA tensor is refused.
class ClassificationProbabilitiesDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  ClassificationBatch decode_classification(const DecoderInput& input,
                                            const ClassificationDecodeRequest& request,
                                            const DeviceContext& device) const override;
};

}  // namespace vie
