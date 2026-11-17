#pragma once

#include "vie/decoder/model_decoder.hpp"

namespace vie {

// Protocol rtdetr_detection_raw on CUDA. The kernel reads the CUDA tensor in
// place. It does not call Tensor::data() or Tensor::to(), and it does not
// copy the raw [query, 4 + class] tensor to the host. Only the boxes that
// survive the confidence, area, and max_detections cuts are copied.
//
// Work is issued on the legacy default stream (the null stream). ORT CUDA EP
// also uses that stream: the backend sets has_user_compute_stream to 0 and
// do_copy_in_default_stream to 1, then synchronizes the device before
// execute() returns. This decoder does not use DeviceContext's private
// stream, because that stream is not the ORT stream. It synchronizes the
// default stream before reading the fault flag and before returning the
// host DetectionBatch. It does not call cudaDeviceSynchronize().
//
// A failure here is not retried on the CPU decoder.
class RtdetrCudaDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  DetectionBatch decode_batch(const DecoderInput& input, const DecoderConfig& config,
                              const DeviceContext& device) const override;
};

// Registry object for rtdetr_detection_raw. CPU goes to RtdetrDecoder. CUDA
// goes to RtdetrCudaDecoder. InferenceEngine still selects by protocol only.
class RtdetrDetectionDecoder : public IModelDecoder {
 public:
  std::string protocol() const override;
  RawPredictions decode(const std::map<std::string, Tensor>& outputs,
                        const DecoderContext& context) const override;
  DetectionBatch decode_batch(const DecoderInput& input, const DecoderConfig& config,
                              const DeviceContext& device) const override;
};

}  // namespace vie
