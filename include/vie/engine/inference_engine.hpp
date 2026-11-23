#pragma once

#include "vie/backend/backend_registry.hpp"
#include "vie/core/device_context.hpp"
#include "vie/decoder/classification.hpp"
#include "vie/decoder/decoder_registry.hpp"
#include "vie/decoder/segmentation.hpp"
#include "vie/model/model_config.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace vie {

// Caller-supplied decode policy. Every field stays unset until the caller
// writes it. infer_segmentation() does not read these values from the manifest
// and does not write them back.
struct SegmentationRuntimeConfig {
  std::optional<float> confidence_threshold;
  std::optional<float> iou_threshold;
  std::optional<ClassSelection> class_selection;
  std::optional<bool> class_aware_nms;
  std::optional<int> pre_nms_top_k;
  std::optional<int> max_detections;
  std::optional<bool> nms_enabled;
  std::optional<bool> retina_masks;
  std::optional<bool> binary_mask_gt_zero;
  std::optional<std::size_t> max_mask_bytes;
};

// Wall time of one infer() call. On CUDA, preprocess and the backend
// synchronize before they return, and the decoder's kept-box copy is inside
// decode_ms. The caller's H2D of ImageInput is outside this.
struct DetectionInferTiming {
  double preprocess_ms = 0;
  double execute_ms = 0;
  double decode_ms = 0;
  double total_ms = 0;
  bool includes_sync = false;
  bool includes_h2d = false;
  bool includes_d2h = false;
};

// top_k is required for each call. It is not written back to the manifest.
struct ClassificationRuntimeConfig {
  std::optional<int> top_k;
};

// execute_ms is the backend. d2h_ms and d2h_bytes are the explicit copy of the
// classification vector. output_bytes is that tensor. validate_ms is the CPU
// numeric check and topk_ms is only the sort. decode_ms is d2h_ms plus those
// two. ORT's own transfers are not included in d2h_bytes.
struct ClassificationInferTiming {
  double preprocess_ms = 0;
  double execute_ms = 0;
  double d2h_ms = 0;
  double validate_ms = 0;
  double topk_ms = 0;
  double decode_ms = 0;
  double total_ms = 0;
  std::size_t output_bytes = 0;
  std::size_t d2h_bytes = 0;
  bool includes_sync = false;
  bool includes_h2d = false;
  bool includes_d2h = false;
};

// Wall time of one infer_segmentation() call. includes_sync is true on CUDA
// because preprocess, the backend, and the decoder synchronize before return.
// The caller's H2D of ImageInput and a later Mask::to_cpu() are outside this.
struct SegmentationInferTiming {
  double load_ms = 0;
  double preprocess_ms = 0;
  double execute_ms = 0;
  double decode_ms = 0;
  double total_ms = 0;
  bool includes_sync = false;
  bool includes_h2d = false;
  bool includes_mask_d2h = false;
};

// Loads a resolved manifest and runs prepare -> execute -> decode.
// Image files and NMS are outside this build. The ONNX Runtime CPU backend is
// available when the library is built with VIE_ENABLE_ONNXRUNTIME. The CUDA
// backend is available only when that build also has the ONNX Runtime GPU
// package. load() throws when a critical field is unresolved or the requested
// backend is absent. Device selection lives in load(), not in the backends.
// One engine is not safe for concurrent load(), infer(), infer_segmentation(),
// or infer_classification() calls. Those calls share the session with no lock;
// overlapping them is a data race. device() is the device committed by the
// last successful load. A failed load leaves the previous session in place,
// and an engine that has never loaded reports the default CPU context.
class InferenceEngine {
 public:
  InferenceEngine();

  void load(const ModelConfig& config);
  bool loaded() const noexcept { return loaded_; }

  const ModelManifest& manifest() const;
  const DeviceContext& device() const noexcept { return device_; }

  BackendOutput execute(const BackendRequest& request);
  // Image entry point. Uses the image's own color and layout as the source
  // and the manifest for the model-side parameters. Missing manifest fields
  // throw UnresolvedField. execute() is unchanged and does not call this.
  PreprocessResult preprocess(const ImageInput& image) const;
  // Preprocess, execute, then decode. Missing decoder or NMS fields throw.
  // execute() does not call this and does not require those fields.
  // A segmentation manifest throws Unsupported. A classification manifest
  // throws Unsupported. This never returns masks or class labels.
  DetectionBatch infer(const ImageInput& image, DetectionInferTiming* timing = nullptr);
  // Preprocess, execute, then decode_segmentation. runtime is the only source
  // of thresholds, NMS, retina_masks, and the mask budget. timing may be null.
  // A detection manifest throws Unsupported. Mask storage outlives this engine.
  SegmentationBatch infer_segmentation(const ImageInput& image, const SegmentationRuntimeConfig& runtime,
                                       SegmentationInferTiming* timing = nullptr);
  // Preprocess, execute, then decode_classification. Only task classification
  // with classification_logits_raw or classification_probabilities_raw is
  // accepted. A CUDA output tensor
  // is copied to CPU here; the decoder then runs Top-K on that CPU tensor.
  ClassificationBatch infer_classification(const ImageInput& image, const ClassificationRuntimeConfig& runtime,
                                           ClassificationInferTiming* timing = nullptr);
  RawPredictions decode(const BackendOutput& output) const;
  RawPredictions run(const BackendRequest& request);

  void synchronize() const;

  BackendRegistry& backends() noexcept { return backends_; }
  DecoderRegistry& decoders() noexcept { return decoders_; }

 private:
  DeviceContext device_;
  BackendRegistry backends_;
  DecoderRegistry decoders_;
  ModelManifest manifest_{};
  std::unique_ptr<IInferenceBackend> backend_;
  std::unique_ptr<IModelDecoder> decoder_;
  bool loaded_ = false;
};

}  // namespace vie
