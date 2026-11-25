#pragma once

#include "vie/core/tensor.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/model_config.hpp"
#include "vie/runtime/execution_resource.hpp"
#include "vie/runtime/runtime_metrics.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vie {

// A single loaded model instance with thread-safe inference
class ModelInstance {
 public:
  ModelInstance(const std::string& id, const ModelConfig& config, std::shared_ptr<MetricsCollector> metrics);
  ~ModelInstance() = default;

  ModelInstance(const ModelInstance&) = delete;
  ModelInstance& operator=(const ModelInstance&) = delete;

  const std::string& id() const noexcept { return id_; }
  const ModelConfig& config() const noexcept { return config_; }
  bool loaded() const noexcept { return engine_.loaded(); }
  const ModelManifest& manifest() const { return engine_.manifest(); }

  // Phase 9.3: The real execution resource of this instance, derived from the
  // committed engine DeviceContext (NOT from ImageInput.device). Because
  // InferenceEngine commits preprocess/backend/decode all to one device and
  // resolve_backend() guarantees OnnxRuntimeCuda <=> Cuda, this is authoritative
  // for a loaded engine. A never-loaded engine (not possible for an instance
  // stored in ModelManager) reports Unknown; the scheduler rejects Unknown.
  ExecutionResource execution_resource() const {
    if (!engine_.loaded()) {
      return ExecutionResource{};  // Unknown
    }
    return execution_resource_from_device(engine_.device());
  }

  // Thread-safe inference methods (serialize concurrent calls)
  DetectionBatch infer(const ImageInput& image, DetectionInferTiming* timing = nullptr);
  // Phase 9.1: Batch inference. Serializes on the same mutex as infer().
  // A Batch>1 call is not concurrent-safe within one instance; parallelism
  // across batches requires multiple ModelInstances. Only task detection
  // with a batch-capable decoder returns results; other tasks throw
  // Unsupported for batch sizes greater than one.
  DetectionBatch infer_batch(const std::vector<ImageInput>& images,
                             DetectionInferTiming* timing = nullptr);

  // Phase 9.3.2: Async-only execution boundary used by Runtime::submit()'s
  // worker path. Semantically identical to infer()/infer_batch() EXCEPT that
  // when this instance runs on a CUDA device and the request image is a
  // Runtime-owned CPU image (already deep-copied at submit), it first stages
  // that CPU image onto the committed CUDA device (synchronous, stream-ordered
  // H2D) with an owning CudaInputBuffer whose lifetime spans the whole
  // preprocess/backend/decode call, then runs inference on the staged image.
  // The synchronous infer()/infer_batch() APIs are intentionally left unchanged:
  // they still require the caller's image device to match the engine device and
  // never accept an external CUDA image through this path. CPU instances do no
  // staging. These methods do not relax the submit-time CUDA ImageInput ban.
  DetectionBatch infer_async(const ImageInput& image, DetectionInferTiming* timing = nullptr);
  DetectionBatch infer_batch_async(const std::vector<ImageInput>& images,
                                   DetectionInferTiming* timing = nullptr);
  SegmentationBatch infer_segmentation(const ImageInput& image, const SegmentationRuntimeConfig& runtime,
                                       SegmentationInferTiming* timing = nullptr);
  ClassificationBatch infer_classification(const ImageInput& image, const ClassificationRuntimeConfig& runtime,
                                           ClassificationInferTiming* timing = nullptr);
  ObbBatch infer_obb(const ImageInput& image, const ObbRuntimeConfig& runtime, ObbInferTiming* timing = nullptr);
  PoseBatch infer_pose(const ImageInput& image, const PoseRuntimeConfig& runtime, PoseInferTiming* timing = nullptr);
  // Thread-safe dense semantic segmentation. Serializes on the same mutex as
  // the other infer methods. Only task semantic_segmentation with protocol
  // deeplabv3_semantic_seg_raw is accepted; the engine reports Unsupported
  // otherwise. This never returns detections or instance masks.
  SemanticSegmentationResult infer_semantic_segmentation(const ImageInput& image,
                                                         SemanticSegmentationInferTiming* timing = nullptr);

  // Thread-safe preprocess
  PreprocessResult preprocess(const ImageInput& image) const;

  // Direct backend execution (thread-safe)
  BackendOutput execute(const BackendRequest& request);

  // Phase 9.3.2.1: Flush ORT profiling for THIS instance's real engine session
  // and return the Chrome-trace JSON path. Empty unless the model was loaded
  // with ModelConfig::ort_enable_profiling. Serializes on the same mutex as the
  // infer methods so it cannot race an in-flight run; it never changes results.
  std::string end_profiling();

 private:
  std::string id_;
  ModelConfig config_;
  InferenceEngine engine_;
  mutable std::mutex mutex_;
  std::shared_ptr<MetricsCollector> metrics_;
};

}  // namespace vie
