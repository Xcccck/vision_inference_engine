#pragma once

#include "vie/core/tensor.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/model_config.hpp"
#include "vie/runtime/runtime_metrics.hpp"

#include <memory>
#include <mutex>
#include <string>

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

  // Thread-safe inference methods (serialize concurrent calls)
  DetectionBatch infer(const ImageInput& image, DetectionInferTiming* timing = nullptr);
  SegmentationBatch infer_segmentation(const ImageInput& image, const SegmentationRuntimeConfig& runtime,
                                       SegmentationInferTiming* timing = nullptr);
  ClassificationBatch infer_classification(const ImageInput& image, const ClassificationRuntimeConfig& runtime,
                                           ClassificationInferTiming* timing = nullptr);
  ObbBatch infer_obb(const ImageInput& image, const ObbRuntimeConfig& runtime, ObbInferTiming* timing = nullptr);
  PoseBatch infer_pose(const ImageInput& image, const PoseRuntimeConfig& runtime, PoseInferTiming* timing = nullptr);

  // Thread-safe preprocess
  PreprocessResult preprocess(const ImageInput& image) const;

  // Direct backend execution (thread-safe)
  BackendOutput execute(const BackendRequest& request);

 private:
  std::string id_;
  ModelConfig config_;
  InferenceEngine engine_;
  mutable std::mutex mutex_;
  std::shared_ptr<MetricsCollector> metrics_;
};

}  // namespace vie
