#include "vie/runtime/model_instance.hpp"

#include "vie/core/error.hpp"

#include <chrono>

namespace vie {

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// RAII helper to record metrics on scope exit
class InferenceGuard {
 public:
  InferenceGuard(std::shared_ptr<MetricsCollector> metrics, const std::string& model_id)
      : metrics_(std::move(metrics)), model_id_(model_id), start_(std::chrono::steady_clock::now()) {}

  ~InferenceGuard() {
    if (metrics_) {
      metrics_->record_inference(model_id_, elapsed_ms(start_), success_);
    }
  }

  void set_success(bool success) { success_ = success; }

  InferenceGuard(const InferenceGuard&) = delete;
  InferenceGuard& operator=(const InferenceGuard&) = delete;

 private:
  std::shared_ptr<MetricsCollector> metrics_;
  std::string model_id_;
  std::chrono::steady_clock::time_point start_;
  bool success_ = true;
};

}  // namespace

ModelInstance::ModelInstance(const std::string& id, const ModelConfig& config,
                             std::shared_ptr<MetricsCollector> metrics)
    : id_(id), config_(config), metrics_(std::move(metrics)) {
  engine_.load(config);
  metrics_->record_model_load();
}

DetectionBatch ModelInstance::infer(const ImageInput& image, DetectionInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  auto result = engine_.infer(image, timing);
  guard.set_success(true);
  return result;
}

SegmentationBatch ModelInstance::infer_segmentation(const ImageInput& image,
                                                     const SegmentationRuntimeConfig& runtime,
                                                     SegmentationInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  auto result = engine_.infer_segmentation(image, runtime, timing);
  guard.set_success(true);
  return result;
}

ClassificationBatch ModelInstance::infer_classification(const ImageInput& image,
                                                         const ClassificationRuntimeConfig& runtime,
                                                         ClassificationInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  auto result = engine_.infer_classification(image, runtime, timing);
  guard.set_success(true);
  return result;
}

ObbBatch ModelInstance::infer_obb(const ImageInput& image, const ObbRuntimeConfig& runtime, ObbInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  auto result = engine_.infer_obb(image, runtime, timing);
  guard.set_success(true);
  return result;
}

PoseBatch ModelInstance::infer_pose(const ImageInput& image, const PoseRuntimeConfig& runtime, PoseInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  auto result = engine_.infer_pose(image, runtime, timing);
  guard.set_success(true);
  return result;
}

PreprocessResult ModelInstance::preprocess(const ImageInput& image) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto start = std::chrono::steady_clock::now();
  auto result = engine_.preprocess(image);
  metrics_->record_preprocess(elapsed_ms(start));
  return result;
}

BackendOutput ModelInstance::execute(const BackendRequest& request) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  auto result = engine_.execute(request);
  guard.set_success(true);
  return result;
}

}  // namespace vie
