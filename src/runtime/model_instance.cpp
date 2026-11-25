#include "vie/runtime/model_instance.hpp"

#include "vie/core/error.hpp"
#include "vie/runtime/cuda_input_staging.hpp"

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

// Phase 9.1: same mutex + metrics pattern as infer(). The whole batch counts
// as one logical inference call for concurrency accounting.
DetectionBatch ModelInstance::infer_batch(const std::vector<ImageInput>& images,
                                          DetectionInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  auto result = engine_.infer_batch(images, timing);
  guard.set_success(true);
  return result;
}

// Phase 9.3.2: async-only staging boundary. Only CUDA instances feeding on a
// Runtime-owned CPU image perform an H2D; the owned CudaInputBuffer lives for
// the whole engine_.infer() call (staging -> preprocess -> backend -> decoder
// -> D2H), which is host-synchronous, so the device input is never freed while
// still in use and is freed on every exception path via RAII. CPU instances and
// already-on-device images take the identical synchronous path as before.
DetectionBatch ModelInstance::infer_async(const ImageInput& image, DetectionInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  const DeviceContext& dev = engine_.device();
  if (dev.device() == DeviceType::Cuda && image.device == DeviceType::Cpu) {
    CudaInputBuffer staged = CudaInputBuffer::stage(image, dev);
    auto result = engine_.infer(staged.image(), timing);
    guard.set_success(true);
    return result;
  }
  auto result = engine_.infer(image, timing);
  guard.set_success(true);
  return result;
}

DetectionBatch ModelInstance::infer_batch_async(const std::vector<ImageInput>& images,
                                                DetectionInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  const DeviceContext& dev = engine_.device();
  if (dev.device() == DeviceType::Cuda) {
    // CUDA stays Batch=1 upstream, but stage defensively: each request gets its
    // OWN device buffer (never shared across requests), held alive for the call.
    std::vector<CudaInputBuffer> buffers;
    buffers.reserve(images.size());
    std::vector<ImageInput> staged_images;
    staged_images.reserve(images.size());
    for (const auto& img : images) {
      if (img.device == DeviceType::Cpu) {
        buffers.push_back(CudaInputBuffer::stage(img, dev));
        staged_images.push_back(buffers.back().image());
      } else {
        staged_images.push_back(img);
      }
    }
    auto result = engine_.infer_batch(staged_images, timing);
    guard.set_success(true);
    return result;
  }
  auto result = engine_.infer_batch(images, timing);
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

SemanticSegmentationResult ModelInstance::infer_semantic_segmentation(const ImageInput& image,
                                                                       SemanticSegmentationInferTiming* timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  InferenceGuard guard(metrics_, id_);
  auto result = engine_.infer_semantic_segmentation(image, timing);
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

// Phase 9.3.2.1: diagnostic flush of the real session profile. Shares the
// instance mutex so it runs strictly after any in-flight inference, ensuring
// every issued ORT node has been recorded before EndProfiling is called.
std::string ModelInstance::end_profiling() {
  std::lock_guard<std::mutex> lock(mutex_);
  return engine_.end_profiling();
}

}  // namespace vie
