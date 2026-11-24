#include "vie/runtime/runtime.hpp"

#include "vie/core/error.hpp"

#include <chrono>
#include <exception>
#include <utility>

namespace vie {

ModelManager::ModelManager(std::shared_ptr<MetricsCollector> metrics, std::size_t max_models)
    : metrics_(std::move(metrics)), max_models_(max_models) {}

std::shared_ptr<ModelInstance> ModelManager::load(const std::string& id, const ModelConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (instances_.find(id) != instances_.end()) {
    throw VieError(ErrorCode::InvalidArgument, "model with id '" + id + "' is already loaded");
  }
  if (instances_.size() >= max_models_) {
    throw VieError(ErrorCode::Unsupported,
                   "maximum number of loaded models (" + std::to_string(max_models_) + ") reached");
  }
  auto instance = std::make_shared<ModelInstance>(id, config, metrics_);
  instances_[id] = instance;
  return instance;
}

void ModelManager::unload(const std::string& id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = instances_.find(id);
  if (it == instances_.end()) {
    throw VieError(ErrorCode::InvalidArgument, "model with id '" + id + "' is not loaded");
  }
  instances_.erase(it);
  metrics_->record_model_unload();
}

std::shared_ptr<ModelInstance> ModelManager::get(const std::string& id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = instances_.find(id);
  if (it == instances_.end()) {
    return nullptr;
  }
  return it->second;
}

bool ModelManager::contains(const std::string& id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return instances_.find(id) != instances_.end();
}

std::vector<std::string> ModelManager::loaded_ids() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> ids;
  ids.reserve(instances_.size());
  for (const auto& [id, instance] : instances_) {
    ids.push_back(id);
  }
  return ids;
}

std::size_t ModelManager::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return instances_.size();
}

Runtime::Runtime(const RuntimeConfig& config)
    : config_(config),
      thread_pool_(config.thread_pool_size),
      metrics_(std::make_shared<MetricsCollector>()),
      models_(metrics_, config.max_loaded_models),
      request_queue_(std::make_unique<RequestQueue>(config.request_queue_capacity)) {
  // Mark as running even if deferred - submit() checks running_
  running_.store(true);
  
  // Auto-start workers unless deferred
  if (!config_.defer_start) {
    launch_workers();
  }
}

Runtime::~Runtime() {
  // Graceful shutdown: wait for all queued requests to complete
  shutdown();
}

void Runtime::start() {
  if (!running_.load()) {
    running_.store(true);
  }
  launch_workers();
}

void Runtime::launch_workers() {
  // Don't launch if workers are already running
  if (!inference_workers_.empty()) {
    return;
  }
  
  // Launch inference worker threads
  const std::size_t worker_count = config_.inference_worker_count;
  if (worker_count == 0) {
    throw VieError(ErrorCode::InvalidArgument, "inference_worker_count must be at least 1");
  }
  
  for (std::size_t i = 0; i < worker_count; ++i) {
    inference_workers_.emplace_back([this] { worker_loop(); });
  }
}

void Runtime::shutdown() {
  if (!running_.load()) {
    return;
  }
  running_.store(false);
  
  // Shutdown the request queue (rejects new pushes, wakes waiting workers)
  request_queue_->shutdown();
  
  // Wait for all inference workers to finish
  for (auto& worker : inference_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  inference_workers_.clear();
}

std::future<DetectionResult> Runtime::submit(const std::string& model_id, ImageInput image) {
  // Check if runtime is running
  if (!running_.load()) {
    throw VieError(ErrorCode::RuntimeNotStarted, "Runtime is not running");
  }
  
  // Resolve model to shared_ptr at submit time. This keeps the model alive
  // even if it is unloaded while the request is queued or executing.
  auto model = models_.get(model_id);
  if (!model) {
    throw VieError(ErrorCode::InvalidModelId, "Model '" + model_id + "' is not loaded");
  }
  
  // Create the request with pre-resolved model reference
  InferenceRequest request;
  request.id = next_request_id_.fetch_add(1);
  request.model_id = model_id;
  request.image = image;
  request.model = model;
  request.submitted_at = std::chrono::steady_clock::now();
  
  // Get the future before moving the request
  std::future<DetectionResult> future = request.promise.get_future();
  
  // Record submission
  metrics_->record_request_submitted();
  
  // Try to push to queue
  if (!request_queue_->push(std::move(request))) {
    metrics_->record_request_rejected();
    throw VieError(ErrorCode::RequestQueueFull, "Request queue is full or shutting down");
  }
  
  // Record acceptance and queue size
  metrics_->record_request_accepted();
  metrics_->record_queue_size(request_queue_->size());
  
  return future;
}

void Runtime::worker_loop() {
  // Continue processing requests until the queue is empty and shutdown
  while (true) {
    InferenceRequest request;
    
    // Wait for a request from the queue
    if (!request_queue_->wait_pop(request)) {
      // Queue is empty and shutdown, exit worker
      break;
    }
    
    // Record queue wait time
    const auto dequeue_time = std::chrono::steady_clock::now();
    const double queue_wait_ms = std::chrono::duration<double, std::milli>(
        dequeue_time - request.submitted_at).count();
    
    // Execute the inference
    const auto execution_start = std::chrono::steady_clock::now();
    bool success = false;
    
    try {
      // Use the pre-resolved model shared_ptr (keeps model alive even if
      // unloaded from ModelManager after submit time)
      if (!request.model) {
        throw VieError(ErrorCode::InvalidModelId, 
                       "Model '" + request.model_id + "' was unloaded before execution");
      }
      
      // Execute inference (single image, extract first result from batch)
      DetectionBatch batch = request.model->infer(request.image);
      if (batch.results.empty()) {
        throw VieError(ErrorCode::Internal, "Inference returned empty results");
      }
      
      // Complete the promise with the first result
      request.promise.set_value(std::move(batch.results[0]));
      success = true;
      
    } catch (...) {
      // Propagate exception through the promise
      try {
        request.promise.set_exception(std::current_exception());
      } catch (...) {
        // Promise already satisfied or broken, ignore
      }
    }
    
    // Record execution time and completion
    const auto execution_end = std::chrono::steady_clock::now();
    const double execution_ms = std::chrono::duration<double, std::milli>(
        execution_end - execution_start).count();
    const double e2e_ms = std::chrono::duration<double, std::milli>(
        execution_end - request.submitted_at).count();
    
    metrics_->record_request_completed(queue_wait_ms, execution_ms, e2e_ms, success);
    metrics_->record_queue_size(request_queue_->size());
  }
}

}  // namespace vie
