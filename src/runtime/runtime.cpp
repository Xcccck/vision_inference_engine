#include "vie/runtime/runtime.hpp"

#include "vie/core/error.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <exception>
#include <utility>
#include <vector>

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
      request_queue_(std::make_unique<RequestQueue>(config.request_queue_capacity)),
      aggregator_(std::make_unique<DynamicBatchAggregator>(config.batch_aggregation, metrics_)),
      // Phase 9.3: device-aware execution scheduler replaces the single batch
      // execution queue. Fallback capacity reuses the existing
      // batch_execution_queue_capacity so default backpressure is unchanged.
      resource_mgr_(std::make_unique<ExecutionResourceManager>(
          config.scheduling, config.batch_execution_queue_capacity, metrics_)) {
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
  if (aggregator_thread_.joinable() || !execution_workers_.empty()) {
    return;
  }

  // Phase 9.2.2.1: Launch the single aggregator thread
  aggregator_thread_ = std::thread([this] { aggregator_loop(); });

  // Launch execution worker threads
  const std::size_t worker_count = std::max<std::size_t>(1, config_.inference_worker_count);
  for (std::size_t i = 0; i < worker_count; ++i) {
    execution_workers_.emplace_back([this] { execution_worker_loop(); });
  }
}

void Runtime::shutdown() {
  std::lock_guard<std::mutex> lock(shutdown_mutex_);
  if (!running_.load()) {
    return;
  }
  running_.store(false);

  // Phase 1: Shutdown the request queue (rejects new pushes, wakes aggregator)
  request_queue_->shutdown();

  // Phase 2: Wait for the aggregator to exit.
  // The aggregator drains the request queue, flushes pending groups, and
  // pushes all formed batches to exec_queue_ before exiting.
  if (aggregator_thread_.joinable()) {
    aggregator_thread_.join();
  }

  // Phase 3: Shutdown the resource scheduler (wakes workers after all batches
  // have been pushed by the aggregator). Workers still drain staged batches.
  resource_mgr_->shutdown();

  // Phase 4: Wait for execution workers to finish remaining batches
  for (auto& worker : execution_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  execution_workers_.clear();
}

std::future<DetectionResult> Runtime::submit(const std::string& model_id, ImageInput image) {
  // Check if runtime is running
  if (!running_.load()) {
    throw VieError(ErrorCode::RuntimeNotStarted, "Runtime is not running");
  }

  // Phase 9.2.2.3: CUDA async submit is not supported.
  // ImageInput::data for a CUDA image is a raw device pointer with no
  // ownership management. The caller may cudaFree() the buffer before the
  // async worker reads it, causing GPU-side use-after-free. Use the
  // synchronous InferenceEngine::infer() API for CUDA inputs instead.
  if (image.device != DeviceType::Cpu) {
    throw VieError(
        ErrorCode::Unsupported,
        std::string("Runtime::submit() does not support ") + to_string(image.device) +
            " device inputs: the asynchronous request pipeline cannot guarantee CUDA buffer "
            "lifetime. Use InferenceEngine::infer() for synchronous CUDA inference.");
  }

  // Resolve model to shared_ptr at submit time. This keeps the model alive
  // even if it is unloaded while the request is queued or executing.
  auto model = models_.get(model_id);
  if (!model) {
    throw VieError(ErrorCode::InvalidModelId, "Model '" + model_id + "' is not loaded");
  }

  // Phase 9.3: Register the model's execution resource with the scheduler so
  // its lane exists with the correct concurrency cap before any batch reaches
  // it, and so an undeterminable (Unknown) resource fails fast at submit rather
  // than being silently treated as CPU. A managed instance is always loaded,
  // so Unknown should not occur; it is rejected defensively.
  const ExecutionResource resource = model->execution_resource();
  if (!resource.valid()) {
    throw VieError(ErrorCode::Unsupported,
                   "Model '" + model_id + "' has an undeterminable execution resource; "
                   "it cannot be scheduled by Runtime::submit()");
  }
  resource_mgr_->register_resource(resource);

  // Create the request with pre-resolved model reference
  InferenceRequest request;
  request.id = next_request_id_.fetch_add(1);
  request.model_id = model_id;
  request.image = image;
  request.model = model;
  request.submitted_at = std::chrono::steady_clock::now();

  // Phase 9.2.2.2/11.1: Deep-copy CPU pixel data so the caller can safely
  // release their buffer immediately after submit() returns. Strided rows
  // (e.g. a cv::Mat ROI) are repacked, so the queued request never reads the
  // source's row padding or past its last valid pixel.
  request.adopt_cpu_pixels();

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

// Phase 9.2.2.1: Aggregator loop - ONLY forms batches, does NOT execute them.
// Formed batches are pushed to the execution queue for workers to process.
void Runtime::aggregator_loop() {
  while (true) {
    // Step 1: Try to flush any expired pending batch
    auto expired_batch = aggregator_->flush_expired();
    if (expired_batch.has_value()) {
      dispatch_batch(std::move(*expired_batch));
      continue;
    }

    // Step 2: Determine how long to wait for next request
    auto deadline = aggregator_->earliest_deadline();
    bool has_pending = (aggregator_->pending_count() > 0);

    InferenceRequest request;
    bool got_request = false;

    if (has_pending && deadline != std::chrono::steady_clock::time_point::max()) {
      got_request = request_queue_->wait_pop_until(deadline, request);
    } else {
      got_request = request_queue_->wait_pop(request);
    }

    // Step 3: If we got a request, add it to the aggregator
    if (got_request) {
      auto batch = aggregator_->add(std::move(request));
      if (batch.has_value()) {
        dispatch_batch(std::move(*batch));
      }
      continue;
    }

    // Step 4: wait_pop returned false
    if (!running_.load() && request_queue_->empty() && aggregator_->pending_count() == 0) {
      break;
    }

    // If queue is shut down, drain remaining requests and flush
    if (request_queue_->is_shutdown()) {
      while (request_queue_->wait_pop(request)) {
        auto batch = aggregator_->add(std::move(request));
        if (batch.has_value()) {
          dispatch_batch(std::move(*batch));
        }
      }
      auto remaining = aggregator_->flush_all();
      for (auto& batch : remaining) {
        dispatch_batch(std::move(batch));
      }
      break;
    }
  }
}

// Push a formed batch into its execution resource's lane. Blocks if the shared
// staging pool is full (backpressure). If the scheduler has been shut down, the
// batch is executed inline as a final fallback so its futures still complete.
void Runtime::dispatch_batch(BatchWorkItem batch) {
  // Determine the resource from the batch's model. A null/Unknown model cannot
  // be routed; execute_batch will fail all its promises, so run it inline.
  ExecutionResource resource;
  if (batch.model) {
    resource = batch.model->execution_resource();
  }
  if (!resource.valid() || resource_mgr_->is_shutdown()) {
    // Not routable, or the scheduler is already shut down (unreachable in the
    // normal ordering where the aggregator joins before scheduler shutdown).
    // `batch` is still intact here because we have not moved it yet.
    execute_batch(batch);
    return;
  }
  if (!resource_mgr_->push(resource, std::move(batch))) {
    // Unreachable in the current design: push() only returns false on shutdown,
    // which we just ruled out. If the invariant is ever broken the batch is
    // moved-from and its promises resolve via broken_promise.
    assert(false && "dispatch_batch: scheduler push failed unexpectedly");
  }
}

// Phase 9.2.2.1: Execution worker loop - pulls batches from the execution
// queue and runs inference. Multiple workers can execute batches for different
// ModelInstances in parallel.
void Runtime::execution_worker_loop() {
  while (true) {
    ScheduledBatch scheduled;
    // pop() blocks until some resource lane has a staged batch with a free
    // permit, then hands back the batch with its lease already held. Returns
    // false only once the scheduler is shut down and every lane is drained.
    if (!resource_mgr_->pop(scheduled)) {
      break;
    }
    // The lease is released automatically when `scheduled` leaves scope at the
    // end of this iteration, including if execute_batch throws.
    execute_batch(scheduled.item);
  }
}

void Runtime::execute_batch(BatchWorkItem& batch) {
  // Execute a formed batch via ModelInstance::infer() or infer_batch()
  const auto exec_start = std::chrono::steady_clock::now();
  const std::size_t batch_size = batch.requests.size();

  DetectionBatch result;
  std::exception_ptr failure;
  try {
    if (!batch.model) {
      throw VieError(ErrorCode::InvalidModelId, "Batch model reference is null");
    }

    // Build the image vector for infer_batch
    std::vector<ImageInput> images;
    images.reserve(batch_size);
    for (const auto& req : batch.requests) {
      images.push_back(req.image);
    }

    // Execute inference: use the async staging boundary. batch_size==1 goes
    // through infer_async() (compatible with fixed-batch models and, for CUDA
    // instances, performs the Runtime-owned CPU->CUDA H2D staging); batch_size>1
    // goes through infer_batch_async() (dynamic-batch CPU models; CUDA is
    // forced to size 1 upstream). CPU models take the unchanged synchronous
    // path inside these methods.
    if (batch_size == 1) {
      result = batch.model->infer_async(images[0]);
      metrics_->record_single_infer();
    } else {
      result = batch.model->infer_batch_async(images);
    }

    // Validate result count matches batch size
    if (result.results.size() != batch_size) {
      throw VieError(ErrorCode::Internal,
                     "infer_batch returned " + std::to_string(result.results.size()) +
                         " results for batch of " + std::to_string(batch_size));
    }
  } catch (...) {
    failure = std::current_exception();
    metrics_->record_batch_execution_failure();
  }

  const bool success = !failure;

  // Record metrics
  const auto exec_end = std::chrono::steady_clock::now();
  const double exec_ms = std::chrono::duration<double, std::milli>(exec_end - exec_start).count();

  // Phase 9.3: record every completion metric BEFORE resolving the futures.
  // A caller that observes a ready future via f.get() is then guaranteed to
  // see the matching metrics already applied. Fulfilling promises first let
  // f.get() outrun the counters, so a test snapshotting right after get()
  // could briefly read total_batches == 0 (a batch counter that lags the
  // completed future). This preserves results, exception propagation and the
  // "set exactly once" guarantee; it only establishes metrics as a
  // happens-before of future readiness.
  metrics_->record_batch_executed(batch_size);
  for (const auto& req : batch.requests) {
    const double queue_wait_ms = std::chrono::duration<double, std::milli>(
        exec_start - req.submitted_at).count();
    metrics_->record_request_completed(queue_wait_ms, exec_ms,
                                        std::chrono::duration<double, std::milli>(exec_end - req.submitted_at).count(),
                                        success);
  }
  metrics_->record_queue_size(request_queue_->size());

  // Distribute results (or the failure) to individual promises last.
  if (success) {
    for (std::size_t i = 0; i < batch_size; ++i) {
      try {
        batch.requests[i].promise.set_value(std::move(result.results[i]));
      } catch (...) {
        // Promise already satisfied or broken, ignore
      }
    }
  } else {
    for (std::size_t i = 0; i < batch_size; ++i) {
      try {
        batch.requests[i].promise.set_exception(failure);
      } catch (...) {
        // Promise already satisfied or broken, ignore
      }
    }
  }
}

}  // namespace vie
