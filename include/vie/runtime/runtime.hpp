#pragma once

#include "vie/model/model_config.hpp"
#include "vie/runtime/batch_execution_queue.hpp"
#include "vie/runtime/dynamic_batch_aggregator.hpp"
#include "vie/runtime/execution_resource_manager.hpp"
#include "vie/runtime/model_instance.hpp"
#include "vie/runtime/request_queue.hpp"
#include "vie/runtime/runtime_config.hpp"
#include "vie/runtime/runtime_metrics.hpp"
#include "vie/runtime/thread_pool.hpp"

#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace vie {

// Manages multiple model instances and their lifecycle
class ModelManager {
 public:
  explicit ModelManager(std::shared_ptr<MetricsCollector> metrics, std::size_t max_models = 16);
  ~ModelManager() = default;

  ModelManager(const ModelManager&) = delete;
  ModelManager& operator=(const ModelManager&) = delete;

  // Load a model and return its instance
  std::shared_ptr<ModelInstance> load(const std::string& id, const ModelConfig& config);

  // Unload a model by id
  void unload(const std::string& id);

  // Get a loaded model instance
  std::shared_ptr<ModelInstance> get(const std::string& id) const;

  // Check if a model is loaded
  bool contains(const std::string& id) const;

  // Get all loaded model ids
  std::vector<std::string> loaded_ids() const;

  // Get count of loaded models
  std::size_t size() const;

 private:
  mutable std::mutex mutex_;
  std::map<std::string, std::shared_ptr<ModelInstance>> instances_;
  std::shared_ptr<MetricsCollector> metrics_;
  std::size_t max_models_;
};

// Main Runtime class that coordinates thread pool and model management
class Runtime {
 public:
  explicit Runtime(const RuntimeConfig& config = RuntimeConfig{});
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  // Access the thread pool
  ThreadPool& thread_pool() noexcept { return thread_pool_; }
  const ThreadPool& thread_pool() const noexcept { return thread_pool_; }

  // Access the model manager
  ModelManager& models() noexcept { return models_; }
  const ModelManager& models() const noexcept { return models_; }

  // Access metrics
  std::shared_ptr<MetricsCollector> metrics() const noexcept { return metrics_; }
  RuntimeMetrics snapshot_metrics() const { return metrics_->snapshot(); }

  // Configuration
  const RuntimeConfig& config() const noexcept { return config_; }
  bool profiling_enabled() const noexcept { return config_.enable_profiling; }

  // Phase 9.2.1: Async request submission
  // Submit an inference request and return a future for the result.
  // The request is queued and executed by a background worker.
  // Throws VieError on submission failure (queue full, shutdown, invalid model).
  //
  // Phase 9.2.2.3: Only CPU ImageInput is supported by this async path.
  // image.device != DeviceType::Cpu is rejected synchronously with
  // ErrorCode::Unsupported before the request reaches the model lookup or
  // the RequestQueue: the pipeline cannot guarantee a caller-owned CUDA
  // device buffer stays valid until the background worker reads it. Use
  // ModelInstance::infer()/infer_batch() or InferenceEngine::infer() for
  // synchronous CUDA inference.
  std::future<DetectionResult> submit(const std::string& model_id, ImageInput image);

  // Phase 12: per-request inference policy. The effective configuration
  // (request > model default > usable manifest value > framework default) is
  // resolved and frozen at submit time. Mutating `options` after this call
  // cannot affect the queued request. Requests whose effective configs differ
  // are never aggregated into the same dynamic batch.
  std::future<DetectionResult> submit(const std::string& model_id, ImageInput image,
                                      const InferenceOptions& options);

  // Start the runtime (launch inference workers). Called automatically in constructor.
  void start();

  // Initiate graceful shutdown. Rejects new requests and waits for queued requests to complete.
  void shutdown();

  // Check if the runtime is running.
  bool is_running() const noexcept { return running_.load(); }

 private:
  void aggregator_loop();
  void execution_worker_loop();
  void launch_workers();
  void execute_batch(BatchWorkItem& batch);
  void dispatch_batch(BatchWorkItem batch);

  RuntimeConfig config_;
  ThreadPool thread_pool_;
  std::shared_ptr<MetricsCollector> metrics_;
  ModelManager models_;

  // Phase 9.2.1: Async request queue
  std::unique_ptr<RequestQueue> request_queue_;

  // Phase 9.2.2: Dynamic batch aggregator (single thread)
  std::unique_ptr<DynamicBatchAggregator> aggregator_;

  // Phase 9.2.2.1 / 9.3: Batch staging + device-aware admission control. The
  // aggregator pushes formed batches into per-resource lanes; execution workers
  // pop a batch only when its resource has a free permit, so a saturated GPU
  // device cannot head-of-line-block CPU or other devices. Replaces the single
  // BatchExecutionQueue as the Runtime's execution hand-off point.
  std::unique_ptr<ExecutionResourceManager> resource_mgr_;

  // Threads
  std::thread aggregator_thread_;
  std::vector<std::thread> execution_workers_;

  std::atomic<bool> running_{false};
  std::atomic<RequestId> next_request_id_{1};
  std::mutex shutdown_mutex_;  // Serializes concurrent shutdown() calls
};

}  // namespace vie
