#pragma once

#include "vie/model/model_config.hpp"
#include "vie/runtime/model_instance.hpp"
#include "vie/runtime/runtime_config.hpp"
#include "vie/runtime/runtime_metrics.hpp"
#include "vie/runtime/thread_pool.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <string>
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

// Main runtime class that coordinates thread pool and model management
class Runtime {
 public:
  explicit Runtime(const RuntimeConfig& config = RuntimeConfig{});
  ~Runtime() = default;

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

 private:
  RuntimeConfig config_;
  ThreadPool thread_pool_;
  std::shared_ptr<MetricsCollector> metrics_;
  ModelManager models_;
};

}  // namespace vie
