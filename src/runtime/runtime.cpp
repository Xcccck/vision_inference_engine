#include "vie/runtime/runtime.hpp"

#include "vie/core/error.hpp"

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
      models_(metrics_, config.max_loaded_models) {}

}  // namespace vie
