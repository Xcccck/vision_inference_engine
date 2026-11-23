#include "vie/backend/backend_registry.hpp"

namespace vie {

void BackendRegistry::register_factory(BackendKind kind, Factory factory) {
  if (!factory) {
    throw VieError(ErrorCode::InvalidArgument, "backend factory is empty");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  factories_[kind] = std::move(factory);
}

bool BackendRegistry::contains(BackendKind kind) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return factories_.find(kind) != factories_.end();
}

std::unique_ptr<IInferenceBackend> BackendRegistry::create(BackendKind kind) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = factories_.find(kind);
  if (found == factories_.end()) {
    throw VieError(ErrorCode::Unsupported,
                   std::string("no backend registered for ") + to_string(kind));
  }
  return found->second();
}

}  // namespace vie
