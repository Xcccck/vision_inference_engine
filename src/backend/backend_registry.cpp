#include "vie/backend/backend_registry.hpp"

namespace vie {

void BackendRegistry::register_factory(BackendKind kind, Factory factory) {
  if (!factory) {
    throw VieError(ErrorCode::InvalidArgument, "backend factory is empty");
  }
  factories_[kind] = std::move(factory);
}

bool BackendRegistry::contains(BackendKind kind) const {
  return factories_.find(kind) != factories_.end();
}

std::unique_ptr<IInferenceBackend> BackendRegistry::create(BackendKind kind) const {
  const auto found = factories_.find(kind);
  if (found == factories_.end()) {
    throw VieError(ErrorCode::Unsupported,
                   std::string("no backend registered for ") + to_string(kind));
  }
  return found->second();
}

}  // namespace vie
