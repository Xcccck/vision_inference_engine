#pragma once

#include "vie/backend/inference_backend.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace vie {

class BackendRegistry {
 public:
  using Factory = std::function<std::unique_ptr<IInferenceBackend>()>;

  void register_factory(BackendKind kind, Factory factory);
  bool contains(BackendKind kind) const;
  std::unique_ptr<IInferenceBackend> create(BackendKind kind) const;

 private:
  mutable std::mutex mutex_;
  std::map<BackendKind, Factory> factories_;
};

}  // namespace vie
