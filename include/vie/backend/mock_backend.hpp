#pragma once

#include "vie/backend/inference_backend.hpp"

namespace vie {

// Test double. Writes a fixed fp32 prefix [1, 2, 3, 4, 0.9, 0] into each
// declared output so a decoder can be checked without a real network.
class MockBackend : public IInferenceBackend {
 public:
  void prepare(const BackendConfig& config) override;
  BackendOutput execute(const BackendRequest& request) override;
  BackendInfo info() const override;

 private:
  BackendConfig config_{};
  bool prepared_ = false;
};

}  // namespace vie
