#pragma once

#include "vie/core/device_context.hpp"

#include <cstddef>
#include <string>

namespace vie {

// Configuration for the runtime system
struct RuntimeConfig {
  std::size_t thread_pool_size = 4;
  DeviceType default_device = DeviceType::Cpu;
  int default_device_id = 0;
  bool enable_profiling = false;
  std::size_t max_loaded_models = 16;
};

}  // namespace vie
