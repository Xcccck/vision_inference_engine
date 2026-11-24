#pragma once

#include "vie/core/device_context.hpp"

#include <cstddef>
#include <string>

namespace vie {

// Configuration for the Runtime system
struct RuntimeConfig {
  std::size_t thread_pool_size = 4;
  DeviceType default_device = DeviceType::Cpu;
  int default_device_id = 0;
  bool enable_profiling = false;
  std::size_t max_loaded_models = 16;
  // Phase 9.2.1: Async request queue configuration
  std::size_t request_queue_capacity = 1024;
  std::size_t inference_worker_count = 2;
  bool defer_start = false;  // If true, constructor won't auto-start workers
};

}  // namespace vie
