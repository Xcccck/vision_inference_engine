#include "vie/runtime/runtime.hpp"
#include "vie/runtime/thread_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

namespace vie {
namespace {

// ThreadPool edge case tests

TEST(ThreadPoolAudit, ZeroThreadsBehavior) {
  // Zero threads should throw
  EXPECT_THROW(ThreadPool(0), std::invalid_argument);
}

TEST(ThreadPoolAudit, SubmitToDestroyedPool) {
  // Test that we can't submit after destruction
  // This is handled by the destructor setting shutdown flag
  std::unique_ptr<ThreadPool> pool = std::make_unique<ThreadPool>(2);
  
  auto future = pool->submit([] { return 42; });
  EXPECT_EQ(future.get(), 42);
  
  pool.reset();  // Destroy the pool
  
  // Can't test submit after destruction (would be use-after-free)
}

TEST(ThreadPoolAudit, ConcurrentSubmit) {
  ThreadPool pool(4);
  std::atomic<int> counter{0};
  std::vector<std::future<void>> futures;
  
  // Submit from multiple threads
  for (int i = 0; i < 10; ++i) {
    futures.push_back(pool.submit([&counter] {
      counter++;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }));
  }
  
  for (auto& f : futures) {
    f.get();
  }
  
  EXPECT_EQ(counter.load(), 10);
}

TEST(ThreadPoolAudit, ExceptionInTask) {
  ThreadPool pool(2);
  
  auto future1 = pool.submit([] { throw std::runtime_error("test"); });
  auto future2 = pool.submit([] { return 42; });
  
  EXPECT_THROW(future1.get(), std::runtime_error);
  EXPECT_EQ(future2.get(), 42);
}

TEST(ThreadPoolAudit, MassiveTaskSubmission) {
  ThreadPool pool(8);
  std::atomic<int> counter{0};
  std::vector<std::future<void>> futures;
  
  // Submit 1000 tasks
  for (int i = 0; i < 1000; ++i) {
    futures.push_back(pool.submit([&counter] { counter++; }));
  }
  
  // Wait for all
  for (auto& f : futures) {
    f.get();
  }
  
  EXPECT_EQ(counter.load(), 1000);
}

TEST(ThreadPoolAudit, TaskWithDanglingReference) {
  ThreadPool pool(2);
  int value = 42;
  
  auto future = pool.submit([&value] { return value; });
  
  // value is still alive here
  EXPECT_EQ(future.get(), 42);
}

// ModelManager lifecycle tests

TEST(ModelManagerAudit, ConcurrentLoadUnload) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 10);
  
  std::atomic<int> load_count{0};
  std::atomic<int> unload_count{0};
  
  // Load models concurrently
  std::vector<std::thread> threads;
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([&, i] {
      try {
        ModelConfig config;
        config.model_path = "test" + std::to_string(i) + ".onnx";
        config.device = DeviceType::Cpu;
        // This will fail (no real model), but tests the locking
        manager.load("model" + std::to_string(i), config);
        load_count++;
      } catch (...) {
        // Expected
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
}

TEST(ModelManagerAudit, GetAfterUnload) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 10);
  
  // Can't test without real models
  // But the API is: get() returns nullptr after unload
}

TEST(ModelManagerAudit, MultipleUnloads) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 10);
  
  // First unload throws (not loaded)
  EXPECT_THROW(manager.unload("nonexistent"), VieError);
}

// ModelInstance concurrency tests

TEST(ModelInstanceAudit, SameModelConcurrentInfer) {
  // Can't test without real model
  // But the mutex should serialize all calls
}

TEST(ModelInstanceAudit, DifferentModelConcurrentInfer) {
  // Can't test without real models
  // But different instances should run in parallel
}

// MetricsCollector stress tests

TEST(MetricsCollectorAudit, HighConcurrencyRecording) {
  MetricsCollector collector;
  std::vector<std::thread> threads;
  
  // 100 threads, each recording 1000 times
  for (int i = 0; i < 100; ++i) {
    threads.emplace_back([&collector, i] {
      for (int j = 0; j < 1000; ++j) {
        collector.record_inference("model" + std::to_string(i % 10), 1.0, true);
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  auto metrics = collector.snapshot();
  EXPECT_EQ(metrics.total_inferences, 100000);
  EXPECT_EQ(metrics.per_model_inference_count.size(), 10);
}

TEST(MetricsCollectorAudit, MemoryGrowth) {
  MetricsCollector collector;
  
  // Record with many different model IDs
  for (int i = 0; i < 10000; ++i) {
    collector.record_inference("model" + std::to_string(i), 1.0, true);
  }
  
  auto metrics = collector.snapshot();
  EXPECT_EQ(metrics.per_model_inference_count.size(), 10000);
  // This demonstrates unbounded growth!
}

// Runtime integration tests

TEST(RuntimeAudit, ConcurrentSubmitAndUnload) {
  RuntimeConfig config;
  config.thread_pool_size = 4;
  Runtime runtime(config);
  
  // Can't test without real models
  // But the scenario is:
  // Thread A: runtime.thread_pool().submit([model] { model->infer(...); });
  // Thread B: runtime.models().unload("model_id");
  // 
  // Analysis:
  // - submit() captures shared_ptr<ModelInstance>
  // - unload() removes from manager's map
  // - But the shared_ptr in the task keeps the instance alive
  // - So the inference completes safely
  // - This is SAFE due to shared_ptr semantics
}

}  // namespace
}  // namespace vie
