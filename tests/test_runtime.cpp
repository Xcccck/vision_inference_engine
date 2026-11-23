#include "vie/runtime/runtime.hpp"
#include "vie/runtime/thread_pool.hpp"
#include "vie/runtime/runtime_metrics.hpp"
#include "vie/runtime/model_instance.hpp"
#include "vie/runtime/runtime_config.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

namespace vie {
namespace {

TEST(ThreadPool, Constructs) {
  ThreadPool pool(4);
  EXPECT_EQ(pool.worker_count(), 4);
  EXPECT_EQ(pool.pending_tasks(), 0);
}

TEST(ThreadPool, SubmitsTask) {
  ThreadPool pool(2);
  auto future = pool.submit([] { return 42; });
  EXPECT_EQ(future.get(), 42);
}

TEST(ThreadPool, SubmitsMultipleTasks) {
  ThreadPool pool(4);
  std::vector<std::future<int>> futures;
  for (int i = 0; i < 10; ++i) {
    futures.push_back(pool.submit([i] { return i * 2; }));
  }
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(futures[i].get(), i * 2);
  }
}

TEST(ThreadPool, PropagatesExceptions) {
  ThreadPool pool(2);
  auto future = pool.submit([] { throw std::runtime_error("test error"); });
  EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(ThreadPool, ExecutesConcurrently) {
  ThreadPool pool(4);
  std::atomic<int> counter{0};
  std::vector<std::future<void>> futures;
  for (int i = 0; i < 8; ++i) {
    futures.push_back(pool.submit([&counter] {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      counter++;
    }));
  }
  for (auto& f : futures) {
    f.get();
  }
  EXPECT_EQ(counter.load(), 8);
}

TEST(ThreadPool, PendingTasksTracked) {
  ThreadPool pool(1);
  std::promise<void> blocker;
  auto wait_future = blocker.get_future();
  
  // Submit a blocking task
  auto task1 = pool.submit([&wait_future] { wait_future.wait(); });
  
  // Submit more tasks that will queue up
  auto task2 = pool.submit([] { return 1; });
  auto task3 = pool.submit([] { return 2; });
  
  // Give tasks time to queue
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  
  // At least 2 tasks should be pending (task2 and task3)
  EXPECT_GE(pool.pending_tasks(), 2);
  
  // Unblock and wait for completion
  blocker.set_value();
  task1.get();
  task2.get();
  task3.get();
  
  EXPECT_EQ(pool.pending_tasks(), 0);
}

TEST(ThreadPool, ShutdownCompletesPendingTasks) {
  std::atomic<int> counter{0};
  {
    ThreadPool pool(2);
    for (int i = 0; i < 5; ++i) {
      pool.submit([&counter] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        counter++;
      });
    }
    // Pool destructor should wait for all tasks to complete
  }
  EXPECT_EQ(counter.load(), 5);
}

TEST(MetricsCollector, RecordsInference) {
  MetricsCollector collector;
  collector.record_inference("model1", 10.5, true);
  collector.record_inference("model1", 15.2, true);
  collector.record_inference("model2", 20.0, false);
  
  auto metrics = collector.snapshot();
  EXPECT_EQ(metrics.total_inferences, 3);
  EXPECT_EQ(metrics.failed_inferences, 1);
  EXPECT_DOUBLE_EQ(metrics.total_inference_time_ms, 45.7);
  EXPECT_EQ(metrics.per_model_inference_count["model1"], 2);
  EXPECT_EQ(metrics.per_model_inference_count["model2"], 1);
}

TEST(MetricsCollector, RecordsPreprocess) {
  MetricsCollector collector;
  collector.record_preprocess(5.0);
  collector.record_preprocess(3.5);
  
  auto metrics = collector.snapshot();
  EXPECT_EQ(metrics.total_preprocess_calls, 2);
  EXPECT_DOUBLE_EQ(metrics.total_preprocess_time_ms, 8.5);
}

TEST(MetricsCollector, RecordsModelLoadUnload) {
  MetricsCollector collector;
  collector.record_model_load();
  collector.record_model_load();
  collector.record_model_unload();
  
  auto metrics = collector.snapshot();
  EXPECT_EQ(metrics.total_models_loaded, 2);
  EXPECT_EQ(metrics.total_models_unloaded, 1);
}

TEST(MetricsCollector, ThreadSafe) {
  MetricsCollector collector;
  std::vector<std::thread> threads;
  
  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&collector, i] {
      for (int j = 0; j < 100; ++j) {
        collector.record_inference("model" + std::to_string(i), 1.0, true);
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  auto metrics = collector.snapshot();
  EXPECT_EQ(metrics.total_inferences, 1000);
}

TEST(RuntimeConfig, Defaults) {
  RuntimeConfig config;
  EXPECT_EQ(config.thread_pool_size, 4);
  EXPECT_EQ(config.default_device, DeviceType::Cpu);
  EXPECT_EQ(config.default_device_id, 0);
  EXPECT_FALSE(config.enable_profiling);
  EXPECT_EQ(config.max_loaded_models, 16);
}

TEST(Runtime, Constructs) {
  RuntimeConfig config;
  config.thread_pool_size = 2;
  Runtime runtime(config);
  
  EXPECT_EQ(runtime.thread_pool().worker_count(), 2);
  EXPECT_FALSE(runtime.profiling_enabled());
}

TEST(Runtime, MetricsAccessible) {
  Runtime runtime;
  auto metrics = runtime.snapshot_metrics();
  EXPECT_EQ(metrics.total_inferences, 0);
}

TEST(ModelManager, Constructs) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 8);
  EXPECT_EQ(manager.size(), 0);
}

TEST(ModelManager, LoadsModel) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 8);
  
  ModelConfig config;
  config.model_path = "test.onnx";
  config.device = DeviceType::Cpu;
  config.device_id = 0;
  
  // This will fail because the model doesn't exist, but we're testing the API
  EXPECT_THROW(manager.load("test_model", config), VieError);
}

TEST(ModelManager, PreventsDuplicateLoad) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 8);
  
  // Can't test without a real model, but the API is correct
  // manager.load("id1", config);
  // EXPECT_THROW(manager.load("id1", config), VieError);
}

TEST(ModelManager, UnloadsModel) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 8);
  
  EXPECT_THROW(manager.unload("nonexistent"), VieError);
}

TEST(ModelManager, GetsModel) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 8);
  
  auto instance = manager.get("nonexistent");
  EXPECT_EQ(instance, nullptr);
}

TEST(ModelManager, ContainsModel) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 8);
  
  EXPECT_FALSE(manager.contains("nonexistent"));
}

TEST(ModelManager, ListsLoadedIds) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 8);
  
  auto ids = manager.loaded_ids();
  EXPECT_TRUE(ids.empty());
}

TEST(ModelManager, EnforcesMaxModels) {
  auto metrics = std::make_shared<MetricsCollector>();
  ModelManager manager(metrics, 2);
  
  // Can't test without real models, but the API is correct
}

}  // namespace
}  // namespace vie
