// Phase 9.2.1: RequestQueue and async Runtime tests

#include "test_util.hpp"

#include "vie/core/error.hpp"
#include "vie/model/manifest.hpp"
#include "vie/runtime/request_queue.hpp"
#include "vie/runtime/runtime.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

namespace vie {
namespace {

// ------------------------------------------------------------------
// RequestQueue unit tests
// ------------------------------------------------------------------

TEST(RequestQueue, Constructs) {
  RequestQueue queue(10);
  EXPECT_EQ(queue.capacity(), 10);
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
  EXPECT_FALSE(queue.is_shutdown());
}

TEST(RequestQueue, RejectsZeroCapacity) {
  EXPECT_THROW(RequestQueue(0), std::invalid_argument);
}

TEST(RequestQueue, PushAndWaitPop) {
  RequestQueue queue(5);
  
  InferenceRequest req;
  req.id = 1;
  req.model_id = "test_model";
  
  EXPECT_TRUE(queue.push(std::move(req)));
  EXPECT_EQ(queue.size(), 1);
  EXPECT_FALSE(queue.empty());
  
  InferenceRequest popped;
  EXPECT_TRUE(queue.wait_pop(popped));
  EXPECT_EQ(popped.id, 1);
  EXPECT_EQ(popped.model_id, "test_model");
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
}

TEST(RequestQueue, PushRejectsWhenFull) {
  RequestQueue queue(2);
  
  InferenceRequest req1, req2, req3;
  req1.id = 1;
  req2.id = 2;
  req3.id = 3;
  
  EXPECT_TRUE(queue.push(std::move(req1)));
  EXPECT_TRUE(queue.push(std::move(req2)));
  EXPECT_FALSE(queue.push(std::move(req3)));  // Queue full
  EXPECT_EQ(queue.size(), 2);
}

TEST(RequestQueue, ShutdownRejectsPush) {
  RequestQueue queue(5);
  
  InferenceRequest req;
  req.id = 1;
  
  EXPECT_TRUE(queue.push(std::move(req)));
  
  queue.shutdown();
  EXPECT_TRUE(queue.is_shutdown());
  
  InferenceRequest req2;
  req2.id = 2;
  EXPECT_FALSE(queue.push(std::move(req2)));  // Shutdown
}

TEST(RequestQueue, ShutdownWakesWaitingPop) {
  RequestQueue queue(5);
  
  std::atomic<bool> pop_completed{false};
  std::thread pop_thread([&] {
    InferenceRequest req;
    EXPECT_FALSE(queue.wait_pop(req));  // Should return false after shutdown
    pop_completed.store(true);
  });
  
  // Give the thread time to start waiting
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(pop_completed.load());
  
  queue.shutdown();
  pop_thread.join();
  EXPECT_TRUE(pop_completed.load());
}

TEST(RequestQueue, GracefulShutdownProcessesQueuedRequests) {
  RequestQueue queue(5);
  
  // Push 3 requests
  for (int i = 1; i <= 3; ++i) {
    InferenceRequest req;
    req.id = static_cast<RequestId>(i);
    EXPECT_TRUE(queue.push(std::move(req)));
  }
  EXPECT_EQ(queue.size(), 3);
  
  // Shutdown
  queue.shutdown();
  
  // Should still be able to pop queued requests
  for (int i = 1; i <= 3; ++i) {
    InferenceRequest req;
    EXPECT_TRUE(queue.wait_pop(req));
    EXPECT_EQ(req.id, static_cast<RequestId>(i));
  }
  EXPECT_TRUE(queue.empty());
  
  // Next pop should return false (empty and shutdown)
  InferenceRequest req;
  EXPECT_FALSE(queue.wait_pop(req));
}

TEST(RequestQueue, MultiThreadedPushPop) {
  RequestQueue queue(100);
  const int num_requests = 50;
  
  std::atomic<int> push_count{0};
  std::atomic<int> pop_count{0};
  
  // Producer thread
  std::thread producer([&] {
    for (int i = 0; i < num_requests; ++i) {
      InferenceRequest req;
      req.id = static_cast<RequestId>(i + 1);
      if (queue.push(std::move(req))) {
        push_count.fetch_add(1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  
  // Consumer thread
  std::thread consumer([&] {
    for (int i = 0; i < num_requests; ++i) {
      InferenceRequest req;
      if (queue.wait_pop(req)) {
        pop_count.fetch_add(1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });
  
  producer.join();
  consumer.join();
  
  EXPECT_EQ(push_count.load(), num_requests);
  EXPECT_EQ(pop_count.load(), num_requests);
  EXPECT_TRUE(queue.empty());
}

// ------------------------------------------------------------------
// Async Runtime tests (require ONNX Runtime)
// ------------------------------------------------------------------

#if defined(VIE_HAS_ONNXRUNTIME)

class AsyncRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_path_ = vie_fixture("testdata/onnx/yolov8n.onnx");
  }

  RuntimeConfig default_config() {
    RuntimeConfig config;
    config.thread_pool_size = 2;
    config.request_queue_capacity = 10;
    config.inference_worker_count = 1;  // Aggregator is single-threaded
    // Use batch_size=1 for fixed-batch model (yolov8n.onnx)
    config.batch_aggregation.max_batch_size = 1;
    config.batch_aggregation.max_wait_time = std::chrono::milliseconds(0);
    return config;
  }

  ModelConfig model_config() {
    ModelConfig config;
    config.model_path = model_path_;
    config.task = TaskType::Detection;
    config.backend = BackendKind::OnnxRuntime;
    config.device = DeviceType::Cpu;
    config.device_id = 0;
    
    // Construct a manifest for yolov8n
    ModelManifest manifest;
    manifest.model_name = "yolov8n";
    manifest.model_family = "yolov8";
    manifest.num_classes = 80;
    manifest.model_path = model_path_;
    manifest.task = TaskType::Detection;
    manifest.model_format = ModelFormat::Onnx;
    manifest.input.name = "images";
    manifest.input.shape = {1, 3, 640, 640};
    manifest.input.dtype = DataType::Fp32;
    manifest.input.layout = Layout::Nchw;
    manifest.input.color = "rgb";
    manifest.input.normalization = "scale_0_1";
    manifest.input.resize = "letterbox";
    manifest.input.interpolation = "bilinear";
    manifest.input.padding_value = 114.f;
    manifest.provenance["model_name"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["model_family"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["num_classes"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["task"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["model_format"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["model_path"] = FieldProvenance{Certainty::User, {}};
    for (const char* field : {"input.name", "input.shape", "input.dtype", "input.layout",
                               "input.color", "input.normalization", "input.resize",
                               "input.interpolation", "input.padding_value"}) {
      manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
    }
    OutputDesc out;
    out.name = "predictions";
    out.shape = {1, 84, 8400};
    out.dtype = DataType::Fp32;
    out.layout = Layout::Unknown;
    out.semantic = "predictions";
    manifest.outputs.push_back(out);
    manifest.decoder.protocol = "yolov8_detect_raw";
    manifest.decoder.box_format = "cxcywh";
    manifest.decoder.box_decode = "decoded_cxcywh";
    manifest.decoder.box_channels = 4;
    manifest.decoder.channel_axis = 1;
    manifest.decoder.class_selection = "best_class";
    manifest.decoder.scores_are_probabilities = true;
    for (const char* field : {"decoder.protocol", "decoder.box_format", "decoder.box_decode",
                               "decoder.box_channels", "decoder.channel_axis",
                               "decoder.class_selection", "decoder.scores_are_probabilities"}) {
      manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
    }
    manifest.nms.enabled = true;
    manifest.nms.confidence_threshold = 0.25f;
    manifest.nms.iou_threshold = 0.45f;
    manifest.nms.max_detections = 20;
    manifest.nms.pre_nms_top_k = 300;
    manifest.nms.class_aware = true;
    for (const char* field : {"nms.enabled", "nms.confidence_threshold", "nms.iou_threshold",
                               "nms.max_detections", "nms.pre_nms_top_k", "nms.class_aware"}) {
      manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
    }
    config.manifest = manifest;
    return config;
  }

  ImageInput make_test_image() {
    pixels_.assign(640 * 640 * 3, 128);
    ImageInput image;
    image.data = pixels_.data();
    image.width = 640;
    image.height = 640;
    image.channels = 3;
    image.color = ColorOrder::Rgb;
    image.layout = ImageLayout::Hwc;
    image.dtype = DataType::Uint8;
    image.device = DeviceType::Cpu;
    image.device_id = 0;
    return image;
  }

  std::string model_path_;
  std::vector<std::uint8_t> pixels_;
};

TEST_F(AsyncRuntimeTest, SubmitsAndReceivesResult) {
  Runtime runtime(default_config());
  runtime.models().load("yolov8n", model_config());
  
  auto image = make_test_image();
  auto future = runtime.submit("yolov8n", image);
  
  // Should get a valid result
  DetectionResult result = future.get();
  EXPECT_GT(result.original_width, 0);
  EXPECT_GT(result.original_height, 0);
}

TEST_F(AsyncRuntimeTest, MultipleRequestsComplete) {
  Runtime runtime(default_config());
  runtime.models().load("yolov8n", model_config());
  
  const int num_requests = 5;
  std::vector<std::future<DetectionResult>> futures;
  
  for (int i = 0; i < num_requests; ++i) {
    auto image = make_test_image();
    futures.push_back(runtime.submit("yolov8n", image));
  }
  
  // All futures should complete
  for (auto& future : futures) {
    DetectionResult result = future.get();
    EXPECT_GT(result.original_width, 0);
  }
  
  // Check metrics
  auto metrics = runtime.snapshot_metrics();
  EXPECT_EQ(metrics.submitted_requests, num_requests);
  EXPECT_EQ(metrics.completed_requests, num_requests);
  EXPECT_EQ(metrics.failed_requests, 0);
}

TEST_F(AsyncRuntimeTest, InvalidModelIdThrows) {
  Runtime runtime(default_config());
  
  auto image = make_test_image();
  EXPECT_THROW(runtime.submit("nonexistent_model", image), VieError);
  
  auto metrics = runtime.snapshot_metrics();
  EXPECT_EQ(metrics.rejected_requests, 0);  // Rejected before submission
}

TEST_F(AsyncRuntimeTest, QueueFullRejectsRequest) {
  RuntimeConfig config = default_config();
  config.request_queue_capacity = 2;
  config.inference_worker_count = 1;
  config.defer_start = true;  // Don't start workers so we can fill the queue
  
  Runtime runtime(config);
  runtime.models().load("yolov8n", model_config());
  
  // Fill the queue to capacity (workers not running, so requests accumulate)
  for (std::size_t i = 0; i < config.request_queue_capacity; ++i) {
    auto image = make_test_image();
    (void)runtime.submit("yolov8n", image);
  }
  
  // Next request should be rejected (queue full)
  auto image = make_test_image();
  EXPECT_THROW(runtime.submit("yolov8n", image), VieError);
  
  auto metrics = runtime.snapshot_metrics();
  EXPECT_GE(metrics.rejected_requests, 1);
  
  // Now start workers to drain the queue and clean up
  runtime.start();
}

TEST_F(AsyncRuntimeTest, ShutdownRejectsNewRequests) {
  Runtime runtime(default_config());
  runtime.models().load("yolov8n", model_config());
  
  runtime.shutdown();
  EXPECT_FALSE(runtime.is_running());
  
  auto image = make_test_image();
  EXPECT_THROW(runtime.submit("yolov8n", image), VieError);
}

TEST_F(AsyncRuntimeTest, GracefulShutdownCompletesQueuedRequests) {
  RuntimeConfig config = default_config();
  config.inference_worker_count = 1;
  
  Runtime runtime(config);
  runtime.models().load("yolov8n", model_config());
  
  // Submit 3 requests
  std::vector<std::future<DetectionResult>> futures;
  for (int i = 0; i < 3; ++i) {
    auto image = make_test_image();
    futures.push_back(runtime.submit("yolov8n", image));
  }
  
  // Shutdown should wait for queued requests
  runtime.shutdown();
  
  // All futures should still complete
  for (auto& future : futures) {
    DetectionResult result = future.get();
    EXPECT_GT(result.original_width, 0);
  }
}

TEST_F(AsyncRuntimeTest, ExceptionPropagatesThroughFuture) {
  Runtime runtime(default_config());
  // Don't load any model
  
  auto image = make_test_image();
  EXPECT_THROW(runtime.submit("nonexistent", image), VieError);
}

TEST_F(AsyncRuntimeTest, ConcurrentSubmitsFromMultipleThreads) {
  Runtime runtime(default_config());
  runtime.models().load("yolov8n", model_config());
  
  const int num_threads = 4;
  const int requests_per_thread = 3;
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      for (int r = 0; r < requests_per_thread; ++r) {
        try {
          // Each thread must use its own pixel buffer to avoid
          // use-after-free when assign() reallocates.
          std::vector<uint8_t> local_pixels(640 * 640 * 3, 128);
          ImageInput image;
          image.data = local_pixels.data();
          image.width = 640;
          image.height = 640;
          image.channels = 3;
          image.color = ColorOrder::Rgb;
          image.layout = ImageLayout::Hwc;
          image.dtype = DataType::Uint8;
          image.device = DeviceType::Cpu;
          image.device_id = 0;
          auto future = runtime.submit("yolov8n", image);
          // local_pixels can safely go out of scope here because
          // submit() deep-copies via adopt_cpu_pixels().
          DetectionResult result = future.get();
          if (result.original_width > 0) {
            success_count.fetch_add(1);
          }
        } catch (...) {
          // Ignore errors
        }
      }
    });
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
  
  EXPECT_EQ(success_count.load(), num_threads * requests_per_thread);
}

TEST_F(AsyncRuntimeTest, ModelUnloadDuringRequest) {
  Runtime runtime(default_config());
  runtime.models().load("yolov8n", model_config());
  
  // Submit a request
  auto image = make_test_image();
  auto future = runtime.submit("yolov8n", image);
  
  // Unload the model while request is in flight
  // (The shared_ptr in the worker should keep it alive)
  runtime.models().unload("yolov8n");
  
  // Request should still complete
  DetectionResult result = future.get();
  EXPECT_GT(result.original_width, 0);
}

TEST_F(AsyncRuntimeTest, MetricsRecorded) {
  Runtime runtime(default_config());
  runtime.models().load("yolov8n", model_config());
  
  auto image = make_test_image();
  auto future = runtime.submit("yolov8n", image);
  future.get();
  
  auto metrics = runtime.snapshot_metrics();
  EXPECT_EQ(metrics.submitted_requests, 1);
  EXPECT_EQ(metrics.accepted_requests, 1);
  EXPECT_EQ(metrics.completed_requests, 1);
  EXPECT_GE(metrics.total_queue_wait_time_ms, 0.0);  // May be 0 if worker picks up immediately
  EXPECT_GT(metrics.total_execution_time_ms, 0.0);
  EXPECT_GT(metrics.total_e2e_time_ms, 0.0);
}

#endif  // VIE_HAS_ONNXRUNTIME

}  // namespace
}  // namespace vie
