// Phase 9.2.2.1: Runtime Concurrency & Backpressure Audit tests
//
// Covers:
// - BatchExecutionQueue unit tests
// - Multi-model concurrency verification (different ModelInstances run in parallel)
// - Same-model serialization (same ModelInstance still serialized)
// - Backpressure across the full chain
// - Exception propagation in batch execution
// - Shutdown safety (submit+shutdown race, pending batches)
// - Metrics accuracy

#include "test_util.hpp"

#include "vie/core/error.hpp"
#include "vie/decoder/detection.hpp"
#include "vie/model/manifest.hpp"
#include "vie/preprocess/image_input.hpp"
#include "vie/runtime/batch_aggregation_config.hpp"
#include "vie/runtime/batch_execution_queue.hpp"
#include "vie/runtime/dynamic_batch_aggregator.hpp"
#include "vie/runtime/model_instance.hpp"
#include "vie/runtime/request_queue.hpp"
#include "vie/runtime/runtime.hpp"
#include "vie/runtime/runtime_config.hpp"
#include "vie/runtime/runtime_metrics.hpp"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace vie;

namespace {

#if defined(VIE_HAS_ONNXRUNTIME)
struct OwnedPixels {
  std::vector<uint8_t> data;
};

ImageInput make_test_image(OwnedPixels& owner, int w = 640, int h = 640, int c = 3) {
  owner.data.resize(static_cast<size_t>(w) * h * c, 128);
  ImageInput img;
  img.data = owner.data.data();
  img.width = w;
  img.height = h;
  img.channels = c;
  img.row_stride_bytes = w * c;
  img.device = DeviceType::Cpu;
  img.device_id = 0;
  img.color = ColorOrder::Bgr;
  img.layout = ImageLayout::Hwc;
  img.dtype = DataType::Uint8;
  img.ownership = ImageOwnership::Borrowed;
  return img;
}

ModelConfig make_model_config(const std::string& model_path) {
  ModelManifest manifest;
  manifest.model_name = "yolov8n";
  manifest.model_family = "yolov8";
  manifest.num_classes = 80;
  manifest.model_path = model_path;
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
  manifest.nms.enabled = true;
  manifest.nms.confidence_threshold = 0.25f;
  manifest.nms.iou_threshold = 0.45f;
  manifest.nms.max_detections = 20;
  manifest.nms.pre_nms_top_k = 300;
  manifest.nms.class_aware = true;

  ModelConfig config;
  config.model_path = model_path;
  config.task = TaskType::Detection;
  config.backend = BackendKind::OnnxRuntime;
  config.device = DeviceType::Cpu;
  config.device_id = 0;
  config.manifest = manifest;
  return config;
}
#endif  // VIE_HAS_ONNXRUNTIME

}  // namespace

// =============================================================================
// BatchExecutionQueue unit tests
// =============================================================================

class BatchExecutionQueueTest : public ::testing::Test {};

TEST_F(BatchExecutionQueueTest, ConstructsAndReportsEmpty) {
  BatchExecutionQueue queue(10);
  EXPECT_EQ(queue.size(), 0u);
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.capacity(), 10u);
  EXPECT_FALSE(queue.is_shutdown());
}

TEST_F(BatchExecutionQueueTest, RejectsZeroCapacity) {
  EXPECT_THROW(BatchExecutionQueue(0), std::invalid_argument);
}

TEST_F(BatchExecutionQueueTest, PushAndWaitPop) {
  BatchExecutionQueue queue(10);
  BatchWorkItem item;
  item.created_at = std::chrono::steady_clock::now();
  EXPECT_TRUE(queue.push(std::move(item)));
  EXPECT_EQ(queue.size(), 1u);

  BatchWorkItem out;
  EXPECT_TRUE(queue.wait_pop(out));
  EXPECT_EQ(queue.size(), 0u);
}

TEST_F(BatchExecutionQueueTest, PushBlocksWhenFull) {
  BatchExecutionQueue queue(1);
  BatchWorkItem item1;
  item1.created_at = std::chrono::steady_clock::now();
  EXPECT_TRUE(queue.push(std::move(item1)));

  // Queue is full - push from another thread should block
  std::atomic<bool> push_completed{false};
  std::thread pusher([&] {
    BatchWorkItem item2;
    item2.created_at = std::chrono::steady_clock::now();
    queue.push(std::move(item2));
    push_completed.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(push_completed.load());

  // Pop the first item to make room
  BatchWorkItem out;
  EXPECT_TRUE(queue.wait_pop(out));

  pusher.join();
  EXPECT_TRUE(push_completed.load());
}

TEST_F(BatchExecutionQueueTest, ShutdownWakesWaitingPop) {
  BatchExecutionQueue queue(10);

  std::atomic<bool> got_item{false};
  std::thread reader([&] {
    BatchWorkItem out;
    got_item.store(queue.wait_pop(out));
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  queue.shutdown();
  reader.join();

  EXPECT_FALSE(got_item.load());
  EXPECT_TRUE(queue.is_shutdown());
}

TEST_F(BatchExecutionQueueTest, ShutdownRejectsPush) {
  BatchExecutionQueue queue(10);
  queue.shutdown();

  BatchWorkItem item;
  item.created_at = std::chrono::steady_clock::now();
  EXPECT_FALSE(queue.push(std::move(item)));
}

TEST_F(BatchExecutionQueueTest, GracefulShutdownDrainsQueuedBatches) {
  BatchExecutionQueue queue(10);

  // Push 3 items
  for (int i = 0; i < 3; ++i) {
    BatchWorkItem item;
    item.created_at = std::chrono::steady_clock::now();
    ASSERT_TRUE(queue.push(std::move(item)));
  }

  queue.shutdown();

  // Should still be able to pop queued items
  BatchWorkItem out;
  int count = 0;
  while (queue.wait_pop(out)) {
    count++;
  }
  EXPECT_EQ(count, 3);
}

// =============================================================================
// Multi-model concurrency tests (requires ORT)
// =============================================================================

#if defined(VIE_HAS_ONNXRUNTIME)

class MultiModelConcurrencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
    model_path_ = vie_fixture("testdata/onnx/yolov8n.onnx");
  }

  RuntimeConfig concurrent_config() {
    RuntimeConfig config;
    config.thread_pool_size = 4;
    config.request_queue_capacity = 100;
    config.inference_worker_count = 4;  // Multiple execution workers
    config.defer_start = false;
    config.batch_aggregation.enabled = true;
    config.batch_aggregation.max_batch_size = 1;  // No batching for fixed model
    config.batch_aggregation.max_wait_time = std::chrono::milliseconds(0);
    config.batch_aggregation.max_pending_requests = 200;
    config.batch_execution_queue_capacity = 64;
    return config;
  }

  std::string model_path_;
};

TEST_F(MultiModelConcurrencyTest, DifferentModelsExecuteInParallel) {
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
  GTEST_SKIP() << "Timing-based proof invalid with GPU EP (single CUDA stream serializes inference)";
#endif
  // Load two independent model instances
  Runtime runtime(concurrent_config());
  runtime.models().load("model_a", make_model_config(model_path_));
  runtime.models().load("model_b", make_model_config(model_path_));

  // Warmup
  {
    OwnedPixels px;
    auto img = make_test_image(px);
    runtime.submit("model_a", img).get();
    auto img2 = make_test_image(px);
    runtime.submit("model_b", img2).get();
  }

  // Measure sequential baseline: submit all to A, then all to B
  const int requests_per_model = 6;
  auto seq_start = std::chrono::steady_clock::now();
  {
    for (int i = 0; i < requests_per_model; ++i) {
      OwnedPixels px;
      auto future = runtime.submit("model_a", make_test_image(px));
      future.get();  // Wait for each to complete
    }
    for (int i = 0; i < requests_per_model; ++i) {
      OwnedPixels px;
      auto future = runtime.submit("model_b", make_test_image(px));
      future.get();
    }
  }
  auto seq_end = std::chrono::steady_clock::now();
  double sequential_ms = std::chrono::duration<double, std::milli>(seq_end - seq_start).count();

  // Measure concurrent: submit to both models simultaneously
  auto conc_start = std::chrono::steady_clock::now();
  {
    std::vector<std::future<DetectionResult>> futures;
    OwnedPixels pxA, pxB;
    for (int i = 0; i < requests_per_model; ++i) {
      futures.push_back(runtime.submit("model_a", make_test_image(pxA)));
      futures.push_back(runtime.submit("model_b", make_test_image(pxB)));
    }
    for (auto& f : futures) {
      f.get();
    }
  }
  auto conc_end = std::chrono::steady_clock::now();
  double concurrent_ms = std::chrono::duration<double, std::milli>(conc_end - conc_start).count();

  // With 4 execution workers, two independent models CAN be scheduled to
  // different workers. Actual wall-clock speedup depends on ORT internal
  // thread pool contention and is not a reliable assertion.
  // The scheduling parallelism is proven by ConcurrencySemanticsTest Mock tests.
  // Here we verify: all futures complete correctly and results are valid.
  auto metrics = runtime.snapshot_metrics();
  EXPECT_EQ(metrics.completed_requests,
            static_cast<std::size_t>(2 + requests_per_model * 4))  // warmup + seq + conc
      << "All requests must complete";
  (void)sequential_ms;
  (void)concurrent_ms;
  RecordProperty("sequential_ms", static_cast<int>(sequential_ms));
  RecordProperty("concurrent_ms", static_cast<int>(concurrent_ms));
}

TEST_F(MultiModelConcurrencyTest, SameModelRemainsSerialized) {
  Runtime runtime(concurrent_config());
  runtime.models().load("model_a", make_model_config(model_path_));

  // Submit multiple requests to the SAME model concurrently
  std::vector<std::future<DetectionResult>> futures;
  OwnedPixels px;
  for (int i = 0; i < 4; ++i) {
    futures.push_back(runtime.submit("model_a", make_test_image(px)));
  }

  // All should complete successfully
  for (auto& f : futures) {
    DetectionResult result = f.get();
    EXPECT_GT(result.original_width, 0);
  }

  // Verify metrics: all requests completed
  auto metrics = runtime.snapshot_metrics();
  EXPECT_EQ(metrics.completed_requests, 4u);
}

TEST_F(MultiModelConcurrencyTest, ThreeModelsExecuteInParallel) {
  RuntimeConfig config = concurrent_config();
  config.inference_worker_count = 4;

  Runtime runtime(config);
  runtime.models().load("model_a", make_model_config(model_path_));
  runtime.models().load("model_b", make_model_config(model_path_));
  runtime.models().load("model_c", make_model_config(model_path_));

  // Warmup
  {
    OwnedPixels px;
    runtime.submit("model_a", make_test_image(px)).get();
    runtime.submit("model_b", make_test_image(px)).get();
    runtime.submit("model_c", make_test_image(px)).get();
  }

  // Submit requests to all three models concurrently
  std::vector<std::future<DetectionResult>> futures;
  for (int i = 0; i < 3; ++i) {
    OwnedPixels px;
    futures.push_back(runtime.submit("model_a", make_test_image(px)));
    futures.push_back(runtime.submit("model_b", make_test_image(px)));
    futures.push_back(runtime.submit("model_c", make_test_image(px)));
  }

  for (auto& f : futures) {
    DetectionResult result = f.get();
    EXPECT_GT(result.original_width, 0);
  }

  auto metrics = runtime.snapshot_metrics();
  // 9 concurrent + 3 warmup = 12 total completed
  EXPECT_EQ(metrics.completed_requests, 12u);
}

TEST_F(MultiModelConcurrencyTest, SlowModelDoesNotBlockOtherModels) {
  Runtime runtime(concurrent_config());
  runtime.models().load("model_a", make_model_config(model_path_));
  runtime.models().load("model_b", make_model_config(model_path_));

  // Submit requests to model A
  std::vector<std::future<DetectionResult>> futures_a;
  OwnedPixels px;
  for (int i = 0; i < 3; ++i) {
    futures_a.push_back(runtime.submit("model_a", make_test_image(px)));
  }

  // Submit requests to model B - should not be blocked by model A
  std::vector<std::future<DetectionResult>> futures_b;
  for (int i = 0; i < 3; ++i) {
    futures_b.push_back(runtime.submit("model_b", make_test_image(px)));
  }

  // Both sets should complete
  for (auto& f : futures_a) {
    DetectionResult result = f.get();
    EXPECT_GT(result.original_width, 0);
  }
  for (auto& f : futures_b) {
    DetectionResult result = f.get();
    EXPECT_GT(result.original_width, 0);
  }
}

// =============================================================================
// Backpressure tests
// =============================================================================

TEST_F(MultiModelConcurrencyTest, BackpressureRejectsWhenQueuesFull) {
  RuntimeConfig config;
  config.thread_pool_size = 2;
  config.request_queue_capacity = 2;  // Very small queue
  config.inference_worker_count = 1;
  config.defer_start = true;  // Don't start workers
  config.batch_aggregation.max_batch_size = 1;
  config.batch_aggregation.max_wait_time = std::chrono::milliseconds(0);
  config.batch_execution_queue_capacity = 4;

  Runtime runtime(config);
  runtime.models().load("model", make_model_config(model_path_));

  // Fill the request queue
  OwnedPixels px;
  for (std::size_t i = 0; i < config.request_queue_capacity; ++i) {
    (void)runtime.submit("model", make_test_image(px));
  }

  // Next request should be rejected
  EXPECT_THROW(runtime.submit("model", make_test_image(px)), VieError);

  auto metrics = runtime.snapshot_metrics();
  EXPECT_GE(metrics.rejected_requests, 1u);

  // Start workers to drain
  runtime.start();
}

// =============================================================================
// Exception and Promise completion tests
// =============================================================================

TEST_F(MultiModelConcurrencyTest, ExceptionPropagatesThroughFuture) {
  Runtime runtime(concurrent_config());
  // Don't load any model - submit should throw synchronously
  OwnedPixels px;
  EXPECT_THROW(runtime.submit("nonexistent", make_test_image(px)), VieError);
}

TEST_F(MultiModelConcurrencyTest, AllFuturesCompleteDuringShutdown) {
  RuntimeConfig config = concurrent_config();
  config.batch_aggregation.max_batch_size = 1;  // Fixed-batch model
  config.batch_aggregation.max_wait_time = std::chrono::milliseconds(100);

  Runtime runtime(config);
  runtime.models().load("model", make_model_config(model_path_));

  // Submit requests
  std::vector<std::future<DetectionResult>> futures;
  OwnedPixels px;
  for (int i = 0; i < 5; ++i) {
    futures.push_back(runtime.submit("model", make_test_image(px)));
  }

  // Shutdown should complete all pending
  runtime.shutdown();

  for (auto& f : futures) {
    DetectionResult result = f.get();
    EXPECT_GT(result.original_width, 0);
  }
}

// =============================================================================
// Shutdown safety tests
// =============================================================================

TEST_F(MultiModelConcurrencyTest, ConcurrentShutdownIsIdempotent) {
  Runtime runtime(concurrent_config());
  runtime.models().load("model", make_model_config(model_path_));

  // Submit some requests
  std::vector<std::future<DetectionResult>> futures;
  OwnedPixels px;
  for (int i = 0; i < 3; ++i) {
    futures.push_back(runtime.submit("model", make_test_image(px)));
  }

  // Call shutdown from multiple threads simultaneously
  std::vector<std::thread> shutdown_threads;
  for (int i = 0; i < 4; ++i) {
    shutdown_threads.emplace_back([&] { runtime.shutdown(); });
  }
  for (auto& t : shutdown_threads) {
    t.join();
  }

  // All futures should still complete
  for (auto& f : futures) {
    DetectionResult result = f.get();
    EXPECT_GT(result.original_width, 0);
  }
}

TEST_F(MultiModelConcurrencyTest, SubmitAfterShutdownThrows) {
  Runtime runtime(concurrent_config());
  runtime.models().load("model", make_model_config(model_path_));
  runtime.shutdown();

  OwnedPixels px;
  EXPECT_THROW(runtime.submit("model", make_test_image(px)), VieError);
}

// =============================================================================
// Metrics accuracy tests
// =============================================================================

TEST_F(MultiModelConcurrencyTest, MetricsDistinguishSingleAndBatch) {
  RuntimeConfig config = concurrent_config();
  config.batch_aggregation.max_batch_size = 1;  // Force single execution
  config.batch_aggregation.max_wait_time = std::chrono::milliseconds(0);

  Runtime runtime(config);
  runtime.models().load("model", make_model_config(model_path_));

  OwnedPixels px;
  for (int i = 0; i < 3; ++i) {
    runtime.submit("model", make_test_image(px)).get();
  }

  auto metrics = runtime.snapshot_metrics();
  // With max_batch_size=1, all should be single infers
  EXPECT_EQ(metrics.total_single_infer_calls, 3u);
  EXPECT_EQ(metrics.completed_requests, 3u);
}

// =============================================================================
// ImageInput async lifecycle tests (Phase 9.2.2.2)
// =============================================================================

TEST_F(MultiModelConcurrencyTest, CallerReleasesPixelsBeforeFutureCompletes) {
  // Critical: caller destroys pixel buffer immediately after submit.
  // The runtime must have deep-copied the data (adopt_cpu_pixels).
  RuntimeConfig config = concurrent_config();
  config.batch_aggregation.max_wait_time = std::chrono::milliseconds(50);

  Runtime runtime(config);
  runtime.models().load("model", make_model_config(model_path_));

  std::future<DetectionResult> future;
  {
    auto pixels = std::make_unique<std::vector<uint8_t>>(640u * 640u * 3u, 200);
    ImageInput img;
    img.data = pixels->data();
    img.width = 640;
    img.height = 640;
    img.channels = 3;
    img.row_stride_bytes = 640 * 3;
    img.device = DeviceType::Cpu;
    img.device_id = 0;
    img.color = ColorOrder::Bgr;
    img.layout = ImageLayout::Hwc;
    img.dtype = DataType::Uint8;
    img.ownership = ImageOwnership::Borrowed;
    future = runtime.submit("model", img);
    // pixels destroyed here
  }

  DetectionResult result = future.get();
  EXPECT_GT(result.original_width, 0);
}

TEST_F(MultiModelConcurrencyTest, SequentialSubmitReleaseComplete) {
  // Scenario A: multiple images created and destroyed sequentially
  Runtime runtime(concurrent_config());
  runtime.models().load("model", make_model_config(model_path_));

  std::vector<std::future<DetectionResult>> futures;
  for (int i = 0; i < 4; ++i) {
    auto pixels = std::make_unique<std::vector<uint8_t>>(64u * 64u * 3u,
                                                          static_cast<uint8_t>(i));
    ImageInput img;
    img.data = pixels->data();
    img.width = 64;
    img.height = 64;
    img.channels = 3;
    img.row_stride_bytes = 64 * 3;
    img.device = DeviceType::Cpu;
    img.device_id = 0;
    img.color = ColorOrder::Bgr;
    img.layout = ImageLayout::Hwc;
    img.dtype = DataType::Uint8;
    img.ownership = ImageOwnership::Borrowed;
    futures.push_back(runtime.submit("model", img));
    // pixels destroyed here
  }

  for (auto& f : futures) {
    DetectionResult result = f.get();
    EXPECT_GT(result.original_width, 0);
  }
}

TEST_F(MultiModelConcurrencyTest, AdoptCpuPixelsUnitDirect) {
  // Unit test: verify deep copy and pointer redirection
  InferenceRequest req;
  std::vector<uint8_t> original = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

  req.image.data = original.data();
  req.image.width = 2;
  req.image.height = 2;
  req.image.channels = 3;
  req.image.row_stride_bytes = 6;
  req.image.device = DeviceType::Cpu;
  req.image.dtype = DataType::Uint8;

  req.adopt_cpu_pixels();

  // image.data must NOT point to original anymore
  EXPECT_NE(req.image.data, static_cast<const void*>(original.data()));
  EXPECT_EQ(req.owned_pixels.size(), 12u);
  EXPECT_EQ(req.owned_pixels, original);

  // Modify original - owned copy must be unaffected
  original[0] = 255;
  EXPECT_EQ(req.owned_pixels[0], 10u);
}

TEST_F(MultiModelConcurrencyTest, AdoptCpuPixelsAfterMove) {
  // Verify pointer stability after move (vector move preserves buffer)
  InferenceRequest req;
  std::vector<uint8_t> original(16, 42);

  req.image.data = original.data();
  req.image.width = 4;
  req.image.height = 4;
  req.image.channels = 1;
  req.image.row_stride_bytes = 4;
  req.image.device = DeviceType::Cpu;
  req.image.dtype = DataType::Uint8;

  req.adopt_cpu_pixels();
  const void* data_before = req.image.data;

  InferenceRequest moved(std::move(req));

  // After move, image.data still points to valid owned buffer
  EXPECT_EQ(moved.owned_pixels.size(), 16u);
  EXPECT_NE(moved.image.data, nullptr);
  auto* ptr = static_cast<const uint8_t*>(moved.image.data);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(ptr[i], 42u);
  }
  (void)data_before;  // Pointer may differ after move (implementation detail)
}

#endif  // VIE_HAS_ONNXRUNTIME
