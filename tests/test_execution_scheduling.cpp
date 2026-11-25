// Phase 9.3: Device-aware Execution Scheduling tests.
//
// Section 1 (ExecutionResource) and Section 2 (ExecutionResourceManager) are
// backend-agnostic and run in every build (A/B/C/D), including CUDA-OFF
// builds: the scheduler models resources as values and does not require a real
// GPU. CUDA resource ISOLATION/ADMISSION semantics are proven here against the
// manager directly (a "Mock Resource Manager"), NOT against multiple physical
// GPUs. Section 3 exercises the Runtime integration where a real backend exists.

#include "test_util.hpp"

#include "vie/core/device_context.hpp"
#include "vie/core/error.hpp"
#include "vie/diagnostics/ort_profile_parser.hpp"
#include "vie/runtime/cuda_input_staging.hpp"
#include "vie/runtime/dynamic_batch_aggregator.hpp"
#include "vie/runtime/execution_resource.hpp"
#include "vie/runtime/execution_resource_manager.hpp"
#include "vie/runtime/execution_scheduling_config.hpp"
#include "vie/runtime/model_instance.hpp"
#include "vie/runtime/runtime.hpp"
#include "vie/runtime/runtime_config.hpp"
#include "vie/runtime/runtime_metrics.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "vie/preprocess/image_input.hpp"

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include <cuda_runtime.h>
#endif

using namespace vie;

namespace {

BatchWorkItem make_item(const std::string& model_id, int count = 1) {
  BatchWorkItem item;
  item.created_at = std::chrono::steady_clock::now();
  for (int i = 0; i < count; ++i) {
    InferenceRequest req;
    req.id = static_cast<RequestId>(i + 1);
    req.model_id = model_id;
    item.requests.push_back(std::move(req));
  }
  return item;
}

constexpr ExecutionResource kCpu{ExecutionResourceType::Cpu, 0};
ExecutionResource cuda0() { return ExecutionResource{ExecutionResourceType::Cuda, 0}; }
ExecutionResource cuda1() { return ExecutionResource{ExecutionResourceType::Cuda, 1}; }

}  // namespace

// =============================================================================
// 1. ExecutionResource identification
// =============================================================================

TEST(ExecutionResourceTest, CpuDeviceMapsToCpuResource) {
  DeviceContext cpu(DeviceType::Cpu, 0);
  ExecutionResource r = execution_resource_from_device(cpu);
  EXPECT_EQ(r.type, ExecutionResourceType::Cpu);
  EXPECT_EQ(r.device_id, 0);
  EXPECT_TRUE(r.valid());
  EXPECT_STREQ(to_string(r.type), "cpu");
}

TEST(ExecutionResourceTest, CudaDeviceMapsToCudaResourceWithDeviceId) {
  // Uses the (DeviceType, device_id) overload so this runs in every build; in a
  // CUDA-OFF build DeviceContext(Cuda, ...) would throw at construction.
  ExecutionResource r = execution_resource_from(DeviceType::Cuda, 2);
  EXPECT_EQ(r.type, ExecutionResourceType::Cuda);
  EXPECT_EQ(r.device_id, 2);
  EXPECT_TRUE(r.valid());
  EXPECT_EQ(r.key(), "cuda:2");
}

TEST(ExecutionResourceTest, DifferentDeviceIdsAreDistinctResources) {
  EXPECT_NE(cuda0(), cuda1());
  EXPECT_EQ(cuda0(), cuda0());
  EXPECT_NE(kCpu, cuda0());
}

TEST(ExecutionResourceTest, UnknownIsValidFalseAndKeyUnknown) {
  ExecutionResource unknown;  // default-constructed
  EXPECT_EQ(unknown.type, ExecutionResourceType::Unknown);
  EXPECT_FALSE(unknown.valid());
  EXPECT_EQ(unknown.key(), "unknown");
}

TEST(ExecutionResourceTest, DefaultEngineGivesCpuNeverUnknown) {
  // A default DeviceContext reports the committed CPU device (device()==Cpu).
  // ModelInstance::execution_resource() guards on loaded() to return Unknown
  // for a never-loaded engine, so the derivation itself never invents Unknown.
  DeviceContext def;
  EXPECT_EQ(def.device(), DeviceType::Cpu);
  EXPECT_EQ(execution_resource_from_device(def).type, ExecutionResourceType::Cpu);
}

// =============================================================================
// 1b. CUDA input staging size/layout checks (Phase 9.3.2, §9.1 CPU-only).
//
// cuda_stage_byte_count() is CUDA-free and allocation-free, so these size /
// stride / overflow rules are proven in every build (A/B/C/D) without a GPU.
// They are the single source of truth CudaInputBuffer::stage() uses to size the
// H2D transfer, so validating them here covers the CPU-side half of the staging
// contract even where no CUDA runtime is linked.
// =============================================================================

namespace {
ImageInput make_stage_image(int w, int h, int c, std::vector<std::uint8_t>& storage,
                            std::size_t row_stride = 0) {
  const std::size_t pitch = row_stride != 0 ? row_stride : static_cast<std::size_t>(w) * c;
  storage.resize(pitch * static_cast<std::size_t>(h), 7);
  ImageInput img;
  img.data = storage.data();
  img.width = w;
  img.height = h;
  img.channels = c;
  img.dtype = DataType::Uint8;
  img.row_stride_bytes = row_stride;
  img.device = DeviceType::Cpu;
  return img;
}
}  // namespace

TEST(CudaInputStagingUnitTest, TightRowByteCountMatchesWidthHeightChannels) {
  std::vector<std::uint8_t> storage;
  ImageInput img = make_stage_image(64, 32, 3, storage);
  // No explicit stride -> width*channels*height, dtype Uint8 (1 byte).
  EXPECT_EQ(cuda_stage_byte_count(img), 64u * 3u * 32u);
}

TEST(CudaInputStagingUnitTest, ExplicitRowStrideIsHonored) {
  std::vector<std::uint8_t> storage;
  // Padded rows: pitch 256 > packed 64*3=192, so bytes = pitch*height.
  ImageInput img = make_stage_image(64, 32, 3, storage, /*row_stride=*/256);
  EXPECT_EQ(cuda_stage_byte_count(img), 256u * 32u);
}

TEST(CudaInputStagingUnitTest, StrideSmallerThanPackedRowThrows) {
  std::vector<std::uint8_t> storage;
  ImageInput img = make_stage_image(64, 32, 3, storage, /*row_stride=*/100);
  EXPECT_THROW(cuda_stage_byte_count(img), VieError);
}

TEST(CudaInputStagingUnitTest, NullDataThrows) {
  ImageInput img;
  img.width = 8;
  img.height = 8;
  img.channels = 3;
  img.dtype = DataType::Uint8;
  img.data = nullptr;
  EXPECT_THROW(cuda_stage_byte_count(img), VieError);
}

TEST(CudaInputStagingUnitTest, NonPositiveDimensionsThrow) {
  std::vector<std::uint8_t> storage;
  EXPECT_THROW(cuda_stage_byte_count(make_stage_image(0, 8, 3, storage)), VieError);
  EXPECT_THROW(cuda_stage_byte_count(make_stage_image(8, 0, 3, storage)), VieError);
  EXPECT_THROW(cuda_stage_byte_count(make_stage_image(8, 8, 0, storage)), VieError);
  // A negative dimension cannot be built through the storage-backed helper (the
  // allocation itself would be malformed), so describe the image directly.
  ImageInput negative;
  negative.data = storage.data();  // non-null
  negative.width = -4;
  negative.height = 8;
  negative.channels = 3;
  negative.dtype = DataType::Uint8;
  EXPECT_THROW(cuda_stage_byte_count(negative), VieError);
}

TEST(CudaInputStagingUnitTest, UnsupportedDtypeThrows) {
  std::vector<std::uint8_t> storage;
  ImageInput img = make_stage_image(8, 8, 3, storage);
  img.dtype = DataType::Unknown;
  EXPECT_THROW(cuda_stage_byte_count(img), VieError);
}

TEST(CudaInputStagingUnitTest, DimensionOverflowThrows) {
  std::vector<std::uint8_t> storage;
  ImageInput img = make_stage_image(8, 8, 3, storage);
  // An explicit row stride near SIZE_MAX times any height > 1 must overflow the
  // final row_bytes*height multiply rather than wrap silently.
  img.row_stride_bytes = std::numeric_limits<std::size_t>::max() - 1;
  img.height = 2;
  EXPECT_THROW(cuda_stage_byte_count(img), VieError);
}

TEST(CudaInputStagingUnitTest, BufferIsMoveOnlyAndDefaultEmpty) {
  // Ownership contract: never copyable, movable, and a default buffer is empty.
  static_assert(!std::is_copy_constructible<CudaInputBuffer>::value, "must not be copyable");
  static_assert(!std::is_copy_assignable<CudaInputBuffer>::value, "must not be copy-assignable");
  static_assert(std::is_move_constructible<CudaInputBuffer>::value, "must be move-constructible");
  static_assert(std::is_move_assignable<CudaInputBuffer>::value, "must be move-assignable");
  CudaInputBuffer empty;
  EXPECT_FALSE(empty.valid());
  EXPECT_EQ(empty.device_bytes(), 0u);
}

// =============================================================================
// 2. ExecutionResourceManager admission / isolation / lifecycle
// =============================================================================

TEST(ExecutionResourceManagerTest, InvalidConfigZeroCudaCapThrows) {
  ExecutionSchedulingConfig cfg;
  cfg.cuda_max_concurrent_per_device = 0;  // 0 must NOT mean unlimited
  EXPECT_THROW(ExecutionResourceManager(cfg, 16), VieError);
}

TEST(ExecutionResourceManagerTest, UnknownResourceRejected) {
  ExecutionSchedulingConfig cfg;
  ExecutionResourceManager mgr(cfg, 16);
  ExecutionResource unknown;
  EXPECT_THROW(mgr.register_resource(unknown), VieError);
  // push with Unknown returns false without touching state.
  EXPECT_FALSE(mgr.push(unknown, make_item("m")));
}

TEST(ExecutionResourceManagerTest, CpuLaneIsUnbounded) {
  ExecutionSchedulingConfig cfg;
  cfg.cpu_max_concurrent = 0;  // unbounded
  ExecutionResourceManager mgr(cfg, 64);
  mgr.register_resource(kCpu);
  EXPECT_EQ(mgr.cap(kCpu), SIZE_MAX);
}

TEST(ExecutionResourceManagerTest, CudaCapEnforcedSerializesSameDevice) {
  // Two batches on the same CUDA device with cap=1: only one may be active at a
  // time. Verified with a lease-holding worker + an atomic active counter, not
  // sleep timing.
  ExecutionSchedulingConfig cfg;
  cfg.cuda_max_concurrent_per_device = 1;
  ExecutionResourceManager mgr(cfg, 64);
  mgr.register_resource(cuda0());

  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(mgr.push(cuda0(), make_item("m", 1)));
  }

  std::atomic<int> observed_concurrency{0};
  std::atomic<int> max_observed{0};
  std::atomic<bool> stop{false};

  auto worker = [&] {
    while (!stop.load()) {
      ScheduledBatch sb;
      if (!mgr.pop(sb)) {
        break;
      }
      int cur = observed_concurrency.fetch_add(1) + 1;
      int prev = max_observed.load();
      while (cur > prev && !max_observed.compare_exchange_weak(prev, cur)) {
      }
      // Hold the permit briefly, then release via lease destruction.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      observed_concurrency.fetch_sub(1);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) threads.emplace_back(worker);

  // Let all 4 drain, then stop.
  while (mgr.queued(cuda0()) > 0 || mgr.active(cuda0()) > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  stop.store(true);
  mgr.shutdown();
  for (auto& t : threads) t.join();

  EXPECT_EQ(max_observed.load(), 1) << "cap=1 must never allow 2 concurrent on same device";
  EXPECT_EQ(mgr.max_active(cuda0()), 1u);
}

TEST(ExecutionResourceManagerTest, DifferentCudaDevicesAreIndependent) {
  // cuda:0 and cuda:1 each cap 1, but together can run 2 concurrently because
  // their lanes and permits are independent.
  ExecutionSchedulingConfig cfg;
  cfg.cuda_max_concurrent_per_device = 1;
  ExecutionResourceManager mgr(cfg, 64);
  mgr.register_resource(cuda0());
  mgr.register_resource(cuda1());

  ASSERT_TRUE(mgr.push(cuda0(), make_item("a")));
  ASSERT_TRUE(mgr.push(cuda1(), make_item("b")));

  ScheduledBatch s0, s1;
  ASSERT_TRUE(mgr.pop(s0));  // takes one
  ASSERT_TRUE(mgr.pop(s1));  // must take the other device, not block
  // Both active simultaneously across devices.
  std::size_t active0 = mgr.active(s0.lease.resource());
  std::size_t active1 = mgr.active(s1.lease.resource());
  EXPECT_EQ(active0 + active1, 2u) << "different devices must not share a permit";
  // Release via lease scope end.
}

TEST(ExecutionResourceManagerTest, LeaseReleasesOnScopeExit) {
  ExecutionSchedulingConfig cfg;
  cfg.cuda_max_concurrent_per_device = 1;
  ExecutionResourceManager mgr(cfg, 64);
  mgr.register_resource(cuda0());
  ASSERT_TRUE(mgr.push(cuda0(), make_item("m")));

  {
    ScheduledBatch sb;
    ASSERT_TRUE(mgr.pop(sb));
    EXPECT_EQ(mgr.active(cuda0()), 1u);
  }  // lease destructor runs here
  EXPECT_EQ(mgr.active(cuda0()), 0u);
}

TEST(ExecutionResourceManagerTest, LeaseReleasesOnException) {
  // Simulate execute throwing: a function-scope ScheduledBatch unwound by an
  // exception must still release its permit exactly once.
  ExecutionSchedulingConfig cfg;
  cfg.cuda_max_concurrent_per_device = 1;
  ExecutionResourceManager mgr(cfg, 64);
  mgr.register_resource(cuda0());
  ASSERT_TRUE(mgr.push(cuda0(), make_item("m")));

  bool caught = false;
  try {
    ScheduledBatch sb;
    if (mgr.pop(sb)) {
      throw std::runtime_error("inference failed");
    }
  } catch (const std::exception&) {
    caught = true;
  }
  EXPECT_TRUE(caught);
  EXPECT_EQ(mgr.active(cuda0()), 0u) << "permit must not leak after an exception";
}

TEST(ExecutionResourceManagerTest, CpuNotBlockedBySaturatedCudaLane) {
  // Head-of-line blocking proof at the scheduler level: a CUDA lane pinned at
  // its cap must not prevent a CPU batch from being popped and executed.
  ExecutionSchedulingConfig cfg;
  cfg.cuda_max_concurrent_per_device = 1;
  ExecutionResourceManager mgr(cfg, 64);
  mgr.register_resource(cuda0());
  mgr.register_resource(kCpu);

  // Occupy the only CUDA permit.
  ASSERT_TRUE(mgr.push(cuda0(), make_item("gpu")));
  ASSERT_TRUE(mgr.push(cuda0(), make_item("gpu2")));  // will be stuck in lane
  ScheduledBatch gpu_hold;
  ASSERT_TRUE(mgr.pop(gpu_hold));
  ASSERT_EQ(gpu_hold.lease.resource(), cuda0());

  // Now submit CPU work; a CPU pop must succeed while CUDA is saturated.
  ASSERT_TRUE(mgr.push(kCpu, make_item("cpu")));
  ScheduledBatch cpu_batch;
  ASSERT_TRUE(mgr.pop(cpu_batch));
  EXPECT_EQ(cpu_batch.lease.resource(), kCpu)
      << "CPU work must be schedulable while a CUDA lane is saturated";
}

TEST(ExecutionResourceManagerTest, ShutdownDrainsAllAcceptedBatches) {
  ExecutionSchedulingConfig cfg;
  cfg.cuda_max_concurrent_per_device = 1;
  ExecutionResourceManager mgr(cfg, 64);
  mgr.register_resource(cuda0());
  for (int i = 0; i < 5; ++i) ASSERT_TRUE(mgr.push(cuda0(), make_item("m")));

  mgr.shutdown();

  // Even after shutdown, staged batches must still be poppable so their futures
  // complete; pop() returns false only once fully drained.
  int drained = 0;
  ScheduledBatch sb;
  while (mgr.pop(sb)) {
    ++drained;  // lease released at end of each iteration
  }
  EXPECT_EQ(drained, 5) << "shutdown must not strand accepted batches";
  EXPECT_EQ(mgr.total_queued(), 0u);
}

TEST(ExecutionResourceManagerTest, BackpressureCapsTotalStaged) {
  // Shared staging capacity bounds total queued across lanes; a producer cannot
  // exceed it, and blocking push resumes after pops free slots.
  ExecutionSchedulingConfig cfg;
  cfg.lane_capacity = 4;
  ExecutionResourceManager mgr(cfg, 64);
  mgr.register_resource(kCpu);

  for (int i = 0; i < 4; ++i) ASSERT_TRUE(mgr.push(kCpu, make_item("m")));
  EXPECT_EQ(mgr.total_queued(), 4u);

  // The 5th push must block because the pool is full.
  std::atomic<bool> pushed{false};
  std::thread producer([&] {
    mgr.push(kCpu, make_item("m"));
    pushed.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_FALSE(pushed.load()) << "push must block at capacity";

  // Free one slot; producer should then complete.
  ScheduledBatch sb;
  ASSERT_TRUE(mgr.pop(sb));
  producer.join();
  EXPECT_TRUE(pushed.load());
}

// =============================================================================
// 3. Runtime integration (needs a real backend to load a model)
// =============================================================================

#if defined(VIE_HAS_ONNXRUNTIME)

class ExecutionSchedulingRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override { model_path_ = vie_fixture("testdata/onnx/yolov8n.onnx"); }

  ModelConfig model_config(BackendKind backend = BackendKind::OnnxRuntime,
                           DeviceType device = DeviceType::Cpu, int device_id = 0) {
    ModelConfig config;
    config.model_path = model_path_;
    config.task = TaskType::Detection;
    config.backend = backend;
    config.device = device;
    config.device_id = device_id;
    // yolov8n.onnx has a few CPU-assigned memcpy nodes, so a CUDA session needs
    // EP fallback to LOAD (the same way the existing CUDA backend tests run it).
    // This is independent of the scheduling questions below; the strict
    // "all nodes on CUDAExecutionProvider" proof lives in the identity.onnx
    // backend test, and the async CPU-image boundary is device-match based.
    config.cuda_cpu_ep_fallback = (device == DeviceType::Cuda);

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
    for (const char* f : {"model_name", "model_family", "num_classes", "task", "model_format",
                          "model_path", "input.name", "input.shape", "input.dtype", "input.layout",
                          "input.color", "input.normalization", "input.resize", "input.interpolation",
                          "input.padding_value"}) {
      manifest.provenance[f] = FieldProvenance{Certainty::User, {}};
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
    for (const char* f : {"decoder.protocol", "decoder.box_format", "decoder.box_decode",
                          "decoder.box_channels", "decoder.channel_axis", "decoder.class_selection",
                          "decoder.scores_are_probabilities", "outputs.0.name", "outputs.0.shape",
                          "outputs.0.dtype", "outputs.0.semantic"}) {
      manifest.provenance[f] = FieldProvenance{Certainty::User, {}};
    }
    manifest.nms.enabled = true;
    manifest.nms.confidence_threshold = 0.25f;
    manifest.nms.iou_threshold = 0.45f;
    manifest.nms.max_detections = 20;
    manifest.nms.pre_nms_top_k = 300;
    manifest.nms.class_aware = true;
    for (const char* f : {"nms.enabled", "nms.confidence_threshold", "nms.iou_threshold",
                          "nms.max_detections", "nms.pre_nms_top_k", "nms.class_aware"}) {
      manifest.provenance[f] = FieldProvenance{Certainty::User, {}};
    }
    config.manifest = manifest;
    return config;
  }

  ImageInput cpu_image() {
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

  RuntimeConfig sched_config() {
    RuntimeConfig config;
    config.thread_pool_size = 4;
    config.inference_worker_count = 4;
    config.request_queue_capacity = 256;
    config.batch_execution_queue_capacity = 64;
    config.batch_aggregation.enabled = true;
    config.batch_aggregation.max_batch_size = 1;
    config.batch_aggregation.max_wait_time = std::chrono::milliseconds(0);
    return config;
  }

  std::string model_path_;
  std::vector<std::uint8_t> pixels_;
};

TEST_F(ExecutionSchedulingRuntimeTest, CpuModelsScheduleAndComplete) {
  Runtime runtime(sched_config());
  auto model = runtime.models().load("det", model_config());

  // The loaded CPU model must expose a CPU execution resource.
  EXPECT_EQ(model->execution_resource().type, ExecutionResourceType::Cpu);

  auto img = cpu_image();
  std::vector<std::future<DetectionResult>> futures;
  for (int i = 0; i < 8; ++i) futures.push_back(runtime.submit("det", img));
  for (auto& f : futures) {
    auto res = f.get();
    EXPECT_GT(res.original_width, 0);
  }

  // execute_batch resolves each promise before recording completion metrics,
  // so f.get() can briefly outrun the counter. shutdown() joins every worker,
  // guaranteeing all metrics are recorded before the snapshot (no sleep needed).
  runtime.shutdown();
  auto m = runtime.snapshot_metrics();
  EXPECT_EQ(m.completed_requests, 8u);
  // CPU scheduling admissions must be recorded and attributed to the cpu lane.
  EXPECT_GE(m.total_resource_admissions, 1u);
  auto it = m.per_resource_admissions.find("cpu");
  ASSERT_NE(it, m.per_resource_admissions.end());
  EXPECT_GE(it->second, 1u);
}

TEST_F(ExecutionSchedulingRuntimeTest, CpuLaneNeverSerializesDifferentInstances) {
  // Two independent CPU instances on the default (unbounded) CPU lane must all
  // complete; Phase 9.3 must not have introduced a global CPU serial lock.
  Runtime runtime(sched_config());
  runtime.models().load("a", model_config());
  runtime.models().load("b", model_config());

  auto img = cpu_image();
  std::vector<std::future<DetectionResult>> futures;
  for (int i = 0; i < 6; ++i) {
    futures.push_back(runtime.submit("a", img));
    futures.push_back(runtime.submit("b", img));
  }
  for (auto& f : futures) f.get();
  // Join workers so completion metrics are fully recorded before asserting.
  runtime.shutdown();
  EXPECT_EQ(runtime.snapshot_metrics().completed_requests, 12u);
}

TEST_F(ExecutionSchedulingRuntimeTest, CudaInputSubmitStillRejected) {
  // Phase 9.2.2.3 rejection must survive the Phase 9.3 scheduling changes.
  RuntimeConfig config = sched_config();
  config.defer_start = true;
  Runtime runtime(config);
  ImageInput cuda_img;
  cuda_img.device = DeviceType::Cuda;
  cuda_img.data = nullptr;
  cuda_img.width = 64;
  cuda_img.height = 64;
  cuda_img.channels = 3;
  cuda_img.dtype = DataType::Uint8;
  try {
    runtime.submit("det", cuda_img);
    FAIL() << "CUDA submit must be rejected";
  } catch (const VieError& e) {
    EXPECT_EQ(e.code(), ErrorCode::Unsupported);
  }
}

// =============================================================================
// 4. Real CUDA Runtime scheduling + input staging integration (Build D only).
//
// Phase 9.3.1 established Conclusion B: the scheduler routed a CPU-image batch
// to the real cuda:N lane and released the permit, but execution FAILED at the
// preprocess boundary because a CPU image was handed to a CUDA engine (no H2D).
//
// Phase 9.3.2 closes that gap. ModelInstance::infer_async()/infer_batch_async()
// (used only by the async worker path) now stage the Runtime-owned CPU image
// onto the committed CUDA device through CudaInputBuffer before calling the
// engine. These tests prove the full asynchronous data path: a CPU ImageInput
// submitted to Runtime drives a genuine CUDA ModelInstance to a real, non-empty
// DetectionResult, the CUDA backend actually executes (execution_device==Cuda,
// no silent CPU fallback), the buffer lifetime spans the call, and the permit is
// released. The synchronous infer() API and the submit-time external-CUDA-image
// ban are unchanged and still verified here.
// =============================================================================
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)

namespace {
// Owns a genuine device buffer so the synchronous CUDA inference test can feed
// an on-device ImageInput (the caller-owned-buffer contract from 9.2.2.3).
struct DeviceBuffer {
  std::uint8_t* ptr = nullptr;
  std::vector<std::uint8_t> host;
  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
};

// Loads the real COCO bus.ppm (P6, 810x1080, RGB) that yolov8n reliably
// detects a bus in at conf>=0.25, so the async CUDA E2E can assert a NON-EMPTY
// result rather than an empty decode on a synthetic image.
ImageInput load_bus(std::vector<std::uint8_t>& pixels) {
  std::ifstream image_file(vie_fixture("testdata/images/bus.ppm"), std::ios::binary);
  std::string magic;
  int width = 0;
  int height = 0;
  int max_value = 0;
  image_file >> magic >> width >> height >> max_value;
  image_file.get();
  EXPECT_EQ(magic, "P6");
  pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u);
  image_file.read(reinterpret_cast<char*>(pixels.data()),
                  static_cast<std::streamsize>(pixels.size()));
  EXPECT_TRUE(image_file);
  ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = ColorOrder::Rgb;
  image.layout = ImageLayout::Hwc;
  image.dtype = DataType::Uint8;
  image.device = DeviceType::Cpu;
  image.device_id = 0;
  return image;
}
}  // namespace

// Reuses the CPU fixture's manifest/config/image builders; only the device and
// backend differ, so the CUDA model is loaded through the exact same validated
// manifest path as the CPU models above.
class CudaSchedulingRuntimeTest : public ExecutionSchedulingRuntimeTest {};

TEST_F(CudaSchedulingRuntimeTest, LoadedCudaModelReportsCudaResource) {
  Runtime runtime(sched_config());
  auto model = runtime.models().load(
      "cuda_det", model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));
  const ExecutionResource r = model->execution_resource();
  EXPECT_EQ(r.type, ExecutionResourceType::Cuda);
  EXPECT_EQ(r.device_id, 0);
  EXPECT_EQ(r.key(), "cuda:0");
}

TEST_F(CudaSchedulingRuntimeTest, CudaBackendExecutesSynchronouslyOnRealGpu) {
  // Proves the CUDA-loaded ModelInstance really runs its inference through the
  // CUDA-configured engine on a genuine on-device image (H2D done by the
  // caller) -> CUDA preprocess -> OnnxRuntimeCuda backend -> decoder, returning
  // a result mapped back to the 640x640 input space and tagged execution_device
  // == Cuda. yolov8n loads with cuda_cpu_ep_fallback=true (a few memcpy nodes),
  // so the strict "every node on CUDAExecutionProvider" guarantee is provided by
  // the identity.onnx backend test; here we only need the CUDA path to execute.
  Runtime runtime(sched_config());
  auto model = runtime.models().load(
      "cuda_det", model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));

  const std::size_t bytes = 640 * 640 * 3;
  DeviceBuffer buf;
  buf.host.assign(bytes, 127);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&buf.ptr), bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(buf.ptr, buf.host.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

  ImageInput img;
  img.data = buf.ptr;
  img.width = 640;
  img.height = 640;
  img.channels = 3;
  img.color = ColorOrder::Rgb;
  img.layout = ImageLayout::Hwc;
  img.dtype = DataType::Uint8;
  img.device = DeviceType::Cuda;
  img.device_id = 0;

  DetectionBatch batch = model->infer(img);
  ASSERT_EQ(batch.results.size(), 1u);
  EXPECT_EQ(batch.results[0].original_width, 640);
  EXPECT_EQ(batch.results[0].original_height, 640);
  // The decoder reports the execution device it actually ran on: a real CUDA
  // run must say Cuda, not a silent CPU fallback.
  EXPECT_EQ(batch.results[0].execution_device, DeviceType::Cuda);
}

TEST_F(CudaSchedulingRuntimeTest, CudaModelCpuImageIsStagedAndSucceedsAsync) {
  // Phase 9.3.2: the async boundary now stages the Runtime-owned CPU image onto
  // cuda:0, runs the real CUDA backend, and returns a valid result. The permit
  // is granted to the cuda lane and released, and the request is counted as a
  // SUCCESS (no preprocess failure, as Phase 9.3.1 Conclusion B required).
  Runtime runtime(sched_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));
  auto f = runtime.submit("cuda_det", cpu_image());
  DetectionResult res;  // must not throw now
  ASSERT_NO_THROW(res = f.get());
  runtime.shutdown();

  auto m = runtime.snapshot_metrics();
  auto it = m.per_resource_admissions.find("cuda:0");
  ASSERT_NE(it, m.per_resource_admissions.end());  // admitted to the CUDA lane
  EXPECT_GE(it->second, 1u);
  EXPECT_EQ(m.completed_requests, 1u);
  EXPECT_EQ(m.failed_requests, 0u);  // staging fixed the preprocess rejection
  // The result really came from the CUDA engine, not a CPU fallback.
  EXPECT_EQ(res.original_width, 640);
  EXPECT_EQ(res.original_height, 640);
  EXPECT_EQ(res.execution_device, DeviceType::Cuda);
}

TEST_F(CudaSchedulingRuntimeTest, CpuAndCudaLanesAreIndependentInRuntime) {
  // With a real CUDA lane present, CPU work must still flow and the CUDA
  // request must now SUCCEED through staging. Proves cross-resource routing,
  // that CPU requests are not blocked behind the cuda lane, and that neither
  // lane's staging affects the other. All 5 requests complete, none fail.
  Runtime runtime(sched_config());
  runtime.models().load("cpu_det", model_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));

  std::vector<std::future<DetectionResult>> cpu_futures;
  for (int i = 0; i < 4; ++i) cpu_futures.push_back(runtime.submit("cpu_det", cpu_image()));
  auto cuda_future = runtime.submit("cuda_det", cpu_image());

  for (auto& f : cpu_futures) f.get();  // succeed, not blocked by the cuda lane
  DetectionResult cuda_res;
  ASSERT_NO_THROW(cuda_res = cuda_future.get());
  runtime.shutdown();

  EXPECT_EQ(cuda_res.execution_device, DeviceType::Cuda);
  auto m = runtime.snapshot_metrics();
  EXPECT_EQ(m.completed_requests, 5u);
  EXPECT_EQ(m.failed_requests, 0u);
  EXPECT_NE(m.per_resource_admissions.find("cpu"), m.per_resource_admissions.end());
  EXPECT_NE(m.per_resource_admissions.find("cuda:0"), m.per_resource_admissions.end());
}

// -----------------------------------------------------------------------------
// 9.3 Real CUDA Runtime E2E: a genuine detection model returns a NON-EMPTY
// DetectionResult through the asynchronous submit() path on a real GPU.
// -----------------------------------------------------------------------------
TEST_F(CudaSchedulingRuntimeTest, AsyncCpuImageDrivesRealCudaDetectionNonEmpty) {
  Runtime runtime(sched_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));

  std::vector<std::uint8_t> pixels;
  ImageInput bus = load_bus(pixels);
  auto f = runtime.submit("cuda_det", bus);
  DetectionResult res;
  ASSERT_NO_THROW(res = f.get());
  runtime.shutdown();

  // Real bus image -> yolov8n must find the bus (matches the CPU baseline in
  // test_integration_regression.cpp). Non-empty proves the H2D actually moved
  // the pixels and the CUDA backend decoded them, not an empty synthetic run.
  EXPECT_EQ(res.execution_device, DeviceType::Cuda);
  EXPECT_EQ(res.original_width, bus.width);
  EXPECT_EQ(res.original_height, bus.height);
  ASSERT_FALSE(res.detections.empty());
  EXPECT_GE(res.detections[0].confidence, 0.25f);

  auto m = runtime.snapshot_metrics();
  EXPECT_EQ(m.failed_requests, 0u);
  EXPECT_NE(m.per_resource_admissions.find("cuda:0"), m.per_resource_admissions.end());
}

// -----------------------------------------------------------------------------
// 9.4 Input ownership: the submit-time deep copy (not the caller buffer) feeds
// staging, so mutating or freeing the caller image after submit is safe.
// -----------------------------------------------------------------------------
TEST_F(CudaSchedulingRuntimeTest, MutatingSourceAfterSubmitDoesNotChangeResult) {
  // Test A: submit, then memset the ORIGINAL caller buffer. The deep-copied,
  // then staged copy must be unaffected. Compare against a pristine baseline.
  Runtime runtime(sched_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));

  std::vector<std::uint8_t> pixels;
  ImageInput bus = load_bus(pixels);
  auto baseline = runtime.submit("cuda_det", bus).get();

  // Second submission; corrupt the caller buffer immediately after submit.
  auto f = runtime.submit("cuda_det", bus);
  std::memset(pixels.data(), 0, pixels.size());
  DetectionResult res = f.get();
  runtime.shutdown();

  ASSERT_FALSE(res.detections.empty());
  // Deep copy happened at submit, so the result must equal the untouched run.
  ASSERT_EQ(res.detections.size(), baseline.detections.size());
  for (std::size_t i = 0; i < res.detections.size(); ++i) {
    EXPECT_NEAR(res.detections[i].confidence, baseline.detections[i].confidence, 1e-4);
  }
}

TEST_F(CudaSchedulingRuntimeTest, FreeingSourceAfterSubmitStillSucceeds) {
  // Test B: the caller pixels vector is destroyed right after submit; the async
  // CUDA path must still succeed because Runtime owns its deep copy.
  Runtime runtime(sched_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));

  std::future<DetectionResult> future;
  {
    std::vector<std::uint8_t> pixels;
    ImageInput bus = load_bus(pixels);
    future = runtime.submit("cuda_det", bus);
    // pixels (and bus.data) go out of scope here.
  }

  DetectionResult res;
  ASSERT_NO_THROW(res = future.get());
  runtime.shutdown();
  EXPECT_EQ(res.execution_device, DeviceType::Cuda);
  ASSERT_FALSE(res.detections.empty());
}

// -----------------------------------------------------------------------------
// 9.2 CUDA buffer tests (Build D): RAII allocation, H2D correctness, device
// guard, and release with no leak across repeated staging.
// -----------------------------------------------------------------------------
TEST_F(CudaSchedulingRuntimeTest, CudaInputBufferStagesAndCopiesExactly) {
  DeviceContext cuda(DeviceType::Cuda, 0);
  std::vector<std::uint8_t> storage;
  ImageInput img = make_stage_image(16, 8, 3, storage);
  // Fill with a recognizable pattern to verify the H2D byte-for-byte.
  for (std::size_t i = 0; i < storage.size(); ++i) storage[i] = static_cast<std::uint8_t>(i & 0xFF);

  CudaInputBuffer buf = CudaInputBuffer::stage(img, cuda);
  ASSERT_TRUE(buf.valid());
  EXPECT_EQ(buf.device_bytes(), 16u * 3u * 8u);
  EXPECT_EQ(buf.image().device, DeviceType::Cuda);
  EXPECT_EQ(buf.image().width, 16);
  EXPECT_EQ(buf.image().height, 8);
  EXPECT_EQ(buf.image().channels, 3);

  // Copy the staged device buffer back to host and compare to the source.
  std::vector<std::uint8_t> roundtrip(storage.size());
  ASSERT_EQ(cudaMemcpy(roundtrip.data(), buf.image().data, storage.size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(std::memcmp(roundtrip.data(), storage.data(), storage.size()), 0);
}

TEST_F(CudaSchedulingRuntimeTest, CudaInputBufferMovePreservesOwnership) {
  DeviceContext cuda(DeviceType::Cuda, 0);
  std::vector<std::uint8_t> storage;
  ImageInput img = make_stage_image(8, 8, 3, storage);
  CudaInputBuffer a = CudaInputBuffer::stage(img, cuda);
  const void* ptr = a.image().data;
  std::size_t bytes = a.device_bytes();

  CudaInputBuffer b(std::move(a));  // move construction
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.image().data, ptr);
  EXPECT_EQ(b.device_bytes(), bytes);

  CudaInputBuffer c;
  c = std::move(b);  // move assignment
  EXPECT_TRUE(c.valid());
  EXPECT_EQ(c.image().data, ptr);
}

TEST_F(CudaSchedulingRuntimeTest, CudaInputBufferRejectsNonCpuSource) {
  DeviceContext cuda(DeviceType::Cuda, 0);
  ImageInput not_cpu;
  not_cpu.device = DeviceType::Cuda;  // pretend an external device image
  not_cpu.width = 4;
  not_cpu.height = 4;
  not_cpu.channels = 3;
  not_cpu.dtype = DataType::Uint8;
  not_cpu.data = reinterpret_cast<const void*>(0x1);  // non-null so it passes null check
  EXPECT_THROW(CudaInputBuffer::stage(not_cpu, cuda), VieError);
}

TEST_F(CudaSchedulingRuntimeTest, CudaInputBufferRepeatedStageFreesWithoutLeak) {
  // Allocate and release many buffers; if the RAII TensorStorage leaked the
  // device memory the repeated cudaMalloc/cudaFree churn would error out.
  DeviceContext cuda(DeviceType::Cuda, 0);
  std::vector<std::uint8_t> storage;
  ImageInput img = make_stage_image(64, 64, 3, storage);
  for (int i = 0; i < 200; ++i) {
    CudaInputBuffer buf = CudaInputBuffer::stage(img, cuda);
    ASSERT_TRUE(buf.valid());
  }
  // Reaching here with valid() true every iteration and no throw is the check.
  SUCCEED();
}

// -----------------------------------------------------------------------------
// 9.5 Exception path: a malformed CPU image that passes submit() but fails at
// the staging boundary surfaces as a future exception, releases its permit and
// staging buffer without polluting global CUDA state, and a later valid
// request still succeeds.
// -----------------------------------------------------------------------------
TEST_F(CudaSchedulingRuntimeTest, StagingFailureRejectsFutureAndLaterRequestSucceeds) {
  Runtime runtime(sched_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));

  // A CPU image with a valid (non-null) buffer but zero width passes submit's
  // device/model checks, then cuda_stage_byte_count rejects it inside the
  // worker, so the failure arrives through the future, not submit().
  std::vector<std::uint8_t> dummy(16, 3);
  ImageInput bad;
  bad.data = dummy.data();
  bad.width = 0;  // invalid dimension -> staging InvalidArgument
  bad.height = 4;
  bad.channels = 3;
  bad.dtype = DataType::Uint8;
  bad.device = DeviceType::Cpu;

  auto bad_future = runtime.submit("cuda_det", bad);
  EXPECT_THROW(bad_future.get(), VieError);

  // The failed request must not have poisoned the lane: a real bus image now
  // still stages and detects.
  std::vector<std::uint8_t> pixels;
  ImageInput bus = load_bus(pixels);
  DetectionResult res = runtime.submit("cuda_det", bus).get();
  runtime.shutdown();

  ASSERT_FALSE(res.detections.empty());
  EXPECT_EQ(res.execution_device, DeviceType::Cuda);
  auto m = runtime.snapshot_metrics();
  EXPECT_GE(m.failed_requests, 1u);   // the staging failure was counted
  EXPECT_GE(m.completed_requests, 1u); // the follow-up succeeded
}

// -----------------------------------------------------------------------------
// 9.6 Concurrency: several CUDA requests on the same lane each own a distinct
// staged buffer. Same-device permit serialization must hold, and no request may
// observe another's input (all results equal the synchronous baseline).
// -----------------------------------------------------------------------------
TEST_F(CudaSchedulingRuntimeTest, ConcurrentCudaRequestsDoNotShareInputBuffer) {
  Runtime runtime(sched_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));

  std::vector<std::uint8_t> pixels;
  ImageInput bus = load_bus(pixels);
  DetectionResult baseline = runtime.submit("cuda_det", bus).get();  // warm, serialized

  std::vector<std::future<DetectionResult>> futures;
  for (int i = 0; i < 6; ++i) futures.push_back(runtime.submit("cuda_det", bus));
  std::vector<DetectionResult> results;
  for (auto& f : futures) results.push_back(f.get());
  runtime.shutdown();

  ASSERT_FALSE(baseline.detections.empty());
  for (const auto& res : results) {
    EXPECT_EQ(res.execution_device, DeviceType::Cuda);
    // Every request decoded its OWN staged copy; none borrowed another's buffer,
    // so each matches the baseline detection set exactly.
    EXPECT_EQ(res.detections.size(), baseline.detections.size());
    for (std::size_t i = 0; i < res.detections.size() && i < baseline.detections.size(); ++i) {
      EXPECT_NEAR(res.detections[i].confidence, baseline.detections[i].confidence, 1e-4);
    }
  }

  auto m = runtime.snapshot_metrics();
  EXPECT_EQ(m.failed_requests, 0u);
  EXPECT_EQ(m.completed_requests, 7u);  // 1 baseline + 6 concurrent
}

// -----------------------------------------------------------------------------
// 10. Performance reference (unified timing). Not a pass/fail gate: Phase 9.3.2
// is correctness, not optimization. It measures three paths under one method so
// the numbers are comparable:
//   * sync CUDA   : ModelInstance::infer() on a caller-owned on-device image
//                   (H2D done by the caller before timing, as the sync API does).
//   * async CUDA  : Runtime::submit(cpu_image) -> future.get()  [includes the
//                   Runtime-owned H2D staging + queue wait + CUDA inference].
//   * async CPU   : Runtime::submit(cpu_image) -> future.get()  [no H2D].
// All async timings start at submit() and end when the future resolves, exactly
// as required; H2D and queue time are inside the async CUDA number, not excluded.
// -----------------------------------------------------------------------------
namespace {
struct PerfStats {
  double mean_ms, p50_ms, p95_ms, max_ms, throughput_rps;
};
PerfStats summarize(std::vector<double>& lat_ms, double wall_ms) {
  std::sort(lat_ms.begin(), lat_ms.end());
  PerfStats s{};
  const std::size_t n = lat_ms.size();
  if (n == 0) return s;
  double sum = 0.0;
  for (double v : lat_ms) sum += v;
  s.mean_ms = sum / static_cast<double>(n);
  auto rank = [&](double p) {
    std::size_t r = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(n)));
    if (r == 0) r = 1;
    if (r > n) r = n;
    return lat_ms[r - 1];
  };
  s.p50_ms = rank(50);
  s.p95_ms = rank(95);
  s.max_ms = lat_ms.back();
  s.throughput_rps = wall_ms > 0 ? static_cast<double>(n) / (wall_ms / 1000.0) : 0.0;
  return s;
}
void print_perf(const char* name, const PerfStats& s, int warmup, int requests) {
  std::cout << "  [perf] " << name << "  warmup=" << warmup << " requests=" << requests
            << " mean=" << std::fixed << std::setprecision(2) << s.mean_ms
            << "ms p50=" << s.p50_ms << "ms p95=" << s.p95_ms << "ms max=" << s.max_ms
            << "ms throughput=" << std::setprecision(1) << s.throughput_rps << " req/s\n";
}
}  // namespace

TEST_F(CudaSchedulingRuntimeTest, PerformanceReferenceReport) {
  constexpr int kWarmup = 10;
  constexpr int kRequests = 50;
  const std::size_t img_bytes = 640u * 640u * 3u;

  // --- async CUDA via staging (Runtime-owned H2D included) ---
  {
    Runtime runtime(sched_config());
    runtime.models().load("cuda_det",
                          model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));
    std::vector<std::uint8_t> pixels;
    ImageInput bus = load_bus(pixels);
    for (int i = 0; i < kWarmup; ++i) runtime.submit("cuda_det", bus).get();
    std::vector<double> lat;
    auto wall0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kRequests; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      runtime.submit("cuda_det", bus).get();
      auto t1 = std::chrono::steady_clock::now();
      lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    auto wall1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    std::cout << "\n=== Phase 9.3.2 CUDA staging performance reference (bus.ppm 810x1080) ===\n";
    print_perf("async CUDA (submit->get, incl. H2D+queue)", summarize(lat, wall), kWarmup, kRequests);
    runtime.shutdown();
  }

  // --- sync CUDA (caller-owned device buffer, no Runtime staging) ---
  {
    Runtime runtime(sched_config());
    auto model = runtime.models().load(
        "cuda_det", model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));
    DeviceBuffer buf;
    buf.host.assign(img_bytes, 127);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&buf.ptr), img_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(buf.ptr, buf.host.data(), img_bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ImageInput img;
    img.data = buf.ptr;
    img.width = 640;
    img.height = 640;
    img.channels = 3;
    img.color = ColorOrder::Rgb;
    img.layout = ImageLayout::Hwc;
    img.dtype = DataType::Uint8;
    img.device = DeviceType::Cuda;
    img.device_id = 0;
    for (int i = 0; i < kWarmup; ++i) model->infer(img);
    std::vector<double> lat;
    auto wall0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kRequests; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      model->infer(img);
      auto t1 = std::chrono::steady_clock::now();
      lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    auto wall1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    // Note: uniform 127 fill (empty detections); this isolates the pipeline cost
    // from decode work and is a different input than the async bus run, so the
    // comparison is directional, not an apples-to-apples H2D delta.
    print_perf("sync CUDA (infer, on-device, uniform fill)", summarize(lat, wall), kWarmup, kRequests);
    runtime.shutdown();
  }

  // --- async CPU (no H2D, reference for CPU non-regression) ---
  {
    Runtime runtime(sched_config());
    runtime.models().load("cpu_det", model_config());
    std::vector<std::uint8_t> pixels;
    ImageInput bus = load_bus(pixels);
    for (int i = 0; i < kWarmup; ++i) runtime.submit("cpu_det", bus).get();
    std::vector<double> lat;
    auto wall0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kRequests; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      runtime.submit("cpu_det", bus).get();
      auto t1 = std::chrono::steady_clock::now();
      lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    auto wall1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    print_perf("async CPU (submit->get, no H2D)", summarize(lat, wall), kWarmup, kRequests);
    EXPECT_EQ(runtime.snapshot_metrics().failed_requests, 0u);
    runtime.shutdown();
  }
}

// -----------------------------------------------------------------------------
// Phase 9.3.2.1: CUDA Execution Provider placement verification.
//
// These tests profile the REAL asynchronous Runtime session (ModelConfig
// .ort_enable_profiling feeds the same Ort::Session the worker's infer_async
// runs), flush it via ModelInstance::end_profiling(), and parse the resulting
// ORT 1.20.1 Chrome trace to attribute every executed node to an Execution
// Provider. They are the node-level evidence Phase 9.3.2's execution_device
// signal could not provide.
// -----------------------------------------------------------------------------
namespace {

bool op_is_memcpy(const std::string& op) {
  return op.find("Memcpy") != std::string::npos || op.find("memcpy") != std::string::npos;
}

const char* provider_label(OrtProviderClass c) {
  switch (c) {
    case OrtProviderClass::Cuda: return "CUDA";
    case OrtProviderClass::Cpu: return "CPU";
    case OrtProviderClass::Other: return "OTHER";
    case OrtProviderClass::Unknown: return "UNKNOWN";
  }
  return "?";
}

// Prints the §6.2 report block plus the per-provider node detail. Purely
// diagnostic; the assertions live in the tests.
void print_ort_stats(const char* label, const OrtProfileStats& s) {
  const std::size_t known = s.cuda_node_events + s.cpu_node_events + s.other_provider_events;
  const double cuda_pct = known ? 100.0 * static_cast<double>(s.cuda_node_events) / static_cast<double>(known) : 0.0;
  const double cpu_pct = known ? 100.0 * static_cast<double>(s.cpu_node_events) / static_cast<double>(known) : 0.0;
  std::cout << "\n=== ORT EP placement: " << label << " ===\n"
            << "  total_trace_events   = " << s.total_trace_events << "\n"
            << "  total_node_events    = " << s.total_node_events << "\n"
            << "  cuda_node_events     = " << s.cuda_node_events << "\n"
            << "  cpu_node_events      = " << s.cpu_node_events << "\n"
            << "  other_provider_events= " << s.other_provider_events << "\n"
            << "  unknown_provider_evts= " << s.unknown_provider_events << "\n"
            << "  transfer_events      = " << s.transfer_events << "\n"
            << "  unique_node_names    = " << s.unique_node_names << "\n"
            << "  CUDA% (of placed)    = " << std::fixed << std::setprecision(2) << cuda_pct << "%\n"
            << "  CPU%  (of placed)    = " << cpu_pct << "%\n";
  std::cout << "  -- CPU-placed nodes --\n";
  for (const OrtNodeEvent& n : s.node_events) {
    if (n.provider_class == OrtProviderClass::Cpu) {
      std::cout << "    [CPU ] " << n.name << "  op=" << n.op_type << "\n";
    }
  }
  std::cout << "  -- memcpy-family nodes --\n";
  for (const OrtNodeEvent& n : s.node_events) {
    if (op_is_memcpy(n.op_type)) {
      std::cout << "    [" << provider_label(n.provider_class) << "] " << n.name
                << "  op=" << n.op_type << "\n";
    }
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// Step 5 / A3 A4 A5 A6 A7 A8: profile the actual async Runtime CUDA session and
// attribute every executed node to an EP. yolov8n runs with cuda_cpu_ep_fallback
// = true; the trace must show most nodes on CUDA, the four Memcpy kernels, and
// the genuine CPU fallback being the two Resize ops (NOT the memcpys).
// -----------------------------------------------------------------------------
TEST_F(CudaSchedulingRuntimeTest, Yolov8nFallbackOnAsyncProfileIsCudaWithResizeFallback) {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path();
  Runtime runtime(sched_config());
  auto cfg = model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0);
  cfg.ort_enable_profiling = true;
  cfg.ort_profile_prefix = (dir / "vie_yolov8n_fallback_on_").string();
  auto model = runtime.models().load("cuda_det", cfg);

  std::vector<std::uint8_t> pixels;
  ImageInput bus = load_bus(pixels);
  DetectionResult res = runtime.submit("cuda_det", bus).get();
  const std::string path = model->end_profiling();
  runtime.shutdown();

  // A4: the real asynchronous CUDA detection path produced a non-empty result.
  ASSERT_FALSE(res.detections.empty());
  EXPECT_EQ(res.execution_device, DeviceType::Cuda);
  // A3: a profile exists and came from the Runtime session just used.
  ASSERT_FALSE(path.empty()) << "end_profiling returned no path";
  std::error_code ec;
  ASSERT_TRUE(fs::exists(path, ec)) << path;
  EXPECT_GT(fs::file_size(path, ec), 0u);

  const OrtProfileStats stats = parse_ort_profile_file(path);
  print_ort_stats("yolov8n fallback=ON (async Runtime)", stats);

  // A5/A6/A7: placement is attributed for every node; none silently unknown.
  EXPECT_GT(stats.total_node_events, 100u);
  EXPECT_EQ(stats.unknown_provider_events, 0u);
  EXPECT_GT(stats.cuda_node_events, 0u);
  EXPECT_GT(stats.cpu_node_events, 0u) << "fallback=ON must expose a genuine CPU fallback";
  // A8: exactly the four Memcpy transfer nodes.
  EXPECT_EQ(stats.transfer_events, 4u);
  EXPECT_EQ(stats.memcpy_names.size(), 4u);

  // The CPU-placed nodes are the two Resize ops, not the memcpys: this is the
  // precise, node-level reason cuda_cpu_ep_fallback is required.
  for (const OrtNodeEvent& n : stats.node_events) {
    if (n.provider_class == OrtProviderClass::Cpu) {
      EXPECT_EQ(n.op_type, "Resize") << "unexpected CPU node " << n.name;
    }
    if (op_is_memcpy(n.op_type)) {
      EXPECT_EQ(n.provider_class, OrtProviderClass::Cuda)
          << "memcpy node " << n.name << " unexpectedly not on CUDA";
    }
  }
}

// -----------------------------------------------------------------------------
// Step 5 / A12 (§11.2): with profiling left at its default (off) the production
// path is byte-for-byte unchanged -- inference still succeeds and NO profile is
// flushed.
// -----------------------------------------------------------------------------
TEST_F(CudaSchedulingRuntimeTest, ProfilingDisabledByDefaultEmitsNoProfile) {
  Runtime runtime(sched_config());
  auto cfg = model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0);
  cfg.ort_enable_profiling = false;  // explicit about the default
  auto model = runtime.models().load("cuda_det", cfg);

  std::vector<std::uint8_t> pixels;
  ImageInput bus = load_bus(pixels);
  DetectionResult res = runtime.submit("cuda_det", bus).get();
  const std::string path = model->end_profiling();
  runtime.shutdown();

  EXPECT_EQ(res.execution_device, DeviceType::Cuda);
  ASSERT_FALSE(res.detections.empty());  // no inference regression
  EXPECT_TRUE(path.empty()) << "no profile must be produced when disabled";
}

// -----------------------------------------------------------------------------
// Step 6 / A9 (§11.3): the same model with cuda_cpu_ep_fallback = false fails at
// SESSION CREATION (ModelInstance load), never reaching submit. Records the real
// ORT error rather than skipping.
// -----------------------------------------------------------------------------
TEST_F(CudaSchedulingRuntimeTest, Yolov8nStrictCudaFallbackOffFailsAtLoad) {
  Runtime runtime(sched_config());
  auto cfg = model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0);
  cfg.cuda_cpu_ep_fallback = false;  // strict CUDA: CPU EP not appended

  std::string stage = "load";
  std::string message;
  bool threw = false;
  try {
    auto model = runtime.models().load("cuda_det", cfg);
    stage = "submit";  // load returned; any throw from here is a submit-stage failure
    std::vector<std::uint8_t> pixels;
    ImageInput bus = load_bus(pixels);
    (void)runtime.submit("cuda_det", bus).get();
  } catch (const VieError& e) {
    threw = true;
    message = e.what();
  }
  runtime.shutdown();

  EXPECT_TRUE(threw) << "strict CUDA must not silently succeed for yolov8n";
  EXPECT_EQ(stage, "load") << "must fail at session creation, not at submit";
  // ORT refuses to build the session because the Resize nodes need the CPU EP.
  EXPECT_NE(message.find("CPU EP"), std::string::npos) << message;
  std::cout << "[strict] stage=" << stage << " error=" << message << "\n";
}

#endif  // VIE_HAS_ONNXRUNTIME_CUDA

#endif  // VIE_HAS_ONNXRUNTIME
