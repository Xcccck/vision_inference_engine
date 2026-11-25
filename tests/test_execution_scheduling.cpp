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
#include "vie/runtime/dynamic_batch_aggregator.hpp"
#include "vie/runtime/execution_resource.hpp"
#include "vie/runtime/execution_resource_manager.hpp"
#include "vie/runtime/execution_scheduling_config.hpp"
#include "vie/runtime/model_instance.hpp"
#include "vie/runtime/runtime.hpp"
#include "vie/runtime/runtime_config.hpp"
#include "vie/runtime/runtime_metrics.hpp"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

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
// 4. Real CUDA Runtime scheduling integration (Build D only).
//
// Phase 9.3.1 audit result (Conclusion B): a CUDA-loaded ModelInstance's
// execution resource IS cuda:N, and the scheduler DOES route its batches into
// the CUDA lane and grant/release exactly one permit. But the current async
// pipeline cannot drive the CUDA backend to *success*: Runtime::submit()
// accepts only CPU ImageInput (Phase 9.2.2.3), the request keeps image.device
// == Cpu (adopt_cpu_pixels), and ImagePreprocessor::execute rejects a CPU
// image against a CUDA execution device ("H2D stays outside preprocess"). So
// an async CUDA batch is admitted to the cuda lane, fails at the preprocess
// boundary, and its permit is released during unwinding.
//
// These tests verify that SCHEDULER integration on a genuine CUDA resource
// lane plus the boundary itself. The CUDA backend's real execution is proven
// separately by the synchronous inference below (a genuine on-device image),
// not by execution_resource() alone. We deliberately do NOT claim a successful
// async CUDA inference.
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

TEST_F(CudaSchedulingRuntimeTest, CudaModelCpuImageIsRoutedToCudaLaneAndReleasesPermit) {
  // The async boundary: submit accepts the CPU image, the scheduler admits the
  // batch to the real cuda:0 lane, execution fails at the preprocess device
  // mismatch, and the permit is released (no hang, future resolves).
  Runtime runtime(sched_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));
  auto f = runtime.submit("cuda_det", cpu_image());
  ASSERT_THROW(f.get(), VieError);  // device mismatch surfaces from preprocess
  runtime.shutdown();

  auto m = runtime.snapshot_metrics();
  auto it = m.per_resource_admissions.find("cuda:0");
  ASSERT_NE(it, m.per_resource_admissions.end());  // admitted to the CUDA lane
  EXPECT_GE(it->second, 1u);
  EXPECT_GE(m.completed_requests, 1u);
  EXPECT_GE(m.failed_requests, 1u);  // permit path finished even on failure
}

TEST_F(CudaSchedulingRuntimeTest, CpuAndCudaLanesAreIndependentInRuntime) {
  // With a real CUDA lane present, CPU work must still flow: CPU futures
  // resolve successfully, the CUDA request is admitted to its own lane and
  // fails only at the preprocess boundary. Proves cross-resource routing and
  // that CPU requests are not blocked behind the cuda lane.
  Runtime runtime(sched_config());
  runtime.models().load("cpu_det", model_config());
  runtime.models().load("cuda_det",
                        model_config(BackendKind::OnnxRuntimeCuda, DeviceType::Cuda, 0));

  std::vector<std::future<DetectionResult>> cpu_futures;
  for (int i = 0; i < 4; ++i) cpu_futures.push_back(runtime.submit("cpu_det", cpu_image()));
  auto cuda_future = runtime.submit("cuda_det", cpu_image());

  for (auto& f : cpu_futures) f.get();  // succeed, not blocked by the cuda lane
  EXPECT_THROW(cuda_future.get(), VieError);
  runtime.shutdown();

  auto m = runtime.snapshot_metrics();
  EXPECT_EQ(m.completed_requests, 5u);
  EXPECT_EQ(m.failed_requests, 1u);
  EXPECT_NE(m.per_resource_admissions.find("cpu"), m.per_resource_admissions.end());
  EXPECT_NE(m.per_resource_admissions.find("cuda:0"), m.per_resource_admissions.end());
}

#endif  // VIE_HAS_ONNXRUNTIME_CUDA

#endif  // VIE_HAS_ONNXRUNTIME
