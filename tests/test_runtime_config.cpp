// Phase 12: Runtime / async inference-configuration tests.
//
// Covers:
// - the legacy submit(model, image) call keeping the framework defaults
// - per-request overrides reaching the real decoder through the async worker
//   path (synthetic mock backend output, never a mock decoder)
// - the effective config being frozen at submit time: mutating the caller's
//   options object after submit() returns cannot affect the queued request
// - concurrent submits keeping their own thresholds
// - model-level defaults from ModelConfig::inference_options, overridable per
//   request
// - invalid options rejected synchronously at submit time (never silently
//   repaired or ignored)
// - the dynamic batch aggregator grouping only identical effective configs
//   (same model + same fingerprint) and separating everything else
// - one ModelInstance serving two threads with different thresholds

#include "test_util.hpp"

#include "vie/config/inference_config.hpp"
#include "vie/model/manifest.hpp"
#include "vie/runtime/batch_aggregation_config.hpp"
#include "vie/runtime/dynamic_batch_aggregator.hpp"
#include "vie/runtime/model_instance.hpp"
#include "vie/runtime/runtime.hpp"

#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using vie::InferenceOptions;
using vie::ModelManifest;
using vie::ParameterSource;
using vie::ResolvedInferenceConfig;
using vie::TaskType;

namespace {

// Mock backend output is {1, 2, 3, 4, 0.9, 0} of fp32 per output tensor:
// one anchor at (cx=1, cy=2, w=3, h=4) with class-0 score 0.9. The default
// framework confidence threshold (0.25) keeps it; a 0.95 override drops it.
ModelManifest mock_detection_manifest() {
  ModelManifest manifest;
  manifest.model_name = "runtime_cfg_det";
  manifest.task = TaskType::Detection;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 2;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  manifest.decoder.protocol = "yolov8_detect_raw";
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.class_selection = "best_class";
  manifest.decoder.scores_are_probabilities = true;
  vie::OutputDesc prediction;
  prediction.name = "predictions";
  prediction.shape = {1, 6, 1};
  prediction.dtype = vie::DataType::Fp32;
  prediction.semantic = "predictions";
  manifest.outputs = {prediction};
  manifest.ensure_provenance();
  return manifest;
}

vie::ModelConfig model_config(const ModelManifest& manifest,
                              const InferenceOptions& defaults = {}) {
  vie::ModelConfig config;
  config.manifest = manifest;
  config.backend = vie::BackendKind::Auto;
  config.device = vie::DeviceType::Cpu;
  config.inference_options = defaults;
  return config;
}

vie::RuntimeConfig runtime_config(std::size_t max_batch_size,
                                  std::chrono::milliseconds max_wait) {
  vie::RuntimeConfig config;
  config.thread_pool_size = 2;
  config.request_queue_capacity = 64;
  config.inference_worker_count = 1;
  config.batch_aggregation.enabled = true;
  config.batch_aggregation.max_batch_size = max_batch_size;
  config.batch_aggregation.max_wait_time = max_wait;
  config.batch_aggregation.max_pending_requests = 64;
  return config;
}

vie::ImageInput tiny_image(std::vector<std::uint8_t>& pixels) {
  pixels.assign(8 * 8 * 3, 9);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = 8;
  image.height = 8;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

InferenceOptions confidence(float value) {
  InferenceOptions options;
  options.detection.confidence_threshold = value;
  return options;
}

// A request with just a model identity and an effective-config fingerprint is
// all the aggregator needs for grouping (execution is out of scope here).
vie::InferenceRequest grouping_request(const std::shared_ptr<vie::ModelInstance>& model,
                                       const std::string& key, vie::RequestId id) {
  vie::InferenceRequest request;
  request.id = id;
  request.model_id = model->id();
  request.model = model;
  request.config_key = key;
  request.submitted_at = std::chrono::steady_clock::now();
  return request;
}

}  // namespace

TEST(RuntimeConfig, LegacySubmitUsesFrameworkDefaults) {
  vie::Runtime runtime(runtime_config(1, std::chrono::milliseconds(0)));
  runtime.models().load("det", model_config(mock_detection_manifest()));

  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);
  const vie::DetectionResult legacy = runtime.submit("det", image).get();
  ASSERT_EQ(legacy.detections.size(), 1u);
  EXPECT_FLOAT_EQ(legacy.detections[0].confidence, 0.9f);
  EXPECT_EQ(legacy.detections[0].class_id, 0);

  // The explicit empty options resolve to the same framework defaults.
  const vie::DetectionResult defaults =
      runtime.submit("det", image, InferenceOptions{}).get();
  ASSERT_EQ(defaults.detections.size(), 1u);
  EXPECT_FLOAT_EQ(defaults.detections[0].confidence, 0.9f);
}

TEST(RuntimeConfig, SubmitAppliesPerRequestOverrides) {
  vie::Runtime runtime(runtime_config(1, std::chrono::milliseconds(0)));
  runtime.models().load("det", model_config(mock_detection_manifest()));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  EXPECT_TRUE(runtime.submit("det", image, confidence(0.95f)).get().detections.empty());

  // A lower threshold keeps the 0.9 anchor.
  EXPECT_EQ(runtime.submit("det", image, confidence(0.5f)).get().detections.size(), 1u);

  // A zero threshold is legal (boundary), not silently repaired.
  EXPECT_EQ(runtime.submit("det", image, confidence(0.0f)).get().detections.size(), 1u);
}

TEST(RuntimeConfig, SubmitFreezesOptionsAtSubmitTime) {
  // Each request waits for the 250 ms deadline, so the caller has ample time
  // to mutate its options object while the request is pending.
  vie::Runtime runtime(runtime_config(4, std::chrono::milliseconds(250)));
  runtime.models().load("det", model_config(mock_detection_manifest()));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  InferenceOptions high = confidence(0.95f);
  InferenceOptions low = confidence(0.0f);
  std::future<vie::DetectionResult> high_future = runtime.submit("det", image, high);
  std::future<vie::DetectionResult> low_future = runtime.submit("det", image, low);

  // Mutate both objects after submit() returned its futures. A runtime that
  // read the caller's options at execution time would invert both results.
  high.detection.confidence_threshold = 0.0f;
  low.detection.confidence_threshold = 0.95f;

  EXPECT_TRUE(high_future.get().detections.empty());
  EXPECT_EQ(low_future.get().detections.size(), 1u);
}

TEST(RuntimeConfig, ConcurrentSubmitsKeepTheirOwnConfig) {
  vie::Runtime runtime(runtime_config(1, std::chrono::milliseconds(0)));
  runtime.models().load("det", model_config(mock_detection_manifest()));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  constexpr int kRequests = 6;
  std::vector<std::future<vie::DetectionResult>> strict;
  std::vector<std::future<vie::DetectionResult>> lenient;
  strict.reserve(kRequests);
  lenient.reserve(kRequests);
  for (int i = 0; i < kRequests; ++i) {
    strict.push_back(runtime.submit("det", image, confidence(0.95f)));
    lenient.push_back(runtime.submit("det", image, confidence(0.5f)));
  }
  for (auto& future : strict) {
    EXPECT_TRUE(future.get().detections.empty());
  }
  for (auto& future : lenient) {
    EXPECT_EQ(future.get().detections.size(), 1u);
  }
}

TEST(RuntimeConfig, ModelDefaultsApplyAndRequestOverridesThem) {
  InferenceOptions model_defaults = confidence(0.95f);
  vie::Runtime runtime(runtime_config(1, std::chrono::milliseconds(0)));
  runtime.models().load("det", model_config(mock_detection_manifest(), model_defaults));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  // No request override: the model default (0.95) hides the 0.9 anchor.
  EXPECT_TRUE(runtime.submit("det", image).get().detections.empty());

  // A request override wins over the model default.
  EXPECT_EQ(runtime.submit("det", image, confidence(0.5f)).get().detections.size(), 1u);

  // The stored model config is untouched by the request override.
  const std::shared_ptr<vie::ModelInstance> model = runtime.models().get("det");
  ASSERT_TRUE(model != nullptr);
  EXPECT_EQ(model->config().inference_options.detection.confidence_threshold, 0.95f);
  const ResolvedInferenceConfig resolved = model->resolve_inference_config();
  EXPECT_FLOAT_EQ(resolved.require_detection().confidence_threshold.value, 0.95f);
  EXPECT_EQ(resolved.require_detection().confidence_threshold.source,
            ParameterSource::ModelDefault);
}

TEST(RuntimeConfig, InvalidOptionsAreRejectedBeforeQueueing) {
  vie::Runtime runtime(runtime_config(1, std::chrono::milliseconds(0)));
  runtime.models().load("det", model_config(mock_detection_manifest()));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   runtime.submit("det", image, confidence(nan)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   runtime.submit("det", image, confidence(inf)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   runtime.submit("det", image, confidence(-0.01f)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   runtime.submit("det", image, confidence(1.01f)));

  // A block that does not belong to the model's task is rejected instead of
  // being silently ignored.
  InferenceOptions wrong;
  wrong.classification.top_k = 2;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, runtime.submit("det", image, wrong));

  // Rejected submissions never entered the queue; the runtime is still usable.
  EXPECT_EQ(runtime.submit("det", image).get().detections.size(), 1u);
  EXPECT_EQ(runtime.snapshot_metrics().rejected_requests, 0u);
}

TEST(RuntimeConfig, AggregatorGroupsOnlyIdenticalConfigs) {
  vie::Runtime runtime(runtime_config(1, std::chrono::milliseconds(0)));
  const std::shared_ptr<vie::ModelInstance> first =
      runtime.models().load("det", model_config(mock_detection_manifest()));
  const std::shared_ptr<vie::ModelInstance> second =
      runtime.models().load("det2", model_config(mock_detection_manifest()));

  vie::BatchAggregationConfig config;
  config.enabled = true;
  config.max_batch_size = 4;
  config.max_wait_time = std::chrono::milliseconds(1000);
  config.max_pending_requests = 16;
  vie::DynamicBatchAggregator aggregator(config, std::make_shared<vie::MetricsCollector>());

  // Same model, same effective config: one pending group.
  EXPECT_FALSE(aggregator.add(grouping_request(first, "conf=0.25", 1)).has_value());
  EXPECT_FALSE(aggregator.add(grouping_request(first, "conf=0.25", 2)).has_value());
  EXPECT_EQ(aggregator.group_count(), 1u);
  EXPECT_EQ(aggregator.pending_count(), 2u);

  // Same model, different effective config: a second group, never merged.
  EXPECT_FALSE(aggregator.add(grouping_request(first, "conf=0.50", 3)).has_value());
  EXPECT_EQ(aggregator.group_count(), 2u);
  EXPECT_EQ(aggregator.pending_count(), 3u);

  // Different model, same effective config: still a separate group.
  EXPECT_FALSE(aggregator.add(grouping_request(second, "conf=0.25", 4)).has_value());
  EXPECT_EQ(aggregator.group_count(), 3u);
  EXPECT_EQ(aggregator.pending_count(), 4u);

  std::vector<vie::BatchWorkItem> batches = aggregator.flush_all();
  ASSERT_EQ(batches.size(), 3u);
  std::size_t same_config_pair = 0;
  std::size_t single_first_other_config = 0;
  std::size_t single_second_model = 0;
  for (const vie::BatchWorkItem& batch : batches) {
    ASSERT_FALSE(batch.requests.empty());
    const std::string& key = batch.requests.front().config_key;
    for (const vie::InferenceRequest& request : batch.requests) {
      EXPECT_EQ(request.config_key, key);
      EXPECT_EQ(request.model.get(), batch.model.get());
    }
    if (batch.model.get() == first.get() && key == "conf=0.25") {
      same_config_pair = batch.requests.size();
    } else if (batch.model.get() == first.get() && key == "conf=0.50") {
      single_first_other_config = batch.requests.size();
    } else if (batch.model.get() == second.get() && key == "conf=0.25") {
      single_second_model = batch.requests.size();
    }
  }
  EXPECT_EQ(same_config_pair, 2u);
  EXPECT_EQ(single_first_other_config, 1u);
  EXPECT_EQ(single_second_model, 1u);
  EXPECT_EQ(aggregator.group_count(), 0u);
  EXPECT_EQ(aggregator.pending_count(), 0u);
}

TEST(RuntimeConfig, SameModelInstanceThreadsUseTheirOwnThreshold) {
  vie::Runtime runtime(runtime_config(1, std::chrono::milliseconds(0)));
  const std::shared_ptr<vie::ModelInstance> model =
      runtime.models().load("det", model_config(mock_detection_manifest()));
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);

  // Resolve per thread and run the snapshot directly on the shared instance.
  // ModelInstance serializes the calls; the results must still follow the
  // config each thread resolved for itself.
  const auto run = [&model, &image](float threshold, std::size_t expected) {
    const ResolvedInferenceConfig resolved =
        model->resolve_inference_config(confidence(threshold));
    for (int i = 0; i < 8; ++i) {
      const vie::DetectionBatch batch = model->infer_async(image, resolved);
      ASSERT_EQ(batch.results.size(), 1u);
      EXPECT_EQ(batch.results[0].detections.size(), expected);
    }
  };

  std::thread strict([&run] { run(0.95f, 0); });
  std::thread lenient([&run] { run(0.5f, 1); });
  strict.join();
  lenient.join();
}