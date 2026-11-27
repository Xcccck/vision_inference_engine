#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/decoder/yolov8_detect_decoder.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <tuple>
#include <vector>

namespace {

vie::GeometryTransform identity(int width, int height) {
  vie::GeometryTransform geometry;
  geometry.original_width = width;
  geometry.original_height = height;
  geometry.resized_width = width;
  geometry.resized_height = height;
  geometry.output_width = width;
  geometry.output_height = height;
  geometry.scale_x = 1;
  geometry.scale_y = 1;
  geometry.uniform_scale = 1;
  return geometry;
}

vie::DecoderConfig cxcywh_config() {
  vie::DecoderConfig config;
  config.output_name = "predictions";
  config.channel_axis = 1;
  config.box_channels = 4;
  config.num_classes = 2;
  config.box_decode = vie::BoxDecode::DecodedCxcywh;
  config.scores_are_probabilities = true;
  config.class_selection = vie::ClassSelection::BestClass;
  config.confidence_threshold = 0.5f;
  config.nms_enabled = true;
  config.class_aware_nms = true;
  config.iou_threshold = 0.5f;
  config.max_detections = 10;
  config.pre_nms_top_k = 100;
  return config;
}

vie::DetectionBatch run_decoder(const std::vector<float>& values, const std::vector<std::int64_t>& shape,
                               const vie::DecoderConfig& config, const vie::GeometryTransform& geometry) {
  vie::Tensor tensor = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, shape, values.data(),
                                              values.size() * sizeof(float));
  vie::DecoderInput input;
  input.outputs.emplace(config.output_name, tensor);
  input.geometry = geometry;
  return vie::YoloV8DetectionDecoder{}.decode_batch(input, config, vie::DeviceContext{});
}

}  // namespace

TEST(DetectionOps, DflSoftmaxExpectationAndDist2bbox) {
  const float logits[] = {0.f, 100.f, 0.f};
  EXPECT_NEAR(vie::dfl_expectation(logits, 3), 1.f, 1e-5f);
  const float tied[] = {0.f, 0.f};
  EXPECT_NEAR(vie::dfl_expectation(tied, 2), 0.5f, 1e-6f);
  const float extreme[] = {1e20f, 0.f, -1e20f};
  EXPECT_NEAR(vie::dfl_expectation(extreme, 3), 0.f, 1e-5f);
  const float bad[] = {0.f, std::nanf(""), 1.f};
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::dfl_expectation(bad, 3));

  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;
  vie::dist2bbox(1.f, 1.f, 1.f, 1.f, 0.5f, 0.5f, 8.f, x1, y1, x2, y2);
  EXPECT_FLOAT_EQ(x1, -4.f);
  EXPECT_FLOAT_EQ(y1, -4.f);
  EXPECT_FLOAT_EQ(x2, 12.f);
  EXPECT_FLOAT_EQ(y2, 12.f);
}

TEST(DetectionOps, LetterboxUsesSeparateScalesAndClips) {
  vie::BoundingBox box{8.f, 7.f, 12.f, 13.f};
  vie::GeometryTransform geometry = identity(40, 40);
  geometry.pad_left = 2;
  geometry.pad_top = 1;
  geometry.scale_x = 0.5;
  geometry.scale_y = 0.25;
  geometry.uniform_scale = 0.25;
  EXPECT_TRUE(vie::map_box_to_original(box, geometry));
  EXPECT_FLOAT_EQ(box.x1, 12.f);
  EXPECT_FLOAT_EQ(box.y1, 24.f);
  EXPECT_FLOAT_EQ(box.x2, 20.f);
  EXPECT_FLOAT_EQ(box.y2, 40.f);

  vie::PreprocessConfig letterbox;
  letterbox.resize_mode = vie::ResizeMode::Letterbox;
  letterbox.target_width = 8;
  letterbox.target_height = 8;
  letterbox.padding_value = 114.f;
  const vie::GeometryTransform odd = vie::make_geometry(letterbox, 5, 3);
  vie::BoundingBox corner{1.f, static_cast<float>(odd.pad_top), 3.f, static_cast<float>(odd.pad_top + 2)};
  EXPECT_TRUE(vie::map_box_to_original(corner, odd));
  EXPECT_NEAR(corner.x1, 1.f / odd.scale_x, 1e-5f);
  EXPECT_FLOAT_EQ(corner.y1, 0.f);
  EXPECT_NE(odd.scale_x, odd.scale_y);
}

TEST(DecoderRegistry, RegistersOnceAndRejectsUnknownProtocols) {
  vie::DecoderRegistry registry;
  registry.register_factory("yolov8_detect_raw", [] { return std::make_unique<vie::YoloV8DetectionDecoder>(); });
  EXPECT_TRUE(registry.contains("yolov8_detect_raw"));
  EXPECT_EQ(registry.create("yolov8_detect_raw")->protocol(), "yolov8_detect_raw");
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   registry.register_factory("yolov8_detect_raw",
                                            [] { return std::make_unique<vie::YoloV8DetectionDecoder>(); }));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, registry.create("yolov8_seg_raw"));
  vie::InferenceEngine engine;
  EXPECT_TRUE(engine.decoders().contains("mock"));
  EXPECT_TRUE(engine.decoders().contains("yolov8_detect_raw"));
}

TEST(YoloV8DecoderCpu, DecodesCxcywhFiltersAndMapsLetterbox) {
  std::vector<float> values(4 + 2, 0.f);
  values[0] = 10.f;
  values[1] = 10.f;
  values[2] = 4.f;
  values[3] = 6.f;
  values[5] = 0.9f;
  vie::GeometryTransform geometry = identity(40, 40);
  geometry.pad_left = 2;
  geometry.pad_top = 1;
  geometry.scale_x = 0.5;
  geometry.scale_y = 0.25;
  const vie::DetectionBatch batch = run_decoder(values, {1, 6, 1}, cxcywh_config(), geometry);
  ASSERT_EQ(batch.results.size(), 1u);
  ASSERT_EQ(batch.results[0].detections.size(), 1u);
  EXPECT_EQ(batch.results[0].detections[0].class_id, 1);
  EXPECT_EQ(batch.results[0].detections[0].query_index, -1);
  EXPECT_EQ(batch.results[0].sort_ms, 0);
  EXPECT_EQ(batch.results[0].d2h_bytes, 0u);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].confidence, 0.9f);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].box.x1, 12.f);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].box.y1, 24.f);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].box.x2, 20.f);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].box.y2, 40.f);
  EXPECT_TRUE(batch.results[0].nms_applied);
  EXPECT_TRUE(batch.results[0].coordinates_mapped);
  EXPECT_EQ(batch.results[0].execution_device, vie::DeviceType::Cpu);
}

TEST(YoloV8DecoderCpu, NmsClassAwarenessMaxDetectionsAndTies) {
  auto two_boxes = [](float score_a, float score_b, int class_a, int class_b) {
    std::vector<float> values(6 * 2, 0.f);
    auto put = [&](int channel, int prediction, float value) {
      values[static_cast<std::size_t>(channel * 2 + prediction)] = value;
    };
    put(0, 0, 10.f);
    put(0, 1, 10.f);
    put(1, 0, 10.f);
    put(1, 1, 10.f);
    put(2, 0, 4.f);
    put(2, 1, 4.f);
    put(3, 0, 4.f);
    put(3, 1, 4.f);
    put(4 + class_a, 0, score_a);
    put(4 + class_b, 1, score_b);
    return values;
  };
  vie::DecoderConfig config = cxcywh_config();
  const vie::DetectionBatch same =
      run_decoder(two_boxes(0.9f, 0.8f, 0, 0), {1, 6, 2}, config, identity(30, 30));
  ASSERT_EQ(same.results[0].detections.size(), 1u);
  EXPECT_FLOAT_EQ(same.results[0].detections[0].confidence, 0.9f);

  const vie::DetectionBatch tied =
      run_decoder(two_boxes(0.9f, 0.9f, 0, 0), {1, 6, 2}, config, identity(30, 30));
  ASSERT_EQ(tied.results[0].detections.size(), 1u);
  EXPECT_FLOAT_EQ(tied.results[0].detections[0].confidence, 0.9f);

  const vie::DetectionBatch other_class =
      run_decoder(two_boxes(0.9f, 0.8f, 0, 1), {1, 6, 2}, config, identity(30, 30));
  EXPECT_EQ(other_class.results[0].detections.size(), 2u);

  config.class_aware_nms = false;
  const vie::DetectionBatch agnostic =
      run_decoder(two_boxes(0.9f, 0.8f, 0, 1), {1, 6, 2}, config, identity(30, 30));
  EXPECT_EQ(agnostic.results[0].detections.size(), 1u);

  config.class_aware_nms = true;
  config.iou_threshold = 0.99f;
  std::vector<float> apart(6 * 2, 0.f);
  auto put = [&](int channel, int prediction, float value) {
    apart[static_cast<std::size_t>(channel * 2 + prediction)] = value;
  };
  put(0, 0, 4.f);
  put(1, 0, 4.f);
  put(2, 0, 2.f);
  put(3, 0, 2.f);
  put(4, 0, 0.7f);
  put(0, 1, 20.f);
  put(1, 1, 20.f);
  put(2, 1, 2.f);
  put(3, 1, 2.f);
  put(4, 1, 0.6f);
  config.max_detections = 1;
  const vie::DetectionBatch capped = run_decoder(apart, {1, 6, 2}, config, identity(30, 30));
  ASSERT_EQ(capped.results[0].detections.size(), 1u);
  EXPECT_FLOAT_EQ(capped.results[0].detections[0].confidence, 0.7f);

  config.confidence_threshold = 0.95f;
  config.max_detections = 10;
  const vie::DetectionBatch empty = run_decoder(apart, {1, 6, 2}, config, identity(30, 30));
  EXPECT_TRUE(empty.results[0].detections.empty());
}

TEST(YoloV8DecoderCpu, MultiLabelDflScalesAndOddLetterbox) {
  vie::DecoderConfig config = cxcywh_config();
  config.class_selection = vie::ClassSelection::AllAboveThreshold;
  std::vector<float> values(6, 0.f);
  values[0] = 8.f;
  values[1] = 8.f;
  values[2] = 4.f;
  values[3] = 4.f;
  values[4] = 0.6f;
  values[5] = 0.7f;
  const vie::DetectionBatch both = run_decoder(values, {1, 6, 1}, config, identity(20, 20));
  ASSERT_EQ(both.results[0].detections.size(), 2u);
  EXPECT_EQ(both.results[0].detections[0].class_id, 1);
  EXPECT_EQ(both.results[0].detections[1].class_id, 0);

  config = cxcywh_config();
  config.box_decode = vie::BoxDecode::Dfl;
  config.box_channels = 8;
  config.reg_max = 2;
  config.num_classes = 1;
  config.levels = {vie::FeatureLevel{8, 1, 1}, vie::FeatureLevel{16, 1, 2}};
  std::vector<float> dfl(9 * 3, 0.f);
  for (int prediction = 0; prediction < 3; ++prediction) {
    for (int side = 0; side < 4; ++side) {
      dfl[static_cast<std::size_t>((side * 2 + 1) * 3 + prediction)] = 20.f;
    }
    dfl[static_cast<std::size_t>(8 * 3 + prediction)] = prediction == 2 ? 0.2f : 0.95f;
  }
  const vie::DetectionBatch scales = run_decoder(dfl, {1, 9, 3}, config, identity(64, 64));
  ASSERT_EQ(scales.results[0].detections.size(), 2u);
  EXPECT_NEAR(scales.results[0].detections[0].box.x2, 12.f, 1e-3f);
  EXPECT_NEAR(scales.results[0].detections[1].box.x2, 24.f, 1e-3f);

  config = cxcywh_config();
  config.scores_are_probabilities = false;
  std::vector<float> logit(6, 0.f);
  logit[0] = 5.f;
  logit[1] = 5.f;
  logit[2] = 2.f;
  logit[3] = 2.f;
  logit[4] = 100.f;
  const vie::DetectionBatch activated = run_decoder(logit, {1, 6, 1}, config, identity(20, 20));
  ASSERT_EQ(activated.results[0].detections.size(), 1u);
  EXPECT_NEAR(activated.results[0].detections[0].confidence, 1.f, 1e-4f);

  std::vector<float> missing(6, 0.f);
  missing[0] = std::numeric_limits<float>::infinity();
  missing[2] = 2.f;
  missing[3] = 2.f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, run_decoder(missing, {1, 6, 1}, cxcywh_config(), identity(8, 8)));
  vie::DecoderConfig bare = cxcywh_config();
  bare.class_selection.reset();
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, run_decoder(values, {1, 6, 1}, bare, identity(8, 8)));
  vie::DecoderInput no_geometry;
  no_geometry.outputs.emplace("predictions", vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32, {1, 6, 1}));
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   vie::YoloV8DetectionDecoder{}.decode_batch(no_geometry, cxcywh_config(), vie::DeviceContext{}));
}

TEST(YoloV8DecoderCpu, ManifestDoesNotInventNmsAndTensorExecuteStillWorks) {
  vie::ModelManifest manifest = vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, vie::decoder_config_from_manifest(manifest));
  vie::InferenceEngine engine;
  vie::ModelConfig loaded;
  loaded.manifest = manifest;
  engine.load(loaded);
  std::vector<std::uint8_t> pixels(16 * 3, 1);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = 4;
  image.height = 4;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, engine.infer(image));
  vie::BackendRequest request;
  request.inputs.emplace("images", vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32, {1, 3, 8, 8}));
  EXPECT_EQ(engine.execute(request).tensors.count("output0"), 1u);
}

#if defined(VIE_HAS_ONNXRUNTIME)
TEST(YoloV8DecoderCpu, RealModelPipelineUsesExplicitTestConfig) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
  const std::string path = vie_fixture("testdata/onnx/yolov8n.onnx");
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult recognition = vie::default_recognizer_registry().recognize(report);
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, recognition, path);
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  manifest.decoder.class_selection = "best_class";
  manifest.nms.enabled = true;
  manifest.nms.confidence_threshold = 0.25f;
  manifest.nms.iou_threshold = 0.45f;
  manifest.nms.max_detections = 20;
  manifest.nms.pre_nms_top_k = 300;
  manifest.nms.class_aware = true;
  const char* fields[] = {"input.color", "input.normalization", "input.resize", "input.interpolation",
                          "input.padding_value", "decoder.class_selection", "nms.enabled",
                          "nms.confidence_threshold", "nms.iou_threshold", "nms.max_detections",
                          "nms.pre_nms_top_k", "nms.class_aware"};
  for (const char* field : fields) {
    manifest.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {"explicit test configuration"}};
  }
  vie::ModelConfig config;
  config.manifest = manifest;
  config.model_path = path;
  config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  engine.load(config);
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(48 * 32 * 3), 0);
  for (int y = 8; y < 24; ++y) {
    for (int x = 12; x < 36; ++x) {
      pixels[static_cast<std::size_t>((y * 48 + x) * 3)] = 220;
      pixels[static_cast<std::size_t>((y * 48 + x) * 3 + 1)] = 30;
      pixels[static_cast<std::size_t>((y * 48 + x) * 3 + 2)] = 30;
    }
  }
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = 48;
  image.height = 32;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  const vie::DetectionBatch batch = engine.infer(image);
  ASSERT_EQ(batch.results.size(), 1u);
  EXPECT_EQ(batch.results[0].original_width, 48);
  EXPECT_EQ(batch.results[0].original_height, 32);
  EXPECT_TRUE(batch.results[0].coordinates_mapped);
  EXPECT_EQ(batch.results[0].protocol, "yolov8_detect_raw");
  for (const vie::Detection& detection : batch.results[0].detections) {
    EXPECT_TRUE(std::isfinite(detection.confidence));
    EXPECT_GE(detection.confidence, 0.25f);
    EXPECT_GE(detection.box.x1, 0.f);
    EXPECT_GE(detection.box.y1, 0.f);
    EXPECT_LE(detection.box.x2, 48.f);
    EXPECT_LE(detection.box.y2, 32.f);
    EXPECT_LT(detection.box.x1, detection.box.x2);
    EXPECT_LT(detection.box.y1, detection.box.y2);
  }
}
#endif

TEST(DecoderBenchmark, CpuStages) {
  constexpr int kPredictions = 8400;
  constexpr int kClasses = 80;
  constexpr int kRuns = 10;
  std::vector<float> values(static_cast<std::size_t>(4 + kClasses) * kPredictions, 0.f);
  for (int i = 0; i < 300; ++i) {
    const int prediction = i * 17;
    values[static_cast<std::size_t>(prediction)] = static_cast<float>((i % 40) + 4);
    values[static_cast<std::size_t>(kPredictions + prediction)] = static_cast<float>((i % 40) + 4);
    values[static_cast<std::size_t>(2 * kPredictions + prediction)] = 8.f;
    values[static_cast<std::size_t>(3 * kPredictions + prediction)] = 8.f;
    values[static_cast<std::size_t>((4 + (i % kClasses)) * kPredictions + prediction)] = 0.9f;
  }
  vie::DecoderConfig config = cxcywh_config();
  config.num_classes = kClasses;
  config.pre_nms_top_k = 1000;
  config.max_detections = 100;
  vie::Tensor tensor = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 4 + kClasses, kPredictions},
                                              values.data(), values.size() * sizeof(float));
  vie::DecoderInput input;
  input.outputs.emplace("predictions", tensor);
  input.geometry = identity(640, 640);
  vie::YoloV8DetectionDecoder decoder;
  decoder.decode_batch(input, config, vie::DeviceContext{});
  std::vector<double> samples;
  int kept = 0;
  int before = 0;
  for (int run = 0; run < kRuns; ++run) {
    const auto start = std::chrono::steady_clock::now();
    const vie::DetectionBatch batch = decoder.decode_batch(input, config, vie::DeviceContext{});
    const auto stop = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    kept = static_cast<int>(batch.results[0].detections.size());
    before = batch.results[0].candidates_before_nms;
  }
  std::vector<double> dfl_samples;
  std::vector<float> logits(16, 0.1f);
  logits[3] = 4.f;
  for (int run = 0; run < kRuns; ++run) {
    const auto start = std::chrono::steady_clock::now();
    float sink = 0.f;
    for (int i = 0; i < kPredictions * 4; ++i) sink += vie::dfl_expectation(logits.data(), 16);
    const auto stop = std::chrono::steady_clock::now();
    dfl_samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    if (sink == 0.f) FAIL();
  }
  auto summarize = [](std::vector<double> values_in) {
    std::sort(values_in.begin(), values_in.end());
    double sum = 0;
    for (double value : values_in) sum += value;
    return std::tuple<double, double, double>{sum / static_cast<double>(values_in.size()),
                                              values_in[values_in.size() / 2],
                                              values_in[static_cast<std::size_t>(values_in.size() * 0.95)]};
  };
  vie::DecoderConfig filter_only = config;
  filter_only.nms_enabled = false;
  std::vector<double> filter_samples;
  for (int run = 0; run < kRuns; ++run) {
    const auto start = std::chrono::steady_clock::now();
    decoder.decode_batch(input, filter_only, vie::DeviceContext{});
    const auto stop = std::chrono::steady_clock::now();
    filter_samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  const auto full = summarize(samples);
  const auto dfl = summarize(dfl_samples);
  const auto filtered = summarize(filter_samples);
  std::ofstream out("/tmp/vie_phase432_cpu.txt");
  out << "predictions=" << kPredictions << " classes=" << kClasses << " kept=" << kept
      << " before_nms=" << before << " runs=" << kRuns << "\n";
  out << "dfl_mean_ms=" << std::get<0>(dfl) << " p50=" << std::get<1>(dfl) << " p95=" << std::get<2>(dfl) << "\n";
  out << "filter_map_mean_ms=" << std::get<0>(filtered) << " p50=" << std::get<1>(filtered)
      << " p95=" << std::get<2>(filtered) << "\n";
  out << "full_mean_ms=" << std::get<0>(full) << " p50=" << std::get<1>(full) << " p95=" << std::get<2>(full) << "\n";
  EXPECT_GT(kept, 0);
}
