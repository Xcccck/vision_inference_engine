#include "test_util.hpp"
#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/decoder/decoder_registry.hpp"
#include "vie/decoder/rtdetr_decoder.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/preprocess/preprocess.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

vie::GeometryTransform stretch_geometry(int source_width, int source_height, int model_width, int model_height) {
  vie::PreprocessConfig config;
  config.resize_mode = vie::ResizeMode::Stretch;
  config.target_width = model_width;
  config.target_height = model_height;
  return vie::make_geometry(config, source_width, source_height);
}

vie::DecoderConfig rtdetr_config(int queries, int classes, float threshold, int max_detections) {
  vie::DecoderConfig config;
  config.output_name = "output0";
  config.channel_axis = 2;
  config.box_channels = 4;
  config.num_classes = classes;
  config.num_queries = queries;
  config.box_decode = vie::BoxDecode::DecodedNormalizedCxcywh;
  config.scores_are_probabilities = true;
  config.class_selection = vie::ClassSelection::BestClass;
  config.confidence_threshold = threshold;
  config.nms_enabled = false;
  config.max_detections = max_detections;
  return config;
}

vie::Tensor prediction_tensor(const std::vector<std::vector<float>>& rows) {
  const int queries = static_cast<int>(rows.size());
  const int channels = static_cast<int>(rows.front().size());
  std::vector<float> packed;
  packed.reserve(static_cast<std::size_t>(queries * channels));
  for (const std::vector<float>& row : rows) packed.insert(packed.end(), row.begin(), row.end());
  return vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32,
                                {1, static_cast<std::int64_t>(queries), static_cast<std::int64_t>(channels)},
                                packed.data(), packed.size() * sizeof(float));
}

vie::DetectionBatch decode_rows(const std::vector<std::vector<float>>& rows, const vie::DecoderConfig& config,
                                const vie::GeometryTransform& geometry) {
  vie::DecoderInput input;
  input.outputs.emplace(config.output_name, prediction_tensor(rows));
  input.geometry = geometry;
  return vie::RtdetrDecoder{}.decode_batch(input, config, vie::DeviceContext{});
}

float iou(const vie::BoundingBox& a, const vie::BoundingBox& b) {
  const float left = std::max(a.x1, b.x1);
  const float top = std::max(a.y1, b.y1);
  const float right = std::min(a.x2, b.x2);
  const float bottom = std::min(a.y2, b.y2);
  const float width = right - left;
  const float height = bottom - top;
  if (width <= 0.f || height <= 0.f) return 0.f;
  const float intersection = width * height;
  const float area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection;
  return area > 0.f ? intersection / area : 0.f;
}

vie::ModelManifest generated_manifest() {
  const std::string path = vie_fixture("testdata/onnx/rtdetr-l.onnx");
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  return vie::ManifestGenerator{}.generate(report, result, path);
}

vie::FieldProvenance user_field(const char* evidence) {
  return vie::FieldProvenance{vie::Certainty::User, {evidence}};
}

vie::ModelManifest runtime_manifest(const vie::ModelManifest& generated) {
  vie::ModelManifest manifest = generated;
  manifest.input.color = "rgb";
  manifest.provenance["input.color"] = user_field(
      "Ultralytics pre_transform converts BGR to RGB; supplied by the caller, not observed in the graph");
  manifest.input.normalization = "scale_0_1";
  manifest.provenance["input.normalization"] = user_field(
      "Ultralytics divides the stretched image by 255; supplied by the caller, not observed in the graph");
  manifest.input.interpolation = "bilinear";
  manifest.provenance["input.interpolation"] = user_field(
      "Ultralytics LetterBox uses cv2.INTER_LINEAR; supplied by the caller, not observed in the graph");
  manifest.outputs[0].shape = {1, static_cast<std::int64_t>(*manifest.decoder.num_queries),
                               static_cast<std::int64_t>(4 + *manifest.num_classes)};
  manifest.provenance["outputs.0.shape"] = user_field(
      "runtime profile observed from execution; the ONNX graph still declares three symbolic dimensions");
  manifest.decoder.class_selection = "best_class";
  manifest.provenance["decoder.class_selection"] = user_field(
      "official postprocess keeps the maximum class probability of each query");
  manifest.nms.confidence_threshold = 0.25f;
  manifest.provenance["nms.confidence_threshold"] =
      user_field("caller threshold, the Ultralytics predict default of 0.25");
  manifest.nms.max_detections = 300;
  manifest.provenance["nms.max_detections"] =
      user_field("caller cap matching Ultralytics max_det; this is not the in-graph TopK");
  return manifest;
}

#if defined(VIE_HAS_ONNXRUNTIME)

vie::ImageInput load_ppm(const std::string& path, std::vector<std::uint8_t>& pixels) {
  std::ifstream image_file(path, std::ios::binary);
  std::string magic;
  int width = 0;
  int height = 0;
  int max_value = 0;
  image_file >> magic >> width >> height >> max_value;
  image_file.get();
  pixels.resize(static_cast<std::size_t>(width * height * 3));
  image_file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

struct OfficialDetection {
  int class_id = -1;
  float confidence = 0.f;
  vie::BoundingBox box;
};

struct OfficialResult {
  int width = 0;
  int height = 0;
  std::vector<OfficialDetection> detections;
  std::vector<std::int64_t> raw_shape;
  std::string raw_file;
  std::vector<std::int64_t> pre_shape;
  std::string pre_file;
};

std::vector<std::int64_t> parse_dims(std::istream& input) {
  std::vector<std::int64_t> dims;
  std::int64_t dim = 0;
  while (input >> dim) dims.push_back(dim);
  return dims;
}

OfficialResult read_official(const std::string& path) {
  std::ifstream input(path);
  OfficialResult result;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    std::string kind;
    row >> kind;
    if (kind == "image") {
      row >> result.width >> result.height;
    } else if (kind == "det") {
      OfficialDetection detection;
      row >> detection.class_id >> detection.confidence >> detection.box.x1 >> detection.box.y1 >> detection.box.x2 >>
          detection.box.y2;
      result.detections.push_back(detection);
    } else if (kind == "raw_shape") {
      result.raw_shape = parse_dims(row);
    } else if (kind == "raw_file") {
      row >> result.raw_file;
    } else if (kind == "pre_shape") {
      result.pre_shape = parse_dims(row);
    } else if (kind == "pre_file") {
      row >> result.pre_file;
    }
  }
  return result;
}

std::vector<float> read_f32(const std::string& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  std::vector<float> values(count);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
  return values;
}

struct MatchStats {
  float min_iou = 1.f;
  float max_corner = 0.f;
  float max_confidence = 0.f;
};

MatchStats match_detections(const std::vector<vie::Detection>& actual, const std::vector<OfficialDetection>& official) {
  std::vector<char> used(official.size(), 0);
  MatchStats stats;
  for (const vie::Detection& detection : actual) {
    int best = -1;
    float best_iou = -1.f;
    for (std::size_t index = 0; index < official.size(); ++index) {
      if (used[index] != 0 || official[index].class_id != detection.class_id) continue;
      const float overlap = iou(detection.box, official[index].box);
      if (overlap > best_iou) {
        best_iou = overlap;
        best = static_cast<int>(index);
      }
    }
    EXPECT_GE(best, 0);
    if (best < 0) continue;
    used[static_cast<std::size_t>(best)] = 1;
    stats.min_iou = std::min(stats.min_iou, best_iou);
    const vie::BoundingBox& expected = official[static_cast<std::size_t>(best)].box;
    stats.max_corner = std::max(stats.max_corner, std::abs(detection.box.x1 - expected.x1));
    stats.max_corner = std::max(stats.max_corner, std::abs(detection.box.y1 - expected.y1));
    stats.max_corner = std::max(stats.max_corner, std::abs(detection.box.x2 - expected.x2));
    stats.max_corner = std::max(stats.max_corner, std::abs(detection.box.y2 - expected.y2));
    stats.max_confidence =
        std::max(stats.max_confidence, std::abs(detection.confidence - official[static_cast<std::size_t>(best)].confidence));
  }
  return stats;
}

#endif

}  // namespace

TEST(RtdetrDecoder, SelectsTheMaxClassAndDoesNotSigmoid) {
  const vie::GeometryTransform geometry = stretch_geometry(100, 50, 640, 640);
  const vie::DecoderConfig config = rtdetr_config(1, 3, 0.1f, 10);
  const vie::DetectionBatch batch = decode_rows({{0.5f, 0.5f, 0.2f, 0.2f, 0.2f, 0.8f, 0.2f}}, config, geometry);
  ASSERT_EQ(batch.results.size(), 1u);
  ASSERT_EQ(batch.results[0].detections.size(), 1u);
  EXPECT_EQ(batch.results[0].detections[0].class_id, 1);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].confidence, 0.8f);
  EXPECT_NEAR(batch.results[0].detections[0].box.x1, 40.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.x2, 60.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.y1, 20.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.y2, 30.f, 1e-3f);
  EXPECT_FALSE(batch.results[0].nms_applied);
  EXPECT_TRUE(batch.results[0].coordinates_mapped);
  EXPECT_EQ(batch.results[0].protocol, "rtdetr_detection_raw");
}

TEST(RtdetrDecoder, UsesSeparateScalesAndClipsToTheImage) {
  const vie::GeometryTransform geometry = stretch_geometry(100, 50, 640, 640);
  EXPECT_NE(geometry.scale_x, geometry.scale_y);
  const vie::DecoderConfig config = rtdetr_config(1, 1, 0.1f, 10);
  const vie::DetectionBatch batch = decode_rows({{0.f, 0.5f, 0.5f, 0.2f, 0.9f}}, config, geometry);
  ASSERT_EQ(batch.results[0].detections.size(), 1u);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].box.x1, 0.f);
  EXPECT_NEAR(batch.results[0].detections[0].box.x2, 25.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.y1, 20.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.y2, 30.f, 1e-3f);
}

TEST(RtdetrDecoder, MappingUsesPadAndDoesNotTreatNormalizedValuesAsPixels) {
  vie::GeometryTransform geometry = stretch_geometry(100, 100, 640, 640);
  geometry.pad_left = 10;
  geometry.pad_top = 20;
  geometry.scale_x = 6.0;
  geometry.scale_y = 4.0;
  const vie::DetectionBatch batch =
      decode_rows({{0.5f, 0.5f, 0.1f, 0.1f, 0.9f}}, rtdetr_config(1, 1, 0.1f, 10), geometry);
  ASSERT_EQ(batch.results[0].detections.size(), 1u);
  EXPECT_NEAR(batch.results[0].detections[0].box.x1, (288.f - 10.f) / 6.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.y1, (288.f - 20.f) / 4.f, 1e-3f);
}

TEST(RtdetrDecoder, FiltersSortsCapsAndKeepsTiesByQueryIndex) {
  const vie::GeometryTransform geometry = stretch_geometry(64, 32, 64, 32);
  const vie::DecoderConfig config = rtdetr_config(4, 2, 0.5f, 2);
  const vie::DetectionBatch batch = decode_rows(
      {{0.2f, 0.3f, 0.1f, 0.2f, 0.4f, 0.1f},
       {0.4f, 0.4f, 0.2f, 0.2f, 0.7f, 0.7f},
       {0.6f, 0.5f, 0.2f, 0.2f, 0.9f, 0.1f},
       {0.8f, 0.6f, 0.1f, 0.2f, 0.6f, 0.2f}},
      config, geometry);
  ASSERT_EQ(batch.results[0].detections.size(), 2u);
  EXPECT_EQ(batch.results[0].candidates_after_filter, 3);
  EXPECT_EQ(batch.results[0].detections[0].class_id, 0);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].confidence, 0.9f);
  EXPECT_EQ(batch.results[0].detections[1].class_id, 0);
  EXPECT_FLOAT_EQ(batch.results[0].detections[1].confidence, 0.7f);
  EXPECT_LT(batch.results[0].detections[1].box.x1, batch.results[0].detections[0].box.x1);

  const vie::DecoderConfig ties = rtdetr_config(3, 1, 0.3f, 10);
  const vie::DetectionBatch tied =
      decode_rows({{0.2f, 0.5f, 0.1f, 0.2f, 0.4f},
                   {0.5f, 0.5f, 0.1f, 0.2f, 0.4f},
                   {0.8f, 0.5f, 0.1f, 0.2f, 0.9f}},
                  ties, geometry);
  ASSERT_EQ(tied.results[0].detections.size(), 3u);
  EXPECT_FLOAT_EQ(tied.results[0].detections[0].confidence, 0.9f);
  EXPECT_LT(tied.results[0].detections[1].box.x1, tied.results[0].detections[2].box.x1);
}

TEST(RtdetrDecoder, QueryIndexIsTheSourceQueryNotTheSortedPosition) {
  const vie::GeometryTransform geometry = stretch_geometry(64, 32, 64, 32);
  const vie::DetectionBatch batch = decode_rows(
      {{0.2f, 0.5f, 0.1f, 0.2f, 0.4f},
       {0.5f, 0.5f, 0.1f, 0.2f, 0.55f},
       {0.5f, 0.5f, 0.f, 0.2f, 0.99f},
       {0.8f, 0.5f, 0.1f, 0.2f, 0.95f},
       {0.7f, 0.5f, 0.1f, 0.2f, 0.55f}},
      rtdetr_config(5, 1, 0.5f, 10), geometry);
  ASSERT_EQ(batch.results[0].detections.size(), 3u);
  EXPECT_EQ(batch.results[0].detections[0].query_index, 3);
  EXPECT_EQ(batch.results[0].detections[1].query_index, 1);
  EXPECT_EQ(batch.results[0].detections[2].query_index, 4);
  EXPECT_NE(batch.results[0].detections[0].query_index, 0);
  EXPECT_NE(batch.results[0].detections[2].query_index, 2);
  EXPECT_EQ(batch.results[0].sort_ms, 0);
  EXPECT_EQ(batch.results[0].d2h_bytes, 0u);
}

TEST(RtdetrDecoder, NonFiniteErrorNamesTheEarliestQueryAndField) {
  const vie::GeometryTransform geometry = stretch_geometry(16, 16, 16, 16);
  try {
    decode_rows({{0.5f, 0.5f, 0.2f, 0.2f, 0.9f},
                 {0.5f, 0.5f, 0.2f, 0.2f, std::nanf("")},
                 {std::nanf(""), 0.5f, 0.2f, 0.2f, 0.9f}},
                rtdetr_config(3, 1, 0.1f, 4), geometry);
    FAIL() << "expected VieError";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::InvalidArgument);
    EXPECT_EQ(std::string(error.what()), "class score is not finite at query 1");
  }
  try {
    decode_rows({{0.5f, std::nanf(""), INFINITY, 0.2f, std::nanf("")}}, rtdetr_config(1, 1, 0.1f, 4), geometry);
    FAIL() << "expected VieError";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::InvalidArgument);
    EXPECT_EQ(std::string(error.what()), "normalized cy is not finite at query 0");
  }
}

TEST(RtdetrDecoder, EmptyResultIsAValidBatchAndOverlapsAreNotSuppressed) {
  const vie::GeometryTransform geometry = stretch_geometry(40, 20, 40, 20);
  const vie::DetectionBatch empty_batch =
      decode_rows({{0.5f, 0.5f, 0.2f, 0.2f, 0.2f}, {0.2f, 0.2f, 0.1f, 0.1f, 0.2f}}, rtdetr_config(2, 1, 0.5f, 10),
                  geometry);
  ASSERT_EQ(empty_batch.results.size(), 1u);
  EXPECT_TRUE(empty_batch.results[0].detections.empty());
  EXPECT_EQ(empty_batch.results[0].original_width, 40);
  EXPECT_EQ(empty_batch.results[0].original_height, 20);
  EXPECT_FALSE(empty_batch.results[0].nms_applied);
  EXPECT_EQ(empty_batch.results[0].candidates_after_filter, 0);

  const vie::DetectionBatch overlaps =
      decode_rows({{0.5f, 0.5f, 0.8f, 0.8f, 0.95f}, {0.52f, 0.5f, 0.8f, 0.8f, 0.9f}},
                  rtdetr_config(2, 1, 0.25f, 10), geometry);
  ASSERT_EQ(overlaps.results[0].detections.size(), 2u);
  EXPECT_GT(iou(overlaps.results[0].detections[0].box, overlaps.results[0].detections[1].box), 0.5f);
  EXPECT_FALSE(overlaps.results[0].nms_applied);

  vie::DecoderConfig topk = rtdetr_config(2, 1, 0.25f, 10);
  topk.pre_nms_top_k = 1;
  const vie::DetectionBatch ignored =
      decode_rows({{0.3f, 0.5f, 0.2f, 0.2f, 0.8f}, {0.7f, 0.5f, 0.2f, 0.2f, 0.7f}}, topk, geometry);
  EXPECT_EQ(ignored.results[0].detections.size(), 2u);
}

TEST(RtdetrDecoder, RejectsNonFiniteScoresAndDisabledNmsIsRequired) {
  const vie::GeometryTransform geometry = stretch_geometry(16, 16, 16, 16);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decode_rows({{0.5f, 0.5f, 0.2f, 0.2f, std::nanf("")}}, rtdetr_config(1, 1, 0.1f, 4), geometry));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decode_rows({{0.5f, 0.5f, 0.2f, 0.2f, INFINITY}},
                                                               rtdetr_config(1, 1, 0.1f, 4), geometry));
  vie::DecoderConfig nms = rtdetr_config(1, 1, 0.1f, 4);
  nms.nms_enabled = true;
  nms.iou_threshold = 0.45f;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, decode_rows({{0.5f, 0.5f, 0.2f, 0.2f, 0.9f}}, nms, geometry));
  vie::DecoderInput missing_geometry;
  missing_geometry.outputs.emplace("output0", prediction_tensor({{0.5f, 0.5f, 0.2f, 0.2f, 0.9f}}));
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   vie::RtdetrDecoder{}.decode_batch(missing_geometry, rtdetr_config(1, 1, 0.1f, 4),
                                                     vie::DeviceContext{}));
}

TEST(RtdetrDecoder, GeneratedManifestStaysSymbolicAndRuntimeProfileIsSeparate) {
  const vie::ModelManifest generated = generated_manifest();
  EXPECT_EQ(generated.outputs[0].shape, (std::vector<std::int64_t>{-1, -1, -1}));
  EXPECT_EQ(generated.find_provenance("outputs.0.shape")->certainty, vie::Certainty::Observed);
  EXPECT_TRUE(generated.input.color.empty());
  EXPECT_TRUE(generated.input.normalization.empty());
  EXPECT_TRUE(generated.input.interpolation.empty());
  EXPECT_EQ(generated.input.resize, "stretch");
  EXPECT_FALSE(*generated.nms.enabled);
  EXPECT_FALSE(generated.nms.pre_nms_top_k.has_value());
  EXPECT_FALSE(generated.nms.iou_threshold.has_value());
  EXPECT_FALSE(vie::ManifestValidator{}.validate(generated).runnable());
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, vie::decoder_config_from_manifest(generated));

  const vie::ModelManifest runtime = runtime_manifest(generated);
  EXPECT_EQ(runtime.outputs[0].shape, (std::vector<std::int64_t>{1, 300, 84}));
  EXPECT_EQ(runtime.find_provenance("outputs.0.shape")->certainty, vie::Certainty::User);
  EXPECT_EQ(runtime.find_provenance("input.color")->certainty, vie::Certainty::User);
  EXPECT_EQ(runtime.input.color, "rgb");
  EXPECT_EQ(runtime.input.normalization, "scale_0_1");
  EXPECT_EQ(runtime.input.interpolation, "bilinear");
  EXPECT_FALSE(runtime.nms.iou_threshold.has_value());
  EXPECT_FALSE(runtime.nms.pre_nms_top_k.has_value());
  const vie::DecoderConfig config = vie::decoder_config_from_manifest(runtime);
  EXPECT_EQ(config.box_decode, vie::BoxDecode::DecodedNormalizedCxcywh);
  EXPECT_EQ(config.num_classes, 80);
  ASSERT_TRUE(config.num_queries.has_value());
  EXPECT_EQ(*config.num_queries, 300);
  EXPECT_FALSE(config.pre_nms_top_k.has_value());
  EXPECT_FALSE(config.iou_threshold.has_value());
  EXPECT_TRUE(vie::ManifestValidator{}.validate(runtime).runnable())
      << vie::format_issues(vie::ManifestValidator{}.validate(runtime).unresolved_critical);

  vie::InferenceEngine engine;
  EXPECT_TRUE(engine.decoders().contains("rtdetr_detection_raw"));
  EXPECT_TRUE(engine.decoders().contains("yolov8_detect_raw"));
  EXPECT_TRUE(engine.decoders().contains("yolov8_seg_raw"));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.decoders().create("not_a_protocol"));
}

#if defined(VIE_HAS_ONNXRUNTIME)

TEST(RtdetrCpu, BusMatchesOfficialPredictAndSeparatesPreprocessFromDecode) {
  const std::string weight = "/tmp/rtdetr_export/rtdetr-l.pt";
  std::ifstream weight_file(weight, std::ios::binary);
  ASSERT_TRUE(weight_file) << "official comparison needs " << weight;
  const std::string image_path = vie_fixture("testdata/images/bus.ppm");
  const std::string out_path = "/tmp/vie_phase53_official.txt";
  const std::string command = "python3 " + vie_fixture("tools/rtdetr_official_predict.py") + " " + image_path + " " +
                              weight + " " + out_path + " >/tmp/vie_phase53_official.log 2>&1";
  ASSERT_EQ(std::system(command.c_str()), 0) << std::ifstream("/tmp/vie_phase53_official.log").rdbuf();
  const OfficialResult official = read_official(out_path);
  ASSERT_EQ(official.detections.size(), 9u);
  ASSERT_EQ(official.raw_shape, (std::vector<std::int64_t>{1, 300, 84}));
  ASSERT_EQ(official.pre_shape, (std::vector<std::int64_t>{1, 3, 640, 640}));

  const vie::ModelManifest runtime = runtime_manifest(generated_manifest());
  const vie::DecoderConfig config = vie::decoder_config_from_manifest(runtime);
  const std::size_t raw_count = static_cast<std::size_t>(300 * 84);
  const std::vector<float> raw = read_f32(official.raw_file, raw_count);
  vie::Tensor raw_tensor = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 300, 84}, raw.data(),
                                                  raw.size() * sizeof(float));
  vie::DecoderInput raw_input;
  raw_input.outputs.emplace(config.output_name, std::move(raw_tensor));
  raw_input.geometry = stretch_geometry(official.width, official.height, 640, 640);
  const vie::DetectionBatch decoded =
      vie::RtdetrDecoder{}.decode_batch(raw_input, config, vie::DeviceContext{});
  ASSERT_EQ(decoded.results[0].detections.size(), official.detections.size());
  const MatchStats decoder_only = match_detections(decoded.results[0].detections, official.detections);
  EXPECT_GT(decoder_only.min_iou, 0.999f);
  EXPECT_LT(decoder_only.max_corner, 0.05f);
  EXPECT_LT(decoder_only.max_confidence, 1e-5f);

  vie::ModelConfig model_config;
  model_config.model_path = vie_fixture("testdata/onnx/rtdetr-l.onnx");
  model_config.device = vie::DeviceType::Cpu;
  model_config.manifest = runtime;
  vie::InferenceEngine engine;
  engine.load(model_config);
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_ppm(image_path, pixels);
  const vie::PreprocessResult prepared = engine.preprocess(image);
  EXPECT_EQ(prepared.geometry.resize_mode, vie::ResizeMode::Stretch);
  EXPECT_NE(prepared.geometry.scale_x, prepared.geometry.scale_y);
  EXPECT_EQ(prepared.geometry.pad_left, 0);
  EXPECT_EQ(prepared.geometry.pad_top, 0);
  const std::vector<float> official_input = read_f32(official.pre_file, static_cast<std::size_t>(1 * 3 * 640 * 640));
  const float* vie_input = prepared.output.data_as<float>();
  float max_abs = 0.f;
  double mean_abs = 0.0;
  int above_lsb = 0;
  for (std::size_t index = 0; index < official_input.size(); ++index) {
    const float delta = std::abs(vie_input[index] - official_input[index]);
    max_abs = std::max(max_abs, delta);
    mean_abs += delta;
    if (delta > 1.f / 255.f) ++above_lsb;
  }
  mean_abs /= static_cast<double>(official_input.size());
  EXPECT_LT(max_abs, 0.01f);
  std::ofstream preprocess_note("/tmp/vie_phase53_preprocess.txt");
  preprocess_note << "max_abs " << max_abs << "\nmean_abs " << mean_abs << "\nabove_1_255 " << above_lsb
                  << "\nscale_x " << prepared.geometry.scale_x << "\nscale_y " << prepared.geometry.scale_y << "\n";

  const vie::DetectionBatch inferred = engine.infer(image);
  ASSERT_EQ(inferred.results[0].detections.size(), official.detections.size());
  EXPECT_FALSE(inferred.results[0].nms_applied);
  const MatchStats full = match_detections(inferred.results[0].detections, official.detections);
  EXPECT_GT(full.min_iou, 0.99f);
  EXPECT_LT(full.max_corner, 1.f);
  EXPECT_LT(full.max_confidence, 1e-2f);
  std::ofstream comparison("/tmp/vie_phase53_comparison.txt");
  comparison << "decoder_min_iou " << decoder_only.min_iou << "\ndecoder_max_corner " << decoder_only.max_corner
             << "\ndecoder_max_confidence " << decoder_only.max_confidence << "\nfull_min_iou " << full.min_iou
             << "\nfull_max_corner " << full.max_corner << "\nfull_max_confidence " << full.max_confidence
             << "\ncount " << inferred.results[0].detections.size() << "\n";
}

TEST(RtdetrCpu, BenchmarkLoadPreprocessExecuteDecode) {
  const vie::ModelManifest runtime = runtime_manifest(generated_manifest());
  vie::ModelConfig model_config;
  model_config.model_path = vie_fixture("testdata/onnx/rtdetr-l.onnx");
  model_config.device = vie::DeviceType::Cpu;
  model_config.manifest = runtime;
  const auto load_start = std::chrono::steady_clock::now();
  vie::InferenceEngine engine;
  engine.load(model_config);
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_ppm(vie_fixture("testdata/images/bus.ppm"), pixels);
  vie::DetectionInferTiming first;
  const vie::DetectionBatch warmup = engine.infer(image, &first);
  ASSERT_EQ(warmup.results[0].detections.size(), 9u);

  constexpr int kRuns = 100;
  std::vector<double> preprocess;
  std::vector<double> execute;
  std::vector<double> decode;
  std::vector<double> total;
  preprocess.reserve(kRuns);
  for (int run = 0; run < kRuns; ++run) {
    vie::DetectionInferTiming timing;
    const vie::DetectionBatch batch = engine.infer(image, &timing);
    EXPECT_EQ(batch.results[0].detections.size(), warmup.results[0].detections.size());
    preprocess.push_back(timing.preprocess_ms);
    execute.push_back(timing.execute_ms);
    decode.push_back(timing.decode_ms);
    total.push_back(timing.total_ms);
  }
  auto summarize = [](std::vector<double> values, std::ostream& out, const char* name) {
    std::sort(values.begin(), values.end());
    double sum = 0;
    for (double value : values) sum += value;
    const double mean = sum / static_cast<double>(values.size());
    const double p50 = values[values.size() / 2];
    const double p95 = values[static_cast<std::size_t>(std::floor(0.95 * values.size()))];
    out << name << " mean " << mean << " p50 " << p50 << " p95 " << p95 << " min " << values.front() << " max "
        << values.back() << "\n";
  };
  std::ofstream out("/tmp/vie_phase53_rtdetr_cpu.txt");
  out << "load_ms " << load_ms << "\n";
  out << "first_preprocess " << first.preprocess_ms << "\nfirst_execute " << first.execute_ms << "\nfirst_decode "
      << first.decode_ms << "\nfirst_e2e " << first.total_ms << "\n";
  out << "steady_runs " << kRuns << "\nincludes_sync " << first.includes_sync << "\nincludes_h2d " << first.includes_h2d
      << "\nincludes_d2h " << first.includes_d2h << "\n";
  summarize(preprocess, out, "preprocess");
  summarize(execute, out, "execute");
  summarize(decode, out, "decode");
  summarize(total, out, "e2e");
}

#endif
