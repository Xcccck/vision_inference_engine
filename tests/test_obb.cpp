#include "onnx_test_writer.hpp"
#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/decoder/obb.hpp"
#include "vie/decoder/yolov8_obb_decoder.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/recognizer/yolov8_detect_recognizer.hpp"
#include "vie/recognizer/yolov8_obb_recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
const char* kRealModel = "/tmp/vie_phase71/yolov8n-obb.onnx";

std::string f32_bytes(float value) {
  std::string raw(4, '\0');
  std::memcpy(raw.data(), &value, 4);
  return raw;
}

OnnxTestWriter wrap_model(const OnnxTestWriter& graph) {
  OnnxTestWriter model;
  model.i64(1, 8);
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

void add_weight(OnnxTestWriter& graph, const std::string& name, const std::vector<std::int64_t>& dims) {
  graph.message(5, tensor_message(name, 1, dims, {}, "", false));
}

void add_conv(OnnxTestWriter& graph, const std::string& name, const std::vector<std::int64_t>& weight) {
  add_weight(graph, name + "_w", weight);
  graph.message(1, node_message("Conv", {name + "_in", name + "_w"}, {name}));
}

OnnxTestWriter obb_graph(float quarter, bool extra_output) {
  OnnxTestWriter graph;
  graph.message(5, tensor_message("quarter", 1, {1}, {}, f32_bytes(quarter), false));
  graph.message(5, tensor_message("pi_scalar", 1, {1}, {}, f32_bytes(static_cast<float>(kPi)), false));
  for (int index = 0; index < 3; ++index) {
    const std::string id = std::to_string(index);
    add_conv(graph, "angle_c" + id, {1, 1, 1, 1});
    graph.message(1, node_message("Reshape", {"angle_c" + id}, {"angle_r" + id}));
    add_conv(graph, "cls_c" + id, {2, 1, 1, 1});
    graph.message(1, node_message("Reshape", {"cls_c" + id}, {"cls_r" + id}));
    add_conv(graph, "box_c" + id, {64, 1, 1, 1});
    graph.message(1, node_message("Reshape", {"box_c" + id}, {"box_r" + id}));
  }
  graph.message(1, node_message("Concat", {"angle_r0", "angle_r1", "angle_r2"}, {"angle_cat"}));
  graph.message(1, node_message("Sigmoid", {"angle_cat"}, {"angle_sig"}));
  graph.message(1, node_message("Sub", {"angle_sig", "quarter"}, {"angle_sub"}));
  graph.message(1, node_message("Mul", {"angle_sub", "pi_scalar"}, {"angle"}));
  graph.message(1, node_message("Concat", {"cls_r0", "cls_r1", "cls_r2"}, {"cls_cat"}));
  graph.message(1, node_message("Sigmoid", {"cls_cat"}, {"cls_sig"}));
  graph.message(1, node_message("Concat", {"box_r0", "box_r1", "box_r2"}, {"box_cat"}));
  graph.message(1, node_message("Softmax", {"box_cat"}, {"box_sm"}));
  add_weight(graph, "dfl_w", {1, 16, 1, 1});
  graph.message(1, node_message("Conv", {"box_sm", "dfl_w"}, {"dfl"}));
  graph.message(1, node_message("Cos", {"angle"}, {"cosine"}));
  graph.message(1, node_message("Mul", {"dfl", "cosine"}, {"box"}));
  graph.message(1, node_message("Concat", {"box", "cls_sig", "angle"}, {"output0"}));
  graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(32),
                                                               static_dim(32)})));
  graph.message(12, value_info("output0", tensor_type(1, true, {static_dim(1), static_dim(7), static_dim(12)})));
  if (extra_output) {
    graph.message(12, value_info("extra", tensor_type(1, true, {static_dim(1), static_dim(7), static_dim(12)})));
  }
  graph.text(2, "obb");
  return graph;
}

vie::ModelAnalysisReport analyze_graph(const OnnxTestWriter& graph) {
  return vie::OnnxAnalyzer{}.analyze_bytes(wrap_model(graph).str(), "obb.onnx", ".");
}

vie::GeometryTransform identity_geometry() {
  vie::GeometryTransform geometry;
  geometry.original_width = 64;
  geometry.original_height = 64;
  geometry.output_width = 64;
  geometry.output_height = 64;
  geometry.resized_width = 64;
  geometry.resized_height = 64;
  geometry.scale_x = 1;
  geometry.scale_y = 1;
  geometry.uniform_scale = 1;
  geometry.resize_mode = vie::ResizeMode::Stretch;
  return geometry;
}

vie::ObbDecodeConfig keep_config() {
  vie::ObbDecodeConfig config;
  config.output_name = "output0";
  config.channel_axis = 1;
  config.num_classes = 2;
  config.class_selection = vie::ClassSelection::BestClass;
  config.confidence_threshold = 0.5f;
  config.nms_enabled = true;
  config.class_aware = true;
  config.iou_threshold = 0.5f;
  config.max_detections = 10;
  config.pre_nms_top_k = 10;
  return config;
}

bool point_in_box(double x, double y, const vie::OrientedBox& box) {
  const double dx = x - box.cx;
  const double dy = y - box.cy;
  const double cosine = std::cos(box.angle);
  const double sine = std::sin(box.angle);
  const double along = dx * cosine + dy * sine;
  const double across = -dx * sine + dy * cosine;
  return std::fabs(along) <= static_cast<double>(box.width) * 0.5 + 1e-2 &&
         std::fabs(across) <= static_cast<double>(box.height) * 0.5 + 1e-2;
}

vie::ModelManifest runnable_obb_manifest() {
  vie::ModelManifest manifest;
  manifest.model_name = "obb";
  manifest.task = vie::TaskType::Obb;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 2;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 32, 32};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  manifest.input.interpolation = "nearest";
  vie::OutputDesc output;
  output.name = "output0";
  output.shape = {1, 7, 4};
  output.dtype = vie::DataType::Fp32;
  output.semantic = "oriented_boxes";
  manifest.outputs.push_back(output);
  manifest.decoder.protocol = "yolov8_obb_raw";
  manifest.decoder.box_format = "cxcywh_angle";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.scores_are_probabilities = true;
  manifest.obb.output_name = "output0";
  manifest.obb.channel_axis = 1;
  manifest.obb.angle_unit = "radian";
  manifest.obb.angle_range = "[-pi/4,3pi/4]";
  manifest.obb.score_type = "probabilities";
  manifest.ensure_provenance();
  return manifest;
}

}  // namespace

TEST(RotatedIou, CoversTheRequiredGeometricCases) {
  vie::OrientedBox box{0.f, 0.f, 10.f, 4.f, 0.f};
  EXPECT_NEAR(vie::rotated_intersection_over_union(box, box), 1.f, 1e-5f);
  vie::OrientedBox separated = box;
  separated.cx = 100.f;
  EXPECT_NEAR(vie::rotated_intersection_over_union(box, separated), 0.f, 1e-5f);

  vie::OrientedBox overlap{2.f, 0.f, 10.f, 4.f, 0.f};
  // Axis-aligned overlap of the same rectangles: intersection 8*4, union 40+40-32.
  EXPECT_NEAR(vie::rotated_intersection_over_union(box, overlap), 32.f / 48.f, 1e-4f);

  vie::OrientedBox inner{0.f, 0.f, 2.f, 2.f, 0.f};
  EXPECT_NEAR(vie::rotated_intersection_over_union(box, inner), 4.f / 40.f, 1e-4f);

  vie::OrientedBox crossed{0.f, 0.f, 10.f, 2.f, static_cast<float>(kPi / 2.0)};
  vie::OrientedBox flat{0.f, 0.f, 10.f, 2.f, 0.f};
  EXPECT_NEAR(vie::rotated_intersection_over_union(flat, crossed), 4.f / 36.f, 1e-3f);

  vie::OrientedBox swapped{0.f, 0.f, 4.f, 10.f, static_cast<float>(kPi / 2.0)};
  EXPECT_NEAR(vie::rotated_intersection_over_union(box, swapped), 1.f, 1e-4f);

  vie::OrientedBox tiny{0.f, 0.f, 1e-4f, 1e-4f, 0.f};
  EXPECT_NEAR(vie::rotated_intersection_over_union(tiny, tiny), 1.f, 1e-4f);
  vie::OrientedBox near_zero = box;
  near_zero.angle = 1e-8f;
  EXPECT_NEAR(vie::rotated_intersection_over_union(box, near_zero), 1.f, 1e-3f);

  const float boundary = static_cast<float>(3.0 * kPi / 4.0 - 1e-4);
  vie::OrientedBox edge{1.f, 2.f, 6.f, 2.f, boundary};
  vie::OrientedBox edge_swap{1.f, 2.f, 2.f, 6.f, boundary - static_cast<float>(kPi / 2.0)};
  EXPECT_NEAR(vie::rotated_intersection_over_union(edge, edge_swap), 1.f, 1e-3f);

  vie::OrientedBox bad = box;
  bad.width = -1.f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::rotated_intersection_over_union(box, bad));
  bad = box;
  bad.angle = std::nanf("");
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::rotated_intersection_over_union(box, bad));
}

TEST(RotatedNms, IsClassAwareAndStable) {
  vie::ObbCandidate first{vie::OrientedBox{0, 0, 10, 4, 0}, 0.9f, 1, 5};
  vie::ObbCandidate duplicate{vie::OrientedBox{0, 0, 10, 4, 0}, 0.8f, 1, 1};
  vie::ObbCandidate other_class{vie::OrientedBox{0, 0, 10, 4, 0}, 0.7f, 2, 3};
  vie::ObbCandidate far{vie::OrientedBox{40, 0, 10, 4, 0}, 0.6f, 1, 4};
  std::vector<vie::ObbCandidate> sorted{first, duplicate, other_class, far};
  const std::vector<int> aware = vie::rotated_nms(sorted, 0.5f, true, 10);
  ASSERT_EQ(aware.size(), 3u);
  EXPECT_EQ(aware[0], 0);
  EXPECT_EQ(aware[1], 2);
  EXPECT_EQ(aware[2], 3);

  vie::ObbCandidate low_index{vie::OrientedBox{0, 30, 4, 4, 0}, 0.5f, 0, 2};
  vie::ObbCandidate high_index{vie::OrientedBox{20, 30, 4, 4, 0}, 0.5f, 0, 9};
  std::vector<vie::ObbCandidate> ties{high_index, low_index};
  std::sort(ties.begin(), ties.end(), [](const vie::ObbCandidate& left, const vie::ObbCandidate& right) {
    if (left.score != right.score) return left.score > right.score;
    return left.anchor_index < right.anchor_index;
  });
  const std::vector<int> kept = vie::rotated_nms(ties, 0.5f, true, 1);
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_EQ(ties[static_cast<std::size_t>(kept[0])].anchor_index, 2);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::rotated_nms(sorted, 1.5f, true, 1));
}

TEST(ObbGeometry, StretchLetterboxAndCenterCrop) {
  vie::GeometryTransform stretch;
  stretch.original_width = 100;
  stretch.original_height = 50;
  stretch.output_width = 200;
  stretch.output_height = 200;
  stretch.scale_x = 2;
  stretch.scale_y = 4;
  stretch.uniform_scale = 0;
  stretch.resize_mode = vie::ResizeMode::Stretch;
  vie::OrientedBox mapped;
  ASSERT_TRUE(vie::map_obb_to_original(vie::OrientedBox{40, 40, 10, 8, 0}, stretch, mapped));
  EXPECT_NEAR(mapped.cx, 20.f, 1e-3f);
  EXPECT_NEAR(mapped.cy, 10.f, 1e-3f);
  EXPECT_NEAR(mapped.width, 5.f, 1e-3f);
  EXPECT_NEAR(mapped.height, 2.f, 1e-3f);
  EXPECT_NEAR(mapped.angle, 0.f, 1e-4f);

  const float angle = static_cast<float>(kPi / 4.0);
  const vie::OrientedBox rotated{40, 40, 10, 8, angle};
  ASSERT_TRUE(vie::map_obb_to_original(rotated, stretch, mapped));
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  const double hw = 5;
  const double hh = 4;
  const double corners[4][2] = {{40 + hw * cosine + (-hh) * sine, 40 + hw * sine + hh * cosine},
                                {40 + hw * cosine - (-hh) * sine, 40 + hw * sine - hh * cosine},
                                {40 - hw * cosine - (-hh) * sine, 40 - hw * sine - hh * cosine},
                                {40 - hw * cosine + (-hh) * sine, 40 - hw * sine + hh * cosine}};
  for (const auto& corner : corners) {
    EXPECT_TRUE(point_in_box(corner[0] / 2.0, corner[1] / 4.0, mapped));
  }

  vie::GeometryTransform letterbox;
  letterbox.original_width = 100;
  letterbox.original_height = 50;
  letterbox.output_width = 100;
  letterbox.output_height = 100;
  letterbox.scale_x = 1;
  letterbox.scale_y = 1;
  letterbox.uniform_scale = 1;
  letterbox.pad_top = 25;
  letterbox.resize_mode = vie::ResizeMode::Letterbox;
  ASSERT_TRUE(vie::map_obb_to_original(vie::OrientedBox{50, 50, 4, 2, 0.4f}, letterbox, mapped));
  EXPECT_NEAR(mapped.cx, 50.f, 1e-4f);
  EXPECT_NEAR(mapped.cy, 25.f, 1e-4f);
  EXPECT_NEAR(mapped.width, 4.f, 1e-4f);
  EXPECT_NEAR(mapped.height, 2.f, 1e-4f);
  EXPECT_NEAR(mapped.angle, 0.4f, 1e-5f);

  vie::GeometryTransform crop;
  crop.original_width = 80;
  crop.original_height = 160;
  crop.output_width = 160;
  crop.output_height = 160;
  crop.scale_x = 2;
  crop.scale_y = 2;
  crop.uniform_scale = 2;
  crop.crop_y = 80;
  crop.resize_mode = vie::ResizeMode::ShortestSide;
  ASSERT_TRUE(vie::map_obb_to_original(vie::OrientedBox{80, 80, 10, 10, 0}, crop, mapped));
  EXPECT_NEAR(mapped.cx, 40.f, 1e-4f);
  EXPECT_NEAR(mapped.cy, 80.f, 1e-4f);
  EXPECT_NEAR(mapped.width, 5.f, 1e-4f);
  EXPECT_NEAR(mapped.height, 5.f, 1e-4f);
  EXPECT_FALSE(vie::map_obb_to_original(vie::OrientedBox{80, -120, 10, 10, 0}, crop, mapped));
  EXPECT_FALSE(vie::map_obb_to_original(vie::OrientedBox{80, -70, 10, 30, 0}, crop, mapped));
}

TEST(YoloV8ObbDecoder, FiltersNmsAndRejectsIllegalTensors) {
  // channels: cx cy w h s0 s1 angle. Two identical boxes and one distant box.
  std::vector<float> values(7 * 3, 0.f);
  auto write = [&](int prediction, int channel, float value) { values[static_cast<std::size_t>(channel * 3 + prediction)] = value; };
  write(0, 0, 10);
  write(0, 1, 10);
  write(0, 2, 4);
  write(0, 3, 2);
  write(0, 4, 0.9f);
  write(0, 5, 0.1f);
  write(0, 6, 0.2f);
  write(1, 0, 10);
  write(1, 1, 10);
  write(1, 2, 4);
  write(1, 3, 2);
  write(1, 4, 0.8f);
  write(1, 5, 0.1f);
  write(1, 6, 0.2f);
  write(2, 0, 40);
  write(2, 1, 40);
  write(2, 2, 4);
  write(2, 3, 2);
  write(2, 4, 0.2f);
  write(2, 5, 0.7f);
  write(2, 6, -0.4f);
  vie::Tensor tensor = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 7, 3}, values.data(),
                                              values.size() * sizeof(float));
  vie::DecoderInput input;
  input.outputs.emplace("output0", tensor);
  input.geometry = identity_geometry();
  const vie::ObbBatch batch = vie::YoloV8ObbDecoder{}.decode_obb(input, keep_config(), vie::DeviceContext{});
  ASSERT_EQ(batch.results.size(), 1u);
  ASSERT_EQ(batch.results[0].objects.size(), 2u);
  EXPECT_EQ(batch.results[0].objects[0].class_id, 0);
  EXPECT_EQ(batch.results[0].objects[0].anchor_index, 0);
  EXPECT_NEAR(batch.results[0].objects[0].score, 0.9f, 1e-6f);
  EXPECT_EQ(batch.results[0].objects[1].class_id, 1);
  EXPECT_EQ(batch.results[0].objects[1].anchor_index, 2);
  EXPECT_NEAR(batch.results[0].objects[1].box.angle, -0.4f, 1e-6f);
  EXPECT_EQ(batch.results[0].protocol, "yolov8_obb_raw");
  EXPECT_EQ(batch.results[0].execution_device, vie::DeviceType::Cpu);

  values[6] = std::nanf("");
  vie::Tensor bad = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 7, 3}, values.data(),
                                           values.size() * sizeof(float));
  vie::DecoderInput bad_input;
  bad_input.outputs.emplace("output0", bad);
  bad_input.geometry = identity_geometry();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::YoloV8ObbDecoder{}.decode_obb(bad_input, keep_config(), vie::DeviceContext{}));

  values[6] = -2.f;
  vie::Tensor negative = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 7, 3}, values.data(),
                                                values.size() * sizeof(float));
  vie::DecoderInput negative_input;
  negative_input.outputs.emplace("output0", negative);
  negative_input.geometry = identity_geometry();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::YoloV8ObbDecoder{}.decode_obb(negative_input, keep_config(), vie::DeviceContext{}));
}

TEST(YoloV8ObbRecognizer, AcceptsTheProvenHeadAndRejectsLookalikes) {
  const vie::ModelAnalysisReport report = analyze_graph(obb_graph(0.25f, false));
  bool saw_quarter = false;
  for (const vie::FloatConstant& constant : report.float_constants) {
    if (constant.name == "quarter") {
      saw_quarter = true;
      EXPECT_NEAR(constant.value, 0.25, 1e-6);
    }
  }
  EXPECT_TRUE(saw_quarter);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_obb");
  EXPECT_EQ(result.selected()->decoder.protocol, "yolov8_obb_raw");
  EXPECT_EQ(*result.selected()->decoder.num_classes, 2);
  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, "obb.onnx");
  EXPECT_EQ(manifest.task, vie::TaskType::Obb);
  EXPECT_EQ(manifest.obb.angle_unit, "radian");
  EXPECT_EQ(manifest.obb.angle_range, "[-pi/4,3pi/4]");
  EXPECT_EQ(manifest.obb.score_type, "probabilities");
  EXPECT_EQ(manifest.decoder.box_format, "cxcywh_angle");
  EXPECT_TRUE(manifest.input.color.empty());
  EXPECT_TRUE(manifest.input.resize.empty());
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_FALSE(validation.runnable());
  const std::string json = manifest.to_json_string();
  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(json);
  EXPECT_EQ(loaded.obb.output_name, "output0");
  EXPECT_EQ(loaded.task, vie::TaskType::Obb);

  const vie::RecognitionResult wrong =
      vie::default_recognizer_registry().recognize(analyze_graph(obb_graph(0.5f, false)));
  EXPECT_EQ(wrong.status, vie::RecognitionStatus::Unknown);
  const vie::RecognitionResult multi =
      vie::default_recognizer_registry().recognize(analyze_graph(obb_graph(0.25f, true)));
  EXPECT_EQ(multi.status, vie::RecognitionStatus::Unknown);

  const std::string detection = vie_fixture("testdata/onnx/yolov8n.onnx");
  std::ifstream input(detection, std::ios::binary);
  ASSERT_TRUE(input);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const vie::ModelAnalysisReport real = vie::OnnxAnalyzer{}.analyze_bytes(buffer.str(), "yolov8n.onnx", ".");
  EXPECT_FALSE(vie::YoloV8ObbRecognizer{}.consider(real).has_value());
  const vie::RecognitionResult kept = vie::default_recognizer_registry().recognize(real);
  EXPECT_EQ(kept.status, vie::RecognitionStatus::Recognized);
  ASSERT_NE(kept.selected(), nullptr);
  EXPECT_EQ(kept.selected()->architecture, "yolov8_detection");
}

TEST(ObbApi, RejectsTheWrongTask) {
  vie::InferenceEngine engine;
  vie::ModelConfig config;
  config.manifest = runnable_obb_manifest();
  config.backend = vie::BackendKind::Mock;
  engine.load(config);
  vie::ImageInput image;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image));
  vie::ModelManifest detection = runnable_obb_manifest();
  detection.task = vie::TaskType::Detection;
  detection.decoder.protocol = "mock";
  detection.decoder.box_format = "xyxy";
  detection.obb = {};
  for (const char* path : {"obb.output_name", "obb.channel_axis", "obb.angle_unit", "obb.angle_range", "obb.score_type"}) {
    detection.provenance.erase(path);
  }
  detection.ensure_provenance();
  vie::InferenceEngine detector;
  vie::ModelConfig detection_config;
  detection_config.manifest = detection;
  detection_config.backend = vie::BackendKind::Mock;
  detector.load(detection_config);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, detector.infer_obb(image, {}));
}

TEST(YoloV8ObbRecognizer, RealExportWhenPresent) {
  std::ifstream input(kRealModel, std::ios::binary);
  if (!input) {
    GTEST_SKIP() << "real yolov8n-obb onnx is not in /tmp/vie_phase71";
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_bytes(buffer.str(), "yolov8n-obb.onnx", ".");
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_obb");
  EXPECT_EQ(result.selected()->decoder.protocol, "yolov8_obb_raw");
  EXPECT_EQ(result.selected()->decoder.num_classes, 15);
  EXPECT_EQ(result.selected()->decoder.channel_axis, 1);
}

#if defined(VIE_HAS_ONNXRUNTIME)
namespace {

struct ReferenceBox {
  float cx = 0.f;
  float cy = 0.f;
  float width = 0.f;
  float height = 0.f;
  float angle = 0.f;
  int class_id = -1;
  float score = 0.f;
  int anchor = -1;
};

const std::filesystem::path kPhaseDir{"/tmp/vie_phase71"};

bool reference_ready() {
  return std::filesystem::is_regular_file(kRealModel) &&
         std::filesystem::is_regular_file(kPhaseDir / "raw_output.f32") &&
         std::filesystem::is_regular_file(kPhaseDir / "reference_boxes.txt") &&
         std::filesystem::is_regular_file(kPhaseDir / "input_u8.bin");
}

vie::ImageInput pattern_image(std::vector<std::uint8_t>& storage) {
  storage.resize(1024 * 1024 * 3);
  std::ifstream input(kPhaseDir / "input_u8.bin", std::ios::binary);
  input.read(reinterpret_cast<char*>(storage.data()), static_cast<std::streamsize>(storage.size()));
  EXPECT_TRUE(static_cast<bool>(input));
  vie::ImageInput image;
  image.data = storage.data();
  image.width = 1024;
  image.height = 1024;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

void supply_test_preprocess(vie::ModelManifest& manifest) {
  auto user = [&](const char* field) {
    manifest.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {"test supplied preprocess"}};
  };
  manifest.input.color = "rgb";
  user("input.color");
  manifest.input.normalization = "scale_0_1";
  user("input.normalization");
  manifest.input.resize = "stretch";
  user("input.resize");
  manifest.input.interpolation = "nearest";
  user("input.interpolation");
}

vie::ModelManifest recognized_manifest() {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(kRealModel);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized);
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, kRealModel);
  supply_test_preprocess(manifest);
  return manifest;
}

vie::ObbRuntimeConfig reference_runtime() {
  vie::ObbRuntimeConfig runtime;
  runtime.class_selection = vie::ClassSelection::BestClass;
  runtime.confidence_threshold = 0.f;
  runtime.nms_enabled = true;
  runtime.class_aware = true;
  runtime.iou_threshold = 0.45f;
  runtime.pre_nms_top_k = 300;
  runtime.max_detections = 20;
  return runtime;
}

std::vector<ReferenceBox> load_reference() {
  std::ifstream input(kPhaseDir / "reference_boxes.txt");
  std::string header;
  std::getline(input, header);
  EXPECT_NE(header.find("confidence 0"), std::string::npos);
  EXPECT_NE(header.find("iou 0.45"), std::string::npos);
  EXPECT_NE(header.find("pre_nms 300"), std::string::npos);
  EXPECT_NE(header.find("max_detections 20"), std::string::npos);
  std::vector<ReferenceBox> boxes;
  ReferenceBox box;
  while (input >> box.cx >> box.cy >> box.width >> box.height >> box.angle >> box.class_id >> box.score >>
         box.anchor) {
    boxes.push_back(box);
  }
  return boxes;
}

void expect_matches_reference(const vie::ObbBatch& batch, const std::vector<ReferenceBox>& expected) {
  ASSERT_EQ(batch.results.size(), 1u);
  const vie::ObbResult& result = batch.results[0];
  EXPECT_EQ(result.image_width, 1024);
  EXPECT_EQ(result.image_height, 1024);
  EXPECT_EQ(result.num_classes, 15);
  EXPECT_EQ(result.execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(result.protocol, "yolov8_obb_raw");
  ASSERT_EQ(result.objects.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const vie::ObbObject& got = result.objects[index];
    const ReferenceBox& want = expected[index];
    EXPECT_EQ(got.class_id, want.class_id) << index;
    EXPECT_EQ(got.anchor_index, want.anchor) << index;
    EXPECT_NEAR(got.score, want.score, 1e-4f) << index;
    EXPECT_NEAR(got.box.cx, want.cx, 1e-2f) << index;
    EXPECT_NEAR(got.box.cy, want.cy, 1e-2f) << index;
    EXPECT_NEAR(got.box.width, want.width, 1e-2f) << index;
    EXPECT_NEAR(got.box.height, want.height, 1e-2f) << index;
    EXPECT_NEAR(got.box.angle, want.angle, 1e-3f) << index;
  }
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(std::floor(fraction * static_cast<double>(values.size() - 1)));
  return values[index];
}

double mean_of(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

}  // namespace

TEST(YoloV8ObbReal, CpuMatchesIndependentReference) {
  if (!reference_ready()) {
    GTEST_SKIP() << "run tools/export_yolov8n_obb.py";
  }
  vie::ModelManifest manifest = recognized_manifest();
  vie::ModelConfig config;
  config.model_path = kRealModel;
  config.manifest = manifest;
  config.backend = vie::BackendKind::OnnxRuntime;
  config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  engine.load(config);
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = pattern_image(pixels);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image));
  vie::ObbInferTiming timing;
  const vie::ObbBatch batch = engine.infer_obb(image, reference_runtime(), &timing);
  vie::BackendRequest request;
  request.inputs.emplace(engine.manifest().input.name, engine.preprocess(image).output);
  const vie::BackendOutput output = engine.execute(request);
  const vie::Tensor& tensor = output.tensors.at("output0").tensor;
  EXPECT_EQ(tensor.shape(), (std::vector<std::int64_t>{1, 20, 21504}));
  const float* data = tensor.data_as<float>();
  std::ifstream reference(kPhaseDir / "raw_output.f32", std::ios::binary);
  std::vector<float> pytorch(tensor.numel());
  reference.read(reinterpret_cast<char*>(pytorch.data()), static_cast<std::streamsize>(pytorch.size() * sizeof(float)));
  ASSERT_TRUE(static_cast<bool>(reference));
  float max_abs = 0.f;
  for (std::size_t index = 0; index < pytorch.size(); ++index) {
    max_abs = std::max(max_abs, std::fabs(data[index] - pytorch[index]));
  }
  EXPECT_LT(max_abs, 1e-3f);
  expect_matches_reference(batch, load_reference());
  EXPECT_EQ(timing.d2h_bytes, 0u);
  EXPECT_EQ(timing.output_bytes, static_cast<std::size_t>(20 * 21504 * 4));
  EXPECT_FALSE(timing.includes_d2h);
  EXPECT_GT(batch.results[0].objects.size(), 0u);
}

TEST(YoloV8ObbReal, CpuTimingUsesRepeatedSamples) {
  if (!reference_ready()) {
    GTEST_SKIP() << "run tools/export_yolov8n_obb.py";
  }
  vie::ModelConfig config;
  config.model_path = kRealModel;
  config.manifest = recognized_manifest();
  config.backend = vie::BackendKind::OnnxRuntime;
  config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  engine.load(config);
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = pattern_image(pixels);
  const vie::ObbRuntimeConfig runtime = reference_runtime();
  constexpr int kWarmup = 3;
  constexpr int kSamples = 10;
  for (int index = 0; index < kWarmup; ++index) engine.infer_obb(image, runtime, nullptr);
  std::vector<double> preprocess_ms;
  std::vector<double> execute_ms;
  std::vector<double> validate_ms;
  std::vector<double> nms_ms;
  std::vector<double> total_ms;
  int candidates = 0;
  int kept = 0;
  std::size_t output_bytes = 0;
  for (int index = 0; index < kSamples; ++index) {
    vie::ObbInferTiming timing;
    engine.infer_obb(image, runtime, &timing);
    preprocess_ms.push_back(timing.preprocess_ms);
    execute_ms.push_back(timing.execute_ms);
    validate_ms.push_back(timing.validate_ms);
    nms_ms.push_back(timing.nms_ms);
    total_ms.push_back(timing.total_ms);
    candidates = timing.candidates_before_nms;
    kept = timing.results_after_nms;
    output_bytes = timing.output_bytes;
  }
  std::ofstream record("/tmp/vie_phase71_cpu.txt");
  record << "device cpu warmup " << kWarmup << " samples " << kSamples << " candidates " << candidates
         << " kept " << kept << " output_bytes " << output_bytes << " d2h_bytes 0\n";
  record << "preprocess mean " << mean_of(preprocess_ms) << " p50 " << percentile(preprocess_ms, 0.50) << " p95 "
         << percentile(preprocess_ms, 0.95) << "\n";
  record << "execute mean " << mean_of(execute_ms) << " p50 " << percentile(execute_ms, 0.50) << " p95 "
         << percentile(execute_ms, 0.95) << "\n";
  record << "validate mean " << mean_of(validate_ms) << " p50 " << percentile(validate_ms, 0.50) << " p95 "
         << percentile(validate_ms, 0.95) << "\n";
  record << "nms mean " << mean_of(nms_ms) << " p50 " << percentile(nms_ms, 0.50) << " p95 "
         << percentile(nms_ms, 0.95) << "\n";
  record << "total mean " << mean_of(total_ms) << " p50 " << percentile(total_ms, 0.50) << " p95 "
         << percentile(total_ms, 0.95) << "\n";
  EXPECT_GT(mean_of(execute_ms), 0.0);
  EXPECT_GT(mean_of(nms_ms), 0.0);
  EXPECT_GE(mean_of(total_ms), mean_of(execute_ms));
}

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
TEST(YoloV8ObbReal, StrictCudaMatchesCpu) {
  if (!reference_ready()) {
    GTEST_SKIP() << "run tools/export_yolov8n_obb.py";
  }
  vie::ModelManifest manifest = recognized_manifest();
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = pattern_image(pixels);
  void* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(&device_pixels, pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput gpu_image = image;
  gpu_image.data = device_pixels;
  gpu_image.device = vie::DeviceType::Cuda;
  const vie::ObbRuntimeConfig runtime = reference_runtime();

  vie::ModelConfig cpu_config;
  cpu_config.model_path = kRealModel;
  cpu_config.manifest = manifest;
  cpu_config.backend = vie::BackendKind::OnnxRuntime;
  cpu_config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine cpu_engine;
  cpu_engine.load(cpu_config);
  const vie::ObbBatch cpu_batch = cpu_engine.infer_obb(image, runtime, nullptr);

  vie::ModelConfig cuda_config = cpu_config;
  cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
  cuda_config.device = vie::DeviceType::Cuda;
  vie::InferenceEngine cuda_engine;
  cuda_engine.load(cuda_config);
  constexpr int kWarmup = 3;
  constexpr int kSamples = 10;
  for (int index = 0; index < kWarmup; ++index) cuda_engine.infer_obb(gpu_image, runtime, nullptr);
  std::vector<double> preprocess_ms;
  std::vector<double> execute_ms;
  std::vector<double> d2h_ms;
  std::vector<double> validate_ms;
  std::vector<double> nms_ms;
  std::vector<double> total_ms;
  vie::ObbBatch cuda_batch;
  vie::ObbInferTiming timing;
  for (int index = 0; index < kSamples; ++index) {
    cuda_batch = cuda_engine.infer_obb(gpu_image, runtime, &timing);
    preprocess_ms.push_back(timing.preprocess_ms);
    execute_ms.push_back(timing.execute_ms);
    d2h_ms.push_back(timing.d2h_ms);
    validate_ms.push_back(timing.validate_ms);
    nms_ms.push_back(timing.nms_ms);
    total_ms.push_back(timing.total_ms);
  }
  EXPECT_EQ(timing.d2h_bytes, static_cast<std::size_t>(20 * 21504 * 4));
  EXPECT_TRUE(timing.includes_d2h);
  EXPECT_EQ(cuda_batch.results[0].execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(cuda_batch.results[0].d2h_bytes, timing.d2h_bytes);

  vie::BackendRequest cpu_request;
  cpu_request.inputs.emplace(cpu_engine.manifest().input.name, cpu_engine.preprocess(image).output);
  const vie::BackendOutput cpu_output = cpu_engine.execute(cpu_request);
  const vie::Tensor& cpu_tensor = cpu_output.tensors.at("output0").tensor;
  vie::BackendRequest cuda_request;
  cuda_request.inputs.emplace(cuda_engine.manifest().input.name, cuda_engine.preprocess(gpu_image).output);
  const vie::BackendOutput cuda_output = cuda_engine.execute(cuda_request);
  const vie::Tensor cuda_tensor =
      cuda_output.tensors.at("output0").tensor.to(vie::DeviceContext(vie::DeviceType::Cpu, 0));
  ASSERT_EQ(cpu_tensor.shape(), cuda_tensor.shape());
  const float* cpu_data = cpu_tensor.data_as<float>();
  const float* cuda_data = cuda_tensor.data_as<float>();
  float max_abs = 0.f;
  float max_box = 0.f;
  float max_score = 0.f;
  float max_angle = 0.f;
  constexpr int kChannels = 20;
  constexpr int kAnchors = 21504;
  ASSERT_EQ(cpu_tensor.numel(), static_cast<std::size_t>(kChannels * kAnchors));
  for (int channel = 0; channel < kChannels; ++channel) {
    for (int anchor = 0; anchor < kAnchors; ++anchor) {
      const float delta = std::fabs(cpu_data[channel * kAnchors + anchor] - cuda_data[channel * kAnchors + anchor]);
      max_abs = std::max(max_abs, delta);
      if (channel < 4) max_box = std::max(max_box, delta);
      else if (channel < 19) max_score = std::max(max_score, delta);
      else max_angle = std::max(max_angle, delta);
    }
  }
  // Measured on this 1024 pattern: box channels differ by at most 0.635 px,
  // class probabilities stay under 1e-4, and angle stays under 0.0028 rad.
  EXPECT_LT(max_box, 0.75f);
  EXPECT_LT(max_score, 1e-4f);
  EXPECT_LT(max_angle, 5e-3f);

  // Scores on this pattern peak near 3e-4, so Execution Provider noise can change
  // which near-tied anchors survive Top-K and rotated NMS. Compare shared anchors.
  int shared = 0;
  for (const vie::ObbObject& cpu : cpu_batch.results[0].objects) {
    const vie::ObbObject* cuda = nullptr;
    for (const vie::ObbObject& candidate : cuda_batch.results[0].objects) {
      if (candidate.anchor_index == cpu.anchor_index) cuda = &candidate;
    }
    if (cuda == nullptr) continue;
    ++shared;
    EXPECT_EQ(cuda->class_id, cpu.class_id) << cpu.anchor_index;
    EXPECT_NEAR(cuda->score, cpu.score, 1e-4f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->box.cx, cpu.box.cx, 0.05f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->box.cy, cpu.box.cy, 0.05f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->box.width, cpu.box.width, 0.05f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->box.height, cpu.box.height, 0.05f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->box.angle, cpu.box.angle, 5e-3f) << cpu.anchor_index;
  }
  EXPECT_EQ(shared, static_cast<int>(cpu_batch.results[0].objects.size()));
  EXPECT_EQ(cuda_batch.results[0].objects.size(), cpu_batch.results[0].objects.size());
  std::ofstream record("/tmp/vie_phase71_cuda.txt");
  record << "device cuda warmup " << kWarmup << " samples " << kSamples << " candidates "
         << timing.candidates_before_nms << " kept " << timing.results_after_nms << " cpu_kept "
         << cpu_batch.results[0].objects.size() << " shared_anchors " << shared << " raw_max_abs " << max_abs
         << " raw_max_box " << max_box << " raw_max_score " << max_score << " raw_max_angle " << max_angle
         << " output_bytes " << timing.output_bytes << " d2h_bytes " << timing.d2h_bytes << "\n";
  record << "preprocess mean " << mean_of(preprocess_ms) << " p50 " << percentile(preprocess_ms, 0.50) << " p95 "
         << percentile(preprocess_ms, 0.95) << "\n";
  record << "execute mean " << mean_of(execute_ms) << " p50 " << percentile(execute_ms, 0.50) << " p95 "
         << percentile(execute_ms, 0.95) << "\n";
  record << "d2h mean " << mean_of(d2h_ms) << " p50 " << percentile(d2h_ms, 0.50) << " p95 "
         << percentile(d2h_ms, 0.95) << "\n";
  record << "validate mean " << mean_of(validate_ms) << " p50 " << percentile(validate_ms, 0.50) << " p95 "
         << percentile(validate_ms, 0.95) << "\n";
  record << "nms mean " << mean_of(nms_ms) << " p50 " << percentile(nms_ms, 0.50) << " p95 "
         << percentile(nms_ms, 0.95) << "\n";
  record << "total mean " << mean_of(total_ms) << " p50 " << percentile(total_ms, 0.50) << " p95 "
         << percentile(total_ms, 0.95) << "\n";
  cudaFree(device_pixels);
}
#endif
#endif
