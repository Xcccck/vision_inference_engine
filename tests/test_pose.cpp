#include "onnx_test_writer.hpp"
#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/decoder/yolov8_pose_decoder.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/recognizer/yolov8_detect_recognizer.hpp"
#include "vie/recognizer/yolov8_pose_recognizer.hpp"

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
#include "vie/backend/onnxruntime_cuda_backend.hpp"
#include <cuda_runtime.h>
#endif

namespace {

const char* kRealModel = "/tmp/vie_phase72/yolov8n-pose.onnx";

std::string f32_bytes(float value) {
  std::string raw(4, '\0');
  std::memcpy(raw.data(), &value, 4);
  return raw;
}

std::string f32_bytes(const std::vector<float>& values) {
  std::string raw(values.size() * 4, '\0');
  if (!values.empty()) std::memcpy(raw.data(), values.data(), raw.size());
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

OnnxTestWriter pose_graph(bool with_mul, bool extra_output, bool with_nms, int channel_count) {
  OnnxTestWriter graph;
  graph.message(5, tensor_message("two", 1, {1}, {}, f32_bytes(2.f), false));
  graph.message(5, tensor_message("stride", 1, {1, 1, 4}, {}, f32_bytes({1.f, 1.f, 1.f, 1.f}), false));
  graph.message(5, tensor_message("anchor", 1, {1, 2, 1}, {}, f32_bytes({1.f, 2.f}), false));
  for (int index = 0; index < 3; ++index) {
    const std::string id = std::to_string(index);
    add_conv(graph, "cls_c" + id, {2, 1, 1, 1});
    graph.message(1, node_message("Reshape", {"cls_c" + id}, {"cls_r" + id}));
    add_conv(graph, "box_c" + id, {64, 1, 1, 1});
    graph.message(1, node_message("Reshape", {"box_c" + id}, {"box_r" + id}));
    add_conv(graph, "kpt_c" + id, {6, 1, 1, 1});
    graph.message(1, node_message("Reshape", {"kpt_c" + id}, {"kpt_r" + id}));
  }
  graph.message(1, node_message("Concat", {"cls_r0", "cls_r1", "cls_r2"}, {"cls_cat"}));
  graph.message(1, node_message("Sigmoid", {"cls_cat"}, {"cls_sig"}));
  graph.message(1, node_message("Concat", {"box_r0", "box_r1", "box_r2"}, {"box_cat"}));
  graph.message(1, node_message("Softmax", {"box_cat"}, {"box_sm"}));
  add_weight(graph, "dfl_w", {1, 16, 1, 1});
  graph.message(1, node_message("Conv", {"box_sm", "dfl_w"}, {"dfl"}));
  graph.message(1, node_message("Sub", {"anchor", "dfl"}, {"x1"}));
  graph.message(1, node_message("Add", {"anchor", "dfl"}, {"x2"}));
  graph.message(1, node_message("Add", {"x1", "x2"}, {"sum"}));
  graph.message(1, node_message("Div", {"sum", "two"}, {"cxy"}));
  graph.message(1, node_message("Sub", {"x2", "x1"}, {"wh"}));
  graph.message(1, node_message("Concat", {"cxy", "wh"}, {"decoded"}));
  graph.message(1, node_message("Mul", {"decoded", "stride"}, {"box"}));
  graph.message(1, node_message("Concat", {"kpt_r0", "kpt_r1", "kpt_r2"}, {"kpt_raw"}));
  graph.message(1, node_message("Reshape", {"kpt_raw"}, {"kpt_view"}));
  graph.message(1, node_message("Slice", {"kpt_view"}, {"xy"}));
  graph.message(1, node_message("Slice", {"kpt_view"}, {"vis"}));
  if (with_mul) graph.message(1, node_message("Mul", {"xy", "two"}, {"xy2"}));
  else graph.message(1, node_message("Identity", {"xy"}, {"xy2"}));
  graph.message(1, node_message("Sigmoid", {"vis"}, {"vis_sig"}));
  graph.message(1, node_message("Concat", {"xy2", "vis_sig"}, {"kpt_pair"}));
  graph.message(1, node_message("Reshape", {"kpt_pair"}, {"kpt_out"}));
  graph.message(1, node_message("Concat", {"box", "cls_sig", "kpt_out"}, {"output0"}));
  if (with_nms) graph.message(1, node_message("NonMaxSuppression", {"box"}, {"nms"}));
  graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(32),
                                                               static_dim(32)})));
  graph.message(12, value_info("output0", tensor_type(1, true, {static_dim(1), static_dim(channel_count),
                                                                static_dim(8)})));
  if (extra_output) {
    graph.message(12, value_info("extra", tensor_type(1, true, {static_dim(1), static_dim(channel_count),
                                                                static_dim(8)})));
  }
  graph.text(2, "pose");
  return graph;
}

vie::ModelAnalysisReport analyze_graph(const OnnxTestWriter& graph) {
  return vie::OnnxAnalyzer{}.analyze_bytes(wrap_model(graph).str(), "pose.onnx", ".");
}

vie::GeometryTransform identity_geometry(int side = 64) {
  vie::GeometryTransform geometry;
  geometry.original_width = side;
  geometry.original_height = side;
  geometry.output_width = side;
  geometry.output_height = side;
  geometry.resized_width = side;
  geometry.resized_height = side;
  geometry.scale_x = 1;
  geometry.scale_y = 1;
  geometry.uniform_scale = 1;
  geometry.resize_mode = vie::ResizeMode::Stretch;
  return geometry;
}

vie::PoseDecodeConfig keep_config() {
  vie::PoseDecodeConfig config;
  config.output_name = "output0";
  config.channel_axis = 1;
  config.num_classes = 2;
  config.keypoint_count = 2;
  config.keypoint_dimensions = 3;
  config.class_selection = vie::ClassSelection::BestClass;
  config.confidence_threshold = 0.5f;
  config.nms_enabled = true;
  config.class_aware = true;
  config.iou_threshold = 0.5f;
  config.max_detections = 10;
  config.pre_nms_top_k = 10;
  return config;
}

vie::Tensor pose_tensor(const std::vector<float>& channel_major, int channels, int predictions) {
  return vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32,
                                {1, static_cast<std::int64_t>(channels), static_cast<std::int64_t>(predictions)},
                                channel_major.data(), channel_major.size() * sizeof(float));
}

void write_channel(std::vector<float>& data, int predictions, int channel, int prediction, float value) {
  data[static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) + static_cast<std::size_t>(prediction)] =
      value;
}

vie::DecoderInput decode_input(const vie::Tensor& tensor, const vie::GeometryTransform& geometry) {
  vie::DecoderInput input;
  input.geometry = geometry;
  input.outputs.emplace("output0", tensor);
  return input;
}

vie::ModelManifest runnable_pose_manifest() {
  vie::ModelManifest manifest;
  manifest.model_name = "pose";
  manifest.task = vie::TaskType::Pose;
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
  output.shape = {1, 12, 4};
  output.dtype = vie::DataType::Fp32;
  output.semantic = "poses";
  manifest.outputs.push_back(output);
  manifest.decoder.protocol = "yolov8_pose_raw";
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.scores_are_probabilities = true;
  manifest.pose.output_name = "output0";
  manifest.pose.channel_axis = 1;
  manifest.pose.keypoint_count = 2;
  manifest.pose.keypoint_dimensions = 3;
  manifest.pose.keypoint_layout = "xy_confidence";
  manifest.pose.score_type = "probabilities";
  manifest.ensure_provenance();
  return manifest;
}

vie::PreprocessConfig basic_config(vie::ResizeMode mode, int width, int height) {
  vie::PreprocessConfig config;
  config.resize_mode = mode;
  config.target_width = width;
  config.target_height = height;
  config.interpolation = vie::Interpolation::Nearest;
  config.input_color = vie::ColorOrder::Rgb;
  config.output_color = vie::ColorOrder::Rgb;
  config.input_layout = vie::ImageLayout::Hwc;
  config.output_layout = vie::ImageLayout::Chw;
  config.output_dtype = vie::DataType::Fp32;
  return config;
}

void expect_roundtrip(const vie::GeometryTransform& geometry, float x, float y) {
  const float model_x = static_cast<float>(static_cast<double>(x) * geometry.scale_x +
                                           static_cast<double>(vie::geometry_model_origin_x(geometry)));
  const float model_y = static_cast<float>(static_cast<double>(y) * geometry.scale_y +
                                           static_cast<double>(vie::geometry_model_origin_y(geometry)));
  float back_x = 0.f;
  float back_y = 0.f;
  bool inside = false;
  vie::map_keypoint_to_original(model_x, model_y, geometry, back_x, back_y, inside);
  EXPECT_NEAR(back_x, x, 1e-4f);
  EXPECT_NEAR(back_y, y, 1e-4f);
  const bool expected = x >= -1e-3f && x <= static_cast<float>(geometry.original_width) + 1e-3f && y >= -1e-3f &&
                        y <= static_cast<float>(geometry.original_height) + 1e-3f;
  EXPECT_EQ(inside, expected);
}

}  // namespace

TEST(PoseGeometry, UsesEachAxisAndKeepsOutsidePoints) {
  vie::GeometryTransform stretch;
  stretch.original_width = 80;
  stretch.original_height = 40;
  stretch.scale_x = 2;
  stretch.scale_y = 4;
  stretch.uniform_scale = 0;
  stretch.resize_mode = vie::ResizeMode::Stretch;
  float x = 0.f;
  float y = 0.f;
  bool inside = true;
  vie::map_keypoint_to_original(10.f, 20.f, stretch, x, y, inside);
  EXPECT_NEAR(x, 5.f, 1e-5f);
  EXPECT_NEAR(y, 5.f, 1e-5f);
  EXPECT_TRUE(inside);
  const float mean = static_cast<float>((stretch.scale_x + stretch.scale_y) * 0.5);
  EXPECT_GT(std::fabs(10.f / mean - x), 0.5f);

  vie::GeometryTransform letterbox;
  letterbox.original_width = 40;
  letterbox.original_height = 30;
  letterbox.scale_x = 2;
  letterbox.scale_y = 2;
  letterbox.pad_left = 10;
  letterbox.pad_top = 4;
  letterbox.resize_mode = vie::ResizeMode::Letterbox;
  vie::map_keypoint_to_original(14.f, 8.f, letterbox, x, y, inside);
  EXPECT_NEAR(x, 2.f, 1e-5f);
  EXPECT_NEAR(y, 2.f, 1e-5f);

  vie::GeometryTransform crop;
  crop.original_width = 100;
  crop.original_height = 80;
  crop.scale_x = 2;
  crop.scale_y = 2;
  crop.crop_x = 8;
  crop.crop_y = 6;
  crop.resize_mode = vie::ResizeMode::ShortestSide;
  vie::map_keypoint_to_original(10.f, 6.f, crop, x, y, inside);
  EXPECT_NEAR(x, 9.f, 1e-5f);
  EXPECT_NEAR(y, 6.f, 1e-5f);
  EXPECT_TRUE(inside);
  vie::map_keypoint_to_original(-4.f, 6.f, crop, x, y, inside);
  EXPECT_NEAR(x, 2.f, 1e-5f);
  EXPECT_TRUE(inside);
  EXPECT_LT(x, static_cast<float>(crop.crop_x));
  vie::map_keypoint_to_original(-100.f, 6.f, crop, x, y, inside);
  EXPECT_NEAR(x, -46.f, 1e-4f);
  EXPECT_FALSE(inside);

  expect_roundtrip(vie::make_geometry(basic_config(vie::ResizeMode::Stretch, 40, 40), 100, 50), 10.f, 20.f);
  vie::PreprocessConfig letter = basic_config(vie::ResizeMode::Letterbox, 64, 64);
  const vie::GeometryTransform made_letter = vie::make_geometry(letter, 30, 20);
  EXPECT_NE(made_letter.scale_x, made_letter.scale_y);
  expect_roundtrip(made_letter, 12.f, 8.f);
  vie::PreprocessConfig shortest = basic_config(vie::ResizeMode::ShortestSide, 16, 16);
  shortest.short_side = 20;
  shortest.crop_width = 16;
  shortest.crop_height = 16;
  const vie::GeometryTransform made_crop = vie::make_geometry(shortest, 100, 40);
  EXPECT_EQ(made_crop.crop_x, 17);
  EXPECT_EQ(made_crop.crop_y, 2);
  expect_roundtrip(made_crop, 10.f, 8.f);
  const float outside_model_x = static_cast<float>(10.0 * made_crop.scale_x - made_crop.crop_x);
  EXPECT_LT(outside_model_x, 0.f);
  vie::map_keypoint_to_original(outside_model_x, 8.f, made_crop, x, y, inside);
  EXPECT_NEAR(x, 10.f, 1e-3f);
  EXPECT_TRUE(inside);
}

TEST(YoloV8PoseDecoder, FiltersNmsAndRejectsIllegalTensors) {
  constexpr int kChannels = 12;
  constexpr int kPredictions = 5;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  auto put = [&](int channel, int prediction, float value) {
    write_channel(values, kPredictions, channel, prediction, value);
  };
  auto box = [&](int prediction, float cx, float cy, float width, float height) {
    put(0, prediction, cx);
    put(1, prediction, cy);
    put(2, prediction, width);
    put(3, prediction, height);
  };
  auto points = [&](int prediction, float x0, float y0, float c0, float x1, float y1, float c1) {
    put(6, prediction, x0);
    put(7, prediction, y0);
    put(8, prediction, c0);
    put(9, prediction, x1);
    put(10, prediction, y1);
    put(11, prediction, c1);
  };
  box(0, 10.f, 10.f, 10.f, 10.f);
  put(4, 0, 0.95f);
  put(5, 0, 0.1f);
  points(0, 5.f, 6.f, 0.4f, -5.f, 8.f, 0.01f);
  box(1, 12.f, 10.f, 10.f, 10.f);
  put(4, 1, 0.90f);
  put(5, 1, 0.1f);
  points(1, 1.f, 1.f, 0.2f, 2.f, 2.f, 0.2f);
  box(2, 12.f, 10.f, 10.f, 10.f);
  put(4, 2, 0.1f);
  put(5, 2, 0.85f);
  points(2, 3.f, 4.f, 0.3f, 4.f, 5.f, 0.3f);
  box(3, 50.f, 50.f, 6.f, 6.f);
  put(4, 3, 0.70f);
  put(5, 3, 0.70f);
  points(3, 48.f, 49.f, 0.6f, 51.f, 52.f, 0.7f);
  box(4, 20.f, 20.f, 4.f, 4.f);
  put(4, 4, 0.2f);
  put(5, 4, 0.2f);
  points(4, std::nanf(""), 1.f, 0.5f, 1.f, 1.f, 0.5f);

  const vie::YoloV8PoseDecoder decoder;
  vie::PoseDecodeConfig config = keep_config();
  const vie::PoseBatch batch =
      decoder.decode_pose(decode_input(pose_tensor(values, kChannels, kPredictions), identity_geometry()), config,
                          vie::DeviceContext{});
  ASSERT_EQ(batch.results.size(), 1u);
  const vie::PoseResult& result = batch.results[0];
  EXPECT_EQ(result.protocol, "yolov8_pose_raw");
  EXPECT_EQ(result.execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(result.num_classes, 2);
  ASSERT_EQ(result.objects.size(), 3u);
  EXPECT_EQ(result.objects[0].anchor_index, 0);
  EXPECT_EQ(result.objects[0].class_id, 0);
  EXPECT_NEAR(result.objects[0].score, 0.95f, 1e-6f);
  EXPECT_NEAR(result.objects[0].cx, 10.f, 1e-4f);
  EXPECT_EQ(result.objects[0].keypoints.size(), 2u);
  EXPECT_NEAR(result.objects[0].keypoints[0].x, 5.f, 1e-4f);
  EXPECT_NEAR(result.objects[0].keypoints[0].confidence, 0.4f, 1e-6f);
  EXPECT_TRUE(result.objects[0].keypoints[0].inside_image);
  EXPECT_NEAR(result.objects[0].keypoints[1].x, -5.f, 1e-4f);
  EXPECT_NEAR(result.objects[0].keypoints[1].confidence, 0.01f, 1e-6f);
  EXPECT_FALSE(result.objects[0].keypoints[1].inside_image);
  EXPECT_EQ(result.objects[1].anchor_index, 2);
  EXPECT_EQ(result.objects[1].class_id, 1);
  EXPECT_EQ(result.objects[2].anchor_index, 3);
  EXPECT_EQ(result.objects[2].class_id, 0);

  config.pre_nms_top_k = 1;
  const vie::PoseBatch top = decoder.decode_pose(
      decode_input(pose_tensor(values, kChannels, kPredictions), identity_geometry()), config, vie::DeviceContext{});
  ASSERT_EQ(top.results[0].objects.size(), 1u);
  EXPECT_EQ(top.results[0].objects[0].anchor_index, 0);

  std::vector<float> tied(static_cast<std::size_t>(kChannels * 2), 0.f);
  auto put_tied = [&](int channel, int prediction, float value) {
    write_channel(tied, 2, channel, prediction, value);
  };
  put_tied(0, 0, 10.f);
  put_tied(1, 0, 10.f);
  put_tied(2, 0, 4.f);
  put_tied(3, 0, 4.f);
  put_tied(4, 0, 0.6f);
  put_tied(0, 1, 40.f);
  put_tied(1, 1, 40.f);
  put_tied(2, 1, 4.f);
  put_tied(3, 1, 4.f);
  put_tied(4, 1, 0.6f);
  config = keep_config();
  const vie::PoseBatch ties = decoder.decode_pose(decode_input(pose_tensor(tied, kChannels, 2), identity_geometry()),
                                                  config, vie::DeviceContext{});
  ASSERT_EQ(ties.results[0].objects.size(), 2u);
  EXPECT_EQ(ties.results[0].objects[0].anchor_index, 0);
  EXPECT_EQ(ties.results[0].objects[1].anchor_index, 1);

  vie::GeometryTransform anisotropic = identity_geometry(80);
  anisotropic.scale_x = 2;
  anisotropic.scale_y = 4;
  anisotropic.uniform_scale = 0;
  std::vector<float> one(static_cast<std::size_t>(kChannels), 0.f);
  write_channel(one, 1, 0, 0, 20.f);
  write_channel(one, 1, 1, 0, 40.f);
  write_channel(one, 1, 2, 0, 8.f);
  write_channel(one, 1, 3, 0, 16.f);
  write_channel(one, 1, 4, 0, 0.8f);
  write_channel(one, 1, 6, 0, 10.f);
  write_channel(one, 1, 7, 0, 20.f);
  write_channel(one, 1, 8, 0, 0.3f);
  const vie::PoseBatch scaled =
      decoder.decode_pose(decode_input(pose_tensor(one, kChannels, 1), anisotropic), config, vie::DeviceContext{});
  ASSERT_EQ(scaled.results[0].objects.size(), 1u);
  EXPECT_NEAR(scaled.results[0].objects[0].cx, 10.f, 1e-4f);
  EXPECT_NEAR(scaled.results[0].objects[0].cy, 10.f, 1e-4f);
  EXPECT_NEAR(scaled.results[0].objects[0].width, 4.f, 1e-4f);
  EXPECT_NEAR(scaled.results[0].objects[0].height, 4.f, 1e-4f);
  EXPECT_NEAR(scaled.results[0].objects[0].keypoints[0].x, 5.f, 1e-4f);
  EXPECT_NEAR(scaled.results[0].objects[0].keypoints[0].y, 5.f, 1e-4f);

  std::vector<float> outside = one;
  write_channel(outside, 1, 0, 0, -100.f);
  write_channel(outside, 1, 1, 0, -100.f);
  write_channel(outside, 1, 2, 0, 2.f);
  write_channel(outside, 1, 3, 0, 2.f);
  const vie::PoseBatch dropped = decoder.decode_pose(
      decode_input(pose_tensor(outside, kChannels, 1), identity_geometry()), config, vie::DeviceContext{});
  EXPECT_TRUE(dropped.results[0].objects.empty());

  std::vector<float> bad = one;
  write_channel(bad, 1, 7, 0, std::nanf(""));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_pose(decode_input(pose_tensor(bad, kChannels, 1), identity_geometry()), config,
                                       vie::DeviceContext{}));
  write_channel(bad, 1, 7, 0, 1.f);
  write_channel(bad, 1, 2, 0, 0.f);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_pose(decode_input(pose_tensor(bad, kChannels, 1), identity_geometry()), config,
                                       vie::DeviceContext{}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_pose(decode_input(pose_tensor(one, kChannels - 1, 1), identity_geometry()), config,
                                       vie::DeviceContext{}));
  vie::PoseDecodeConfig wrong_count = config;
  wrong_count.keypoint_count = 5;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_pose(decode_input(pose_tensor(one, kChannels, 1), identity_geometry()), wrong_count,
                                       vie::DeviceContext{}));
  vie::PoseDecodeConfig bad_threshold = config;
  bad_threshold.confidence_threshold = 1.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   decoder.decode_pose(decode_input(pose_tensor(one, kChannels, 1), identity_geometry()), bad_threshold,
                                       vie::DeviceContext{}));
  vie::PoseDecodeConfig bad_iou = config;
  bad_iou.iou_threshold = -0.1f;
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   decoder.decode_pose(decode_input(pose_tensor(one, kChannels, 1), identity_geometry()), bad_iou,
                                       vie::DeviceContext{}));
  vie::PoseDecodeConfig bad_top = config;
  bad_top.pre_nms_top_k = 4097;
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   decoder.decode_pose(decode_input(pose_tensor(one, kChannels, 1), identity_geometry()), bad_top,
                                       vie::DeviceContext{}));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, decoder.decode({}, {}));
}

TEST(YoloV8PoseDecoder, LayoutsNmsPoliciesAndEmptyCandidates) {
  constexpr int kChannels = 12;
  // cx, cy, width, height, class 0, class 1, then (x, y, confidence) twice.
  const std::vector<std::vector<float>> people{
      {10.f, 10.f, 10.f, 10.f, 0.95f, 0.10f, 5.f, 6.f, 0.4f, -5.f, 8.f, 0.01f},
      {40.f, 40.f, 8.f, 8.f, 0.20f, 0.85f, 37.f, 38.f, 0.3f, 39.f, 41.f, 0.9f}};
  auto channel_major = [&](const std::vector<std::vector<float>>& rows) {
    const int predictions = static_cast<int>(rows.size());
    std::vector<float> flat(static_cast<std::size_t>(kChannels * predictions), 0.f);
    for (int prediction = 0; prediction < predictions; ++prediction) {
      for (int channel = 0; channel < kChannels; ++channel) {
        flat[static_cast<std::size_t>(channel * predictions + prediction)] =
            rows[static_cast<std::size_t>(prediction)][static_cast<std::size_t>(channel)];
      }
    }
    return flat;
  };
  auto prediction_major = [&](const std::vector<std::vector<float>>& rows) {
    std::vector<float> flat;
    for (const std::vector<float>& row : rows) flat.insert(flat.end(), row.begin(), row.end());
    return flat;
  };

  const vie::YoloV8PoseDecoder decoder;
  const vie::GeometryTransform geometry = identity_geometry();
  vie::PoseDecodeConfig axis1 = keep_config();
  const vie::PoseBatch rows = decoder.decode_pose(
      decode_input(pose_tensor(channel_major(people), kChannels, 2), geometry), axis1, vie::DeviceContext{});
  vie::PoseDecodeConfig axis2 = keep_config();
  axis2.channel_axis = 2;
  const std::vector<float> flat = prediction_major(people);
  const vie::Tensor transposed = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 2, kChannels},
                                                        flat.data(), flat.size() * sizeof(float));
  const vie::PoseBatch columns = decoder.decode_pose(decode_input(transposed, geometry), axis2, vie::DeviceContext{});
  ASSERT_EQ(rows.results[0].objects.size(), 2u);
  ASSERT_EQ(columns.results[0].objects.size(), rows.results[0].objects.size());
  for (std::size_t index = 0; index < rows.results[0].objects.size(); ++index) {
    const vie::PoseObject& left = rows.results[0].objects[index];
    const vie::PoseObject& right = columns.results[0].objects[index];
    EXPECT_EQ(left.anchor_index, right.anchor_index) << index;
    EXPECT_EQ(left.class_id, right.class_id) << index;
    EXPECT_FLOAT_EQ(left.score, right.score) << index;
    EXPECT_FLOAT_EQ(left.cx, right.cx) << index;
    EXPECT_FLOAT_EQ(left.height, right.height) << index;
    ASSERT_EQ(left.keypoints.size(), right.keypoints.size());
    for (std::size_t point = 0; point < left.keypoints.size(); ++point) {
      EXPECT_FLOAT_EQ(left.keypoints[point].x, right.keypoints[point].x) << index << " " << point;
      EXPECT_FLOAT_EQ(left.keypoints[point].confidence, right.keypoints[point].confidence) << index << " " << point;
      EXPECT_EQ(left.keypoints[point].inside_image, right.keypoints[point].inside_image) << index << " " << point;
    }
  }

  // The same box on two anchors whose best classes differ.
  const std::vector<std::vector<float>> overlap{
      {20.f, 20.f, 10.f, 10.f, 0.95f, 0.10f, 18.f, 22.f, 0.5f, 21.f, 19.f, 0.6f},
      {20.f, 20.f, 10.f, 10.f, 0.10f, 0.90f, 18.f, 22.f, 0.5f, 21.f, 19.f, 0.6f}};
  const std::vector<float> overlap_data = channel_major(overlap);
  const vie::Tensor overlap_tensor = pose_tensor(overlap_data, kChannels, 2);
  vie::PoseDecodeConfig class_aware = keep_config();
  class_aware.class_aware = true;
  const vie::PoseBatch separated =
      decoder.decode_pose(decode_input(overlap_tensor, geometry), class_aware, vie::DeviceContext{});
  ASSERT_EQ(separated.results[0].objects.size(), 2u);
  EXPECT_EQ(separated.results[0].objects[0].class_id, 0);
  EXPECT_EQ(separated.results[0].objects[1].class_id, 1);
  EXPECT_EQ(separated.results[0].candidates_before_nms, 2);
  vie::PoseDecodeConfig shared = keep_config();
  shared.class_aware = false;
  const vie::PoseBatch merged =
      decoder.decode_pose(decode_input(overlap_tensor, geometry), shared, vie::DeviceContext{});
  ASSERT_EQ(merged.results[0].objects.size(), 1u);
  EXPECT_EQ(merged.results[0].objects[0].anchor_index, 0);

  // NMS off keeps every mapped candidate in sorted order, up to max_detections.
  vie::PoseDecodeConfig no_nms = keep_config();
  no_nms.nms_enabled = false;
  no_nms.iou_threshold.reset();
  no_nms.class_aware.reset();
  const vie::PoseBatch unfiltered =
      decoder.decode_pose(decode_input(overlap_tensor, geometry), no_nms, vie::DeviceContext{});
  ASSERT_EQ(unfiltered.results[0].objects.size(), 2u);
  EXPECT_EQ(unfiltered.results[0].objects[0].anchor_index, 0);
  EXPECT_EQ(unfiltered.results[0].objects[1].anchor_index, 1);
  EXPECT_EQ(unfiltered.results[0].candidates_before_nms, 2);

  // Every class above the threshold becomes its own candidate on one anchor.
  const std::vector<std::vector<float>> both{
      {20.f, 20.f, 10.f, 10.f, 0.95f, 0.90f, 18.f, 22.f, 0.5f, 21.f, 19.f, 0.6f}};
  const vie::Tensor both_tensor = pose_tensor(channel_major(both), kChannels, 1);
  vie::PoseDecodeConfig all_classes = keep_config();
  all_classes.class_selection = vie::ClassSelection::AllAboveThreshold;
  const vie::PoseBatch multi =
      decoder.decode_pose(decode_input(both_tensor, geometry), all_classes, vie::DeviceContext{});
  ASSERT_EQ(multi.results[0].objects.size(), 2u);
  EXPECT_EQ(multi.results[0].objects[0].class_id, 0);
  EXPECT_NEAR(multi.results[0].objects[0].score, 0.95f, 1e-6f);
  EXPECT_EQ(multi.results[0].objects[1].class_id, 1);
  EXPECT_NEAR(multi.results[0].objects[1].score, 0.90f, 1e-6f);
  EXPECT_EQ(multi.results[0].objects[0].anchor_index, 0);
  EXPECT_EQ(multi.results[0].objects[1].anchor_index, 0);

  // Nothing above the threshold is an empty result, not an error.
  vie::PoseDecodeConfig nothing = keep_config();
  nothing.confidence_threshold = 0.99f;
  const vie::PoseBatch empty =
      decoder.decode_pose(decode_input(overlap_tensor, geometry), nothing, vie::DeviceContext{});
  EXPECT_TRUE(empty.results[0].objects.empty());
  EXPECT_EQ(empty.results[0].candidates_after_filter, 0);
  EXPECT_EQ(empty.results[0].candidates_before_nms, 0);
  EXPECT_GT(empty.results[0].output_bytes, 0u);

  // A bad channel in a kept anchor fails the whole call.
  auto at = [](int channel, int prediction) { return static_cast<std::size_t>(channel * 2 + prediction); };
  std::vector<float> bad_box = overlap_data;
  bad_box[at(0, 0)] = std::nanf("");
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decoder.decode_pose(
                     decode_input(pose_tensor(bad_box, kChannels, 2), geometry), keep_config(), vie::DeviceContext{}));
  std::vector<float> bad_confidence = overlap_data;
  bad_confidence[at(8, 0)] = std::nanf("");
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decoder.decode_pose(
                     decode_input(pose_tensor(bad_confidence, kChannels, 2), geometry), keep_config(),
                     vie::DeviceContext{}));
  std::vector<float> bad_probability = overlap_data;
  bad_probability[at(4, 0)] = 1.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decoder.decode_pose(
                     decode_input(pose_tensor(bad_probability, kChannels, 2), geometry), keep_config(),
                     vie::DeviceContext{}));

  // A bad keypoint inside an anchor that the threshold already dropped is not
  // read, so it does not fail the call.
  std::vector<std::vector<float>> half{people};
  half[1][4] = 0.2f;
  half[1][5] = 0.3f;
  std::vector<float> half_data = channel_major(half);
  half_data[at(8, 1)] = std::nanf("");
  const vie::PoseBatch survived = decoder.decode_pose(
      decode_input(pose_tensor(half_data, kChannels, 2), geometry), keep_config(), vie::DeviceContext{});
  ASSERT_EQ(survived.results[0].objects.size(), 1u);
  EXPECT_EQ(survived.results[0].objects[0].anchor_index, 0);
}

TEST(YoloV8PoseRecognizer, AcceptsTheProvenHeadAndRejectsLookalikes) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
  const vie::ModelAnalysisReport report = analyze_graph(pose_graph(true, false, false, 12));
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_pose");
  EXPECT_EQ(result.selected()->decoder.protocol, "yolov8_pose_raw");
  EXPECT_EQ(*result.selected()->decoder.num_classes, 2);
  EXPECT_EQ(*result.selected()->decoder.keypoint_count, 2);
  EXPECT_EQ(*result.selected()->decoder.keypoint_dimensions, 3);
  EXPECT_EQ(result.selected()->decoder.keypoint_layout, "xy_confidence");
  EXPECT_FALSE(vie::YoloV8DetectRecognizer{}.consider(report).has_value());
  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, "pose.onnx");
  EXPECT_EQ(manifest.task, vie::TaskType::Pose);
  EXPECT_EQ(manifest.pose.keypoint_count, 2);
  EXPECT_EQ(manifest.pose.keypoint_layout, "xy_confidence");
  EXPECT_EQ(manifest.decoder.box_decode, "decoded_cxcywh");
  EXPECT_TRUE(manifest.input.color.empty());
  EXPECT_TRUE(manifest.input.resize.empty());
  EXPECT_TRUE(manifest.input.normalization.empty());
  EXPECT_FALSE(manifest.nms.enabled.has_value());
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_FALSE(validation.runnable());
  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(manifest.to_json_string());
  EXPECT_EQ(loaded.pose.output_name, "output0");
  EXPECT_EQ(loaded.pose.keypoint_dimensions, 3);
  EXPECT_EQ(loaded.task, vie::TaskType::Pose);

  EXPECT_EQ(vie::default_recognizer_registry().recognize(analyze_graph(pose_graph(false, false, false, 12))).status,
            vie::RecognitionStatus::Unknown);
  EXPECT_EQ(vie::default_recognizer_registry().recognize(analyze_graph(pose_graph(true, true, false, 12))).status,
            vie::RecognitionStatus::Unknown);
  EXPECT_EQ(vie::default_recognizer_registry().recognize(analyze_graph(pose_graph(true, false, true, 12))).status,
            vie::RecognitionStatus::Unknown);
  EXPECT_EQ(vie::default_recognizer_registry().recognize(analyze_graph(pose_graph(true, false, false, 11))).status,
            vie::RecognitionStatus::Unknown);

  const std::string detection = vie_fixture("testdata/onnx/yolov8n.onnx");
  std::ifstream input(detection, std::ios::binary);
  ASSERT_TRUE(input);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const vie::ModelAnalysisReport real = vie::OnnxAnalyzer{}.analyze_bytes(buffer.str(), "yolov8n.onnx", ".");
  EXPECT_FALSE(vie::YoloV8PoseRecognizer{}.consider(real).has_value());
  const vie::RecognitionResult kept = vie::default_recognizer_registry().recognize(real);
  EXPECT_EQ(kept.status, vie::RecognitionStatus::Recognized);
  ASSERT_NE(kept.selected(), nullptr);
  EXPECT_EQ(kept.selected()->architecture, "yolov8_detection");

  vie::ModelManifest stolen = runnable_pose_manifest();
  stolen.task = vie::TaskType::Detection;
  stolen.decoder.protocol = "yolov8_detect_raw";
  stolen.decoder.box_decode.clear();
  stolen.provenance["task"] = vie::FieldProvenance{vie::Certainty::User, {"test"}};
  stolen.provenance["decoder.protocol"] = vie::FieldProvenance{vie::Certainty::User, {"test"}};
  const vie::ManifestValidation rejected = vie::ManifestValidator{}.validate(stolen);
  EXPECT_FALSE(rejected.schema_ok());
}

TEST(PoseApi, RejectsTheWrongTask) {
  vie::InferenceEngine engine;
  vie::ModelConfig config;
  config.manifest = runnable_pose_manifest();
  config.backend = vie::BackendKind::Mock;
  engine.load(config);
  vie::ImageInput image;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image));
  vie::ModelManifest detection = runnable_pose_manifest();
  detection.task = vie::TaskType::Detection;
  detection.decoder.protocol = "mock";
  detection.decoder.box_format = "xyxy";
  detection.decoder.box_decode.clear();
  detection.pose = {};
  for (const char* path : {"pose.output_name", "pose.channel_axis", "pose.keypoint_count", "pose.keypoint_dimensions",
                           "pose.keypoint_layout", "pose.score_type", "decoder.box_decode"}) {
    detection.provenance.erase(path);
  }
  detection.ensure_provenance();
  vie::InferenceEngine detector;
  vie::ModelConfig detection_config;
  detection_config.manifest = detection;
  detection_config.backend = vie::BackendKind::Mock;
  detector.load(detection_config);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, detector.infer_pose(image, {}));
}

TEST(PoseManifest, RejectsIncompleteAndInconsistentPoseFields) {
  auto has_issue = [](const std::vector<vie::SchemaIssue>& issues, const std::string& path) {
    for (const vie::SchemaIssue& issue : issues) {
      if (issue.path == path) return true;
    }
    return false;
  };
  const vie::ManifestValidation valid = vie::ManifestValidator{}.validate(runnable_pose_manifest());
  EXPECT_TRUE(valid.schema_ok()) << vie::format_issues(valid.schema_issues);
  EXPECT_TRUE(valid.runnable()) << vie::format_issues(valid.unresolved_critical);

  // An unset pose field is a critical gap, not a schema error.
  for (const char* path : {"pose.output_name", "pose.channel_axis", "pose.keypoint_count", "pose.keypoint_dimensions",
                           "pose.keypoint_layout", "pose.score_type"}) {
    vie::ModelManifest manifest = runnable_pose_manifest();
    const std::string key(path);
    if (key == "pose.output_name") manifest.pose.output_name.clear();
    if (key == "pose.channel_axis") manifest.pose.channel_axis.reset();
    if (key == "pose.keypoint_count") manifest.pose.keypoint_count.reset();
    if (key == "pose.keypoint_dimensions") manifest.pose.keypoint_dimensions.reset();
    if (key == "pose.keypoint_layout") manifest.pose.keypoint_layout.clear();
    if (key == "pose.score_type") manifest.pose.score_type.clear();
    manifest.provenance.erase(path);
    manifest.ensure_provenance();
    const vie::ManifestValidation missing = vie::ManifestValidator{}.validate(manifest);
    EXPECT_TRUE(missing.schema_ok()) << path << ": " << vie::format_issues(missing.schema_issues);
    EXPECT_TRUE(has_issue(missing.unresolved_critical, path)) << path;
    EXPECT_FALSE(missing.runnable()) << path;
  }

  vie::ModelManifest dimensions = runnable_pose_manifest();
  dimensions.pose.keypoint_dimensions = 2;
  EXPECT_TRUE(has_issue(vie::ManifestValidator{}.validate(dimensions).schema_issues, "pose.keypoint_dimensions"));

  vie::ModelManifest layout = runnable_pose_manifest();
  layout.pose.keypoint_layout = "xy";
  EXPECT_TRUE(has_issue(vie::ManifestValidator{}.validate(layout).schema_issues, "pose.keypoint_layout"));

  vie::ModelManifest score_type = runnable_pose_manifest();
  score_type.pose.score_type = "logits";
  EXPECT_TRUE(has_issue(vie::ManifestValidator{}.validate(score_type).schema_issues, "pose.score_type"));

  vie::ModelManifest axis_mismatch = runnable_pose_manifest();
  axis_mismatch.pose.channel_axis = 2;
  EXPECT_TRUE(
      has_issue(vie::ManifestValidator{}.validate(axis_mismatch).schema_issues, "pose.channel_axis"));

  vie::ModelManifest wrong_name = runnable_pose_manifest();
  wrong_name.pose.output_name = "output1";
  EXPECT_TRUE(has_issue(vie::ManifestValidator{}.validate(wrong_name).schema_issues, "pose.output_name"));

  vie::ModelManifest wrong_length = runnable_pose_manifest();
  wrong_length.outputs[0].shape = {1, 11, 4};
  EXPECT_TRUE(has_issue(vie::ManifestValidator{}.validate(wrong_length).schema_issues, "pose.channel_axis"));

  vie::ModelManifest wrong_protocol = runnable_pose_manifest();
  wrong_protocol.decoder.protocol = "yolov8_detect_raw";
  wrong_protocol.decoder.box_decode.clear();
  EXPECT_TRUE(
      has_issue(vie::ManifestValidator{}.validate(wrong_protocol).schema_issues, "decoder.protocol"));

  // A detection manifest may not smuggle pose fields.
  const std::vector<std::string> pose_paths{"pose.output_name", "pose.channel_axis", "pose.keypoint_count",
                                             "pose.keypoint_dimensions", "pose.keypoint_layout", "pose.score_type"};
  vie::ModelManifest detection_only = runnable_pose_manifest();
  detection_only.task = vie::TaskType::Detection;
  detection_only.decoder.protocol = "mock";
  detection_only.decoder.box_format = "xyxy";
  detection_only.decoder.box_decode.clear();
  detection_only.provenance.erase("decoder.box_decode");
  detection_only.ensure_provenance();
  EXPECT_TRUE(has_issue(vie::ManifestValidator{}.validate(detection_only).schema_issues, "pose"));

  // The pose object survives a round trip and rejects illegal types and values.
  const std::string text = runnable_pose_manifest().to_json_string(-1);
  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(text);
  EXPECT_EQ(*loaded.pose.keypoint_count, 2);
  EXPECT_EQ(*loaded.pose.keypoint_dimensions, 3);
  EXPECT_EQ(loaded.pose.keypoint_layout, "xy_confidence");
  auto patched = [&text](const std::string& from, const std::string& to) {
    std::string out = text;
    const std::size_t found = out.find(from);
    EXPECT_NE(found, std::string::npos) << from;
    out.replace(found, from.size(), to);
    return out;
  };
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   vie::ModelManifest::from_json_string(
                       patched("\"keypoint_dimensions\":3", "\"keypoint_dimensions\":5")));
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   vie::ModelManifest::from_json_string(
                       patched("\"keypoint_count\":2", "\"keypoint_count\":0")));
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   vie::ModelManifest::from_json_string(
                       patched("\"pose\":{\"channel_axis\":1", "\"pose\":{\"channel_axis\":3")));
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   vie::ModelManifest::from_json_string(
                       patched("\"keypoint_layout\":\"xy_confidence\"", "\"keypoint_layout\":\"xy\"")));

  vie::ModelManifest without_pose = runnable_pose_manifest();
  without_pose.task = vie::TaskType::Detection;
  without_pose.decoder.protocol = "mock";
  without_pose.decoder.box_format = "xyxy";
  without_pose.decoder.box_decode.clear();
  without_pose.pose = {};
  for (const std::string& path : pose_paths) without_pose.provenance.erase(path);
  without_pose.provenance.erase("decoder.box_decode");
  without_pose.ensure_provenance();
  const std::string pose_free = without_pose.to_json_string(-1);
  EXPECT_EQ(pose_free.find("\"pose\":{"), std::string::npos) << pose_free;
  EXPECT_TRUE(vie::ModelManifest::from_json_string(pose_free).pose.output_name.empty());
}

TEST(YoloV8PoseRecognizer, RealExportWhenPresent) {
  std::ifstream input(kRealModel, std::ios::binary);
  if (!input) {
    GTEST_SKIP() << "real yolov8n-pose onnx is not in /tmp/vie_phase72";
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_bytes(buffer.str(), "yolov8n-pose.onnx", ".");
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_pose");
  EXPECT_EQ(result.selected()->decoder.protocol, "yolov8_pose_raw");
  EXPECT_EQ(result.selected()->decoder.channel_axis, 1);
  EXPECT_EQ(result.selected()->decoder.keypoint_layout, "xy_confidence");
}

#if defined(VIE_HAS_ONNXRUNTIME)
namespace {

struct ReferencePose {
  float cx = 0.f;
  float cy = 0.f;
  float width = 0.f;
  float height = 0.f;
  int class_id = -1;
  float score = 0.f;
  int anchor = -1;
  std::vector<vie::Keypoint> keypoints;
};

const std::filesystem::path kPhaseDir{"/tmp/vie_phase72"};

bool reference_ready() {
  return std::filesystem::is_regular_file(kRealModel) &&
         std::filesystem::is_regular_file(kPhaseDir / "raw_output.f32") &&
         std::filesystem::is_regular_file(kPhaseDir / "reference_poses.txt") &&
         std::filesystem::is_regular_file(kPhaseDir / "input_u8.bin") &&
         std::filesystem::is_regular_file(kPhaseDir / "raw_shape.txt");
}

int model_side() {
  std::ifstream shape(kPhaseDir / "raw_shape.txt");
  int batch = 0;
  int channels = 0;
  int anchors = 0;
  shape >> batch >> channels >> anchors;
  std::ifstream side_file(kPhaseDir / "image_side.txt");
  int side = 0;
  side_file >> side;
  return side;
}

vie::ImageInput pattern_image(std::vector<std::uint8_t>& storage) {
  const int side = model_side();
  storage.resize(static_cast<std::size_t>(side * side * 3));
  std::ifstream input(kPhaseDir / "input_u8.bin", std::ios::binary);
  input.read(reinterpret_cast<char*>(storage.data()), static_cast<std::streamsize>(storage.size()));
  EXPECT_TRUE(static_cast<bool>(input));
  vie::ImageInput image;
  image.data = storage.data();
  image.width = side;
  image.height = side;
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
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, kRealModel);
  supply_test_preprocess(manifest);
  return manifest;
}

vie::PoseRuntimeConfig reference_runtime() {
  vie::PoseRuntimeConfig runtime;
  runtime.class_selection = vie::ClassSelection::BestClass;
  runtime.confidence_threshold = 0.f;
  runtime.nms_enabled = true;
  runtime.class_aware = true;
  runtime.iou_threshold = 0.45f;
  runtime.pre_nms_top_k = 300;
  runtime.max_detections = 20;
  return runtime;
}

std::vector<ReferencePose> load_reference() {
  std::ifstream input(kPhaseDir / "reference_poses.txt");
  std::string header;
  std::getline(input, header);
  EXPECT_NE(header.find("confidence 0"), std::string::npos);
  EXPECT_NE(header.find("iou 0.45"), std::string::npos);
  EXPECT_NE(header.find("pre_nms 300"), std::string::npos);
  EXPECT_NE(header.find("max_detections 20"), std::string::npos);
  std::vector<ReferencePose> poses;
  int count = 0;
  int keypoints = 0;
  input >> count >> keypoints;
  for (int index = 0; index < count; ++index) {
    ReferencePose pose;
    input >> pose.cx >> pose.cy >> pose.width >> pose.height >> pose.class_id >> pose.score >> pose.anchor;
    pose.keypoints.resize(static_cast<std::size_t>(keypoints));
    for (int point = 0; point < keypoints; ++point) {
      int inside = 0;
      input >> pose.keypoints[static_cast<std::size_t>(point)].x >> pose.keypoints[static_cast<std::size_t>(point)].y >>
          pose.keypoints[static_cast<std::size_t>(point)].confidence >> inside;
      pose.keypoints[static_cast<std::size_t>(point)].inside_image = inside != 0;
    }
    poses.push_back(std::move(pose));
  }
  EXPECT_TRUE(static_cast<bool>(input) || input.eof());
  return poses;
}

void expect_matches_reference(const vie::PoseBatch& batch, const std::vector<ReferencePose>& expected) {
  ASSERT_EQ(batch.results.size(), 1u);
  const vie::PoseResult& result = batch.results[0];
  const int side = model_side();
  EXPECT_EQ(result.image_width, side);
  EXPECT_EQ(result.image_height, side);
  EXPECT_EQ(result.execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(result.protocol, "yolov8_pose_raw");
  ASSERT_EQ(result.objects.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const vie::PoseObject& got = result.objects[index];
    const ReferencePose& want = expected[index];
    EXPECT_EQ(got.class_id, want.class_id) << index;
    EXPECT_EQ(got.anchor_index, want.anchor) << index;
    EXPECT_NEAR(got.score, want.score, 1e-4f) << index;
    EXPECT_NEAR(got.cx, want.cx, 1e-2f) << index;
    EXPECT_NEAR(got.cy, want.cy, 1e-2f) << index;
    EXPECT_NEAR(got.width, want.width, 1e-2f) << index;
    EXPECT_NEAR(got.height, want.height, 1e-2f) << index;
    ASSERT_EQ(got.keypoints.size(), want.keypoints.size());
    for (std::size_t point = 0; point < want.keypoints.size(); ++point) {
      EXPECT_NEAR(got.keypoints[point].x, want.keypoints[point].x, 1e-2f) << index << " " << point;
      EXPECT_NEAR(got.keypoints[point].y, want.keypoints[point].y, 1e-2f) << index << " " << point;
      EXPECT_NEAR(got.keypoints[point].confidence, want.keypoints[point].confidence, 1e-4f) << index << " " << point;
      EXPECT_EQ(got.keypoints[point].inside_image, want.keypoints[point].inside_image) << index << " " << point;
    }
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

TEST(YoloV8PoseReal, CpuMatchesIndependentReference) {
  if (!reference_ready()) {
    GTEST_SKIP() << "run tools/export_yolov8n_pose.py";
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
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image));
  vie::PoseInferTiming timing;
  const vie::PoseBatch batch = engine.infer_pose(image, reference_runtime(), &timing);
  vie::BackendRequest request;
  request.inputs.emplace(engine.manifest().input.name, engine.preprocess(image).output);
  const vie::BackendOutput output = engine.execute(request);
  const vie::Tensor& tensor = output.tensors.at("output0").tensor;
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
  {
    std::ofstream note("/tmp/vie_phase72_cpu_raw.txt");
    note << "pytorch_vs_ort_cpu_max_abs " << max_abs << " objects " << batch.results[0].objects.size()
         << " output_bytes " << tensor.nbytes() << "\n";
  }
  expect_matches_reference(batch, load_reference());
  EXPECT_EQ(timing.d2h_bytes, 0u);
  EXPECT_FALSE(timing.includes_d2h);
  EXPECT_EQ(timing.output_bytes, tensor.nbytes());
  EXPECT_GT(batch.results[0].objects.size(), 0u);
}

TEST(YoloV8PoseReal, CpuTimingUsesRepeatedSamples) {
  if (!reference_ready()) {
    GTEST_SKIP() << "run tools/export_yolov8n_pose.py";
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
  const vie::PoseRuntimeConfig runtime = reference_runtime();
  constexpr int kWarmup = 3;
  constexpr int kSamples = 10;
  for (int index = 0; index < kWarmup; ++index) engine.infer_pose(image, runtime, nullptr);
  std::vector<double> preprocess_ms;
  std::vector<double> execute_ms;
  std::vector<double> validate_ms;
  std::vector<double> pose_decode_ms;
  std::vector<double> nms_ms;
  std::vector<double> decode_ms;
  std::vector<double> total_ms;
  int candidates = 0;
  int kept = 0;
  std::size_t output_bytes = 0;
  for (int index = 0; index < kSamples; ++index) {
    vie::PoseInferTiming timing;
    engine.infer_pose(image, runtime, &timing);
    preprocess_ms.push_back(timing.preprocess_ms);
    execute_ms.push_back(timing.execute_ms);
    validate_ms.push_back(timing.validate_ms);
    pose_decode_ms.push_back(timing.pose_decode_ms);
    nms_ms.push_back(timing.nms_ms);
    decode_ms.push_back(timing.decode_ms);
    total_ms.push_back(timing.total_ms);
    candidates = timing.candidates_before_nms;
    kept = timing.results_after_nms;
    output_bytes = timing.output_bytes;
  }
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
  std::ofstream record("/tmp/vie_phase72_cpu_ep_in_gpu_build.txt");
#else
  std::ofstream record("/tmp/vie_phase72_cpu.txt");
#endif
  record << "device cpu warmup " << kWarmup << " samples " << kSamples << " candidates " << candidates
         << " kept " << kept << " output_bytes " << output_bytes << " d2h_bytes 0\n";
  record << "preprocess mean " << mean_of(preprocess_ms) << " p50 " << percentile(preprocess_ms, 0.50) << " p95 "
         << percentile(preprocess_ms, 0.95) << "\n";
  record << "execute mean " << mean_of(execute_ms) << " p50 " << percentile(execute_ms, 0.50) << " p95 "
         << percentile(execute_ms, 0.95) << "\n";
  record << "validate mean " << mean_of(validate_ms) << " p50 " << percentile(validate_ms, 0.50) << " p95 "
         << percentile(validate_ms, 0.95) << "\n";
  record << "pose_decode mean " << mean_of(pose_decode_ms) << " p50 " << percentile(pose_decode_ms, 0.50) << " p95 "
         << percentile(pose_decode_ms, 0.95) << "\n";
  record << "nms mean " << mean_of(nms_ms) << " p50 " << percentile(nms_ms, 0.50) << " p95 "
         << percentile(nms_ms, 0.95) << "\n";
  record << "decode mean " << mean_of(decode_ms) << " p50 " << percentile(decode_ms, 0.50) << " p95 "
         << percentile(decode_ms, 0.95) << "\n";
  record << "total mean " << mean_of(total_ms) << " p50 " << percentile(total_ms, 0.50) << " p95 "
         << percentile(total_ms, 0.95) << "\n";
  EXPECT_GT(mean_of(execute_ms), 0.0);
  EXPECT_GT(mean_of(total_ms), 0.0);
}

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
TEST(YoloV8PoseReal, StrictCudaMatchesCpu) {
  if (!reference_ready()) {
    GTEST_SKIP() << "run tools/export_yolov8n_pose.py";
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
  const vie::PoseRuntimeConfig runtime = reference_runtime();

  vie::ModelConfig cpu_config;
  cpu_config.model_path = kRealModel;
  cpu_config.manifest = manifest;
  cpu_config.backend = vie::BackendKind::OnnxRuntime;
  cpu_config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine cpu_engine;
  cpu_engine.load(cpu_config);
  const vie::PoseBatch cpu_batch = cpu_engine.infer_pose(image, runtime, nullptr);

  vie::ModelConfig cuda_config = cpu_config;
  cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
  cuda_config.device = vie::DeviceType::Cuda;
  ASSERT_FALSE(cuda_config.cuda_cpu_ep_fallback);
  EXPECT_FALSE(vie::OnnxRuntimeCudaBackend{}.cpu_ep_fallback());
  vie::InferenceEngine cuda_engine;
  cuda_engine.load(cuda_config);
  constexpr int kWarmup = 3;
  constexpr int kSamples = 10;
  for (int index = 0; index < kWarmup; ++index) cuda_engine.infer_pose(gpu_image, runtime, nullptr);
  std::vector<double> preprocess_ms;
  std::vector<double> execute_ms;
  std::vector<double> d2h_ms;
  std::vector<double> validate_ms;
  std::vector<double> pose_decode_ms;
  std::vector<double> nms_ms;
  std::vector<double> decode_ms;
  std::vector<double> total_ms;
  vie::PoseBatch cuda_batch;
  vie::PoseInferTiming timing;
  for (int index = 0; index < kSamples; ++index) {
    cuda_batch = cuda_engine.infer_pose(gpu_image, runtime, &timing);
    preprocess_ms.push_back(timing.preprocess_ms);
    execute_ms.push_back(timing.execute_ms);
    d2h_ms.push_back(timing.d2h_ms);
    validate_ms.push_back(timing.validate_ms);
    pose_decode_ms.push_back(timing.pose_decode_ms);
    nms_ms.push_back(timing.nms_ms);
    decode_ms.push_back(timing.decode_ms);
    total_ms.push_back(timing.total_ms);
  }
  // Phase 14: the pose kernels keep the raw tensor on the device and copy back
  // only the kept objects and their keypoints.
  const std::size_t kept = cuda_batch.results[0].objects.size();
  const std::size_t keypoints = static_cast<std::size_t>(manifest.pose.keypoint_count.value_or(0));
  EXPECT_EQ(timing.results_after_nms, static_cast<int>(kept));
  EXPECT_EQ(timing.d2h_bytes, kept * 28u + kept * keypoints * 16u);
  EXPECT_GT(timing.d2h_bytes, 0u);
  EXPECT_LT(timing.d2h_bytes, timing.output_bytes);
  EXPECT_TRUE(timing.includes_d2h);
  EXPECT_EQ(cuda_batch.results[0].execution_device, vie::DeviceType::Cuda);
  EXPECT_EQ(cuda_batch.results[0].execution_device_id, 0);
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
  const int channels = static_cast<int>(cpu_tensor.shape()[1]);
  const int anchors = static_cast<int>(cpu_tensor.shape()[2]);
  const int classes = cpu_engine.manifest().num_classes.value_or(0);
  const int keypoint_base = 4 + classes;
  float max_abs = 0.f;
  float max_box = 0.f;
  float max_score = 0.f;
  float max_keypoint = 0.f;
  for (int channel = 0; channel < channels; ++channel) {
    for (int anchor = 0; anchor < anchors; ++anchor) {
      const float delta = std::fabs(cpu_data[channel * anchors + anchor] - cuda_data[channel * anchors + anchor]);
      max_abs = std::max(max_abs, delta);
      if (channel < 4) max_box = std::max(max_box, delta);
      else if (channel < keypoint_base) max_score = std::max(max_score, delta);
      else max_keypoint = std::max(max_keypoint, delta);
    }
  }
  // Measured on this 640 pattern: box channels differ by at most 0.623 px,
  // class probabilities stay under 7e-4, and keypoint channels stay under 0.403.
  EXPECT_LT(max_box, 0.75f);
  EXPECT_LT(max_score, 1e-3f);
  EXPECT_LT(max_keypoint, 0.5f);

  int shared = 0;
  for (const vie::PoseObject& cpu : cpu_batch.results[0].objects) {
    const vie::PoseObject* cuda = nullptr;
    for (const vie::PoseObject& candidate : cuda_batch.results[0].objects) {
      if (candidate.anchor_index == cpu.anchor_index && candidate.class_id == cpu.class_id) cuda = &candidate;
    }
    if (cuda == nullptr) continue;
    ++shared;
    EXPECT_NEAR(cuda->score, cpu.score, 1e-3f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->cx, cpu.cx, 0.75f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->cy, cpu.cy, 0.75f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->width, cpu.width, 0.75f) << cpu.anchor_index;
    EXPECT_NEAR(cuda->height, cpu.height, 0.75f) << cpu.anchor_index;
    ASSERT_EQ(cuda->keypoints.size(), cpu.keypoints.size());
    for (std::size_t point = 0; point < cpu.keypoints.size(); ++point) {
      EXPECT_NEAR(cuda->keypoints[point].x, cpu.keypoints[point].x, 0.5f);
      EXPECT_NEAR(cuda->keypoints[point].y, cpu.keypoints[point].y, 0.5f);
      EXPECT_NEAR(cuda->keypoints[point].confidence, cpu.keypoints[point].confidence, 1e-3f);
    }
  }
  EXPECT_EQ(shared, static_cast<int>(cpu_batch.results[0].objects.size()));
  EXPECT_EQ(cuda_batch.results[0].objects.size(), cpu_batch.results[0].objects.size());
  std::ofstream record("/tmp/vie_phase72_cuda.txt");
  record << "device cuda warmup " << kWarmup << " samples " << kSamples << " candidates "
         << timing.candidates_before_nms << " kept " << timing.results_after_nms << " cpu_kept "
         << cpu_batch.results[0].objects.size() << " shared_anchors " << shared << " raw_max_abs " << max_abs
         << " raw_max_box " << max_box << " raw_max_score " << max_score << " raw_max_keypoint " << max_keypoint
         << " output_bytes " << timing.output_bytes << " d2h_bytes " << timing.d2h_bytes << "\n";
  record << "preprocess mean " << mean_of(preprocess_ms) << " p50 " << percentile(preprocess_ms, 0.50) << " p95 "
         << percentile(preprocess_ms, 0.95) << "\n";
  record << "execute mean " << mean_of(execute_ms) << " p50 " << percentile(execute_ms, 0.50) << " p95 "
         << percentile(execute_ms, 0.95) << "\n";
  record << "d2h mean " << mean_of(d2h_ms) << " p50 " << percentile(d2h_ms, 0.50) << " p95 "
         << percentile(d2h_ms, 0.95) << "\n";
  record << "validate mean " << mean_of(validate_ms) << " p50 " << percentile(validate_ms, 0.50) << " p95 "
         << percentile(validate_ms, 0.95) << "\n";
  record << "pose_decode mean " << mean_of(pose_decode_ms) << " p50 " << percentile(pose_decode_ms, 0.50) << " p95 "
         << percentile(pose_decode_ms, 0.95) << "\n";
  record << "nms mean " << mean_of(nms_ms) << " p50 " << percentile(nms_ms, 0.50) << " p95 "
         << percentile(nms_ms, 0.95) << "\n";
  record << "decode mean " << mean_of(decode_ms) << " p50 " << percentile(decode_ms, 0.50) << " p95 "
         << percentile(decode_ms, 0.95) << "\n";
  record << "total mean " << mean_of(total_ms) << " p50 " << percentile(total_ms, 0.50) << " p95 "
         << percentile(total_ms, 0.95) << "\n";
  cudaFree(device_pixels);
}
#endif
#endif
