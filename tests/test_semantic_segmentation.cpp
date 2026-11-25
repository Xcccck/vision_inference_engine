#include "onnx_test_writer.hpp"
#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/decoder/decoder_registry.hpp"
#include "vie/decoder/semantic_segmentation.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/deeplabv3_semantic_segmentation_recognizer.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using vie::DataType;
using vie::DeviceContext;
using vie::DeviceType;
using vie::GeometryTransform;
using vie::Tensor;

constexpr std::size_t kFloat = sizeof(float);

Tensor logits_tensor(std::vector<std::int64_t> shape, std::vector<float> data) {
  return Tensor::from_host(DeviceContext{}, DataType::Fp32, std::move(shape), data.data(),
                           data.size() * kFloat);
}

GeometryTransform identity_geometry(int w, int h) {
  GeometryTransform geometry;
  geometry.original_width = w;
  geometry.original_height = h;
  geometry.resized_width = w;
  geometry.resized_height = h;
  geometry.output_width = w;
  geometry.output_height = h;
  geometry.scale_x = 1.0;
  geometry.scale_y = 1.0;
  geometry.uniform_scale = 1.0;
  return geometry;
}

std::vector<float> make_ramp(int channels, int height, int width) {
  std::vector<float> data(static_cast<std::size_t>(channels) * height * width, 0.f);
  for (int c = 0; c < channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int winner = (x + y) % channels;
        const std::size_t index =
            static_cast<std::size_t>(c) * height * width + static_cast<std::size_t>(y) * width + x;
        data[index] = c == winner ? 1.f : 0.f;
      }
    }
  }
  return data;
}

int expected_winner(int channels, int x, int y) { return (x + y) % channels; }

}  // namespace

// ---- §12.1 decoder data structure and validation ----

TEST(SemanticDecoder, ProducesClassMapOnIdentityGrid) {
  const int c = 3;
  const int h = 4;
  const int w = 5;
  vie::DecoderInput input;
  input.outputs.emplace("out", logits_tensor({1, c, h, w}, make_ramp(c, h, w)));
  input.geometry = identity_geometry(w, h);
  vie::SemanticSegmentationDecodeRequest request;
  request.output_name = "out";
  request.num_classes = c;
  request.channel_axis = 1;
  request.logits_semantics = "per_pixel_class_logits";

  const vie::SemanticSegmentationBatch batch =
      vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(input, request,
                                                                      DeviceContext(DeviceType::Cpu, 0));
  ASSERT_EQ(batch.results.size(), 1u);
  const vie::SemanticSegmentationResult& result = batch.results.front();
  EXPECT_EQ(result.protocol, "deeplabv3_semantic_seg_raw");
  EXPECT_EQ(result.width, w);
  EXPECT_EQ(result.height, h);
  EXPECT_EQ(result.original_width, w);
  EXPECT_EQ(result.original_height, h);
  EXPECT_EQ(result.num_classes, c);
  EXPECT_EQ(result.execution_device, DeviceType::Cpu);
  EXPECT_EQ(result.class_ids.size(), static_cast<std::size_t>(w) * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      EXPECT_EQ(result.class_ids[static_cast<std::size_t>(y) * w + x], expected_winner(c, x, y));
    }
  }
}

TEST(SemanticDecoder, RejectsCpuDeviceRequestOnCuda) {
  vie::DecoderInput input;
  input.outputs.emplace("out", logits_tensor({1, 2, 2, 2}, std::vector<float>(8, 0.f)));
  input.geometry = identity_geometry(2, 2);
  vie::SemanticSegmentationDecodeRequest request;
  request.output_name = "out";
  request.num_classes = 2;
  request.logits_semantics = "per_pixel_class_logits";
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                       input, request, DeviceContext(DeviceType::Cuda, 0)));
}

TEST(SemanticDecoder, RejectsNonFiniteLogits) {
  const std::vector<float> nan_data{1.f, std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f, 1.f, 0.f};
  const std::vector<float> inf_data{1.f, std::numeric_limits<float>::infinity(), 0.f, 0.f, 1.f, 0.f};
  for (const std::vector<float>& data : {nan_data, inf_data}) {
    vie::DecoderInput input;
    input.outputs.emplace("out", logits_tensor({1, 2, 1, 3}, data));
    input.geometry = identity_geometry(3, 1);
    vie::SemanticSegmentationDecodeRequest request;
    request.output_name = "out";
    request.num_classes = 2;
    request.logits_semantics = "per_pixel_class_logits";
    VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                     vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                         input, request, DeviceContext(DeviceType::Cpu, 0)));
  }
}

TEST(SemanticDecoder, RejectsBadRankDtypeBatchAndChannelMismatch) {
  vie::SemanticSegmentationDecodeRequest request;
  request.output_name = "out";
  request.num_classes = 2;
  request.logits_semantics = "per_pixel_class_logits";

  // rank 3
  {
    vie::DecoderInput input;
    input.outputs.emplace("out", logits_tensor({1, 2, 4}, std::vector<float>(8, 0.f)));
    input.geometry = identity_geometry(2, 2);
    VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                     vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                         input, request, DeviceContext(DeviceType::Cpu, 0)));
  }
  // batch != 1
  {
    vie::DecoderInput input;
    input.outputs.emplace("out", logits_tensor({2, 2, 2, 2}, std::vector<float>(16, 0.f)));
    input.geometry = identity_geometry(2, 2);
    VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                     vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                         input, request, DeviceContext(DeviceType::Cpu, 0)));
  }
  // channel != num_classes
  {
    vie::DecoderInput input;
    input.outputs.emplace("out", logits_tensor({1, 3, 2, 2}, std::vector<float>(12, 0.f)));
    input.geometry = identity_geometry(2, 2);
    VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                     vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                         input, request, DeviceContext(DeviceType::Cpu, 0)));
  }
  // wrong dtype (u8)
  {
    vie::DecoderInput input;
    input.outputs.emplace("out",
                          Tensor::zeros(DeviceContext{}, DataType::Uint8, {1, 2, 2, 2}));
    input.geometry = identity_geometry(2, 2);
    VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                     vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                         input, request, DeviceContext(DeviceType::Cpu, 0)));
  }
}

TEST(SemanticDecoder, RejectsMissingFieldsAndGeometry) {
  vie::DecoderInput input;
  input.outputs.emplace("out", logits_tensor({1, 2, 2, 2}, std::vector<float>(8, 0.f)));
  input.geometry = identity_geometry(2, 2);

  vie::SemanticSegmentationDecodeRequest no_output;
  no_output.num_classes = 2;
  no_output.logits_semantics = "per_pixel_class_logits";
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                       input, no_output, DeviceContext(DeviceType::Cpu, 0)));

  vie::SemanticSegmentationDecodeRequest few_classes;
  few_classes.output_name = "out";
  few_classes.num_classes = 1;
  few_classes.logits_semantics = "per_pixel_class_logits";
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                       input, few_classes, DeviceContext(DeviceType::Cpu, 0)));

  vie::SemanticSegmentationDecodeRequest wrong_semantics;
  wrong_semantics.output_name = "out";
  wrong_semantics.num_classes = 2;
  wrong_semantics.logits_semantics = "probability_map";
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                       input, wrong_semantics, DeviceContext(DeviceType::Cpu, 0)));

  vie::DecoderInput no_geometry;
  no_geometry.outputs.emplace("out", logits_tensor({1, 2, 2, 2}, std::vector<float>(8, 0.f)));
  vie::SemanticSegmentationDecodeRequest ok;
  ok.output_name = "out";
  ok.num_classes = 2;
  ok.logits_semantics = "per_pixel_class_logits";
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                       no_geometry, ok, DeviceContext(DeviceType::Cpu, 0)));
}

TEST(SemanticDecoder, TieKeepsSmallestClassIndex) {
  // Pixel 0: classes {5,5,5} -> 0. Pixel 1: {0,5,5} -> 1.
  vie::DecoderInput input;
  input.outputs.emplace(
      "out", logits_tensor({1, 3, 1, 2}, {5.f, 0.f, 5.f, 5.f, 5.f, 5.f}));
  input.geometry = identity_geometry(2, 1);
  vie::SemanticSegmentationDecodeRequest request;
  request.output_name = "out";
  request.num_classes = 3;
  request.logits_semantics = "per_pixel_class_logits";
  const vie::SemanticSegmentationBatch batch =
      vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(input, request,
                                                                      DeviceContext(DeviceType::Cpu, 0));
  ASSERT_EQ(batch.results.size(), 1u);
  EXPECT_EQ(batch.results.front().class_ids[0], 0);
  EXPECT_EQ(batch.results.front().class_ids[1], 1);
}

// ---- §12.2 geometry / coordinate recovery ----

namespace {

vie::SemanticSegmentationResult decode_with_geometry(vie::DecoderInput input,
                                                     const vie::GeometryTransform& geometry, int classes) {
  input.geometry = geometry;
  vie::SemanticSegmentationDecodeRequest request;
  request.output_name = "out";
  request.num_classes = classes;
  request.channel_axis = 1;
  request.logits_semantics = "per_pixel_class_logits";
  request.output_interpolation = "bilinear";
  request.class_id_interpolation = "nearest";
  vie::SemanticSegmentationBatch batch = vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
      input, request, DeviceContext(DeviceType::Cpu, 0));
  return std::move(batch.results.front());
}

}  // namespace

TEST(SemanticGeometry, LandscapeAndPortraitRecoverOriginalSize) {
  for (const std::pair<int, int>& wh : {std::make_pair(10, 3), std::make_pair(3, 10)}) {
    const int w = wh.first;
    const int h = wh.second;
    const int c = 2;
    vie::DecoderInput input;
    input.outputs.emplace("out", logits_tensor({1, c, h, w}, make_ramp(c, h, w)));
    const vie::SemanticSegmentationResult result = decode_with_geometry(input, identity_geometry(w, h), c);
    EXPECT_EQ(result.width, w);
    EXPECT_EQ(result.height, h);
    EXPECT_EQ(result.class_ids.size(), static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        EXPECT_EQ(result.class_ids[static_cast<std::size_t>(y) * w + x], expected_winner(c, x, y));
      }
    }
  }
}

TEST(SemanticGeometry, StretchScaleMapsOriginalToModelGrid) {
  // Original 3x3, model-input grid 6x6 (scale 2, origin 0). Logits already at grid.
  const int grid = 6;
  const int c = 4;
  vie::DecoderInput input;
  input.outputs.emplace("out", logits_tensor({1, c, grid, grid}, make_ramp(c, grid, grid)));
  GeometryTransform geometry = identity_geometry(3, 3);
  geometry.output_width = grid;
  geometry.output_height = grid;
  geometry.resized_width = grid;
  geometry.resized_height = grid;
  geometry.scale_x = 2.0;
  geometry.scale_y = 2.0;
  const vie::SemanticSegmentationResult result = decode_with_geometry(input, geometry, c);
  EXPECT_EQ(result.width, 3);
  EXPECT_EQ(result.height, 3);
  ASSERT_EQ(result.class_ids.size(), 9u);
  for (int oy = 0; oy < 3; ++oy) {
    for (int ox = 0; ox < 3; ++ox) {
      const int mx = ox * 2;
      const int my = oy * 2;
      EXPECT_EQ(result.class_ids[static_cast<std::size_t>(oy) * 3 + ox], expected_winner(c, mx, my));
    }
  }
}

TEST(SemanticGeometry, LetterboxPaddingNeverPollutes) {
  // Model-input grid 4x4, content occupies x in [1,2], y in [1,2] (pad 1 each side).
  // Original 2x2, scale 1, origin (1,1). Padding pixels carry an out-of-content class
  // that must never appear in the result.
  const int grid = 4;
  const int c = 3;
  std::vector<float> data(static_cast<std::size_t>(c) * grid * grid, 0.f);
  auto at = [&](int ch, int y, int x) -> float& {
    return data[static_cast<std::size_t>(ch) * grid * grid + static_cast<std::size_t>(y) * grid + x];
  };
  // Padding ring is dominated by class 2 (the polluting class).
  for (int i = 0; i < grid; ++i) {
    at(2, 0, i) = 10.f;
    at(2, grid - 1, i) = 10.f;
    at(2, i, 0) = 10.f;
    at(2, i, grid - 1) = 10.f;
  }
  // Content region uses a two-class ramp (classes 0/1 only) so class 2 is the
  // polluting class that can only come from the padding ring.
  for (int y = 1; y <= 2; ++y) {
    for (int x = 1; x <= 2; ++x) {
      const int winner = (x + y) % 2;
      for (int ch = 0; ch < c; ++ch) at(ch, y, x) = ch == winner ? 5.f : 0.f;
    }
  }
  vie::DecoderInput input;
  input.outputs.emplace("out", logits_tensor({1, c, grid, grid}, data));
  GeometryTransform geometry = identity_geometry(2, 2);
  geometry.output_width = grid;
  geometry.output_height = grid;
  geometry.resized_width = grid;
  geometry.resized_height = grid;
  geometry.scale_x = 1.0;
  geometry.scale_y = 1.0;
  geometry.pad_left = 1;
  geometry.pad_top = 1;
  geometry.pad_right = 1;
  geometry.pad_bottom = 1;
  const vie::SemanticSegmentationResult result = decode_with_geometry(input, geometry, c);
  EXPECT_EQ(result.width, 2);
  EXPECT_EQ(result.height, 2);
  ASSERT_EQ(result.class_ids.size(), 4u);
  // Sampled model coords: ox {0,1} -> mx = ox + 1 = {1,2}. Same for y.
  for (int oy = 0; oy < 2; ++oy) {
    for (int ox = 0; ox < 2; ++ox) {
      const int mx = ox + 1;
      const int my = oy + 1;
      EXPECT_EQ(result.class_ids[static_cast<std::size_t>(oy) * 2 + ox], (mx + my) % 2);
      EXPECT_NE(result.class_ids[static_cast<std::size_t>(oy) * 2 + ox], 2) << "padding leaked into output";
    }
  }
}

TEST(SemanticGeometry, OddPaddingAndNonIntegerScaleKeepSizeAndRange) {
  const int grid = 7;
  const int c = 3;
  vie::DecoderInput input;
  input.outputs.emplace("out", logits_tensor({1, c, grid, grid}, make_ramp(c, grid, grid)));
  GeometryTransform geometry = identity_geometry(3, 2);
  geometry.output_width = grid;
  geometry.output_height = grid;
  geometry.resized_width = grid;
  geometry.resized_height = grid;
  geometry.scale_x = 1.7;
  geometry.scale_y = 2.3;
  geometry.pad_left = 1;  // odd padding
  geometry.pad_top = 3;
  const vie::SemanticSegmentationResult result = decode_with_geometry(input, geometry, c);
  EXPECT_EQ(result.width, 3);
  EXPECT_EQ(result.height, 2);
  ASSERT_EQ(result.class_ids.size(), 6u);
  for (const std::int32_t id : result.class_ids) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, c);
  }
}

TEST(SemanticGeometry, LowResolutionLogitsResizeToGrid) {
  // Logits 2x2 uniformly class 1; model-input grid and original 4x4. Exercises the
  // bilinear logits-resize path and must not introduce other classes.
  const int c = 2;
  std::vector<float> data(static_cast<std::size_t>(c) * 2 * 2, 0.f);
  auto at = [&](int ch, int y, int x) -> float& {
    return data[static_cast<std::size_t>(ch) * 2 * 2 + static_cast<std::size_t>(y) * 2 + x];
  };
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
      at(1, y, x) = 1.f;
    }
  }
  vie::DecoderInput input;
  input.outputs.emplace("out", logits_tensor({1, c, 2, 2}, data));
  const vie::SemanticSegmentationResult result = decode_with_geometry(input, identity_geometry(4, 4), c);
  EXPECT_EQ(result.logits_width, 2);
  EXPECT_EQ(result.logits_height, 2);
  EXPECT_EQ(result.width, 4);
  EXPECT_EQ(result.height, 4);
  ASSERT_EQ(result.class_ids.size(), 16u);
  for (const std::int32_t id : result.class_ids) EXPECT_EQ(id, 1);
}

// ---- §12.3 recognizer and manifest ----

namespace {

OnnxTestWriter wrap_model(const OnnxTestWriter& graph) {
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 18));
  return model;
}

// A DeepLabV3-shaped dense graph: Conv backbone -> 1x1 class Conv -> Resize ->
// out. head selects a post-processing op so the negative cases can force the
// recognizer to decline.
OnnxTestWriter seg_graph(int classes, int grid, int in_channels, const std::string& head,
                         bool kernel1x1) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Conv", {"image", "bw"}, {"feat"}));
  graph.message(1, node_message("Conv", {"feat", "cw"}, {"low"}));
  if (head == "argmax") {
    graph.message(1, node_message("ArgMax", {"low"}, {"out"}));
  } else if (head == "softmax") {
    graph.message(1, node_message("Softmax", {"low"}, {"prob"}));
    graph.message(1, node_message("Resize", {"prob"}, {"out"}));
  } else if (head == "identity") {
    graph.message(1, node_message("Identity", {"low"}, {"out"}));
  } else {
    graph.message(1, node_message("Resize", {"low"}, {"out"}));
  }
  const int kk = kernel1x1 ? 1 : 3;
  graph.message(5, tensor_message("bw", 1, {8, in_channels, 3, 3}, {}, "", false));
  graph.message(5, tensor_message("cw", 1, {classes, 8, kk, kk}, {}, "", false));
  graph.message(11, value_info("image", tensor_type(1, true, {static_dim(1), static_dim(in_channels),
                                                              static_dim(grid), static_dim(grid)})));
  graph.message(12, value_info("out", tensor_type(1, true, {static_dim(1), static_dim(classes),
                                                            static_dim(grid), static_dim(grid)})));
  return wrap_model(graph);
}

vie::RecognitionResult recognize(const OnnxTestWriter& model) {
  const vie::ModelAnalysisReport report =
      vie::OnnxAnalyzer{}.analyze_bytes(model.str(), "not_a_filename_clue.onnx", ".");
  return vie::default_recognizer_registry().recognize(report);
}

bool has_evidence(const vie::RecognitionCandidate& candidate, const std::string& code) {
  for (const vie::RecognitionEvidence& evidence : candidate.evidence) {
    if (evidence.code == code && !evidence.detail.empty()) return true;
  }
  return false;
}

}  // namespace

TEST(SemanticRecognizer, DenseLogitGraphIsRecognized) {
  const vie::RecognitionResult result = recognize(seg_graph(21, 32, 3, "resize", true));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "deeplabv3_semantic_segmentation");
  EXPECT_EQ(result.selected()->decoder.protocol, "deeplabv3_semantic_seg_raw");
  EXPECT_EQ(result.selected()->decoder.output_name, "out");
  EXPECT_EQ(result.selected()->decoder.num_classes, 21);
  EXPECT_EQ(result.selected()->decoder.channel_axis, 1);
  EXPECT_TRUE(has_evidence(*result.selected(), "raw_logits_head"));
  EXPECT_TRUE(has_evidence(*result.selected(), "terminal_1x1_conv"));
  for (const vie::RecognitionEvidence& evidence : result.selected()->evidence) {
    EXPECT_EQ(evidence.detail.find("not_a_filename_clue"), std::string::npos);
  }
}

TEST(SemanticRecognizer, IdentityHeadIsStillLogits) {
  const vie::RecognitionResult result = recognize(seg_graph(5, 16, 3, "identity", true));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "deeplabv3_semantic_segmentation");
}

TEST(SemanticRecognizer, DeclinesPostProcessedAndWrongShapes) {
  // ArgMax head is not raw logits.
  EXPECT_NE(recognize(seg_graph(4, 16, 3, "argmax", true)).status, vie::RecognitionStatus::Recognized);
  // Softmax in the path.
  EXPECT_NE(recognize(seg_graph(4, 16, 3, "softmax", true)).status, vie::RecognitionStatus::Recognized);
  // 3x3 class conv, not a 1x1 dense projection.
  EXPECT_NE(recognize(seg_graph(4, 16, 3, "resize", false)).status, vie::RecognitionStatus::Recognized);
  // Non-3-channel input.
  EXPECT_NE(recognize(seg_graph(4, 16, 4, "resize", true)).status, vie::RecognitionStatus::Recognized);
  // Single class is not a segmentation task.
  EXPECT_NE(recognize(seg_graph(1, 16, 3, "resize", true)).status, vie::RecognitionStatus::Recognized);
}

TEST(SemanticManifest, GeneratorFillsTaskAndHead) {
  const OnnxTestWriter model = seg_graph(21, 32, 3, "resize", true);
  const vie::ModelAnalysisReport report =
      vie::OnnxAnalyzer{}.analyze_bytes(model.str(), "deeplabv3.onnx", ".");
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  ASSERT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  const vie::ModelManifest manifest =
      vie::ManifestGenerator{}.generate(report, result, "deeplabv3.onnx");
  EXPECT_EQ(manifest.task, vie::TaskType::SemanticSegmentation);
  EXPECT_EQ(manifest.num_classes, 21);
  EXPECT_EQ(manifest.decoder.protocol, "deeplabv3_semantic_seg_raw");
  EXPECT_EQ(manifest.semantic_segmentation.output_name, "out");
  ASSERT_TRUE(manifest.semantic_segmentation.channel_axis.has_value());
  EXPECT_EQ(*manifest.semantic_segmentation.channel_axis, 1);
  EXPECT_EQ(manifest.semantic_segmentation.logits_semantics, "per_pixel_class_logits");
  EXPECT_EQ(manifest.semantic_segmentation.output_interpolation, "bilinear");
  EXPECT_EQ(manifest.semantic_segmentation.class_id_interpolation, "nearest");

  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok());
  // The generated manifest is structurally valid but intentionally not fully
  // runnable until the user supplies preprocessing provenance (color,
  // normalization, resize). A fully-provenanced manifest loads and runs, which
  // the SemanticEngine.MockCpuEndToEndRestoresOriginalSize test proves.
  EXPECT_FALSE(validation.runnable());

  const std::string text = manifest.to_json_string();
  const vie::ModelManifest round_trip = vie::ModelManifest::from_json_string(text);
  EXPECT_EQ(round_trip.task, vie::TaskType::SemanticSegmentation);
  EXPECT_EQ(round_trip.semantic_segmentation.logits_semantics, "per_pixel_class_logits");
  EXPECT_EQ(round_trip.semantic_segmentation.class_id_interpolation, "nearest");
}

TEST(SemanticManifest, ValidatorRejectsCrossTaskReuse) {
  vie::ModelManifest base;
  base.model_name = "m";
  base.model_format = vie::ModelFormat::Onnx;
  base.task = vie::TaskType::SemanticSegmentation;
  base.num_classes = 4;
  base.input.name = "image";
  base.input.shape = {1, 3, 16, 16};
  base.input.dtype = vie::DataType::Fp32;
  base.input.layout = vie::Layout::Nchw;
  base.decoder.protocol = "deeplabv3_semantic_seg_raw";
  vie::OutputDesc out;
  out.name = "logits";
  out.shape = {1, 4, 16, 16};
  out.dtype = vie::DataType::Fp32;
  base.outputs = {out};
  base.semantic_segmentation.output_name = "logits";
  base.semantic_segmentation.channel_axis = 1;
  base.semantic_segmentation.logits_semantics = "per_pixel_class_logits";
  base.semantic_segmentation.output_interpolation = "bilinear";
  base.semantic_segmentation.class_id_interpolation = "nearest";

  EXPECT_TRUE(vie::ManifestValidator{}.validate(base).schema_ok());

  // Instance-segmentation config cannot describe a semantic model.
  {
    vie::ModelManifest bad = base;
    bad.segmentation.prototype_name = "logits";
    EXPECT_FALSE(vie::ManifestValidator{}.validate(bad).schema_ok());
  }
  // channel_axis must be 1.
  {
    vie::ModelManifest bad = base;
    bad.semantic_segmentation.channel_axis = 3;
    EXPECT_FALSE(vie::ManifestValidator{}.validate(bad).schema_ok());
  }
  // num_classes must exceed 1.
  {
    vie::ModelManifest bad = base;
    bad.num_classes = 1;
    EXPECT_FALSE(vie::ManifestValidator{}.validate(bad).schema_ok());
  }
  // Semantic fields on a non-semantic task are rejected.
  {
    vie::ModelManifest bad = base;
    bad.task = vie::TaskType::Detection;
    bad.decoder.protocol = "yolov8_detect_raw";
    EXPECT_FALSE(vie::ManifestValidator{}.validate(bad).schema_ok());
  }
}

// ---- §12.4 engine dispatch (CPU-only, mock backend) ----

namespace {

vie::ModelManifest mock_semantic_manifest() {
  vie::ModelManifest manifest;
  manifest.model_name = "mock_semantic";
  manifest.task = vie::TaskType::SemanticSegmentation;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 4;
  manifest.input.name = "image";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  manifest.input.interpolation = "bilinear";
  manifest.decoder.protocol = "deeplabv3_semantic_seg_raw";
  vie::OutputDesc prediction;
  prediction.name = "logits";
  prediction.shape = {1, 4, 2, 2};
  prediction.dtype = vie::DataType::Fp32;
  prediction.layout = vie::Layout::Nchw;
  prediction.semantic = "semantic_logits";
  manifest.outputs = {prediction};
  manifest.semantic_segmentation.output_name = "logits";
  manifest.semantic_segmentation.channel_axis = 1;
  manifest.semantic_segmentation.logits_semantics = "per_pixel_class_logits";
  manifest.semantic_segmentation.output_interpolation = "bilinear";
  manifest.semantic_segmentation.class_id_interpolation = "nearest";
  manifest.ensure_provenance();
  return manifest;
}

vie::ImageInput solid_image(std::vector<std::uint8_t>& pixels, int w, int h) {
  pixels.assign(static_cast<std::size_t>(w) * h * 3, 128);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = w;
  image.height = h;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

}  // namespace

TEST(SemanticEngine, MockCpuEndToEndRestoresOriginalSize) {
  vie::InferenceEngine engine;
  vie::ModelConfig config;
  config.manifest = mock_semantic_manifest();
  engine.load(config);
  ASSERT_TRUE(engine.loaded());

  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = solid_image(pixels, 4, 4);
  const vie::SemanticSegmentationResult result = engine.infer_semantic_segmentation(image);
  EXPECT_EQ(result.width, 4);
  EXPECT_EQ(result.height, 4);
  EXPECT_EQ(result.num_classes, 4);
  EXPECT_EQ(result.execution_device, vie::DeviceType::Cpu);
  ASSERT_EQ(result.class_ids.size(), 16u);
  for (const std::int32_t id : result.class_ids) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, 4);
  }
}

TEST(SemanticEngine, DetectionEntryPointRejectsSemanticTask) {
  vie::InferenceEngine engine;
  vie::ModelConfig config;
  config.manifest = mock_semantic_manifest();
  engine.load(config);
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = solid_image(pixels, 4, 4);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(image));
}

TEST(SemanticEngine, UnloadedEngineRejectsSemanticCall) {
  vie::InferenceEngine engine;
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = solid_image(pixels, 4, 4);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_semantic_segmentation(image));
}
