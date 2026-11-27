#include "onnx_test_writer.hpp"
#include "test_util.hpp"
#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/decoder/detection.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/recognizer/yolov8_seg_recognizer.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

OnnxTestWriter float_weight(std::string_view name, const std::vector<std::int64_t>& dims) {
  return tensor_message(name, 1, dims, {}, std::string(4, '\0'), true);
}

OnnxTestWriter value(std::string_view name, const std::vector<std::int64_t>& dims) {
  std::vector<TestDim> shape;
  shape.reserve(dims.size());
  for (const std::int64_t dim : dims) shape.push_back(static_dim(dim));
  return value_info(name, tensor_type(1, true, shape));
}

struct SegGraphOptions {
  int class_channels = 80;
  int mask_channels = 32;
  int proto_channels = 32;
  int prediction_channels = 116;
  bool with_dfl = true;
  bool sigmoid_on_coefficients = false;
  bool with_prototype = true;
  bool identity = false;
  std::string names;
};

OnnxTestWriter segmentation_model(const SegGraphOptions& options) {
  OnnxTestWriter graph;
  graph.text(2, "yolov8n-seg");
  if (options.identity) {
    graph.message(1, node_message("Identity", {"images"}, {"output0"}));
    graph.message(1, node_message("Identity", {"images"}, {"output1"}));
  } else {
    graph.message(5, float_weight("w_dfl", {1, 16, 1, 1}));
    graph.message(5, float_weight("w_cls0", {options.class_channels, 8, 1, 1}));
    graph.message(5, float_weight("w_cls1", {options.class_channels, 8, 1, 1}));
    graph.message(5, float_weight("w_cls2", {options.class_channels, 8, 1, 1}));
    graph.message(5, float_weight("w_mask0", {options.mask_channels, 8, 1, 1}));
    graph.message(5, float_weight("w_mask1", {options.mask_channels, 8, 1, 1}));
    graph.message(5, float_weight("w_mask2", {options.mask_channels, 8, 1, 1}));
    graph.message(5, float_weight("w_proto", {options.proto_channels, 8, 1, 1}));
    graph.message(5, float_weight("w_up", {8, 8, 2, 2}));
    graph.message(5, float_weight("shape4", {3}));

    if (options.with_dfl) {
      graph.message(1, node_message("Softmax", {"dfl_in"}, {"dfl_sm"}));
      graph.message(1, node_message("Conv", {"dfl_sm", "w_dfl"}, {"dfl_out"}));
    } else {
      graph.message(1, node_message("Constant", {}, {"dfl_out"}));
    }
    graph.message(1, node_message("Constant", {}, {"stride"}));
    graph.message(1, node_message("Constant", {}, {"half"}));
    graph.message(1, node_message("Div", {"dfl_out", "half"}, {"div_out"}));
    graph.message(1, node_message("Sub", {"half", "dfl_out"}, {"sub_out"}));
    graph.message(1, node_message("Concat", {"div_out", "sub_out"}, {"decoded"}));
    graph.message(1, node_message("Mul", {"decoded", "stride"}, {"box"}));

    graph.message(1, node_message("Conv", {"images", "w_cls0"}, {"cls0"}));
    graph.message(1, node_message("Conv", {"images", "w_cls1"}, {"cls1"}));
    graph.message(1, node_message("Conv", {"images", "w_cls2"}, {"cls2"}));
    graph.message(1, node_message("Reshape", {"cls0", "shape4"}, {"rcls0"}));
    graph.message(1, node_message("Reshape", {"cls1", "shape4"}, {"rcls1"}));
    graph.message(1, node_message("Reshape", {"cls2", "shape4"}, {"rcls2"}));
    graph.message(1, node_message("Concat", {"rcls0", "rcls1", "rcls2"}, {"class_cat"}));
    graph.message(1, node_message("Sigmoid", {"class_cat"}, {"class_sig"}));

    graph.message(1, node_message("Conv", {"images", "w_mask0"}, {"mask0"}));
    graph.message(1, node_message("Conv", {"images", "w_mask1"}, {"mask1"}));
    graph.message(1, node_message("Conv", {"images", "w_mask2"}, {"mask2"}));
    graph.message(1, node_message("Reshape", {"mask0", "shape4"}, {"rmask0"}));
    graph.message(1, node_message("Reshape", {"mask1", "shape4"}, {"rmask1"}));
    graph.message(1, node_message("Reshape", {"mask2", "shape4"}, {"rmask2"}));
    graph.message(1, node_message("Concat", {"rmask0", "rmask1", "rmask2"}, {"mask_cat"}));
    const std::string coefficients = options.sigmoid_on_coefficients ? "mask_sig" : "mask_cat";
    if (options.sigmoid_on_coefficients) {
      graph.message(1, node_message("Sigmoid", {"mask_cat"}, {"mask_sig"}));
    }
    graph.message(1, node_message("Concat", {"box", "class_sig", coefficients}, {"output0"}));

    if (options.with_prototype) {
      graph.message(1, node_message("ConvTranspose", {"images", "w_up"}, {"up"}));
      graph.message(1, node_message("Conv", {"up", "w_proto"}, {"proto_conv"}));
      graph.message(1, node_message("Sigmoid", {"proto_conv"}, {"proto_sig"}));
      graph.message(1, node_message("Mul", {"proto_conv", "proto_sig"}, {"output1"}));
    } else {
      graph.message(1, node_message("Identity", {"images"}, {"output1"}));
    }
  }

  graph.message(11, value("images", {1, 3, 64, 64}));
  graph.message(12, value("output0", {1, options.prediction_channels, 100}));
  if (!options.identity || true) {
    graph.message(12, value("output1", {1, options.proto_channels, 16, 16}));
  }
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  if (!options.names.empty()) model.message(14, string_entry("names", options.names));
  return model;
}

vie::ModelAnalysisReport analyze(const OnnxTestWriter& model) {
  return vie::OnnxAnalyzer{}.analyze_bytes(model.str(), "synthetic.onnx", ".");
}

vie::RecognitionResult recognize(const vie::ModelAnalysisReport& report) {
  return vie::default_recognizer_registry().recognize(report);
}

}  // namespace

TEST(YoloV8Seg, RealExportIsRecognizedFromGraphNotFilename) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n-seg.onnx");
  const std::string path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const vie::ModelAnalysisReport report =
      vie::OnnxAnalyzer{}.analyze_bytes(buffer.str(), "renamed_not_a_clue.bin", ".");
  const vie::RecognitionResult result = recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_segmentation");
  EXPECT_EQ(result.selected()->family, "yolov8");
  EXPECT_EQ(result.selected()->decoder.protocol, "yolov8_seg_raw");
  EXPECT_EQ(result.selected()->decoder.output_name, "output0");
  EXPECT_EQ(result.selected()->decoder.prototype_name, "output1");
  ASSERT_TRUE(result.selected()->decoder.num_classes.has_value());
  EXPECT_EQ(*result.selected()->decoder.num_classes, 80);
  ASSERT_TRUE(result.selected()->decoder.mask_channels.has_value());
  EXPECT_EQ(*result.selected()->decoder.mask_channels, 32);
  EXPECT_EQ(result.selected()->decoder.box_format, "cxcywh");
  EXPECT_TRUE(result.selected()->decoder.class_scores_are_probabilities);
  EXPECT_EQ(result.selected()->decoder.mask_coefficient_activation, "none");
  EXPECT_EQ(result.selected()->decoder.prototype_activation, "model_output_features");
  ASSERT_TRUE(result.selected()->decoder.has_objectness.has_value());
  EXPECT_FALSE(*result.selected()->decoder.has_objectness);
  EXPECT_EQ(result.selected()->decoder.prototype_channel_axis, 1);
  EXPECT_EQ(result.selected()->decoder.prototype_height, 160);
  EXPECT_EQ(result.selected()->decoder.prototype_width, 160);
  for (const vie::RecognitionEvidence& evidence : result.selected()->evidence) {
    EXPECT_EQ(evidence.detail.find("renamed_not_a_clue"), std::string::npos);
    EXPECT_EQ(evidence.detail.find("segment"), std::string::npos);
  }

  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
  EXPECT_EQ(manifest.task, vie::TaskType::Segmentation);
  EXPECT_EQ(manifest.model_family, "yolov8");
  EXPECT_EQ(manifest.decoder.protocol, "yolov8_seg_raw");
  EXPECT_EQ(manifest.decoder.box_format, "cxcywh");
  EXPECT_EQ(manifest.decoder.box_decode, "decoded_cxcywh");
  ASSERT_TRUE(manifest.decoder.scores_are_probabilities.has_value());
  EXPECT_TRUE(*manifest.decoder.scores_are_probabilities);
  ASSERT_TRUE(manifest.num_classes.has_value());
  EXPECT_EQ(*manifest.num_classes, 80);
  ASSERT_EQ(manifest.outputs.size(), 2u);
  EXPECT_EQ(manifest.outputs[0].name, "output0");
  EXPECT_EQ(manifest.outputs[0].shape, (std::vector<std::int64_t>{1, 116, 8400}));
  EXPECT_EQ(manifest.outputs[0].semantic, "cxcywh_class_probabilities_mask_coefficients");
  EXPECT_EQ(manifest.outputs[1].name, "output1");
  EXPECT_EQ(manifest.outputs[1].shape, (std::vector<std::int64_t>{1, 32, 160, 160}));
  EXPECT_EQ(manifest.outputs[1].semantic, "mask_prototypes");
  EXPECT_EQ(manifest.outputs[1].layout, vie::Layout::Nchw);
  ASSERT_TRUE(manifest.segmentation.mask_channels.has_value());
  EXPECT_EQ(*manifest.segmentation.mask_channels, 32);
  EXPECT_EQ(manifest.segmentation.prototype_name, "output1");
  EXPECT_EQ(manifest.segmentation.prototype_layout, vie::Layout::Nchw);
  EXPECT_EQ(manifest.segmentation.prototype_height, 160);
  EXPECT_EQ(manifest.segmentation.prototype_width, 160);
  EXPECT_EQ(manifest.segmentation.mask_coefficient_activation, "none");
  EXPECT_EQ(manifest.segmentation.prototype_activation, "model_output_features");
  ASSERT_TRUE(manifest.segmentation.has_objectness.has_value());
  EXPECT_FALSE(*manifest.segmentation.has_objectness);
  EXPECT_EQ(manifest.segmentation.class_names.size(), 80u);
  EXPECT_EQ(manifest.segmentation.class_names[0], "person");
  EXPECT_TRUE(manifest.input.color.empty());
  EXPECT_TRUE(manifest.input.normalization.empty());
  EXPECT_TRUE(manifest.input.resize.empty());
  EXPECT_TRUE(manifest.decoder.class_selection.empty());
  EXPECT_FALSE(manifest.nms.confidence_threshold.has_value());
  EXPECT_FALSE(manifest.segmentation.retina_masks.has_value());
  EXPECT_FALSE(manifest.segmentation.binary_mask_gt_zero.has_value());
  EXPECT_FALSE(manifest.segmentation.max_mask_bytes.has_value());

  const auto certainty = [&](const char* field) {
    const vie::FieldProvenance* record = manifest.find_provenance(field);
    EXPECT_NE(record, nullptr) << field;
    return record == nullptr ? vie::Certainty::User : record->certainty;
  };
  EXPECT_EQ(certainty("input.shape"), vie::Certainty::Observed);
  EXPECT_EQ(certainty("outputs.0.shape"), vie::Certainty::Observed);
  EXPECT_EQ(certainty("outputs.1.shape"), vie::Certainty::Observed);
  EXPECT_EQ(certainty("task"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("num_classes"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("segmentation.mask_channels"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("segmentation.class_names"), vie::Certainty::Metadata);
  EXPECT_EQ(certainty("input.color"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("decoder.class_selection"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("nms.confidence_threshold"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("segmentation.retina_masks"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("segmentation.binary_mask_gt_zero"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("segmentation.max_mask_bytes"), vie::Certainty::Unresolved);

  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(manifest.to_json_string(2));
  EXPECT_EQ(loaded.task, vie::TaskType::Segmentation);
  EXPECT_EQ(loaded.outputs[0].semantic, manifest.outputs[0].semantic);
  EXPECT_EQ(loaded.outputs[1].semantic, manifest.outputs[1].semantic);
  EXPECT_EQ(loaded.segmentation.class_names, manifest.segmentation.class_names);
  EXPECT_EQ(loaded.find_provenance("segmentation.retina_masks")->certainty, vie::Certainty::Unresolved);
  EXPECT_FALSE(loaded.segmentation.retina_masks.has_value());
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(loaded);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_FALSE(validation.runnable());
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, vie::decoder_config_from_manifest(loaded));
}

TEST(YoloV8Seg, DetectionModelStaysDetection) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
  const std::string path = vie_fixture("testdata/onnx/yolov8n.onnx");
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_bytes(buffer.str(), "detect.onnx", ".");
  const vie::RecognitionResult result = recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_detection");
  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
  EXPECT_EQ(manifest.task, vie::TaskType::Detection);
  EXPECT_EQ(manifest.decoder.protocol, "yolov8_detect_raw");
  EXPECT_FALSE(manifest.segmentation.mask_channels.has_value());
  EXPECT_EQ(manifest.to_json_string().find("\"segmentation\""), std::string::npos);
  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(manifest.to_json_string(2));
  EXPECT_EQ(loaded.task, vie::TaskType::Detection);
  EXPECT_EQ(loaded.outputs.size(), 1u);
  EXPECT_EQ(loaded.outputs[0].semantic, "cxcywh_then_class_scores");
}

TEST(YoloV8Seg, SameShapeIdentityIsUnknown) {
  SegGraphOptions options;
  options.identity = true;
  const vie::RecognitionResult result = recognize(analyze(segmentation_model(options)));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
}

TEST(YoloV8Seg, MissingPrototypeIsUnknown) {
  SegGraphOptions options;
  options.with_prototype = false;
  const vie::RecognitionResult result = recognize(analyze(segmentation_model(options)));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
}

TEST(YoloV8Seg, PrototypeChannelMismatchIsUnknown) {
  SegGraphOptions options;
  options.proto_channels = 16;
  const vie::RecognitionResult result = recognize(analyze(segmentation_model(options)));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
}

TEST(YoloV8Seg, ClassChannelMismatchIsUnknown) {
  SegGraphOptions options;
  options.class_channels = 79;
  const vie::RecognitionResult result = recognize(analyze(segmentation_model(options)));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
}

TEST(YoloV8Seg, UnsupportedOutputShapeIsUnknown) {
  SegGraphOptions options;
  options.prediction_channels = 84;
  const vie::RecognitionResult result = recognize(analyze(segmentation_model(options)));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
}

TEST(YoloV8Seg, MissingDflEvidenceIsUnknown) {
  SegGraphOptions options;
  options.with_dfl = false;
  const vie::RecognitionResult result = recognize(analyze(segmentation_model(options)));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
}

TEST(YoloV8Seg, ActivatedCoefficientsAreUnknown) {
  SegGraphOptions options;
  options.sigmoid_on_coefficients = true;
  const vie::RecognitionResult result = recognize(analyze(segmentation_model(options)));
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
}

TEST(YoloV8Seg, ConflictingClassNamesStayUnresolved) {
  SegGraphOptions options;
  options.names = "{0: 'person', 1: 'bicycle'}";
  const vie::ModelAnalysisReport report = analyze(segmentation_model(options));
  const vie::RecognitionResult result = recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, "synthetic.onnx");
  EXPECT_TRUE(manifest.segmentation.class_names.empty());
  const vie::FieldProvenance* names = manifest.find_provenance("segmentation.class_names");
  ASSERT_NE(names, nullptr);
  EXPECT_EQ(names->certainty, vie::Certainty::Unresolved);
  EXPECT_NE(names->evidence.front().find("does not match"), std::string::npos);
  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(manifest.to_json_string());
  EXPECT_TRUE(loaded.segmentation.class_names.empty());
  EXPECT_EQ(loaded.find_provenance("segmentation.class_names")->certainty, vie::Certainty::Unresolved);
}

TEST(YoloV8Seg, SyntheticGraphIsRecognizedWithoutDefaults) {
  const vie::ModelAnalysisReport report = analyze(segmentation_model(SegGraphOptions{}));
  const vie::RecognitionResult result = recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(*result.selected()->decoder.num_classes, 80);
  EXPECT_EQ(*result.selected()->decoder.mask_channels, 32);
  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, "synthetic.onnx");
  EXPECT_TRUE(manifest.segmentation.class_names.empty());
  EXPECT_EQ(manifest.find_provenance("segmentation.class_names")->certainty, vie::Certainty::Unresolved);
  EXPECT_FALSE(manifest.nms.enabled.has_value());
  EXPECT_FALSE(manifest.segmentation.retina_masks.has_value());
}
