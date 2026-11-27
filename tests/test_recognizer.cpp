#include "onnx_test_writer.hpp"
#include "test_util.hpp"
#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/recognizer/yolov8_detect_recognizer.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <utility>

namespace {

OnnxTestWriter simple_model() {
  OnnxTestWriter graph;
  graph.message(1, node_message("Relu", {"x"}, {"y"}));
  graph.text(2, "simple");
  graph.message(11, value_info("x", tensor_type(1, true, {static_dim(1), static_dim(4)})));
  graph.message(12, value_info("y", tensor_type(1, true, {static_dim(1), static_dim(4)})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

vie::ModelAnalysisReport analyze_simple() {
  return vie::OnnxAnalyzer{}.analyze_bytes(simple_model().str(), "simple.onnx", ".");
}

class FixedCandidateRecognizer : public vie::IModelRecognizer {
 public:
  FixedCandidateRecognizer(std::string id, std::string architecture, int evidence_items)
      : id_(std::move(id)), architecture_(std::move(architecture)), evidence_items_(evidence_items) {}

  std::string id() const override { return id_; }

  std::optional<vie::RecognitionCandidate> consider(const vie::ModelAnalysisReport&) const override {
    vie::RecognitionCandidate candidate;
    candidate.recognizer_id = id_;
    candidate.architecture = architecture_;
    candidate.family = architecture_;
    for (int index = 0; index < evidence_items_; ++index) {
      candidate.evidence.push_back(
          {"evidence_" + std::to_string(index), id_ + " structural check " + std::to_string(index)});
    }
    candidate.decoder.protocol = id_ + "_protocol";
    return candidate;
  }

 private:
  std::string id_;
  std::string architecture_;
  int evidence_items_ = 0;
};

OnnxTestWriter shape_only_yolo_lookalike() {
  OnnxTestWriter graph;
  graph.message(1, node_message("Identity", {"images"}, {"predictions"}));
  graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(640),
                                                               static_dim(640)})));
  graph.message(12, value_info("predictions", tensor_type(1, true, {static_dim(1), static_dim(84),
                                                                    static_dim(8400)})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.text(2, "yolov8n.onnx");
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

}  // namespace

TEST(RecognizerRegistry, UnknownWhenNothingMatches) {
  vie::ModelRecognizerRegistry registry;
  registry.add(std::make_unique<vie::YoloV8DetectRecognizer>());
  const vie::ModelAnalysisReport report = analyze_simple();
  const vie::RecognitionResult result = registry.recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
  EXPECT_EQ(result.selected(), nullptr);
  EXPECT_TRUE(result.candidates.empty());
}

TEST(RecognizerRegistry, AmbiguousWhenEvidenceCountsTie) {
  vie::ModelRecognizerRegistry registry;
  registry.add(std::make_unique<FixedCandidateRecognizer>("alpha", "alpha_net", 3));
  registry.add(std::make_unique<FixedCandidateRecognizer>("beta", "beta_net", 3));
  const vie::ModelAnalysisReport report = analyze_simple();
  const vie::RecognitionResult result = registry.recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Ambiguous);
  EXPECT_EQ(result.selected(), nullptr);
  ASSERT_EQ(result.candidates.size(), 2u);
  EXPECT_EQ(result.candidates[0].evidence.size(), result.candidates[1].evidence.size());

  const vie::ModelManifest manifest =
      vie::ManifestGenerator{}.generate(report, result, "simple.onnx");
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok());
  EXPECT_FALSE(validation.runnable());
  EXPECT_EQ(manifest.task, vie::TaskType::Unknown);
  EXPECT_TRUE(manifest.model_family.empty());
  EXPECT_TRUE(manifest.input.color.empty());
}

TEST(RecognizerRegistry, HigherEvidenceCountWinsWithoutAScore) {
  vie::ModelRecognizerRegistry registry;
  registry.add(std::make_unique<FixedCandidateRecognizer>("alpha", "alpha_net", 2));
  registry.add(std::make_unique<FixedCandidateRecognizer>("beta", "beta_net", 4));
  const vie::ModelAnalysisReport report = analyze_simple();
  const vie::RecognitionResult result = registry.recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized);
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "beta_net");
  EXPECT_EQ(result.selected()->evidence.size(), 4u);
}

TEST(YoloV8Detect, RealExportIsRecognizedAndManifestStaysPartial) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
  const std::string path = vie_fixture("testdata/onnx/yolov8n.onnx");
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const vie::ModelAnalysisReport report =
      vie::OnnxAnalyzer{}.analyze_bytes(buffer.str(), "renamed_not_a_clue.bin", ".");
  vie::ModelRecognizerRegistry registry;
  registry.add(std::make_unique<vie::YoloV8DetectRecognizer>());
  const vie::RecognitionResult result = registry.recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_detection");
  EXPECT_EQ(result.selected()->family, "yolov8");
  EXPECT_EQ(result.selected()->decoder.protocol, "yolov8_detect_raw");
  EXPECT_EQ(result.selected()->decoder.box_format, "cxcywh");
  EXPECT_EQ(result.selected()->decoder.channel_order, "cxcywh_then_class_scores");
  EXPECT_EQ(result.selected()->decoder.channel_axis, 1);
  ASSERT_TRUE(result.selected()->decoder.num_classes.has_value());
  EXPECT_EQ(*result.selected()->decoder.num_classes, 80);
  EXPECT_GE(result.selected()->evidence.size(), 6u);
  for (const vie::RecognitionEvidence& evidence : result.selected()->evidence) {
    EXPECT_EQ(evidence.detail.find("renamed_not_a_clue"), std::string::npos);
    EXPECT_FALSE(evidence.detail.empty());
  }

  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_FALSE(validation.runnable());
  EXPECT_EQ(manifest.task, vie::TaskType::Detection);
  EXPECT_EQ(manifest.model_family, "yolov8");
  EXPECT_EQ(manifest.model_format, vie::ModelFormat::Onnx);
  EXPECT_EQ(manifest.input.name, "images");
  EXPECT_EQ(manifest.input.dtype, vie::DataType::Fp32);
  EXPECT_EQ(manifest.input.layout, vie::Layout::Nchw);
  EXPECT_EQ(manifest.input.shape, (std::vector<std::int64_t>{1, 3, 640, 640}));
  EXPECT_TRUE(manifest.input.color.empty());
  EXPECT_TRUE(manifest.input.normalization.empty());
  EXPECT_TRUE(manifest.input.resize.empty());
  ASSERT_EQ(manifest.outputs.size(), 1u);
  EXPECT_EQ(manifest.outputs[0].name, "predictions");
  EXPECT_EQ(manifest.outputs[0].shape, (std::vector<std::int64_t>{1, 84, 8400}));
  EXPECT_EQ(manifest.outputs[0].semantic, "cxcywh_then_class_scores");
  EXPECT_EQ(manifest.decoder.protocol, "yolov8_detect_raw");
  EXPECT_EQ(manifest.decoder.box_format, "cxcywh");
  EXPECT_EQ(manifest.decoder.box_decode, "decoded_cxcywh");
  ASSERT_TRUE(manifest.decoder.scores_are_probabilities.has_value());
  EXPECT_TRUE(*manifest.decoder.scores_are_probabilities);
  ASSERT_TRUE(manifest.decoder.box_channels.has_value());
  EXPECT_EQ(*manifest.decoder.box_channels, 4);
  ASSERT_TRUE(manifest.decoder.channel_axis.has_value());
  EXPECT_EQ(*manifest.decoder.channel_axis, 1);
  ASSERT_TRUE(manifest.decoder.reg_max.has_value());
  EXPECT_GE(*manifest.decoder.reg_max, 2);
  EXPECT_TRUE(manifest.decoder.class_selection.empty());
  ASSERT_TRUE(manifest.num_classes.has_value());
  EXPECT_EQ(*manifest.num_classes, 80);
  EXPECT_FALSE(manifest.nms.enabled.has_value());
  EXPECT_FALSE(manifest.nms.confidence_threshold.has_value());
  EXPECT_TRUE(manifest.model_version.empty());

  const auto certainty = [&](const char* field) {
    const vie::FieldProvenance* record = manifest.find_provenance(field);
    EXPECT_NE(record, nullptr) << field;
    return record == nullptr ? vie::Certainty::User : record->certainty;
  };
  EXPECT_EQ(certainty("task"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("model_family"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("num_classes"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("input.layout"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("decoder.protocol"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("decoder.box_format"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("input.name"), vie::Certainty::Observed);
  EXPECT_EQ(certainty("input.shape"), vie::Certainty::Observed);
  EXPECT_EQ(certainty("input.color"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("input.normalization"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("input.resize"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("nms.enabled"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("model_version"), vie::Certainty::Unresolved);
  EXPECT_FALSE(manifest.find_provenance("task")->evidence.empty());

  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(manifest.to_json_string(2));
  EXPECT_EQ(loaded.task, vie::TaskType::Detection);
  EXPECT_EQ(loaded.find_provenance("task")->certainty, vie::Certainty::Inferred);
  EXPECT_TRUE(loaded.input.color.empty());
}

TEST(YoloV8Detect, ShapeAndFilenameAloneAreNotAccepted) {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_bytes(
      shape_only_yolo_lookalike().str(), "yolov8n.onnx", ".");
  EXPECT_EQ(report.outputs[0].name, "predictions");
  ASSERT_EQ(report.outputs[0].type.declared_shape.dimensions.size(), 3u);
  EXPECT_EQ(report.outputs[0].type.declared_shape.dimensions[1].value, 84);
  EXPECT_EQ(report.outputs[0].type.declared_shape.dimensions[2].value, 8400);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown);
  EXPECT_EQ(result.selected(), nullptr);
}
