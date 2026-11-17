#include "onnx_test_writer.hpp"
#include "test_util.hpp"
#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/recognizer/rtdetr_recognizer.hpp"

#include <string>
#include <vector>

namespace {

OnnxTestWriter rtdetr_head(bool box_log, bool class_sigmoid, bool with_topk) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Relu", {"images"}, {"feat"}));
  graph.message(1, node_message("Relu", {"feat"}, {"ref_a"}));
  graph.message(1, node_message("Relu", {"feat"}, {"ref_b"}));
  graph.message(1, node_message("Div", {"ref_a", "ref_b"}, {"ratio"}));
  if (box_log) {
    graph.message(1, node_message("Log", {"ratio"}, {"inverse"}));
  } else {
    graph.message(1, node_message("Relu", {"ratio"}, {"inverse"}));
  }
  std::string data = "feat";
  if (with_topk) {
    graph.message(1, node_message("Constant", {}, {"k"}, {attr_int("value_int", 300)}));
    graph.message(1, node_message("TopK", {"feat", "k"}, {"values", "indices"},
                                 {attr_int("axis", 1), attr_int("largest", 1), attr_int("sorted", 1)}));
    graph.message(1, node_message("Gather", {"feat", "indices"}, {"selected"}, {attr_int("axis", 1)}));
    data = "selected";
  }
  graph.message(1, node_message("MatMul", {data, "w_box"}, {"box_mm"}));
  graph.message(1, node_message("Add", {"box_mm", "b_box"}, {"box_linear"}));
  graph.message(1, node_message("Add", {"box_linear", "inverse"}, {"refined"}));
  graph.message(1, node_message("Sigmoid", {"refined"}, {"box_sig"}));
  graph.message(1, node_message("MatMul", {data, "w_cls"}, {"cls_mm"}));
  graph.message(1, node_message("Add", {"cls_mm", "b_cls"}, {"cls_linear"}));
  if (class_sigmoid) {
    graph.message(1, node_message("Sigmoid", {"cls_linear"}, {"cls_sig"}));
  } else {
    graph.message(1, node_message("Softmax", {"cls_linear"}, {"cls_sig"}, {attr_int("axis", -1)}));
  }
  graph.message(1, node_message("Concat", {"box_sig", "cls_sig"}, {"output0"}, {attr_int("axis", -1)}));
  graph.message(5, tensor_message("w_box", 1, {256, 4}, {}, "", false));
  graph.message(5, tensor_message("b_box", 1, {4}, {}, "", false));
  graph.message(5, tensor_message("w_cls", 1, {256, 5}, {}, "", false));
  graph.message(5, tensor_message("b_cls", 1, {5}, {}, "", false));
  graph.text(2, "synthetic_rtdetr");
  graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(64),
                                                              static_dim(64)})));
  graph.message(12, value_info("output0", tensor_type(1, true, {symbol_dim("batch"), symbol_dim("queries"),
                                                               symbol_dim("channels")})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

vie::ModelAnalysisReport analyze_head(bool box_log, bool class_sigmoid, bool with_topk) {
  return vie::OnnxAnalyzer{}.analyze_bytes(rtdetr_head(box_log, class_sigmoid, with_topk).str(), "head.onnx", ".");
}

}  // namespace

TEST(RtdetrRecognizer, RealExportIsRecognizedWithoutClaimingAStaticOutput) {
  const std::string path = vie_fixture("testdata/onnx/rtdetr-l.onnx");
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  EXPECT_TRUE(report.warnings.empty());
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->recognizer_id, "rtdetr_detection");
  EXPECT_EQ(result.selected()->architecture, "rtdetr_detection");
  EXPECT_FALSE(result.selected()->evidence.empty());
  for (const vie::RecognitionEvidence& evidence : result.selected()->evidence) {
    EXPECT_FALSE(evidence.detail.empty());
    EXPECT_EQ(evidence.detail.find("rtdetr-l"), std::string::npos);
  }

  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_FALSE(validation.runnable());
  EXPECT_EQ(manifest.task, vie::TaskType::Detection);
  EXPECT_EQ(manifest.model_family, "rtdetr");
  EXPECT_EQ(manifest.decoder.protocol, "rtdetr_detection_raw");
  EXPECT_EQ(manifest.input.name, "images");
  EXPECT_EQ(manifest.input.shape, (std::vector<std::int64_t>{1, 3, 640, 640}));
  EXPECT_EQ(manifest.input.dtype, vie::DataType::Fp32);
  EXPECT_EQ(manifest.input.resize, "stretch");
  EXPECT_TRUE(manifest.input.color.empty());
  EXPECT_TRUE(manifest.input.normalization.empty());
  ASSERT_EQ(manifest.outputs.size(), 1u);
  EXPECT_EQ(manifest.outputs[0].name, "output0");
  EXPECT_EQ(manifest.outputs[0].semantic, "predictions");
  EXPECT_EQ(manifest.outputs[0].shape, (std::vector<std::int64_t>{-1, -1, -1}));
  EXPECT_EQ(manifest.decoder.box_format, "cxcywh_normalized");
  EXPECT_EQ(manifest.decoder.box_decode, "decoded_normalized_cxcywh");
  ASSERT_TRUE(manifest.decoder.box_channels.has_value());
  EXPECT_EQ(*manifest.decoder.box_channels, 4);
  ASSERT_TRUE(manifest.decoder.channel_axis.has_value());
  EXPECT_EQ(*manifest.decoder.channel_axis, 2);
  ASSERT_TRUE(manifest.decoder.scores_are_probabilities.has_value());
  EXPECT_TRUE(*manifest.decoder.scores_are_probabilities);
  ASSERT_TRUE(manifest.num_classes.has_value());
  EXPECT_EQ(*manifest.num_classes, 80);
  ASSERT_TRUE(manifest.decoder.num_queries.has_value());
  EXPECT_EQ(*manifest.decoder.num_queries, 300);
  ASSERT_TRUE(manifest.nms.enabled.has_value());
  EXPECT_FALSE(*manifest.nms.enabled);
  EXPECT_FALSE(manifest.nms.pre_nms_top_k.has_value());
  EXPECT_FALSE(manifest.nms.confidence_threshold.has_value());
  EXPECT_FALSE(manifest.nms.iou_threshold.has_value());
  EXPECT_TRUE(manifest.decoder.class_selection.empty());

  const auto certainty = [&](const char* field) {
    const vie::FieldProvenance* record = manifest.find_provenance(field);
    EXPECT_NE(record, nullptr) << field;
    return record == nullptr ? vie::Certainty::User : record->certainty;
  };
  EXPECT_EQ(certainty("outputs.0.shape"), vie::Certainty::Observed);
  EXPECT_EQ(certainty("decoder.num_queries"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("decoder.box_format"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("decoder.box_decode"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("input.resize"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("nms.enabled"), vie::Certainty::Inferred);
  EXPECT_EQ(certainty("input.color"), vie::Certainty::Unresolved);
  EXPECT_EQ(certainty("input.normalization"), vie::Certainty::Unresolved);
  EXPECT_FALSE(manifest.find_provenance("decoder.num_queries")->evidence.empty());

  const std::string json = manifest.to_json_string(2);
  EXPECT_EQ(json.find("\"segmentation\""), std::string::npos);
  EXPECT_NE(json.find("\"num_queries\""), std::string::npos);
  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(json);
  EXPECT_EQ(loaded.task, manifest.task);
  EXPECT_EQ(loaded.model_family, manifest.model_family);
  EXPECT_EQ(loaded.decoder.protocol, manifest.decoder.protocol);
  EXPECT_EQ(loaded.outputs[0].semantic, "predictions");
  EXPECT_EQ(loaded.outputs[0].shape, manifest.outputs[0].shape);
  EXPECT_EQ(loaded.decoder.box_format, manifest.decoder.box_format);
  EXPECT_EQ(loaded.decoder.box_decode, manifest.decoder.box_decode);
  EXPECT_EQ(loaded.decoder.num_queries, manifest.decoder.num_queries);
  EXPECT_EQ(loaded.num_classes, manifest.num_classes);
  EXPECT_EQ(loaded.nms.enabled, manifest.nms.enabled);
  EXPECT_EQ(loaded.input.resize, manifest.input.resize);
  EXPECT_TRUE(loaded.input.color.empty());
  EXPECT_EQ(loaded.find_provenance("outputs.0.shape")->certainty, vie::Certainty::Observed);
  EXPECT_EQ(loaded.find_provenance("decoder.num_queries")->certainty, vie::Certainty::Inferred);

  vie::ModelManifest runnable = loaded;
  runnable.input.color = "rgb";
  runnable.input.normalization = "scale_0_1";
  runnable.input.interpolation = "bilinear";
  runnable.provenance["input.color"] = {vie::Certainty::User, {"test supplies the color"}};
  runnable.provenance["input.normalization"] = {vie::Certainty::User, {"test supplies the normalization"}};
  runnable.provenance["input.interpolation"] = {vie::Certainty::User, {"test supplies the interpolation"}};
  runnable.outputs[0].shape = {1, 300, 84};
  runnable.provenance["outputs.0.shape"] = {vie::Certainty::User, {"test supplies the runtime shape"}};
  vie::ModelConfig config;
  config.model_path = path;
  config.device = vie::DeviceType::Cpu;
  config.manifest = runnable;
  vie::InferenceEngine engine;
#if defined(VIE_HAS_ONNXRUNTIME)
  engine.load(config);
  EXPECT_TRUE(engine.loaded());
  EXPECT_EQ(engine.manifest().decoder.protocol, "rtdetr_detection_raw");
  EXPECT_EQ(engine.manifest().outputs[0].shape, (std::vector<std::int64_t>{1, 300, 84}));
  EXPECT_EQ(engine.manifest().find_provenance("outputs.0.shape")->certainty, vie::Certainty::User);
#else
  try {
    engine.load(config);
    FAIL() << "this build has no ONNX Runtime backend";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::Unsupported) << error.what();
    EXPECT_FALSE(engine.loaded());
    EXPECT_NE(std::string(error.what()).find("no backend"), std::string::npos) << error.what();
  }
#endif
}

TEST(RtdetrRecognizer, YoloDetectionIsNotRtdetrAndKeepsItsManifest) {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(vie_fixture("testdata/onnx/yolov8n.onnx"));
  EXPECT_FALSE(vie::RtdetrRecognizer{}.consider(report).has_value());
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_detection");
  EXPECT_EQ(result.selected()->decoder.protocol, "yolov8_detect_raw");
  const vie::ModelManifest manifest =
      vie::ManifestGenerator{}.generate(report, result, vie_fixture("testdata/onnx/yolov8n.onnx"));
  EXPECT_EQ(manifest.model_family, "yolov8");
  EXPECT_EQ(manifest.decoder.box_decode, "decoded_cxcywh");
  EXPECT_EQ(manifest.outputs[0].semantic, "cxcywh_then_class_scores");
  EXPECT_TRUE(manifest.input.resize.empty());
  EXPECT_FALSE(manifest.nms.enabled.has_value());
  EXPECT_FALSE(manifest.decoder.num_queries.has_value());
  EXPECT_EQ(manifest.to_json_string().find("\"segmentation\""), std::string::npos);
  EXPECT_EQ(manifest.to_json_string().find("num_queries"), std::string::npos);
}

TEST(RtdetrRecognizer, YoloSegmentationIsNotRtdetrAndKeepsItsManifest) {
  const vie::ModelAnalysisReport report =
      vie::OnnxAnalyzer{}.analyze_file(vie_fixture("testdata/onnx/yolov8n-seg.onnx"));
  EXPECT_FALSE(vie::RtdetrRecognizer{}.consider(report).has_value());
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->architecture, "yolov8_segmentation");
  EXPECT_EQ(result.selected()->decoder.protocol, "yolov8_seg_raw");
  const vie::ModelManifest manifest =
      vie::ManifestGenerator{}.generate(report, result, vie_fixture("testdata/onnx/yolov8n-seg.onnx"));
  EXPECT_EQ(manifest.task, vie::TaskType::Segmentation);
  EXPECT_EQ(manifest.model_family, "yolov8");
  EXPECT_TRUE(manifest.segmentation.mask_channels.has_value());
  EXPECT_NE(manifest.to_json_string().find("\"segmentation\""), std::string::npos);
  EXPECT_EQ(manifest.to_json_string().find("rtdetr_detection_raw"), std::string::npos);
  EXPECT_EQ(manifest.to_json_string().find("num_queries"), std::string::npos);
}

TEST(RtdetrRecognizer, IdentityTopkAndShapeAloneAreUnknown) {
  OnnxTestWriter identity_graph;
  identity_graph.message(1, node_message("Identity", {"images"}, {"output0"}));
  identity_graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                                       static_dim(8)})));
  identity_graph.message(12, value_info("output0", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                                        static_dim(8)})));
  OnnxTestWriter identity;
  identity.i64(1, 10);
  identity.message(7, identity_graph);
  identity.message(8, opset_import("", 17));
  const vie::RecognitionResult identity_result = vie::default_recognizer_registry().recognize(
      vie::OnnxAnalyzer{}.analyze_bytes(identity.str(), "identity.onnx", "."));
  EXPECT_EQ(identity_result.status, vie::RecognitionStatus::Unknown);

  OnnxTestWriter topk_graph;
  topk_graph.message(1, node_message("Constant", {}, {"k"}, {attr_int("value_int", 300)}));
  topk_graph.message(1, node_message("TopK", {"images", "k"}, {"values", "indices"}, {attr_int("axis", 1)}));
  topk_graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(64),
                                                                   static_dim(64)})));
  topk_graph.message(12, value_info("values", tensor_type(1, true, {static_dim(1), static_dim(300), static_dim(64)})));
  OnnxTestWriter topk;
  topk.i64(1, 10);
  topk.message(7, topk_graph);
  topk.message(8, opset_import("", 17));
  const vie::RecognitionResult topk_result = vie::default_recognizer_registry().recognize(
      vie::OnnxAnalyzer{}.analyze_bytes(topk.str(), "topk.onnx", "."));
  EXPECT_EQ(topk_result.status, vie::RecognitionStatus::Unknown);

  OnnxTestWriter shape_graph;
  shape_graph.message(1, node_message("Identity", {"images"}, {"output0"}));
  shape_graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(640),
                                                                    static_dim(640)})));
  shape_graph.message(12, value_info("output0", tensor_type(1, true, {static_dim(1), static_dim(300), static_dim(84)})));
  OnnxTestWriter shape;
  shape.i64(1, 10);
  shape.text(2, "rtdetr-l.onnx");
  shape.message(7, shape_graph);
  shape.message(8, opset_import("", 17));
  const vie::RecognitionResult shape_result = vie::default_recognizer_registry().recognize(
      vie::OnnxAnalyzer{}.analyze_bytes(shape.str(), "rtdetr-l.onnx", "."));
  EXPECT_EQ(shape_result.status, vie::RecognitionStatus::Unknown);
}

TEST(RtdetrRecognizer, SyntheticHeadNeedsBoxClassAndQuerySelection) {
  const vie::ModelAnalysisReport positive = analyze_head(true, true, true);
  const vie::RecognitionResult recognized = vie::default_recognizer_registry().recognize(positive);
  EXPECT_EQ(recognized.status, vie::RecognitionStatus::Recognized) << recognized.explanation;
  ASSERT_NE(recognized.selected(), nullptr);
  EXPECT_EQ(recognized.selected()->recognizer_id, "rtdetr_detection");
  ASSERT_TRUE(recognized.selected()->decoder.num_classes.has_value());
  EXPECT_EQ(*recognized.selected()->decoder.num_classes, 5);
  ASSERT_TRUE(recognized.selected()->decoder.num_queries.has_value());
  EXPECT_EQ(*recognized.selected()->decoder.num_queries, 300);

  const vie::RecognitionResult broken_box = vie::default_recognizer_registry().recognize(analyze_head(false, true, true));
  EXPECT_EQ(broken_box.status, vie::RecognitionStatus::Unknown);
  const vie::RecognitionResult broken_class =
      vie::default_recognizer_registry().recognize(analyze_head(true, false, true));
  EXPECT_EQ(broken_class.status, vie::RecognitionStatus::Unknown);
  const vie::RecognitionResult no_queries =
      vie::default_recognizer_registry().recognize(analyze_head(true, true, false));
  EXPECT_EQ(no_queries.status, vie::RecognitionStatus::Unknown);
}
