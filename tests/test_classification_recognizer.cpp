#include "onnx_test_writer.hpp"
#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/classification_recognizer.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/model_recognizer.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <filesystem>
#include <string>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/engine/inference_engine.hpp"
#endif
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include "vie/backend/onnxruntime_cuda_backend.hpp"

#include <cuda_runtime.h>
#endif

namespace {

OnnxTestWriter wrap_model(const OnnxTestWriter& graph) {
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

vie::RecognitionResult recognize_model(const OnnxTestWriter& model, const char* name) {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_bytes(model.str(), name, ".");
  return vie::default_recognizer_registry().recognize(report);
}

OnnxTestWriter gemm_classifier(int classes, int features, const std::string& tail) {
  OnnxTestWriter graph;
  graph.message(1, node_message("GlobalAveragePool", {"images"}, {"pooled"}));
  graph.message(1, node_message("Flatten", {"pooled"}, {"flat"}, {attr_int("axis", 1)}));
  graph.message(1, node_message("Gemm", {"flat", "weight", "bias"}, {"pre"}, {attr_int("transB", 1)}));
  if (tail == "softmax") {
    graph.message(1, node_message("Softmax", {"pre"}, {"logits"}, {attr_int("axis", 1)}));
  } else if (tail == "sigmoid") {
    graph.message(1, node_message("Sigmoid", {"pre"}, {"logits"}));
  } else if (tail == "normalize") {
    graph.message(1, node_message("LpNormalization", {"pre"}, {"logits"}, {attr_int("axis", 1)}));
  } else {
    graph.message(1, node_message("Identity", {"pre"}, {"logits"}));
  }
  graph.message(5, tensor_message("weight", 1, {classes, features}, {}, "", false));
  graph.message(5, tensor_message("bias", 1, {classes}, {}, "", false));
  graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                              static_dim(8)})));
  graph.message(12, value_info("logits", tensor_type(1, true, {static_dim(1), static_dim(classes)})));
  return wrap_model(graph);
}

bool has_evidence(const vie::RecognitionCandidate& candidate, const std::string& code) {
  for (const vie::RecognitionEvidence& evidence : candidate.evidence) {
    if (evidence.code == code && !evidence.detail.empty()) return true;
  }
  return false;
}

class NamedCandidate : public vie::IModelRecognizer {
 public:
  NamedCandidate(std::string architecture, int evidence_items)
      : architecture_(std::move(architecture)), evidence_items_(evidence_items) {}

  std::string id() const override { return architecture_; }

  std::optional<vie::RecognitionCandidate> consider(const vie::ModelAnalysisReport&) const override {
    vie::RecognitionCandidate candidate;
    candidate.recognizer_id = architecture_;
    candidate.architecture = architecture_;
    for (int index = 0; index < evidence_items_; ++index) {
      candidate.evidence.push_back({"item", architecture_});
    }
    return candidate;
  }

 private:
  std::string architecture_;
  int evidence_items_ = 0;
};

}  // namespace

TEST(ClassificationRecognizer, SyntheticGemmHeadIsLogits) {
  const vie::RecognitionResult result =
      recognize_model(gemm_classifier(5, 8, "identity"), "not_a_filename_clue.onnx");
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->recognizer_id, "classification_logits");
  EXPECT_EQ(result.selected()->architecture, "single_label_classification");
  EXPECT_TRUE(result.selected()->family.empty());
  EXPECT_EQ(result.selected()->decoder.protocol, "classification_logits_raw");
  EXPECT_EQ(result.selected()->decoder.output_name, "logits");
  EXPECT_EQ(result.selected()->decoder.num_classes, 5);
  ASSERT_TRUE(result.selected()->decoder.class_axis.has_value());
  EXPECT_EQ(*result.selected()->decoder.class_axis, 1);
  EXPECT_EQ(result.selected()->decoder.score_type, "logits");
  EXPECT_TRUE(has_evidence(*result.selected(), "single_runtime_input"));
  EXPECT_TRUE(has_evidence(*result.selected(), "single_output"));
  EXPECT_TRUE(has_evidence(*result.selected(), "rank2_output"));
  EXPECT_TRUE(has_evidence(*result.selected(), "class_dimension"));
  EXPECT_TRUE(has_evidence(*result.selected(), "batch_dimension"));
  EXPECT_TRUE(has_evidence(*result.selected(), "linear_classification_head"));
  EXPECT_TRUE(has_evidence(*result.selected(), "global_pool_path"));
  EXPECT_TRUE(has_evidence(*result.selected(), "logits_output"));
  EXPECT_TRUE(has_evidence(*result.selected(), "single_label_output"));
  for (const vie::RecognitionEvidence& evidence : result.selected()->evidence) {
    EXPECT_EQ(evidence.detail.find("not_a_filename_clue"), std::string::npos);
  }

  const vie::ModelAnalysisReport report =
      vie::OnnxAnalyzer{}.analyze_bytes(gemm_classifier(5, 8, "identity").str(), "synthetic.onnx", ".");
  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, "synthetic.onnx");
  EXPECT_EQ(manifest.task, vie::TaskType::Classification);
  EXPECT_TRUE(manifest.model_family.empty());
  EXPECT_EQ(manifest.num_classes, 5);
  EXPECT_EQ(manifest.decoder.protocol, "classification_logits_raw");
  EXPECT_EQ(manifest.classification.output_name, "logits");
  ASSERT_TRUE(manifest.classification.class_axis.has_value());
  EXPECT_EQ(*manifest.classification.class_axis, 1);
  EXPECT_EQ(manifest.classification.score_type, "logits");
  EXPECT_TRUE(manifest.classification.class_names.empty());
  EXPECT_TRUE(manifest.input.color.empty());
  EXPECT_TRUE(manifest.input.normalization.empty());
  EXPECT_TRUE(manifest.input.resize.empty());
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_FALSE(validation.runnable());
}

TEST(ClassificationRecognizer, MatMulPlusBiasIsAccepted) {
  OnnxTestWriter graph;
  graph.message(1, node_message("GlobalMaxPool", {"images"}, {"pooled"}));
  graph.message(1, node_message("Flatten", {"pooled"}, {"flat"}, {attr_int("axis", 1)}));
  graph.message(1, node_message("MatMul", {"flat", "weight"}, {"product"}));
  graph.message(1, node_message("Add", {"product", "bias"}, {"logits"}));
  graph.message(5, tensor_message("weight", 1, {8, 4}, {}, "", false));
  graph.message(5, tensor_message("bias", 1, {4}, {}, "", false));
  graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(4),
                                                              static_dim(4)})));
  graph.message(12, value_info("logits", tensor_type(1, true, {symbol_dim("batch"), static_dim(4)})));
  const vie::RecognitionResult result = recognize_model(wrap_model(graph), "matmul.onnx");
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->decoder.num_classes, 4);
  EXPECT_EQ(result.selected()->decoder.score_type, "logits");
  EXPECT_TRUE(has_evidence(*result.selected(), "batch_dimension"));
  bool saw_symbol = false;
  for (const vie::RecognitionEvidence& evidence : result.selected()->evidence) {
    if (evidence.detail.find("symbolic") != std::string::npos) saw_symbol = true;
  }
  EXPECT_TRUE(saw_symbol);
}

TEST(ClassificationRecognizer, DeclinesStructuresThatAreNotSingleLabelLogits) {
  auto expect_unknown = [](const OnnxTestWriter& model, const char* name) {
    const vie::RecognitionResult result = recognize_model(model, name);
    EXPECT_EQ(result.status, vie::RecognitionStatus::Unknown) << name << " " << result.explanation;
    EXPECT_EQ(result.selected(), nullptr);
  };

  expect_unknown(gemm_classifier(5, 8, "softmax"), "softmax");
  expect_unknown(gemm_classifier(5, 8, "sigmoid"), "sigmoid");
  expect_unknown(gemm_classifier(5, 8, "normalize"), "normalize");
  expect_unknown(gemm_classifier(1, 8, "identity"), "regression");

  OnnxTestWriter pool_only;
  pool_only.message(1, node_message("GlobalAveragePool", {"images"}, {"pooled"}));
  pool_only.message(1, node_message("Flatten", {"pooled"}, {"logits"}, {attr_int("axis", 1)}));
  pool_only.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                                  static_dim(8)})));
  pool_only.message(12, value_info("logits", tensor_type(1, true, {static_dim(1), static_dim(32)})));
  expect_unknown(wrap_model(pool_only), "pool_only");

  OnnxTestWriter embedding;
  embedding.message(1, node_message("Gemm", {"features", "weight"}, {"logits"}, {attr_int("transB", 1)}));
  embedding.message(5, tensor_message("weight", 1, {16, 32}, {}, "", false));
  embedding.message(11, value_info("features", tensor_type(1, true, {static_dim(1), static_dim(32)})));
  embedding.message(12, value_info("logits", tensor_type(1, true, {static_dim(1), static_dim(16)})));
  expect_unknown(wrap_model(embedding), "embedding");

  OnnxTestWriter matmul_only;
  matmul_only.message(1, node_message("MatMul", {"features", "weight"}, {"logits"}));
  matmul_only.message(5, tensor_message("weight", 1, {8, 4}, {}, "", false));
  matmul_only.message(11, value_info("features", tensor_type(1, true, {static_dim(1), static_dim(8)})));
  matmul_only.message(12, value_info("logits", tensor_type(1, true, {static_dim(1), static_dim(4)})));
  expect_unknown(wrap_model(matmul_only), "matmul_only");

  OnnxTestWriter rank3_graph;
  rank3_graph.message(1, node_message("GlobalAveragePool", {"images"}, {"pooled"}));
  rank3_graph.message(1, node_message("Flatten", {"pooled"}, {"flat"}, {attr_int("axis", 1)}));
  rank3_graph.message(1, node_message("Gemm", {"flat", "weight", "bias"}, {"logits"}, {attr_int("transB", 1)}));
  rank3_graph.message(5, tensor_message("weight", 1, {5, 8}, {}, "", false));
  rank3_graph.message(5, tensor_message("bias", 1, {5}, {}, "", false));
  rank3_graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                                    static_dim(8)})));
  rank3_graph.message(12, value_info("logits", tensor_type(1, true, {static_dim(1), static_dim(5), static_dim(1)})));
  expect_unknown(wrap_model(rank3_graph), "rank3");

  OnnxTestWriter leading;
  leading.message(1, node_message("GlobalAveragePool", {"images"}, {"pooled"}));
  leading.message(1, node_message("Gemm", {"pooled", "weight", "bias"}, {"logits"}, {attr_int("transB", 1)}));
  leading.message(5, tensor_message("weight", 1, {1, 8}, {}, "", false));
  leading.message(5, tensor_message("bias", 1, {1}, {}, "", false));
  leading.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                                static_dim(8)})));
  leading.message(12, value_info("logits", tensor_type(1, true, {static_dim(5), static_dim(1)})));
  expect_unknown(wrap_model(leading), "leading_class_axis");

  OnnxTestWriter multi_out;
  multi_out.message(1, node_message("GlobalAveragePool", {"images"}, {"pooled"}));
  multi_out.message(1, node_message("Flatten", {"pooled"}, {"flat"}, {attr_int("axis", 1)}));
  multi_out.message(1, node_message("Gemm", {"flat", "weight", "bias"}, {"logits"}, {attr_int("transB", 1)}));
  multi_out.message(5, tensor_message("weight", 1, {5, 8}, {}, "", false));
  multi_out.message(5, tensor_message("bias", 1, {5}, {}, "", false));
  multi_out.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                                  static_dim(8)})));
  multi_out.message(12, value_info("logits", tensor_type(1, true, {static_dim(1), static_dim(5)})));
  multi_out.message(12, value_info("pooled", tensor_type(1, true, {static_dim(1), static_dim(8), static_dim(1),
                                                                  static_dim(1)})));
  expect_unknown(wrap_model(multi_out), "multi_output");

  OnnxTestWriter multi_in;
  multi_in.message(1, node_message("GlobalAveragePool", {"images"}, {"pooled"}));
  multi_in.message(1, node_message("Flatten", {"pooled"}, {"flat"}, {attr_int("axis", 1)}));
  multi_in.message(1, node_message("Gemm", {"flat", "weight", "bias"}, {"logits"}, {attr_int("transB", 1)}));
  multi_in.message(5, tensor_message("weight", 1, {5, 8}, {}, "", false));
  multi_in.message(5, tensor_message("bias", 1, {5}, {}, "", false));
  multi_in.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                                 static_dim(8)})));
  multi_in.message(11, value_info("extra", tensor_type(1, true, {static_dim(1), static_dim(4)})));
  multi_in.message(12, value_info("logits", tensor_type(1, true, {static_dim(1), static_dim(5)})));
  expect_unknown(wrap_model(multi_in), "multi_input");
}

TEST(ClassificationRecognizer, AnotherTaskWithMoreEvidenceStaysAmbiguous) {
  vie::ModelRecognizerRegistry registry;
  registry.add(std::make_unique<NamedCandidate>("yolov8_detection", 2));
  registry.add(std::make_unique<NamedCandidate>("single_label_classification", 9));
  const vie::ModelAnalysisReport report =
      vie::OnnxAnalyzer{}.analyze_bytes(gemm_classifier(5, 8, "identity").str(), "tie.onnx", ".");
  const vie::RecognitionResult result = registry.recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Ambiguous);
  EXPECT_EQ(result.selected(), nullptr);
  EXPECT_NE(result.explanation.find("classification"), std::string::npos);
}

TEST(ClassificationRecognizer, ExistingDetectorsStaySelected) {
  const auto check = [](const char* path, const char* recognizer) {
    const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(vie_fixture(path));
    const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
    EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << path << " " << result.explanation;
    ASSERT_NE(result.selected(), nullptr);
    EXPECT_EQ(result.selected()->recognizer_id, recognizer);
    const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
    EXPECT_EQ(manifest.to_json_string().find("\"classification\""), std::string::npos) << path;
  };
  check("testdata/onnx/yolov8n.onnx", "yolov8_detection");
  check("testdata/onnx/yolov8n-seg.onnx", "yolov8_segmentation");
  check("testdata/onnx/rtdetr-l.onnx", "rtdetr_detection");

  OnnxTestWriter identity;
  identity.message(1, node_message("Identity", {"x"}, {"y"}));
  identity.message(11, value_info("x", tensor_type(1, true, {static_dim(1), static_dim(4)})));
  identity.message(12, value_info("y", tensor_type(1, true, {static_dim(1), static_dim(4)})));
  const vie::RecognitionResult unknown = recognize_model(wrap_model(identity), "identity.onnx");
  EXPECT_EQ(unknown.status, vie::RecognitionStatus::Unknown);
}

TEST(ClassificationRecognizer, UntrainedResNetIsRecognizedWithoutImageNetClaims) {
  const std::string path = "/tmp/resnet50_untrained_static.onnx";
  ASSERT_TRUE(std::filesystem::exists(path));
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << result.explanation;
  ASSERT_NE(result.selected(), nullptr);
  EXPECT_EQ(result.selected()->recognizer_id, "classification_logits");
  EXPECT_EQ(result.selected()->family, "cnn_classifier");
  EXPECT_EQ(result.selected()->decoder.output_name, "logits");
  EXPECT_EQ(result.selected()->decoder.num_classes, 1000);
  EXPECT_EQ(result.selected()->decoder.score_type, "logits");
  EXPECT_EQ(result.selected()->decoder.protocol, "classification_logits_raw");
  for (const vie::RecognitionEvidence& evidence : result.selected()->evidence) {
    EXPECT_EQ(evidence.detail.find("ImageNet"), std::string::npos);
    EXPECT_EQ(evidence.detail.find("resnet50"), std::string::npos);
  }

  const vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
  EXPECT_EQ(manifest.task, vie::TaskType::Classification);
  EXPECT_EQ(manifest.model_family, "cnn_classifier");
  EXPECT_EQ(manifest.num_classes, 1000);
  EXPECT_EQ(manifest.classification.output_name, "logits");
  ASSERT_TRUE(manifest.classification.class_axis.has_value());
  EXPECT_EQ(*manifest.classification.class_axis, 1);
  EXPECT_EQ(manifest.classification.score_type, "logits");
  EXPECT_TRUE(manifest.classification.class_names.empty());
  EXPECT_TRUE(manifest.input.color.empty());
  EXPECT_TRUE(manifest.input.normalization.empty());
  EXPECT_TRUE(manifest.input.resize.empty());
  EXPECT_EQ(manifest.input.layout, vie::Layout::Nchw);
  EXPECT_EQ(manifest.outputs.at(0).name, "logits");
  EXPECT_EQ(manifest.outputs.at(0).semantic, "logits");
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_FALSE(validation.runnable());
  EXPECT_NE(manifest.to_json_string().find("\"classification\""), std::string::npos);
}

#if defined(VIE_HAS_ONNXRUNTIME)
namespace {

void supply_preprocess(vie::ModelManifest& manifest) {
  auto user = [&](const char* field) {
    manifest.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {"test supplied preprocess"}};
  };
  manifest.input.color = "rgb";
  user("input.color");
  manifest.input.normalization = "scale_0_1";
  user("input.normalization");
  manifest.input.resize = "stretch";
  user("input.resize");
  manifest.input.interpolation = "bilinear";
  user("input.interpolation");
}

vie::ImageInput gray_image(std::vector<std::uint8_t>& storage) {
  storage.assign(16 * 16 * 3, 0);
  vie::ImageInput image;
  image.data = storage.data();
  image.width = 16;
  image.height = 16;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

}  // namespace

TEST(ClassificationRecognizer, GeneratedManifestRunsInferClassification) {
  const std::string path = "/tmp/resnet50_untrained_static.onnx";
  ASSERT_TRUE(std::filesystem::exists(path));
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
  EXPECT_TRUE(manifest.input.color.empty());
  supply_preprocess(manifest);

  vie::ModelConfig config;
  config.model_path = path;
  config.manifest = manifest;
  config.backend = vie::BackendKind::OnnxRuntime;
  config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  engine.load(config);
  std::vector<std::uint8_t> pixels;
  vie::ClassificationRuntimeConfig runtime;
  runtime.top_k = 5;
  const vie::ClassificationBatch batch = engine.infer_classification(gray_image(pixels), runtime);
  ASSERT_EQ(batch.results.size(), 1u);
  EXPECT_EQ(batch.results[0].num_classes, 1000);
  EXPECT_EQ(batch.results[0].output_bytes, 4000u);
  EXPECT_EQ(batch.results[0].d2h_bytes, 0u);
  EXPECT_EQ(batch.results[0].labels.size(), 5u);
  EXPECT_EQ(batch.results[0].score_type, "logits");

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
  vie::OnnxRuntimeCudaBackend cuda_backend;
  EXPECT_FALSE(cuda_backend.cpu_ep_fallback());
  vie::ModelConfig gpu_cpu = config;
  gpu_cpu.device = vie::DeviceType::Cpu;
  gpu_cpu.backend = vie::BackendKind::OnnxRuntime;
  vie::InferenceEngine gpu_package_cpu;
  gpu_package_cpu.load(gpu_cpu);
  const vie::ClassificationBatch cpu_batch =
      gpu_package_cpu.infer_classification(gray_image(pixels), runtime);
  EXPECT_EQ(cpu_batch.results.at(0).d2h_bytes, 0u);
  EXPECT_EQ(cpu_batch.results.at(0).labels.size(), 5u);

  void* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(&device_pixels, pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput gpu_image = gray_image(pixels);
  gpu_image.data = device_pixels;
  gpu_image.device = vie::DeviceType::Cuda;
  vie::ModelConfig cuda_config = config;
  cuda_config.device = vie::DeviceType::Cuda;
  cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
  vie::InferenceEngine cuda_engine;
  cuda_engine.load(cuda_config);
  vie::ClassificationInferTiming timing;
  const vie::ClassificationBatch cuda_batch = cuda_engine.infer_classification(gpu_image, runtime, &timing);
  EXPECT_EQ(timing.d2h_bytes, 4000u);
  EXPECT_EQ(timing.output_bytes, 4000u);
  EXPECT_EQ(cuda_batch.results.at(0).execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(cuda_batch.results.at(0).labels.size(), 5u);
  cudaFree(device_pixels);
#endif
}
#endif
