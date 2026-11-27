#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/engine/inference_engine.hpp"
#endif
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

struct ExportedModel {
  const char* file;
  const char* score_type;
  const char* protocol;
};

const std::filesystem::path kModelDir{"/tmp/vie_phase66"};

const ExportedModel kModels[] = {
    {"resnet18_untrained_opset17.onnx", "logits", "classification_logits_raw"},
    {"mobilenet_v2_untrained_opset17.onnx", "logits", "classification_logits_raw"},
    {"mobilenet_v3_small_untrained_opset17.onnx", "logits", "classification_logits_raw"},
    {"efficientnet_b0_untrained_opset17.onnx", "logits", "classification_logits_raw"},
    {"resnet18_untrained_opset17_softmax.onnx", "probabilities", "classification_probabilities_raw"},
};

bool models_ready() {
  if (!std::filesystem::is_directory(kModelDir)) return false;
  for (const ExportedModel& model : kModels) {
    if (!std::filesystem::is_regular_file(kModelDir / model.file)) return false;
  }
  return true;
}

}  // namespace

TEST(ClassificationRealModels, RecognizerAcceptsExportedGraphsWithoutInferringPreprocess) {
  if (!models_ready()) {
    GTEST_SKIP() << "optional models are absent; run tools/export_classification_models.py";
  }
  for (const ExportedModel& model : kModels) {
    const std::filesystem::path path = kModelDir / model.file;
    const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path.string());
    ASSERT_EQ(report.runtime_inputs.size(), 1u) << model.file;
    ASSERT_EQ(report.outputs.size(), 1u) << model.file;
    EXPECT_EQ(report.runtime_inputs[0].type.declared_shape.dimensions.size(), 4u) << model.file;
    const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
    EXPECT_EQ(result.status, vie::RecognitionStatus::Recognized) << model.file << " " << result.explanation;
    ASSERT_NE(result.selected(), nullptr) << model.file;
    EXPECT_EQ(result.selected()->decoder.protocol, model.protocol) << model.file;
    EXPECT_EQ(result.selected()->decoder.score_type, model.score_type) << model.file;
    EXPECT_EQ(result.selected()->decoder.num_classes, 1000) << model.file;
    ASSERT_TRUE(result.selected()->decoder.class_axis.has_value()) << model.file;
    EXPECT_EQ(*result.selected()->decoder.class_axis, 1) << model.file;
    EXPECT_EQ(result.selected()->family, "cnn_classifier") << model.file;
    for (const vie::RecognitionEvidence& evidence : result.selected()->evidence) {
      EXPECT_EQ(evidence.detail.find("ImageNet"), std::string::npos) << model.file;
    }

    vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path.string());
    EXPECT_EQ(manifest.decoder.protocol, model.protocol) << model.file;
    EXPECT_EQ(manifest.classification.score_type, model.score_type) << model.file;
    EXPECT_EQ(manifest.classification.output_name, result.selected()->decoder.output_name) << model.file;
    EXPECT_TRUE(manifest.input.color.empty()) << model.file;
    EXPECT_TRUE(manifest.input.normalization.empty()) << model.file;
    EXPECT_TRUE(manifest.input.resize.empty()) << model.file;
    EXPECT_EQ(manifest.outputs.at(0).semantic, model.score_type) << model.file;
    const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
    EXPECT_TRUE(validation.schema_ok()) << model.file << " " << vie::format_issues(validation.schema_issues);
    EXPECT_FALSE(validation.runnable()) << model.file;
  }
}

#if defined(VIE_HAS_ONNXRUNTIME)
namespace {

std::vector<int> reference_order(const std::vector<float>& scores) {
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    const float left_score = scores[static_cast<std::size_t>(left)];
    const float right_score = scores[static_cast<std::size_t>(right)];
    if (left_score != right_score) return left_score > right_score;
    return left < right;
  });
  return order;
}

vie::ImageInput pattern_image(std::vector<std::uint8_t>& storage) {
  storage.resize(224 * 224 * 3);
  for (std::size_t index = 0; index < storage.size(); ++index) {
    storage[index] = static_cast<std::uint8_t>((index * 17 + 3) % 251);
  }
  vie::ImageInput image;
  image.data = storage.data();
  image.width = 224;
  image.height = 224;
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

std::vector<float> raw_scores(vie::InferenceEngine& engine, const vie::ImageInput& image,
                              const std::string& output_name) {
  vie::BackendRequest request;
  request.inputs.emplace(engine.manifest().input.name, engine.preprocess(image).output);
  const vie::BackendOutput output = engine.execute(request);
  const vie::Tensor& tensor = output.tensors.at(output_name).tensor;
  EXPECT_EQ(tensor.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(tensor.shape(), (std::vector<std::int64_t>{1, 1000}));
  const float* data = tensor.data_as<float>();
  return std::vector<float>(data, data + tensor.numel());
}

void expect_matches_raw_order(const vie::ClassificationBatch& batch, const std::vector<float>& scores,
                               const std::string& score_type) {
  const std::vector<int> expected = reference_order(scores);
  ASSERT_EQ(batch.results.size(), 1u);
  ASSERT_EQ(batch.results[0].labels.size(), 5u);
  EXPECT_EQ(batch.results[0].score_type, score_type);
  EXPECT_EQ(batch.results[0].num_classes, 1000);
  EXPECT_EQ(batch.results[0].top_k, 5);
  EXPECT_EQ(batch.results[0].execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(batch.results[0].output_bytes, 4000u);
  for (int rank = 0; rank < 5; ++rank) {
    const int class_id = expected[static_cast<std::size_t>(rank)];
    EXPECT_EQ(batch.results[0].labels[static_cast<std::size_t>(rank)].class_id, class_id);
    EXPECT_FLOAT_EQ(batch.results[0].labels[static_cast<std::size_t>(rank)].score,
                    scores[static_cast<std::size_t>(class_id)]);
    EXPECT_GE(class_id, 0);
    EXPECT_LT(class_id, 1000);
    EXPECT_TRUE(std::isfinite(scores[static_cast<std::size_t>(class_id)]));
  }
  if (score_type == "logits") {
    const double maximum = static_cast<double>(*std::max_element(scores.begin(), scores.end()));
    double partition = 0.0;
    for (const float score : scores) partition += std::exp(static_cast<double>(score) - maximum);
    bool differs_from_softmax = false;
    for (const float score : scores) {
      const double probability = std::exp(static_cast<double>(score) - maximum) / partition;
      if (std::fabs(probability - static_cast<double>(score)) > 1e-3) differs_from_softmax = true;
    }
    EXPECT_TRUE(differs_from_softmax);
  } else {
    for (const vie::Classification& label : batch.results[0].labels) {
      EXPECT_GE(label.score, 0.f);
      EXPECT_LE(label.score, 1.f);
    }
  }
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(std::floor(fraction * static_cast<double>(values.size() - 1)));
  return values[index];
}

}  // namespace

TEST(ClassificationRealModels, CpuEndToEndKeepsRawScores) {
  if (!models_ready()) {
    GTEST_SKIP() << "optional models are absent; run tools/export_classification_models.py";
  }
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = pattern_image(pixels);
  for (const ExportedModel& model : kModels) {
    const std::filesystem::path path = kModelDir / model.file;
    const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path.string());
    const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
    ASSERT_EQ(result.status, vie::RecognitionStatus::Recognized) << model.file;
    vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path.string());
    supply_test_preprocess(manifest);
    vie::ModelConfig config;
    config.model_path = path.string();
    config.manifest = manifest;
    config.backend = vie::BackendKind::OnnxRuntime;
    config.device = vie::DeviceType::Cpu;
    vie::InferenceEngine engine;
    engine.load(config);
    vie::ClassificationRuntimeConfig runtime;
    runtime.top_k = 5;
    vie::ClassificationInferTiming timing;
    const vie::ClassificationBatch batch = engine.infer_classification(image, runtime, &timing);
    const std::vector<float> scores = raw_scores(engine, image, manifest.classification.output_name);
    expect_matches_raw_order(batch, scores, model.score_type);
    const std::filesystem::path reference_path = path.parent_path() / (path.stem().string() + ".pytorch.f32");
    if (std::filesystem::is_regular_file(reference_path)) {
      std::ifstream reference(reference_path, std::ios::binary);
      std::vector<float> pytorch(scores.size());
      reference.read(reinterpret_cast<char*>(pytorch.data()),
                     static_cast<std::streamsize>(pytorch.size() * sizeof(float)));
      ASSERT_TRUE(reference) << model.file;
      float max_abs = 0.f;
      for (std::size_t index = 0; index < scores.size(); ++index) {
        max_abs = std::max(max_abs, std::fabs(scores[index] - pytorch[index]));
      }
      EXPECT_LT(max_abs, 1e-4f) << model.file;
    }
    EXPECT_EQ(timing.d2h_bytes, 0u) << model.file;
    EXPECT_EQ(timing.output_bytes, 4000u) << model.file;
    EXPECT_FALSE(timing.includes_d2h) << model.file;
  }
}

TEST(ClassificationRealModels, ResNet18TimingUsesRepeatedSamples) {
  const std::filesystem::path path = kModelDir / "resnet18_untrained_opset17.onnx";
  if (!std::filesystem::is_regular_file(path)) {
    GTEST_SKIP() << "resnet18 export is absent";
  }
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path.string());
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path.string());
  supply_test_preprocess(manifest);
  vie::ModelConfig config;
  config.model_path = path.string();
  config.manifest = manifest;
  config.backend = vie::BackendKind::OnnxRuntime;
  config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  engine.load(config);
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = pattern_image(pixels);
  vie::ClassificationRuntimeConfig runtime;
  runtime.top_k = 5;
  constexpr int kWarmup = 3;
  constexpr int kSamples = 20;
  for (int index = 0; index < kWarmup; ++index) engine.infer_classification(image, runtime, nullptr);
  std::vector<double> preprocess_ms;
  std::vector<double> execute_ms;
  std::vector<double> validate_ms;
  std::vector<double> topk_ms;
  std::vector<double> total_ms;
  for (int index = 0; index < kSamples; ++index) {
    vie::ClassificationInferTiming timing;
    engine.infer_classification(image, runtime, &timing);
    preprocess_ms.push_back(timing.preprocess_ms);
    execute_ms.push_back(timing.execute_ms);
    validate_ms.push_back(timing.validate_ms);
    topk_ms.push_back(timing.topk_ms);
    total_ms.push_back(timing.total_ms);
  }
  auto mean = [](const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  };
  std::ofstream record("/tmp/vie_phase66_resnet18_cpu.txt");
  record << "model resnet18_untrained classes 1000 input 224x224 dtype fp32 device cpu "
         << "includes_d2h 0 warmup " << kWarmup << " samples " << kSamples << "\n";
  record << "preprocess mean " << mean(preprocess_ms) << " p50 " << percentile(preprocess_ms, 0.50)
         << " p95 " << percentile(preprocess_ms, 0.95) << "\n";
  record << "execute mean " << mean(execute_ms) << " p50 " << percentile(execute_ms, 0.50) << " p95 "
         << percentile(execute_ms, 0.95) << "\n";
  record << "validate mean " << mean(validate_ms) << " p50 " << percentile(validate_ms, 0.50) << " p95 "
         << percentile(validate_ms, 0.95) << "\n";
  record << "topk mean " << mean(topk_ms) << " p50 " << percentile(topk_ms, 0.50) << " p95 "
         << percentile(topk_ms, 0.95) << "\n";
  record << "total mean " << mean(total_ms) << " p50 " << percentile(total_ms, 0.50) << " p95 "
         << percentile(total_ms, 0.95) << "\n";
  EXPECT_GT(mean(execute_ms), 0.0);
  EXPECT_GE(mean(total_ms), mean(execute_ms));
}

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
TEST(ClassificationRealModels, StrictCudaAgreesWithCpuOnRawTopK) {
  if (!models_ready()) {
    GTEST_SKIP() << "optional models are absent; run tools/export_classification_models.py";
  }
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = pattern_image(pixels);
  void* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(&device_pixels, pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput gpu_image = image;
  gpu_image.data = device_pixels;
  gpu_image.device = vie::DeviceType::Cuda;
  for (const ExportedModel& model : kModels) {
    const std::filesystem::path path = kModelDir / model.file;
    const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path.string());
    const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
    vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path.string());
    supply_test_preprocess(manifest);
    vie::ClassificationRuntimeConfig runtime;
    runtime.top_k = 5;
    vie::ModelConfig cpu_config;
    cpu_config.model_path = path.string();
    cpu_config.manifest = manifest;
    cpu_config.backend = vie::BackendKind::OnnxRuntime;
    cpu_config.device = vie::DeviceType::Cpu;
    vie::InferenceEngine cpu_engine;
    cpu_engine.load(cpu_config);
    const vie::ClassificationBatch cpu_batch = cpu_engine.infer_classification(image, runtime, nullptr);

    vie::ModelConfig cuda_config = cpu_config;
    cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
    cuda_config.device = vie::DeviceType::Cuda;
    vie::InferenceEngine cuda_engine;
    cuda_engine.load(cuda_config);
    vie::ClassificationInferTiming timing;
    const vie::ClassificationBatch cuda_batch = cuda_engine.infer_classification(gpu_image, runtime, &timing);
    ASSERT_EQ(cuda_batch.results[0].labels.size(), cpu_batch.results[0].labels.size()) << model.file;
    // Phase 13: top_k = 5 -> 5 * 8 bytes on the bus instead of the full logits.
    EXPECT_EQ(timing.d2h_bytes, 40u) << model.file;
    EXPECT_TRUE(timing.includes_d2h) << model.file;
    EXPECT_EQ(cuda_batch.results[0].execution_device, vie::DeviceType::Cuda) << model.file;
    EXPECT_EQ(cuda_batch.results[0].score_type, model.score_type) << model.file;
    const float gap = std::fabs(cpu_batch.results[0].labels[0].score - cpu_batch.results[0].labels[1].score);
    for (std::size_t index = 0; index < cpu_batch.results[0].labels.size(); ++index) {
      if (gap > 1e-4f) {
        EXPECT_EQ(cuda_batch.results[0].labels[index].class_id, cpu_batch.results[0].labels[index].class_id)
            << model.file;
      }
      EXPECT_NEAR(cuda_batch.results[0].labels[index].score, cpu_batch.results[0].labels[index].score, 1e-3f)
          << model.file;
    }
  }
  cudaFree(device_pixels);
}
#endif
#endif
