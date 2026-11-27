// Phase 13 end-to-end checks against real ONNX exports:
//  * Classification: torchvision ResNet18 (untrained) exported to opset 17 by
//    tools/export_classification_models.py; 1000 classes, input "input"
//    [1, 3, 224, 224], outputs "logits" and (softmax export) "probabilities".
//  * Semantic segmentation: DeepLabV3-MobileNetV3-Large (COCO/VOC weights) at
//    testdata/onnx/deeplabv3_mobilenet_v3_large_520_opset18.onnx; 21 classes,
//    input "image" [1, 3, 520, 520], output "out" [1, 21, 520, 520].
// Only compiled when VIE_ENABLE_CUDA is on.

#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/backend/onnxruntime_cuda_backend.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using vie::DeviceType;

const std::filesystem::path kModelDir{"/tmp/vie_phase66"};

void mark_user(vie::ModelManifest& manifest, const char* field, const char* evidence) {
  manifest.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {evidence}};
}

vie::ModelManifest classification_manifest(const std::string& output_name,
                                           const std::string& protocol,
                                           const std::string& score_type) {
  vie::ModelManifest manifest;
  manifest.schema_version = "1.0";
  manifest.model_name = "resnet18_untrained";
  manifest.task = vie::TaskType::Classification;
  manifest.model_format = vie::ModelFormat::Onnx;
  manifest.num_classes = 1000;
  manifest.input.name = "input";
  manifest.input.shape = {1, 3, 224, 224};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  manifest.input.interpolation = "bilinear";
  mark_user(manifest, "input.color", "torchvision ToTensor input, supplied by the caller");
  mark_user(manifest, "input.normalization", "divide by 255, supplied by the caller");
  mark_user(manifest, "input.resize", "export contract, supplied by the caller");
  mark_user(manifest, "input.interpolation", "bilinear resize, supplied by the caller");
  vie::OutputDesc output;
  output.name = output_name;
  output.shape = {1, 1000};
  output.dtype = vie::DataType::Fp32;
  output.semantic = score_type;
  manifest.outputs.push_back(std::move(output));
  manifest.decoder.protocol = protocol;
  manifest.classification.output_name = output_name;
  manifest.classification.class_axis = 1;
  manifest.classification.score_type = score_type;
  manifest.ensure_provenance();
  return manifest;
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

vie::ImageInput load_bus(std::vector<std::uint8_t>& pixels) {
  std::ifstream image_file(vie_fixture("testdata/images/bus.ppm"), std::ios::binary);
  std::string magic;
  int width = 0;
  int height = 0;
  int max_value = 0;
  image_file >> magic >> width >> height >> max_value;
  image_file.get();
  EXPECT_EQ(magic, "P6");
  pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
  image_file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  EXPECT_TRUE(image_file);
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

vie::ModelConfig semantic_config(DeviceType device) {
  const std::string path = vie_fixture("testdata/onnx/deeplabv3_mobilenet_v3_large_520_opset18.onnx");
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  manifest.input.interpolation = "bilinear";
  mark_user(manifest, "input.color", "BGR->RGB conversion, supplied by the caller");
  mark_user(manifest, "input.normalization", "divide by 255, supplied by the caller");
  mark_user(manifest, "input.resize", "export contract, supplied by the caller");
  mark_user(manifest, "input.interpolation", "bilinear resize, supplied by the caller");
  vie::ModelConfig config;
  config.model_path = path;
  config.manifest = std::move(manifest);
  config.device = device;
  config.backend = device == DeviceType::Cuda ? vie::BackendKind::OnnxRuntimeCuda
                                              : vie::BackendKind::OnnxRuntime;
  return config;
}

vie::ImageInput cuda_image(vie::ImageInput image, const std::vector<std::uint8_t>& storage,
                           void*& device_pixels) {
  device_pixels = nullptr;
  EXPECT_EQ(cudaMalloc(&device_pixels, storage.size()), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(device_pixels, storage.data(), storage.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  image.data = device_pixels;
  image.device = DeviceType::Cuda;
  return image;
}

struct ClassificationSample {
  vie::ClassificationBatch batch;
  double execute_ms = 0;
  double d2h_ms = 0;
  double validate_ms = 0;
  double topk_ms = 0;
  double decode_ms = 0;
  double total_ms = 0;
  double d2h_bytes = 0;
};

ClassificationSample measure_classification(vie::InferenceEngine& engine,
                                            const vie::ImageInput& image,
                                            const vie::ClassificationRuntimeConfig& runtime) {
  constexpr int kSamples = 5;
  for (int index = 0; index < 2; ++index) engine.infer_classification(image, runtime, nullptr);
  ClassificationSample sample;
  for (int index = 0; index < kSamples; ++index) {
    vie::ClassificationInferTiming timing;
    sample.batch = engine.infer_classification(image, runtime, &timing);
    sample.execute_ms += timing.execute_ms;
    sample.d2h_ms += timing.d2h_ms;
    sample.validate_ms += timing.validate_ms;
    sample.topk_ms += timing.topk_ms;
    sample.decode_ms += timing.decode_ms;
    sample.total_ms += timing.total_ms;
    sample.d2h_bytes = static_cast<double>(timing.d2h_bytes);
  }
  sample.execute_ms /= kSamples;
  sample.d2h_ms /= kSamples;
  sample.validate_ms /= kSamples;
  sample.topk_ms /= kSamples;
  sample.decode_ms /= kSamples;
  sample.total_ms /= kSamples;
  return sample;
}

}  // namespace

TEST(CudaDecoderE2E, ClassificationTopKMatchesCpuOnResNet18Logits) {
  const std::filesystem::path path = kModelDir / "resnet18_untrained_opset17.onnx";
  if (!std::filesystem::is_regular_file(path)) {
    GTEST_SKIP() << "run tools/export_classification_models.py to create " << path;
  }
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = pattern_image(pixels);

  vie::ModelConfig cpu_config;
  cpu_config.model_path = path.string();
  cpu_config.manifest = classification_manifest("logits", "classification_logits_raw", "logits");
  cpu_config.backend = vie::BackendKind::OnnxRuntime;
  cpu_config.device = DeviceType::Cpu;
  vie::InferenceEngine cpu_engine;
  cpu_engine.load(cpu_config);

  vie::ModelConfig cuda_config = cpu_config;
  cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
  cuda_config.device = DeviceType::Cuda;
  ASSERT_FALSE(cuda_config.cuda_cpu_ep_fallback);
  EXPECT_FALSE(vie::OnnxRuntimeCudaBackend{}.cpu_ep_fallback());
  vie::InferenceEngine cuda_engine;
  cuda_engine.load(cuda_config);

  void* device_pixels = nullptr;
  const vie::ImageInput gpu_image = cuda_image(image, pixels, device_pixels);
  ASSERT_NE(device_pixels, nullptr);

  vie::ClassificationRuntimeConfig runtime;
  runtime.top_k = 5;
  const ClassificationSample cpu = measure_classification(cpu_engine, image, runtime);
  const ClassificationSample gpu = measure_classification(cuda_engine, gpu_image, runtime);

  ASSERT_EQ(cpu.batch.results.size(), 1u);
  ASSERT_EQ(gpu.batch.results.size(), 1u);
  ASSERT_EQ(cpu.batch.results[0].labels.size(), 5u);
  ASSERT_EQ(gpu.batch.results[0].labels.size(), 5u);
  EXPECT_EQ(gpu.batch.results[0].execution_device, DeviceType::Cuda);
  EXPECT_EQ(gpu.batch.results[0].num_classes, 1000);
  EXPECT_EQ(gpu.batch.results[0].output_bytes, 4000u);
  EXPECT_EQ(gpu.batch.results[0].d2h_bytes, 5u * 8u);
  // Decoder parity is exact and covered by CudaDecoder* unit tests that feed
  // the same logits to both paths. Here the two engines run different EP
  // kernels, so the scores may disagree in the last places (~1e-4 observed);
  // the repo convention for cross-EP comparisons is 1e-3.
  constexpr float kCrossEpTolerance = 1e-3f;
  for (std::size_t rank = 0; rank < 5; ++rank) {
    EXPECT_EQ(cpu.batch.results[0].labels[rank].class_id,
              gpu.batch.results[0].labels[rank].class_id)
        << "rank " << rank;
    EXPECT_NEAR(cpu.batch.results[0].labels[rank].score,
                gpu.batch.results[0].labels[rank].score, kCrossEpTolerance)
        << "rank " << rank;
  }

  std::ofstream record("/tmp/vie_phase13_e2e.txt", std::ios::app);
  record << "classification resnet18 logits top_k 5\n";
  record << "  cpu execute_ms " << cpu.execute_ms << " validate_ms " << cpu.validate_ms
         << " topk_ms " << cpu.topk_ms << " decode_ms " << cpu.decode_ms << " total_ms "
         << cpu.total_ms << " d2h_bytes 0\n";
  record << "  cuda execute_ms " << gpu.execute_ms << " validate_ms " << gpu.validate_ms
         << " topk_ms " << gpu.topk_ms << " d2h_ms " << gpu.d2h_ms << " decode_ms "
         << gpu.decode_ms << " total_ms " << gpu.total_ms << " d2h_bytes " << gpu.d2h_bytes
         << " full_logits_bytes 4000\n";

  cudaFree(device_pixels);
}

TEST(CudaDecoderE2E, ClassificationProbabilitiesMatchCpuOnSoftmaxResNet18) {
  const std::filesystem::path path = kModelDir / "resnet18_untrained_opset17_softmax.onnx";
  if (!std::filesystem::is_regular_file(path)) {
    GTEST_SKIP() << "run tools/export_classification_models.py to create " << path;
  }
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = pattern_image(pixels);

  vie::ModelConfig cpu_config;
  cpu_config.model_path = path.string();
  cpu_config.manifest =
      classification_manifest("probabilities", "classification_probabilities_raw", "probabilities");
  cpu_config.backend = vie::BackendKind::OnnxRuntime;
  cpu_config.device = DeviceType::Cpu;
  vie::InferenceEngine cpu_engine;
  cpu_engine.load(cpu_config);

  vie::ModelConfig cuda_config = cpu_config;
  cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
  cuda_config.device = DeviceType::Cuda;
  vie::InferenceEngine cuda_engine;
  cuda_engine.load(cuda_config);

  void* device_pixels = nullptr;
  const vie::ImageInput gpu_image = cuda_image(image, pixels, device_pixels);
  ASSERT_NE(device_pixels, nullptr);

  vie::ClassificationRuntimeConfig runtime;
  runtime.top_k = 5;
  const ClassificationSample cpu = measure_classification(cpu_engine, image, runtime);
  const ClassificationSample gpu = measure_classification(cuda_engine, gpu_image, runtime);

  ASSERT_EQ(cpu.batch.results[0].labels.size(), 5u);
  ASSERT_EQ(gpu.batch.results[0].labels.size(), 5u);
  EXPECT_EQ(gpu.batch.results[0].execution_device, DeviceType::Cuda);
  EXPECT_EQ(gpu.batch.results[0].d2h_bytes, 5u * 8u);
  for (std::size_t rank = 0; rank < 5; ++rank) {
    EXPECT_EQ(cpu.batch.results[0].labels[rank].class_id,
              gpu.batch.results[0].labels[rank].class_id)
        << "rank " << rank;
    EXPECT_NEAR(cpu.batch.results[0].labels[rank].score,
                gpu.batch.results[0].labels[rank].score, 1e-5f)
        << "rank " << rank;
    EXPECT_GE(gpu.batch.results[0].labels[rank].score, 0.f);
    EXPECT_LE(gpu.batch.results[0].labels[rank].score, 1.f);
  }

  std::ofstream record("/tmp/vie_phase13_e2e.txt", std::ios::app);
  record << "classification resnet18 probabilities top_k 5\n";
  record << "  cpu execute_ms " << cpu.execute_ms << " validate_ms " << cpu.validate_ms
         << " topk_ms " << cpu.topk_ms << " decode_ms " << cpu.decode_ms << " total_ms "
         << cpu.total_ms << " d2h_bytes 0\n";
  record << "  cuda execute_ms " << gpu.execute_ms << " validate_ms " << gpu.validate_ms
         << " topk_ms " << gpu.topk_ms << " d2h_ms " << gpu.d2h_ms << " decode_ms "
         << gpu.decode_ms << " total_ms " << gpu.total_ms << " d2h_bytes " << gpu.d2h_bytes
         << " full_logits_bytes 4000\n";

  cudaFree(device_pixels);
}

TEST(CudaDecoderE2E, SemanticDeepLabV3ClassMapMatchesCpu) {
  VIE_REQUIRE_MODEL("testdata/onnx/deeplabv3_mobilenet_v3_large_520_opset18.onnx");
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_bus(pixels);

  vie::InferenceEngine cpu_engine;
  cpu_engine.load(semantic_config(DeviceType::Cpu));

  vie::InferenceEngine cuda_engine;
  cuda_engine.load(semantic_config(DeviceType::Cuda));
  EXPECT_FALSE(vie::OnnxRuntimeCudaBackend{}.cpu_ep_fallback());

  void* device_pixels = nullptr;
  const vie::ImageInput gpu_image = cuda_image(image, pixels, device_pixels);
  ASSERT_NE(device_pixels, nullptr);

  constexpr int kSamples = 3;
  cuda_engine.infer_semantic_segmentation(gpu_image);
  cpu_engine.infer_semantic_segmentation(image);

  vie::SemanticSegmentationResult cpu;
  vie::SemanticSegmentationResult gpu;
  vie::SemanticSegmentationInferTiming cpu_timing;
  vie::SemanticSegmentationInferTiming gpu_timing;
  double cpu_total = 0;
  double gpu_total = 0;
  for (int index = 0; index < kSamples; ++index) {
    cpu = cpu_engine.infer_semantic_segmentation(image, &cpu_timing);
    gpu = cuda_engine.infer_semantic_segmentation(gpu_image, &gpu_timing);
    cpu_total += cpu_timing.total_ms;
    gpu_total += gpu_timing.total_ms;
  }
  cpu_total /= kSamples;
  gpu_total /= kSamples;

  ASSERT_EQ(cpu.class_ids.size(), gpu.class_ids.size());
  ASSERT_EQ(gpu.width, image.width);
  ASSERT_EQ(gpu.height, image.height);
  EXPECT_EQ(gpu.num_classes, 21);
  EXPECT_EQ(gpu.execution_device, DeviceType::Cuda);
  const std::size_t logits_bytes =
      static_cast<std::size_t>(21) * 520 * 520 * sizeof(float);
  EXPECT_EQ(gpu.output_bytes, logits_bytes);
  EXPECT_EQ(gpu.d2h_bytes, static_cast<std::size_t>(image.width) * image.height * sizeof(std::int32_t));
  EXPECT_TRUE(gpu_timing.includes_d2h);
  EXPECT_FALSE(cpu_timing.includes_d2h);

  std::size_t matches = 0;
  for (std::size_t index = 0; index < cpu.class_ids.size(); ++index) {
    if (cpu.class_ids[index] == gpu.class_ids[index]) ++matches;
    EXPECT_GE(gpu.class_ids[index], 0);
    EXPECT_LT(gpu.class_ids[index], 21);
  }
  const double pixel_accuracy =
      static_cast<double>(matches) / static_cast<double>(cpu.class_ids.size());

  // Mean IoU over classes present in either map; classes absent from both are
  // left out instead of counting as a perfect match.
  std::vector<std::size_t> intersection(21, 0);
  std::vector<std::size_t> union_count(21, 0);
  for (std::size_t index = 0; index < cpu.class_ids.size(); ++index) {
    const std::size_t left = static_cast<std::size_t>(cpu.class_ids[index]);
    const std::size_t right = static_cast<std::size_t>(gpu.class_ids[index]);
    ++union_count[left];
    if (left != right) ++union_count[right];
    if (left == right) ++intersection[left];
  }
  int present = 0;
  double iou_sum = 0;
  for (int cls = 0; cls < 21; ++cls) {
    if (union_count[static_cast<std::size_t>(cls)] == 0) continue;
    ++present;
    iou_sum += static_cast<double>(intersection[static_cast<std::size_t>(cls)]) /
               static_cast<double>(union_count[static_cast<std::size_t>(cls)]);
  }
  const double mean_iou = present == 0 ? 0.0 : iou_sum / present;

  // Decoder parity is proven exactly in SemanticCudaDecoder with identical
  // logits. Here the two execution providers run the same graph with different
  // kernels, so the maps can differ on a few borderline pixels; the floor
  // documents the measured EP-level agreement instead of hiding it.
  EXPECT_GE(pixel_accuracy, 0.99);
  EXPECT_GE(mean_iou, 0.99);

  std::ofstream record("/tmp/vie_phase13_e2e.txt", std::ios::app);
  record << "semantic deeplabv3 520x520 classes 21 original " << image.width << "x"
         << image.height << " samples " << kSamples << "\n";
  record << "  pixel_accuracy " << pixel_accuracy << " mean_iou " << mean_iou
         << " mismatched_pixels " << (cpu.class_ids.size() - matches) << "\n";
  record << "  cpu validate_ms " << cpu_timing.validate_ms << " argmax_ms " << cpu_timing.argmax_ms
         << " decode_ms " << cpu_timing.decode_ms << " total_ms " << cpu_total << " d2h_bytes 0\n";
  record << "  cuda validate_ms " << gpu_timing.validate_ms << " argmax_ms "
         << gpu_timing.argmax_ms << " d2h_ms " << gpu_timing.d2h_ms << " decode_ms "
         << gpu_timing.decode_ms << " total_ms " << gpu_total << " d2h_bytes " << gpu_timing.d2h_bytes
         << " full_logits_bytes " << logits_bytes << "\n";

  cudaFree(device_pixels);
}