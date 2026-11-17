#include "test_util.hpp"

#include "vie/decoder/classification.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest.hpp"
#include "vie/model/manifest_validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/backend/onnxruntime_backend.hpp"
#endif
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include "vie/backend/onnxruntime_cuda_backend.hpp"

#include <cuda_runtime.h>
#endif

namespace {

vie::Tensor fp32_logits(const std::vector<std::int64_t>& shape, const std::vector<float>& values) {
  vie::Tensor tensor = vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32, shape);
  float* data = tensor.mutable_data<float>();
  for (std::size_t i = 0; i < values.size() && i < tensor.numel(); ++i) data[i] = values[i];
  return tensor;
}

vie::ClassificationDecodeRequest logits_request(int classes, int top_k) {
  vie::ClassificationDecodeRequest request;
  request.output_name = "logits";
  request.class_axis = 1;
  request.num_classes = classes;
  request.top_k = top_k;
  request.score_type = "logits";
  return request;
}

vie::DecoderInput named_logits(vie::Tensor tensor) {
  vie::DecoderInput input;
  input.outputs.emplace("logits", std::move(tensor));
  return input;
}

std::vector<vie::Classification> reference_top_k(const std::vector<float>& scores, int top_k) {
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    if (scores[static_cast<std::size_t>(left)] != scores[static_cast<std::size_t>(right)]) {
      return scores[static_cast<std::size_t>(left)] > scores[static_cast<std::size_t>(right)];
    }
    return left < right;
  });
  std::vector<vie::Classification> labels;
  for (int rank = 0; rank < top_k; ++rank) {
    vie::Classification label;
    label.class_id = order[static_cast<std::size_t>(rank)];
    label.score = scores[static_cast<std::size_t>(label.class_id)];
    labels.push_back(label);
  }
  return labels;
}

void expect_matches_reference(const std::vector<float>& scores, int top_k,
                              const std::vector<std::string>& names = {}) {
  vie::ClassificationDecodeRequest request = logits_request(static_cast<int>(scores.size()), top_k);
  request.class_names = names;
  const vie::ClassificationBatch batch = vie::ClassificationDecoder{}.decode_classification(
      named_logits(fp32_logits({1, static_cast<std::int64_t>(scores.size())}, scores)), request,
      vie::DeviceContext{});
  const std::vector<vie::Classification> expected = reference_top_k(scores, top_k);
  ASSERT_EQ(batch.results.size(), 1u);
  ASSERT_EQ(batch.results[0].labels.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(batch.results[0].labels[i].class_id, expected[i].class_id);
    EXPECT_FLOAT_EQ(batch.results[0].labels[i].score, expected[i].score);
    if (!names.empty()) {
      EXPECT_EQ(batch.results[0].labels[i].class_name, names[static_cast<std::size_t>(expected[i].class_id)]);
    } else {
      EXPECT_TRUE(batch.results[0].labels[i].class_name.empty());
    }
  }
}

std::string thrown_message(const std::function<void()>& call) {
  try {
    call();
  } catch (const vie::VieError& error) {
    return error.what();
  }
  return {};
}

vie::ModelManifest classification_manifest(int classes, const std::string& protocol) {
  vie::ModelManifest manifest;
  manifest.schema_version = "1.0";
  manifest.model_name = "classification_fixture";
  manifest.task = vie::TaskType::Classification;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = classes;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  manifest.input.interpolation = "bilinear";
  vie::OutputDesc output;
  output.name = "logits";
  output.shape = {1, classes};
  output.dtype = vie::DataType::Fp32;
  manifest.outputs.push_back(std::move(output));
  manifest.decoder.protocol = protocol;
  manifest.classification.output_name = "logits";
  manifest.classification.class_axis = 1;
  manifest.classification.score_type = "logits";
  return manifest;
}

vie::ImageInput tiny_image(std::vector<std::uint8_t>& storage) {
  storage.assign(8 * 8 * 3, 7);
  vie::ImageInput image;
  image.data = storage.data();
  image.width = 8;
  image.height = 8;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

}  // namespace

TEST(ClassificationManifest, RoundTripKeepsClassificationFields) {
  vie::ModelManifest manifest = classification_manifest(3, "classification_logits_raw");
  manifest.classification.class_names = {"cat", "dog", "bird"};
  manifest.classification.score_type = "probabilities";
  const vie::ModelManifest loaded =
      vie::ModelManifest::from_json_string(manifest.to_json_string());
  EXPECT_EQ(loaded.task, vie::TaskType::Classification);
  EXPECT_EQ(loaded.num_classes, 3);
  EXPECT_EQ(loaded.classification.output_name, "logits");
  ASSERT_TRUE(loaded.classification.class_axis.has_value());
  EXPECT_EQ(*loaded.classification.class_axis, 1);
  EXPECT_EQ(loaded.classification.score_type, "probabilities");
  EXPECT_EQ(loaded.classification.class_names, (std::vector<std::string>{"cat", "dog", "bird"}));
  const vie::ManifestValidation report = vie::ManifestValidator{}.validate(loaded);
  EXPECT_TRUE(report.schema_ok()) << vie::format_issues(report.schema_issues);
  EXPECT_TRUE(report.runnable()) << vie::format_issues(report.unresolved_critical);
}

TEST(ClassificationManifest, DetectionJsonOmitsClassification) {
  const vie::ModelManifest detection =
      vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  const std::string json = detection.to_json_string();
  EXPECT_EQ(json.find("\"classification\""), std::string::npos);
  const vie::ModelManifest again = vie::ModelManifest::from_json_string(json);
  EXPECT_TRUE(again.classification.output_name.empty());
  EXPECT_FALSE(again.classification.class_axis.has_value());
  EXPECT_TRUE(again.classification.score_type.empty());
  EXPECT_TRUE(again.classification.class_names.empty());
  EXPECT_EQ(again.task, vie::TaskType::Detection);
  EXPECT_EQ(again.decoder.protocol, "mock");
}

TEST(ClassificationManifest, SegmentationJsonOmitsClassification) {
  vie::ModelManifest manifest;
  manifest.schema_version = "1.0";
  manifest.task = vie::TaskType::Segmentation;
  manifest.num_classes = 1;
  manifest.segmentation.prototype_name = "protos";
  manifest.ensure_provenance();
  const std::string json = manifest.to_json_string();
  EXPECT_NE(json.find("\"segmentation\""), std::string::npos);
  EXPECT_EQ(json.find("\"classification\""), std::string::npos);
}

TEST(ClassificationManifest, MissingObjectDoesNotGuessSemantics) {
  constexpr char kJson[] = R"({
    "schema_version": "1.0",
    "task": "classification",
    "num_classes": 4,
    "decoder": {"protocol": "classification_logits_raw"},
    "outputs": [{"name": "logits", "shape": [1, 4], "dtype": "fp32"}]
  })";
  const vie::ModelManifest manifest = vie::ModelManifest::from_json_string(kJson);
  EXPECT_TRUE(manifest.classification.output_name.empty());
  EXPECT_FALSE(manifest.classification.class_axis.has_value());
  EXPECT_TRUE(manifest.classification.score_type.empty());
  const vie::ManifestValidation report = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(report.schema_ok()) << vie::format_issues(report.schema_issues);
  EXPECT_FALSE(report.runnable());
  const std::string unresolved = vie::format_issues(report.unresolved_critical);
  EXPECT_NE(unresolved.find("classification.output_name"), std::string::npos);
  EXPECT_NE(unresolved.find("classification.class_axis"), std::string::npos);
  EXPECT_NE(unresolved.find("classification.score_type"), std::string::npos);
}

TEST(ClassificationManifest, RejectsNameCountAndIllegalAxis) {
  vie::ModelManifest mismatch = classification_manifest(3, "classification_logits_raw");
  mismatch.classification.class_names = {"only_one"};
  const vie::ManifestValidation names = vie::ManifestValidator{}.validate(mismatch);
  EXPECT_FALSE(names.schema_ok());

  EXPECT_THROW(vie::ModelManifest::from_json_string(R"({
    "schema_version": "1.0",
    "classification": {"class_axis": -1}
  })"),
               vie::VieError);

  vie::ModelManifest axis = classification_manifest(4, "classification_logits_raw");
  axis.classification.class_axis = 2;
  const vie::ManifestValidation illegal = vie::ManifestValidator{}.validate(axis);
  EXPECT_FALSE(illegal.schema_ok());
}

TEST(ClassificationDecoder, TopKMatchesIndependentReference) {
  expect_matches_reference({0.2f, 1.5f, -0.4f, 0.7f}, 1);
  expect_matches_reference({0.2f, 1.5f, -0.4f, 0.7f}, 3);
  expect_matches_reference({-8.f, -0.25f, -3.f, -8.f}, 2);
  expect_matches_reference({1.0e20f, -1.0e20f, 1.0e19f}, 2);
  expect_matches_reference({2.f, 5.f, 5.f, 1.f}, 3, {"a", "b", "c", "d"});
}

TEST(ClassificationDecoder, RejectsNonFiniteTopKAndShape) {
  vie::ClassificationDecoder decoder;
  const vie::DeviceContext cpu;
  auto scores = [](std::vector<float> values) {
    return named_logits(fp32_logits({1, static_cast<std::int64_t>(values.size())}, values));
  };
  const std::string nan = thrown_message([&] {
    decoder.decode_classification(scores({1.f, 2.f, std::nanf("")}), logits_request(3, 1), cpu);
  });
  EXPECT_NE(nan.find("logits"), std::string::npos);
  EXPECT_NE(nan.find("class_id 2"), std::string::npos);
  EXPECT_NE(nan.find("NaN"), std::string::npos);

  const std::string positive_inf = thrown_message([&] {
    decoder.decode_classification(
        scores({1.f, std::numeric_limits<float>::infinity(), 0.f}), logits_request(3, 1), cpu);
  });
  EXPECT_NE(positive_inf.find("class_id 1"), std::string::npos);
  EXPECT_NE(positive_inf.find("infinite"), std::string::npos);

  const std::string negative_inf = thrown_message([&] {
    decoder.decode_classification(
        scores({-std::numeric_limits<float>::infinity(), 1.f}), logits_request(2, 1), cpu);
  });
  EXPECT_NE(negative_inf.find("class_id 0"), std::string::npos);
  EXPECT_NE(negative_inf.find("infinite"), std::string::npos);

  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_classification(scores({1.f, 2.f}), logits_request(2, 0), cpu));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_classification(scores({1.f, 2.f, 3.f}), logits_request(3, 4), cpu));

  vie::ClassificationDecodeRequest dtype = logits_request(2, 1);
  vie::DecoderInput wrong_dtype;
  wrong_dtype.outputs.emplace("logits", vie::Tensor::zeros(cpu, vie::DataType::Int32, {1, 2}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decoder.decode_classification(wrong_dtype, dtype, cpu));

  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_classification(named_logits(fp32_logits({4}, {1.f, 2.f, 3.f, 4.f})),
                                                 logits_request(4, 1), cpu));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_classification(
                       named_logits(fp32_logits({1, 1, 4}, {1.f, 2.f, 3.f, 4.f})), logits_request(4, 1), cpu));

  vie::ClassificationDecodeRequest axis = logits_request(4, 1);
  axis.class_axis = 0;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_classification(scores({1.f, 2.f, 3.f, 4.f}), axis, cpu));

  vie::ClassificationDecodeRequest names = logits_request(2, 1);
  names.class_names = {"only"};
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_classification(scores({1.f, 2.f}), names, cpu));

  vie::ClassificationDecodeRequest probabilities = logits_request(2, 1);
  probabilities.score_type = "probabilities";
  const std::string unsupported = thrown_message([&] {
    decoder.decode_classification(scores({1.f, 2.f}), probabilities, cpu);
  });
  EXPECT_NE(unsupported.find("Softmax"), std::string::npos);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   decoder.decode_classification(scores({1.f, 2.f}), probabilities, cpu));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   decoder.decode_classification(scores({1.f, 2.f}), logits_request(2, 1),
                                                 vie::DeviceContext(vie::DeviceType::Cuda, 0)));
}

TEST(ClassificationDecoder, RejectsANonContiguousTensor) {
  vie::Tensor packed = fp32_logits({1, 4}, {1.f, 2.f, 3.f, 4.f});
  vie::Tensor viewed = packed.view({1, 2}, {4, 2}, 0);
  EXPECT_FALSE(viewed.contiguous());
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::ClassificationDecoder{}.decode_classification(
                       named_logits(std::move(viewed)), logits_request(2, 1), vie::DeviceContext{}));
}

TEST(ClassificationDecoder, ClassAxisZeroReadsTheSlowDimension) {
  vie::ClassificationDecodeRequest request = logits_request(3, 2);
  request.class_axis = 0;
  const vie::ClassificationBatch batch = vie::ClassificationDecoder{}.decode_classification(
      named_logits(fp32_logits({3, 1}, {-2.f, 4.f, 4.f})), request, vie::DeviceContext{});
  const std::vector<vie::Classification> expected = reference_top_k({-2.f, 4.f, 4.f}, 2);
  ASSERT_EQ(batch.results[0].labels.size(), 2u);
  EXPECT_EQ(batch.results[0].labels[0].class_id, expected[0].class_id);
  EXPECT_EQ(batch.results[0].labels[1].class_id, expected[1].class_id);
  EXPECT_FLOAT_EQ(batch.results[0].labels[0].score, expected[0].score);
  EXPECT_EQ(batch.results[0].output_bytes, 12u);
  EXPECT_EQ(batch.results[0].d2h_bytes, 0u);
  EXPECT_EQ(batch.results[0].execution_device, vie::DeviceType::Cpu);
}

TEST(ClassificationEngine, MockTopKAndTaskDispatch) {
  vie::InferenceEngine engine;
  vie::ModelConfig config;
  config.manifest = classification_manifest(4, "classification_logits_raw");
  config.backend = vie::BackendKind::Auto;
  config.device = vie::DeviceType::Cpu;
  engine.load(config);
  EXPECT_TRUE(engine.loaded());
  EXPECT_EQ(engine.decoders().create("classification_logits_raw")->protocol(),
            "classification_logits_raw");

  vie::ImageInput blank;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.infer(blank));

  std::vector<std::uint8_t> pixels;
  vie::ClassificationRuntimeConfig runtime;
  runtime.top_k = 2;
  vie::ClassificationInferTiming timing;
  const vie::ClassificationBatch batch = engine.infer_classification(tiny_image(pixels), runtime, &timing);
  ASSERT_EQ(batch.results.size(), 1u);
  ASSERT_EQ(batch.results[0].labels.size(), 2u);
  EXPECT_EQ(batch.results[0].labels[0].class_id, 3);
  EXPECT_FLOAT_EQ(batch.results[0].labels[0].score, 4.f);
  EXPECT_EQ(batch.results[0].labels[1].class_id, 2);
  EXPECT_FLOAT_EQ(batch.results[0].labels[1].score, 3.f);
  EXPECT_EQ(batch.results[0].output_bytes, 16u);
  EXPECT_EQ(batch.results[0].d2h_bytes, 0u);
  EXPECT_EQ(timing.d2h_bytes, 0u);
  EXPECT_FALSE(timing.includes_d2h);
  EXPECT_GE(timing.execute_ms, 0.0);
  EXPECT_GE(timing.topk_ms, 0.0);
  EXPECT_DOUBLE_EQ(timing.decode_ms, timing.d2h_ms + timing.topk_ms);

  vie::InferenceEngine detection;
  vie::ModelConfig detection_config;
  detection_config.manifest = vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  detection.load(detection_config);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   detection.infer_classification(tiny_image(pixels), runtime));
  vie::BackendRequest request;
  request.inputs.emplace("images", vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32, {1, 3, 8, 8}));
  const vie::RawPredictions predictions = detection.run(request);
  ASSERT_EQ(predictions.detections.size(), 1u);
  EXPECT_FLOAT_EQ(predictions.detections[0].score, 0.9f);

  vie::ModelConfig mismatch;
  mismatch.manifest = classification_manifest(4, "mock");
  vie::InferenceEngine wrong_protocol;
  wrong_protocol.load(mismatch);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   wrong_protocol.infer_classification(tiny_image(pixels), runtime));
}

TEST(ClassificationEngine, MissingBackendIsExplicit) {
#if !defined(VIE_HAS_ONNXRUNTIME)
  vie::ModelConfig config;
  config.manifest = classification_manifest(4, "classification_logits_raw");
  config.manifest->model_format = vie::ModelFormat::Onnx;
  config.manifest->model_path = "missing.onnx";
  config.backend = vie::BackendKind::OnnxRuntime;
  config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(config));
  EXPECT_FALSE(engine.loaded());
#elif !defined(VIE_HAS_ONNXRUNTIME_CUDA)
  vie::ModelConfig config;
  config.manifest = classification_manifest(4, "classification_logits_raw");
  config.manifest->model_format = vie::ModelFormat::Onnx;
  config.manifest->model_path = "missing.onnx";
  config.backend = vie::BackendKind::OnnxRuntimeCuda;
  config.device = vie::DeviceType::Cuda;
  vie::InferenceEngine engine;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(config));
  EXPECT_FALSE(engine.loaded());
#else
  GTEST_SKIP() << "this build has both ONNX Runtime backends";
#endif
}

#if defined(VIE_HAS_ONNXRUNTIME)
namespace {

vie::ModelManifest resnet_manifest() {
  vie::ModelManifest manifest;
  manifest.schema_version = "1.0";
  manifest.model_name = "resnet50_untrained";
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
  vie::OutputDesc output;
  output.name = "logits";
  output.shape = {1, 1000};
  output.dtype = vie::DataType::Fp32;
  manifest.outputs.push_back(std::move(output));
  manifest.decoder.protocol = "classification_logits_raw";
  manifest.classification.output_name = "logits";
  manifest.classification.class_axis = 1;
  manifest.classification.score_type = "logits";
  manifest.ensure_provenance();
  return manifest;
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

void record_path(std::ostream& out, const char* name, const vie::ClassificationInferTiming& timing,
                 const vie::ClassificationBatch& batch) {
  out << name << " execute_ms " << timing.execute_ms << " d2h_ms " << timing.d2h_ms << " d2h_bytes "
      << timing.d2h_bytes << " topk_ms " << timing.topk_ms << " decode_ms " << timing.decode_ms
      << " output_bytes " << timing.output_bytes << " labels " << batch.results.at(0).labels.size()
      << " top_class " << batch.results.at(0).labels.at(0).class_id << "\n";
}

}  // namespace

TEST(ClassificationOnnx, UntrainedResNetExecutesAndDecodes) {
  const std::string model = "/tmp/resnet50_untrained_static.onnx";
  ASSERT_TRUE(std::filesystem::exists(model)) << model;
  vie::ModelConfig config;
  config.model_path = model;
  config.manifest = resnet_manifest();
  config.backend = vie::BackendKind::OnnxRuntime;
  config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  engine.load(config);

  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = gray_image(pixels);
  vie::ClassificationRuntimeConfig runtime;
  runtime.top_k = 5;
  vie::ClassificationInferTiming warmup_timing;
  engine.infer_classification(image, runtime, &warmup_timing);

  vie::ClassificationInferTiming timing;
  const vie::ClassificationBatch batch = engine.infer_classification(image, runtime, &timing);
  ASSERT_EQ(batch.results.size(), 1u);
  ASSERT_EQ(batch.results[0].labels.size(), 5u);
  EXPECT_EQ(batch.results[0].num_classes, 1000);
  EXPECT_EQ(batch.results[0].output_bytes, 4000u);
  EXPECT_EQ(timing.output_bytes, 4000u);
  EXPECT_EQ(timing.d2h_bytes, 0u);
  EXPECT_EQ(batch.results[0].execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(batch.results[0].score_type, "logits");
  for (const vie::Classification& label : batch.results[0].labels) {
    EXPECT_TRUE(std::isfinite(label.score));
    EXPECT_GE(label.class_id, 0);
    EXPECT_LT(label.class_id, 1000);
  }

  vie::BackendRequest request;
  request.inputs.emplace("input", engine.preprocess(image).output);
  const vie::BackendOutput output = engine.execute(request);
  const vie::Tensor& logits = output.tensors.at("logits").tensor;
  EXPECT_EQ(logits.shape(), (std::vector<std::int64_t>{1, 1000}));
  EXPECT_EQ(logits.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(logits.device(), vie::DeviceType::Cpu);
  EXPECT_EQ(logits.nbytes(), 4000u);

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
  vie::OnnxRuntimeCudaBackend cuda_backend;
  EXPECT_FALSE(cuda_backend.cpu_ep_fallback());
  vie::ModelConfig cuda_config = config;
  cuda_config.device = vie::DeviceType::Cuda;
  cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
  vie::InferenceEngine cuda_engine;
  cuda_engine.load(cuda_config);
  void* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(&device_pixels, pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput gpu_image = image;
  gpu_image.data = device_pixels;
  gpu_image.device = vie::DeviceType::Cuda;
  cuda_engine.infer_classification(gpu_image, runtime, &warmup_timing);
  vie::ClassificationInferTiming cuda_timing;
  const vie::ClassificationBatch cuda_batch =
      cuda_engine.infer_classification(gpu_image, runtime, &cuda_timing);
  ASSERT_EQ(cuda_batch.results[0].labels.size(), 5u);
  EXPECT_EQ(cuda_timing.output_bytes, 4000u);
  EXPECT_EQ(cuda_timing.d2h_bytes, 4000u);
  EXPECT_EQ(cuda_batch.results[0].d2h_bytes, 4000u);
  EXPECT_TRUE(cuda_timing.includes_d2h);
  EXPECT_EQ(cuda_batch.results[0].execution_device, vie::DeviceType::Cpu);
  EXPECT_LT(cuda_batch.results[0].labels.size() * sizeof(float), cuda_timing.d2h_bytes);

  const vie::PreprocessResult prepared = cuda_engine.preprocess(gpu_image);
  vie::BackendRequest cuda_request;
  cuda_request.inputs.emplace("input", prepared.output);
  const vie::BackendOutput cuda_output = cuda_engine.execute(cuda_request);
  const vie::Tensor& cuda_logits = cuda_output.tensors.at("logits").tensor;
  EXPECT_TRUE(cuda_logits.is_cuda());
  EXPECT_EQ(cuda_logits.shape(), (std::vector<std::int64_t>{1, 1000}));
  EXPECT_EQ(cuda_logits.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(cuda_logits.nbytes(), 4000u);
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, cuda_logits.device_data()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

  vie::ModelConfig gpu_cpu = config;
  gpu_cpu.device = vie::DeviceType::Cpu;
  gpu_cpu.backend = vie::BackendKind::OnnxRuntime;
  vie::InferenceEngine gpu_package_cpu;
  gpu_package_cpu.load(gpu_cpu);
  vie::ClassificationInferTiming gpu_cpu_timing;
  const vie::ClassificationBatch gpu_cpu_batch =
      gpu_package_cpu.infer_classification(image, runtime, &gpu_cpu_timing);
  EXPECT_EQ(gpu_cpu_timing.d2h_bytes, 0u);
  EXPECT_EQ(gpu_cpu_batch.results[0].labels.size(), 5u);

  std::ofstream record("/tmp/vie_phase62_gpu_pkg.txt");
  record_path(record, "gpu_package_cpu_ep", gpu_cpu_timing, gpu_cpu_batch);
  record_path(record, "strict_cuda_ep", cuda_timing, cuda_batch);
  cudaFree(device_pixels);
#else
  std::ofstream record("/tmp/vie_phase62_cpu_pkg.txt");
  record_path(record, "cpu_package_cpu_ep", timing, batch);
#endif
}
#endif
