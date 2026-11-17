#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/decoder/yolov8_seg_decoder.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/backend/onnxruntime_backend.hpp"
#endif
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

vie::ModelManifest mock_segmentation_manifest() {
  vie::ModelManifest manifest;
  manifest.model_name = "mock_seg";
  manifest.task = vie::TaskType::Segmentation;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 80;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 8, 8};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  manifest.input.interpolation = "bilinear";
  manifest.decoder.protocol = "yolov8_seg_raw";
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.scores_are_probabilities = true;
  vie::OutputDesc prediction;
  prediction.name = "output0";
  prediction.shape = {1, 116, 4};
  prediction.dtype = vie::DataType::Fp32;
  prediction.semantic = "cxcywh_class_probabilities_mask_coefficients";
  vie::OutputDesc prototype;
  prototype.name = "output1";
  prototype.shape = {1, 32, 2, 2};
  prototype.dtype = vie::DataType::Fp32;
  prototype.layout = vie::Layout::Nchw;
  prototype.semantic = "mask_prototypes";
  manifest.outputs = {prediction, prototype};
  manifest.segmentation.mask_channels = 32;
  manifest.segmentation.prototype_name = "output1";
  manifest.segmentation.prototype_layout = vie::Layout::Nchw;
  manifest.segmentation.prototype_channel_axis = 1;
  manifest.segmentation.prototype_height = 2;
  manifest.segmentation.prototype_width = 2;
  manifest.segmentation.mask_coefficient_activation = "none";
  manifest.segmentation.prototype_activation = "model_output_features";
  manifest.segmentation.has_objectness = false;
  manifest.ensure_provenance();
  return manifest;
}

vie::SegmentationRuntimeConfig runtime_ready() {
  vie::SegmentationRuntimeConfig config;
  config.confidence_threshold = 0.25f;
  config.iou_threshold = 0.45f;
  config.class_selection = vie::ClassSelection::BestClass;
  config.class_aware_nms = true;
  config.pre_nms_top_k = 300;
  config.max_detections = 20;
  config.nms_enabled = true;
  config.retina_masks = true;
  config.binary_mask_gt_zero = true;
  config.max_mask_bytes = 64ull << 20;
  return config;
}

vie::ImageInput tiny_image(std::vector<std::uint8_t>& pixels) {
  pixels.assign(4 * 4 * 3, 1);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = 4;
  image.height = 4;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

void load_mock_segmentation(vie::InferenceEngine& engine) {
  vie::ModelConfig config;
  config.manifest = mock_segmentation_manifest();
  engine.load(config);
}

void expect_config_error(vie::SegmentationRuntimeConfig runtime, vie::ErrorCode code, const char* field) {
  vie::InferenceEngine engine;
  load_mock_segmentation(engine);
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);
  try {
    engine.infer_segmentation(image, runtime);
    FAIL() << "expected " << field;
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), code) << error.what();
    EXPECT_NE(std::string(error.what()).find(field), std::string::npos) << error.what();
  }
}

#if defined(VIE_HAS_ONNXRUNTIME)
struct LoadedImage {
  std::vector<std::uint8_t> pixels;
  vie::ImageInput image;
};

LoadedImage load_bus() {
  std::ifstream image_file(vie_fixture("testdata/images/bus.ppm"), std::ios::binary);
  std::string magic;
  int image_width = 0;
  int image_height = 0;
  int max_value = 0;
  image_file >> magic >> image_width >> image_height >> max_value;
  image_file.get();
  EXPECT_EQ(magic, "P6");
  EXPECT_EQ(max_value, 255);
  LoadedImage loaded;
  loaded.pixels.resize(static_cast<std::size_t>(image_width * image_height * 3));
  image_file.read(reinterpret_cast<char*>(loaded.pixels.data()),
                  static_cast<std::streamsize>(loaded.pixels.size()));
  EXPECT_TRUE(image_file);
  loaded.image.data = loaded.pixels.data();
  loaded.image.width = image_width;
  loaded.image.height = image_height;
  loaded.image.channels = 3;
  loaded.image.color = vie::ColorOrder::Rgb;
  loaded.image.layout = vie::ImageLayout::Hwc;
  loaded.image.dtype = vie::DataType::Uint8;
  return loaded;
}

vie::ModelManifest recognized_segmentation_manifest(const std::string& path) {
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult recognition = vie::default_recognizer_registry().recognize(report);
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, recognition, path);
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  const char* fields[] = {"input.color", "input.normalization", "input.resize", "input.interpolation",
                          "input.padding_value"};
  for (const char* field : fields) {
    manifest.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {"explicit test configuration"}};
  }
  return manifest;
}

void expect_same_batch(const vie::SegmentationBatch& actual, const vie::SegmentationBatch& expected) {
  ASSERT_EQ(actual.instances.size(), expected.instances.size());
  EXPECT_EQ(actual.original_width, expected.original_width);
  EXPECT_EQ(actual.original_height, expected.original_height);
  for (std::size_t i = 0; i < actual.instances.size(); ++i) {
    EXPECT_EQ(actual.instances[i].detection.class_id, expected.instances[i].detection.class_id);
    EXPECT_FLOAT_EQ(actual.instances[i].detection.confidence, expected.instances[i].detection.confidence);
    EXPECT_FLOAT_EQ(actual.instances[i].detection.box.x1, expected.instances[i].detection.box.x1);
    EXPECT_FLOAT_EQ(actual.instances[i].detection.box.y1, expected.instances[i].detection.box.y1);
    EXPECT_FLOAT_EQ(actual.instances[i].detection.box.x2, expected.instances[i].detection.box.x2);
    EXPECT_FLOAT_EQ(actual.instances[i].detection.box.y2, expected.instances[i].detection.box.y2);
    const vie::Mask host = actual.instances[i].mask.to_cpu();
    ASSERT_EQ(host.width(), expected.instances[i].mask.width());
    ASSERT_EQ(host.height(), expected.instances[i].mask.height());
    EXPECT_EQ(std::memcmp(host.binary_data(), expected.instances[i].mask.binary_data(),
                          static_cast<std::size_t>(host.width() * host.height())),
              0);
  }
}

struct StageStats {
  double mean = 0;
  double p50 = 0;
  double p95 = 0;
};

StageStats summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  StageStats stats;
  double sum = 0;
  for (const double value : values) sum += value;
  stats.mean = sum / static_cast<double>(values.size());
  stats.p50 = values[values.size() / 2];
  const std::size_t p95_index = std::min(values.size() - 1, static_cast<std::size_t>(0.95 * values.size()));
  stats.p95 = values[p95_index];
  return stats;
}
#endif

}  // namespace

TEST(EngineSegmentation, UnloadedAndTaskBoundaries) {
  vie::InferenceEngine unloaded;
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, unloaded.infer_segmentation(image, runtime_ready()));

  vie::InferenceEngine detection;
  vie::ModelConfig detection_config;
  detection_config.manifest = vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  detection.load(detection_config);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, detection.infer_segmentation(image, runtime_ready()));
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, detection.infer(image));

  vie::InferenceEngine segmentation;
  load_mock_segmentation(segmentation);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, segmentation.infer(image));
  vie::ModelConfig wrong_protocol;
  wrong_protocol.manifest = mock_segmentation_manifest();
  wrong_protocol.manifest->decoder.protocol = "mock";
  vie::InferenceEngine other;
  other.load(wrong_protocol);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, other.infer_segmentation(image, runtime_ready()));
}

TEST(EngineSegmentation, MissingAndIllegalRuntimeConfigNamesTheField) {
  vie::SegmentationRuntimeConfig config = runtime_ready();
  config.confidence_threshold.reset();
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "confidence_threshold");
  config = runtime_ready();
  config.iou_threshold.reset();
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "iou_threshold");
  config = runtime_ready();
  config.class_selection.reset();
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "class_selection");
  config = runtime_ready();
  config.retina_masks.reset();
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "retina_masks");
  config = runtime_ready();
  config.binary_mask_gt_zero.reset();
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "binary_mask_gt_zero");
  config = runtime_ready();
  config.binary_mask_gt_zero = false;
  expect_config_error(config, vie::ErrorCode::Unsupported, "M > 0");
  config = runtime_ready();
  config.max_mask_bytes.reset();
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "max_mask_bytes");
  config = runtime_ready();
  config.nms_enabled.reset();
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "nms.enabled");
  config = runtime_ready();
  config.confidence_threshold = 1.5f;
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "confidence_threshold");
  config = runtime_ready();
  config.iou_threshold = -0.1f;
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "iou_threshold");
  config = runtime_ready();
  config.pre_nms_top_k = 0;
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "pre_nms_top_k");
  config = runtime_ready();
  config.pre_nms_top_k = 4097;
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "pre_nms_top_k");
  config = runtime_ready();
  config.max_detections = 0;
  expect_config_error(config, vie::ErrorCode::UnresolvedField, "max_detections");

  vie::ModelManifest poisoned = mock_segmentation_manifest();
  poisoned.nms.confidence_threshold = 0.99f;
  poisoned.nms.iou_threshold = 0.1f;
  poisoned.nms.enabled = true;
  poisoned.nms.class_aware = false;
  poisoned.segmentation.retina_masks = false;
  const char* policy_fields[] = {"nms.enabled", "nms.confidence_threshold", "nms.iou_threshold",
                                 "nms.class_aware", "segmentation.retina_masks"};
  for (const char* field : policy_fields) {
    poisoned.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {"manifest policy that must be ignored"}};
  }
  vie::ModelConfig reloaded;
  reloaded.manifest = poisoned;
  vie::InferenceEngine ignores_manifest_policy;
  ignores_manifest_policy.load(reloaded);
  vie::SegmentationRuntimeConfig missing = runtime_ready();
  missing.confidence_threshold.reset();
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = tiny_image(pixels);
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   ignores_manifest_policy.infer_segmentation(image, missing));
  EXPECT_TRUE(ignores_manifest_policy.manifest().decoder.class_selection.empty());
  EXPECT_FLOAT_EQ(*ignores_manifest_policy.manifest().nms.confidence_threshold, 0.99f);
}

TEST(EngineSegmentation, ConflictingManifestShapeIsRejected) {
  vie::ModelConfig config;
  config.manifest = mock_segmentation_manifest();
  config.manifest->segmentation.prototype_height = 9;
  vie::InferenceEngine engine;
  engine.load(config);
  std::vector<std::uint8_t> pixels;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.infer_segmentation(tiny_image(pixels), runtime_ready()));
}

#if defined(VIE_HAS_ONNXRUNTIME)
TEST(EngineSegmentation, CpuEndToEndMatchesDirectPipeline) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  vie::ModelConfig config;
  config.manifest = recognized_segmentation_manifest(path);
  config.model_path = path;
  config.device = vie::DeviceType::Cpu;
  const LoadedImage bus = load_bus();
  vie::SegmentationBatch owned;
  {
    vie::InferenceEngine engine;
    const auto load_start = std::chrono::steady_clock::now();
    engine.load(config);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    EXPECT_FALSE(engine.manifest().segmentation.retina_masks.has_value());
    EXPECT_TRUE(engine.manifest().decoder.class_selection.empty());
    vie::SegmentationInferTiming timing;
    owned = engine.infer_segmentation(bus.image, runtime_ready(), &timing);
    EXPECT_GT(load_ms, 0);
    EXPECT_GT(timing.preprocess_ms, 0);
    EXPECT_GT(timing.execute_ms, 0);
    EXPECT_GT(timing.decode_ms, 0);
    EXPECT_FALSE(timing.includes_h2d);
    EXPECT_FALSE(timing.includes_mask_d2h);
    EXPECT_FALSE(timing.includes_sync);
  }
  ASSERT_FALSE(owned.instances.empty());
  EXPECT_EQ(owned.execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(owned.instances[0].mask.device(), vie::DeviceType::Cpu);
  EXPECT_EQ(owned.instances[0].mask.width(), bus.image.width);
  EXPECT_EQ(owned.instances[0].mask.height(), bus.image.height);
  for (const vie::InstanceSegmentation& instance : owned.instances) {
    EXPECT_GE(instance.detection.confidence, 0.25f);
    const std::uint8_t* pixels = instance.mask.binary_data();
    for (int i = 0; i < instance.mask.width() * instance.mask.height(); ++i) {
      ASSERT_TRUE(pixels[i] == 0 || pixels[i] == 1);
    }
  }

  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(vie::preprocess_config_from_manifest(*config.manifest, bus.image.color, bus.image.layout));
  const vie::PreprocessResult prepared = preprocessor.execute(bus.image, vie::DeviceContext{});
  vie::BackendConfig backend_config;
  backend_config.model_path = path;
  backend_config.model_format = vie::ModelFormat::Onnx;
  backend_config.device = vie::DeviceType::Cpu;
  backend_config.input.name = "images";
  backend_config.input.shape = {1, 3, 640, 640};
  backend_config.input.dtype = vie::DataType::Fp32;
  vie::OnnxRuntimeBackend backend;
  backend.prepare(backend_config);
  vie::BackendRequest request;
  request.inputs.emplace("images", prepared.output);
  const vie::BackendOutput output = backend.execute(request);
  vie::DecoderConfig decoder;
  decoder.output_name = "output0";
  decoder.prototype_name = "output1";
  decoder.channel_axis = 1;
  decoder.box_channels = 4;
  decoder.num_classes = 80;
  decoder.mask_channels = 32;
  decoder.box_decode = vie::BoxDecode::DecodedCxcywh;
  decoder.scores_are_probabilities = true;
  decoder.class_selection = vie::ClassSelection::BestClass;
  decoder.confidence_threshold = 0.25f;
  decoder.nms_enabled = true;
  decoder.class_aware_nms = true;
  decoder.iou_threshold = 0.45f;
  decoder.max_detections = 20;
  decoder.pre_nms_top_k = 300;
  decoder.prototype_channel_axis = 1;
  decoder.retina_masks = true;
  decoder.binary_mask_gt_zero = true;
  decoder.max_mask_bytes = 64ull << 20;
  vie::DecoderInput decoded;
  decoded.geometry = prepared.geometry;
  decoded.outputs.emplace("output0", output.tensors.at("output0").tensor);
  decoded.outputs.emplace("output1", output.tensors.at("output1").tensor);
  const vie::SegmentationBatch direct =
      vie::YoloV8SegmentationDecoder{}.decode_segmentation(decoded, decoder, vie::DeviceContext{});
  expect_same_batch(owned, direct);

  vie::InferenceEngine budget_engine;
  budget_engine.load(config);
  vie::SegmentationRuntimeConfig budget = runtime_ready();
  budget.max_mask_bytes = 1;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, budget_engine.infer_segmentation(bus.image, budget));
}

TEST(EngineSegmentationBenchmark, CpuRealImage) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  vie::ModelConfig config;
  config.manifest = recognized_segmentation_manifest(path);
  config.model_path = path;
  vie::InferenceEngine engine;
  const auto load_start = std::chrono::steady_clock::now();
  engine.load(config);
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
  const LoadedImage bus = load_bus();
  vie::SegmentationInferTiming first;
  const vie::SegmentationBatch first_batch = engine.infer_segmentation(bus.image, runtime_ready(), &first);
  ASSERT_FALSE(first_batch.instances.empty());
  constexpr int kRuns = 100;
  std::vector<double> preprocess(kRuns);
  std::vector<double> execute(kRuns);
  std::vector<double> decode(kRuns);
  std::vector<double> total(kRuns);
  for (int i = 0; i < kRuns; ++i) {
    vie::SegmentationInferTiming timing;
    const vie::SegmentationBatch batch = engine.infer_segmentation(bus.image, runtime_ready(), &timing);
    ASSERT_EQ(batch.instances.size(), first_batch.instances.size());
    preprocess[static_cast<std::size_t>(i)] = timing.preprocess_ms;
    execute[static_cast<std::size_t>(i)] = timing.execute_ms;
    decode[static_cast<std::size_t>(i)] = timing.decode_ms;
    total[static_cast<std::size_t>(i)] = timing.total_ms;
  }
  const StageStats prep = summarize(preprocess);
  const StageStats ran = summarize(execute);
  const StageStats decoded = summarize(decode);
  const StageStats all = summarize(total);
  std::ofstream out("/tmp/vie_phase433_engine_cpu.txt");
  out << "model=yolov8n-seg.onnx image=" << bus.image.width << "x" << bus.image.height
      << " ep=cpu instances=" << first_batch.instances.size() << " mask=" << first_batch.instances[0].mask.width()
      << "x" << first_batch.instances[0].mask.height()
      << " includes_h2d=0 includes_mask_d2h=0 includes_sync=0 runs=100\n";
  out << "load_ms=" << load_ms << "\n";
  out << "first preprocess=" << first.preprocess_ms << " execute=" << first.execute_ms
      << " decode=" << first.decode_ms << " total=" << first.total_ms << "\n";
  out << "steady preprocess mean/p50/p95 " << prep.mean << " " << prep.p50 << " " << prep.p95 << "\n";
  out << "steady execute mean/p50/p95 " << ran.mean << " " << ran.p50 << " " << ran.p95 << "\n";
  out << "steady decode mean/p50/p95 " << decoded.mean << " " << decoded.p50 << " " << decoded.p95 << "\n";
  out << "steady e2e mean/p50/p95 " << all.mean << " " << all.p50 << " " << all.p95 << "\n";
  EXPECT_GT(all.mean, decoded.mean);
}
#endif

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
TEST(EngineSegmentation, CudaEndToEndKeepsMasksOnDevice) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  const LoadedImage bus = load_bus();
  vie::ModelConfig cpu_config;
  cpu_config.manifest = recognized_segmentation_manifest(path);
  cpu_config.model_path = path;
  cpu_config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine cpu_engine;
  cpu_engine.load(cpu_config);
  const vie::SegmentationBatch cpu_batch = cpu_engine.infer_segmentation(bus.image, runtime_ready());

  std::uint8_t* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_pixels), bus.pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, bus.pixels.data(), bus.pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput device_image = bus.image;
  device_image.data = device_pixels;
  device_image.device = vie::DeviceType::Cuda;
  device_image.device_id = 0;

  vie::ModelConfig cuda_config = cpu_config;
  cuda_config.device = vie::DeviceType::Cuda;
  cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
  vie::InferenceEngine cuda_engine;
  cuda_engine.load(cuda_config);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, cuda_engine.infer_segmentation(bus.image, runtime_ready()));
  vie::SegmentationInferTiming timing;
  const vie::SegmentationBatch cuda_batch = cuda_engine.infer_segmentation(device_image, runtime_ready(), &timing);
  cudaFree(device_pixels);
  ASSERT_FALSE(cuda_batch.instances.empty());
  EXPECT_EQ(cuda_batch.execution_device, vie::DeviceType::Cuda);
  EXPECT_TRUE(timing.includes_sync);
  EXPECT_FALSE(timing.includes_h2d);
  EXPECT_FALSE(timing.includes_mask_d2h);
  for (const vie::InstanceSegmentation& instance : cuda_batch.instances) {
    EXPECT_EQ(instance.mask.device(), vie::DeviceType::Cuda);
    EXPECT_EQ(instance.mask.device_id(), 0);
    const vie::Mask host = instance.mask.to_cpu();
    const std::uint8_t* pixels = host.binary_data();
    for (int i = 0; i < host.width() * host.height(); ++i) ASSERT_TRUE(pixels[i] == 0 || pixels[i] == 1);
  }

  const vie::PreprocessResult prepared = cpu_engine.preprocess(bus.image);
  vie::BackendRequest request;
  request.inputs.emplace("images", prepared.output);
  const vie::BackendOutput cpu_outputs = cpu_engine.execute(request);
  ASSERT_TRUE(cpu_outputs.tensors.at("output0").tensor.is_cpu());
  ASSERT_TRUE(cpu_outputs.tensors.at("output1").tensor.is_cpu());
  vie::DecoderConfig decoder;
  decoder.output_name = "output0";
  decoder.prototype_name = "output1";
  decoder.channel_axis = 1;
  decoder.box_channels = 4;
  decoder.num_classes = 80;
  decoder.mask_channels = 32;
  decoder.box_decode = vie::BoxDecode::DecodedCxcywh;
  decoder.scores_are_probabilities = true;
  decoder.class_selection = vie::ClassSelection::BestClass;
  decoder.confidence_threshold = 0.25f;
  decoder.nms_enabled = true;
  decoder.class_aware_nms = true;
  decoder.iou_threshold = 0.45f;
  decoder.max_detections = 20;
  decoder.pre_nms_top_k = 300;
  decoder.prototype_channel_axis = 1;
  decoder.retina_masks = true;
  decoder.binary_mask_gt_zero = true;
  decoder.max_mask_bytes = 64ull << 20;
  vie::DecoderInput cpu_input;
  cpu_input.geometry = prepared.geometry;
  cpu_input.outputs.emplace("output0", cpu_outputs.tensors.at("output0").tensor);
  cpu_input.outputs.emplace("output1", cpu_outputs.tensors.at("output1").tensor);
  const vie::SegmentationBatch cpu_decoder =
      vie::YoloV8SegmentationDecoder{}.decode_segmentation(cpu_input, decoder, vie::DeviceContext{});
  vie::DecoderInput gpu_input;
  gpu_input.geometry = prepared.geometry;
  const vie::DeviceContext gpu{vie::DeviceType::Cuda, 0};
  gpu_input.outputs.emplace("output0", cpu_outputs.tensors.at("output0").tensor.to(gpu));
  gpu_input.outputs.emplace("output1", cpu_outputs.tensors.at("output1").tensor.to(gpu));
  EXPECT_EQ(gpu_input.outputs.at("output0").device(), vie::DeviceType::Cuda);
  EXPECT_EQ(gpu_input.outputs.at("output1").device(), vie::DeviceType::Cuda);
  const vie::SegmentationBatch gpu_decoder =
      vie::YoloV8SegmentationDecoder{}.decode_segmentation(gpu_input, decoder, gpu);
  expect_same_batch(gpu_decoder, cpu_decoder);

  int class_mismatches = 0;
  int box_mismatches = 0;
  int pixel_mismatches = 0;
  const std::size_t shared = std::min(cpu_batch.instances.size(), cuda_batch.instances.size());
  for (std::size_t i = 0; i < shared; ++i) {
    if (cpu_batch.instances[i].detection.class_id != cuda_batch.instances[i].detection.class_id) ++class_mismatches;
    const vie::BoundingBox& left = cpu_batch.instances[i].detection.box;
    const vie::BoundingBox& right = cuda_batch.instances[i].detection.box;
    if (left.x1 != right.x1 || left.y1 != right.y1 || left.x2 != right.x2 || left.y2 != right.y2) ++box_mismatches;
    const vie::Mask cuda_host = cuda_batch.instances[i].mask.to_cpu();
    if (cuda_host.width() == cpu_batch.instances[i].mask.width() &&
        cuda_host.height() == cpu_batch.instances[i].mask.height()) {
      const std::uint8_t* a = cpu_batch.instances[i].mask.binary_data();
      const std::uint8_t* b = cuda_host.binary_data();
      for (int pixel = 0; pixel < cuda_host.width() * cuda_host.height(); ++pixel) {
        if (a[pixel] != b[pixel]) ++pixel_mismatches;
      }
    }
  }
  std::ofstream out("/tmp/vie_phase433_engine_cuda.txt");
  out << "cpu_ep_instances=" << cpu_batch.instances.size() << " cuda_ep_instances=" << cuda_batch.instances.size()
      << " class_mismatches=" << class_mismatches << " box_mismatches=" << box_mismatches
      << " pixel_mismatches=" << pixel_mismatches << "\n";
  out << "same_tensor_decoder_match=1 cuda_masks_device=cuda fallback=0\n";
  out << "engine_cuda preprocess=" << timing.preprocess_ms << " execute=" << timing.execute_ms
      << " decode=" << timing.decode_ms << " total=" << timing.total_ms
      << " includes_sync=1 includes_h2d=0 includes_mask_d2h=0\n";
}

TEST(EngineSegmentationBenchmark, CudaRealImage) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  const LoadedImage bus = load_bus();
  std::uint8_t* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_pixels), bus.pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, bus.pixels.data(), bus.pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput device_image = bus.image;
  device_image.data = device_pixels;
  device_image.device = vie::DeviceType::Cuda;
  device_image.device_id = 0;
  vie::ModelConfig config;
  config.manifest = recognized_segmentation_manifest(path);
  config.model_path = path;
  config.device = vie::DeviceType::Cuda;
  config.backend = vie::BackendKind::OnnxRuntimeCuda;
  vie::InferenceEngine engine;
  const auto load_start = std::chrono::steady_clock::now();
  engine.load(config);
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
  vie::SegmentationInferTiming first;
  const vie::SegmentationBatch first_batch = engine.infer_segmentation(device_image, runtime_ready(), &first);
  ASSERT_FALSE(first_batch.instances.empty());
  constexpr int kRuns = 100;
  std::vector<double> preprocess(kRuns);
  std::vector<double> execute(kRuns);
  std::vector<double> decode(kRuns);
  std::vector<double> total(kRuns);
  for (int i = 0; i < kRuns; ++i) {
    vie::SegmentationInferTiming timing;
    const vie::SegmentationBatch batch = engine.infer_segmentation(device_image, runtime_ready(), &timing);
    ASSERT_EQ(batch.instances.size(), first_batch.instances.size());
    EXPECT_EQ(batch.instances[0].mask.device(), vie::DeviceType::Cuda);
    preprocess[static_cast<std::size_t>(i)] = timing.preprocess_ms;
    execute[static_cast<std::size_t>(i)] = timing.execute_ms;
    decode[static_cast<std::size_t>(i)] = timing.decode_ms;
    total[static_cast<std::size_t>(i)] = timing.total_ms;
  }
  cudaFree(device_pixels);
  const StageStats prep = summarize(preprocess);
  const StageStats ran = summarize(execute);
  const StageStats decoded = summarize(decode);
  const StageStats all = summarize(total);
  std::ofstream out("/tmp/vie_phase433_engine_cuda_bench.txt");
  out << "model=yolov8n-seg.onnx image=" << bus.image.width << "x" << bus.image.height
      << " ep=cuda_strict instances=" << first_batch.instances.size()
      << " mask=" << first_batch.instances[0].mask.width() << "x" << first_batch.instances[0].mask.height()
      << " includes_h2d=0 includes_mask_d2h=0 includes_sync=1 runs=100\n";
  out << "load_ms=" << load_ms << "\n";
  out << "first preprocess=" << first.preprocess_ms << " execute=" << first.execute_ms
      << " decode=" << first.decode_ms << " total=" << first.total_ms << "\n";
  out << "steady preprocess mean/p50/p95 " << prep.mean << " " << prep.p50 << " " << prep.p95 << "\n";
  out << "steady execute mean/p50/p95 " << ran.mean << " " << ran.p50 << " " << ran.p95 << "\n";
  out << "steady decode mean/p50/p95 " << decoded.mean << " " << decoded.p50 << " " << decoded.p95 << "\n";
  out << "steady e2e mean/p50/p95 " << all.mean << " " << all.p50 << " " << all.p95 << "\n";
}
#endif
