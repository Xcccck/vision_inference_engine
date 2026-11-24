// Phase 9.1: Batch Tensor, Batch Preprocess, and Batch inference tests.
// The static test model testdata/onnx/yolov8n.onnx is fixed Batch=1, so
// end-to-end Batch>1 coverage uses testdata/onnx_dynamic/yolov8n_batch.onnx
// exported with dynamic batch. When that file is missing the batch E2E tests
// GTEST_SKIP rather than faking success.

#include "test_util.hpp"

#include "vie/core/tensor.hpp"
#include "vie/decoder/detection.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/model_config.hpp"
#include "vie/preprocess/preprocess.hpp"
#include "vie/runtime/model_instance.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sys/stat.h>
#include <vector>

namespace {

using namespace vie;

#if defined(VIE_HAS_ONNXRUNTIME)
bool file_exists(const std::string& path) {
  struct stat buffer {};
  return ::stat(path.c_str(), &buffer) == 0;
}
#endif

// Builds an ImageInput from a synthetic pixel buffer: constant color per
// channel. Every image in a test batch gets its own fill so we can compare
// outputs across the single and batch paths without crosstalk.
struct OwnedImage {
  std::vector<std::uint8_t> pixels;
  ImageInput input;
};

OwnedImage make_image(int width, int height, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  OwnedImage image;
  image.pixels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u, 0u);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t base = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                static_cast<std::size_t>(x)) *
                               3u;
      image.pixels[base + 0] = r;
      image.pixels[base + 1] = g;
      image.pixels[base + 2] = b;
    }
  }
  image.input.data = image.pixels.data();
  image.input.width = width;
  image.input.height = height;
  image.input.channels = 3;
  image.input.color = ColorOrder::Rgb;
  image.input.layout = ImageLayout::Hwc;
  image.input.dtype = DataType::Uint8;
  image.input.device = DeviceType::Cpu;
  image.input.device_id = 0;
  return image;
}

PreprocessConfig default_config() {
  PreprocessConfig config;
  config.target_height = 64;
  config.target_width = 64;
  config.resize_mode = ResizeMode::Letterbox;
  config.interpolation = Interpolation::Bilinear;
  config.padding_value = 114.f;
  config.input_color = ColorOrder::Rgb;
  config.output_color = ColorOrder::Rgb;
  config.input_layout = ImageLayout::Hwc;
  config.output_layout = ImageLayout::Chw;
  config.output_dtype = DataType::Fp32;
  config.normalize_enabled = true;
  config.scale = 1.f / 255.f;
  config.mean = {0.f, 0.f, 0.f};
  config.std = {1.f, 1.f, 1.f};
  return config;
}

// ------------------------------------------------------------------
// Tensor Batch shape handling
// ------------------------------------------------------------------

TEST(BatchTensor, CpuAllocatesNcHW) {
  DeviceContext cpu(DeviceType::Cpu, 0);
  Tensor t = Tensor::zeros(cpu, DataType::Fp32, {4, 3, 8, 8});
  EXPECT_EQ(t.rank(), 4u);
  EXPECT_EQ(t.shape()[0], 4);
  EXPECT_EQ(t.numel(), 4u * 3u * 8u * 8u);
  EXPECT_TRUE(t.contiguous());
  EXPECT_TRUE(t.is_cpu());
}

TEST(BatchTensor, ClonePreservesBatch) {
  DeviceContext cpu(DeviceType::Cpu, 0);
  Tensor t = Tensor::zeros(cpu, DataType::Fp32, {2, 3, 4, 4});
  float* data = t.mutable_data<float>();
  for (std::size_t i = 0; i < t.numel(); ++i) data[i] = static_cast<float>(i);
  Tensor clone = t.clone();
  ASSERT_EQ(clone.shape(), t.shape());
  EXPECT_NE(clone.data(), t.data());
  EXPECT_EQ(std::memcmp(clone.data(), t.data(), t.nbytes()), 0);
}

TEST(BatchTensor, RejectsNonPositiveBatch) {
  DeviceContext cpu(DeviceType::Cpu, 0);
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, Tensor::zeros(cpu, DataType::Fp32, {0, 3, 4, 4}));
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, Tensor::zeros(cpu, DataType::Fp32, {-1, 3, 4, 4}));
}

TEST(BatchTensor, ViewSampleSliceSharesStorage) {
  DeviceContext cpu(DeviceType::Cpu, 0);
  Tensor batch = Tensor::zeros(cpu, DataType::Fp32, {2, 3, 4, 4});
  // Sample 1 starts at element 24 (2 * 3 * 4).
  Tensor slice = batch.view({1, 3, 4, 4}, {48, 16, 4, 1}, 24u * sizeof(float));
  EXPECT_TRUE(slice.shares_storage(batch));
  EXPECT_EQ(slice.data(), batch.data() + 24u * sizeof(float));
}

// ------------------------------------------------------------------
// Preprocess Batch
// ------------------------------------------------------------------

TEST(PreprocessBatch, CpuEmptyBatchThrows) {
  ImagePreprocessor pre;
  pre.configure(default_config());
  std::vector<ImageInput> empty;
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument,
                   pre.execute_batch(empty, DeviceContext(DeviceType::Cpu, 0)));
}

TEST(PreprocessBatch, CpuMatchesSingleImageResults) {
  ImagePreprocessor pre;
  pre.configure(default_config());
  DeviceContext cpu(DeviceType::Cpu, 0);
  auto a = make_image(48, 32, 10, 20, 30);
  auto b = make_image(64, 48, 200, 100, 50);
  auto c = make_image(32, 32, 7, 7, 7);
  const std::vector<ImageInput> images = {a.input, b.input, c.input};

  const PreprocessResult single_a = pre.execute(a.input, cpu);
  const PreprocessResult single_b = pre.execute(b.input, cpu);
  const PreprocessResult single_c = pre.execute(c.input, cpu);
  const BatchPreprocessResult batch = pre.execute_batch(images, cpu);

  ASSERT_EQ(batch.output.shape().size(), 4u);
  EXPECT_EQ(batch.output.shape()[0], 3);
  EXPECT_EQ(batch.output.shape()[1], 3);
  EXPECT_EQ(batch.output.shape()[2], 64);
  EXPECT_EQ(batch.output.shape()[3], 64);
  EXPECT_EQ(batch.geometries.size(), 3u);

  const std::size_t per_image = 3u * 64u * 64u * sizeof(float);
  const float* base = batch.output.data_as<float>();
  EXPECT_EQ(std::memcmp(base, single_a.output.data(), per_image), 0);
  EXPECT_EQ(std::memcmp(base + (per_image / sizeof(float)), single_b.output.data(), per_image), 0);
  EXPECT_EQ(std::memcmp(base + 2u * (per_image / sizeof(float)), single_c.output.data(), per_image), 0);
}

TEST(PreprocessBatch, CpuPreservesPerImageGeometry) {
  ImagePreprocessor pre;
  pre.configure(default_config());
  DeviceContext cpu(DeviceType::Cpu, 0);
  auto a = make_image(100, 50, 1, 2, 3);   // landscape, resized_w = 64, resized_h = 32
  auto b = make_image(50, 100, 4, 5, 6);   // portrait, resized_w = 32, resized_h = 64
  const BatchPreprocessResult batch = pre.execute_batch({a.input, b.input}, cpu);
  EXPECT_EQ(batch.geometries[0].original_width, 100);
  EXPECT_EQ(batch.geometries[0].original_height, 50);
  EXPECT_EQ(batch.geometries[1].original_width, 50);
  EXPECT_EQ(batch.geometries[1].original_height, 100);
  // Landscape vs portrait swap the resized extent along each axis; sharing
  // one GeometryTransform across the batch would make that impossible.
  EXPECT_EQ(batch.geometries[0].resized_width, batch.geometries[1].resized_height);
  EXPECT_EQ(batch.geometries[0].resized_height, batch.geometries[1].resized_width);
  EXPECT_NE(batch.geometries[0].resized_width, batch.geometries[0].resized_height);
}

TEST(PreprocessBatch, CpuRejectsDeviceMismatchPerIndex) {
  ImagePreprocessor pre;
  pre.configure(default_config());
  DeviceContext cpu(DeviceType::Cpu, 0);
  auto a = make_image(32, 32, 1, 2, 3);
  ImageInput b = a.input;
  b.device = DeviceType::Cuda;  // invalid for CPU execution
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, pre.execute_batch({a.input, b}, cpu));
}

// ------------------------------------------------------------------
// YOLOv8 Detection Batch (E2E, requires dynamic batch model)
// ------------------------------------------------------------------

#if defined(VIE_HAS_ONNXRUNTIME)

class Yolov8BatchE2E : public ::testing::Test {
 protected:
  void SetUp() override {
    model_path_ = vie_fixture("testdata/onnx_dynamic/yolov8n_batch.onnx");
    if (!file_exists(model_path_)) {
      GTEST_SKIP() << "dynamic batch yolov8n model not available: " << model_path_;
    }
  }

  ModelConfig config_for(std::int64_t batch_dim) {
    // Manually construct a manifest for the dynamic batch model; the
    // recognizer cannot resolve the shape when batch is dynamic in this
    // phase, so we hand it the fields the pipeline needs.
    ModelManifest manifest;
    manifest.model_name = "yolov8n_batch";
    manifest.model_family = "yolov8";
    manifest.num_classes = 80;
    manifest.model_path = model_path_;
    manifest.task = TaskType::Detection;
    manifest.model_format = ModelFormat::Onnx;
    manifest.input.name = "images";
    manifest.input.shape = {batch_dim, 3, 640, 640};
    manifest.input.dtype = DataType::Fp32;
    manifest.input.layout = Layout::Nchw;
    manifest.input.color = "rgb";
    manifest.input.normalization = "scale_0_1";
    manifest.input.resize = "letterbox";
    manifest.input.interpolation = "bilinear";
    manifest.input.padding_value = 114.f;
    manifest.provenance["model_name"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["model_family"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["num_classes"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["task"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["model_format"] = FieldProvenance{Certainty::User, {}};
    manifest.provenance["model_path"] = FieldProvenance{Certainty::User, {}};
    for (const char* field : {"input.name", "input.shape", "input.dtype", "input.layout",
                               "input.color", "input.normalization", "input.resize",
                               "input.interpolation", "input.padding_value"}) {
      manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
    }
    OutputDesc out;
    out.name = "output0";
    out.shape = {batch_dim, 84, 8400};
    out.dtype = DataType::Fp32;
    out.layout = Layout::Unknown;
    out.semantic = "predictions";
    manifest.outputs.push_back(out);
    manifest.decoder.protocol = "yolov8_detect_raw";
    manifest.decoder.box_format = "cxcywh";
    manifest.decoder.box_decode = "decoded_cxcywh";
    manifest.decoder.box_channels = 4;
    manifest.decoder.channel_axis = 1;
    manifest.decoder.class_selection = "best_class";
    manifest.decoder.scores_are_probabilities = true;
    for (const char* field : {"decoder.protocol", "decoder.box_format", "decoder.box_decode",
                               "decoder.box_channels", "decoder.channel_axis",
                               "decoder.class_selection", "decoder.scores_are_probabilities"}) {
      manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
    }
    manifest.nms.enabled = true;
    manifest.nms.confidence_threshold = 0.25f;
    manifest.nms.iou_threshold = 0.45f;
    manifest.nms.max_detections = 20;
    manifest.nms.pre_nms_top_k = 300;
    manifest.nms.class_aware = true;
    for (const char* field : {"nms.enabled", "nms.confidence_threshold", "nms.iou_threshold",
                               "nms.max_detections", "nms.pre_nms_top_k", "nms.class_aware"}) {
      manifest.provenance[field] = FieldProvenance{Certainty::User, {}};
    }

    ModelConfig config;
    config.model_path = model_path_;
    config.task = TaskType::Detection;
    config.backend = BackendKind::OnnxRuntime;
    config.device = DeviceType::Cpu;
    config.manifest = manifest;
    return config;
  }

  std::string model_path_;
};

TEST_F(Yolov8BatchE2E, BatchOneMatchesSingle) {
  InferenceEngine engine;
  engine.load(config_for(1));
  auto image = make_image(640, 640, 128, 64, 32);
  DetectionBatch single = engine.infer(image.input);
  DetectionBatch batch = engine.infer_batch({image.input});
  ASSERT_EQ(single.results.size(), 1u);
  ASSERT_EQ(batch.results.size(), 1u);
  ASSERT_EQ(batch.results[0].detections.size(), single.results[0].detections.size());
  for (std::size_t i = 0; i < single.results[0].detections.size(); ++i) {
    const Detection& a = single.results[0].detections[i];
    const Detection& b = batch.results[0].detections[i];
    EXPECT_FLOAT_EQ(a.box.x1, b.box.x1);
    EXPECT_FLOAT_EQ(a.confidence, b.confidence);
    EXPECT_EQ(a.class_id, b.class_id);
  }
}

TEST_F(Yolov8BatchE2E, BatchFourProducesFourResults) {
  InferenceEngine engine;
  engine.load(config_for(-1));  // dynamic batch manifest
  auto a = make_image(320, 240, 10, 20, 30);
  auto b = make_image(640, 480, 100, 150, 200);
  auto c = make_image(800, 600, 30, 60, 90);
  auto d = make_image(500, 500, 200, 100, 50);
  DetectionBatch batch = engine.infer_batch({a.input, b.input, c.input, d.input});
  ASSERT_EQ(batch.results.size(), 4u);
  EXPECT_EQ(batch.results[0].original_width, 320);
  EXPECT_EQ(batch.results[1].original_width, 640);
  EXPECT_EQ(batch.results[2].original_width, 800);
  EXPECT_EQ(batch.results[3].original_width, 500);
}

TEST_F(Yolov8BatchE2E, BatchMatchesPerImageInference) {
  // The Batch path must produce exactly the same Detection list as running
  // infer() on each image, i.e. NMS and geometry stay isolated per sample.
  InferenceEngine batch_engine;
  batch_engine.load(config_for(-1));
  InferenceEngine single_engine;
  single_engine.load(config_for(1));
  auto a = make_image(400, 300, 20, 40, 60);
  auto b = make_image(640, 480, 180, 90, 40);
  DetectionBatch batch = batch_engine.infer_batch({a.input, b.input});
  DetectionBatch single_a = single_engine.infer(a.input);
  DetectionBatch single_b = single_engine.infer(b.input);
  ASSERT_EQ(batch.results.size(), 2u);
  ASSERT_EQ(batch.results[0].detections.size(), single_a.results[0].detections.size());
  ASSERT_EQ(batch.results[1].detections.size(), single_b.results[0].detections.size());
  for (std::size_t i = 0; i < batch.results[0].detections.size(); ++i) {
    EXPECT_FLOAT_EQ(batch.results[0].detections[i].box.x1, single_a.results[0].detections[i].box.x1);
    EXPECT_FLOAT_EQ(batch.results[0].detections[i].confidence,
                    single_a.results[0].detections[i].confidence);
  }
  for (std::size_t i = 0; i < batch.results[1].detections.size(); ++i) {
    EXPECT_FLOAT_EQ(batch.results[1].detections[i].box.x1, single_b.results[0].detections[i].box.x1);
    EXPECT_FLOAT_EQ(batch.results[1].detections[i].confidence,
                    single_b.results[0].detections[i].confidence);
  }
}

TEST_F(Yolov8BatchE2E, EmptyBatchRejected) {
  InferenceEngine engine;
  engine.load(config_for(-1));
  std::vector<ImageInput> empty;
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, engine.infer_batch(empty));
}

TEST_F(Yolov8BatchE2E, MixedDevicesRejected) {
  InferenceEngine engine;
  engine.load(config_for(-1));
  auto a = make_image(64, 64, 1, 2, 3);
  ImageInput b = a.input;
  b.device = DeviceType::Cuda;
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, engine.infer_batch({a.input, b}));
}

#endif  // VIE_HAS_ONNXRUNTIME

}  // namespace
