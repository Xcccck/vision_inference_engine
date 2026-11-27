// Phase 11: optional OpenCV cv::Mat -> ImageInput adapter.
//
// These tests only compile in builds with -DVIE_ENABLE_OPENCV=ON (the source is
// added conditionally in CMakeLists.txt). Core ImageInput tests stay in
// tests/test_preprocess.cpp and are unaffected.

#include "test_util.hpp"

#include "vie/core/error.hpp"
#include "vie/opencv/image_input.hpp"
#include "vie/preprocess/preprocess.hpp"
#include "vie/runtime/inference_request.hpp"

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/decoder/detection.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest.hpp"
#include "vie/model/model_config.hpp"
#include "vie/runtime/runtime.hpp"
#endif

#include <opencv2/core.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace vie {
namespace {

// ------------------------------------------------------------------
// Type mapping: only the four documented OpenCV types are accepted.
// ------------------------------------------------------------------

TEST(OpenCvInput, MapsGray8UC1) {
  cv::Mat mat(48, 64, CV_8UC1, cv::Scalar(7));
  const ImageInput image = from_cv_mat(mat);
  EXPECT_EQ(image.data, static_cast<const void*>(mat.data));
  EXPECT_EQ(image.width, 64);
  EXPECT_EQ(image.height, 48);
  EXPECT_EQ(image.channels, 1);
  EXPECT_EQ(image.color, ColorOrder::Gray);
  EXPECT_EQ(image.layout, ImageLayout::Hwc);
  EXPECT_EQ(image.dtype, DataType::Uint8);
  EXPECT_EQ(image.device, DeviceType::Cpu);
  EXPECT_EQ(image.device_id, 0);
  EXPECT_EQ(image.ownership, ImageOwnership::Borrowed);
  EXPECT_EQ(image.row_stride_bytes, mat.step[0]);
}

TEST(OpenCvInput, MapsBgr8UC3ByDefault) {
  cv::Mat mat(48, 64, CV_8UC3, cv::Scalar(10, 20, 30));
  const ImageInput image = from_cv_mat(mat);
  EXPECT_EQ(image.data, static_cast<const void*>(mat.data));
  EXPECT_EQ(image.channels, 3);
  EXPECT_EQ(image.color, ColorOrder::Bgr);
  EXPECT_EQ(image.dtype, DataType::Uint8);
  EXPECT_EQ(image.row_stride_bytes, static_cast<std::size_t>(64) * 3);
}

TEST(OpenCvInput, HonorsExplicitRgbColorOrder) {
  cv::Mat mat(8, 8, CV_8UC3);
  const ImageInput image = from_cv_mat(mat, ColorOrder::Rgb);
  EXPECT_EQ(image.channels, 3);
  EXPECT_EQ(image.color, ColorOrder::Rgb);
}

TEST(OpenCvInput, MapsGray32FC1) {
  cv::Mat mat(16, 32, CV_32FC1, cv::Scalar(0.5f));
  const ImageInput image = from_cv_mat(mat);
  EXPECT_EQ(image.channels, 1);
  EXPECT_EQ(image.color, ColorOrder::Gray);
  EXPECT_EQ(image.dtype, DataType::Fp32);
  EXPECT_EQ(image.row_stride_bytes, static_cast<std::size_t>(32) * 4);
}

TEST(OpenCvInput, MapsBgr32FC3) {
  cv::Mat mat(16, 32, CV_32FC3, cv::Scalar(0.1f, 0.2f, 0.3f));
  const ImageInput image = from_cv_mat(mat);
  EXPECT_EQ(image.channels, 3);
  EXPECT_EQ(image.color, ColorOrder::Bgr);
  EXPECT_EQ(image.dtype, DataType::Fp32);
  EXPECT_EQ(image.row_stride_bytes, static_cast<std::size_t>(32) * 3 * 4);
}

TEST(OpenCvInput, RejectsBgra8UC4) {
  // ImageInput has no alpha channel. Dropping it here would silently discard
  // caller data, so the adapter refuses instead.
  cv::Mat mat(8, 8, CV_8UC4);
  VIE_EXPECT_ERROR(ErrorCode::Unsupported, from_cv_mat(mat));
}

TEST(OpenCvInput, RejectsTwoChannelMat) {
  cv::Mat mat(8, 8, CV_8UC2);
  VIE_EXPECT_ERROR(ErrorCode::Unsupported, from_cv_mat(mat));
}

TEST(OpenCvInput, Rejects16BitDepth) {
  cv::Mat mat(8, 8, CV_16UC1);
  VIE_EXPECT_ERROR(ErrorCode::Unsupported, from_cv_mat(mat));
}

TEST(OpenCvInput, Rejects64BitDepth) {
  cv::Mat mat(8, 8, CV_64FC1);
  VIE_EXPECT_ERROR(ErrorCode::Unsupported, from_cv_mat(mat));
}

TEST(OpenCvInput, RejectsEmptyMat) {
  cv::Mat mat;
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, from_cv_mat(mat));
}

TEST(OpenCvInput, RejectsGrayOrderForThreeChannelMat) {
  cv::Mat mat(8, 8, CV_8UC3);
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, from_cv_mat(mat, ColorOrder::Gray));
}

// ------------------------------------------------------------------
// Geometry: ROI rows are not tightly packed; stride must come from step[0].
// ------------------------------------------------------------------

TEST(OpenCvInput, RoiUsesParentStepNotWidthTimesChannels) {
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(1, 2, 3));
  const cv::Mat roi = frame(cv::Rect(20, 20, 300, 200));
  ASSERT_FALSE(roi.isContinuous());

  const ImageInput image = from_cv_mat(roi);
  EXPECT_EQ(image.data, static_cast<const void*>(roi.data));
  EXPECT_EQ(image.width, 300);
  EXPECT_EQ(image.height, 200);
  EXPECT_EQ(image.row_stride_bytes, frame.step[0]);
  EXPECT_GT(image.row_stride_bytes, static_cast<std::size_t>(300) * 3);
}

TEST(OpenCvInput, PreprocessReadsRoiRowsThroughStride) {
  // A ROI and a tightly packed clone hold the same pixels in the same order.
  // Preprocess must therefore produce byte-identical output for both; if the
  // adapter had used width*channels as stride, the ROI output would be skewed.
  cv::Mat frame(480, 640, CV_8UC3);
  cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
  const cv::Mat roi = frame(cv::Rect(20, 20, 300, 200));
  const cv::Mat packed = roi.clone();
  ASSERT_FALSE(roi.isContinuous());
  ASSERT_TRUE(packed.isContinuous());

  PreprocessConfig config;
  config.resize_mode = ResizeMode::None;
  config.target_width = 300;
  config.target_height = 200;
  config.input_color = ColorOrder::Bgr;
  config.output_color = ColorOrder::Bgr;
  config.output_layout = ImageLayout::Hwc;
  config.output_dtype = DataType::Uint8;
  ImagePreprocessor preprocessor;
  preprocessor.configure(config);

  const PreprocessResult from_roi = preprocessor.execute(from_cv_mat(roi), DeviceContext(DeviceType::Cpu, 0));
  const PreprocessResult from_packed =
      preprocessor.execute(from_cv_mat(packed), DeviceContext(DeviceType::Cpu, 0));

  const std::size_t bytes = static_cast<std::size_t>(300) * 200 * 3;
  ASSERT_EQ(from_roi.output.numel(), from_packed.output.numel());
  EXPECT_EQ(std::memcmp(from_roi.output.data(), from_packed.output.data(), bytes), 0);
}

// ------------------------------------------------------------------
// Memory lifetime: borrowed view, shallow copy, caller-owned storage.
// ------------------------------------------------------------------

TEST(OpenCvInput, BorrowsMatStorageWithoutCopying) {
  cv::Mat mat(4, 4, CV_8UC1, cv::Scalar(0));
  ImageInput image = from_cv_mat(mat);
  const std::uint8_t before = static_cast<const std::uint8_t*>(image.data)[5];
  mat.at<std::uint8_t>(1, 1) = 200;
  const std::uint8_t after = static_cast<const std::uint8_t*>(image.data)[5];
  EXPECT_NE(before, after);
  EXPECT_EQ(after, 200);
}

TEST(OpenCvInput, ShallowCopySharesPixelBuffer) {
  cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat shallow = mat;  // refcount bump, same allocation
  ASSERT_EQ(shallow.data, mat.data);

  ImageInput image = from_cv_mat(shallow);
  shallow.setTo(cv::Scalar(9, 8, 7));
  EXPECT_EQ(static_cast<const std::uint8_t*>(image.data)[0], 9);
  EXPECT_EQ(static_cast<const std::uint8_t*>(image.data)[1], 8);
  EXPECT_EQ(static_cast<const std::uint8_t*>(image.data)[2], 7);
}

TEST(OpenCvInput, RoiLifetimeIsParentLifetime) {
  // The ROI Mat is a header over the parent allocation; the borrowed pointer
  // stays valid exactly as long as the parent (not the header) is alive.
  cv::Mat parent(64, 64, CV_8UC1, cv::Scalar(5));
  ImageInput image;
  {
    const cv::Mat roi = parent(cv::Rect(8, 8, 16, 16));
    image = from_cv_mat(roi);
    EXPECT_EQ(image.data, static_cast<const void*>(roi.data));
  }  // ROI header destroyed; parent allocation still alive.
  EXPECT_EQ(static_cast<const std::uint8_t*>(image.data)[0], 5);
}

// ------------------------------------------------------------------
// Real-model E2E: cv::Mat path must match the direct ImageInput path.
// ------------------------------------------------------------------

#if defined(VIE_HAS_ONNXRUNTIME)
namespace {

ImageInput load_bus_pixels(std::vector<std::uint8_t>& pixels, int& width, int& height) {
  std::ifstream image_file(vie_fixture("testdata/images/bus.ppm"), std::ios::binary);
  std::string magic;
  int max_value = 0;
  image_file >> magic >> width >> height >> max_value;
  image_file.get();
  if (magic != "P6" || width <= 0 || height <= 0) {
    throw std::runtime_error("bus.ppm is not a valid P6 image");
  }
  pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
  image_file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = ColorOrder::Rgb;
  image.layout = ImageLayout::Hwc;
  image.dtype = DataType::Uint8;
  image.row_stride_bytes = static_cast<std::size_t>(width) * 3;
  return image;
}

ModelConfig detection_config() {
  ModelConfig config;
  config.model_path = vie_fixture("testdata/onnx/yolov8n.onnx");
  config.manifest = ModelManifest::load_file(
      vie_fixture("tools/diagnostics/model_deployment/yolov8n.manifest.json"));
  config.device = DeviceType::Cpu;
  return config;
}

void expect_same_detections(const DetectionResult& a, const DetectionResult& b) {
  ASSERT_EQ(a.detections.size(), b.detections.size());
  for (std::size_t i = 0; i < a.detections.size(); ++i) {
    EXPECT_EQ(a.detections[i].class_id, b.detections[i].class_id);
    EXPECT_NEAR(a.detections[i].confidence, b.detections[i].confidence, 1e-4f);
    EXPECT_NEAR(a.detections[i].box.x1, b.detections[i].box.x1, 1e-2f);
    EXPECT_NEAR(a.detections[i].box.y1, b.detections[i].box.y1, 1e-2f);
    EXPECT_NEAR(a.detections[i].box.x2, b.detections[i].box.x2, 1e-2f);
    EXPECT_NEAR(a.detections[i].box.y2, b.detections[i].box.y2, 1e-2f);
  }
}

// Allocates a frame padded by left/top/right/bottom and copies the source
// pixels into that rectangle, so the ROI content is byte-identical to pixels.
cv::Mat frame_with_pixels(const std::vector<std::uint8_t>& pixels, int width, int height,
                          int left, int top, int right, int bottom) {
  cv::Mat frame(height + top + bottom, width + left + right, CV_8UC3);
  frame.setTo(cv::Scalar(0, 0, 0));
  cv::Mat area = frame(cv::Rect(left, top, width, height));
  for (int y = 0; y < height; ++y) {
    std::memcpy(area.ptr(y), pixels.data() + static_cast<std::size_t>(y) * width * 3,
                static_cast<std::size_t>(width) * 3);
  }
  return frame;
}

}  // namespace

TEST(OpenCvInputRealModel, MatchesDirectImageInputPath) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");

  std::vector<std::uint8_t> pixels;
  int width = 0;
  int height = 0;
  const ImageInput direct = load_bus_pixels(pixels, width, height);

  // Same bytes in an OpenCV Mat. The picture was decoded as RGB, so Rgb is the
  // honest color order; nothing is swapped behind the caller's back.
  cv::Mat mat(height, width, CV_8UC3);
  std::memcpy(mat.data, pixels.data(), pixels.size());
  const ImageInput via_cv = from_cv_mat(mat, ColorOrder::Rgb);

  InferenceEngine engine;
  engine.load(detection_config());
  const DetectionResult direct_result = engine.infer(direct).results.at(0);
  const DetectionResult cv_result = engine.infer(via_cv).results.at(0);

  ASSERT_FALSE(direct_result.detections.empty());
  expect_same_detections(direct_result, cv_result);
}

TEST(OpenCvInputAsync, AdoptPacksRoiStrideAndBytes) {
  // The request copy must describe a packed image, not the ROI's parent stride.
  cv::Mat frame(40, 60, CV_8UC3);
  cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
  const cv::Mat roi = frame(cv::Rect(31, 9, 20, 13));
  ASSERT_FALSE(roi.isContinuous());
  const cv::Mat packed = roi.clone();

  InferenceRequest request;
  request.image = from_cv_mat(roi);
  request.adopt_cpu_pixels();

  ASSERT_EQ(request.owned_pixels.size(), 20u * 13u * 3u);
  EXPECT_EQ(request.image.row_stride_bytes, 20u * 3u);
  EXPECT_EQ(request.image.data, static_cast<const void*>(request.owned_pixels.data()));
  EXPECT_EQ(std::memcmp(request.owned_pixels.data(), packed.data, packed.total() * packed.elemSize()),
            0);
}

TEST(OpenCvInputAsync, RoiSubmitSurvivesCallerMutationAtEveryParentEdge) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");

  std::vector<std::uint8_t> pixels;
  int width = 0;
  int height = 0;
  const ImageInput direct = load_bus_pixels(pixels, width, height);

  InferenceEngine reference_engine;
  reference_engine.load(detection_config());
  const DetectionResult expected = reference_engine.infer(direct).results.at(0);
  ASSERT_FALSE(expected.detections.empty());

  RuntimeConfig config;
  config.enable_profiling = false;
  Runtime runtime(config);
  runtime.models().load("yolov8n", detection_config());

  struct Placement {
    const char* name;
    int left;
    int top;
    int right;
    int bottom;
  };
  const Placement placements[] = {
      {"continuous", 0, 0, 0, 0},
      {"center", 13, 9, 17, 11},
      {"right_edge", 21, 7, 0, 13},
      {"bottom_edge", 11, 17, 13, 0},
      {"bottom_right", 17, 13, 0, 0},
      {"top_left", 0, 0, 23, 19},
  };

  for (const Placement& placement : placements) {
    SCOPED_TRACE(placement.name);
    cv::Mat frame = frame_with_pixels(pixels, width, height, placement.left, placement.top,
                                      placement.right, placement.bottom);
    cv::Mat roi = frame(cv::Rect(placement.left, placement.top, width, height));

    std::future<DetectionResult> future =
        runtime.submit("yolov8n", from_cv_mat(roi, ColorOrder::Rgb));

    // submit() must have repacked the strided rows before returning. Destroy
    // every caller-side handle before the queued request runs.
    frame.setTo(cv::Scalar(0, 0, 0));
    roi.release();
    frame.release();

    const DetectionResult actual = future.get();
    expect_same_detections(expected, actual);
  }

  runtime.shutdown();
}

TEST(OpenCvInputAsync, NestedRoiSubmitSurvivesCallerMutation) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");

  std::vector<std::uint8_t> pixels;
  int width = 0;
  int height = 0;
  const ImageInput direct = load_bus_pixels(pixels, width, height);

  InferenceEngine reference_engine;
  reference_engine.load(detection_config());
  const DetectionResult expected = reference_engine.infer(direct).results.at(0);
  ASSERT_FALSE(expected.detections.empty());

  RuntimeConfig config;
  config.enable_profiling = false;
  Runtime runtime(config);
  runtime.models().load("yolov8n", detection_config());

  cv::Mat frame = frame_with_pixels(pixels, width, height, 20, 12, 6, 4);
  cv::Mat outer = frame(cv::Rect(5, 3, width + 15, height + 9));
  // Outer starts 15px left of / 9px above the pixels, so an inner ROI at
  // (15, 9) is exactly the pixels and ends on outer's bottom-right corner.
  cv::Mat inner = outer(cv::Rect(15, 9, width, height));
  ASSERT_EQ(inner.step[0], frame.step[0]);

  std::future<DetectionResult> future = runtime.submit("yolov8n", from_cv_mat(inner, ColorOrder::Rgb));
  frame.setTo(cv::Scalar(0, 0, 0));
  inner.release();
  outer.release();
  frame.release();

  const DetectionResult actual = future.get();
  expect_same_detections(expected, actual);
  runtime.shutdown();
}

TEST(OpenCvInputAsync, SubmitDeepCopiesBeforeReturning) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");

  std::vector<std::uint8_t> pixels;
  int width = 0;
  int height = 0;
  const ImageInput direct = load_bus_pixels(pixels, width, height);

  // Synchronous reference from pristine pixels (separate engine instance).
  InferenceEngine reference_engine;
  reference_engine.load(detection_config());
  const DetectionResult expected = reference_engine.infer(direct).results.at(0);
  ASSERT_FALSE(expected.detections.empty());

  cv::Mat mat(height, width, CV_8UC3);
  std::memcpy(mat.data, pixels.data(), pixels.size());

  RuntimeConfig config;
  config.enable_profiling = false;
  Runtime runtime(config);
  runtime.models().load("yolov8n", detection_config());

  std::future<DetectionResult> future = runtime.submit("yolov8n", from_cv_mat(mat, ColorOrder::Rgb));

  // submit() must have deep-copied the CPU pixels (adopt_cpu_pixels). Overwrite
  // and then release the Mat immediately; the queued request must not care.
  mat.setTo(cv::Scalar(0, 0, 0));
  mat.release();

  const DetectionResult actual = future.get();
  expect_same_detections(expected, actual);
  runtime.shutdown();
}

#endif  // VIE_HAS_ONNXRUNTIME

}  // namespace
}  // namespace vie