#include "test_util.hpp"

#include "vie/engine/inference_engine.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

namespace {

vie::PreprocessConfig base_config() {
  vie::PreprocessConfig config;
  config.resize_mode = vie::ResizeMode::None;
  config.target_width = 2;
  config.target_height = 2;
  config.interpolation = vie::Interpolation::Nearest;
  config.input_color = vie::ColorOrder::Rgb;
  config.output_color = vie::ColorOrder::Rgb;
  config.input_layout = vie::ImageLayout::Hwc;
  config.output_layout = vie::ImageLayout::Chw;
  config.output_dtype = vie::DataType::Fp32;
  config.padding_value = 0.f;
  return config;
}

vie::ImageInput image_from(const std::vector<std::uint8_t>& pixels, int width, int height, int channels,
                           vie::ColorOrder color, vie::ImageLayout layout, std::size_t stride = 0) {
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = channels;
  image.color = color;
  image.layout = layout;
  image.dtype = vie::DataType::Uint8;
  image.row_stride_bytes = stride;
  image.device = vie::DeviceType::Cpu;
  return image;
}

const float* chw(const vie::Tensor& tensor) { return tensor.data_as<float>(); }

}  // namespace

TEST(PreprocessCpu, RgbBgrAndLayoutConversion) {
  const std::vector<std::uint8_t> rgb = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  vie::ImagePreprocessor preprocessor;
  vie::PreprocessConfig config = base_config();
  preprocessor.configure(config);
  const vie::PreprocessResult same =
      preprocessor.execute(image_from(rgb, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc),
                           vie::DeviceContext{});
  EXPECT_EQ(same.output.shape(), (std::vector<std::int64_t>{1, 3, 2, 2}));
  EXPECT_EQ(same.output.layout(), vie::Layout::Nchw);
  EXPECT_FLOAT_EQ(chw(same.output)[0], 10.f);
  EXPECT_FLOAT_EQ(chw(same.output)[4], 20.f);
  EXPECT_FLOAT_EQ(chw(same.output)[8], 30.f);
  EXPECT_FLOAT_EQ(chw(same.output)[3], 100.f);

  config.input_color = vie::ColorOrder::Bgr;
  config.output_color = vie::ColorOrder::Rgb;
  preprocessor.configure(config);
  const vie::PreprocessResult swapped =
      preprocessor.execute(image_from(rgb, 2, 2, 3, vie::ColorOrder::Bgr, vie::ImageLayout::Hwc),
                           vie::DeviceContext{});
  EXPECT_FLOAT_EQ(chw(swapped.output)[0], 30.f);
  EXPECT_FLOAT_EQ(chw(swapped.output)[4], 20.f);
  EXPECT_FLOAT_EQ(chw(swapped.output)[8], 10.f);

  config.output_layout = vie::ImageLayout::Hwc;
  config.output_dtype = vie::DataType::Uint8;
  config.input_color = vie::ColorOrder::Rgb;
  config.output_color = vie::ColorOrder::Bgr;
  preprocessor.configure(config);
  const vie::PreprocessResult hwc =
      preprocessor.execute(image_from(rgb, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc),
                           vie::DeviceContext{});
  EXPECT_EQ(hwc.output.shape(), (std::vector<std::int64_t>{1, 2, 2, 3}));
  EXPECT_EQ(hwc.output.layout(), vie::Layout::Nhwc);
  EXPECT_EQ(hwc.output.dtype(), vie::DataType::Uint8);
  EXPECT_EQ(hwc.output.data_as<std::uint8_t>()[0], 30);
  EXPECT_EQ(hwc.output.data_as<std::uint8_t>()[1], 20);
  EXPECT_EQ(hwc.output.data_as<std::uint8_t>()[2], 10);
}

TEST(PreprocessCpu, GrayChwInputAndNormalize) {
  const std::vector<std::uint8_t> gray = {0, 255, 128, 64};
  vie::PreprocessConfig config = base_config();
  config.input_color = vie::ColorOrder::Gray;
  config.normalize_enabled = true;
  config.scale = 1.f / 255.f;
  config.mean = {0.f, 0.f, 0.f};
  config.std = {1.f, 1.f, 1.f};
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(config);
  const vie::PreprocessResult result =
      preprocessor.execute(image_from(gray, 2, 2, 1, vie::ColorOrder::Gray, vie::ImageLayout::Hwc),
                           vie::DeviceContext{});
  EXPECT_NEAR(chw(result.output)[1], 1.f, 1e-6);
  EXPECT_NEAR(chw(result.output)[5], 1.f, 1e-6);
  EXPECT_NEAR(chw(result.output)[2], 128.f / 255.f, 1e-6);

  std::vector<std::uint8_t> planes(12);
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 4; ++i) planes[static_cast<std::size_t>(c * 4 + i)] = static_cast<std::uint8_t>(10 * c + i);
  }
  config = base_config();
  config.input_layout = vie::ImageLayout::Chw;
  config.normalize_enabled = false;
  preprocessor.configure(config);
  const vie::PreprocessResult chw_in =
      preprocessor.execute(image_from(planes, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Chw),
                           vie::DeviceContext{});
  EXPECT_FLOAT_EQ(chw(chw_in.output)[0], 0.f);
  EXPECT_FLOAT_EQ(chw(chw_in.output)[3], 3.f);
  EXPECT_FLOAT_EQ(chw(chw_in.output)[4], 10.f);
}

TEST(PreprocessCpu, ResizeLetterboxGeometryAndPixels) {
  vie::PreprocessConfig config = base_config();
  config.resize_mode = vie::ResizeMode::Stretch;
  config.target_width = 1;
  config.target_height = 1;
  config.interpolation = vie::Interpolation::Bilinear;
  config.output_dtype = vie::DataType::Uint8;
  const std::vector<std::uint8_t> pixels = {0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30};
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(config);
  const vie::PreprocessResult reduced =
      preprocessor.execute(image_from(pixels, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc),
                           vie::DeviceContext{});
  EXPECT_EQ(reduced.output.data_as<std::uint8_t>()[0], 15);
  EXPECT_DOUBLE_EQ(reduced.geometry.scale_x, 0.5);
  EXPECT_DOUBLE_EQ(reduced.geometry.scale_y, 0.5);
  EXPECT_DOUBLE_EQ(reduced.geometry.uniform_scale, 0);

  config.resize_mode = vie::ResizeMode::Letterbox;
  config.target_width = 8;
  config.target_height = 8;
  config.interpolation = vie::Interpolation::Nearest;
  config.padding_value = 114.f;
  config.output_dtype = vie::DataType::Uint8;
  config.output_layout = vie::ImageLayout::Hwc;
  const std::vector<std::uint8_t> odd(5 * 3 * 3, 7);
  preprocessor.configure(config);
  const vie::GeometryTransform geometry = vie::make_geometry(config, 5, 3);
  EXPECT_EQ(geometry.resized_width, 8);
  EXPECT_EQ(geometry.resized_height, 5);
  EXPECT_EQ(geometry.pad_left, 0);
  EXPECT_EQ(geometry.pad_right, 0);
  EXPECT_EQ(geometry.pad_top, 1);
  EXPECT_EQ(geometry.pad_bottom, 2);
  EXPECT_NE(geometry.scale_x, geometry.scale_y);
  EXPECT_EQ(geometry.pad_top + geometry.resized_height + geometry.pad_bottom, 8);

  const vie::PreprocessResult letter =
      preprocessor.execute(image_from(odd, 5, 3, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc),
                           vie::DeviceContext{});
  EXPECT_EQ(letter.geometry.pad_top, 1);
  EXPECT_EQ(letter.output.data_as<std::uint8_t>()[0], 114);
  const std::size_t content = static_cast<std::size_t>((1 * 8) * 3);
  EXPECT_EQ(letter.output.data_as<std::uint8_t>()[content], 7);
}

TEST(PreprocessCpu, FloatInputKeepsValues) {
  const std::vector<float> pixels = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f,
                                     7.5f, 8.5f, 9.5f, 10.5f, 11.5f, 12.5f};
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = 2;
  image.height = 2;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Fp32;
  vie::PreprocessConfig config = base_config();
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(config);
  const vie::PreprocessResult result = preprocessor.execute(image, vie::DeviceContext{});
  EXPECT_FLOAT_EQ(chw(result.output)[0], 1.5f);
  EXPECT_FLOAT_EQ(chw(result.output)[4], 2.5f);
  EXPECT_FLOAT_EQ(chw(result.output)[8], 3.5f);
}

TEST(PreprocessCpu, NoncontiguousRowStrideIsNotReadAsPixels) {
  std::vector<std::uint8_t> rows(2 * 8, 255);
  rows[0] = 1;
  rows[1] = 2;
  rows[2] = 3;
  rows[3] = 4;
  rows[4] = 5;
  rows[5] = 6;
  rows[8] = 7;
  rows[9] = 8;
  rows[10] = 9;
  rows[11] = 10;
  rows[12] = 11;
  rows[13] = 12;
  vie::PreprocessConfig config = base_config();
  config.output_dtype = vie::DataType::Uint8;
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(config);
  const vie::PreprocessResult result = preprocessor.execute(
      image_from(rows, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc, 8), vie::DeviceContext{});
  EXPECT_EQ(result.output.data_as<std::uint8_t>()[0], 1);
  EXPECT_EQ(result.output.data_as<std::uint8_t>()[2], 7);
  for (int i = 0; i < 12; ++i) EXPECT_NE(result.output.data_as<std::uint8_t>()[i], 255);
}

TEST(PreprocessCpu, RejectsIllegalInputAndConfig) {
  vie::ImagePreprocessor preprocessor;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, preprocessor.configure(vie::PreprocessConfig{}));
  vie::PreprocessConfig config = base_config();
  config.std = {0.f, 1.f, 1.f};
  config.mean = {0.f, 0.f};
  config.normalize_enabled = true;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, preprocessor.configure(config));
  config = base_config();
  config.normalize_enabled = true;
  config.scale = 1.f;
  config.mean = {0.f, 0.f, 0.f};
  config.std = {1.f, 0.f, 1.f};
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, preprocessor.configure(config));

  config = base_config();
  preprocessor.configure(config);
  std::vector<std::uint8_t> pixels(12, 1);
  vie::ImageInput image = image_from(pixels, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc);
  image.data = nullptr;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, preprocessor.validate(image));
  image = image_from(pixels, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc);
  image.channels = 4;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, preprocessor.validate(image));
  image = image_from(pixels, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc, 2);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, preprocessor.validate(image));
  image = image_from(pixels, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc);
  image.device = vie::DeviceType::Cuda;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   preprocessor.execute(image, vie::DeviceContext{}));

  config.resize_mode = vie::ResizeMode::None;
  config.target_width = 4;
  preprocessor.configure(config);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   preprocessor.execute(image_from(pixels, 2, 2, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc),
                                        vie::DeviceContext{}));
}

TEST(PreprocessCpu, ManifestDoesNotInventMissingParameters) {
  vie::ModelManifest manifest = vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   vie::preprocess_config_from_manifest(manifest, vie::ColorOrder::Bgr, vie::ImageLayout::Hwc));
  EXPECT_TRUE(manifest.input.interpolation.empty());
  EXPECT_FALSE(manifest.input.padding_value.has_value());

  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  const vie::PreprocessConfig config =
      vie::preprocess_config_from_manifest(manifest, vie::ColorOrder::Bgr, vie::ImageLayout::Hwc);
  EXPECT_EQ(config.resize_mode, vie::ResizeMode::Letterbox);
  EXPECT_EQ(config.output_color, vie::ColorOrder::Rgb);
  EXPECT_EQ(config.input_color, vie::ColorOrder::Bgr);
  EXPECT_EQ(config.target_width, 8);
  EXPECT_TRUE(config.normalize_enabled);
  EXPECT_FLOAT_EQ(config.padding_value, 114.f);
  EXPECT_EQ(config.interpolation, vie::Interpolation::Bilinear);

  manifest.input.resize = "crop";
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::preprocess_config_from_manifest(manifest, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc));

  vie::InferenceEngine engine;
  vie::ModelConfig loaded;
  loaded.manifest = vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
  engine.load(loaded);
  std::vector<std::uint8_t> pixels(4 * 4 * 3, 1);
  vie::ImageInput image = image_from(pixels, 4, 4, 3, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc);
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, engine.preprocess(image));
  vie::BackendRequest request;
  request.inputs.emplace("images", vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32, {1, 3, 8, 8}));
  EXPECT_EQ(engine.execute(request).tensors.count("output0"), 1u);

  manifest = loaded.manifest.value();
  manifest.input.interpolation = "nearest";
  manifest.input.padding_value = 0.f;
  manifest.provenance["input.interpolation"] = vie::FieldProvenance{vie::Certainty::User, {"test"}};
  manifest.provenance["input.padding_value"] = vie::FieldProvenance{vie::Certainty::User, {"test"}};
  const std::string json = manifest.to_json_string();
  const vie::ModelManifest round = vie::ModelManifest::from_json_string(json);
  EXPECT_EQ(round.input.interpolation, "nearest");
  ASSERT_TRUE(round.input.padding_value.has_value());
  EXPECT_FLOAT_EQ(*round.input.padding_value, 0.f);
}

TEST(PreprocessBenchmark, CpuStages) {
  constexpr int kSrcW = 640;
  constexpr int kSrcH = 480;
  constexpr int kDst = 640;
  constexpr int kRuns = 20;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSrcW * kSrcH * 3), 17);
  std::vector<std::uint8_t> square(static_cast<std::size_t>(kDst * kDst * 3), 17);
  auto time_runs = [&](const vie::PreprocessConfig& config, const std::vector<std::uint8_t>& source, int width,
                       int height) {
    vie::ImagePreprocessor preprocessor;
    preprocessor.configure(config);
    const vie::ImageInput image =
        image_from(source, width, height, 3, config.input_color, vie::ImageLayout::Hwc);
    preprocessor.execute(image, vie::DeviceContext{});
    std::vector<double> samples;
    samples.reserve(kRuns);
    for (int i = 0; i < kRuns; ++i) {
      const auto start = std::chrono::steady_clock::now();
      preprocessor.execute(image, vie::DeviceContext{});
      const auto stop = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    double sum = 0;
    for (double sample : samples) sum += sample;
    return std::tuple<double, double, double>{sum / static_cast<double>(kRuns), samples[kRuns / 2],
                                              samples[static_cast<std::size_t>(kRuns * 0.95)]};
  };

  vie::PreprocessConfig resize = base_config();
  resize.resize_mode = vie::ResizeMode::Stretch;
  resize.target_width = kDst;
  resize.target_height = kDst;
  resize.interpolation = vie::Interpolation::Bilinear;
  resize.output_layout = vie::ImageLayout::Hwc;
  resize.output_dtype = vie::DataType::Uint8;
  vie::PreprocessConfig letter = resize;
  letter.resize_mode = vie::ResizeMode::Letterbox;
  letter.padding_value = 114.f;
  vie::PreprocessConfig normalize = base_config();
  normalize.target_width = kDst;
  normalize.target_height = kDst;
  normalize.normalize_enabled = true;
  normalize.scale = 1.f / 255.f;
  normalize.mean = {0.f, 0.f, 0.f};
  normalize.std = {1.f, 1.f, 1.f};
  vie::PreprocessConfig full = letter;
  full.input_color = vie::ColorOrder::Bgr;
  full.output_color = vie::ColorOrder::Rgb;
  full.output_layout = vie::ImageLayout::Chw;
  full.output_dtype = vie::DataType::Fp32;
  full.normalize_enabled = true;
  full.scale = 1.f / 255.f;
  full.mean = {0.f, 0.f, 0.f};
  full.std = {1.f, 1.f, 1.f};

  const auto resize_stat = time_runs(resize, pixels, kSrcW, kSrcH);
  const auto letter_stat = time_runs(letter, pixels, kSrcW, kSrcH);
  const auto normalize_stat = time_runs(normalize, square, kDst, kDst);
  const auto full_stat = time_runs(full, pixels, kSrcW, kSrcH);
  std::ofstream out("/tmp/vie_phase431_cpu.txt");
  out << "image=640x480 output=640x640 dtype=uint8_or_fp32 device=cpu sync=none runs=" << kRuns << "\n";
  out << "resize_mean_ms=" << std::get<0>(resize_stat) << " p50=" << std::get<1>(resize_stat)
      << " p95=" << std::get<2>(resize_stat) << "\n";
  out << "letterbox_mean_ms=" << std::get<0>(letter_stat) << " p50=" << std::get<1>(letter_stat)
      << " p95=" << std::get<2>(letter_stat) << "\n";
  out << "normalize_mean_ms=" << std::get<0>(normalize_stat) << " p50=" << std::get<1>(normalize_stat)
      << " p95=" << std::get<2>(normalize_stat) << "\n";
  out << "full_mean_ms=" << std::get<0>(full_stat) << " p50=" << std::get<1>(full_stat)
      << " p95=" << std::get<2>(full_stat) << "\n";
  EXPECT_GT(std::get<0>(full_stat), 0);
}
