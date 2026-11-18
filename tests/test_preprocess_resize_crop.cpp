#include "test_util.hpp"

#include "vie/decoder/detection.hpp"
#include "vie/model/manifest.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#endif

namespace {

int ref_round(int numerator, int multiplier, int denominator) {
  const long long product = static_cast<long long>(numerator) * multiplier;
  return static_cast<int>((product * 2 + denominator) / (denominator * 2));
}

struct RefSize {
  int resized_w = 0;
  int resized_h = 0;
  int crop_x = 0;
  int crop_y = 0;
};

RefSize ref_shortest(int width, int height, int side, int crop_w, int crop_h) {
  RefSize size;
  if (width < height) {
    size.resized_w = side;
    size.resized_h = ref_round(height, side, width);
  } else if (height < width) {
    size.resized_h = side;
    size.resized_w = ref_round(width, side, height);
  } else {
    size.resized_w = side;
    size.resized_h = side;
  }
  size.crop_x = (size.resized_w - crop_w) / 2;
  size.crop_y = (size.resized_h - crop_h) / 2;
  return size;
}

float ref_source(int content, int source_size, int resized_size) {
  return (static_cast<float>(content) + 0.5f) * (static_cast<float>(source_size) / static_cast<float>(resized_size)) -
         0.5f;
}

int ref_nearest(float coordinate, int limit) {
  const int index = static_cast<int>(coordinate >= 0.f ? coordinate + 0.5f : coordinate - 0.5f);
  if (index < 0) return 0;
  if (index >= limit) return limit - 1;
  return index;
}

int ref_clamp(int value, int limit) {
  if (value < 0) return 0;
  if (value >= limit) return limit - 1;
  return value;
}

float ref_sample(const std::vector<float>& plane, int width, int height, float src_x, float src_y, bool nearest) {
  if (nearest) {
    return plane[static_cast<std::size_t>(ref_nearest(src_y, height) * width + ref_nearest(src_x, width))];
  }
  const int x0 = static_cast<int>(std::floor(src_x));
  const int y0 = static_cast<int>(std::floor(src_y));
  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);
  const auto at = [&](int x, int y) {
    return plane[static_cast<std::size_t>(ref_clamp(y, height) * width + ref_clamp(x, width))];
  };
  const float top = at(x0, y0) * (1.f - fx) + at(x0 + 1, y0) * fx;
  const float bottom = at(x0, y0 + 1) * (1.f - fx) + at(x0 + 1, y0 + 1) * fx;
  return top * (1.f - fy) + bottom * fy;
}

vie::PreprocessConfig shortest_config(int side, int crop_w, int crop_h, vie::Interpolation interpolation) {
  vie::PreprocessConfig config;
  config.resize_mode = vie::ResizeMode::ShortestSide;
  config.short_side = side;
  config.crop_width = crop_w;
  config.crop_height = crop_h;
  config.target_width = crop_w;
  config.target_height = crop_h;
  config.interpolation = interpolation;
  config.input_color = vie::ColorOrder::Rgb;
  config.output_color = vie::ColorOrder::Rgb;
  config.input_layout = vie::ImageLayout::Hwc;
  config.output_layout = vie::ImageLayout::Chw;
  config.output_dtype = vie::DataType::Fp32;
  return config;
}

vie::ImageInput image_u8(const std::vector<std::uint8_t>& pixels, int width, int height,
                         vie::ColorOrder color = vie::ColorOrder::Rgb) {
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = color;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

void expect_code(const std::function<void()>& call, vie::ErrorCode code) {
  try {
    call();
    FAIL() << "expected VieError";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), code) << error.what();
  }
}

}  // namespace

TEST(PreprocessResize, ShortestSideRoundsTheLongEdgeAndPinsTheShortEdge) {
  const vie::GeometryTransform wide = vie::make_geometry(shortest_config(4, 3, 3, vie::Interpolation::Nearest), 7, 5);
  const RefSize wide_ref = ref_shortest(7, 5, 4, 3, 3);
  EXPECT_EQ(wide.resized_width, 6);
  EXPECT_EQ(wide.resized_height, 4);
  EXPECT_EQ(wide.resized_width, wide_ref.resized_w);
  EXPECT_EQ(wide.resized_height, wide_ref.resized_h);
  EXPECT_EQ(wide.crop_x, 1);
  EXPECT_EQ(wide.crop_y, 0);
  EXPECT_EQ(wide.output_width, 3);
  EXPECT_EQ(wide.output_height, 3);
  EXPECT_EQ(wide.pad_left, 0);
  EXPECT_EQ(wide.pad_top, 0);
  EXPECT_NEAR(wide.uniform_scale, 4.0 / 5.0, 1e-12);
  EXPECT_NEAR(wide.scale_y, 4.0 / 5.0, 1e-12);
  EXPECT_NEAR(wide.scale_x, 6.0 / 7.0, 1e-12);

  const vie::GeometryTransform tall = vie::make_geometry(shortest_config(4, 2, 3, vie::Interpolation::Nearest), 5, 7);
  EXPECT_EQ(tall.resized_width, 4);
  EXPECT_EQ(tall.resized_height, 6);
  EXPECT_EQ(tall.crop_x, 1);
  EXPECT_EQ(tall.crop_y, 1);

  const vie::GeometryTransform square =
      vie::make_geometry(shortest_config(5, 5, 5, vie::Interpolation::Bilinear), 10, 10);
  EXPECT_EQ(square.resized_width, 5);
  EXPECT_EQ(square.resized_height, 5);
  EXPECT_EQ(square.crop_x, 0);
  EXPECT_EQ(square.crop_y, 0);

  const vie::GeometryTransform extreme =
      vie::make_geometry(shortest_config(8, 4, 4, vie::Interpolation::Nearest), 1000, 1);
  EXPECT_EQ(extreme.resized_height, 8);
  EXPECT_EQ(extreme.resized_width, 8000);
  EXPECT_EQ(extreme.crop_x, (8000 - 4) / 2);
  EXPECT_EQ(extreme.crop_y, (8 - 4) / 2);
  EXPECT_GE(extreme.resized_width, extreme.output_width);
  EXPECT_GE(extreme.resized_height, extreme.output_height);
}

TEST(PreprocessResize, CenterCropRejectsAnOversizedWindowAndIllegalSizes) {
  expect_code([] { vie::validate_preprocess_config(shortest_config(4, 5, 4, vie::Interpolation::Nearest)); },
              vie::ErrorCode::InvalidArgument);
  expect_code([] { vie::validate_preprocess_config(shortest_config(0, 2, 2, vie::Interpolation::Nearest)); },
              vie::ErrorCode::InvalidArgument);
  expect_code([] { vie::validate_preprocess_config(shortest_config(4, 0, 2, vie::Interpolation::Nearest)); },
              vie::ErrorCode::InvalidArgument);
  vie::PreprocessConfig negative = shortest_config(4, 2, 2, vie::Interpolation::Nearest);
  negative.crop_height = -3;
  expect_code([&] { vie::validate_preprocess_config(negative); }, vie::ErrorCode::InvalidArgument);
  vie::PreprocessConfig mismatch = shortest_config(8, 4, 4, vie::Interpolation::Nearest);
  mismatch.target_width = 5;
  expect_code([&] { vie::validate_preprocess_config(mismatch); }, vie::ErrorCode::InvalidArgument);
}

TEST(PreprocessResize, NearestAndBilinearMatchAnIndependentSampler) {
  const std::vector<std::uint8_t> row = {0, 0, 0, 10, 10, 10};
  vie::ImagePreprocessor preprocessor;
  vie::PreprocessConfig nearest = shortest_config(2, 2, 2, vie::Interpolation::Nearest);
  preprocessor.configure(nearest);
  const vie::PreprocessResult nearest_result =
      preprocessor.execute(image_u8(row, 2, 1), vie::DeviceContext{});
  const float* nearest_data = nearest_result.output.data_as<float>();
  EXPECT_FLOAT_EQ(nearest_data[0], 0.f);
  EXPECT_FLOAT_EQ(nearest_data[1], 10.f);
  EXPECT_FLOAT_EQ(nearest_data[2], 0.f);
  EXPECT_FLOAT_EQ(nearest_data[3], 10.f);

  vie::PreprocessConfig bilinear = shortest_config(2, 2, 2, vie::Interpolation::Bilinear);
  preprocessor.configure(bilinear);
  const vie::PreprocessResult bilinear_result =
      preprocessor.execute(image_u8(row, 2, 1), vie::DeviceContext{});
  EXPECT_FLOAT_EQ(bilinear_result.output.data_as<float>()[0], 2.5f);

  std::mt19937 rng(64);
  std::uniform_int_distribution<int> pixel(0, 255);
  const int cases[][4] = {{8, 5, 4, 3}, {5, 9, 4, 4}, {6, 6, 5, 2}, {3, 4, 3, 3}};
  for (const auto& item : cases) {
    const int width = item[0];
    const int height = item[1];
    const int side = item[2];
    const int crop = item[3];
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width * height * 3));
    for (std::uint8_t& value : bytes) value = static_cast<std::uint8_t>(pixel(rng));
    std::vector<float> red(static_cast<std::size_t>(width * height));
    std::vector<float> green(red.size());
    std::vector<float> blue(red.size());
    for (int i = 0; i < width * height; ++i) {
      red[static_cast<std::size_t>(i)] = bytes[static_cast<std::size_t>(i * 3)];
      green[static_cast<std::size_t>(i)] = bytes[static_cast<std::size_t>(i * 3 + 1)];
      blue[static_cast<std::size_t>(i)] = bytes[static_cast<std::size_t>(i * 3 + 2)];
    }
    for (const vie::Interpolation interpolation : {vie::Interpolation::Nearest, vie::Interpolation::Bilinear}) {
      const vie::PreprocessConfig config = shortest_config(side, crop, crop, interpolation);
      preprocessor.configure(config);
      const vie::PreprocessResult result =
          preprocessor.execute(image_u8(bytes, width, height), vie::DeviceContext{});
      const RefSize expected = ref_shortest(width, height, side, crop, crop);
      EXPECT_EQ(result.geometry.resized_width, expected.resized_w);
      EXPECT_EQ(result.geometry.resized_height, expected.resized_h);
      EXPECT_EQ(result.geometry.crop_x, expected.crop_x);
      EXPECT_EQ(result.geometry.crop_y, expected.crop_y);
      const bool use_nearest = interpolation == vie::Interpolation::Nearest;
      const float* out = result.output.data_as<float>();
      for (int y = 0; y < crop; ++y) {
        for (int x = 0; x < crop; ++x) {
          const float src_x = ref_source(x + expected.crop_x, width, expected.resized_w);
          const float src_y = ref_source(y + expected.crop_y, height, expected.resized_h);
          const float channels[3] = {ref_sample(red, width, height, src_x, src_y, use_nearest),
                                     ref_sample(green, width, height, src_x, src_y, use_nearest),
                                     ref_sample(blue, width, height, src_x, src_y, use_nearest)};
          for (int c = 0; c < 3; ++c) {
            const int index = (c * crop + y) * crop + x;
            EXPECT_FLOAT_EQ(out[index], channels[c]) << width << "x" << height << " @" << x << "," << y;
          }
        }
      }
    }
  }
}

TEST(PreprocessResize, NormalizationOrderScaleMeanStdAndColor) {
  vie::PreprocessConfig config;
  config.resize_mode = vie::ResizeMode::None;
  config.target_width = 1;
  config.target_height = 1;
  config.input_color = vie::ColorOrder::Bgr;
  config.output_color = vie::ColorOrder::Rgb;
  config.input_layout = vie::ImageLayout::Hwc;
  config.output_layout = vie::ImageLayout::Chw;
  config.output_dtype = vie::DataType::Fp32;
  config.normalize_enabled = true;
  config.scale = 1.f / 255.f;
  config.mean = {0.5f, 0.25f, 0.f};
  config.std = {0.5f, 0.5f, 2.f};
  const std::vector<std::uint8_t> bgr = {10, 20, 30};
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(config);
  const vie::PreprocessResult normalized =
      preprocessor.execute(image_u8(bgr, 1, 1, vie::ColorOrder::Bgr), vie::DeviceContext{});
  const float* value = normalized.output.data_as<float>();
  const float red = (30.f * config.scale - 0.5f) / 0.5f;
  const float green = (20.f * config.scale - 0.25f) / 0.5f;
  const float blue = (10.f * config.scale - 0.f) / 2.f;
  EXPECT_FLOAT_EQ(value[0], red);
  EXPECT_FLOAT_EQ(value[1], green);
  EXPECT_FLOAT_EQ(value[2], blue);

  const std::vector<float> floats = {30.f, 20.f, 10.f};
  config.input_color = vie::ColorOrder::Rgb;
  config.output_color = vie::ColorOrder::Rgb;
  preprocessor.configure(config);
  vie::ImageInput float_image = image_u8(bgr, 1, 1);
  float_image.data = floats.data();
  float_image.dtype = vie::DataType::Fp32;
  float_image.color = vie::ColorOrder::Rgb;
  const vie::PreprocessResult from_float_result = preprocessor.execute(float_image, vie::DeviceContext{});
  const float* from_float = from_float_result.output.data_as<float>();
  EXPECT_FLOAT_EQ(from_float[0], (30.f * config.scale - 0.5f) / 0.5f);

  config.std = {1.f, 0.f, 1.f};
  expect_code([&] { preprocessor.configure(config); }, vie::ErrorCode::InvalidArgument);
  config.std = {1.f, 1.f};
  config.mean = {1.f, 1.f};
  expect_code([&] { preprocessor.configure(config); }, vie::ErrorCode::InvalidArgument);
}

TEST(PreprocessResize, GeometryMapsForwardAndBackWithoutChangingLetterbox) {
  vie::PreprocessConfig letterbox;
  letterbox.resize_mode = vie::ResizeMode::Letterbox;
  letterbox.target_width = 8;
  letterbox.target_height = 8;
  letterbox.padding_value = 114.f;
  const vie::GeometryTransform old_letter = vie::make_geometry(letterbox, 5, 3);
  EXPECT_EQ(old_letter.resized_width, 8);
  EXPECT_EQ(old_letter.resized_height, 5);
  EXPECT_EQ(old_letter.pad_left, 0);
  EXPECT_EQ(old_letter.pad_top, 1);
  EXPECT_EQ(old_letter.pad_bottom, 2);
  EXPECT_EQ(old_letter.crop_x, 0);
  EXPECT_EQ(old_letter.crop_y, 0);

  vie::PreprocessConfig stretch;
  stretch.resize_mode = vie::ResizeMode::Stretch;
  stretch.target_width = 4;
  stretch.target_height = 2;
  const vie::GeometryTransform old_stretch = vie::make_geometry(stretch, 8, 2);
  EXPECT_DOUBLE_EQ(old_stretch.scale_x, 0.5);
  EXPECT_DOUBLE_EQ(old_stretch.scale_y, 1);
  EXPECT_DOUBLE_EQ(old_stretch.uniform_scale, 0);
  EXPECT_EQ(old_stretch.crop_x, 0);

  const vie::GeometryTransform crop = vie::make_geometry(shortest_config(4, 3, 3, vie::Interpolation::Nearest), 7, 5);
  const int origin_x = vie::geometry_model_origin_x(crop);
  const int origin_y = vie::geometry_model_origin_y(crop);
  EXPECT_EQ(origin_x, -crop.crop_x);
  EXPECT_EQ(origin_y, 0);
  const float model_x = 1.25f;
  const float model_y = 0.5f;
  const double original_x = (static_cast<double>(model_x) - origin_x) / crop.scale_x;
  const double original_y = (static_cast<double>(model_y) - origin_y) / crop.scale_y;
  EXPECT_NEAR(original_x * crop.scale_x + origin_x, model_x, 1e-9);
  EXPECT_NEAR(original_y * crop.scale_y + origin_y, model_y, 1e-9);
  vie::BoundingBox box;
  box.x1 = model_x;
  box.x2 = model_x + 0.5f;
  box.y1 = model_y;
  box.y2 = model_y + 0.5f;
  EXPECT_TRUE(vie::map_box_to_original(box, crop));
  EXPECT_NEAR(box.x1, static_cast<float>(original_x), 1e-5);
  EXPECT_NEAR(box.y1, static_cast<float>(original_y), 1e-5);

  vie::BoundingBox edge;
  edge.x1 = -100.f;
  edge.x2 = 100.f;
  edge.y1 = -100.f;
  edge.y2 = 100.f;
  EXPECT_TRUE(vie::map_box_to_original(edge, crop));
  EXPECT_FLOAT_EQ(edge.x1, 0.f);
  EXPECT_FLOAT_EQ(edge.y1, 0.f);
  EXPECT_FLOAT_EQ(edge.x2, static_cast<float>(crop.original_width));
  EXPECT_FLOAT_EQ(edge.y2, static_cast<float>(crop.original_height));
}

TEST(PreprocessResize, ManifestCarriesShortestSideAndExplicitNormalization) {
  vie::ModelManifest manifest;
  manifest.model_name = "classifier";
  manifest.task = vie::TaskType::Classification;
  manifest.model_format = vie::ModelFormat::Onnx;
  manifest.num_classes = 4;
  manifest.input.name = "input";
  manifest.input.shape = {1, 3, 3, 4};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.resize = "shortest_side";
  manifest.input.short_side = 4;
  manifest.input.crop_width = 4;
  manifest.input.crop_height = 3;
  manifest.input.interpolation = "bilinear";
  manifest.input.normalization = "explicit";
  manifest.input.scale = 1.f / 255.f;
  manifest.input.mean = {0.1f, 0.2f, 0.3f};
  manifest.input.std = {1.f, 1.f, 1.f};
  manifest.outputs.push_back(vie::OutputDesc{"logits", {1, 4}, vie::DataType::Fp32, vie::Layout::Unknown, "logits"});
  manifest.decoder.protocol = "classification_logits_raw";
  manifest.classification.output_name = "logits";
  manifest.classification.class_axis = 1;
  manifest.classification.score_type = "logits";
  manifest.ensure_provenance();
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_TRUE(validation.runnable()) << vie::format_issues(validation.unresolved_critical);
  const std::string json = manifest.to_json_string();
  EXPECT_NE(json.find("\"resize\": \"shortest_side\""), std::string::npos);
  EXPECT_NE(json.find("\"short_side\": 4"), std::string::npos);
  EXPECT_NE(json.find("\"normalization\": \"explicit\""), std::string::npos);
  const vie::ModelManifest loaded = vie::ModelManifest::from_json_string(json);
  EXPECT_EQ(loaded.input.short_side, 4);
  EXPECT_EQ(loaded.input.crop_width, 4);
  EXPECT_EQ(loaded.input.crop_height, 3);
  ASSERT_TRUE(loaded.input.scale.has_value());
  EXPECT_FLOAT_EQ(*loaded.input.scale, 1.f / 255.f);
  EXPECT_EQ(loaded.input.mean.size(), 3u);

  vie::ModelManifest detection;
  detection.input.resize = "stretch";
  detection.ensure_provenance();
  EXPECT_EQ(detection.to_json_string().find("short_side"), std::string::npos);
  EXPECT_EQ(detection.to_json_string().find("\"mean\""), std::string::npos);

  manifest.input.std = {1.f, 0.f, 1.f};
  manifest.ensure_provenance();
  const vie::ManifestValidation zero_std = vie::ManifestValidator{}.validate(manifest);
  EXPECT_FALSE(zero_std.schema_ok());

  manifest.input.std = {1.f, 1.f, 1.f};
  manifest.input.crop_width = 8;
  manifest.ensure_provenance();
  const vie::ManifestValidation too_wide = vie::ManifestValidator{}.validate(manifest);
  EXPECT_FALSE(too_wide.schema_ok());

  vie::ModelManifest missing = manifest;
  missing.input.crop_width = 4;
  missing.input.short_side.reset();
  missing.provenance.erase("input.short_side");
  missing.ensure_provenance();
  const vie::ManifestValidation unresolved = vie::ManifestValidator{}.validate(missing);
  EXPECT_TRUE(unresolved.schema_ok()) << vie::format_issues(unresolved.schema_issues);
  EXPECT_FALSE(unresolved.runnable());

  const vie::PreprocessConfig config =
      vie::preprocess_config_from_manifest(loaded, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc);
  EXPECT_EQ(config.resize_mode, vie::ResizeMode::ShortestSide);
  EXPECT_EQ(config.short_side, 4);
  EXPECT_EQ(config.crop_width, 4);
  EXPECT_EQ(config.crop_height, 3);
  EXPECT_TRUE(config.normalize_enabled);
  EXPECT_FLOAT_EQ(config.mean[2], 0.3f);
}

TEST(PreprocessResize, CpuTimingUsesTheSameImageForStretchAndShortestSide) {
  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kWidth * kHeight * 3));
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const std::size_t index = static_cast<std::size_t>(y * kWidth + x) * 3;
      pixels[index] = static_cast<std::uint8_t>(x & 255);
      pixels[index + 1] = static_cast<std::uint8_t>(y & 255);
      pixels[index + 2] = static_cast<std::uint8_t>((x + y) & 255);
    }
  }
  vie::PreprocessConfig shortest = shortest_config(256, 224, 224, vie::Interpolation::Bilinear);
  shortest.normalize_enabled = true;
  shortest.scale = 1.f / 255.f;
  shortest.mean = {0.485f, 0.456f, 0.406f};
  shortest.std = {0.229f, 0.224f, 0.225f};
  vie::PreprocessConfig stretch = shortest;
  stretch.resize_mode = vie::ResizeMode::Stretch;
  stretch.short_side = 0;
  stretch.crop_width = 0;
  stretch.crop_height = 0;
  stretch.target_width = 224;
  stretch.target_height = 224;
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(shortest);
  const vie::ImageInput image = image_u8(pixels, kWidth, kHeight);
  (void)preprocessor.execute(image, vie::DeviceContext{});
  const auto shortest_start = std::chrono::steady_clock::now();
  const vie::PreprocessResult shortest_result = preprocessor.execute(image, vie::DeviceContext{});
  const double shortest_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - shortest_start).count();
  preprocessor.configure(stretch);
  (void)preprocessor.execute(image, vie::DeviceContext{});
  const auto stretch_start = std::chrono::steady_clock::now();
  const vie::PreprocessResult stretch_result = preprocessor.execute(image, vie::DeviceContext{});
  const double stretch_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stretch_start).count();
  EXPECT_EQ(shortest_result.output.shape(), (std::vector<std::int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(shortest_result.geometry.resized_height, 256);
  EXPECT_EQ(shortest_result.geometry.output_width, 224);
  EXPECT_EQ(stretch_result.geometry.resized_width, 224);
  EXPECT_EQ(stretch_result.geometry.resized_height, 224);
  std::ofstream out("/tmp/vie_phase64_preprocess_timing.txt");
  out << "image=640x480 uint8 hwc rgb\n";
  out << "color_convert=no channel swap; rgb to rgb\n";
  out << "normalize=yes scale=1/255 mean/std set layout=nchw\n";
  out << "h2d=no device=cpu warmup=1 measured=1\n";
  out << "shortest_resized=" << shortest_result.geometry.resized_width << "x"
      << shortest_result.geometry.resized_height << "\n";
  out << "shortest_crop=" << shortest_result.geometry.output_width << "x" << shortest_result.geometry.output_height
      << " origin=" << shortest_result.geometry.crop_x << "," << shortest_result.geometry.crop_y << "\n";
  out << "shortest_preprocess_ms=" << shortest_ms << "\n";
  out << "stretch_resized=224x224 stretch_preprocess_ms=" << stretch_ms << "\n";
  EXPECT_GT(shortest_ms, 0);
  EXPECT_GT(stretch_ms, 0);
}

#if defined(VIE_HAS_ONNXRUNTIME)
TEST(PreprocessResize, UntrainedResNetRunsShortestSideCenterCropOnCpu) {
  const std::string path = "/tmp/resnet50_untrained_static.onnx";
  ASSERT_TRUE(std::filesystem::exists(path));
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult recognition = vie::default_recognizer_registry().recognize(report);
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, recognition, path);
  EXPECT_TRUE(manifest.input.resize.empty());
  EXPECT_TRUE(manifest.input.normalization.empty());
  EXPECT_FALSE(manifest.input.short_side.has_value());

  auto user = [&](const char* field) {
    manifest.provenance[field] = vie::FieldProvenance{vie::Certainty::User, {"test supplied preprocess"}};
  };
  manifest.input.color = "rgb";
  user("input.color");
  manifest.input.resize = "shortest_side";
  user("input.resize");
  manifest.input.short_side = 256;
  user("input.short_side");
  manifest.input.crop_width = 224;
  user("input.crop_width");
  manifest.input.crop_height = 224;
  user("input.crop_height");
  manifest.input.interpolation = "bilinear";
  user("input.interpolation");
  manifest.input.normalization = "explicit";
  user("input.normalization");
  manifest.input.scale = 1.f / 255.f;
  user("input.scale");
  manifest.input.mean = {0.485f, 0.456f, 0.406f};
  user("input.mean");
  manifest.input.std = {0.229f, 0.224f, 0.225f};
  user("input.std");
  manifest.ensure_provenance();
  const vie::ManifestValidation validation = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(validation.schema_ok()) << vie::format_issues(validation.schema_issues);
  EXPECT_TRUE(validation.runnable()) << vie::format_issues(validation.unresolved_critical);

  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kWidth * kHeight * 3), 0);
  for (int i = 0; i < kWidth * kHeight; ++i) pixels[static_cast<std::size_t>(i * 3)] = static_cast<std::uint8_t>(i & 255);
  const vie::PreprocessConfig config =
      vie::preprocess_config_from_manifest(manifest, vie::ColorOrder::Rgb, vie::ImageLayout::Hwc);
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(config);
  const vie::PreprocessResult prepared = preprocessor.execute(image_u8(pixels, kWidth, kHeight), vie::DeviceContext{});
  EXPECT_EQ(prepared.output.shape(), (std::vector<std::int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(prepared.output.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(prepared.geometry.resized_height, 256);
  EXPECT_EQ(prepared.geometry.output_width, 224);
  EXPECT_EQ(prepared.geometry.output_height, 224);
  for (float value : std::vector<float>(prepared.output.data_as<float>(),
                                        prepared.output.data_as<float>() + prepared.output.numel())) {
    ASSERT_TRUE(std::isfinite(value));
  }

  vie::ModelConfig model;
  model.model_path = path;
  model.manifest = manifest;
  model.backend = vie::BackendKind::OnnxRuntime;
  model.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  engine.load(model);
  vie::ClassificationRuntimeConfig runtime;
  runtime.top_k = 5;
  (void)engine.infer_classification(image_u8(pixels, kWidth, kHeight), runtime);
  vie::ClassificationInferTiming timing;
  const vie::ClassificationBatch batch =
      engine.infer_classification(image_u8(pixels, kWidth, kHeight), runtime, &timing);
  ASSERT_EQ(batch.results.size(), 1u);
  EXPECT_EQ(batch.results[0].num_classes, 1000);
  EXPECT_EQ(batch.results[0].output_bytes, 4000u);
  EXPECT_EQ(batch.results[0].labels.size(), 5u);
  EXPECT_EQ(batch.results[0].score_type, "logits");
  for (const vie::Classification& label : batch.results[0].labels) {
    EXPECT_TRUE(std::isfinite(label.score));
    EXPECT_GE(label.class_id, 0);
    EXPECT_LT(label.class_id, 1000);
  }
  std::ofstream out("/tmp/vie_phase64_resnet_timing.txt");
  out << "image=640x480 uint8 hwc rgb color_swap=no normalize=explicit bilinear nchw h2d=no\n";
  out << "resized=" << prepared.geometry.resized_width << "x" << prepared.geometry.resized_height
      << " crop=224x224\n";
  out << "preprocess_ms=" << timing.preprocess_ms << " execute_ms=" << timing.execute_ms
      << " decode_ms=" << timing.decode_ms << " topk_ms=" << timing.topk_ms << " total_ms=" << timing.total_ms
      << "\n";
  EXPECT_GT(timing.preprocess_ms, 0);
  EXPECT_GT(timing.execute_ms, 0);
  EXPECT_GT(timing.total_ms, 0);
}
#endif
