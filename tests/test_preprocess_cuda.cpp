#include "test_util.hpp"

#include "vie/preprocess/preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

namespace {

void cuda_check(cudaError_t status) { ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status); }

vie::PreprocessConfig letterbox_config() {
  vie::PreprocessConfig config;
  config.resize_mode = vie::ResizeMode::Letterbox;
  config.target_width = 8;
  config.target_height = 6;
  config.interpolation = vie::Interpolation::Bilinear;
  config.input_color = vie::ColorOrder::Bgr;
  config.output_color = vie::ColorOrder::Rgb;
  config.input_layout = vie::ImageLayout::Hwc;
  config.output_layout = vie::ImageLayout::Chw;
  config.output_dtype = vie::DataType::Fp32;
  config.normalize_enabled = true;
  config.scale = 1.f / 255.f;
  config.mean = {0.f, 0.f, 0.f};
  config.std = {1.f, 1.f, 1.f};
  config.padding_value = 114.f;
  return config;
}

std::vector<std::uint8_t> make_pixels(int width, int height) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width * height * 3));
  for (int i = 0; i < width * height; ++i) {
    pixels[static_cast<std::size_t>(i * 3)] = static_cast<std::uint8_t>(i % 251);
    pixels[static_cast<std::size_t>(i * 3 + 1)] = static_cast<std::uint8_t>((i * 3) % 251);
    pixels[static_cast<std::size_t>(i * 3 + 2)] = static_cast<std::uint8_t>((i * 5) % 251);
  }
  return pixels;
}

vie::ImageInput host_image(const std::vector<std::uint8_t>& pixels, int width, int height) {
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = vie::ColorOrder::Bgr;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  image.device = vie::DeviceType::Cpu;
  return image;
}

struct DeviceImage {
  std::uint8_t* data = nullptr;
  explicit DeviceImage(const std::vector<std::uint8_t>& pixels) {
    cuda_check(cudaMalloc(&data, pixels.size()));
    cuda_check(cudaMemcpy(data, pixels.data(), pixels.size(), cudaMemcpyHostToDevice));
  }
  ~DeviceImage() {
    if (data != nullptr) cudaFree(data);
  }
  DeviceImage(const DeviceImage&) = delete;
  DeviceImage& operator=(const DeviceImage&) = delete;
};

vie::ImageInput device_view(std::uint8_t* data, int width, int height) {
  vie::ImageInput image = host_image({}, width, height);
  image.data = data;
  image.device = vie::DeviceType::Cuda;
  image.device_id = 0;
  return image;
}

void expect_close(const vie::Tensor& cpu, const vie::Tensor& cuda_on_host) {
  ASSERT_EQ(cpu.shape(), cuda_on_host.shape());
  ASSERT_EQ(cpu.dtype(), cuda_on_host.dtype());
  if (cpu.dtype() == vie::DataType::Uint8) {
    const auto* left = cpu.data_as<std::uint8_t>();
    const auto* right = cuda_on_host.data_as<std::uint8_t>();
    for (std::size_t i = 0; i < cpu.numel(); ++i) EXPECT_EQ(left[i], right[i]) << i;
    return;
  }
  const float* left = cpu.data_as<float>();
  const float* right = cuda_on_host.data_as<float>();
  for (std::size_t i = 0; i < cpu.numel(); ++i) {
    EXPECT_NEAR(left[i], right[i], 1e-4) << i;
  }
}

}  // namespace

TEST(PreprocessCuda, MatchesCpuOnDeviceAndKeepsGeometry) {
  const std::vector<std::uint8_t> pixels = make_pixels(5, 3);
  const vie::PreprocessConfig config = letterbox_config();
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(config);
  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  const vie::PreprocessResult host = preprocessor.execute(host_image(pixels, 5, 3), cpu);
  DeviceImage device(pixels);
  vie::ImageInput input = device_view(device.data, 5, 3);
  const vie::PreprocessResult first = preprocessor.execute(input, gpu);
  EXPECT_TRUE(first.output.is_cuda());
  EXPECT_EQ(first.output.device_id(), 0);
  EXPECT_EQ(first.output.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(first.output.layout(), vie::Layout::Nchw);
  cudaPointerAttributes attributes{};
  cuda_check(cudaPointerGetAttributes(&attributes, first.output.device_data()));
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
  EXPECT_EQ(first.geometry.resized_width, host.geometry.resized_width);
  EXPECT_EQ(first.geometry.resized_height, host.geometry.resized_height);
  EXPECT_EQ(first.geometry.pad_left, host.geometry.pad_left);
  EXPECT_EQ(first.geometry.pad_top, host.geometry.pad_top);
  EXPECT_EQ(first.geometry.pad_right, host.geometry.pad_right);
  EXPECT_EQ(first.geometry.pad_bottom, host.geometry.pad_bottom);
  EXPECT_DOUBLE_EQ(first.geometry.scale_x, host.geometry.scale_x);
  EXPECT_DOUBLE_EQ(first.geometry.scale_y, host.geometry.scale_y);
  expect_close(host.output, first.output.to(cpu));

  const vie::PreprocessResult second = preprocessor.execute(input, gpu);
  expect_close(first.output.to(cpu), second.output.to(cpu));
  cudaFree(device.data);
  device.data = nullptr;
  expect_close(host.output, first.output.to(cpu));

  vie::PreprocessConfig bytes = config;
  bytes.output_dtype = vie::DataType::Uint8;
  bytes.normalize_enabled = false;
  bytes.interpolation = vie::Interpolation::Nearest;
  bytes.resize_mode = vie::ResizeMode::Stretch;
  preprocessor.configure(bytes);
  DeviceImage again(pixels);
  const vie::PreprocessResult cpu_bytes = preprocessor.execute(host_image(pixels, 5, 3), cpu);
  const vie::PreprocessResult gpu_bytes = preprocessor.execute(device_view(again.data, 5, 3), gpu);
  expect_close(cpu_bytes.output, gpu_bytes.output.to(cpu));

  const std::vector<float> floats = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f, 12.5f};
  vie::PreprocessConfig same;
  same.resize_mode = vie::ResizeMode::None;
  same.target_width = 2;
  same.target_height = 2;
  same.interpolation = vie::Interpolation::Nearest;
  same.input_color = vie::ColorOrder::Rgb;
  same.output_color = vie::ColorOrder::Rgb;
  same.output_layout = vie::ImageLayout::Chw;
  same.output_dtype = vie::DataType::Fp32;
  preprocessor.configure(same);
  float* device_float = nullptr;
  cuda_check(cudaMalloc(&device_float, floats.size() * sizeof(float)));
  cuda_check(cudaMemcpy(device_float, floats.data(), floats.size() * sizeof(float), cudaMemcpyHostToDevice));
  vie::ImageInput host_float;
  host_float.data = floats.data();
  host_float.width = 2;
  host_float.height = 2;
  host_float.channels = 3;
  host_float.color = vie::ColorOrder::Rgb;
  host_float.layout = vie::ImageLayout::Hwc;
  host_float.dtype = vie::DataType::Fp32;
  vie::ImageInput device_float_image = host_float;
  device_float_image.data = device_float;
  device_float_image.device = vie::DeviceType::Cuda;
  const vie::PreprocessResult cpu_float = preprocessor.execute(host_float, cpu);
  const vie::PreprocessResult gpu_float = preprocessor.execute(device_float_image, gpu);
  expect_close(cpu_float.output, gpu_float.output.to(cpu));
  cudaFree(device_float);
}

TEST(PreprocessCuda, RejectsHostInputAndBadImages) {
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(letterbox_config());
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  std::vector<std::uint8_t> pixels = make_pixels(5, 3);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, preprocessor.execute(host_image(pixels, 5, 3), gpu));
  vie::ImageInput bad = device_view(pixels.data(), 5, 3);
  bad.data = nullptr;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, preprocessor.execute(bad, gpu));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::DeviceContext(vie::DeviceType::Cuda, 99));
}

TEST(PreprocessBenchmark, CudaStages) {
  constexpr int kSrcW = 640;
  constexpr int kSrcH = 480;
  constexpr int kDst = 640;
  constexpr int kRuns = 20;
  const std::vector<std::uint8_t> pixels = make_pixels(kSrcW, kSrcH);
  const std::vector<std::uint8_t> square = make_pixels(kDst, kDst);
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  DeviceImage source(pixels);
  DeviceImage square_device(square);

  auto time_device = [&](const vie::PreprocessConfig& config, std::uint8_t* data, int width, int height) {
    vie::ImagePreprocessor preprocessor;
    preprocessor.configure(config);
    vie::ImageInput image = device_view(data, width, height);
    image.color = config.input_color;
    preprocessor.execute(image, gpu);
    std::vector<double> wall;
    std::vector<double> kernel;
    for (int i = 0; i < kRuns; ++i) {
      const auto start = std::chrono::steady_clock::now();
      const vie::PreprocessResult result = preprocessor.execute(image, gpu);
      const auto stop = std::chrono::steady_clock::now();
      wall.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
      kernel.push_back(result.device_ms);
    }
    auto summarize = [](std::vector<double> samples) {
      std::sort(samples.begin(), samples.end());
      double sum = 0;
      for (double sample : samples) sum += sample;
      const double count = static_cast<double>(samples.size());
      return std::tuple<double, double, double>{sum / count, samples[samples.size() / 2],
                                                samples[static_cast<std::size_t>(count * 0.95)]};
    };
    return std::tuple<std::tuple<double, double, double>, std::tuple<double, double, double>>{summarize(wall),
                                                                                             summarize(kernel)};
  };

  vie::PreprocessConfig resize;
  resize.resize_mode = vie::ResizeMode::Stretch;
  resize.target_width = kDst;
  resize.target_height = kDst;
  resize.interpolation = vie::Interpolation::Bilinear;
  resize.input_color = vie::ColorOrder::Rgb;
  resize.output_color = vie::ColorOrder::Rgb;
  resize.output_layout = vie::ImageLayout::Hwc;
  resize.output_dtype = vie::DataType::Uint8;
  vie::PreprocessConfig letter = resize;
  letter.resize_mode = vie::ResizeMode::Letterbox;
  letter.padding_value = 114.f;
  vie::PreprocessConfig normalize;
  normalize.resize_mode = vie::ResizeMode::None;
  normalize.target_width = kDst;
  normalize.target_height = kDst;
  normalize.interpolation = vie::Interpolation::Nearest;
  normalize.input_color = vie::ColorOrder::Rgb;
  normalize.output_color = vie::ColorOrder::Rgb;
  normalize.output_layout = vie::ImageLayout::Chw;
  normalize.output_dtype = vie::DataType::Fp32;
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

  std::vector<double> copies;
  for (int i = 0; i < kRuns; ++i) {
    const auto start = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpy(source.data, pixels.data(), pixels.size(), cudaMemcpyHostToDevice));
    cuda_check(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    copies.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  std::sort(copies.begin(), copies.end());
  double copy_sum = 0;
  for (double sample : copies) copy_sum += sample;

  const auto resize_stat = time_device(resize, source.data, kSrcW, kSrcH);
  const auto letter_stat = time_device(letter, source.data, kSrcW, kSrcH);
  const auto normalize_stat = time_device(normalize, square_device.data, kDst, kDst);
  const auto full_stat = time_device(full, source.data, kSrcW, kSrcH);
  std::ofstream out("/tmp/vie_phase431_cuda.txt");
  out << "image=640x480 output=640x640 device=cuda:0 runs=" << kRuns << "\n";
  auto write_pair = [&](const char* name, const auto& stat) {
    out << name << "_wall_mean_ms=" << std::get<0>(std::get<0>(stat)) << " p50=" << std::get<1>(std::get<0>(stat))
        << " p95=" << std::get<2>(std::get<0>(stat)) << "\n";
    out << name << "_kernel_sync_mean_ms=" << std::get<0>(std::get<1>(stat))
        << " p50=" << std::get<1>(std::get<1>(stat)) << " p95=" << std::get<2>(std::get<1>(stat)) << "\n";
  };
  out << "sync=cudaEventSynchronize around the fused kernel; wall time also includes output allocation\n";
  out << "h2d_mean_ms=" << copy_sum / kRuns << " p50=" << copies[kRuns / 2]
      << " p95=" << copies[static_cast<std::size_t>(kRuns * 0.95)] << "\n";
  write_pair("resize", resize_stat);
  write_pair("letterbox", letter_stat);
  write_pair("normalize", normalize_stat);
  write_pair("full", full_stat);
  EXPECT_GT(std::get<0>(std::get<1>(full_stat)), 0);
}
