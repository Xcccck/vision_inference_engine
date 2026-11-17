#include "test_util.hpp"

#include "vie/decoder/yolov8_detect_decoder.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <tuple>
#include <vector>

namespace {

void cuda_check(cudaError_t status) { ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status); }

vie::GeometryTransform identity(int width, int height) {
  vie::GeometryTransform geometry;
  geometry.original_width = width;
  geometry.original_height = height;
  geometry.resized_width = width;
  geometry.resized_height = height;
  geometry.output_width = width;
  geometry.output_height = height;
  geometry.scale_x = 1;
  geometry.scale_y = 1;
  geometry.uniform_scale = 1;
  return geometry;
}

vie::DecoderConfig config_for(int classes) {
  vie::DecoderConfig config;
  config.output_name = "predictions";
  config.channel_axis = 1;
  config.box_channels = 4;
  config.num_classes = classes;
  config.box_decode = vie::BoxDecode::DecodedCxcywh;
  config.scores_are_probabilities = true;
  config.class_selection = vie::ClassSelection::BestClass;
  config.confidence_threshold = 0.5f;
  config.nms_enabled = true;
  config.class_aware_nms = true;
  config.iou_threshold = 0.5f;
  config.max_detections = 100;
  config.pre_nms_top_k = 1000;
  return config;
}

}  // namespace

TEST(YoloV8DecoderCuda, MatchesCpuOnSyntheticPredictions) {
  constexpr int kPredictions = 64;
  constexpr int kClasses = 4;
  std::vector<float> values(static_cast<std::size_t>(4 + kClasses) * kPredictions, 0.f);
  for (int i = 0; i < 8; ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(10 + i * 3);
    values[static_cast<std::size_t>(kPredictions + i)] = 12.f;
    values[static_cast<std::size_t>(2 * kPredictions + i)] = i == 1 ? 6.f : 4.f;
    values[static_cast<std::size_t>(3 * kPredictions + i)] = 4.f;
    values[static_cast<std::size_t>((4 + (i % kClasses)) * kPredictions + i)] = 0.4f + 0.07f * static_cast<float>(i);
  }
  values[static_cast<std::size_t>(1)] = values[0];
  values[static_cast<std::size_t>(kPredictions + 1)] = values[static_cast<std::size_t>(kPredictions)];
  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  vie::Tensor host = vie::Tensor::from_host(cpu, vie::DataType::Fp32, {1, 4 + kClasses, kPredictions}, values.data(),
                                            values.size() * sizeof(float));
  vie::Tensor device = host.to(gpu);
  vie::DecoderConfig config = config_for(kClasses);
  vie::GeometryTransform geometry = identity(80, 40);
  geometry.pad_left = 3;
  geometry.pad_top = 1;
  geometry.scale_x = 0.5;
  geometry.scale_y = 0.8;
  vie::DecoderInput input;
  input.geometry = geometry;
  input.outputs.emplace("predictions", host);
  const vie::DetectionBatch cpu_batch = vie::YoloV8DetectionDecoder{}.decode_batch(input, config, cpu);
  input.outputs.clear();
  input.outputs.emplace("predictions", device);
  const vie::DetectionBatch first = vie::YoloV8DetectionDecoder{}.decode_batch(input, config, gpu);
  const vie::DetectionBatch second = vie::YoloV8DetectionDecoder{}.decode_batch(input, config, gpu);
  ASSERT_EQ(first.results[0].detections.size(), cpu_batch.results[0].detections.size());
  EXPECT_EQ(first.results[0].execution_device, vie::DeviceType::Cuda);
  EXPECT_EQ(first.results[0].execution_device_id, 0);
  EXPECT_GT(first.results[0].gpu_execute_ms, 0);
  EXPECT_EQ(first.results[0].candidates_before_nms, cpu_batch.results[0].candidates_before_nms);
  for (std::size_t i = 0; i < first.results[0].detections.size(); ++i) {
    EXPECT_EQ(first.results[0].detections[i].class_id, cpu_batch.results[0].detections[i].class_id);
    EXPECT_NEAR(first.results[0].detections[i].confidence, cpu_batch.results[0].detections[i].confidence, 1e-5f);
    EXPECT_NEAR(first.results[0].detections[i].box.x1, cpu_batch.results[0].detections[i].box.x1, 1e-4f);
    EXPECT_NEAR(first.results[0].detections[i].box.y1, cpu_batch.results[0].detections[i].box.y1, 1e-4f);
    EXPECT_NEAR(first.results[0].detections[i].box.x2, cpu_batch.results[0].detections[i].box.x2, 1e-4f);
    EXPECT_NEAR(first.results[0].detections[i].box.y2, cpu_batch.results[0].detections[i].box.y2, 1e-4f);
    EXPECT_EQ(second.results[0].detections[i].class_id, first.results[0].detections[i].class_id);
  }
  vie::DecoderInput mismatch;
  mismatch.geometry = geometry;
  mismatch.outputs.emplace("predictions", host);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::YoloV8DetectionDecoder{}.decode_batch(mismatch, config, gpu));
  std::vector<float> bad = values;
  bad[0] = std::nanf("");
  vie::Tensor bad_host = vie::Tensor::from_host(cpu, vie::DataType::Fp32, {1, 4 + kClasses, kPredictions}, bad.data(),
                                                bad.size() * sizeof(float));
  input.outputs.clear();
  input.outputs.emplace("predictions", bad_host.to(gpu));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::YoloV8DetectionDecoder{}.decode_batch(input, config, gpu));
  vie::DecoderConfig dfl = config_for(1);
  dfl.box_decode = vie::BoxDecode::Dfl;
  dfl.box_channels = 8;
  dfl.reg_max = 2;
  dfl.num_classes = 1;
  dfl.levels = {vie::FeatureLevel{8, 1, 1}};
  std::vector<float> dfl_values(9, 0.f);
  dfl_values[1] = 20.f;
  dfl_values[3] = 20.f;
  dfl_values[5] = 20.f;
  dfl_values[7] = 20.f;
  dfl_values[8] = 0.95f;
  vie::Tensor dfl_host = vie::Tensor::from_host(cpu, vie::DataType::Fp32, {1, 9, 1}, dfl_values.data(),
                                                dfl_values.size() * sizeof(float));
  vie::DecoderInput dfl_input;
  dfl_input.geometry = identity(32, 32);
  dfl_input.outputs.emplace("predictions", dfl_host);
  const vie::DetectionBatch dfl_cpu = vie::YoloV8DetectionDecoder{}.decode_batch(dfl_input, dfl, cpu);
  dfl_input.outputs.clear();
  dfl_input.outputs.emplace("predictions", dfl_host.to(gpu));
  const vie::DetectionBatch dfl_gpu = vie::YoloV8DetectionDecoder{}.decode_batch(dfl_input, dfl, gpu);
  ASSERT_EQ(dfl_gpu.results[0].detections.size(), dfl_cpu.results[0].detections.size());
  ASSERT_EQ(dfl_cpu.results[0].detections.size(), 1u);
  EXPECT_NEAR(dfl_gpu.results[0].detections[0].box.x2, dfl_cpu.results[0].detections[0].box.x2, 1e-4f);

  cudaPointerAttributes attributes{};
  cuda_check(cudaPointerGetAttributes(&attributes, device.device_data()));
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
}

TEST(DecoderBenchmark, CudaStages) {
  constexpr int kPredictions = 8400;
  constexpr int kClasses = 80;
  constexpr int kRuns = 10;
  std::vector<float> values(static_cast<std::size_t>(4 + kClasses) * kPredictions, 0.f);
  for (int i = 0; i < 1000; ++i) {
    const int prediction = i * 7;
    values[static_cast<std::size_t>(prediction)] = static_cast<float>((i % 50) + 4);
    values[static_cast<std::size_t>(kPredictions + prediction)] = static_cast<float>((i % 40) + 4);
    values[static_cast<std::size_t>(2 * kPredictions + prediction)] = 10.f;
    values[static_cast<std::size_t>(3 * kPredictions + prediction)] = 8.f;
    values[static_cast<std::size_t>((4 + (i % kClasses)) * kPredictions + prediction)] = 0.85f;
  }
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  vie::Tensor device = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 4 + kClasses, kPredictions},
                                              values.data(), values.size() * sizeof(float))
                           .to(gpu);
  vie::DecoderConfig config = config_for(kClasses);
  config.pre_nms_top_k = 1000;
  config.max_detections = 300;
  vie::DecoderInput input;
  input.geometry = identity(640, 640);
  input.outputs.emplace("predictions", device);
  vie::YoloV8DetectionDecoder decoder;
  const vie::DetectionBatch warmup = decoder.decode_batch(input, config, gpu);
  std::vector<double> gpu_samples;
  std::vector<double> wall_samples;
  std::vector<double> copy_samples;
  for (int run = 0; run < kRuns; ++run) {
    const auto start = std::chrono::steady_clock::now();
    const vie::DetectionBatch batch = decoder.decode_batch(input, config, gpu);
    const auto stop = std::chrono::steady_clock::now();
    wall_samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    gpu_samples.push_back(batch.results[0].gpu_execute_ms);
    copy_samples.push_back(batch.results[0].d2h_ms);
  }
  auto summarize = [](std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    double sum = 0;
    for (double sample : samples) sum += sample;
    return std::tuple<double, double, double>{sum / static_cast<double>(samples.size()),
                                              samples[samples.size() / 2],
                                              samples[static_cast<std::size_t>(samples.size() * 0.95)]};
  };
  vie::DecoderConfig filter_only = config;
  filter_only.nms_enabled = false;
  std::vector<double> filter_samples;
  for (int run = 0; run < kRuns; ++run) {
    const auto start = std::chrono::steady_clock::now();
    const vie::DetectionBatch batch = decoder.decode_batch(input, filter_only, gpu);
    const auto stop = std::chrono::steady_clock::now();
    filter_samples.push_back(batch.results[0].gpu_execute_ms);
    (void)start;
    (void)stop;
  }
  const auto filtered = summarize(filter_samples);
  const auto wall = summarize(wall_samples);
  const auto gpu_time = summarize(gpu_samples);
  const auto copy = summarize(copy_samples);
  std::ofstream out("/tmp/vie_phase432_cuda.txt");
  out << "predictions=" << kPredictions << " classes=" << kClasses
      << " before_nms=" << warmup.results[0].candidates_before_nms
      << " kept=" << warmup.results[0].detections.size() << " runs=" << kRuns << "\n";
  out << "sync=cudaEventSynchronize around emit/sort/map/nms; D2H of kept boxes is separate\n";
  out << "filter_map_gpu_mean_ms=" << std::get<0>(filtered) << " p50=" << std::get<1>(filtered)
      << " p95=" << std::get<2>(filtered) << "\n";
  out << "gpu_mean_ms=" << std::get<0>(gpu_time) << " p50=" << std::get<1>(gpu_time)
      << " p95=" << std::get<2>(gpu_time) << "\n";
  out << "d2h_mean_ms=" << std::get<0>(copy) << " p50=" << std::get<1>(copy) << " p95=" << std::get<2>(copy) << "\n";
  out << "wall_mean_ms=" << std::get<0>(wall) << " p50=" << std::get<1>(wall) << " p95=" << std::get<2>(wall) << "\n";
  EXPECT_GT(warmup.results[0].detections.size(), 0u);
}
