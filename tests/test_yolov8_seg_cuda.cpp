#include "test_util.hpp"

#include "vie/decoder/yolov8_seg_decoder.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <tuple>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/backend/onnxruntime_backend.hpp"
#endif
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include "vie/backend/onnxruntime_cuda_backend.hpp"
#endif

namespace {

void put(std::vector<float>& data, int predictions, int channel, int prediction, float value) {
  data[static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) +
       static_cast<std::size_t>(prediction)] = value;
}

vie::DecoderConfig seg_config(int classes, int mask_channels) {
  vie::DecoderConfig config;
  config.output_name = "output0";
  config.prototype_name = "output1";
  config.channel_axis = 1;
  config.box_channels = 4;
  config.num_classes = classes;
  config.mask_channels = mask_channels;
  config.box_decode = vie::BoxDecode::DecodedCxcywh;
  config.scores_are_probabilities = true;
  config.class_selection = vie::ClassSelection::BestClass;
  config.confidence_threshold = 0.5f;
  config.nms_enabled = true;
  config.class_aware_nms = true;
  config.iou_threshold = 0.5f;
  config.max_detections = 32;
  config.pre_nms_top_k = 300;
  config.prototype_channel_axis = 1;
  config.retina_masks = false;
  config.binary_mask_gt_zero = true;
  config.max_mask_bytes = 64ull << 20;
  return config;
}

vie::DecoderInput make_input(const std::vector<float>& predictions, const std::vector<std::int64_t>& prediction_shape,
                            const std::vector<float>& proto, const std::vector<std::int64_t>& proto_shape,
                            const vie::GeometryTransform& geometry, const vie::DeviceContext& device) {
  vie::DecoderInput input;
  input.geometry = geometry;
  input.outputs.emplace("output0", vie::Tensor::from_host(device, vie::DataType::Fp32, prediction_shape,
                                                         predictions.data(), predictions.size() * sizeof(float)));
  input.outputs.emplace("output1", vie::Tensor::from_host(device, vie::DataType::Fp32, proto_shape, proto.data(),
                                                         proto.size() * sizeof(float)));
  return input;
}

void expect_same_batch(const vie::SegmentationBatch& cpu, const vie::SegmentationBatch& gpu) {
  ASSERT_EQ(cpu.instances.size(), gpu.instances.size());
  ASSERT_EQ(cpu.original_width, gpu.original_width);
  ASSERT_EQ(cpu.original_height, gpu.original_height);
  for (std::size_t i = 0; i < cpu.instances.size(); ++i) {
    EXPECT_EQ(cpu.instances[i].detection.class_id, gpu.instances[i].detection.class_id);
    EXPECT_FLOAT_EQ(cpu.instances[i].detection.confidence, gpu.instances[i].detection.confidence);
    EXPECT_FLOAT_EQ(cpu.instances[i].detection.box.x1, gpu.instances[i].detection.box.x1);
    EXPECT_FLOAT_EQ(cpu.instances[i].detection.box.y1, gpu.instances[i].detection.box.y1);
    EXPECT_FLOAT_EQ(cpu.instances[i].detection.box.x2, gpu.instances[i].detection.box.x2);
    EXPECT_FLOAT_EQ(cpu.instances[i].detection.box.y2, gpu.instances[i].detection.box.y2);
    const vie::Mask cpu_host = cpu.instances[i].mask.device() == vie::DeviceType::Cpu ? cpu.instances[i].mask
                                                                                     : cpu.instances[i].mask.to_cpu();
    const vie::Mask host = gpu.instances[i].mask.device() == vie::DeviceType::Cpu ? gpu.instances[i].mask
                                                                                  : gpu.instances[i].mask.to_cpu();
    ASSERT_EQ(host.width(), cpu_host.width());
    ASSERT_EQ(host.height(), cpu_host.height());
    const int pixels = host.width() * host.height();
    int mismatched = 0;
    int max_edge = 0;
    for (int p = 0; p < pixels; ++p) {
      EXPECT_TRUE(host.binary_data()[p] == 0 || host.binary_data()[p] == 1);
      if (host.binary_data()[p] != cpu_host.binary_data()[p]) {
        ++mismatched;
        const int x = p % host.width();
        const int y = p / host.width();
        const int edge = std::min(std::min(x, host.width() - 1 - x), std::min(y, host.height() - 1 - y));
        max_edge = std::max(max_edge, edge);
      }
    }
    EXPECT_EQ(mismatched, 0) << "instance " << i << " max interior distance " << max_edge;
  }
}

vie::SegmentationBatch decode_both(const std::vector<float>& predictions, const std::vector<std::int64_t>& prediction_shape,
                                  const std::vector<float>& proto, const std::vector<std::int64_t>& proto_shape,
                                  vie::DecoderConfig config, const vie::GeometryTransform& geometry) {
  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  const vie::DecoderInput host = make_input(predictions, prediction_shape, proto, proto_shape, geometry, cpu);
  const vie::DecoderInput device = make_input(predictions, prediction_shape, proto, proto_shape, geometry, gpu);
  const vie::SegmentationBatch cpu_batch = vie::YoloV8SegmentationDecoder{}.decode_segmentation(host, config, cpu);
  const vie::SegmentationBatch gpu_batch = vie::YoloV8SegmentationDecoder{}.decode_segmentation(device, config, gpu);
  EXPECT_EQ(gpu_batch.execution_device, vie::DeviceType::Cuda);
  expect_same_batch(cpu_batch, gpu_batch);
  return gpu_batch;
}

}  // namespace

TEST(YoloV8SegCuda, MatchesCpuOnReconstructionCropAndRetina) {
  constexpr int kPredictions = 3;
  constexpr int kChannels = 4 + 2 + 2;
  std::vector<float> predictions(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  put(predictions, kPredictions, 0, 0, 1.f);
  put(predictions, kPredictions, 1, 0, 1.f);
  put(predictions, kPredictions, 2, 0, 2.f);
  put(predictions, kPredictions, 3, 0, 2.f);
  put(predictions, kPredictions, 4, 0, 0.9f);
  put(predictions, kPredictions, 6, 0, 2.f);
  put(predictions, kPredictions, 7, 0, -0.5f);
  put(predictions, kPredictions, 0, 1, 1.f);
  put(predictions, kPredictions, 1, 1, 1.f);
  put(predictions, kPredictions, 2, 1, 2.f);
  put(predictions, kPredictions, 3, 1, 2.f);
  put(predictions, kPredictions, 5, 1, 0.7f);
  put(predictions, kPredictions, 6, 1, -4.f);
  put(predictions, kPredictions, 4, 2, 0.1f);
  const std::vector<float> proto = {1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, -4.f};
  vie::GeometryTransform geometry;
  geometry.original_width = 4;
  geometry.original_height = 3;
  geometry.output_width = 2;
  geometry.output_height = 2;
  geometry.resized_width = 2;
  geometry.resized_height = 2;
  geometry.pad_left = 0;
  geometry.pad_top = 0;
  geometry.scale_x = 0.5;
  geometry.scale_y = 2.0 / 3.0;
  geometry.uniform_scale = 0;
  vie::DecoderConfig config = seg_config(2, 2);
  const vie::SegmentationBatch low =
      decode_both(predictions, {1, kChannels, kPredictions}, proto, {1, 2, 2, 2}, config, geometry);
  ASSERT_EQ(low.instances.size(), 2u);
  config.retina_masks = true;
  const vie::SegmentationBatch high =
      decode_both(predictions, {1, kChannels, kPredictions}, proto, {1, 2, 2, 2}, config, geometry);
  ASSERT_EQ(high.instances.size(), 2u);
  EXPECT_EQ(high.instances[0].mask.width(), 4);
  EXPECT_EQ(high.instances[0].mask.height(), 3);
  EXPECT_EQ(low.instances[0].mask.width(), 2);
  EXPECT_EQ(low.instances[0].mask.height(), 2);
}

TEST(YoloV8SegCuda, CoversEmptyOddLetterboxStretchAndRepeat) {
  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  std::vector<float> quiet(7, 0.f);
  quiet[2] = 2.f;
  quiet[3] = 2.f;
  quiet[4] = 0.1f;
  const std::vector<float> proto = {1.f, -1.f, 0.5f, 2.f, -3.f, 0.25f};
  vie::PreprocessConfig odd_config;
  odd_config.resize_mode = vie::ResizeMode::Letterbox;
  odd_config.target_width = 5;
  odd_config.target_height = 5;
  odd_config.interpolation = vie::Interpolation::Bilinear;
  odd_config.padding_value = 114.f;
  const vie::GeometryTransform odd = vie::make_geometry(odd_config, 2, 3);
  EXPECT_EQ(odd.pad_left % 2, 1);
  EXPECT_NE(odd.scale_x, odd.scale_y);
  vie::DecoderConfig config = seg_config(2, 1);
  config.retina_masks = true;
  const vie::SegmentationBatch empty =
      decode_both(quiet, {1, 7, 1}, proto, {1, 1, 2, 3}, config, odd);
  EXPECT_TRUE(empty.instances.empty());

  std::vector<float> one(7, 0.f);
  one[0] = 2.5f;
  one[1] = 2.5f;
  one[2] = 5.f;
  one[3] = 5.f;
  one[4] = 0.95f;
  one[6] = -0.2f;
  const vie::SegmentationBatch first = decode_both(one, {1, 7, 1}, proto, {1, 1, 2, 3}, config, odd);
  const vie::SegmentationBatch second = decode_both(one, {1, 7, 1}, proto, {1, 1, 2, 3}, config, odd);
  ASSERT_EQ(first.instances.size(), 1u);
  expect_same_batch(first, second);

  vie::PreprocessConfig stretch_config;
  stretch_config.resize_mode = vie::ResizeMode::Stretch;
  stretch_config.target_width = 4;
  stretch_config.target_height = 1;
  stretch_config.interpolation = vie::Interpolation::Bilinear;
  const vie::GeometryTransform stretch = vie::make_geometry(stretch_config, 2, 1);
  EXPECT_NE(stretch.scale_x, stretch.scale_y);
  config.retina_masks = false;
  std::vector<float> stretch_pred(7, 0.f);
  stretch_pred[0] = 2.f;
  stretch_pred[1] = 0.5f;
  stretch_pred[2] = 4.f;
  stretch_pred[3] = 1.f;
  stretch_pred[4] = 0.9f;
  stretch_pred[6] = 1.f;
  const std::vector<float> stretch_proto = {0.f, 0.f, 10.f, 0.f};
  const vie::SegmentationBatch stretched =
      decode_both(stretch_pred, {1, 7, 1}, stretch_proto, {1, 1, 1, 4}, config, stretch);
  ASSERT_EQ(stretched.instances.size(), 1u);
  EXPECT_EQ(stretched.instances[0].mask.width(), 4);
  EXPECT_EQ(stretched.instances[0].mask.height(), 1);
}

TEST(YoloV8SegCuda, RejectsBadDeviceNanAndBudgetAndKeepsMaskAfterInputDies) {
  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  std::vector<float> predictions(7, 0.f);
  predictions[0] = 1.f;
  predictions[1] = 1.f;
  predictions[2] = 2.f;
  predictions[3] = 2.f;
  predictions[4] = 0.9f;
  predictions[6] = 1.f;
  const std::vector<float> proto = {1.f, 1.f, 1.f, 1.f};
  vie::GeometryTransform geometry;
  geometry.original_width = 2;
  geometry.original_height = 2;
  geometry.output_width = 2;
  geometry.output_height = 2;
  geometry.scale_x = 1;
  geometry.scale_y = 1;
  vie::DecoderConfig config = seg_config(2, 1);
  vie::DecoderInput mixed = make_input(predictions, {1, 7, 1}, proto, {1, 1, 2, 2}, geometry, cpu);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::YoloV8SegmentationDecoder{}.decode_segmentation(mixed, config, gpu));
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  EXPECT_EQ(device_count, 1);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::DeviceContext(vie::DeviceType::Cuda, 1));
  std::vector<float> bad_proto = proto;
  bad_proto[0] = std::numeric_limits<float>::quiet_NaN();
  vie::DecoderInput nan_input = make_input(predictions, {1, 7, 1}, bad_proto, {1, 1, 2, 2}, geometry, gpu);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::YoloV8SegmentationDecoder{}.decode_segmentation(nan_input, config, gpu));
  vie::DecoderConfig budget = config;
  budget.max_mask_bytes = 3;
  vie::DecoderInput device_input = make_input(predictions, {1, 7, 1}, proto, {1, 1, 2, 2}, geometry, gpu);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::YoloV8SegmentationDecoder{}.decode_segmentation(device_input, budget, gpu));

  vie::SegmentationBatch surviving;
  {
    vie::DecoderInput temporary = make_input(predictions, {1, 7, 1}, proto, {1, 1, 2, 2}, geometry, gpu);
    surviving = vie::YoloV8SegmentationDecoder{}.decode_segmentation(temporary, config, gpu);
  }
  ASSERT_EQ(surviving.instances.size(), 1u);
  EXPECT_EQ(surviving.instances[0].mask.device(), vie::DeviceType::Cuda);
  const vie::Mask host = surviving.instances[0].mask.to_cpu();
  EXPECT_EQ(host.binary_data()[0], 1);
  vie::YoloV8SegmentationDecoder decoder;
  const vie::SegmentationBatch again = decoder.decode_segmentation(device_input, config, gpu);
  expect_same_batch(again, surviving);
}

#if defined(VIE_HAS_ONNXRUNTIME)
TEST(YoloV8SegCuda, RealCpuEpOutputsDriveCudaDecoder) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  vie::PreprocessConfig prep;
  prep.resize_mode = vie::ResizeMode::Letterbox;
  prep.target_width = 640;
  prep.target_height = 640;
  prep.interpolation = vie::Interpolation::Bilinear;
  prep.input_color = vie::ColorOrder::Rgb;
  prep.output_color = vie::ColorOrder::Rgb;
  prep.normalize_enabled = true;
  prep.scale = 1.f / 255.f;
  prep.mean = {0.f, 0.f, 0.f};
  prep.std = {1.f, 1.f, 1.f};
  prep.padding_value = 114.f;
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(prep);
  std::ifstream image_file(vie_fixture("testdata/images/bus.ppm"), std::ios::binary);
  std::string magic;
  int image_width = 0;
  int image_height = 0;
  int max_value = 0;
  image_file >> magic >> image_width >> image_height >> max_value;
  image_file.get();
  ASSERT_EQ(magic, "P6");
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(image_width * image_height * 3));
  image_file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  ASSERT_TRUE(image_file);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = image_width;
  image.height = image_height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  const vie::PreprocessResult prepared = preprocessor.execute(image, cpu);
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
  const vie::Tensor& prediction = output.tensors.at("output0").tensor;
  const vie::Tensor& prototype = output.tensors.at("output1").tensor;
  const int mask_channels = static_cast<int>(prototype.shape()[1]);
  vie::DecoderConfig config = seg_config(static_cast<int>(prediction.shape()[1]) - 4 - mask_channels, mask_channels);
  config.retina_masks = true;
  config.confidence_threshold = 0.25f;
  config.iou_threshold = 0.45f;
  config.max_detections = 20;
  config.pre_nms_top_k = 300;
  vie::DecoderInput host_input;
  host_input.geometry = prepared.geometry;
  host_input.outputs.emplace("output0", prediction);
  host_input.outputs.emplace("output1", prototype);
  vie::DecoderInput device_input;
  device_input.geometry = prepared.geometry;
  device_input.outputs.emplace("output0", prediction.to(gpu));
  device_input.outputs.emplace("output1", prototype.to(gpu));
  const vie::SegmentationBatch cpu_batch =
      vie::YoloV8SegmentationDecoder{}.decode_segmentation(host_input, config, cpu);
  const vie::SegmentationBatch gpu_batch =
      vie::YoloV8SegmentationDecoder{}.decode_segmentation(device_input, config, gpu);
  expect_same_batch(cpu_batch, gpu_batch);
  std::size_t free_before = 0;
  std::size_t total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);
  vie::CudaSegmentationTiming timing;
  const vie::SegmentationBatch timed =
      vie::YoloV8SegmentationDecoder{}.decode_cuda_timed(device_input, config, gpu, &timing);
  std::size_t free_after = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_after, &total), cudaSuccess);
  EXPECT_EQ(timed.instances.size(), cpu_batch.instances.size());
  EXPECT_FALSE(timing.masks_copied_to_host);
  std::ofstream out("/tmp/vie_phase433_cuda_real.txt");
  out << "label=cuda_decoder_on_cpu_ep_tensors\n";
  out << "output0=" << prediction.shape()[1] << "x" << prediction.shape()[2]
      << " output1=" << prototype.shape()[1] << "x" << prototype.shape()[2] << "x" << prototype.shape()[3] << "\n";
  out << "instances=" << cpu_batch.instances.size() << " mask=" << image_width << "x" << image_height
      << " exact_cpu_match=1\n";
  out << "gpu_ms=" << timing.gpu_ms << " wall_ms=" << timing.wall_ms << " d2h_ms=" << timing.d2h_ms
      << " mask_alloc_ms=" << timing.mask_alloc_ms << "\n";
  out << "free_before=" << free_before << " free_after=" << free_after
      << " retained_bytes=" << (free_before > free_after ? free_before - free_after : 0) << "\n";

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
  vie::OnnxRuntimeCudaBackend cuda_backend;
  vie::BackendConfig strict = backend_config;
  strict.device = vie::DeviceType::Cuda;
  strict.device_id = 0;
  bool strict_ok = false;
  std::string strict_error;
  try {
    cuda_backend.prepare(strict);
    strict_ok = true;
  } catch (const vie::VieError& error) {
    strict_error = error.what();
  }
  out << "strict_cuda_ep=" << (strict_ok ? "created" : "failed") << "\n";
  if (!strict_ok) out << "strict_error=" << strict_error << "\n";
  if (strict_ok) {
    void* device_pixels = nullptr;
    ASSERT_EQ(cudaMalloc(&device_pixels, pixels.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
    vie::ImageInput gpu_source = image;
    gpu_source.data = device_pixels;
    gpu_source.device = vie::DeviceType::Cuda;
    gpu_source.device_id = 0;
    try {
      const vie::PreprocessResult gpu_image = preprocessor.execute(gpu_source, gpu);
      vie::BackendRequest gpu_request;
      gpu_request.inputs.emplace("images", gpu_image.output);
      const vie::BackendOutput gpu_output = cuda_backend.execute(gpu_request);
      vie::DecoderInput e2e;
      e2e.geometry = gpu_image.geometry;
      e2e.outputs.emplace("output0", gpu_output.tensors.at("output0").tensor);
      e2e.outputs.emplace("output1", gpu_output.tensors.at("output1").tensor);
      EXPECT_TRUE(e2e.outputs.at("output0").is_cuda());
      EXPECT_TRUE(e2e.outputs.at("output1").is_cuda());
      const vie::SegmentationBatch e2e_batch =
          vie::YoloV8SegmentationDecoder{}.decode_segmentation(e2e, config, gpu);
      EXPECT_EQ(e2e_batch.execution_device, vie::DeviceType::Cuda);
      for (const vie::InstanceSegmentation& instance : e2e_batch.instances) {
        EXPECT_EQ(instance.mask.device(), vie::DeviceType::Cuda);
        const vie::Mask host = instance.mask.to_cpu();
        int illegal = 0;
        for (int p = 0; p < host.width() * host.height(); ++p) {
          if (host.binary_data()[p] > 1) ++illegal;
        }
        EXPECT_EQ(illegal, 0);
      }
      out << "cuda_end_to_end=1 instances=" << e2e_batch.instances.size()
          << " cpu_ep_instances=" << cpu_batch.instances.size() << " fallback=0\n";
    } catch (const vie::VieError& error) {
      out << "cuda_end_to_end=0 run_error=" << error.what() << "\n";
    }
    cudaFree(device_pixels);
  } else {
    out << "cuda_end_to_end=0\n";
  }
#else
  out << "strict_cuda_ep=not_built\n";
  out << "cuda_end_to_end=0\n";
#endif
}
#endif

TEST(YoloV8SegCudaBenchmark, CpuAndCudaSameInputs) {
  constexpr int kPredictions = 8400;
  constexpr int kClasses = 80;
  constexpr int kRuns = 100;
  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  struct Case {
    const char* name;
    int instances;
    int channels;
    int proto;
    bool retina;
    int mask_w;
    int mask_h;
  };
  const Case cases[] = {
      {"inst1_c32_proto160_retina_false_80", 1, 32, 160, false, 80, 80},
      {"inst5_c32_proto160_retina_false_160", 5, 32, 160, false, 160, 160},
      {"inst10_c8_proto40_retina_true_120x80", 10, 8, 40, true, 120, 80},
      {"inst30_c32_proto80_retina_true_160x120", 30, 32, 80, true, 160, 120},
  };
  std::ofstream out("/tmp/vie_phase433_cuda_bench.txt");
  out << "runs=" << kRuns << " includes_allocation=1 includes_sync=1 masks_not_copied=1\n";
  for (const Case& item : cases) {
    const int channels = 4 + kClasses + item.channels;
    std::vector<float> predictions(static_cast<std::size_t>(channels) * kPredictions, 0.f);
    const int space = item.retina ? 80 : item.mask_w;
    const int columns = std::max(1, space / 8);
    for (int i = 0; i < item.instances; ++i) {
      const int prediction = i * 30;
      put(predictions, kPredictions, 0, prediction, static_cast<float>(4 + (i % columns) * 8));
      put(predictions, kPredictions, 1, prediction, static_cast<float>(4 + (i / columns) * 8));
      put(predictions, kPredictions, 2, prediction, 4.f);
      put(predictions, kPredictions, 3, prediction, 4.f);
      put(predictions, kPredictions, 4 + (i % kClasses), prediction, 0.91f);
      for (int channel = 0; channel < item.channels; ++channel) {
        put(predictions, kPredictions, 4 + kClasses + channel, prediction, channel % 2 == 0 ? 0.4f : -0.2f);
      }
    }
    std::vector<float> proto(static_cast<std::size_t>(item.channels * item.proto * item.proto), 0.15f);
    vie::DecoderConfig config = seg_config(kClasses, item.channels);
    config.retina_masks = item.retina;
    config.max_detections = item.instances;
    config.pre_nms_top_k = 300;
    vie::GeometryTransform geometry;
    if (item.retina) {
      geometry.original_width = item.mask_w;
      geometry.original_height = item.mask_h;
      geometry.output_width = 80;
      geometry.output_height = 80;
      geometry.scale_x = 80.0 / item.mask_w;
      geometry.scale_y = 80.0 / item.mask_h;
      geometry.pad_left = 1;
      geometry.pad_top = 1;
    } else {
      geometry.original_width = item.mask_w;
      geometry.original_height = item.mask_h;
      geometry.output_width = item.mask_w;
      geometry.output_height = item.mask_h;
      geometry.scale_x = 1;
      geometry.scale_y = 1;
    }
    vie::DecoderInput host = make_input(predictions, {1, channels, kPredictions}, proto,
                                       {1, item.channels, item.proto, item.proto}, geometry, cpu);
    vie::DecoderInput device = make_input(predictions, {1, channels, kPredictions}, proto,
                                         {1, item.channels, item.proto, item.proto}, geometry, gpu);
    vie::YoloV8SegmentationDecoder decoder;
    vie::SegmentationTiming cpu_once;
    vie::CudaSegmentationTiming gpu_once;
    const vie::SegmentationBatch cpu_batch = decoder.decode_timed(host, config, cpu, &cpu_once);
    const vie::SegmentationBatch gpu_batch = decoder.decode_cuda_timed(device, config, gpu, &gpu_once);
    ASSERT_EQ(cpu_batch.instances.size(), gpu_batch.instances.size());
    ASSERT_EQ(gpu_once.instances, item.instances);
    expect_same_batch(cpu_batch, gpu_batch);
    std::vector<vie::SegmentationTiming> cpu_samples;
    std::vector<vie::CudaSegmentationTiming> gpu_samples;
    for (int run = 0; run < kRuns; ++run) {
      vie::SegmentationTiming cpu_sample;
      vie::CudaSegmentationTiming gpu_sample;
      decoder.decode_timed(host, config, cpu, &cpu_sample);
      decoder.decode_cuda_timed(device, config, gpu, &gpu_sample);
      cpu_samples.push_back(cpu_sample);
      gpu_samples.push_back(gpu_sample);
    }
    auto summarize = [](auto rows, auto field) {
      std::vector<double> values;
      for (const auto& row : rows) values.push_back(row.*field);
      std::sort(values.begin(), values.end());
      double sum = 0;
      for (double value : values) sum += value;
      const std::size_t p95 = static_cast<std::size_t>(values.size() * 0.95);
      return std::tuple<double, double, double>{sum / static_cast<double>(values.size()), values[values.size() / 2],
                                                values[std::min(p95, values.size() - 1)]};
    };
    const auto cpu_full = summarize(cpu_samples, &vie::SegmentationTiming::total_ms);
    const auto cpu_detect = summarize(cpu_samples, &vie::SegmentationTiming::detection_ms);
    const auto cpu_filter = summarize(cpu_samples, &vie::SegmentationTiming::filter_ms);
    const auto cpu_nms = summarize(cpu_samples, &vie::SegmentationTiming::nms_ms);
    const auto cpu_gather = summarize(cpu_samples, &vie::SegmentationTiming::gather_ms);
    const auto cpu_recon = summarize(cpu_samples, &vie::SegmentationTiming::reconstruct_ms);
    const auto cpu_crop = summarize(cpu_samples, &vie::SegmentationTiming::crop_ms);
    const auto cpu_resize = summarize(cpu_samples, &vie::SegmentationTiming::resize_ms);
    const auto gpu_full = summarize(gpu_samples, &vie::CudaSegmentationTiming::gpu_ms);
    const auto gpu_wall = summarize(gpu_samples, &vie::CudaSegmentationTiming::wall_ms);
    const auto gpu_detect = summarize(gpu_samples, &vie::CudaSegmentationTiming::detection_ms);
    const auto gpu_sort = summarize(gpu_samples, &vie::CudaSegmentationTiming::sort_ms);
    const auto gpu_nms = summarize(gpu_samples, &vie::CudaSegmentationTiming::nms_ms);
    const auto gpu_gather = summarize(gpu_samples, &vie::CudaSegmentationTiming::gather_ms);
    const auto gpu_recon = summarize(gpu_samples, &vie::CudaSegmentationTiming::reconstruct_ms);
    const auto gpu_crop = summarize(gpu_samples, &vie::CudaSegmentationTiming::crop_ms);
    const auto gpu_resize = summarize(gpu_samples, &vie::CudaSegmentationTiming::resize_ms);
    const auto gpu_alloc = summarize(gpu_samples, &vie::CudaSegmentationTiming::mask_alloc_ms);
    const auto gpu_d2h = summarize(gpu_samples, &vie::CudaSegmentationTiming::d2h_ms);
    auto write = [&](const char* label, const std::tuple<double, double, double>& value) {
      out << label << "_mean=" << std::get<0>(value) << " p50=" << std::get<1>(value) << " p95=" << std::get<2>(value)
          << "\n";
    };
    out << "case=" << item.name << " instances=" << gpu_once.instances << " proto=" << item.channels << "x" << item.proto
        << "x" << item.proto << " mask=" << gpu_once.mask_width << "x" << gpu_once.mask_height << "\n";
    write("cpu_detection", cpu_detect);
    write("cpu_filter", cpu_filter);
    write("cpu_nms", cpu_nms);
    write("cpu_gather", cpu_gather);
    write("cpu_reconstruct", cpu_recon);
    write("cpu_crop", cpu_crop);
    write("cpu_resize", cpu_resize);
    write("cpu_full", cpu_full);
    write("cuda_detection_includes_sort_nms", gpu_detect);
    write("cuda_sort", gpu_sort);
    write("cuda_nms", gpu_nms);
    write("cuda_gather", gpu_gather);
    write("cuda_reconstruct", gpu_recon);
    write("cuda_crop", gpu_crop);
    write("cuda_resize", gpu_resize);
    write("cuda_gpu", gpu_full);
    write("cuda_wall", gpu_wall);
    write("cuda_mask_alloc", gpu_alloc);
    write("cuda_d2h_candidates_only", gpu_d2h);
  }
}
