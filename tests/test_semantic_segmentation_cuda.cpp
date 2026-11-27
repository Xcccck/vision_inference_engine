// Phase 13: Semantic Segmentation CUDA ArgMax decoder. Only compiled when
// VIE_ENABLE_CUDA is on; the CPU-only builds never see these sources.

#include "test_util.hpp"

#include "vie/decoder/semantic_segmentation.hpp"
#include "vie/decoder/semantic_segmentation_cuda_decoder.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using vie::DataType;
using vie::DeviceContext;
using vie::DeviceType;
using vie::GeometryTransform;
using vie::SemanticSegmentationBatch;
using vie::SemanticSegmentationDecodeRequest;
using vie::SemanticSegmentationResult;

const DeviceContext kCpu;
const DeviceContext kCuda(DeviceType::Cuda, 0);

SemanticSegmentationDecodeRequest request_for(int classes) {
  SemanticSegmentationDecodeRequest request;
  request.output_name = "out";
  request.num_classes = classes;
  request.channel_axis = 1;
  request.logits_semantics = "per_pixel_class_logits";
  request.output_interpolation = "bilinear";
  request.class_id_interpolation = "nearest";
  return request;
}

vie::Tensor host_logits(const std::vector<float>& data, int classes, int height, int width) {
  return vie::Tensor::from_host(kCpu, DataType::Fp32,
                                {1, classes, height, width}, data.data(),
                                data.size() * sizeof(float));
}

GeometryTransform identity_geometry(int width, int height) {
  GeometryTransform geometry;
  geometry.original_width = width;
  geometry.original_height = height;
  geometry.resized_width = width;
  geometry.resized_height = height;
  geometry.output_width = width;
  geometry.output_height = height;
  geometry.scale_x = 1.0;
  geometry.scale_y = 1.0;
  geometry.uniform_scale = 1.0;
  return geometry;
}

// Letterbox: the model grid is larger than the scaled image and the content is
// shifted by pad_left/pad_top.
GeometryTransform letterbox_geometry(int original_width, int original_height, int grid_width,
                                     int grid_height, int pad_left, int pad_top, double scale) {
  GeometryTransform geometry;
  geometry.original_width = original_width;
  geometry.original_height = original_height;
  geometry.resized_width = static_cast<int>(std::lround(original_width * scale));
  geometry.resized_height = static_cast<int>(std::lround(original_height * scale));
  geometry.output_width = grid_width;
  geometry.output_height = grid_height;
  geometry.scale_x = scale;
  geometry.scale_y = scale;
  geometry.uniform_scale = scale;
  geometry.pad_left = pad_left;
  geometry.pad_top = pad_top;
  geometry.pad_right = grid_width - geometry.resized_width - pad_left;
  geometry.pad_bottom = grid_height - geometry.resized_height - pad_top;
  return geometry;
}

std::vector<float> pattern_logits(int classes, int height, int width) {
  std::vector<float> data(static_cast<std::size_t>(classes) * height * width);
  for (int c = 0; c < classes; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::size_t index = (static_cast<std::size_t>(c) * height + y) * width + x;
        data[index] = static_cast<float>(
            std::sin(static_cast<double>(c) * 1.7 + static_cast<double>(y) * 0.23 +
                     static_cast<double>(x) * 0.11) *
            3.25);
      }
    }
  }
  return data;
}

SemanticSegmentationBatch decode_cpu(const std::vector<float>& data, int classes, int height,
                                     int width, const GeometryTransform& geometry) {
  vie::DecoderInput input;
  input.outputs.emplace("out", host_logits(data, classes, height, width));
  input.geometry = geometry;
  return vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
      input, request_for(classes), kCpu);
}

SemanticSegmentationBatch decode_cuda(const std::vector<float>& data, int classes, int height,
                                      int width, const GeometryTransform& geometry) {
  vie::DecoderInput input;
  input.outputs.emplace("out", host_logits(data, classes, height, width).to(kCuda));
  input.geometry = geometry;
  return vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
      input, request_for(classes), kCuda);
}

void expect_same(const SemanticSegmentationBatch& cpu, const SemanticSegmentationBatch& cuda,
                 int classes, int height, int width, const GeometryTransform& geometry) {
  ASSERT_EQ(cpu.results.size(), 1u);
  ASSERT_EQ(cuda.results.size(), 1u);
  const SemanticSegmentationResult& expected = cpu.results.front();
  const SemanticSegmentationResult& actual = cuda.results.front();
  EXPECT_EQ(expected.width, actual.width);
  EXPECT_EQ(expected.height, actual.height);
  EXPECT_EQ(expected.num_classes, actual.num_classes);
  EXPECT_EQ(expected.logits_width, actual.logits_width);
  EXPECT_EQ(expected.logits_height, actual.logits_height);
  EXPECT_EQ(expected.logits_width, width);
  EXPECT_EQ(expected.logits_height, height);
  ASSERT_EQ(actual.class_ids.size(),
            static_cast<std::size_t>(geometry.original_width) *
                static_cast<std::size_t>(geometry.original_height));
  EXPECT_EQ(expected.class_ids, actual.class_ids);
  EXPECT_EQ(actual.execution_device, DeviceType::Cuda);
  EXPECT_EQ(actual.output_bytes, static_cast<std::size_t>(classes) * height * width * sizeof(float));
  EXPECT_EQ(actual.d2h_bytes, static_cast<std::size_t>(geometry.original_width) *
                                  static_cast<std::size_t>(geometry.original_height) * sizeof(std::int32_t));
  int mismatches = 0;
  for (std::size_t index = 0; index < actual.class_ids.size(); ++index) {
    if (actual.class_ids[index] != expected.class_ids[index]) ++mismatches;
    EXPECT_GE(actual.class_ids[index], 0);
    EXPECT_LT(actual.class_ids[index], classes);
  }
  EXPECT_EQ(mismatches, 0);
}

std::string decode_error(const std::vector<float>& data, int classes, int height, int width,
                         const GeometryTransform& geometry, bool cuda,
                         SemanticSegmentationDecodeRequest request) {
  try {
    vie::DecoderInput input;
    if (cuda) {
      input.outputs.emplace("out", host_logits(data, classes, height, width).to(kCuda));
    } else {
      input.outputs.emplace("out", host_logits(data, classes, height, width));
    }
    input.geometry = geometry;
    if (cuda) {
      vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(input, request, kCuda);
    } else {
      vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(input, request, kCpu);
    }
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::InvalidArgument) << error.what();
    return error.what();
  }
  ADD_FAILURE() << "expected VieError";
  return {};
}

}  // namespace

TEST(SemanticCudaDecoder, MatchesCpuOnIdentityGrids) {
  struct Case {
    int classes;
    int height;
    int width;
  };
  const Case cases[] = {{2, 1, 1}, {2, 3, 5}, {3, 4, 5}, {3, 5, 4}, {7, 9, 6}, {21, 16, 13}};
  for (const Case& item : cases) {
    const std::vector<float> data = pattern_logits(item.classes, item.height, item.width);
    const GeometryTransform geometry = identity_geometry(item.width, item.height);
    const SemanticSegmentationBatch cpu =
        decode_cpu(data, item.classes, item.height, item.width, geometry);
    const SemanticSegmentationBatch cuda =
        decode_cuda(data, item.classes, item.height, item.width, geometry);
    expect_same(cpu, cuda, item.classes, item.height, item.width, geometry);
  }
}

TEST(SemanticCudaDecoder, MatchesCpuWithLetterboxOriginAndStretchScale) {
  constexpr int kClasses = 5;
  constexpr int kHeight = 12;
  constexpr int kWidth = 10;
  const GeometryTransform geometry = letterbox_geometry(9, 7, 10, 12, 2, 3, 0.75);
  const std::vector<float> data = pattern_logits(kClasses, kHeight, kWidth);
  const SemanticSegmentationBatch cpu = decode_cpu(data, kClasses, kHeight, kWidth, geometry);
  const SemanticSegmentationBatch cuda = decode_cuda(data, kClasses, kHeight, kWidth, geometry);
  expect_same(cpu, cuda, kClasses, kHeight, kWidth, geometry);
  ASSERT_EQ(cpu.results[0].class_ids.size(), 63u);
}

TEST(SemanticCudaDecoder, MatchesCpuWhenLogitsNeedBilinearResize) {
  constexpr int kClasses = 4;
  constexpr int kLogitsHeight = 3;
  constexpr int kLogitsWidth = 4;
  constexpr int kGridHeight = 11;
  constexpr int kGridWidth = 13;
  const GeometryTransform geometry = identity_geometry(kGridWidth, kGridHeight);
  const std::vector<float> data = pattern_logits(kClasses, kLogitsHeight, kLogitsWidth);
  const SemanticSegmentationBatch cpu =
      decode_cpu(data, kClasses, kLogitsHeight, kLogitsWidth, geometry);
  const SemanticSegmentationBatch cuda =
      decode_cuda(data, kClasses, kLogitsHeight, kLogitsWidth, geometry);
  expect_same(cpu, cuda, kClasses, kLogitsHeight, kLogitsWidth, geometry);
  EXPECT_EQ(cuda.results[0].class_ids.size(), static_cast<std::size_t>(kGridWidth) * kGridHeight);
}

TEST(SemanticCudaDecoder, TiesKeepTheSmallestClassId) {
  constexpr int kClasses = 6;
  constexpr int kHeight = 5;
  constexpr int kWidth = 4;
  const std::vector<float> zeros(static_cast<std::size_t>(kClasses) * kHeight * kWidth, 2.f);
  const GeometryTransform geometry = identity_geometry(kWidth, kHeight);
  const SemanticSegmentationBatch cpu = decode_cpu(zeros, kClasses, kHeight, kWidth, geometry);
  const SemanticSegmentationBatch cuda = decode_cuda(zeros, kClasses, kHeight, kWidth, geometry);
  expect_same(cpu, cuda, kClasses, kHeight, kWidth, geometry);
  for (const std::int32_t id : cuda.results[0].class_ids) {
    EXPECT_EQ(id, 0);
  }

  // All channels carry the same value: every pixel is an exact tie and both
  // decoders must keep class 0.
  std::vector<float> mirrored(static_cast<std::size_t>(kClasses) * kHeight * kWidth);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const float value = static_cast<float>(std::sin(static_cast<double>(y) * 0.31 +
                                                      static_cast<double>(x) * 0.17));
      for (int c = 0; c < kClasses; ++c) {
        mirrored[(static_cast<std::size_t>(c) * kHeight + y) * kWidth + x] = value;
      }
    }
  }
  const SemanticSegmentationBatch mirrored_cpu =
      decode_cpu(mirrored, kClasses, kHeight, kWidth, geometry);
  const SemanticSegmentationBatch mirrored_cuda =
      decode_cuda(mirrored, kClasses, kHeight, kWidth, geometry);
  expect_same(mirrored_cpu, mirrored_cuda, kClasses, kHeight, kWidth, geometry);
  for (const std::int32_t id : mirrored_cuda.results[0].class_ids) {
    EXPECT_EQ(id, 0);
  }
}

TEST(SemanticCudaDecoder, RejectsNonFiniteLogitsWithTheSameCoordinates) {
  constexpr int kClasses = 3;
  constexpr int kHeight = 4;
  constexpr int kWidth = 5;
  const GeometryTransform geometry = identity_geometry(kWidth, kHeight);
  struct Case {
    int c;
    int y;
    int x;
    float value;
    const char* reason;
  };
  const Case cases[] = {
      {1, 2, 3, std::numeric_limits<float>::quiet_NaN(), "NaN"},
      {2, 0, 4, std::numeric_limits<float>::infinity(), "infinite"},
      {0, 3, 1, -std::numeric_limits<float>::infinity(), "infinite"},
  };
  for (const Case& item : cases) {
    std::vector<float> data = pattern_logits(kClasses, kHeight, kWidth);
    data[(static_cast<std::size_t>(item.c) * kHeight + item.y) * kWidth + item.x] = item.value;
    const std::string cpu = decode_error(data, kClasses, kHeight, kWidth, geometry, false,
                                         request_for(kClasses));
    const std::string cuda = decode_error(data, kClasses, kHeight, kWidth, geometry, true,
                                          request_for(kClasses));
    EXPECT_EQ(cpu, cuda);
    const std::string coordinates =
        "[c=" + std::to_string(item.c) + ", y=" + std::to_string(item.y) + ", x=" +
        std::to_string(item.x) + "]";
    EXPECT_NE(cpu.find(coordinates), std::string::npos) << cpu;
    EXPECT_NE(cpu.find(item.reason), std::string::npos) << cpu;
  }
}

TEST(SemanticCudaDecoder, RejectsInvalidTensorAndRequestContracts) {
  constexpr int kClasses = 3;
  constexpr int kHeight = 4;
  constexpr int kWidth = 5;
  const GeometryTransform geometry = identity_geometry(kWidth, kHeight);
  const std::vector<float> data = pattern_logits(kClasses, kHeight, kWidth);
  const SemanticSegmentationDecodeRequest request = request_for(kClasses);

  vie::DecoderInput cpu_tensor;
  cpu_tensor.outputs.emplace("out", host_logits(data, kClasses, kHeight, kWidth));
  cpu_tensor.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       cpu_tensor, request, kCuda));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
                       cpu_tensor, request, DeviceContext(DeviceType::Cuda, 0)));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       cpu_tensor, request, kCpu));

  vie::DecoderInput cuda_tensor;
  cuda_tensor.outputs.emplace("out", host_logits(data, kClasses, kHeight, kWidth).to(kCuda));
  cuda_tensor.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       cuda_tensor, request, DeviceContext(DeviceType::Cuda, 1)));

  vie::DecoderInput int_tensor;
  int_tensor.outputs.emplace("out", vie::Tensor::zeros(kCuda, DataType::Int32, {1, kClasses, kHeight, kWidth}));
  int_tensor.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       int_tensor, request, kCuda));

  vie::DecoderInput rank3;
  rank3.outputs.emplace("out", vie::Tensor::zeros(kCuda, DataType::Fp32, {kClasses, kHeight, kWidth}));
  rank3.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       rank3, request, kCuda));

  vie::DecoderInput batch2;
  batch2.outputs.emplace("out", vie::Tensor::zeros(kCuda, DataType::Fp32, {2, kClasses, kHeight, kWidth}));
  batch2.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       batch2, request, kCuda));

  vie::DecoderInput wrong_classes;
  wrong_classes.outputs.emplace("out",
                                vie::Tensor::zeros(kCuda, DataType::Fp32, {1, kClasses + 1, kHeight, kWidth}));
  wrong_classes.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       wrong_classes, request, kCuda));

  // Tensor::zeros already rejects a zero-sized dimension, so the decoder can
  // never observe an empty spatial extent; assert that guard where it lives.
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Tensor::zeros(kCuda, DataType::Fp32, {1, kClasses, 0, kWidth}));

  vie::Tensor wide = vie::Tensor::zeros(kCuda, DataType::Fp32, {1, kClasses, kHeight, 2 * kWidth});
  vie::Tensor strided = wide.view({1, kClasses, kHeight, kWidth}, {kClasses * kHeight * 2 * kWidth,
                                                                   kHeight * 2 * kWidth, 2 * kWidth, 2},
                                  0);
  ASSERT_FALSE(strided.contiguous());
  vie::DecoderInput strided_input;
  strided_input.outputs.emplace("out", strided);
  strided_input.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       strided_input, request, kCuda));

  vie::DecoderInput no_geometry;
  no_geometry.outputs.emplace("out", host_logits(data, kClasses, kHeight, kWidth).to(kCuda));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       no_geometry, request, kCuda));

  SemanticSegmentationDecodeRequest one_class = request_for(1);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       cuda_tensor, one_class, kCuda));
  SemanticSegmentationDecodeRequest bad_axis = request_for(kClasses);
  bad_axis.channel_axis = 0;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       cuda_tensor, bad_axis, kCuda));
  SemanticSegmentationDecodeRequest bad_semantics = request_for(kClasses);
  bad_semantics.logits_semantics = "probabilities";
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(
                       cuda_tensor, bad_semantics, kCuda));
}

TEST(SemanticCudaDecoder, DispatcherRoutesByDevice) {
  constexpr int kClasses = 4;
  constexpr int kHeight = 6;
  constexpr int kWidth = 7;
  const GeometryTransform geometry = identity_geometry(kWidth, kHeight);
  const std::vector<float> data = pattern_logits(kClasses, kHeight, kWidth);
  vie::SemanticSegmentationDispatchDecoder dispatcher;
  EXPECT_EQ(dispatcher.protocol(), "deeplabv3_semantic_seg_raw");

  vie::DecoderInput cpu_input;
  cpu_input.outputs.emplace("out", host_logits(data, kClasses, kHeight, kWidth));
  cpu_input.geometry = geometry;
  const SemanticSegmentationBatch cpu =
      dispatcher.decode_semantic_segmentation(cpu_input, request_for(kClasses), kCpu);
  EXPECT_EQ(cpu.results[0].execution_device, DeviceType::Cpu);
  EXPECT_EQ(cpu.results[0].d2h_bytes, 0u);

  vie::DecoderInput cuda_input;
  cuda_input.outputs.emplace("out", host_logits(data, kClasses, kHeight, kWidth).to(kCuda));
  cuda_input.geometry = geometry;
  const SemanticSegmentationBatch cuda =
      dispatcher.decode_semantic_segmentation(cuda_input, request_for(kClasses), kCuda);
  EXPECT_EQ(cuda.results[0].execution_device, DeviceType::Cuda);
  expect_same(cpu, cuda, kClasses, kHeight, kWidth, geometry);
}

TEST(SemanticCudaDecoder, RecordsDecoderTimingAndPayload) {
  constexpr int kClasses = 21;
  constexpr int kHeight = 65;
  constexpr int kWidth = 65;
  constexpr int kOriginalWidth = 160;
  constexpr int kOriginalHeight = 120;
  const std::vector<float> data = pattern_logits(kClasses, kHeight, kWidth);
  const GeometryTransform geometry = identity_geometry(kOriginalWidth, kOriginalHeight);

  // Identity for the map but the logits grid differs from the original size,
  // so the resize path is measured too.
  GeometryTransform resized = geometry;
  resized.output_width = kWidth;
  resized.output_height = kHeight;
  resized.scale_x = static_cast<double>(kWidth) / kOriginalWidth;
  resized.scale_y = static_cast<double>(kHeight) / kOriginalHeight;

  vie::DecoderInput cuda_input;
  cuda_input.outputs.emplace("out", host_logits(data, kClasses, kHeight, kWidth).to(kCuda));
  cuda_input.geometry = resized;
  vie::DecoderInput cpu_input;
  cpu_input.outputs.emplace("out", host_logits(data, kClasses, kHeight, kWidth));
  cpu_input.geometry = resized;

  constexpr int kWarmup = 2;
  constexpr int kSamples = 5;
  for (int index = 0; index < kWarmup; ++index) {
    vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(cuda_input, request_for(kClasses),
                                                                       kCuda);
    vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(cpu_input, request_for(kClasses),
                                                                   kCpu);
  }
  double cpu_validate = 0;
  double cpu_argmax = 0;
  double cuda_validate = 0;
  double cuda_argmax = 0;
  double cuda_d2h = 0;
  std::size_t d2h_bytes = 0;
  for (int index = 0; index < kSamples; ++index) {
    const SemanticSegmentationBatch cpu = vie::SemanticSegmentationDecoder{}.decode_semantic_segmentation(
        cpu_input, request_for(kClasses), kCpu);
    const SemanticSegmentationBatch cuda =
        vie::SemanticSegmentationCudaDecoder{}.decode_semantic_segmentation(cuda_input,
                                                                           request_for(kClasses), kCuda);
    ASSERT_EQ(cpu.results[0].class_ids, cuda.results[0].class_ids);
    cpu_validate += cpu.results[0].validate_ms;
    cpu_argmax += cpu.results[0].argmax_ms;
    cuda_validate += cuda.results[0].validate_ms;
    cuda_argmax += cuda.results[0].argmax_ms;
    cuda_d2h += cuda.results[0].d2h_ms;
    d2h_bytes = cuda.results[0].d2h_bytes;
  }
  EXPECT_EQ(d2h_bytes, static_cast<std::size_t>(kOriginalWidth) * kOriginalHeight * 4);

  std::ofstream record("/tmp/vie_phase13_semantic_decoder.txt");
  record << "classes " << kClasses << " logits " << kWidth << "x" << kHeight << " original "
         << kOriginalWidth << "x" << kOriginalHeight << " warmup " << kWarmup << " samples "
         << kSamples << "\n";
  record << "cpu validate_ms_mean " << cpu_validate / kSamples << " argmax_ms_mean "
         << cpu_argmax / kSamples << " d2h_bytes 0\n";
  record << "cuda validate_ms_mean " << cuda_validate / kSamples << " argmax_ms_mean "
         << cuda_argmax / kSamples << " d2h_ms_mean " << cuda_d2h / kSamples << " d2h_bytes "
         << d2h_bytes << " full_logits_bytes "
         << static_cast<std::size_t>(kClasses) * kHeight * kWidth * sizeof(float) << "\n";
}