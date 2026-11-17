#include "test_util.hpp"

#include "vie/decoder/decoder_registry.hpp"
#include "vie/decoder/mock_decoder.hpp"
#include "vie/decoder/yolov8_detect_decoder.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#if defined(VIE_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

class SegmentationOnlyDecoder : public vie::IModelDecoder {
 public:
  vie::RawPredictions decode(const std::map<std::string, vie::Tensor>&,
                             const vie::DecoderContext&) const override {
    return {};
  }

  vie::SegmentationBatch decode_segmentation(const vie::DecoderInput&, const vie::DecoderConfig&,
                                             const vie::DeviceContext&) const override {
    vie::SegmentationBatch batch;
    batch.original_width = 3;
    batch.original_height = 2;
    batch.execution_device = vie::DeviceType::Cpu;
    batch.protocol = "test_seg_only";
    return batch;
  }
};

}  // namespace

TEST(MaskCpu, BinaryAndProbabilityUseDistinctStorage) {
  const std::uint8_t binary_pixels[] = {0, 1, 1, 0};
  vie::Mask binary = vie::Mask::binary(vie::DeviceContext{}, 2, 2, binary_pixels, 4);
  EXPECT_EQ(binary.width(), 2);
  EXPECT_EQ(binary.height(), 2);
  EXPECT_EQ(binary.kind(), vie::MaskKind::Binary);
  EXPECT_EQ(binary.layout(), vie::MaskLayout::Hw);
  EXPECT_EQ(binary.dtype(), vie::DataType::Uint8);
  EXPECT_EQ(binary.device(), vie::DeviceType::Cpu);
  EXPECT_EQ(binary.device_id(), 0);
  ASSERT_EQ(binary.shape().size(), 2u);
  EXPECT_EQ(binary.shape()[0], 2);
  EXPECT_EQ(binary.shape()[1], 2);
  EXPECT_EQ(binary.binary_data()[1], 1);
  EXPECT_EQ(binary.tensor().nbytes(), 4u);

  const float probability_pixels[] = {0.f, 0.25f, 1.f};
  vie::Mask probability = vie::Mask::probability(vie::DeviceContext{}, 3, 1, probability_pixels, 3);
  EXPECT_EQ(probability.kind(), vie::MaskKind::Probability);
  EXPECT_EQ(probability.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(probability.device(), vie::DeviceType::Cpu);
  EXPECT_FLOAT_EQ(probability.probability_data()[1], 0.25f);
  EXPECT_EQ(probability.tensor().nbytes(), 12u);

  vie::Mask zeros = vie::Mask::binary(vie::DeviceContext{}, 2, 1);
  EXPECT_EQ(zeros.binary_data()[0], 0);
  EXPECT_EQ(zeros.binary_data()[1], 0);

  const std::uint8_t illegal_binary[] = {2};
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Mask::binary(vie::DeviceContext{}, 1, 1, illegal_binary, 1));
  const float logit = 2.f;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Mask::probability(vie::DeviceContext{}, 1, 1, &logit, 1));
  const float missing = std::numeric_limits<float>::quiet_NaN();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Mask::probability(vie::DeviceContext{}, 1, 1, &missing, 1));
}

TEST(MaskCpu, RejectsBadShapeDtypeAndNonContiguousStorage) {
  vie::Tensor fp32 = vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32, {2, 2});
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::Mask::from_tensor(vie::MaskKind::Binary, fp32));
  vie::Tensor bytes = vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Uint8, {2, 2});
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::Mask::from_tensor(vie::MaskKind::Probability, bytes));
  vie::Tensor rank3 = vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Uint8, {1, 2, 2});
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::Mask::from_tensor(vie::MaskKind::Binary, rank3));
  vie::Tensor wide = vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Uint8, {2, 4});
  vie::Tensor strided = wide.view({2, 2}, {4, 1}, 0);
  EXPECT_FALSE(strided.contiguous());
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, vie::Mask::from_tensor(vie::MaskKind::Binary, strided));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::Mask::binary(vie::DeviceContext{}, 0, 2));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Mask::binary(vie::DeviceContext{}, 2, 2, static_cast<const std::uint8_t*>(nullptr), 4));
}

TEST(MaskCpu, StorageOutlivesTheOriginalMask) {
  const std::uint8_t pixels[] = {1, 0, 0, 1};
  vie::Mask copy;
  {
    vie::Mask original = vie::Mask::binary(vie::DeviceContext{}, 2, 2, pixels, 4);
    copy = original;
    EXPECT_TRUE(copy.tensor().shares_storage(original.tensor()));
  }
  EXPECT_EQ(copy.binary_data()[0], 1);
  EXPECT_EQ(copy.binary_data()[3], 1);
  vie::Tensor retired = copy.tensor();
  copy = vie::Mask{};
  EXPECT_EQ(retired.data_as<std::uint8_t>()[3], 1);
}

TEST(SegmentationBatch, KeepsDetectionAndMaskPaired) {
  const std::uint8_t first_pixels[] = {1, 0};
  const std::uint8_t second_pixels[] = {0, 1};
  vie::InstanceSegmentation first;
  first.detection.box = {1.f, 2.f, 3.f, 4.f};
  first.detection.confidence = 0.8f;
  first.detection.class_id = 4;
  first.mask = vie::Mask::binary(vie::DeviceContext{}, 2, 1, first_pixels, 2);
  vie::InstanceSegmentation second;
  second.detection.class_id = 7;
  second.detection.confidence = 0.3f;
  second.mask = vie::Mask::binary(vie::DeviceContext{}, 2, 1, second_pixels, 2);

  vie::SegmentationBatch batch;
  batch.instances.push_back(first);
  batch.instances.push_back(std::move(second));
  batch.original_width = 20;
  batch.original_height = 10;
  batch.execution_device = vie::DeviceType::Cpu;
  batch.execution_device_id = 0;
  batch.protocol = "unspecified";

  ASSERT_EQ(batch.instances.size(), 2u);
  EXPECT_EQ(batch.instances[0].detection.class_id, 4);
  EXPECT_EQ(batch.instances[0].mask.binary_data()[0], 1);
  EXPECT_EQ(batch.instances[1].detection.class_id, 7);
  EXPECT_EQ(batch.instances[1].mask.binary_data()[1], 1);
  EXPECT_EQ(batch.original_width, 20);
  EXPECT_EQ(batch.original_height, 10);
  EXPECT_EQ(batch.execution_device, vie::DeviceType::Cpu);
  EXPECT_TRUE(batch.instances[0].mask.tensor().shares_storage(first.mask.tensor()));
}

TEST(DecoderInterface, SegmentationDefaultsToUnsupportedAndDetectionBatchStays) {
  vie::DecoderInput input;
  vie::DecoderConfig config;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::YoloV8DetectionDecoder{}.decode_segmentation(input, config, vie::DeviceContext{}));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::MockDecoder{}.decode_segmentation(input, config, vie::DeviceContext{}));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::MockDecoder{}.decode_batch(input, config, vie::DeviceContext{}));

  vie::DecoderRegistry registry;
  registry.register_factory("test_seg_only", [] { return std::make_unique<SegmentationOnlyDecoder>(); });
  const std::unique_ptr<vie::IModelDecoder> decoder = registry.create("test_seg_only");
  const vie::SegmentationBatch batch = decoder->decode_segmentation(input, config, vie::DeviceContext{});
  EXPECT_EQ(batch.protocol, "test_seg_only");
  EXPECT_EQ(batch.original_width, 3);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, decoder->decode_batch(input, config, vie::DeviceContext{}));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, registry.create("yolov8_seg_raw"));
}

#if defined(VIE_HAS_CUDA)
TEST(MaskCuda, StoresOnDeviceAndCopiesBackOnlyWhenAsked) {
  vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  const std::uint8_t binary_pixels[] = {1, 0, 1, 1};
  vie::Mask binary = vie::Mask::binary(gpu, 2, 2, binary_pixels, 4);
  EXPECT_EQ(binary.device(), vie::DeviceType::Cuda);
  EXPECT_EQ(binary.device_id(), 0);
  EXPECT_EQ(binary.dtype(), vie::DataType::Uint8);
  EXPECT_EQ(binary.kind(), vie::MaskKind::Binary);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, binary.binary_data());
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, binary.tensor().data());
  ASSERT_NE(binary.tensor().device_data(), nullptr);
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, binary.tensor().device_data()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

  vie::Mask host = binary.to_cpu();
  EXPECT_EQ(host.device(), vie::DeviceType::Cpu);
  EXPECT_EQ(host.kind(), vie::MaskKind::Binary);
  EXPECT_EQ(host.binary_data()[0], 1);
  EXPECT_EQ(host.binary_data()[1], 0);
  EXPECT_FALSE(host.tensor().shares_storage(binary.tensor()));

  const float probability_pixels[] = {0.5f, 1.f};
  vie::Mask probability = vie::Mask::probability(gpu, 2, 1, probability_pixels, 2);
  EXPECT_EQ(probability.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(probability.device(), vie::DeviceType::Cuda);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, probability.probability_data());
  vie::Mask probability_host = probability.to_cpu();
  EXPECT_FLOAT_EQ(probability_host.probability_data()[0], 0.5f);
  EXPECT_FLOAT_EQ(probability_host.probability_data()[1], 1.f);

  vie::Tensor cpu_bytes = vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Uint8, {1, 1});
  vie::Mask cpu_mask = vie::Mask::from_tensor(vie::MaskKind::Binary, cpu_bytes);
  EXPECT_EQ(cpu_mask.device(), vie::DeviceType::Cpu);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, cpu_mask.tensor().device_data());
}
#endif
