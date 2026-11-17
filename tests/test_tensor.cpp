#include "test_util.hpp"

#include "vie/core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace {

vie::DeviceContext host() { return vie::DeviceContext(vie::DeviceType::Cpu, 0); }

}  // namespace

TEST(TensorTest, ZerosUseRowMajorStrides) {
  const vie::Tensor tensor = vie::Tensor::zeros(host(), vie::DataType::Fp32, {1, 3, 4, 4});
  EXPECT_EQ(tensor.numel(), 48u);
  EXPECT_EQ(tensor.nbytes(), 192u);
  EXPECT_TRUE(tensor.contiguous());
  EXPECT_EQ(tensor.device(), vie::DeviceType::Cpu);
  const std::vector<std::int64_t> expected{48, 16, 4, 1};
  EXPECT_EQ(tensor.strides(), expected);
  EXPECT_FLOAT_EQ(tensor.data_as<float>()[0], 0.f);
}

TEST(TensorTest, ViewSharesStorageAndCloneDoesNot) {
  vie::Tensor full = vie::Tensor::zeros(host(), vie::DataType::Fp32, {2, 2});
  float* values = full.mutable_data<float>();
  values[0] = 1.f;
  values[1] = 2.f;
  values[2] = 3.f;
  values[3] = 4.f;

  const vie::Tensor tail = full.view({2}, {1}, 2 * sizeof(float));
  EXPECT_TRUE(tail.shares_storage(full));
  EXPECT_TRUE(tail.contiguous());
  EXPECT_FLOAT_EQ(tail.data_as<float>()[0], 3.f);

  vie::Tensor cloned = tail.clone();
  EXPECT_FALSE(cloned.shares_storage(full));
  values[2] = 9.f;
  EXPECT_FLOAT_EQ(tail.data_as<float>()[0], 9.f);
  EXPECT_FLOAT_EQ(cloned.data_as<float>()[0], 3.f);
  EXPECT_FLOAT_EQ(cloned.data_as<float>()[1], 4.f);

  const vie::Tensor column = full.view({2}, {2}, 0);
  EXPECT_FALSE(column.contiguous());
  const vie::Tensor column_copy = column.clone();
  EXPECT_FALSE(column_copy.shares_storage(full));
  EXPECT_FLOAT_EQ(column_copy.data_as<float>()[0], 1.f);
  EXPECT_FLOAT_EQ(column_copy.data_as<float>()[1], 9.f);
}

TEST(TensorTest, RejectsCudaAndInvalidShapes) {
#if defined(VIE_HAS_CUDA)
  const vie::DeviceContext cuda(vie::DeviceType::Cuda, 0);
  EXPECT_EQ(cuda.device(), vie::DeviceType::Cuda);
  EXPECT_EQ(cuda.device_id(), 0);
  cuda.synchronize();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::DeviceContext(vie::DeviceType::Cuda, 99));
#else
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::DeviceContext(vie::DeviceType::Cuda, 0));
#endif
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Tensor::zeros(host(), vie::DataType::Fp32, {}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Tensor::zeros(host(), vie::DataType::Fp32, {1, 0, 2}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Tensor::zeros(host(), vie::DataType::Fp32, {1, -1}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::DeviceContext(vie::DeviceType::Cpu, -1));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::DeviceContext(vie::DeviceType::Cuda, -1));
}

TEST(TensorTest, CpuCopyKeepsValuesAndRejectsDevicePointer) {
  const float values[] = {1.f, 2.f, 3.f, 4.f};
  vie::Tensor source = vie::Tensor::from_host(host(), vie::DataType::Fp32, {2, 2}, values, sizeof(values));
  vie::Tensor destination = vie::Tensor::zeros(host(), vie::DataType::Fp32, {2, 2});
  destination.copy_from(source);
  EXPECT_FLOAT_EQ(destination.data_as<float>()[3], 4.f);

  vie::Tensor moved = source.to(host());
  EXPECT_FALSE(moved.shares_storage(source));
  source.mutable_data<float>()[0] = 8.f;
  EXPECT_FLOAT_EQ(moved.data_as<float>()[0], 1.f);
  EXPECT_TRUE(source.is_cpu());
  EXPECT_FALSE(source.is_cuda());
  EXPECT_EQ(source.storage_bytes(), source.nbytes());
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, source.device_data());
}

TEST(TensorTest, RejectsViewsOutsideStorage) {
  const vie::Tensor tensor = vie::Tensor::zeros(host(), vie::DataType::Fp32, {4});
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, tensor.view({2}, {1}, 3 * sizeof(float)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, tensor.view({1}, {1}, 1));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, tensor.view({2}, {-1}, 0));
}
