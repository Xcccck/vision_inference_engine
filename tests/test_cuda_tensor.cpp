#include "test_util.hpp"

#include "vie/core/tensor.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace {

vie::DeviceContext cpu() { return vie::DeviceContext(vie::DeviceType::Cpu, 0); }

vie::DeviceContext cuda0() { return vie::DeviceContext(vie::DeviceType::Cuda, 0); }

void expect_cuda_clean() {
  const cudaError_t status = cudaGetLastError();
  EXPECT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

std::vector<float> read_floats(const vie::Tensor& tensor) {
  const vie::Tensor host = tensor.is_cpu() ? tensor.clone() : tensor.to(cpu());
  const float* data = host.data_as<float>();
  return std::vector<float>(data, data + host.numel());
}

void expect_values(const vie::Tensor& tensor, const std::vector<float>& expected) {
  const std::vector<float> actual = read_floats(tensor);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
  }
}

}  // namespace

TEST(CudaTensorTest, ContextRejectsIllegalDeviceAndKeepsStream) {
  const vie::DeviceContext first = cuda0();
  const vie::DeviceContext second = first;
  EXPECT_EQ(first.device(), vie::DeviceType::Cuda);
  EXPECT_EQ(first.device_id(), 0);
  EXPECT_EQ(second.execution_state(), first.execution_state());
  first.synchronize();
  second.synchronize();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::DeviceContext(vie::DeviceType::Cuda, 99));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::DeviceContext(vie::DeviceType::Cuda, -1));
  expect_cuda_clean();
}

TEST(CudaTensorTest, AllocationIsDeviceMemoryAndZerosAreZero) {
  const vie::Tensor tensor = vie::Tensor::zeros(cuda0(), vie::DataType::Fp32, {2, 3});
  EXPECT_TRUE(tensor.is_cuda());
  EXPECT_FALSE(tensor.is_cpu());
  EXPECT_EQ(tensor.device_id(), 0);
  EXPECT_EQ(tensor.storage_bytes(), tensor.nbytes());
  EXPECT_NE(tensor.device_data(), nullptr);

  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, tensor.device_data()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
  EXPECT_EQ(attributes.device, 0);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, tensor.data());
  expect_values(tensor, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
  expect_cuda_clean();
}

TEST(CudaTensorTest, CopiesRoundTripBetweenCpuAndCuda) {
  const vie::DeviceContext device = cuda0();
  const float values[] = {1.5f, -2.f, 3.25f, 4.f};
  const vie::Tensor source =
      vie::Tensor::from_host(cpu(), vie::DataType::Fp32, {4}, values, sizeof(values));
  vie::Tensor gpu = vie::Tensor::zeros(device, vie::DataType::Fp32, {4});
  gpu.copy_from(source);
  expect_values(gpu, {1.5f, -2.f, 3.25f, 4.f});

  vie::Tensor alias = vie::Tensor::zeros(device, vie::DataType::Fp32, {4});
  alias.copy_from(gpu);
  expect_values(alias, {1.5f, -2.f, 3.25f, 4.f});

  const float replaced[] = {8.f, 7.f, 6.f, 5.f};
  gpu.copy_from(vie::Tensor::from_host(cpu(), vie::DataType::Fp32, {4}, replaced, sizeof(replaced)));
  expect_values(gpu, {8.f, 7.f, 6.f, 5.f});
  expect_values(alias, {1.5f, -2.f, 3.25f, 4.f});

  const vie::Tensor back = gpu.to(cpu());
  EXPECT_TRUE(back.is_cpu());
  expect_values(back, {8.f, 7.f, 6.f, 5.f});
  expect_cuda_clean();
}

TEST(CudaTensorTest, CloneIsIndependentAndViewSharesStorage) {
  const float values[] = {1.f, 2.f, 3.f, 4.f};
  vie::Tensor original =
      vie::Tensor::from_host(cuda0(), vie::DataType::Fp32, {2, 2}, values, sizeof(values));
  vie::Tensor cloned = original.clone();
  EXPECT_FALSE(cloned.shares_storage(original));
  EXPECT_TRUE(cloned.contiguous());
  EXPECT_TRUE(cloned.is_cuda());

  const float changed[] = {9.f, 8.f, 7.f, 6.f};
  original.copy_from(
      vie::Tensor::from_host(cpu(), vie::DataType::Fp32, {2, 2}, changed, sizeof(changed)));
  expect_values(original, {9.f, 8.f, 7.f, 6.f});
  expect_values(cloned, {1.f, 2.f, 3.f, 4.f});

  vie::Tensor viewed = original.view({2}, {1}, 2 * sizeof(float));
  EXPECT_TRUE(viewed.shares_storage(original));
  EXPECT_EQ(viewed.byte_offset(), 2 * sizeof(float));
  expect_values(viewed, {7.f, 6.f});

  vie::Tensor surviving;
  {
    vie::Tensor temporary =
        vie::Tensor::from_host(cuda0(), vie::DataType::Fp32, {4}, values, sizeof(values));
    surviving = temporary.view({2}, {1}, sizeof(float));
    EXPECT_TRUE(surviving.shares_storage(temporary));
  }
  expect_values(surviving, {2.f, 3.f});
  expect_cuda_clean();
}

TEST(CudaTensorTest, NonContiguousClonePreservesStridesAndOffset) {
  const float values[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  const vie::Tensor original =
      vie::Tensor::from_host(cuda0(), vie::DataType::Fp32, {2, 3}, values, sizeof(values));
  const vie::Tensor column = original.view({2}, {3}, sizeof(float));
  EXPECT_FALSE(column.contiguous());
  EXPECT_EQ(column.byte_offset(), sizeof(float));

  const vie::Tensor packed = column.clone();
  EXPECT_FALSE(packed.shares_storage(original));
  EXPECT_TRUE(packed.contiguous());
  EXPECT_TRUE(packed.is_cuda());
  EXPECT_EQ(packed.byte_offset(), 0u);
  expect_values(packed, {2.f, 5.f});

  const vie::Tensor on_cpu = column.to(cpu());
  EXPECT_TRUE(on_cpu.is_cpu());
  EXPECT_TRUE(on_cpu.contiguous());
  expect_values(on_cpu, {2.f, 5.f});
  expect_cuda_clean();
}

TEST(CudaTensorTest, RejectsBadShapeDtypeAndCapacity) {
  const vie::DeviceContext device = cuda0();
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Tensor::zeros(device, vie::DataType::Unknown, {2}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::Tensor::zeros(device, vie::DataType::Fp32, {}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Tensor::zeros(device, vie::DataType::Fp32, {1, 0, 2}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Tensor::zeros(device, vie::DataType::Int32, {1, -1}));

  vie::Tensor gpu = vie::Tensor::zeros(device, vie::DataType::Fp32, {4});
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, gpu.view({2}, {1}, 3 * sizeof(float)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, gpu.view({2}, {-1}, 0));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::Tensor::from_host(device, vie::DataType::Fp32, {4},
                                                                          nullptr, 16));
  const float short_source[] = {1.f};
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::Tensor::from_host(device, vie::DataType::Fp32, {4}, short_source,
                                          sizeof(short_source)));

  const vie::Tensor cpu_tensor = vie::Tensor::zeros(cpu(), vie::DataType::Int32, {4});
  try {
    gpu.copy_from(cpu_tensor);
    FAIL() << "expected VieError";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::InvalidArgument);
    const std::string message = error.what();
    EXPECT_NE(message.find("cuda:0"), std::string::npos) << message;
    EXPECT_NE(message.find("cpu:0"), std::string::npos) << message;
  }
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, cpu_tensor.device_data());
  expect_cuda_clean();
}

TEST(CudaTensorTest, IntegerRoundTripAndRepeatedRelease) {
  const std::int32_t values[] = {-1, 0, 42, 1000};
  const vie::Tensor gpu =
      vie::Tensor::from_host(cuda0(), vie::DataType::Int32, {4}, values, sizeof(values));
  const vie::Tensor back = gpu.to(cpu());
  const std::int32_t* data = back.data_as<std::int32_t>();
  EXPECT_EQ(data[0], -1);
  EXPECT_EQ(data[1], 0);
  EXPECT_EQ(data[2], 42);
  EXPECT_EQ(data[3], 1000);

  for (int i = 0; i < 32; ++i) {
    vie::Tensor temporary = vie::Tensor::zeros(cuda0(), vie::DataType::Fp32, {128, 128});
    EXPECT_EQ(temporary.storage_bytes(), 128u * 128u * sizeof(float));
  }
  expect_cuda_clean();
}

TEST(CudaTensorTest, RecordsAllocationAndTransferTime) {
  constexpr std::int64_t n = 1 * 3 * 640 * 640;
  const std::vector<std::int64_t> shape{1, 3, 640, 640};
  std::vector<float> host(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<float>(i % 251);
  }
  using clock = std::chrono::steady_clock;
  const auto cpu_begin = clock::now();
  vie::Tensor cpu_tensor =
      vie::Tensor::from_host(cpu(), vie::DataType::Fp32, shape, host.data(), host.size() * sizeof(float));
  const auto cpu_end = clock::now();

  const auto cuda_begin = clock::now();
  vie::Tensor gpu = vie::Tensor::zeros(cuda0(), vie::DataType::Fp32, shape);
  const auto cuda_end = clock::now();
  const auto cuda_again_begin = clock::now();
  vie::Tensor gpu_again = vie::Tensor::zeros(cuda0(), vie::DataType::Fp32, shape);
  const auto cuda_again_end = clock::now();
  gpu_again = vie::Tensor();

  const auto h2d_begin = clock::now();
  gpu.copy_from(cpu_tensor);
  const auto h2d_end = clock::now();

  vie::Tensor other = vie::Tensor::zeros(cuda0(), vie::DataType::Fp32, shape);
  const auto d2d_begin = clock::now();
  other.copy_from(gpu);
  const auto d2d_end = clock::now();

  vie::Tensor back = vie::Tensor::zeros(cpu(), vie::DataType::Fp32, shape);
  const auto d2h_begin = clock::now();
  back.copy_from(gpu);
  const auto d2h_end = clock::now();

  const auto clone_begin = clock::now();
  const vie::Tensor cloned = gpu.clone();
  const auto clone_end = clock::now();

  auto ms = [](clock::time_point begin, clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  };
  EXPECT_EQ(back.numel(), host.size());
  EXPECT_FLOAT_EQ(back.data_as<float>()[0], host[0]);
  EXPECT_FLOAT_EQ(back.data_as<float>()[host.size() / 2], host[host.size() / 2]);
  EXPECT_FLOAT_EQ(back.data_as<float>()[host.size() - 1], host[host.size() - 1]);
  const vie::Tensor cloned_back = cloned.to(cpu());
  EXPECT_FLOAT_EQ(cloned_back.data_as<float>()[host.size() - 1], host[host.size() - 1]);

  std::ofstream out("/tmp/vie_phase421_memory.txt");
  ASSERT_TRUE(out.is_open());
  out << "shape=1x3x640x640 fp32 bytes=" << cpu_tensor.nbytes() << "\n";
  out << "cpu_from_host_ms=" << ms(cpu_begin, cpu_end) << "\n";
  out << "cuda_zeros_first_malloc_memset_sync_ms=" << ms(cuda_begin, cuda_end) << "\n";
  out << "cuda_zeros_second_malloc_memset_sync_ms=" << ms(cuda_again_begin, cuda_again_end) << "\n";
  out << "h2d_copy_plus_sync_ms=" << ms(h2d_begin, h2d_end) << "\n";
  out << "d2d_copy_plus_sync_ms=" << ms(d2d_begin, d2d_end) << "\n";
  out << "d2h_copy_plus_sync_ms=" << ms(d2h_begin, d2h_end) << "\n";
  out << "cuda_clone_alloc_copy_sync_ms=" << ms(clone_begin, clone_end) << "\n";
  out << "note=copy wall time includes synchronization before return; "
         "cuda zeros includes allocation, memset, and stream sync\n";
  expect_cuda_clean();
}
