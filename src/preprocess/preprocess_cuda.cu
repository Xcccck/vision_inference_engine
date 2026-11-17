#include "preprocess_cuda.hpp"

#include "preprocess_math.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace vie {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  throw VieError(ErrorCode::Internal,
                 std::string(operation) + " failed: " + cudaGetErrorString(status));
}

__device__ float read_sample(const std::uint8_t* u8, const float* f32, int is_u8, std::size_t stride,
                             int layout_hwc, int height, int channels, int x, int y, int channel) {
  const std::size_t element = is_u8 ? 1 : 4;
  std::size_t offset = 0;
  if (layout_hwc) {
    offset = static_cast<std::size_t>(y) * stride +
             (static_cast<std::size_t>(x) * static_cast<std::size_t>(channels) +
              static_cast<std::size_t>(channel)) *
                 element;
  } else {
    offset = static_cast<std::size_t>(channel) * static_cast<std::size_t>(height) * stride +
             static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * element;
  }
  if (is_u8) return static_cast<float>(u8[offset]);
  return *reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(f32) + offset);
}

__device__ float sample_channel(const std::uint8_t* u8, const float* f32, int is_u8, std::size_t stride,
                                int layout_hwc, int width, int height, int channels, int channel,
                                float src_x, float src_y, int nearest) {
  if (nearest) {
    return read_sample(u8, f32, is_u8, stride, layout_hwc, height, channels, nearest_index(src_x, width),
                       nearest_index(src_y, height), channel);
  }
  const int x0 = static_cast<int>(floorf(src_x));
  const int y0 = static_cast<int>(floorf(src_y));
  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);
  const float v00 = read_sample(u8, f32, is_u8, stride, layout_hwc, height, channels, clamp_index(x0, width),
                                clamp_index(y0, height), channel);
  const float v10 = read_sample(u8, f32, is_u8, stride, layout_hwc, height, channels,
                                clamp_index(x0 + 1, width), clamp_index(y0, height), channel);
  const float v01 = read_sample(u8, f32, is_u8, stride, layout_hwc, height, channels, clamp_index(x0, width),
                                clamp_index(y0 + 1, height), channel);
  const float v11 = read_sample(u8, f32, is_u8, stride, layout_hwc, height, channels,
                                clamp_index(x0 + 1, width), clamp_index(y0 + 1, height), channel);
  const float top = v00 * (1.f - fx) + v10 * fx;
  const float bottom = v01 * (1.f - fx) + v11 * fx;
  return top * (1.f - fy) + bottom * fy;
}

__global__ void preprocess_kernel(const std::uint8_t* u8, const float* f32, int is_u8,
                                  std::size_t stride, int layout_hwc, int src_w, int src_h,
                                  int channels, int swap, int nearest, int normalize, float scale,
                                  float mean0, float mean1, float mean2, float std0, float std1,
                                  float std2, float padding, int out_w, int out_h, int resized_w,
                                  int resized_h, int pad_left, int pad_top, int output_chw,
                                  int output_u8, std::uint8_t* out_u8, float* out_f32) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= out_w || y >= out_h) return;
  const int inside = x >= pad_left && y >= pad_top && x < pad_left + resized_w && y < pad_top + resized_h;
  float src_x = 0.f;
  float src_y = 0.f;
  if (inside) {
    src_x = source_coordinate(x - pad_left, src_w, resized_w);
    src_y = source_coordinate(y - pad_top, src_h, resized_h);
  }
  const float mean[3] = {mean0, mean1, mean2};
  const float stddev[3] = {std0, std1, std2};
  for (int c = 0; c < 3; ++c) {
    float value = padding;
    if (inside) {
      const int channel = source_channel(c, channels, swap);
      value = sample_channel(u8, f32, is_u8, stride, layout_hwc, src_w, src_h, channels, channel, src_x,
                             src_y, nearest);
    }
    value = normalize_sample(value, normalize, scale, mean[c], stddev[c]);
    const int index = output_chw ? (c * out_h + y) * out_w + x : (y * out_w + x) * 3 + c;
    if (output_u8) {
      out_u8[index] = static_cast<std::uint8_t>(round_to_uint8(value));
    } else {
      out_f32[index] = value;
    }
  }
}

std::size_t resolved_stride(const ImageInput& image) {
  const std::size_t element = image.dtype == DataType::Uint8 ? 1 : 4;
  const std::size_t packed =
      image.layout == ImageLayout::Hwc
          ? static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.channels) * element
          : static_cast<std::size_t>(image.width) * element;
  return image.row_stride_bytes == 0 ? packed : image.row_stride_bytes;
}

}  // namespace

PreprocessResult preprocess_cuda(const PreprocessConfig& config, const ImageInput& image,
                                 const DeviceContext& device, const GeometryTransform& geometry) {
  if (image.device != DeviceType::Cuda) {
    throw VieError(ErrorCode::InvalidArgument, "CUDA preprocess requires a CUDA image");
  }
  cuda_check(cudaSetDevice(device.device_id()), "cudaSetDevice");

  std::vector<std::int64_t> shape;
  Layout layout = Layout::Unknown;
  if (config.output_layout == ImageLayout::Chw) {
    shape = {1, 3, geometry.output_height, geometry.output_width};
    layout = Layout::Nchw;
  } else {
    shape = {1, geometry.output_height, geometry.output_width, 3};
    layout = Layout::Nhwc;
  }
  Tensor output = Tensor::zeros(device, config.output_dtype, shape);
  output.set_layout(layout);

  const bool is_u8 = image.dtype == DataType::Uint8;
  const float mean0 = config.mean.size() == 3 ? config.mean[0] : 0.f;
  const float mean1 = config.mean.size() == 3 ? config.mean[1] : 0.f;
  const float mean2 = config.mean.size() == 3 ? config.mean[2] : 0.f;
  const float std0 = config.std.size() == 3 ? config.std[0] : 1.f;
  const float std1 = config.std.size() == 3 ? config.std[1] : 1.f;
  const float std2 = config.std.size() == 3 ? config.std[2] : 1.f;
  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(geometry.output_width) + 15) / 16,
            (static_cast<unsigned>(geometry.output_height) + 15) / 16);
  struct EventGuard {
    cudaEvent_t event = nullptr;
    ~EventGuard() {
      if (event != nullptr) cudaEventDestroy(event);
    }
  };
  EventGuard started;
  EventGuard stopped;
  cuda_check(cudaEventCreate(&started.event), "cudaEventCreate");
  cuda_check(cudaEventCreate(&stopped.event), "cudaEventCreate");
  cuda_check(cudaEventRecord(started.event), "cudaEventRecord");
  preprocess_kernel<<<grid, block>>>(
      is_u8 ? static_cast<const std::uint8_t*>(image.data) : nullptr,
      is_u8 ? nullptr : static_cast<const float*>(image.data), is_u8 ? 1 : 0, resolved_stride(image),
      image.layout == ImageLayout::Hwc ? 1 : 0, image.width, image.height, image.channels,
      config.input_color != config.output_color ? 1 : 0,
      config.interpolation == Interpolation::Nearest ? 1 : 0, config.normalize_enabled ? 1 : 0,
      config.scale, mean0, mean1, mean2, std0, std1, std2, config.padding_value, geometry.output_width,
      geometry.output_height, geometry.resized_width, geometry.resized_height, geometry.pad_left,
      geometry.pad_top, config.output_layout == ImageLayout::Chw ? 1 : 0,
      config.output_dtype == DataType::Uint8 ? 1 : 0,
      config.output_dtype == DataType::Uint8 ? reinterpret_cast<std::uint8_t*>(output.device_data())
                                            : nullptr,
      config.output_dtype == DataType::Fp32 ? reinterpret_cast<float*>(output.device_data()) : nullptr);
  cuda_check(cudaGetLastError(), "preprocess kernel launch");
  cuda_check(cudaEventRecord(stopped.event), "cudaEventRecord");
  cuda_check(cudaEventSynchronize(stopped.event), "cudaEventSynchronize");
  float elapsed = 0.f;
  cuda_check(cudaEventElapsedTime(&elapsed, started.event, stopped.event), "cudaEventElapsedTime");

  PreprocessResult result;
  result.output = std::move(output);
  result.geometry = geometry;
  result.device_ms = static_cast<double>(elapsed);
  return result;
}

}  // namespace vie
