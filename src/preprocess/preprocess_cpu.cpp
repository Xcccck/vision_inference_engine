#include "preprocess_cpu.hpp"

#include "preprocess_math.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vie {
namespace {

std::size_t element_size(DataType dtype) { return dtype == DataType::Uint8 ? 1 : 4; }

std::size_t resolved_stride(const ImageInput& image) {
  const std::size_t element = element_size(image.dtype);
  const std::size_t packed =
      image.layout == ImageLayout::Hwc
          ? static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.channels) * element
          : static_cast<std::size_t>(image.width) * element;
  return image.row_stride_bytes == 0 ? packed : image.row_stride_bytes;
}

float read_sample(const std::byte* base, std::size_t stride, ImageLayout layout, DataType dtype,
                  int height, int channels, int x, int y, int channel) {
  const std::size_t element = element_size(dtype);
  std::size_t offset = 0;
  if (layout == ImageLayout::Hwc) {
    offset = static_cast<std::size_t>(y) * stride +
             (static_cast<std::size_t>(x) * static_cast<std::size_t>(channels) +
              static_cast<std::size_t>(channel)) *
                 element;
  } else {
    offset = static_cast<std::size_t>(channel) * static_cast<std::size_t>(height) * stride +
             static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * element;
  }
  if (dtype == DataType::Uint8) {
    return static_cast<float>(reinterpret_cast<const std::uint8_t*>(base)[offset]);
  }
  float value = 0.f;
  std::memcpy(&value, base + offset, sizeof(value));
  return value;
}

float sample_channel(const std::byte* base, std::size_t stride, const ImageInput& image, int channel,
                     float src_x, float src_y, Interpolation interpolation) {
  if (interpolation == Interpolation::Nearest) {
    const int x = nearest_index(src_x, image.width);
    const int y = nearest_index(src_y, image.height);
    return read_sample(base, stride, image.layout, image.dtype, image.height, image.channels, x, y,
                       channel);
  }
  const int x0 = static_cast<int>(std::floor(src_x));
  const int y0 = static_cast<int>(std::floor(src_y));
  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float v00 = read_sample(base, stride, image.layout, image.dtype, image.height, image.channels,
                                clamp_index(x0, image.width), clamp_index(y0, image.height), channel);
  const float v10 = read_sample(base, stride, image.layout, image.dtype, image.height, image.channels,
                                clamp_index(x1, image.width), clamp_index(y0, image.height), channel);
  const float v01 = read_sample(base, stride, image.layout, image.dtype, image.height, image.channels,
                                clamp_index(x0, image.width), clamp_index(y1, image.height), channel);
  const float v11 = read_sample(base, stride, image.layout, image.dtype, image.height, image.channels,
                                clamp_index(x1, image.width), clamp_index(y1, image.height), channel);
  const float top = v00 * (1.f - fx) + v10 * fx;
  const float bottom = v01 * (1.f - fx) + v11 * fx;
  return top * (1.f - fy) + bottom * fy;
}

}  // namespace

PreprocessResult preprocess_cpu(const PreprocessConfig& config, const ImageInput& image,
                                const DeviceContext& device, const GeometryTransform& geometry) {
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

  const std::byte* base = static_cast<const std::byte*>(image.data);
  const std::size_t stride = resolved_stride(image);
  const int swap = config.input_color != config.output_color ? 1 : 0;
  const bool normalize = config.normalize_enabled;
  float* floats = config.output_dtype == DataType::Fp32 ? output.mutable_data<float>() : nullptr;
  std::uint8_t* bytes =
      config.output_dtype == DataType::Uint8 ? output.mutable_data<std::uint8_t>() : nullptr;

  for (int y = 0; y < geometry.output_height; ++y) {
    for (int x = 0; x < geometry.output_width; ++x) {
      const int origin_x = geometry_model_origin_x(geometry);
      const int origin_y = geometry_model_origin_y(geometry);
      const int content_x = x - origin_x;
      const int content_y = y - origin_y;
      const bool inside = content_x >= 0 && content_y >= 0 && content_x < geometry.resized_width &&
                          content_y < geometry.resized_height;
      float src_x = 0.f;
      float src_y = 0.f;
      if (inside) {
        src_x = source_coordinate(content_x, image.width, geometry.resized_width);
        src_y = source_coordinate(content_y, image.height, geometry.resized_height);
      }
      for (int c = 0; c < 3; ++c) {
        float value = config.padding_value;
        if (inside) {
          const int channel = source_channel(c, image.channels, swap);
          value = sample_channel(base, stride, image, channel, src_x, src_y, config.interpolation);
        }
        if (normalize) {
          value = normalize_sample(value, 1, config.scale, config.mean[static_cast<std::size_t>(c)],
                                   config.std[static_cast<std::size_t>(c)]);
        }
        const int index = config.output_layout == ImageLayout::Chw
                              ? (c * geometry.output_height + y) * geometry.output_width + x
                              : (y * geometry.output_width + x) * 3 + c;
        if (floats != nullptr) {
          floats[index] = value;
        } else {
          bytes[index] = static_cast<std::uint8_t>(round_to_uint8(value));
        }
      }
    }
  }

  PreprocessResult result;
  result.output = std::move(output);
  result.geometry = geometry;
  return result;
}

}  // namespace vie
