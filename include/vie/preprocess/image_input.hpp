#pragma once

#include "vie/core/types.hpp"

#include <cstddef>

namespace vie {

enum class ColorOrder { Rgb, Bgr, Gray };

enum class ImageLayout { Hwc, Chw };

// Borrowed: data stays owned by the caller and must remain valid for the
// whole execute() call. The preprocessor never frees it and never wraps it in
// a Tensor that would free it.
enum class ImageOwnership { Borrowed };

// One image. data is a host pointer when device is CPU and a CUDA device
// pointer when device is CUDA. row_stride_bytes is the distance between rows.
// For HWC that row contains all channels. For CHW it is the row stride inside
// one channel plane, and planes are contiguous. Zero means tightly packed.
struct ImageInput {
  const void* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;
  ColorOrder color = ColorOrder::Rgb;
  ImageLayout layout = ImageLayout::Hwc;
  DataType dtype = DataType::Uint8;
  std::size_t row_stride_bytes = 0;
  DeviceType device = DeviceType::Cpu;
  int device_id = 0;
  ImageOwnership ownership = ImageOwnership::Borrowed;
};

const char* to_string(ColorOrder color);
const char* to_string(ImageLayout layout);

}  // namespace vie
