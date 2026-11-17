#pragma once

// Independent scalar mask reference. It does not call YoloV8SegmentationDecoder.
// Crop is the half-open float rule x1 <= x < x2, y1 <= y < y2. Bilinear matches
// source = (index + 0.5) * (in / out) - 0.5 with edge replication.

#include "vie/decoder/detection.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace vie_ref {

inline float bilinear(const std::vector<float>& source, int width, int height, double x, double y) {
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const float wx = static_cast<float>(x - static_cast<double>(x0));
  const float wy = static_cast<float>(y - static_cast<double>(y0));
  const auto at = [&](int px, int py) {
    if (px < 0) px = 0;
    if (py < 0) py = 0;
    if (px >= width) px = width - 1;
    if (py >= height) py = height - 1;
    return source[static_cast<std::size_t>(py) * static_cast<std::size_t>(width) + static_cast<std::size_t>(px)];
  };
  const float top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * wx;
  const float bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * wx;
  return top + (bottom - top) * wy;
}

inline void zero_outside(std::vector<float>& image, int width, int height, float x1, float y1, float x2, float y2) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      if (!(fx >= x1 && fx < x2 && fy >= y1 && fy < y2)) {
        image[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] = 0.f;
      }
    }
  }
}

inline std::vector<std::uint8_t> binary_mask(const float* proto, int channels, int proto_h, int proto_w,
                                            const std::vector<float>& coefficients, const vie::BoundingBox& model_box,
                                            const vie::BoundingBox& original_box, const vie::GeometryTransform& geometry,
                                            bool retina) {
  const int mask_w = retina ? geometry.original_width : geometry.output_width;
  const int mask_h = retina ? geometry.original_height : geometry.output_height;
  const std::size_t proto_pixels = static_cast<std::size_t>(proto_h) * static_cast<std::size_t>(proto_w);
  std::vector<float> logits(proto_pixels, 0.f);
  for (int y = 0; y < proto_h; ++y) {
    for (int x = 0; x < proto_w; ++x) {
      float sum = 0.f;
      const std::size_t spatial = static_cast<std::size_t>(y) * static_cast<std::size_t>(proto_w) +
                                  static_cast<std::size_t>(x);
      for (int channel = 0; channel < channels; ++channel) {
        sum += coefficients[static_cast<std::size_t>(channel)] *
               proto[static_cast<std::size_t>(channel) * proto_pixels + spatial];
      }
      logits[spatial] = sum;
    }
  }
  const double proto_per_model_x = static_cast<double>(proto_w) / static_cast<double>(geometry.output_width);
  const double proto_per_model_y = static_cast<double>(proto_h) / static_cast<double>(geometry.output_height);
  std::vector<float> scaled(static_cast<std::size_t>(mask_w) * static_cast<std::size_t>(mask_h), 0.f);
  if (!retina) {
    vie::BoundingBox proto_box;
    proto_box.x1 = static_cast<float>(static_cast<double>(model_box.x1) * proto_per_model_x);
    proto_box.y1 = static_cast<float>(static_cast<double>(model_box.y1) * proto_per_model_y);
    proto_box.x2 = static_cast<float>(static_cast<double>(model_box.x2) * proto_per_model_x);
    proto_box.y2 = static_cast<float>(static_cast<double>(model_box.y2) * proto_per_model_y);
    zero_outside(logits, proto_w, proto_h, proto_box.x1, proto_box.y1, proto_box.x2, proto_box.y2);
    for (int y = 0; y < mask_h; ++y) {
      const double src_y = (static_cast<double>(y) + 0.5) * proto_per_model_y - 0.5;
      for (int x = 0; x < mask_w; ++x) {
        const double src_x = (static_cast<double>(x) + 0.5) * proto_per_model_x - 0.5;
        scaled[static_cast<std::size_t>(y) * static_cast<std::size_t>(mask_w) + static_cast<std::size_t>(x)] =
            bilinear(logits, proto_w, proto_h, src_x, src_y);
      }
    }
  } else {
    for (int y = 0; y < mask_h; ++y) {
      const double src_y =
          (static_cast<double>(geometry.pad_top) + (static_cast<double>(y) + 0.5) * geometry.scale_y) *
              proto_per_model_y -
          0.5;
      for (int x = 0; x < mask_w; ++x) {
        const double src_x =
            (static_cast<double>(geometry.pad_left) + (static_cast<double>(x) + 0.5) * geometry.scale_x) *
                proto_per_model_x -
            0.5;
        scaled[static_cast<std::size_t>(y) * static_cast<std::size_t>(mask_w) + static_cast<std::size_t>(x)] =
            bilinear(logits, proto_w, proto_h, src_x, src_y);
      }
    }
    zero_outside(scaled, mask_w, mask_h, original_box.x1, original_box.y1, original_box.x2, original_box.y2);
  }
  std::vector<std::uint8_t> binary(scaled.size(), 0);
  for (std::size_t i = 0; i < scaled.size(); ++i) binary[i] = scaled[i] > 0.f ? 1 : 0;
  return binary;
}

}  // namespace vie_ref
