#pragma once

// Shared sampling formulas. Host and device both use these so CPU and CUDA
// agree on coordinates, rounding, and edge replication.

#if defined(__CUDACC__)
#define VIE_HD __host__ __device__
#else
#define VIE_HD
#endif

namespace vie {

VIE_HD inline int clamp_index(int value, int limit) {
  if (value < 0) return 0;
  if (value >= limit) return limit - 1;
  return value;
}

VIE_HD inline float source_coordinate(int content_index, int source_size, int resized_size) {
  return (static_cast<float>(content_index) + 0.5f) *
             (static_cast<float>(source_size) / static_cast<float>(resized_size)) -
         0.5f;
}

VIE_HD inline int nearest_index(float coordinate, int limit) {
  int index = static_cast<int>(coordinate >= 0.f ? coordinate + 0.5f : coordinate - 0.5f);
  return clamp_index(index, limit);
}

VIE_HD inline int source_channel(int output_channel, int input_channels, int swap_rb) {
  if (input_channels == 1) return 0;
  if (swap_rb && (output_channel == 0 || output_channel == 2)) return 2 - output_channel;
  return output_channel;
}

VIE_HD inline float normalize_sample(float value, int enabled, float scale, float mean, float stddev) {
  if (!enabled) return value;
  return (value * scale - mean) / stddev;
}

VIE_HD inline int round_to_uint8(float value) {
  if (value <= 0.f) return 0;
  if (value >= 255.f) return 255;
  return static_cast<int>(value + 0.5f);
}

}  // namespace vie

#undef VIE_HD
