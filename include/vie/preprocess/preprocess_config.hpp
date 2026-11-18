#pragma once

#include "vie/preprocess/image_input.hpp"

#include <vector>

namespace vie {

enum class ResizeMode { None, Stretch, Letterbox, ShortestSide };

enum class Interpolation { Nearest, Bilinear };

// Explicit preprocess parameters. Nothing here is inferred from a model name.
// mean and std are in output-channel order and are required only when
// normalize_enabled is true. padding_value is the numeric sample written into
// letterbox borders before normalization.
struct PreprocessConfig {
  ResizeMode resize_mode = ResizeMode::None;
  int target_width = 0;
  int target_height = 0;
  Interpolation interpolation = Interpolation::Nearest;
  ColorOrder input_color = ColorOrder::Rgb;
  ColorOrder output_color = ColorOrder::Rgb;
  ImageLayout input_layout = ImageLayout::Hwc;
  ImageLayout output_layout = ImageLayout::Chw;
  DataType output_dtype = DataType::Fp32;
  bool normalize_enabled = false;
  float scale = 1.f;
  std::vector<float> mean;
  std::vector<float> std;
  float padding_value = 0.f;
  // ShortestSide only. The short edge is pinned to short_side, the long edge
  // is rounded, then a center crop of crop_width by crop_height is taken.
  // target_width and target_height must equal that crop. Other modes keep
  // these at 0.
  int short_side = 0;
  int crop_width = 0;
  int crop_height = 0;
};

const char* to_string(ResizeMode mode);
const char* to_string(Interpolation interpolation);

// Throws InvalidArgument when the config itself is not executable. Image size
// is checked later, against a real input.
void validate_preprocess_config(const PreprocessConfig& config);

}  // namespace vie
