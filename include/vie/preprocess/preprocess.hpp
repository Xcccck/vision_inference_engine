#pragma once

#include "vie/preprocess/preprocess_config.hpp"

#include "vie/core/device_context.hpp"
#include "vie/core/tensor.hpp"
#include "vie/model/manifest.hpp"

namespace vie {

// Model-independent geometry. scale_x and scale_y are the ratios after integer
// rounding of the resized content box, so a letterbox of a non-square image
// can have scale_x != scale_y. uniform_scale is the ratio chosen before that
// rounding. Decoder mapping is:
//   x_original = (x_model - pad_left) / scale_x
//   y_original = (y_model - pad_top) / scale_y
struct GeometryTransform {
  int original_width = 0;
  int original_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int output_width = 0;
  int output_height = 0;
  double scale_x = 1;
  double scale_y = 1;
  double uniform_scale = 1;
  int pad_left = 0;
  int pad_top = 0;
  int pad_right = 0;
  int pad_bottom = 0;
  ResizeMode resize_mode = ResizeMode::None;
};

struct PreprocessResult {
  Tensor output;
  GeometryTransform geometry;
  // CUDA: kernel launch plus cudaDeviceSynchronize, excluding output allocation.
  // CPU: 0. The caller times the whole execute() call.
  double device_ms = 0;
};

// Sampling convention, shared by CPU and CUDA:
// source = (content_index + 0.5) * (source_size / resized_size) - 0.5
// Nearest uses floor(source + 0.5). Bilinear uses floor and replicates edges.
// Letterbox content size is round(source * uniform_scale), then clamped so it
// still fits the target. Leftover pixels are split with integer division:
// pad_left = pad / 2, pad_right = pad - pad_left. Stretch has no single ratio,
// so its uniform_scale is 0 and only scale_x and scale_y are meaningful.
GeometryTransform make_geometry(const PreprocessConfig& config, int source_width, int source_height);

class ImagePreprocessor {
 public:
  void configure(PreprocessConfig config);
  void validate(const ImageInput& image) const;
  PreprocessResult execute(const ImageInput& image, const DeviceContext& device) const;

  const PreprocessConfig& config() const { return config_; }

 private:
  PreprocessConfig config_{};
  bool configured_ = false;
};

// source_color and source_layout come from the image, not from the model.
// Manifest color is the model input color (the preprocess output). Missing
// interpolation, a letterbox without padding_value, or an unknown required
// field throws UnresolvedField. crop is a known manifest word and is rejected
// here as Unsupported. This does not invent YOLO defaults.
PreprocessConfig preprocess_config_from_manifest(const ModelManifest& manifest,
                                                 ColorOrder source_color,
                                                 ImageLayout source_layout);

}  // namespace vie
