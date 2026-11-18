#pragma once

#include "vie/preprocess/preprocess_config.hpp"

#include "vie/core/device_context.hpp"
#include "vie/core/tensor.hpp"
#include "vie/model/manifest.hpp"

namespace vie {

// Model-independent geometry. scale_x and scale_y are the ratios after integer
// rounding of the resized content box, so a letterbox of a non-square image
// can have scale_x != scale_y. uniform_scale is the ratio chosen before that
// rounding. Letterbox stores the border in pad_*. Shortest-side center crop
// stores the window origin in crop_*; its pads stay 0 and resized_* is the
// image before the crop. Both modes use one origin:
//   origin_x = pad_left - crop_x
//   x_original = (x_model - origin_x) / scale_x
//   x_model = x_original * scale_x + origin_x
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
  int crop_x = 0;
  int crop_y = 0;
  ResizeMode resize_mode = ResizeMode::None;
};

inline int geometry_model_origin_x(const GeometryTransform& geometry) {
  return geometry.pad_left - geometry.crop_x;
}

inline int geometry_model_origin_y(const GeometryTransform& geometry) {
  return geometry.pad_top - geometry.crop_y;
}

struct PreprocessResult {
  Tensor output;
  GeometryTransform geometry;
  // CUDA: kernel launch plus cudaDeviceSynchronize, excluding output allocation.
  // CPU: 0. The caller times the whole execute() call.
  double device_ms = 0;
};

// Sampling convention, shared by CPU and CUDA:
// source = (content_index + 0.5) * (source_size / resized_size) - 0.5
// Nearest uses floor(source + 0.5), with halves rounding away from zero.
// Bilinear uses floor and replicates edges. This is the half-pixel convention.
// It is not the torchvision or PIL bilinear kernel.
// Letterbox content size is round(source * uniform_scale), then clamped so it
// still fits the target. Leftover pixels are split with integer division:
// pad_left = pad / 2, pad_right = pad - pad_left. Stretch has no single ratio,
// so its uniform_scale is 0 and only scale_x and scale_y are meaningful.
// Shortest-side resize pins the short edge to short_side. The long edge is
// round-half-away-from-zero of long * short_side / short, computed in integer
// arithmetic. Center crop then keeps crop_width by crop_height. An odd leftover
// gives the extra pixel to the right and to the bottom:
// crop_x = (resized_width - crop_width) / 2.
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
