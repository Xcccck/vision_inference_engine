#include "vie/preprocess/preprocess.hpp"

#include "preprocess_cpu.hpp"

#if defined(VIE_HAS_CUDA)
#include "preprocess_cuda.hpp"
#endif

#include <cmath>
#include <sstream>

namespace vie {
namespace {

std::string device_label(DeviceType device, int device_id) {
  std::ostringstream stream;
  stream << to_string(device) << ':' << device_id;
  return stream.str();
}

void require_resolved(bool ready, const std::string& path) {
  if (!ready) {
    throw VieError(ErrorCode::UnresolvedField, path + " is unresolved");
  }
}

}  // namespace

const char* to_string(ColorOrder color) {
  switch (color) {
    case ColorOrder::Rgb:
      return "rgb";
    case ColorOrder::Bgr:
      return "bgr";
    case ColorOrder::Gray:
      return "gray";
  }
  return "unknown";
}

const char* to_string(ImageLayout layout) {
  switch (layout) {
    case ImageLayout::Hwc:
      return "hwc";
    case ImageLayout::Chw:
      return "chw";
  }
  return "unknown";
}

const char* to_string(ResizeMode mode) {
  switch (mode) {
    case ResizeMode::None:
      return "none";
    case ResizeMode::Stretch:
      return "stretch";
    case ResizeMode::Letterbox:
      return "letterbox";
  }
  return "unknown";
}

const char* to_string(Interpolation interpolation) {
  switch (interpolation) {
    case Interpolation::Nearest:
      return "nearest";
    case Interpolation::Bilinear:
      return "bilinear";
  }
  return "unknown";
}

void validate_preprocess_config(const PreprocessConfig& config) {
  if (config.target_width <= 0 || config.target_height <= 0) {
    throw VieError(ErrorCode::InvalidArgument, "target width and height must be positive");
  }
  if (config.output_color == ColorOrder::Gray) {
    throw VieError(ErrorCode::InvalidArgument, "output color must be rgb or bgr");
  }
  if (config.output_dtype != DataType::Uint8 && config.output_dtype != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument, "output dtype must be uint8 or fp32");
  }
  if (!std::isfinite(config.padding_value)) {
    throw VieError(ErrorCode::InvalidArgument, "padding_value must be finite");
  }
  if (!config.normalize_enabled) return;
  if (!std::isfinite(config.scale)) {
    throw VieError(ErrorCode::InvalidArgument, "normalize scale must be finite");
  }
  if (config.mean.size() != 3 || config.std.size() != 3) {
    throw VieError(ErrorCode::InvalidArgument, "normalize mean and std must each have 3 values");
  }
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(config.mean[static_cast<std::size_t>(i)]) ||
        !std::isfinite(config.std[static_cast<std::size_t>(i)])) {
      throw VieError(ErrorCode::InvalidArgument, "normalize mean and std must be finite");
    }
    if (config.std[static_cast<std::size_t>(i)] == 0.f) {
      throw VieError(ErrorCode::InvalidArgument, "normalize std must be non-zero");
    }
  }
}

GeometryTransform make_geometry(const PreprocessConfig& config, int source_width, int source_height) {
  if (source_width <= 0 || source_height <= 0) {
    throw VieError(ErrorCode::InvalidArgument, "image width and height must be positive");
  }
  GeometryTransform geometry;
  geometry.original_width = source_width;
  geometry.original_height = source_height;
  geometry.output_width = config.target_width;
  geometry.output_height = config.target_height;
  geometry.resize_mode = config.resize_mode;

  if (config.resize_mode == ResizeMode::None) {
    if (config.target_width != source_width || config.target_height != source_height) {
      throw VieError(ErrorCode::InvalidArgument,
                     "resize mode none requires the target size to equal the image size");
    }
    geometry.resized_width = source_width;
    geometry.resized_height = source_height;
    geometry.scale_x = 1;
    geometry.scale_y = 1;
    geometry.uniform_scale = 1;
    return geometry;
  }

  if (config.resize_mode == ResizeMode::Stretch) {
    geometry.resized_width = config.target_width;
    geometry.resized_height = config.target_height;
    geometry.scale_x = static_cast<double>(config.target_width) / static_cast<double>(source_width);
    geometry.scale_y =
        static_cast<double>(config.target_height) / static_cast<double>(source_height);
    geometry.uniform_scale = 0;
    return geometry;
  }

  const double scale_w =
      static_cast<double>(config.target_width) / static_cast<double>(source_width);
  const double scale_h =
      static_cast<double>(config.target_height) / static_cast<double>(source_height);
  const double uniform = scale_w < scale_h ? scale_w : scale_h;
  int resized_w = static_cast<int>(std::lround(static_cast<double>(source_width) * uniform));
  int resized_h = static_cast<int>(std::lround(static_cast<double>(source_height) * uniform));
  if (resized_w < 1) resized_w = 1;
  if (resized_h < 1) resized_h = 1;
  if (resized_w > config.target_width) resized_w = config.target_width;
  if (resized_h > config.target_height) resized_h = config.target_height;
  geometry.resized_width = resized_w;
  geometry.resized_height = resized_h;
  geometry.uniform_scale = uniform;
  geometry.scale_x = static_cast<double>(resized_w) / static_cast<double>(source_width);
  geometry.scale_y = static_cast<double>(resized_h) / static_cast<double>(source_height);
  const int pad_w = config.target_width - resized_w;
  const int pad_h = config.target_height - resized_h;
  geometry.pad_left = pad_w / 2;
  geometry.pad_right = pad_w - geometry.pad_left;
  geometry.pad_top = pad_h / 2;
  geometry.pad_bottom = pad_h - geometry.pad_top;
  return geometry;
}

void ImagePreprocessor::configure(PreprocessConfig config) {
  validate_preprocess_config(config);
  config_ = std::move(config);
  configured_ = true;
}

void ImagePreprocessor::validate(const ImageInput& image) const {
  if (!configured_) {
    throw VieError(ErrorCode::InvalidArgument, "preprocessor is not configured");
  }
  if (image.data == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "image data is null");
  }
  if (image.ownership != ImageOwnership::Borrowed) {
    throw VieError(ErrorCode::InvalidArgument, "image ownership is not supported");
  }
  if (image.width <= 0 || image.height <= 0) {
    throw VieError(ErrorCode::InvalidArgument, "image width and height must be positive");
  }
  if (image.dtype != DataType::Uint8 && image.dtype != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument, "image dtype must be uint8 or fp32");
  }
  if (image.device != DeviceType::Cpu && image.device != DeviceType::Cuda) {
    throw VieError(ErrorCode::InvalidArgument, "image device must be cpu or cuda");
  }
  if (image.device_id < 0) {
    throw VieError(ErrorCode::InvalidArgument, "image device id must be >= 0");
  }
  const int expected_channels = image.color == ColorOrder::Gray ? 1 : 3;
  if (image.channels != expected_channels) {
    throw VieError(ErrorCode::InvalidArgument,
                   std::string("image channels do not match color ") + to_string(image.color));
  }
  if (image.color != config_.input_color || image.layout != config_.input_layout) {
    throw VieError(ErrorCode::InvalidArgument,
                   "image color or layout does not match the preprocess config");
  }
  const std::size_t element = image.dtype == DataType::Uint8 ? 1 : 4;
  const std::size_t packed =
      image.layout == ImageLayout::Hwc
          ? static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.channels) * element
          : static_cast<std::size_t>(image.width) * element;
  if (image.row_stride_bytes != 0 &&
      (image.row_stride_bytes < packed || image.row_stride_bytes % element != 0)) {
    throw VieError(ErrorCode::InvalidArgument, "image row_stride_bytes is smaller than one row or is misaligned");
  }
}

PreprocessResult ImagePreprocessor::execute(const ImageInput& image, const DeviceContext& device) const {
  validate(image);
  if (image.device != device.device() || image.device_id != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "image device " + device_label(image.device, image.device_id) +
                       " does not match execution device " +
                       device_label(device.device(), device.device_id()));
  }
  const GeometryTransform geometry = make_geometry(config_, image.width, image.height);
  if (device.device() == DeviceType::Cpu) {
    if (image.device != DeviceType::Cpu) {
      throw VieError(ErrorCode::InvalidArgument, "CPU preprocess cannot read a CUDA image");
    }
    return preprocess_cpu(config_, image, device, geometry);
  }
  if (device.device() == DeviceType::Cuda) {
#if defined(VIE_HAS_CUDA)
    return preprocess_cuda(config_, image, device, geometry);
#else
    throw VieError(ErrorCode::Unsupported, "CUDA preprocess is not available in this build");
#endif
  }
  throw VieError(ErrorCode::InvalidArgument, "execution device must be cpu or cuda");
}

PreprocessConfig preprocess_config_from_manifest(const ModelManifest& manifest, ColorOrder source_color,
                                                 ImageLayout source_layout) {
  require_resolved(!manifest.input.color.empty(), "input.color");
  require_resolved(!manifest.input.normalization.empty(), "input.normalization");
  require_resolved(!manifest.input.resize.empty(), "input.resize");
  require_resolved(!manifest.input.interpolation.empty(), "input.interpolation");
  require_resolved(manifest.input.dtype == DataType::Fp32 || manifest.input.dtype == DataType::Uint8,
                   "input.dtype");
  require_resolved(manifest.input.layout == Layout::Nchw || manifest.input.layout == Layout::Nhwc,
                   "input.layout");
  if (!is_concrete_shape(manifest.input.shape) || manifest.input.shape.size() != 4) {
    throw VieError(ErrorCode::UnresolvedField, "input.shape is unresolved");
  }

  PreprocessConfig config;
  config.input_color = source_color;
  config.input_layout = source_layout;
  if (manifest.input.color == "rgb") {
    config.output_color = ColorOrder::Rgb;
  } else if (manifest.input.color == "bgr") {
    config.output_color = ColorOrder::Bgr;
  } else {
    throw VieError(ErrorCode::InvalidArgument, "input.color is not rgb or bgr");
  }
  config.output_layout =
      manifest.input.layout == Layout::Nchw ? ImageLayout::Chw : ImageLayout::Hwc;
  config.output_dtype = manifest.input.dtype;
  config.interpolation =
      manifest.input.interpolation == "nearest" ? Interpolation::Nearest : Interpolation::Bilinear;
  if (manifest.input.interpolation != "nearest" && manifest.input.interpolation != "bilinear") {
    throw VieError(ErrorCode::InvalidArgument, "input.interpolation is not nearest or bilinear");
  }

  if (manifest.input.layout == Layout::Nchw) {
    if (manifest.input.shape[0] != 1 || manifest.input.shape[1] != 3) {
      throw VieError(ErrorCode::InvalidArgument, "NCHW preprocess expects shape [1, 3, H, W]");
    }
    config.target_height = static_cast<int>(manifest.input.shape[2]);
    config.target_width = static_cast<int>(manifest.input.shape[3]);
  } else {
    if (manifest.input.shape[0] != 1 || manifest.input.shape[3] != 3) {
      throw VieError(ErrorCode::InvalidArgument, "NHWC preprocess expects shape [1, H, W, 3]");
    }
    config.target_height = static_cast<int>(manifest.input.shape[1]);
    config.target_width = static_cast<int>(manifest.input.shape[2]);
  }

  if (manifest.input.resize == "letterbox") {
    config.resize_mode = ResizeMode::Letterbox;
    require_resolved(manifest.input.padding_value.has_value(), "input.padding_value");
    config.padding_value = *manifest.input.padding_value;
  } else if (manifest.input.resize == "stretch") {
    config.resize_mode = ResizeMode::Stretch;
  } else if (manifest.input.resize == "crop") {
    throw VieError(ErrorCode::Unsupported, "resize mode crop is not implemented");
  } else {
    throw VieError(ErrorCode::InvalidArgument, "input.resize is not letterbox or stretch");
  }

  if (manifest.input.normalization == "none" || manifest.input.normalization == "scale_1_255") {
    config.normalize_enabled = false;
  } else if (manifest.input.normalization == "scale_0_1") {
    config.normalize_enabled = true;
    config.scale = 1.f / 255.f;
    config.mean = {0.f, 0.f, 0.f};
    config.std = {1.f, 1.f, 1.f};
  } else if (manifest.input.normalization == "imagenet") {
    config.normalize_enabled = true;
    config.scale = 1.f / 255.f;
    config.mean = {0.485f, 0.456f, 0.406f};
    config.std = {0.229f, 0.224f, 0.225f};
  } else {
    throw VieError(ErrorCode::UnresolvedField, "input.normalization is unresolved");
  }
  validate_preprocess_config(config);
  return config;
}

}  // namespace vie
