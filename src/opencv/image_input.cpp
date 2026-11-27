#include "vie/opencv/image_input.hpp"

#if defined(VIE_HAS_OPENCV)

#include "vie/core/error.hpp"

#include <string>

namespace vie {
namespace {

[[noreturn]] void unsupported_type(const cv::Mat& mat) {
  throw VieError(ErrorCode::Unsupported,
                 "from_cv_mat: OpenCV type " + std::to_string(mat.type()) +
                     " is not supported; ImageInput accepts CV_8UC1, CV_8UC3, CV_32FC1 and "
                     "CV_32FC3 only. Convert depth/channels explicitly before adapting "
                     "(for example drop BGRA alpha with cv::cvtColor).");
}

}  // namespace

ImageInput from_cv_mat(const cv::Mat& mat, ColorOrder three_channel_color) {
  if (mat.empty() || mat.data == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "from_cv_mat: the input cv::Mat is empty");
  }
  if (mat.dims != 2) {
    throw VieError(ErrorCode::Unsupported,
                   "from_cv_mat: only 2-D cv::Mat images are supported, got " +
                       std::to_string(mat.dims) + " dimensions");
  }
  if (mat.cols <= 0 || mat.rows <= 0) {
    throw VieError(ErrorCode::InvalidArgument, "from_cv_mat: the input cv::Mat has no pixels");
  }

  ImageInput image;
  image.data = mat.data;
  image.width = mat.cols;
  image.height = mat.rows;
  image.layout = ImageLayout::Hwc;
  // mat.step[0] is the exact distance between rows. For a continuous image it
  // equals width * elemSize; for an ROI it is the parent's step, which is what
  // the preprocessor must use to walk the borrowed rows.
  image.row_stride_bytes = mat.step[0];
  image.device = DeviceType::Cpu;
  image.device_id = 0;
  image.ownership = ImageOwnership::Borrowed;

  switch (mat.type()) {
    case CV_8UC1:
      image.channels = 1;
      image.color = ColorOrder::Gray;
      image.dtype = DataType::Uint8;
      break;
    case CV_8UC3:
      if (three_channel_color == ColorOrder::Gray) {
        throw VieError(ErrorCode::InvalidArgument,
                       "from_cv_mat: a 3-channel cv::Mat cannot carry Gray color order");
      }
      image.channels = 3;
      image.color = three_channel_color;
      image.dtype = DataType::Uint8;
      break;
    case CV_32FC1:
      image.channels = 1;
      image.color = ColorOrder::Gray;
      image.dtype = DataType::Fp32;
      break;
    case CV_32FC3:
      if (three_channel_color == ColorOrder::Gray) {
        throw VieError(ErrorCode::InvalidArgument,
                       "from_cv_mat: a 3-channel cv::Mat cannot carry Gray color order");
      }
      image.channels = 3;
      image.color = three_channel_color;
      image.dtype = DataType::Fp32;
      break;
    default:
      unsupported_type(mat);
  }
  return image;
}

}  // namespace vie

#endif  // VIE_HAS_OPENCV