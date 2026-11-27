#pragma once

// Optional OpenCV adapter. This header is intentionally *not* part of
// vie/vie.hpp: the core library never includes OpenCV, and only builds with
// VIE_ENABLE_OPENCV=ON define VIE_HAS_OPENCV and compile the adapter.
//
// The adapter does not decode anything. The caller already owns a decoded
// cv::Mat (from imread, a camera SDK, a frame grabber, ...) and hands its
// pixels to VIE through the same borrowed ImageInput the core already uses.

#include "vie/preprocess/image_input.hpp"

#if defined(VIE_HAS_OPENCV)

#include <opencv2/core.hpp>

namespace vie {

// Adapts one already-decoded cv::Mat to a borrowed ImageInput.
//
// Ownership. The returned ImageInput.borrows mat's pixel buffer (ImageInput is
// borrow-only by design); nothing is copied. The cv::Mat and its underlying
// allocation must stay alive for the whole synchronous
// InferenceEngine::infer()/infer_batch() call that consumes the ImageInput.
// For Runtime::submit(), the request deep-copies CPU pixels inside submit()
// and repacks strided rows (a ROI is not continuous) into a tightly packed
// buffer, so the cv::Mat may be released as soon as submit() returns.
//
// Geometry. layout is Hwc, row_stride_bytes is mat.step[0] (never
// width*elemSize). Consequently a non-continuous ROI such as
//   cv::Mat roi = frame(cv::Rect(20, 20, 300, 200));
// is adapted correctly: VIE preprocess reads every row through that stride and
// never interprets the padding bytes between rows.
//
// Type mapping. No depth conversion, no channel dropping, no color swap:
//   CV_8UC1  -> 1 channel, Gray,  Uint8
//   CV_8UC3  -> 3 channels, three_channel_color, Uint8
//   CV_32FC1 -> 1 channel, Gray,  Fp32
//   CV_32FC3 -> 3 channels, three_channel_color, Fp32
// Everything else - CV_8UC4/BGRA, CV_8UC2, CV_16U, CV_64F, ... - throws
// ErrorCode::Unsupported. ImageInput has no alpha or 16-bit path, and silently
// dropping alpha or converting depth would hide data loss from the caller.
//
// Color. three_channel_color defaults to Bgr because that is OpenCV's own
// convention for CV_8UC3/CV_32FC3. Pass ColorOrder::Rgb explicitly when the
// buffer was filled in RGB order. The adapter never swaps channels itself; the
// existing PreprocessConfig still decides the model's input color.
//
// Throws ErrorCode::InvalidArgument for an empty Mat and Unsupported for any
// type outside the table above.
ImageInput from_cv_mat(const cv::Mat& mat, ColorOrder three_channel_color = ColorOrder::Bgr);

}  // namespace vie

#endif  // VIE_HAS_OPENCV