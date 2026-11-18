#include "vie/core/types.hpp"

#include <limits>

namespace vie {
namespace {

[[noreturn]] void bad_token(std::string_view kind, std::string_view text, std::string_view allowed) {
  throw VieError(ErrorCode::SchemaError, std::string(kind) + " has unsupported value '" +
                                              std::string(text) + "'. Expected one of: " +
                                              std::string(allowed));
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw VieError(ErrorCode::InvalidArgument, "tensor extent overflows size_t");
  }
  return lhs * rhs;
}

}  // namespace

DataType data_type_from_string(std::string_view text) {
  if (text == "unknown") return DataType::Unknown;
  if (text == "fp32") return DataType::Fp32;
  if (text == "fp16") return DataType::Fp16;
  if (text == "int8") return DataType::Int8;
  if (text == "int32") return DataType::Int32;
  if (text == "uint8") return DataType::Uint8;
  if (text == "bool") return DataType::Bool;
  if (text == "int64") return DataType::Int64;
  bad_token("dtype", text, "unknown, fp32, fp16, int8, int32, int64, uint8, bool");
}

DeviceType device_type_from_string(std::string_view text) {
  if (text == "cpu") return DeviceType::Cpu;
  if (text == "cuda") return DeviceType::Cuda;
  bad_token("device", text, "cpu, cuda");
}

Layout layout_from_string(std::string_view text) {
  if (text == "unknown") return Layout::Unknown;
  if (text == "nchw") return Layout::Nchw;
  if (text == "nhwc") return Layout::Nhwc;
  if (text == "bcn") return Layout::Bcn;
  bad_token("layout", text, "unknown, nchw, nhwc, bcn");
}

Certainty certainty_from_string(std::string_view text) {
  if (text == "observed") return Certainty::Observed;
  if (text == "metadata") return Certainty::Metadata;
  if (text == "inferred") return Certainty::Inferred;
  if (text == "user") return Certainty::User;
  if (text == "unresolved") return Certainty::Unresolved;
  bad_token("certainty", text, "observed, metadata, inferred, user, unresolved");
}

TaskType task_type_from_string(std::string_view text) {
  if (text == "unknown") return TaskType::Unknown;
  if (text == "detection") return TaskType::Detection;
  if (text == "segmentation") return TaskType::Segmentation;
  if (text == "pose") return TaskType::Pose;
  if (text == "obb") return TaskType::Obb;
  if (text == "classification") return TaskType::Classification;
  bad_token("task", text, "unknown, detection, segmentation, pose, obb, classification");
}

ModelFormat model_format_from_string(std::string_view text) {
  if (text == "unspecified") return ModelFormat::Unspecified;
  if (text == "onnx") return ModelFormat::Onnx;
  if (text == "tensorrt_engine") return ModelFormat::TensorRtEngine;
  if (text == "mock") return ModelFormat::Mock;
  bad_token("model_format", text, "unspecified, onnx, tensorrt_engine, mock");
}

BackendKind backend_kind_from_string(std::string_view text) {
  if (text == "auto") return BackendKind::Auto;
  if (text == "onnxruntime") return BackendKind::OnnxRuntime;
  if (text == "onnxruntime_cuda") return BackendKind::OnnxRuntimeCuda;
  if (text == "tensorrt") return BackendKind::TensorRT;
  if (text == "mock") return BackendKind::Mock;
  bad_token("backend", text, "auto, onnxruntime, onnxruntime_cuda, tensorrt, mock");
}

bool is_concrete_shape(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) return false;
  for (const std::int64_t dim : shape) {
    if (dim <= 0) return false;
  }
  return true;
}

std::vector<std::int64_t> row_major_strides(const std::vector<std::int64_t>& shape) {
  if (!is_concrete_shape(shape)) {
    throw VieError(ErrorCode::InvalidArgument, "row-major strides require a concrete shape");
  }
  std::vector<std::int64_t> strides(shape.size(), 1);
  std::int64_t extent = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = extent;
    if (shape[i] != 0 &&
        extent > std::numeric_limits<std::int64_t>::max() / shape[i]) {
      throw VieError(ErrorCode::InvalidArgument, "tensor shape overflows stride computation");
    }
    extent *= shape[i];
  }
  return strides;
}

std::size_t checked_numel(const std::vector<std::int64_t>& shape) {
  if (!is_concrete_shape(shape)) {
    throw VieError(ErrorCode::InvalidArgument, "numel requires a concrete shape");
  }
  std::size_t count = 1;
  for (const std::int64_t dim : shape) {
    count = checked_mul(count, static_cast<std::size_t>(dim));
  }
  return count;
}

bool is_known_color(std::string_view text) { return text == "rgb" || text == "bgr"; }

bool is_known_normalization(std::string_view text) {
  return text == "scale_0_1" || text == "scale_1_255" || text == "imagenet" || text == "none" ||
         text == "explicit";
}

bool is_known_resize(std::string_view text) {
  return text == "letterbox" || text == "stretch" || text == "crop" || text == "shortest_side";
}

bool is_known_box_format(std::string_view text) {
  return text == "xyxy" || text == "cxcywh" || text == "xywh" || text == "xyxy_normalized" ||
         text == "cxcywh_normalized";
}

bool task_needs_box_format(TaskType task) {
  return task == TaskType::Detection || task == TaskType::Segmentation || task == TaskType::Pose ||
         task == TaskType::Obb;
}

bool task_needs_num_classes(TaskType task) { return task != TaskType::Unknown; }

}  // namespace vie
