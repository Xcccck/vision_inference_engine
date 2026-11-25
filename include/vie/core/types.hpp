#pragma once

#include "vie/core/error.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vie {

enum class DataType {
  Unknown,
  Fp32,
  Fp16,
  Int8,
  Int32,
  Uint8,
  Bool,
  Int64,
};

enum class DeviceType {
  Cpu,
  Cuda,
};

enum class Layout {
  Unknown,
  Nchw,
  Nhwc,
  Bcn,
};

// Where a manifest value came from. Unresolved is a real state, not a
// low confidence score.
enum class Certainty {
  Observed,
  Metadata,
  Inferred,
  User,
  Unresolved,
};

enum class TaskType {
  Unknown,
  Detection,
  Segmentation,
  Pose,
  Obb,
  Classification,
  // Dense per-pixel semantic segmentation. Distinct from Segmentation, which
  // is instance segmentation (per-object masks). The two never share a result
  // type, decoder, or recognizer.
  SemanticSegmentation,
};

enum class ModelFormat {
  Unspecified,
  Onnx,
  TensorRtEngine,
  Mock,
};

enum class BackendKind {
  Auto,
  OnnxRuntime,
  OnnxRuntimeCuda,
  TensorRT,
  Mock,
};

// Copied: the backend does not keep this storage.
// ViewUntilNextExecute: the storage is backend-owned and dies on the next
// execute() of that backend.
enum class TensorLifetime {
  Copied,
  ViewUntilNextExecute,
};

struct FieldProvenance {
  Certainty certainty = Certainty::Unresolved;
  std::vector<std::string> evidence;
};

struct TensorSpec {
  std::string name;
  std::vector<std::int64_t> shape;
  DataType dtype = DataType::Unknown;
  Layout layout = Layout::Unknown;
};

inline std::size_t dtype_size(DataType dtype) {
  switch (dtype) {
    case DataType::Fp32:
    case DataType::Int32:
      return 4;
    case DataType::Fp16:
      return 2;
    case DataType::Int8:
    case DataType::Uint8:
    case DataType::Bool:
      return 1;
    case DataType::Int64:
      return 8;
    case DataType::Unknown:
      break;
  }
  throw VieError(ErrorCode::InvalidArgument, "dtype size requested for an unknown dtype");
}

inline const char* to_string(DataType dtype) {
  switch (dtype) {
    case DataType::Unknown:
      return "unknown";
    case DataType::Fp32:
      return "fp32";
    case DataType::Fp16:
      return "fp16";
    case DataType::Int8:
      return "int8";
    case DataType::Int32:
      return "int32";
    case DataType::Uint8:
      return "uint8";
    case DataType::Bool:
      return "bool";
    case DataType::Int64:
      return "int64";
  }
  return "unknown";
}

inline const char* to_string(DeviceType device) {
  switch (device) {
    case DeviceType::Cpu:
      return "cpu";
    case DeviceType::Cuda:
      return "cuda";
  }
  return "cpu";
}

inline const char* to_string(Layout layout) {
  switch (layout) {
    case Layout::Unknown:
      return "unknown";
    case Layout::Nchw:
      return "nchw";
    case Layout::Nhwc:
      return "nhwc";
    case Layout::Bcn:
      return "bcn";
  }
  return "unknown";
}

inline const char* to_string(Certainty certainty) {
  switch (certainty) {
    case Certainty::Observed:
      return "observed";
    case Certainty::Metadata:
      return "metadata";
    case Certainty::Inferred:
      return "inferred";
    case Certainty::User:
      return "user";
    case Certainty::Unresolved:
      return "unresolved";
  }
  return "unresolved";
}

inline const char* to_string(TaskType task) {
  switch (task) {
    case TaskType::Unknown:
      return "unknown";
    case TaskType::Detection:
      return "detection";
    case TaskType::Segmentation:
      return "segmentation";
    case TaskType::Pose:
      return "pose";
    case TaskType::Obb:
      return "obb";
    case TaskType::Classification:
      return "classification";
    case TaskType::SemanticSegmentation:
      return "semantic_segmentation";
  }
  return "unknown";
}

inline const char* to_string(ModelFormat format) {
  switch (format) {
    case ModelFormat::Unspecified:
      return "unspecified";
    case ModelFormat::Onnx:
      return "onnx";
    case ModelFormat::TensorRtEngine:
      return "tensorrt_engine";
    case ModelFormat::Mock:
      return "mock";
  }
  return "unspecified";
}

inline const char* to_string(BackendKind kind) {
  switch (kind) {
    case BackendKind::Auto:
      return "auto";
    case BackendKind::OnnxRuntime:
      return "onnxruntime";
    case BackendKind::OnnxRuntimeCuda:
      return "onnxruntime_cuda";
    case BackendKind::TensorRT:
      return "tensorrt";
    case BackendKind::Mock:
      return "mock";
  }
  return "auto";
}

inline const char* to_string(TensorLifetime lifetime) {
  switch (lifetime) {
    case TensorLifetime::Copied:
      return "copied";
    case TensorLifetime::ViewUntilNextExecute:
      return "view_until_next_execute";
  }
  return "copied";
}

DataType data_type_from_string(std::string_view text);
DeviceType device_type_from_string(std::string_view text);
Layout layout_from_string(std::string_view text);
Certainty certainty_from_string(std::string_view text);
TaskType task_type_from_string(std::string_view text);
ModelFormat model_format_from_string(std::string_view text);
BackendKind backend_kind_from_string(std::string_view text);

bool is_concrete_shape(const std::vector<std::int64_t>& shape);
std::vector<std::int64_t> row_major_strides(const std::vector<std::int64_t>& shape);
std::size_t checked_numel(const std::vector<std::int64_t>& shape);

bool is_known_color(std::string_view text);
bool is_known_normalization(std::string_view text);
bool is_known_resize(std::string_view text);
bool is_known_box_format(std::string_view text);

bool task_needs_box_format(TaskType task);
bool task_needs_num_classes(TaskType task);

}  // namespace vie
