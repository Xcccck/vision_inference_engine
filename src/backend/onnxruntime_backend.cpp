#include "vie/backend/onnxruntime_backend.hpp"

#include "onnxruntime_cxx_api.h"

#include "cpu_provider_factory.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <utility>

namespace vie {
namespace {

using Clock = std::chrono::steady_clock;

struct Endpoint {
  std::string name;
  DataType dtype = DataType::Unknown;
  std::vector<std::int64_t> shape;
};

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) out << ",";
    out << shape[index];
  }
  out << "]";
  return out.str();
}

[[noreturn]] void fail_load(ErrorCode code, const std::string& message) {
  throw VieError(code, "onnxruntime load failed: " + message);
}

[[noreturn]] void fail_input(const std::string& message) {
  throw VieError(ErrorCode::InvalidArgument, "input validation failed: " + message);
}

[[noreturn]] void fail_inference(const std::string& message) {
  throw VieError(ErrorCode::Internal, "onnxruntime inference failed: " + message);
}

[[noreturn]] void fail_output(const std::string& message) {
  throw VieError(ErrorCode::Internal, "output conversion failed: " + message);
}

DataType dtype_from_ort(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::Fp32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::Fp16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DataType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::Uint8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DataType::Bool;
    default:
      return DataType::Unknown;
  }
}

ONNXTensorElementDataType ort_from_dtype(DataType dtype) {
  switch (dtype) {
    case DataType::Fp32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::Fp16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::Int8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DataType::Int32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::Int64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::Uint8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DataType::Bool:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case DataType::Unknown:
      break;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

GraphOptimizationLevel ort_optimization(OnnxRuntimeOptimization level) {
  switch (level) {
    case OnnxRuntimeOptimization::Disabled:
      return ORT_DISABLE_ALL;
    case OnnxRuntimeOptimization::Basic:
      return ORT_ENABLE_BASIC;
    case OnnxRuntimeOptimization::Extended:
      return ORT_ENABLE_EXTENDED;
    case OnnxRuntimeOptimization::All:
      return ORT_ENABLE_ALL;
  }
  return ORT_ENABLE_ALL;
}

const Endpoint* find_endpoint(const std::vector<Endpoint>& endpoints, const std::string& name) {
  for (const Endpoint& endpoint : endpoints) {
    if (endpoint.name == name) return &endpoint;
  }
  return nullptr;
}

const TensorSpec* find_spec(const BackendConfig& config, const std::string& name, bool input) {
  if (input) {
    if (config.input.name == name) return &config.input;
    return nullptr;
  }
  for (const TensorSpec& spec : config.outputs) {
    if (spec.name == name) return &spec;
  }
  return nullptr;
}

void check_model_shape(const std::string& name, const Tensor& tensor, const Endpoint& model) {
  if (tensor.shape().size() != model.shape.size()) {
    fail_input("input tensor '" + name + "' rank " + std::to_string(tensor.shape().size()) +
               " does not match model rank " + std::to_string(model.shape.size()));
  }
  for (std::size_t axis = 0; axis < model.shape.size(); ++axis) {
    const std::int64_t expected = model.shape[axis];
    const std::int64_t actual = tensor.shape()[axis];
    if (expected <= 0) {
      if (actual <= 0) {
        fail_input("input tensor '" + name + "' has a non-positive dynamic dimension");
      }
      continue;
    }
    if (actual != expected) {
      fail_input("input tensor '" + name + "' shape " + shape_text(tensor.shape()) +
                 " does not match model shape " + shape_text(model.shape));
    }
  }
}

void check_profile(const std::string& name, const Tensor& tensor, const TensorSpec* spec) {
  if (spec == nullptr) return;
  if (spec->dtype != DataType::Unknown && tensor.dtype() != spec->dtype) {
    fail_input("input tensor '" + name + "' dtype " + to_string(tensor.dtype()) +
               " does not match the declared profile dtype " + to_string(spec->dtype));
  }
  if (is_concrete_shape(spec->shape) && tensor.shape() != spec->shape) {
    fail_input("input tensor '" + name + "' shape " + shape_text(tensor.shape()) +
               " does not match the declared profile " + shape_text(spec->shape));
  }
}

Endpoint read_endpoint(const Ort::Session& session, bool input, std::size_t index,
                       Ort::AllocatorWithDefaultOptions& allocator) {
  Ort::AllocatedStringPtr allocated =
      input ? session.GetInputNameAllocated(index, allocator)
            : session.GetOutputNameAllocated(index, allocator);
  if (allocated.get() == nullptr) {
    fail_load(ErrorCode::SchemaError, "the model returned an empty tensor name");
  }
  Ort::TypeInfo type_info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    fail_load(ErrorCode::Unsupported,
              std::string(input ? "input" : "output") + " '" + allocated.get() +
                  "' is not a tensor");
  }
  const Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
  Endpoint endpoint;
  endpoint.name = allocated.get();
  endpoint.dtype = dtype_from_ort(tensor_info.GetElementType());
  endpoint.shape = tensor_info.GetShape();
  if (endpoint.dtype == DataType::Unknown) {
    fail_load(ErrorCode::Unsupported, std::string(input ? "input" : "output") + " '" +
                                          endpoint.name +
                                          "' has a dtype that this build cannot represent");
  }
  return endpoint;
}

}  // namespace

struct OnnxRuntimeBackend::Impl {
  int intra_op_threads = 0;
  int inter_op_threads = 0;
  OnnxRuntimeOptimization optimization = OnnxRuntimeOptimization::All;
  int log_severity = 2;
  bool prepared = false;
  OnnxRuntimeTiming timing;
  BackendConfig config;
  Ort::Env env{nullptr};
  Ort::SessionOptions options{nullptr};
  std::unique_ptr<Ort::Session> session;
  std::vector<Endpoint> inputs;
  std::vector<Endpoint> outputs;

  void release_session() {
    prepared = false;
    session.reset();
    inputs.clear();
    outputs.clear();
  }
};

OnnxRuntimeBackend::OnnxRuntimeBackend() : impl_(std::make_unique<Impl>()) {}

OnnxRuntimeBackend::~OnnxRuntimeBackend() = default;

void OnnxRuntimeBackend::set_intra_op_threads(int threads) {
  if (threads < 0) {
    throw VieError(ErrorCode::InvalidArgument, "intra-op thread count must be >= 0");
  }
  impl_->intra_op_threads = threads;
}

void OnnxRuntimeBackend::set_inter_op_threads(int threads) {
  if (threads < 0) {
    throw VieError(ErrorCode::InvalidArgument, "inter-op thread count must be >= 0");
  }
  impl_->inter_op_threads = threads;
}

void OnnxRuntimeBackend::set_graph_optimization(OnnxRuntimeOptimization level) {
  impl_->optimization = level;
}

void OnnxRuntimeBackend::set_log_severity(int severity) {
  if (severity < 0 || severity > 4) {
    throw VieError(ErrorCode::InvalidArgument,
                   "log severity must be in the inclusive range [0, 4]");
  }
  impl_->log_severity = severity;
}

void OnnxRuntimeBackend::prepare(const BackendConfig& config) {
  if (config.model_format != ModelFormat::Onnx) {
    throw VieError(ErrorCode::Unsupported,
                   "the ONNX Runtime backend only executes model format 'onnx'");
  }
  if (config.device != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "the ONNX Runtime CPU backend only runs on CPU");
  }
  if (config.device_id < 0) {
    fail_load(ErrorCode::InvalidArgument, "device id must be >= 0");
  }
  if (config.model_path.empty()) {
    fail_load(ErrorCode::InvalidArgument, "model path is empty");
  }
  std::error_code file_error;
  const bool regular = std::filesystem::is_regular_file(config.model_path, file_error);
  if (file_error || !regular) {
    fail_load(ErrorCode::InvalidArgument, "model file does not exist: " + config.model_path);
  }

  impl_->release_session();
  const Clock::time_point started = Clock::now();
  try {
    impl_->env = Ort::Env(static_cast<OrtLoggingLevel>(impl_->log_severity), "vie.onnxruntime");
    impl_->options = Ort::SessionOptions{};
    if (impl_->intra_op_threads > 0) {
      impl_->options.SetIntraOpNumThreads(impl_->intra_op_threads);
    }
    if (impl_->inter_op_threads > 0) {
      impl_->options.SetInterOpNumThreads(impl_->inter_op_threads);
    }
    impl_->options.SetGraphOptimizationLevel(ort_optimization(impl_->optimization));
    impl_->options.SetLogSeverityLevel(impl_->log_severity);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(impl_->options, 1));
    impl_->session =
        std::make_unique<Ort::Session>(impl_->env, config.model_path.c_str(), impl_->options);

    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t input_count = impl_->session->GetInputCount();
    const std::size_t output_count = impl_->session->GetOutputCount();
    if (input_count == 0 || output_count == 0) {
      fail_load(ErrorCode::SchemaError, "the model has no tensor inputs or no tensor outputs");
    }
    impl_->inputs.reserve(input_count);
    impl_->outputs.reserve(output_count);
    for (std::size_t index = 0; index < input_count; ++index) {
      impl_->inputs.push_back(read_endpoint(*impl_->session, true, index, allocator));
    }
    for (std::size_t index = 0; index < output_count; ++index) {
      impl_->outputs.push_back(read_endpoint(*impl_->session, false, index, allocator));
    }

    std::vector<std::string> declared;
    for (const TensorSpec& spec : config.outputs) {
      if (spec.name.empty()) {
        fail_load(ErrorCode::InvalidArgument, "a declared output has an empty name");
      }
      if (find_endpoint(impl_->outputs, spec.name) == nullptr) {
        fail_load(ErrorCode::InvalidArgument,
                  "declared output '" + spec.name + "' is not a model output");
      }
      for (const std::string& previous : declared) {
        if (previous == spec.name) {
          fail_load(ErrorCode::InvalidArgument,
                    "declared output '" + spec.name + "' is duplicated");
        }
      }
      declared.push_back(spec.name);
    }

    impl_->config = config;
    impl_->prepared = true;
    impl_->timing = {};
    impl_->timing.load_ms = elapsed_ms(started);
  } catch (const VieError&) {
    impl_->release_session();
    throw;
  } catch (const Ort::Exception& error) {
    impl_->release_session();
    const OrtErrorCode code = error.GetOrtErrorCode();
    if (code == ORT_NO_SUCHFILE) {
      fail_load(ErrorCode::InvalidArgument, error.what());
    }
    if (code == ORT_INVALID_PROTOBUF || code == ORT_INVALID_GRAPH || code == ORT_NO_MODEL ||
        code == ORT_FAIL || code == ORT_INVALID_ARGUMENT) {
      fail_load(ErrorCode::SchemaError, error.what());
    }
    fail_load(ErrorCode::Internal, error.what());
  } catch (const std::exception& error) {
    impl_->release_session();
    fail_load(ErrorCode::Internal, error.what());
  }
}

BackendOutput OnnxRuntimeBackend::execute(const BackendRequest& request) {
  if (!impl_->prepared || impl_->session == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "onnxruntime backend is not prepared");
  }

  const double load_ms = impl_->timing.load_ms;
  impl_->timing = {};
  impl_->timing.load_ms = load_ms;

  std::vector<std::string> missing;
  for (const Endpoint& endpoint : impl_->inputs) {
    if (request.inputs.find(endpoint.name) == request.inputs.end()) missing.push_back(endpoint.name);
  }
  std::vector<std::string> unexpected;
  for (const auto& entry : request.inputs) {
    if (find_endpoint(impl_->inputs, entry.first) == nullptr) unexpected.push_back(entry.first);
  }
  if (!missing.empty() && !unexpected.empty()) {
    fail_input("missing input tensor '" + missing.front() + "'; unexpected input tensor '" +
               unexpected.front() + "'");
  }
  if (!missing.empty()) fail_input("missing input tensor '" + missing.front() + "'");
  if (!unexpected.empty()) fail_input("unexpected input tensor '" + unexpected.front() + "'");

  std::vector<Tensor> packed;
  packed.reserve(impl_->inputs.size());
  std::vector<const Tensor*> bound;
  bound.reserve(impl_->inputs.size());
  for (const Endpoint& endpoint : impl_->inputs) {
    const Tensor& input = request.inputs.at(endpoint.name);
    if (input.empty()) fail_input("input tensor '" + endpoint.name + "' has no storage");
    if (input.device() != DeviceType::Cpu) {
      fail_input("input tensor '" + endpoint.name + "' is not a CPU tensor");
    }
    if (input.dtype() != endpoint.dtype) {
      fail_input("input tensor '" + endpoint.name + "' dtype " + to_string(input.dtype()) +
                 " does not match model dtype " + to_string(endpoint.dtype));
    }
    check_model_shape(endpoint.name, input, endpoint);
    check_profile(endpoint.name, input, find_spec(impl_->config, endpoint.name, true));
    if (input.contiguous()) {
      bound.push_back(&input);
      continue;
    }
    const Clock::time_point copy_started = Clock::now();
    try {
      packed.push_back(input.clone());
    } catch (const VieError& error) {
      fail_input(std::string("could not pack non-contiguous input '") + endpoint.name + "': " +
                 error.what());
    }
    impl_->timing.last_input_copy_ms += elapsed_ms(copy_started);
    impl_->timing.input_was_copied = true;
    bound.push_back(&packed.back());
  }

  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> values;
  std::vector<const char*> input_names;
  values.reserve(bound.size());
  input_names.reserve(bound.size());
  for (std::size_t index = 0; index < bound.size(); ++index) {
    const Tensor& tensor = *bound[index];
    input_names.push_back(impl_->inputs[index].name.c_str());
    values.push_back(Ort::Value::CreateTensor(
        memory, const_cast<void*>(static_cast<const void*>(tensor.data())), tensor.nbytes(),
        tensor.shape().data(), tensor.shape().size(), ort_from_dtype(tensor.dtype())));
  }

  std::vector<const char*> output_names;
  output_names.reserve(impl_->outputs.size());
  for (const Endpoint& endpoint : impl_->outputs) output_names.push_back(endpoint.name.c_str());

  std::vector<Ort::Value> results;
  const Clock::time_point run_started = Clock::now();
  try {
    Ort::RunOptions run_options;
    results = impl_->session->Run(run_options, input_names.data(), values.data(), values.size(),
                                  output_names.data(), output_names.size());
  } catch (const Ort::Exception& error) {
    fail_inference(error.what());
  } catch (const std::exception& error) {
    fail_inference(error.what());
  }
  impl_->timing.last_execute_ms = elapsed_ms(run_started);

  if (results.size() != impl_->outputs.size()) {
    fail_output("runtime returned " + std::to_string(results.size()) + " tensor(s), expected " +
                std::to_string(impl_->outputs.size()));
  }

  const DeviceContext host(DeviceType::Cpu, impl_->config.device_id);
  BackendOutput output;
  const Clock::time_point copy_started = Clock::now();
  try {
    for (std::size_t index = 0; index < results.size(); ++index) {
      const std::string& name = impl_->outputs[index].name;
      Ort::Value& value = results[index];
      if (!value.IsTensor()) fail_output("output '" + name + "' is not a tensor");
      const Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
      const DataType dtype = dtype_from_ort(info.GetElementType());
      if (dtype == DataType::Unknown) {
        fail_output("output '" + name + "' has a dtype that this build cannot represent");
      }
      std::vector<std::int64_t> shape = info.GetShape();
      for (const std::int64_t dim : shape) {
        if (dim <= 0) {
          fail_output("output '" + name + "' has a non-positive dimension in shape " +
                      shape_text(shape));
        }
      }
      const TensorSpec* spec = find_spec(impl_->config, name, false);
      if (spec != nullptr && spec->dtype != DataType::Unknown && spec->dtype != dtype) {
        fail_output("output '" + name + "' dtype " + to_string(dtype) +
                    " does not match the declared profile dtype " + to_string(spec->dtype));
      }
      if (spec != nullptr && is_concrete_shape(spec->shape) && spec->shape != shape) {
        fail_output("output '" + name + "' shape " + shape_text(shape) +
                    " does not match the declared profile " + shape_text(spec->shape));
      }
      const void* raw = value.GetTensorRawData();
      if (raw == nullptr) fail_output("output '" + name + "' has no data");
      Tensor tensor = Tensor::zeros(host, dtype, shape);
      std::memcpy(tensor.data(), raw, tensor.nbytes());
      output.tensors.emplace(name, NamedTensor{std::move(tensor), TensorLifetime::Copied});
    }
  } catch (const VieError&) {
    throw;
  } catch (const Ort::Exception& error) {
    fail_output(error.what());
  } catch (const std::exception& error) {
    fail_output(error.what());
  }
  impl_->timing.last_output_copy_ms = elapsed_ms(copy_started);
  return output;
}

BackendInfo OnnxRuntimeBackend::info() const {
  BackendInfo info;
  info.name = "onnxruntime";
  info.kind = BackendKind::OnnxRuntime;
  info.device = DeviceType::Cpu;
  info.supports_cuda = false;
  return info;
}

const OnnxRuntimeTiming& OnnxRuntimeBackend::timing() const { return impl_->timing; }

}  // namespace vie
