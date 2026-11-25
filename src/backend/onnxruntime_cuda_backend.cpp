#include "vie/backend/onnxruntime_cuda_backend.hpp"

#include "../core/cuda_memory.hpp"

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

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
  throw VieError(code, "onnxruntime cuda load failed: " + message);
}

[[noreturn]] void fail_input(const std::string& message) {
  throw VieError(ErrorCode::InvalidArgument, "input validation failed: " + message);
}

[[noreturn]] void fail_inference(const std::string& message) {
  throw VieError(ErrorCode::Internal, "onnxruntime cuda inference failed: " + message);
}

[[noreturn]] void fail_output(const std::string& message) {
  throw VieError(ErrorCode::Internal, "output conversion failed: " + message);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  throw VieError(ErrorCode::Internal, std::string(operation) + " failed: " + cudaGetErrorString(status));
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
              std::string(input ? "input" : "output") + " '" + allocated.get() + "' is not a tensor");
  }
  const Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
  Endpoint endpoint;
  endpoint.name = allocated.get();
  endpoint.dtype = dtype_from_ort(tensor_info.GetElementType());
  endpoint.shape = tensor_info.GetShape();
  if (endpoint.dtype == DataType::Unknown) {
    fail_load(ErrorCode::Unsupported, std::string(input ? "input" : "output") + " '" + endpoint.name +
                                          "' has a dtype that this build cannot represent");
  }
  return endpoint;
}

void require_cuda_memory(const Ort::ConstMemoryInfo& memory, int device_id, const std::string& what) {
  if (memory.GetDeviceType() != OrtMemoryInfoDeviceType_GPU || memory.GetDeviceId() != device_id) {
    throw VieError(ErrorCode::Internal, what + " is not CUDA:" + std::to_string(device_id) +
                                            " memory (allocator '" + memory.GetAllocatorName() +
                                            "', device type " + std::to_string(static_cast<int>(memory.GetDeviceType())) +
                                            ", device " + std::to_string(memory.GetDeviceId()) + ")");
  }
}

}  // namespace

struct OnnxRuntimeCudaBackend::Impl {
  int intra_op_threads = 0;
  int inter_op_threads = 0;
  OnnxRuntimeOptimization optimization = OnnxRuntimeOptimization::All;
  int log_severity = 2;
  bool cpu_ep_fallback = false;
  bool profiling_enabled = false;
  std::string profiling_prefix;
  bool prepared = false;
  bool device_ready = false;
  int device_id = 0;
  OnnxRuntimeCudaTiming timing;
  BackendConfig config;
  DeviceContext device;
  std::mutex log_mu;
  std::string session_log;
  Ort::Env env{nullptr};
  Ort::SessionOptions options{nullptr};
  std::unique_ptr<Ort::Session> session;
  std::vector<Endpoint> inputs;
  std::vector<Endpoint> outputs;

  void quiesce() noexcept {
    if (!device_ready) return;
    cudaSetDevice(device_id);
    cudaDeviceSynchronize();
  }

  void release_session() noexcept {
    quiesce();
    prepared = false;
    session.reset();
    inputs.clear();
    outputs.clear();
  }

  void append_log(const char* message) {
    if (message == nullptr) return;
    std::lock_guard<std::mutex> guard(log_mu);
    constexpr std::size_t kLimit = 2 * 1024 * 1024;
    if (session_log.size() >= kLimit) return;
    session_log.append(message);
    session_log.push_back('\n');
  }
};

OnnxRuntimeCudaBackend::OnnxRuntimeCudaBackend() : impl_(std::make_unique<Impl>()) {}

OnnxRuntimeCudaBackend::~OnnxRuntimeCudaBackend() {
  if (impl_ != nullptr) impl_->release_session();
}

void OnnxRuntimeCudaBackend::set_intra_op_threads(int threads) {
  if (threads < 0) {
    throw VieError(ErrorCode::InvalidArgument, "intra-op thread count must be >= 0");
  }
  impl_->intra_op_threads = threads;
}

void OnnxRuntimeCudaBackend::set_inter_op_threads(int threads) {
  if (threads < 0) {
    throw VieError(ErrorCode::InvalidArgument, "inter-op thread count must be >= 0");
  }
  impl_->inter_op_threads = threads;
}

void OnnxRuntimeCudaBackend::set_graph_optimization(OnnxRuntimeOptimization level) {
  impl_->optimization = level;
}

void OnnxRuntimeCudaBackend::set_log_severity(int severity) {
  if (severity < 0 || severity > 4) {
    throw VieError(ErrorCode::InvalidArgument, "log severity must be in the inclusive range [0, 4]");
  }
  impl_->log_severity = severity;
}

void OnnxRuntimeCudaBackend::set_cpu_ep_fallback(bool enabled) { impl_->cpu_ep_fallback = enabled; }

void OnnxRuntimeCudaBackend::set_profiling(bool enabled, std::string output_prefix) {
  impl_->profiling_enabled = enabled;
  impl_->profiling_prefix = std::move(output_prefix);
}

void OnnxRuntimeCudaBackend::prepare(const BackendConfig& config) {
  if (config.model_format != ModelFormat::Onnx) {
    throw VieError(ErrorCode::Unsupported,
                   "the ONNX Runtime CUDA backend only executes model format 'onnx'");
  }
  if (config.device != DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported, "the ONNX Runtime CUDA backend only runs on CUDA");
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

  int device_count = 0;
  try {
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  } catch (const VieError& error) {
    fail_load(ErrorCode::Internal, error.what());
  }
  if (config.device_id >= device_count) {
    fail_load(ErrorCode::InvalidArgument, "CUDA device id " + std::to_string(config.device_id) +
                                             " is invalid; device count is " + std::to_string(device_count));
  }
  try {
    check_cuda(cudaSetDevice(config.device_id), "cudaSetDevice");
  } catch (const VieError& error) {
    fail_load(ErrorCode::Internal, error.what());
  }

  impl_->release_session();
  impl_->device_id = config.device_id;
  impl_->device_ready = true;
  {
    std::lock_guard<std::mutex> guard(impl_->log_mu);
    impl_->session_log.clear();
  }
  const Clock::time_point started = Clock::now();
  try {
    auto logger = [](void* param, OrtLoggingLevel, const char*, const char*, const char*,
                     const char* message) {
      static_cast<Impl*>(param)->append_log(message);
    };
    impl_->env = Ort::Env(static_cast<OrtLoggingLevel>(impl_->log_severity), "vie.onnxruntime.cuda",
                          logger, impl_.get());
    impl_->options = Ort::SessionOptions{};
    if (impl_->intra_op_threads > 0) {
      impl_->options.SetIntraOpNumThreads(impl_->intra_op_threads);
    }
    if (impl_->inter_op_threads > 0) {
      impl_->options.SetInterOpNumThreads(impl_->inter_op_threads);
    }
    impl_->options.SetGraphOptimizationLevel(ort_optimization(impl_->optimization));
    impl_->options.SetLogSeverityLevel(impl_->log_severity);
    impl_->options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback,
                                  impl_->cpu_ep_fallback ? "0" : "1");

    // Phase 9.3.2.1: When profiling is opted in, enable it on the SAME session
    // options that create the inference session (never a second session). This
    // only records telemetry; it does not change the CUDA EP, CPU fallback, or
    // the execution path. Disabled by default, so this branch is skipped in the
    // normal production path.
    if (impl_->profiling_enabled && !impl_->profiling_prefix.empty()) {
      impl_->options.EnableProfiling(impl_->profiling_prefix.c_str());
    }

    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = config.device_id;
    cuda_options.do_copy_in_default_stream = 1;
    cuda_options.has_user_compute_stream = 0;
    cuda_options.user_compute_stream = nullptr;
    impl_->options.AppendExecutionProvider_CUDA(cuda_options);

    impl_->device = DeviceContext(DeviceType::Cuda, config.device_id);
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
        fail_load(ErrorCode::InvalidArgument, "declared output '" + spec.name + "' is not a model output");
      }
      for (const std::string& previous : declared) {
        if (previous == spec.name) {
          fail_load(ErrorCode::InvalidArgument, "declared output '" + spec.name + "' is duplicated");
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
    if (code == ORT_NO_SUCHFILE) fail_load(ErrorCode::InvalidArgument, error.what());
    if (code == ORT_INVALID_PROTOBUF || code == ORT_INVALID_GRAPH || code == ORT_NO_MODEL) {
      fail_load(ErrorCode::SchemaError, error.what());
    }
    fail_load(ErrorCode::Internal, error.what());
  } catch (const std::exception& error) {
    impl_->release_session();
    fail_load(ErrorCode::Internal, error.what());
  }
}

BackendOutput OnnxRuntimeCudaBackend::execute(const BackendRequest& request) {
  if (!impl_->prepared || impl_->session == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "onnxruntime cuda backend is not prepared");
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

  std::vector<const Tensor*> bound;
  bound.reserve(impl_->inputs.size());
  for (const Endpoint& endpoint : impl_->inputs) {
    const Tensor& input = request.inputs.at(endpoint.name);
    if (input.empty()) fail_input("input tensor '" + endpoint.name + "' has no storage");
    if (input.device() != DeviceType::Cuda || input.device_id() != impl_->device_id) {
      fail_input("input tensor '" + endpoint.name + "' must be cuda:" +
                 std::to_string(impl_->device_id) + ", got " + std::string(to_string(input.device())) +
                 ":" + std::to_string(input.device_id()));
    }
    if (!input.contiguous()) {
      throw VieError(ErrorCode::Unsupported, "input tensor '" + endpoint.name +
                                                "' is not contiguous; the CUDA backend does not copy it");
    }
    if (input.dtype() != endpoint.dtype) {
      fail_input("input tensor '" + endpoint.name + "' dtype " + to_string(input.dtype()) +
                 " does not match model dtype " + to_string(endpoint.dtype));
    }
    check_model_shape(endpoint.name, input, endpoint);
    check_profile(endpoint.name, input, find_spec(impl_->config, endpoint.name, true));
    bound.push_back(&input);
  }

  try {
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice");
    Ort::MemoryInfo cuda_memory("Cuda", OrtDeviceAllocator, impl_->device_id, OrtMemTypeDefault);
    require_cuda_memory(cuda_memory.GetConst(), impl_->device_id, "CUDA memory info");

    const Clock::time_point bind_started = Clock::now();
    Ort::IoBinding binding(*impl_->session);
    std::vector<Ort::Value> values;
    values.reserve(bound.size());
    for (std::size_t index = 0; index < bound.size(); ++index) {
      const Tensor& tensor = *bound[index];
      Ort::Value value = Ort::Value::CreateTensor(
          cuda_memory, const_cast<void*>(static_cast<const void*>(tensor.device_data())), tensor.nbytes(),
          tensor.shape().data(), tensor.shape().size(), ort_from_dtype(tensor.dtype()));
      require_cuda_memory(value.GetTensorMemoryInfo(), impl_->device_id,
                          "bound input '" + impl_->inputs[index].name + "'");
      if (value.GetTensorRawData() != tensor.device_data()) {
        fail_input("input tensor '" + impl_->inputs[index].name +
                   "' was not borrowed; ORT holds a different address");
      }
      binding.BindInput(impl_->inputs[index].name.c_str(), value);
      values.push_back(std::move(value));
    }
    for (const Endpoint& endpoint : impl_->outputs) {
      binding.BindOutput(endpoint.name.c_str(), cuda_memory);
    }
    impl_->timing.last_bind_ms = elapsed_ms(bind_started);

    const Clock::time_point input_sync_started = Clock::now();
    binding.SynchronizeInputs();
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    impl_->timing.last_input_sync_ms = elapsed_ms(input_sync_started);

    const Clock::time_point run_started = Clock::now();
    Ort::RunOptions run_options;
    impl_->session->Run(run_options, binding);
    impl_->timing.last_run_ms = elapsed_ms(run_started);

    const Clock::time_point sync_started = Clock::now();
    binding.SynchronizeOutputs();
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    impl_->timing.last_sync_ms = elapsed_ms(sync_started);

    std::vector<Ort::Value> results = binding.GetOutputValues();
    if (results.size() != impl_->outputs.size()) {
      fail_output("runtime returned " + std::to_string(results.size()) + " tensor(s), expected " +
                  std::to_string(impl_->outputs.size()));
    }

    BackendOutput output;
    for (std::size_t index = 0; index < results.size(); ++index) {
      const std::string& name = impl_->outputs[index].name;
      Ort::Value& value = results[index];
      if (!value.IsTensor()) fail_output("output '" + name + "' is not a tensor");
      require_cuda_memory(value.GetTensorMemoryInfo(), impl_->device_id, "output '" + name + "'");
      const Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
      const DataType dtype = dtype_from_ort(info.GetElementType());
      if (dtype == DataType::Unknown) {
        fail_output("output '" + name + "' has a dtype that this build cannot represent");
      }
      std::vector<std::int64_t> shape = info.GetShape();
      for (const std::int64_t dim : shape) {
        if (dim <= 0) {
          fail_output("output '" + name + "' has a non-positive dimension in shape " + shape_text(shape));
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

      const Clock::time_point alloc_started = Clock::now();
      Tensor tensor = Tensor::zeros(impl_->device, dtype, shape);
      impl_->timing.last_output_alloc_ms += elapsed_ms(alloc_started);
      const Clock::time_point copy_started = Clock::now();
      copy_device_memory(nullptr, DeviceType::Cuda, impl_->device_id, static_cast<const std::byte*>(raw),
                         DeviceType::Cuda, impl_->device_id, tensor.device_data(), tensor.nbytes());
      impl_->timing.last_output_copy_ms += elapsed_ms(copy_started);
      output.tensors.emplace(name, NamedTensor{std::move(tensor), TensorLifetime::Copied});
    }
    return output;
  } catch (const VieError&) {
    throw;
  } catch (const Ort::Exception& error) {
    fail_inference(error.what());
  } catch (const std::exception& error) {
    fail_inference(error.what());
  }
}

BackendInfo OnnxRuntimeCudaBackend::info() const {
  BackendInfo info;
  info.name = "onnxruntime-cuda";
  info.kind = BackendKind::OnnxRuntimeCuda;
  info.device = DeviceType::Cuda;
  info.supports_cuda = true;
  return info;
}

const OnnxRuntimeCudaTiming& OnnxRuntimeCudaBackend::timing() const { return impl_->timing; }

bool OnnxRuntimeCudaBackend::cpu_ep_fallback() const { return impl_->cpu_ep_fallback; }

std::string OnnxRuntimeCudaBackend::end_profiling() {
  if (!impl_->profiling_enabled || impl_->session == nullptr) return {};
  // Flushes the profile to <prefix><timestamp>.json and returns the real path.
  // Safe to call once inference has completed; does not touch the session state
  // used for further runs. Called before release_session()/destruction so the
  // file handle is closed deterministically (§4.2 requirements 6-8).
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr path = impl_->session->EndProfilingAllocated(allocator);
  return path.get() != nullptr ? std::string(path.get()) : std::string();
}

std::string OnnxRuntimeCudaBackend::session_log() const {
  std::lock_guard<std::mutex> guard(impl_->log_mu);
  return impl_->session_log;
}

}  // namespace vie
