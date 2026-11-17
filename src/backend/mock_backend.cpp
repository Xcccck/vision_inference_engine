#include "vie/backend/mock_backend.hpp"

#include <algorithm>
#include <array>

namespace vie {

void MockBackend::prepare(const BackendConfig& config) {
  if (config.model_format != ModelFormat::Mock) {
    throw VieError(ErrorCode::Unsupported, "the mock backend only prepares model format 'mock'");
  }
  if (config.device != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "the mock backend only runs on CPU");
  }
  if (config.device_id < 0) {
    throw VieError(ErrorCode::InvalidArgument, "device id must be >= 0");
  }
  if (config.input.name.empty() || !is_concrete_shape(config.input.shape) ||
      config.input.dtype == DataType::Unknown) {
    throw VieError(ErrorCode::InvalidArgument, "mock backend input spec is incomplete");
  }
  if (config.outputs.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "mock backend requires at least one output");
  }
  for (const TensorSpec& output : config.outputs) {
    if (output.name.empty() || !is_concrete_shape(output.shape) ||
        output.dtype == DataType::Unknown) {
      throw VieError(ErrorCode::InvalidArgument, "mock backend output spec is incomplete");
    }
  }
  config_ = config;
  prepared_ = true;
}

BackendOutput MockBackend::execute(const BackendRequest& request) {
  if (!prepared_) {
    throw VieError(ErrorCode::InvalidArgument, "mock backend is not prepared");
  }
  if (request.inputs.size() != 1) {
    throw VieError(ErrorCode::InvalidArgument, "mock backend expects exactly one input tensor");
  }
  const auto found = request.inputs.find(config_.input.name);
  if (found == request.inputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "missing input tensor '" + config_.input.name + "'");
  }
  const Tensor& input = found->second;
  if (input.device() != DeviceType::Cpu) {
    throw VieError(ErrorCode::Unsupported, "mock backend inputs must reside on CPU");
  }
  if (!input.contiguous()) {
    throw VieError(ErrorCode::InvalidArgument, "mock backend input must be contiguous");
  }
  if (input.dtype() != config_.input.dtype) {
    throw VieError(ErrorCode::InvalidArgument, "input dtype does not match the manifest");
  }
  if (input.shape() != config_.input.shape) {
    throw VieError(ErrorCode::InvalidArgument, "input shape does not match the manifest");
  }

  const DeviceContext host(DeviceType::Cpu, config_.device_id);
  BackendOutput output;
  constexpr std::array<float, 6> kPrefix{1.f, 2.f, 3.f, 4.f, 0.9f, 0.f};
  for (const TensorSpec& spec : config_.outputs) {
    Tensor tensor = Tensor::zeros(host, spec.dtype, spec.shape);
    tensor.set_layout(spec.layout);
    if (spec.dtype == DataType::Fp32) {
      float* values = tensor.mutable_data<float>();
      const std::size_t count = std::min(tensor.numel(), kPrefix.size());
      std::copy(kPrefix.begin(), kPrefix.begin() + static_cast<std::ptrdiff_t>(count), values);
    }
    output.tensors.emplace(spec.name, NamedTensor{std::move(tensor), TensorLifetime::Copied});
  }
  return output;
}

BackendInfo MockBackend::info() const {
  BackendInfo info;
  info.name = "mock";
  info.kind = BackendKind::Mock;
  info.device = prepared_ ? config_.device : DeviceType::Cpu;
  info.supports_cuda = false;
  return info;
}

}  // namespace vie
