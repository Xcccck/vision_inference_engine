#include "onnx_test_writer.hpp"
#include "test_util.hpp"

#include "vie/backend/onnxruntime_backend.hpp"
#include "vie/backend/onnxruntime_cuda_backend.hpp"
#include "vie/engine/inference_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path fixture_dir() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "vie-phase422";
  std::filesystem::create_directories(dir);
  return dir;
}

std::string write_model(const std::string& name, const OnnxTestWriter& model) {
  const std::filesystem::path path = fixture_dir() / name;
  std::ofstream output(path, std::ios::binary);
  EXPECT_TRUE(output) << path.string();
  output.write(model.str().data(), static_cast<std::streamsize>(model.str().size()));
  EXPECT_TRUE(output);
  return path.string();
}

OnnxTestWriter identity_model(const std::vector<TestDim>& shape, std::string_view input,
                              std::string_view output) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Identity", {std::string(input)}, {std::string(output)}));
  graph.text(2, "identity");
  graph.message(11, value_info(input, tensor_type(1, true, shape)));
  graph.message(12, value_info(output, tensor_type(1, true, shape)));
  OnnxTestWriter model;
  model.i64(1, 8);
  model.message(7, graph);
  model.message(8, opset_import("", 13));
  return model;
}

OnnxTestWriter add_model() {
  OnnxTestWriter graph;
  graph.message(1, node_message("Add", {"a", "b"}, {"sum"}));
  graph.text(2, "add");
  const std::vector<TestDim> shape{static_dim(4)};
  graph.message(11, value_info("a", tensor_type(1, true, shape)));
  graph.message(11, value_info("b", tensor_type(1, true, shape)));
  graph.message(12, value_info("sum", tensor_type(1, true, shape)));
  OnnxTestWriter model;
  model.i64(1, 8);
  model.message(7, graph);
  model.message(8, opset_import("", 13));
  return model;
}

OnnxTestWriter add_sub_model() {
  OnnxTestWriter graph;
  graph.message(1, node_message("Add", {"left", "right"}, {"total"}));
  graph.message(1, node_message("Sub", {"left", "right"}, {"difference"}));
  graph.text(2, "add_sub");
  const std::vector<TestDim> shape{static_dim(2)};
  graph.message(11, value_info("left", tensor_type(1, true, shape)));
  graph.message(11, value_info("right", tensor_type(1, true, shape)));
  graph.message(12, value_info("total", tensor_type(1, true, shape)));
  graph.message(12, value_info("difference", tensor_type(1, true, shape)));
  OnnxTestWriter model;
  model.i64(1, 8);
  model.message(7, graph);
  model.message(8, opset_import("", 13));
  return model;
}

vie::DeviceContext cpu() { return vie::DeviceContext(vie::DeviceType::Cpu, 0); }

vie::DeviceContext gpu() { return vie::DeviceContext(vie::DeviceType::Cuda, 0); }

vie::BackendConfig cuda_config(const std::string& path) {
  vie::BackendConfig config;
  config.model_path = path;
  config.model_format = vie::ModelFormat::Onnx;
  config.device = vie::DeviceType::Cuda;
  config.device_id = 0;
  return config;
}

vie::Tensor cuda_tensor(const std::vector<float>& values, std::vector<std::int64_t> shape) {
  return vie::Tensor::from_host(gpu(), vie::DataType::Fp32, std::move(shape), values.data(),
                                values.size() * sizeof(float));
}

std::vector<float> to_cpu(const vie::Tensor& tensor) {
  const vie::Tensor host = tensor.to(cpu());
  const float* data = host.data_as<float>();
  return std::vector<float>(data, data + host.numel());
}

void expect_float(const vie::Tensor& tensor, const std::vector<float>& expected) {
  const std::vector<float> actual = to_cpu(tensor);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(actual[index], expected[index]) << index;
  }
}

void expect_failure(vie::ErrorCode code, const std::string& snippet, const std::function<void()>& call) {
  try {
    call();
    ADD_FAILURE() << "expected VieError containing '" << snippet << "'";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), code) << error.what();
    EXPECT_NE(std::string(error.what()).find(snippet), std::string::npos) << error.what();
  }
}

void expect_device_pointer(const vie::Tensor& tensor) {
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, tensor.device_data()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
  EXPECT_EQ(attributes.device, tensor.device_id());
}

bool maps_contain(const std::string& snippet) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find(snippet) != std::string::npos) return true;
  }
  return false;
}

struct Summary {
  double mean = 0;
  double p50 = 0;
  double p95 = 0;
  double min = 0;
  double max = 0;
};

Summary summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  Summary summary;
  summary.min = samples.front();
  summary.max = samples.back();
  summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                 static_cast<double>(samples.size());
  summary.p50 = samples[(samples.size() - 1) / 2];
  summary.p95 = samples[samples.size() * 95 / 100];
  if (samples.size() == 100) {
    summary.p50 = samples[49];
    summary.p95 = samples[94];
  }
  return summary;
}

}  // namespace

TEST(OnnxRuntimeCudaBackend, IdentityAddAndMultipleOutputsStayOnDevice) {
  const std::string identity_path =
      write_model("identity.onnx", identity_model({static_dim(4)}, "images", "predictions"));
  vie::OnnxRuntimeCudaBackend backend;
  backend.set_log_severity(0);
  backend.set_intra_op_threads(1);
  EXPECT_FALSE(backend.cpu_ep_fallback());
  backend.prepare(cuda_config(identity_path));
  EXPECT_GT(backend.timing().load_ms, 0.0);
  EXPECT_EQ(backend.info().kind, vie::BackendKind::OnnxRuntimeCuda);
  EXPECT_TRUE(backend.info().supports_cuda);

  const std::string log = backend.session_log();
  EXPECT_NE(log.find("All nodes placed on [CUDAExecutionProvider]. Number of nodes: 1"),
            std::string::npos)
      << log;
  EXPECT_EQ(log.find("placed on [CPUExecutionProvider]"), std::string::npos) << log;
  EXPECT_TRUE(maps_contain("/opt/onnxruntime-linux-x64-gpu-1.20.1/lib/libonnxruntime.so"));
  EXPECT_FALSE(maps_contain("/opt/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so"));
  EXPECT_TRUE(maps_contain("libonnxruntime_providers_cuda.so"));

  const vie::Tensor input = cuda_tensor({1.f, 2.f, 3.f, 4.f}, {4});
  expect_device_pointer(input);
  const void* address = input.device_data();
  vie::BackendRequest request;
  request.inputs.emplace("images", input);
  const vie::BackendOutput output = backend.execute(request);
  EXPECT_EQ(input.device_data(), address);
  expect_float(input, {1.f, 2.f, 3.f, 4.f});
  ASSERT_EQ(output.tensors.count("predictions"), 1u);
  const vie::Tensor& predictions = output.tensors.at("predictions").tensor;
  EXPECT_TRUE(predictions.is_cuda());
  EXPECT_EQ(predictions.device_id(), 0);
  EXPECT_EQ(predictions.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(predictions.shape(), (std::vector<std::int64_t>{4}));
  EXPECT_EQ(output.tensors.at("predictions").lifetime, vie::TensorLifetime::Copied);
  expect_device_pointer(predictions);
  expect_float(predictions, {1.f, 2.f, 3.f, 4.f});
  EXPECT_FALSE(predictions.shares_storage(input));

  vie::OnnxRuntimeCudaBackend add;
  add.prepare(cuda_config(write_model("add.onnx", add_model())));
  vie::BackendRequest add_request;
  add_request.inputs.emplace("a", cuda_tensor({1.f, 2.f, 3.f, 4.f}, {4}));
  add_request.inputs.emplace("b", cuda_tensor({10.f, 20.f, 30.f, 40.f}, {4}));
  const vie::BackendOutput sum = add.execute(add_request);
  expect_float(sum.tensors.at("sum").tensor, {11.f, 22.f, 33.f, 44.f});

  vie::OnnxRuntimeCudaBackend multi;
  multi.prepare(cuda_config(write_model("add_sub.onnx", add_sub_model())));
  vie::BackendRequest multi_request;
  multi_request.inputs.emplace("left", cuda_tensor({5.f, 9.f}, {2}));
  multi_request.inputs.emplace("right", cuda_tensor({1.f, 4.f}, {2}));
  const vie::BackendOutput both = multi.execute(multi_request);
  EXPECT_EQ(both.tensors.size(), 2u);
  expect_float(both.tensors.at("total").tensor, {6.f, 13.f});
  expect_float(both.tensors.at("difference").tensor, {4.f, 5.f});
}

TEST(OnnxRuntimeCudaBackend, RepeatsExecutesAndKeepsOutputAfterDestruction) {
  const std::string path =
      write_model("lifetime.onnx", identity_model({static_dim(2)}, "x", "y"));
  vie::BackendOutput kept;
  {
    vie::OnnxRuntimeCudaBackend backend;
    backend.prepare(cuda_config(path));
    vie::BackendRequest first_request;
    first_request.inputs.emplace("x", cuda_tensor({8.f, 9.f}, {2}));
    const vie::BackendOutput first = backend.execute(first_request);
    vie::BackendRequest second_request;
    second_request.inputs.emplace("x", cuda_tensor({3.f, 4.f}, {2}));
    const vie::BackendOutput second = backend.execute(second_request);
    EXPECT_FALSE(first.tensors.at("y").tensor.shares_storage(second.tensors.at("y").tensor));
    expect_float(first.tensors.at("y").tensor, {8.f, 9.f});
    expect_float(second.tensors.at("y").tensor, {3.f, 4.f});
    expect_float(first_request.inputs.at("x"), {8.f, 9.f});
    kept = first;
  }
  expect_float(kept.tensors.at("y").tensor, {8.f, 9.f});
  expect_device_pointer(kept.tensors.at("y").tensor);
}

TEST(OnnxRuntimeCudaBackend, RejectsBadInputsAndUnavailableDevices) {
  const std::string path =
      write_model("reject.onnx", identity_model({static_dim(4)}, "images", "predictions"));
  vie::OnnxRuntimeCudaBackend backend;
  backend.prepare(cuda_config(path));

  vie::BackendRequest cpu_input;
  cpu_input.inputs.emplace("images", vie::Tensor::zeros(cpu(), vie::DataType::Fp32, {4}));
  expect_failure(vie::ErrorCode::InvalidArgument, "must be cuda:0", [&] { backend.execute(cpu_input); });

  vie::Tensor full = cuda_tensor({1.f, 2.f, 3.f, 4.f}, {4});
  const vie::Tensor strided = full.view({2}, {2}, 0);
  EXPECT_FALSE(strided.contiguous());
  vie::BackendRequest strided_request;
  strided_request.inputs.emplace("images", strided);
  expect_failure(vie::ErrorCode::Unsupported, "not contiguous", [&] { backend.execute(strided_request); });

  vie::BackendRequest missing;
  expect_failure(vie::ErrorCode::InvalidArgument, "missing input tensor", [&] { backend.execute(missing); });
  vie::BackendRequest extra = cpu_input;
  extra.inputs.emplace("other", cuda_tensor({1.f}, {1}));
  expect_failure(vie::ErrorCode::InvalidArgument, "unexpected input tensor", [&] { backend.execute(extra); });

  vie::BackendRequest dtype;
  dtype.inputs.emplace("images", vie::Tensor::zeros(gpu(), vie::DataType::Int32, {4}));
  expect_failure(vie::ErrorCode::InvalidArgument, "does not match model dtype", [&] { backend.execute(dtype); });
  vie::BackendRequest shape;
  shape.inputs.emplace("images", cuda_tensor({1.f, 2.f}, {2}));
  expect_failure(vie::ErrorCode::InvalidArgument, "does not match model shape", [&] { backend.execute(shape); });

  vie::OnnxRuntimeBackend cpu_backend;
  cpu_backend.prepare([&] {
    vie::BackendConfig config = cuda_config(path);
    config.device = vie::DeviceType::Cpu;
    return config;
  }());
  vie::BackendRequest gpu_input;
  gpu_input.inputs.emplace("images", full);
  expect_failure(vie::ErrorCode::InvalidArgument, "is not a CPU tensor", [&] { cpu_backend.execute(gpu_input); });

  expect_failure(vie::ErrorCode::InvalidArgument, "device count is", [&] {
    vie::OnnxRuntimeCudaBackend bad_device;
    vie::BackendConfig config = cuda_config(path);
    config.device_id = 99;
    bad_device.prepare(config);
  });

  expect_failure(vie::ErrorCode::InvalidArgument, "model file does not exist", [&] {
    vie::OnnxRuntimeCudaBackend missing_file;
    missing_file.prepare(cuda_config((fixture_dir() / "missing.onnx").string()));
  });
  const std::filesystem::path corrupt = fixture_dir() / "corrupt.onnx";
  {
    std::ofstream output(corrupt, std::ios::binary);
    const char bytes[] = "this is not an onnx model";
    output.write(bytes, sizeof(bytes) - 1);
  }
  expect_failure(vie::ErrorCode::SchemaError, "onnxruntime cuda load failed:", [&] {
    vie::OnnxRuntimeCudaBackend broken;
    broken.prepare(cuda_config(corrupt.string()));
  });

  expect_failure(vie::ErrorCode::Internal, "fallback to CPU EP has been explicitly disabled", [&] {
    vie::OnnxRuntimeCudaBackend strict;
    strict.prepare(cuda_config(vie_fixture("testdata/onnx/yolov8n.onnx")));
  });
}

TEST(OnnxRuntimeCudaBackend, EngineSelectsCudaAndDoesNotFallBack) {
  const std::string path =
      write_model("engine.onnx", identity_model({static_dim(1), static_dim(6)}, "values", "output0"));
  vie::ModelManifest manifest;
  manifest.model_name = "identity6";
  manifest.task = vie::TaskType::Classification;
  manifest.model_format = vie::ModelFormat::Onnx;
  manifest.model_path = path;
  manifest.num_classes = 1;
  manifest.input.name = "values";
  manifest.input.shape = {1, 6};
  manifest.input.dtype = vie::DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "stretch";
  vie::OutputDesc output_desc;
  output_desc.name = "output0";
  output_desc.shape = {1, 6};
  output_desc.dtype = vie::DataType::Fp32;
  manifest.outputs.push_back(output_desc);
  manifest.decoder.protocol = "mock";

  vie::ModelConfig config;
  config.manifest = manifest;
  config.backend = vie::BackendKind::Auto;
  config.device = vie::DeviceType::Cuda;
  config.device_id = 0;
  vie::InferenceEngine engine;
  EXPECT_TRUE(engine.backends().contains(vie::BackendKind::OnnxRuntimeCuda));
  engine.load(config);
  EXPECT_TRUE(engine.loaded());
  EXPECT_EQ(engine.device().device(), vie::DeviceType::Cuda);

  const float values[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  vie::BackendRequest request;
  request.inputs.emplace("values", vie::Tensor::from_host(gpu(), vie::DataType::Fp32, {1, 6}, values,
                                                          sizeof(values)));
  const vie::BackendOutput output = engine.execute(request);
  expect_float(output.tensors.at("output0").tensor, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

  vie::InferenceEngine explicit_cpu_kind;
  config.backend = vie::BackendKind::OnnxRuntime;
  explicit_cpu_kind.load(config);
  const vie::BackendOutput selected = explicit_cpu_kind.execute(request);
  EXPECT_TRUE(selected.tensors.at("output0").tensor.is_cuda());

  vie::InferenceEngine rejected;
  config.backend = vie::BackendKind::OnnxRuntimeCuda;
  config.device = vie::DeviceType::Cpu;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, rejected.load(config));
  EXPECT_FALSE(rejected.loaded());
  EXPECT_EQ(rejected.device().device(), vie::DeviceType::Cpu);

  vie::InferenceEngine bad_device;
  config.device = vie::DeviceType::Cuda;
  config.device_id = 99;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, bad_device.load(config));
  EXPECT_FALSE(bad_device.loaded());
  EXPECT_EQ(bad_device.device().device(), vie::DeviceType::Cpu);
}

TEST(OnnxRuntimeCudaBackend, RunsYoloV8nOnCuda) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n.onnx");
  vie::OnnxRuntimeCudaBackend backend;
  backend.set_intra_op_threads(4);
  backend.set_inter_op_threads(1);
  backend.set_graph_optimization(vie::OnnxRuntimeOptimization::All);
  backend.set_log_severity(1);
  backend.set_cpu_ep_fallback(true);
  backend.prepare(cuda_config(path));
  EXPECT_TRUE(backend.cpu_ep_fallback());
  const std::string log = backend.session_log();
  EXPECT_NE(log.find("CUDAExecutionProvider"), std::string::npos) << log;
  std::ofstream placement("/tmp/vie_phase422_yolov8_placement.txt");
  ASSERT_TRUE(placement.is_open());
  placement << "cpu_ep_fallback=1\n";
  std::istringstream lines(log);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find("placed on") != std::string::npos || line.find("Memcpy") != std::string::npos ||
        line.find("CUDAExecutionProvider") != std::string::npos) {
      placement << line << "\n";
    }
  }
  vie::BackendRequest request;
  request.inputs.emplace("images", vie::Tensor::zeros(gpu(), vie::DataType::Fp32, {1, 3, 640, 640}));
  const vie::BackendOutput output = backend.execute(request);
  ASSERT_EQ(output.tensors.count("predictions"), 1u);
  const vie::Tensor& predictions = output.tensors.at("predictions").tensor;
  EXPECT_TRUE(predictions.is_cuda());
  EXPECT_EQ(predictions.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(predictions.shape(), (std::vector<std::int64_t>{1, 84, 8400}));
  EXPECT_EQ(output.tensors.at("predictions").lifetime, vie::TensorLifetime::Copied);
  expect_device_pointer(predictions);
  const std::vector<float> host = to_cpu(predictions);
  ASSERT_EQ(host.size(), predictions.numel());
  for (std::size_t index = 0; index < host.size(); ++index) {
    ASSERT_TRUE(std::isfinite(host[index])) << index;
  }
}

TEST(OnnxRuntimeCudaBackend, ComparesYoloV8nCpuAndCuda) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n.onnx");
  constexpr int kWarmup = 10;
  constexpr int kRuns = 100;
  const std::vector<std::int64_t> shape{1, 3, 640, 640};

  vie::OnnxRuntimeBackend cpu_backend;
  cpu_backend.set_intra_op_threads(4);
  cpu_backend.set_inter_op_threads(1);
  cpu_backend.set_graph_optimization(vie::OnnxRuntimeOptimization::All);
  cpu_backend.prepare([&] {
    vie::BackendConfig config = cuda_config(path);
    config.device = vie::DeviceType::Cpu;
    return config;
  }());
  const double cpu_load_ms = cpu_backend.timing().load_ms;
  vie::BackendRequest cpu_request;
  cpu_request.inputs.emplace("images", vie::Tensor::zeros(cpu(), vie::DataType::Fp32, shape));
  cpu_backend.execute(cpu_request);
  std::vector<double> cpu_wall;
  cpu_wall.reserve(kRuns);
  for (int index = 0; index < kRuns; ++index) {
    const auto started = std::chrono::steady_clock::now();
    cpu_backend.execute(cpu_request);
    cpu_wall.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                           .count());
  }

  vie::OnnxRuntimeCudaBackend cuda_backend;
  cuda_backend.set_intra_op_threads(4);
  cuda_backend.set_inter_op_threads(1);
  cuda_backend.set_graph_optimization(vie::OnnxRuntimeOptimization::All);
  cuda_backend.set_cpu_ep_fallback(true);
  cuda_backend.prepare(cuda_config(path));
  const double cuda_load_ms = cuda_backend.timing().load_ms;

  const auto alloc_begin = std::chrono::steady_clock::now();
  vie::Tensor gpu_input = vie::Tensor::zeros(gpu(), vie::DataType::Fp32, shape);
  const double alloc_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_begin).count();
  const auto alloc_again_begin = std::chrono::steady_clock::now();
  vie::Tensor gpu_again = vie::Tensor::zeros(gpu(), vie::DataType::Fp32, shape);
  const double alloc_again_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_again_begin)
          .count();
  gpu_again = vie::Tensor{};

  vie::Tensor cpu_input = vie::Tensor::zeros(cpu(), vie::DataType::Fp32, shape);
  const auto h2d_begin = std::chrono::steady_clock::now();
  gpu_input.copy_from(cpu_input);
  const double h2d_once_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h2d_begin).count();

  vie::BackendRequest gpu_request;
  gpu_request.inputs.emplace("images", gpu_input);
  for (int index = 0; index < kWarmup; ++index) cuda_backend.execute(gpu_request);

  std::vector<double> ready_wall;
  std::vector<double> bind_ms;
  std::vector<double> run_ms;
  std::vector<double> sync_ms;
  std::vector<double> output_alloc_ms;
  std::vector<double> d2d_ms;
  ready_wall.reserve(kRuns);
  for (int index = 0; index < kRuns; ++index) {
    const auto started = std::chrono::steady_clock::now();
    const vie::BackendOutput output = cuda_backend.execute(gpu_request);
    ready_wall.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    bind_ms.push_back(cuda_backend.timing().last_bind_ms);
    run_ms.push_back(cuda_backend.timing().last_run_ms);
    sync_ms.push_back(cuda_backend.timing().last_sync_ms);
    output_alloc_ms.push_back(cuda_backend.timing().last_output_alloc_ms);
    d2d_ms.push_back(cuda_backend.timing().last_output_copy_ms);
    ASSERT_EQ(output.tensors.at("predictions").tensor.shape(), (std::vector<std::int64_t>{1, 84, 8400}));
  }

  vie::Tensor download = vie::Tensor::zeros(cpu(), vie::DataType::Fp32, {1, 84, 8400});
  std::vector<double> h2d_ms_samples;
  std::vector<double> execute_ms_samples;
  std::vector<double> d2h_ms_samples;
  std::vector<double> e2e_ms_samples;
  for (int index = 0; index < kRuns; ++index) {
    const auto started = std::chrono::steady_clock::now();
    const auto copied = std::chrono::steady_clock::now();
    gpu_input.copy_from(cpu_input);
    const auto executed = std::chrono::steady_clock::now();
    const vie::BackendOutput output = cuda_backend.execute(gpu_request);
    const auto returned = std::chrono::steady_clock::now();
    download.copy_from(output.tensors.at("predictions").tensor);
    const auto finished = std::chrono::steady_clock::now();
    auto ms = [](std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    h2d_ms_samples.push_back(ms(copied, executed));
    execute_ms_samples.push_back(ms(executed, returned));
    d2h_ms_samples.push_back(ms(returned, finished));
    e2e_ms_samples.push_back(ms(started, finished));
  }

  const Summary cpu_summary = summarize(cpu_wall);
  const Summary ready_summary = summarize(ready_wall);
  const Summary e2e_summary = summarize(e2e_ms_samples);
  std::ofstream out("/tmp/vie_phase422_yolov8.txt");
  ASSERT_TRUE(out.is_open());
  out << "model=" << path << "\n";
  out << "cuda_cpu_ep_fallback=1\n";
  out << "cuda_execution=mixed\n";
  out << "input=1x3x640x640 fp32 zeros\n";
  out << "warmup=" << kWarmup << " measured=" << kRuns << "\n";
  out << "cpu_session_ms=" << cpu_load_ms << "\n";
  out << "cuda_session_ms=" << cuda_load_ms << "\n";
  out << "cuda_input_alloc_first_ms=" << alloc_ms << "\n";
  out << "cuda_input_alloc_second_ms=" << alloc_again_ms << "\n";
  out << "h2d_once_ms=" << h2d_once_ms << "\n";
  out << "cpu_execute_mean_ms=" << cpu_summary.mean << " p50=" << cpu_summary.p50
      << " p95=" << cpu_summary.p95 << " min=" << cpu_summary.min << " max=" << cpu_summary.max << "\n";
  out << "cuda_ready_input_execute_mean_ms=" << ready_summary.mean << " p50=" << ready_summary.p50
      << " p95=" << ready_summary.p95 << " min=" << ready_summary.min << " max=" << ready_summary.max
      << "\n";
  out << "cuda_bind_mean_ms=" << summarize(bind_ms).mean << "\n";
  out << "cuda_run_mean_ms=" << summarize(run_ms).mean << "\n";
  out << "cuda_sync_after_run_mean_ms=" << summarize(sync_ms).mean << "\n";
  out << "cuda_output_alloc_mean_ms=" << summarize(output_alloc_ms).mean << "\n";
  out << "cuda_d2d_mean_ms=" << summarize(d2d_ms).mean << "\n";
  out << "path_b_h2d_mean_ms=" << summarize(h2d_ms_samples).mean << "\n";
  out << "path_b_execute_mean_ms=" << summarize(execute_ms_samples).mean << "\n";
  out << "path_b_d2h_mean_ms=" << summarize(d2h_ms_samples).mean << "\n";
  out << "path_b_e2e_mean_ms=" << e2e_summary.mean << " p50=" << e2e_summary.p50
      << " p95=" << e2e_summary.p95 << " min=" << e2e_summary.min << " max=" << e2e_summary.max << "\n";
  out << "note=run_ms is ORT Run wall time; sync_after_run_ms is SynchronizeOutputs plus "
         "cudaDeviceSynchronize. GPU time is their sum. Session creation is excluded from execute samples.\n";
  EXPECT_GT(cpu_summary.mean, 0.0);
  EXPECT_GT(ready_summary.mean, 0.0);
}
