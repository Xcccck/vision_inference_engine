#include "onnx_test_writer.hpp"
#include "test_util.hpp"

#include "vie/backend/onnxruntime_backend.hpp"
#include "vie/engine/inference_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace {

std::filesystem::path fixture_dir() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "vie-phase41";
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

OnnxTestWriter identity_model(int element_type, const std::vector<TestDim>& shape,
                              std::string_view input, std::string_view output) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Identity", {std::string(input)}, {std::string(output)}));
  graph.text(2, "identity");
  graph.message(11, value_info(input, tensor_type(element_type, true, shape)));
  graph.message(12, value_info(output, tensor_type(element_type, true, shape)));
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

vie::BackendConfig cpu_config(const std::string& path) {
  vie::BackendConfig config;
  config.model_path = path;
  config.model_format = vie::ModelFormat::Onnx;
  config.device = vie::DeviceType::Cpu;
  return config;
}

vie::Tensor host_tensor(vie::DataType dtype, std::vector<std::int64_t> shape, const void* bytes,
                        std::size_t size) {
  return vie::Tensor::from_host(vie::DeviceContext{}, dtype, std::move(shape), bytes, size);
}

void expect_failure(vie::ErrorCode code, const std::string& snippet,
                    const std::function<void()>& call) {
  try {
    call();
    ADD_FAILURE() << "expected VieError containing '" << snippet << "'";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), code) << error.what();
    EXPECT_NE(std::string(error.what()).find(snippet), std::string::npos) << error.what();
  }
}

}  // namespace

TEST(OnnxRuntimeBackend, LoadsIdentityAndChecksNumericResult) {
  const std::string path =
      write_model("identity_f32.onnx",
                  identity_model(1, {static_dim(1), static_dim(4)}, "images", "predictions"));
  vie::OnnxRuntimeBackend backend;
  backend.set_intra_op_threads(1);
  backend.set_graph_optimization(vie::OnnxRuntimeOptimization::Basic);
  backend.set_log_severity(3);
  EXPECT_EQ(backend.info().name, "onnxruntime");
  EXPECT_EQ(backend.info().kind, vie::BackendKind::OnnxRuntime);
  EXPECT_FALSE(backend.info().supports_cuda);

  backend.prepare(cpu_config(path));
  EXPECT_GT(backend.timing().load_ms, 0.0);

  const float input[] = {1.5f, -2.f, 3.25f, 4.f};
  vie::BackendRequest request;
  vie::Tensor tensor = host_tensor(vie::DataType::Fp32, {1, 4}, input, sizeof(input));
  tensor.set_layout(vie::Layout::Nchw);
  request.inputs.emplace("images", std::move(tensor));
  const vie::BackendOutput output = backend.execute(request);

  ASSERT_EQ(output.tensors.size(), 1u);
  ASSERT_EQ(output.tensors.count("predictions"), 1u);
  const vie::NamedTensor& named = output.tensors.at("predictions");
  EXPECT_EQ(named.lifetime, vie::TensorLifetime::Copied);
  EXPECT_EQ(named.tensor.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(named.tensor.layout(), vie::Layout::Unknown);
  EXPECT_EQ(named.tensor.shape(), (std::vector<std::int64_t>{1, 4}));
  EXPECT_TRUE(named.tensor.contiguous());
  const float* values = named.tensor.data_as<float>();
  for (int index = 0; index < 4; ++index) EXPECT_FLOAT_EQ(values[index], input[index]);
  EXPECT_FALSE(backend.timing().input_was_copied);
  EXPECT_DOUBLE_EQ(backend.timing().last_input_copy_ms, 0.0);
  EXPECT_GT(backend.timing().last_execute_ms, 0.0);
  EXPECT_GT(backend.timing().last_output_copy_ms, 0.0);
}

TEST(OnnxRuntimeBackend, RunsNamedMultiInputMultiOutput) {
  const std::string path = write_model("add_sub.onnx", add_sub_model());
  vie::OnnxRuntimeBackend backend;
  backend.set_intra_op_threads(1);
  backend.prepare(cpu_config(path));

  const float left[] = {10.f, 20.f};
  const float right[] = {3.f, 5.f};
  vie::BackendRequest request;
  request.inputs.emplace("right", host_tensor(vie::DataType::Fp32, {2}, right, sizeof(right)));
  request.inputs.emplace("left", host_tensor(vie::DataType::Fp32, {2}, left, sizeof(left)));
  const vie::BackendOutput output = backend.execute(request);

  ASSERT_EQ(output.tensors.count("total"), 1u);
  ASSERT_EQ(output.tensors.count("difference"), 1u);
  EXPECT_EQ(output.tensors.at("total").tensor.shape(), (std::vector<std::int64_t>{2}));
  EXPECT_EQ(output.tensors.at("difference").tensor.shape(), (std::vector<std::int64_t>{2}));
  EXPECT_FLOAT_EQ(output.tensors.at("total").tensor.data_as<float>()[0], 13.f);
  EXPECT_FLOAT_EQ(output.tensors.at("total").tensor.data_as<float>()[1], 25.f);
  EXPECT_FLOAT_EQ(output.tensors.at("difference").tensor.data_as<float>()[0], 7.f);
  EXPECT_FLOAT_EQ(output.tensors.at("difference").tensor.data_as<float>()[1], 15.f);
}

TEST(OnnxRuntimeBackend, AcceptsDynamicShapesAndRejectsIllegalOnes) {
  const std::string path = write_model(
      "dynamic_batch.onnx",
      identity_model(1, {symbol_dim("batch"), static_dim(4)}, "features", "features_out"));
  vie::OnnxRuntimeBackend backend;
  backend.set_intra_op_threads(1);
  vie::BackendConfig config = cpu_config(path);
  config.input.name = "features";
  config.input.shape = {-1, 4};
  config.input.dtype = vie::DataType::Fp32;
  backend.prepare(config);

  const float one_batch[] = {1.f, 2.f, 3.f, 4.f};
  const float three_batch[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
  vie::BackendRequest first;
  first.inputs.emplace("features",
                       host_tensor(vie::DataType::Fp32, {1, 4}, one_batch, sizeof(one_batch)));
  const vie::BackendOutput first_out = backend.execute(first);
  EXPECT_EQ(first_out.tensors.at("features_out").tensor.shape(), (std::vector<std::int64_t>{1, 4}));

  vie::BackendRequest third;
  third.inputs.emplace("features", host_tensor(vie::DataType::Fp32, {3, 4}, three_batch,
                                               sizeof(three_batch)));
  const vie::BackendOutput third_out = backend.execute(third);
  EXPECT_EQ(third_out.tensors.at("features_out").tensor.shape(), (std::vector<std::int64_t>{3, 4}));
  EXPECT_FLOAT_EQ(third_out.tensors.at("features_out").tensor.data_as<float>()[11], 12.f);

  vie::BackendRequest bad_static = first;
  bad_static.inputs["features"] = host_tensor(vie::DataType::Fp32, {1, 5}, three_batch,
                                              5 * sizeof(float));
  expect_failure(vie::ErrorCode::InvalidArgument, "input validation failed:",
                 [&] { backend.execute(bad_static); });

  vie::BackendRequest bad_rank = first;
  bad_rank.inputs["features"] = host_tensor(vie::DataType::Fp32, {4}, one_batch, sizeof(one_batch));
  expect_failure(vie::ErrorCode::InvalidArgument, "does not match model rank",
                 [&] { backend.execute(bad_rank); });
}

TEST(OnnxRuntimeBackend, ConcreteProfileDoesNotRewriteDynamicAxes) {
  const std::string path = write_model(
      "dynamic_profile.onnx",
      identity_model(1, {symbol_dim("batch"), static_dim(4)}, "features", "features_out"));
  const float values[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};

  vie::OnnxRuntimeBackend frozen;
  vie::BackendConfig concrete = cpu_config(path);
  concrete.input.name = "features";
  concrete.input.shape = {1, 4};
  concrete.input.dtype = vie::DataType::Fp32;
  frozen.prepare(concrete);
  vie::BackendRequest batch3;
  batch3.inputs.emplace("features",
                        host_tensor(vie::DataType::Fp32, {3, 4}, values, sizeof(values)));
  expect_failure(vie::ErrorCode::InvalidArgument, "declared profile",
                 [&] { frozen.execute(batch3); });

  vie::OnnxRuntimeBackend open;
  vie::BackendConfig dynamic = cpu_config(path);
  dynamic.input.name = "features";
  dynamic.input.shape = {-1, 4};
  dynamic.input.dtype = vie::DataType::Fp32;
  open.prepare(dynamic);
  const vie::BackendOutput output = open.execute(batch3);
  EXPECT_EQ(output.tensors.at("features_out").tensor.shape(), (std::vector<std::int64_t>{3, 4}));
}

TEST(OnnxRuntimeBackend, PassesSupportedElementTypes) {
  struct Case {
    const char* file;
    int onnx_type;
    vie::DataType dtype;
    std::vector<unsigned char> bytes;
  };
  const std::int32_t int32_values[] = {-1, 0, 42};
  const std::int64_t int64_values[] = {1, -2, 99};
  const std::uint8_t uint8_values[] = {0, 255, 7};
  const std::uint8_t bool_values[] = {1, 0, 1};
  const std::uint16_t fp16_values[] = {0x3C00, 0x4000, 0x0000};
  std::vector<Case> cases = {
      {"id_i32.onnx", 6, vie::DataType::Int32, {}},
      {"id_i64.onnx", 7, vie::DataType::Int64, {}},
      {"id_u8.onnx", 2, vie::DataType::Uint8, {}},
      {"id_bool.onnx", 9, vie::DataType::Bool, {}},
      {"id_f16.onnx", 10, vie::DataType::Fp16, {}},
  };
  auto fill = [](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    return std::vector<unsigned char>(bytes, bytes + size);
  };
  cases[0].bytes = fill(int32_values, sizeof(int32_values));
  cases[1].bytes = fill(int64_values, sizeof(int64_values));
  cases[2].bytes = fill(uint8_values, sizeof(uint8_values));
  cases[3].bytes = fill(bool_values, sizeof(bool_values));
  cases[4].bytes = fill(fp16_values, sizeof(fp16_values));

  for (const Case& item : cases) {
    const std::string path =
        write_model(item.file, identity_model(item.onnx_type, {static_dim(3)}, "x", "y"));
    vie::OnnxRuntimeBackend backend;
    backend.set_intra_op_threads(1);
    backend.prepare(cpu_config(path));
    vie::BackendRequest request;
    request.inputs.emplace("x", host_tensor(item.dtype, {3}, item.bytes.data(), item.bytes.size()));
    const vie::BackendOutput output = backend.execute(request);
    ASSERT_EQ(output.tensors.count("y"), 1u) << item.file;
    const vie::Tensor& tensor = output.tensors.at("y").tensor;
    EXPECT_EQ(tensor.dtype(), item.dtype) << item.file;
    EXPECT_EQ(tensor.shape(), (std::vector<std::int64_t>{3})) << item.file;
    EXPECT_EQ(tensor.nbytes(), item.bytes.size()) << item.file;
    EXPECT_EQ(std::memcmp(tensor.data(), item.bytes.data(), item.bytes.size()), 0) << item.file;
  }
}

TEST(OnnxRuntimeBackend, RejectsBadInputsMissingAndCorruptModels) {
  const std::string path =
      write_model("boundary.onnx", identity_model(1, {static_dim(1), static_dim(4)}, "images", "y"));
  vie::OnnxRuntimeBackend backend;
  backend.set_intra_op_threads(1);
  expect_failure(vie::ErrorCode::InvalidArgument, "not prepared",
                 [&] { backend.execute(vie::BackendRequest{}); });

  vie::BackendConfig config = cpu_config(path);
  config.outputs.push_back(vie::TensorSpec{"missing_out", {1, 4}, vie::DataType::Fp32, {}});
  expect_failure(vie::ErrorCode::InvalidArgument, "onnxruntime load failed:",
                 [&] { backend.prepare(config); });

  config.outputs.clear();
  config.outputs.push_back(vie::TensorSpec{"y", {1, 3}, vie::DataType::Fp32, {}});
  backend.prepare(config);

  const float values[] = {1.f, 2.f, 3.f, 4.f};
  vie::BackendRequest ok;
  ok.inputs.emplace("images", host_tensor(vie::DataType::Fp32, {1, 4}, values, sizeof(values)));
  expect_failure(vie::ErrorCode::Internal, "output conversion failed:", [&] { backend.execute(ok); });

  vie::OnnxRuntimeBackend ready;
  ready.set_intra_op_threads(1);
  ready.prepare(cpu_config(path));

  vie::BackendRequest unknown;
  unknown.inputs.emplace("nope", ok.inputs.at("images"));
  expect_failure(vie::ErrorCode::InvalidArgument, "unexpected input tensor",
                 [&] { ready.execute(unknown); });

  expect_failure(vie::ErrorCode::InvalidArgument, "missing input tensor",
                 [&] { ready.execute(vie::BackendRequest{}); });

  vie::BackendRequest extra = ok;
  extra.inputs.emplace("other", ok.inputs.at("images"));
  expect_failure(vie::ErrorCode::InvalidArgument, "unexpected input tensor",
                [&] { ready.execute(extra); });

  const std::int32_t wrong_type[] = {1, 2, 3, 4};
  vie::BackendRequest dtype = ok;
  dtype.inputs["images"] =
      host_tensor(vie::DataType::Int32, {1, 4}, wrong_type, sizeof(wrong_type));
  expect_failure(vie::ErrorCode::InvalidArgument, "does not match model dtype",
                 [&] { ready.execute(dtype); });

  vie::BackendRequest shape = ok;
  shape.inputs["images"] = host_tensor(vie::DataType::Fp32, {1, 3}, values, 3 * sizeof(float));
  expect_failure(vie::ErrorCode::InvalidArgument, "does not match model shape",
                 [&] { ready.execute(shape); });

  vie::BackendRequest empty = ok;
  empty.inputs["images"] = vie::Tensor{};
  expect_failure(vie::ErrorCode::InvalidArgument, "has no storage", [&] { ready.execute(empty); });

  vie::BackendConfig missing = cpu_config((fixture_dir() / "missing.onnx").string());
  expect_failure(vie::ErrorCode::InvalidArgument, "model file does not exist",
                 [&] { ready.prepare(missing); });

  const std::filesystem::path corrupt = fixture_dir() / "corrupt.onnx";
  {
    std::ofstream output(corrupt, std::ios::binary);
    const char bytes[] = "this is not an onnx model";
    output.write(bytes, sizeof(bytes) - 1);
  }
  expect_failure(vie::ErrorCode::SchemaError, "onnxruntime load failed:", [&] {
    vie::OnnxRuntimeBackend broken;
    broken.prepare(cpu_config(corrupt.string()));
  });

  vie::BackendConfig mock_format = cpu_config(path);
  mock_format.model_format = vie::ModelFormat::Mock;
  expect_failure(vie::ErrorCode::Unsupported, "only executes model format 'onnx'", [&] {
    vie::OnnxRuntimeBackend other;
    other.prepare(mock_format);
  });

  vie::BackendConfig cuda = cpu_config(path);
  cuda.device = vie::DeviceType::Cuda;
  expect_failure(vie::ErrorCode::Unsupported, "only runs on CPU", [&] {
    vie::OnnxRuntimeBackend other;
    other.prepare(cuda);
  });
}

TEST(OnnxRuntimeBackend, MaterializesNonContiguousInput) {
  const std::string path =
      write_model("strided.onnx", identity_model(1, {static_dim(1), static_dim(2)}, "x", "y"));
  vie::Tensor full = vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32, {1, 4});
  float* values = full.mutable_data<float>();
  values[0] = 1.f;
  values[1] = 2.f;
  values[2] = 3.f;
  values[3] = 4.f;
  const vie::Tensor view = full.view({1, 2}, {4, 2}, 0);
  EXPECT_FALSE(view.contiguous());

  vie::OnnxRuntimeBackend backend;
  backend.set_intra_op_threads(1);
  backend.prepare(cpu_config(path));
  vie::BackendRequest request;
  request.inputs.emplace("x", view);
  const vie::BackendOutput output = backend.execute(request);
  EXPECT_TRUE(backend.timing().input_was_copied);
  EXPECT_GT(backend.timing().last_input_copy_ms, 0.0);
  EXPECT_FLOAT_EQ(output.tensors.at("y").tensor.data_as<float>()[0], 1.f);
  EXPECT_FLOAT_EQ(output.tensors.at("y").tensor.data_as<float>()[1], 3.f);
}

TEST(OnnxRuntimeBackend, KeepsOutputAliveAfterBackendDestruction) {
  const std::string path =
      write_model("lifetime.onnx", identity_model(1, {static_dim(2)}, "x", "y"));
  const float input[] = {8.f, 9.f};
  vie::BackendOutput output;
  {
    auto backend = std::make_unique<vie::OnnxRuntimeBackend>();
    backend->set_intra_op_threads(1);
    backend->prepare(cpu_config(path));
    vie::BackendRequest request;
    request.inputs.emplace("x", host_tensor(vie::DataType::Fp32, {2}, input, sizeof(input)));
    const vie::BackendOutput first = backend->execute(request);
    const vie::BackendOutput second = backend->execute(request);
    EXPECT_FALSE(first.tensors.at("y").tensor.shares_storage(second.tensors.at("y").tensor));
    output = first;
    backend.reset();
  }
  EXPECT_EQ(output.tensors.at("y").lifetime, vie::TensorLifetime::Copied);
  EXPECT_FLOAT_EQ(output.tensors.at("y").tensor.data_as<float>()[0], 8.f);
  EXPECT_FLOAT_EQ(output.tensors.at("y").tensor.data_as<float>()[1], 9.f);
}

TEST(OnnxRuntimeBackend, EngineExecutesOnnxWithoutPreprocessing) {
  const std::string path =
      write_model("engine_identity.onnx", identity_model(1, {static_dim(1), static_dim(6)}, "values",
                                                          "output0"));
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
  vie::OutputDesc output;
  output.name = "output0";
  output.shape = {1, 6};
  output.dtype = vie::DataType::Fp32;
  manifest.outputs.push_back(output);
  manifest.decoder.protocol = "mock";

  vie::ModelConfig config;
  config.manifest = manifest;
  config.backend = vie::BackendKind::Auto;
  config.device = vie::DeviceType::Cpu;
  vie::InferenceEngine engine;
  engine.load(config);

  const float input[] = {10.f, 20.f, 30.f, 40.f, 0.5f, 0.f};
  vie::BackendRequest request;
  request.inputs.emplace("values", host_tensor(vie::DataType::Fp32, {1, 6}, input, sizeof(input)));
  const vie::RawPredictions predictions = engine.run(request);
  ASSERT_EQ(predictions.detections.size(), 1u);
  EXPECT_FLOAT_EQ(predictions.detections[0].box.x1, 10.f);
  EXPECT_FLOAT_EQ(predictions.detections[0].box.y1, 20.f);
  EXPECT_FLOAT_EQ(predictions.detections[0].box.x2, 30.f);
  EXPECT_FLOAT_EQ(predictions.detections[0].box.y2, 40.f);
  EXPECT_FLOAT_EQ(predictions.detections[0].score, 0.5f);
  EXPECT_EQ(predictions.detections[0].class_id, 0);
}

TEST(OnnxRuntimeBackend, RunsYoloV8nCpuAndRecordsTiming) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n.onnx");
  vie::OnnxRuntimeBackend backend;
  backend.set_intra_op_threads(4);
  backend.set_inter_op_threads(1);
  backend.set_graph_optimization(vie::OnnxRuntimeOptimization::All);
  backend.set_log_severity(2);
  backend.prepare(cpu_config(path));
  const double load_ms = backend.timing().load_ms;
  EXPECT_GT(load_ms, 0.0);

  vie::BackendRequest request;
  request.inputs.emplace("images", vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32,
                                                      {1, 3, 640, 640}));
  const vie::BackendOutput first = backend.execute(request);
  ASSERT_EQ(first.tensors.size(), 1u);
  ASSERT_EQ(first.tensors.count("predictions"), 1u);
  const vie::Tensor& predictions = first.tensors.at("predictions").tensor;
  EXPECT_EQ(predictions.dtype(), vie::DataType::Fp32);
  EXPECT_EQ(predictions.shape(), (std::vector<std::int64_t>{1, 84, 8400}));
  EXPECT_EQ(first.tensors.at("predictions").lifetime, vie::TensorLifetime::Copied);
  const float* values = predictions.data_as<float>();
  for (std::size_t index = 0; index < predictions.numel(); ++index) {
    ASSERT_TRUE(std::isfinite(values[index])) << index;
  }
  EXPECT_FALSE(backend.timing().input_was_copied);
  EXPECT_GT(backend.timing().last_output_copy_ms, 0.0);
  const double single_ms = backend.timing().last_execute_ms;

  constexpr int kRuns = 100;
  std::vector<double> samples;
  samples.reserve(kRuns);
  for (int index = 0; index < kRuns; ++index) {
    const auto started = std::chrono::steady_clock::now();
    const vie::BackendOutput output = backend.execute(request);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    ASSERT_EQ(output.tensors.at("predictions").tensor.shape(),
              (std::vector<std::int64_t>{1, 84, 8400}));
    samples.push_back(wall_ms);
  }
  std::vector<double> ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  const double mean = sum / static_cast<double>(kRuns);
  const double p50 = ordered[49];
  const double p95 = ordered[94];

  const std::filesystem::path report = fixture_dir() / "yolov8n_bench.txt";
  std::ofstream out(report);
  out << "model=" << path << "\n";
  out << "intra_op_threads=4\n";
  out << "inter_op_threads=1\n";
  out << "graph_optimization=all\n";
  out << "input=1x3x640x640 fp32 contiguous\n";
  out << "output=predictions 1x84x8400 fp32 copied\n";
  out << "load_ms=" << load_ms << "\n";
  out << "single_execute_ms=" << single_ms << "\n";
  out << "runs=" << kRuns << "\n";
  out << "mean_wall_ms=" << mean << "\n";
  out << "p50_wall_ms=" << p50 << "\n";
  out << "p95_wall_ms=" << p95 << "\n";
  out << "min_wall_ms=" << ordered.front() << "\n";
  out << "max_wall_ms=" << ordered.back() << "\n";
  out << "last_input_copy_ms=" << backend.timing().last_input_copy_ms << "\n";
  out << "last_output_copy_ms=" << backend.timing().last_output_copy_ms << "\n";
  out << "input_was_copied=" << (backend.timing().input_was_copied ? 1 : 0) << "\n";
  EXPECT_TRUE(out);
  EXPECT_LT(p50, p95);
  EXPECT_GT(mean, 0.0);
}
