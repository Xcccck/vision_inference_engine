#include "test_util.hpp"

#include "vie/decoder/mock_decoder.hpp"
#include "vie/engine/inference_engine.hpp"

namespace {

vie::ModelManifest load_mock() {
  return vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
}

vie::ModelConfig mock_config() {
  vie::ModelConfig config;
  config.manifest = load_mock();
  config.backend = vie::BackendKind::Auto;
  config.device = vie::DeviceType::Cpu;
  return config;
}

vie::BackendRequest input_request() {
  vie::BackendRequest request;
  request.inputs.emplace("images", vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32,
                                                      {1, 3, 8, 8}));
  return request;
}

}  // namespace

TEST(EngineTest, MockPipelineDecodesNamedOutput) {
  vie::InferenceEngine engine;
  EXPECT_TRUE(engine.backends().contains(vie::BackendKind::Mock));
  EXPECT_TRUE(engine.decoders().contains("mock"));
  const std::unique_ptr<vie::IInferenceBackend> created =
      engine.backends().create(vie::BackendKind::Mock);
  EXPECT_EQ(created->info().name, "mock");
  EXPECT_FALSE(created->info().supports_cuda);

  engine.load(mock_config());
  EXPECT_TRUE(engine.loaded());
  const vie::FieldProvenance* family = engine.manifest().find_provenance("model_family");
  ASSERT_NE(family, nullptr);
  EXPECT_EQ(family->certainty, vie::Certainty::Unresolved);
  EXPECT_EQ(engine.manifest().outputs.at(0).layout, vie::Layout::Unknown);

  const vie::BackendRequest request = input_request();
  const vie::BackendOutput first = engine.execute(request);
  const vie::BackendOutput second = engine.execute(request);
  ASSERT_EQ(first.tensors.count("output0"), 1u);
  EXPECT_EQ(first.tensors.at("output0").lifetime, vie::TensorLifetime::Copied);
  EXPECT_FALSE(first.tensors.at("output0").tensor.shares_storage(second.tensors.at("output0").tensor));

  const vie::RawPredictions predictions = engine.run(request);
  EXPECT_EQ(predictions.protocol, "mock");
  EXPECT_EQ(predictions.box_format, "xyxy");
  ASSERT_EQ(predictions.detections.size(), 1u);
  EXPECT_FLOAT_EQ(predictions.detections[0].box.x1, 1.f);
  EXPECT_FLOAT_EQ(predictions.detections[0].box.y1, 2.f);
  EXPECT_FLOAT_EQ(predictions.detections[0].box.x2, 3.f);
  EXPECT_FLOAT_EQ(predictions.detections[0].box.y2, 4.f);
  EXPECT_FLOAT_EQ(predictions.detections[0].score, 0.9f);
  EXPECT_EQ(predictions.detections[0].class_id, 0);
  engine.synchronize();
}

TEST(EngineTest, RejectsUnresolvedAndUnavailableBackends) {
  vie::InferenceEngine engine;
  vie::ModelConfig missing_color = mock_config();
  missing_color.manifest->input.color.clear();
  missing_color.manifest->provenance["input.color"] =
      vie::FieldProvenance{vie::Certainty::Unresolved, {}};
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField, engine.load(missing_color));
  EXPECT_FALSE(engine.loaded());

  vie::ModelConfig cuda = mock_config();
  cuda.device = vie::DeviceType::Cuda;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(cuda));

  vie::ModelConfig ort = mock_config();
  ort.backend = vie::BackendKind::OnnxRuntime;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(ort));

  vie::ModelConfig trt = mock_config();
  trt.backend = vie::BackendKind::TensorRT;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(trt));

  vie::ModelConfig onnx = mock_config();
  onnx.manifest->model_format = vie::ModelFormat::Onnx;
  onnx.manifest->provenance["model_format"] =
      vie::FieldProvenance{vie::Certainty::User, {"test"}};
  if (engine.backends().contains(vie::BackendKind::OnnxRuntime)) {
    VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.load(onnx));
  } else {
    VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(onnx));
  }

  vie::ModelConfig unknown_protocol = mock_config();
  unknown_protocol.manifest->decoder.protocol = "yolo_raw_v8";
  unknown_protocol.manifest->provenance["decoder.protocol"] =
      vie::FieldProvenance{vie::Certainty::User, {"test"}};
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(unknown_protocol));

  vie::ModelConfig no_manifest;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, engine.load(no_manifest));
  EXPECT_FALSE(engine.loaded());
}

TEST(EngineTest, RejectsInputShapeMismatch) {
  vie::InferenceEngine engine;
  engine.load(mock_config());
  vie::BackendRequest request;
  request.inputs.emplace("images", vie::Tensor::zeros(vie::DeviceContext{}, vie::DataType::Fp32,
                                                      {1, 3, 4, 4}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, engine.run(request));
}

TEST(EngineTest, MockDecoderUsesOutputNameAndClassCount) {
  const float values[6] = {0.f, 1.f, 2.f, 3.f, 0.5f, 3.f};
  const vie::Tensor tensor = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 6},
                                                    values, sizeof(values));
  vie::DecoderContext context;
  context.protocol = "mock";
  context.box_format = "xyxy";
  context.num_classes = 1;
  vie::TensorSpec spec;
  spec.name = "output0";
  spec.shape = {1, 6};
  spec.dtype = vie::DataType::Fp32;
  context.outputs.push_back(spec);

  std::map<std::string, vie::Tensor> outputs;
  outputs.emplace("other", tensor);
  const vie::MockDecoder decoder;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decoder.decode(outputs, context));

  outputs.clear();
  outputs.emplace("output0", tensor);
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decoder.decode(outputs, context));

  const float short_values[2] = {1.f, 2.f};
  outputs["output0"] = vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {2},
                                              short_values, sizeof(short_values));
  context.num_classes.reset();
  const vie::RawPredictions empty = decoder.decode(outputs, context);
  EXPECT_TRUE(empty.detections.empty());
}
