#include "onnx_test_writer.hpp"
#include "test_util.hpp"
#include "vie/analyzer/onnx_analyzer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void write_bytes(const std::filesystem::path& path, const std::string& bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output) << path.string();
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(output);
}

const vie::ValueEndpoint* find_endpoint(const std::vector<vie::ValueEndpoint>& endpoints,
                                        const std::string& name) {
  for (const vie::ValueEndpoint& endpoint : endpoints) {
    if (endpoint.name == name) return &endpoint;
  }
  return nullptr;
}

const vie::InitializerInfo* find_initializer(const std::vector<vie::InitializerInfo>& initializers,
                                             const std::string& name) {
  for (const vie::InitializerInfo& initializer : initializers) {
    if (initializer.name == name) return &initializer;
  }
  return nullptr;
}

bool has_warning(const vie::ModelAnalysisReport& report, const std::string& code) {
  for (const vie::AnalysisWarning& warning : report.warnings) {
    if (warning.code == code) return true;
  }
  return false;
}

std::int64_t operator_count(const vie::ModelAnalysisReport& report, const std::string& domain,
                            const std::string& op_type) {
  for (const vie::OperatorStat& stat : report.operators) {
    if (stat.domain == domain && stat.op_type == op_type) return stat.count;
  }
  return 0;
}

void expect_static(const vie::TensorShape& shape, const std::vector<std::int64_t>& expected) {
  ASSERT_TRUE(shape.rank_known);
  ASSERT_EQ(shape.dimensions.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(shape.dimensions[i].kind, vie::DimensionKind::Static);
    EXPECT_EQ(shape.dimensions[i].value, expected[i]);
  }
}

vie::ModelAnalysisReport analyze(const OnnxTestWriter& model, const std::string& name) {
  vie::OnnxAnalyzer analyzer;
  return analyzer.analyze_bytes(model.str(), name, ".");
}

OnnxTestWriter fixed_shapes_model() {
  OnnxTestWriter scale = tensor_message("scale", 1, {1}, {}, std::string(4, '\0'), false);
  OnnxTestWriter graph;
  graph.message(1, node_message("Mul", {"images", "scale"}, {"scaled"}));
  graph.message(1, node_message("Relu", {"scaled"}, {"y"}));
  graph.text(2, "fixed");
  graph.message(5, scale);
  graph.message(11, value_info("images", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                               static_dim(8)})));
  graph.message(12, value_info("y", tensor_type(1, true, {static_dim(1), static_dim(3), static_dim(8),
                                                          static_dim(8)})));

  OnnxTestWriter model;
  model.i64(1, 10);
  model.text(2, "vie-test");
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  model.message(14, string_entry("author", "vie"));
  return model;
}

OnnxTestWriter dynamic_multi_io_model() {
  OnnxTestWriter graph;
  graph.message(1, node_message("Relu", {"a"}, {"a_out"}));
  graph.message(1, node_message("Identity", {"b"}, {"b_out"}));
  graph.message(1, node_message("Probe", {"a"}, {"seq"}, {}, "com.vie.test", "probe"));
  graph.text(2, "dynamic");
  graph.message(11, value_info("a", tensor_type(1, true, {symbol_dim("batch"), static_dim(3),
                                                          symbol_dim("height"), symbol_dim("width")})));
  graph.message(11, value_info("b", tensor_type(7, true, {symbol_dim("batch")})));
  graph.message(11, value_info("c", tensor_type(2, true, {unknown_dim(), static_dim(8), static_dim(8)})));
  graph.message(12, value_info("a_out", tensor_type(1, true, {static_dim(1), static_dim(4)})));
  graph.message(12, value_info("b_out", tensor_type(7, true, {symbol_dim("batch")})));
  graph.message(12, value_info("seq", sequence_of_float()));
  graph.message(13, value_info("a_out", tensor_type(1, true, {static_dim(9), static_dim(9), static_dim(9),
                                                              static_dim(9)})));

  OnnxTestWriter model;
  model.i64(1, 10);
  model.text(2, "vie-test");
  model.message(7, graph);
  model.message(8, opset_import("", 13));
  model.message(8, opset_import("ai.onnx.ml", 3));
  return model;
}

OnnxTestWriter branch_graph(std::string_view op_type) {
  OnnxTestWriter graph;
  graph.message(1, node_message(op_type, {"x"}, {"t"}));
  graph.message(12, value_info("t", tensor_type(10, false, {})));
  return graph;
}

OnnxTestWriter external_tensor(std::string_view name, int data_type, const std::vector<std::int64_t>& dims,
                               std::string_view location, std::string_view length) {
  OnnxTestWriter tensor = tensor_message(name, data_type, dims, {}, {}, false);
  tensor.message(13, string_entry("location", location));
  tensor.message(13, string_entry("offset", "0"));
  tensor.message(13, string_entry("length", length));
  tensor.i64(14, 1);
  return tensor;
}

OnnxTestWriter external_subgraph_model() {
  OnnxTestWriter graph;
  graph.message(1, node_message("MatMul", {"x", "w"}, {"mm"}, {}, "", "matmul"));
  graph.message(1, node_message("If", {"cond"}, {"y_if"},
                                {attr_graph("then_branch", branch_graph("Relu")),
                                 attr_graph("else_branch", branch_graph("Abs"))},
                                "", "choose"));
  graph.text(2, "external");
  graph.message(5, tensor_message("bias", 10, {4}, {}, std::string(8, '\0'), false));
  graph.message(5, external_tensor("w", 1, {4, 4}, "weights.bin", "64"));
  graph.message(5, external_tensor("gone", 6, {2}, "missing.bin", "8"));
  graph.message(11, value_info("x", tensor_type(10, true, {static_dim(2), static_dim(4)})));
  graph.message(11, value_info("cond", tensor_type(9, true, {})));
  graph.message(11, value_info("scale64", tensor_type(11, true, {static_dim(1)})));
  graph.message(11, value_info("bias", tensor_type(10, true, {static_dim(4)})));
  graph.message(12, value_info("mm", tensor_type(1, true, {static_dim(2), static_dim(5)})));
  graph.message(12, value_info("y_if", tensor_type(10, false, {})));

  OnnxTestWriter model;
  model.i64(1, 10);
  model.text(2, "vie-test");
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

}  // namespace

TEST(OnnxAnalyzer, FixedShapesKeepDeclaredAndPropagatedApart) {
  const std::filesystem::path path = vie_fixture("testdata/onnx/fixed_shapes.onnx");
  write_bytes(path, fixed_shapes_model().str());

  vie::OnnxAnalyzer analyzer;
  const vie::ModelAnalysisReport report = analyzer.analyze_file(path.string());

  EXPECT_EQ(report.scope, "onnx_structure");
  EXPECT_EQ(report.ir_version, 10);
  EXPECT_EQ(report.producer_name, "vie-test");
  ASSERT_EQ(report.metadata.size(), 1u);
  EXPECT_EQ(report.metadata[0].first, "author");
  ASSERT_EQ(report.opsets.size(), 1u);
  EXPECT_EQ(report.opsets[0].raw_domain, "");
  EXPECT_EQ(report.opsets[0].domain, "ai.onnx");
  EXPECT_EQ(report.opsets[0].version, 17);
  ASSERT_EQ(report.runtime_inputs.size(), 1u);
  expect_static(report.runtime_inputs[0].type.declared_shape, {1, 3, 8, 8});
  EXPECT_EQ(report.runtime_inputs[0].type.element_type.runtime_type, vie::DataType::Fp32);
  ASSERT_EQ(report.outputs.size(), 1u);
  expect_static(report.outputs[0].type.declared_shape, {1, 3, 8, 8});
  const vie::PropagatedShape* propagated = vie::find_propagated_shape(report, "y");
  ASSERT_NE(propagated, nullptr);
  expect_static(propagated->shape, {1, 3, 8, 8});
  EXPECT_EQ(propagated->domain, "ai.onnx");
  EXPECT_EQ(propagated->op_type, "Relu");
  EXPECT_EQ(operator_count(report, "ai.onnx", "Mul"), 1);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Relu"), 1);
  ASSERT_NE(find_initializer(report.initializers, "scale"), nullptr);
  EXPECT_TRUE(report.runtime_inputs.size() == 1u);
  EXPECT_FALSE(has_warning(report, "declared_shape_conflicts_with_propagation"));
}

TEST(OnnxAnalyzer, DynamicShapesSymbolsAndMultipleDomains) {
  const std::filesystem::path path = vie_fixture("testdata/onnx/dynamic_multi_io.onnx");
  write_bytes(path, dynamic_multi_io_model().str());

  vie::OnnxAnalyzer analyzer;
  const vie::ModelAnalysisReport report = analyzer.analyze_file(path.string());

  ASSERT_EQ(report.opsets.size(), 2u);
  EXPECT_EQ(report.opsets[1].raw_domain, "ai.onnx.ml");
  EXPECT_EQ(report.opsets[1].domain, "ai.onnx.ml");
  ASSERT_EQ(report.runtime_inputs.size(), 3u);
  const vie::ValueEndpoint* input_a = find_endpoint(report.runtime_inputs, "a");
  ASSERT_NE(input_a, nullptr);
  ASSERT_EQ(input_a->type.declared_shape.dimensions.size(), 4u);
  EXPECT_EQ(input_a->type.declared_shape.dimensions[0].kind, vie::DimensionKind::Symbolic);
  EXPECT_EQ(input_a->type.declared_shape.dimensions[0].symbol, "batch");
  EXPECT_EQ(input_a->type.declared_shape.dimensions[2].symbol, "height");
  const vie::ValueEndpoint* input_b = find_endpoint(report.runtime_inputs, "b");
  ASSERT_NE(input_b, nullptr);
  EXPECT_EQ(input_b->type.element_type.name, "int64");
  EXPECT_FALSE(input_b->type.element_type.runtime_type.has_value());
  const vie::ValueEndpoint* input_c = find_endpoint(report.runtime_inputs, "c");
  ASSERT_NE(input_c, nullptr);
  EXPECT_EQ(input_c->type.element_type.runtime_type, vie::DataType::Uint8);
  EXPECT_EQ(input_c->type.declared_shape.dimensions[0].kind, vie::DimensionKind::Unknown);
  EXPECT_NE(input_c->type.declared_shape.dimensions[0].kind, vie::DimensionKind::Static);

  const vie::ValueEndpoint* output_a = find_endpoint(report.outputs, "a_out");
  ASSERT_NE(output_a, nullptr);
  expect_static(output_a->type.declared_shape, {1, 4});
  const vie::PropagatedShape* propagated = vie::find_propagated_shape(report, "a_out");
  ASSERT_NE(propagated, nullptr);
  EXPECT_EQ(propagated->shape.dimensions[0].symbol, "batch");
  EXPECT_EQ(propagated->shape.dimensions[3].symbol, "width");
  ASSERT_EQ(report.value_infos.size(), 1u);
  expect_static(report.value_infos[0].type.declared_shape, {9, 9, 9, 9});
  EXPECT_TRUE(has_warning(report, "value_info_disagrees_with_declared"));
  EXPECT_TRUE(has_warning(report, "declared_shape_conflicts_with_propagation"));

  const vie::ValueEndpoint* sequence = find_endpoint(report.outputs, "seq");
  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(sequence->type.kind, vie::OnnxTypeKind::Sequence);
  EXPECT_EQ(sequence->type.nested_kind, vie::OnnxTypeKind::Tensor);
  EXPECT_FALSE(sequence->type.declared_shape.rank_known);
  EXPECT_EQ(operator_count(report, "com.vie.test", "Probe"), 1);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Probe"), 0);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Relu"), 1);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Identity"), 1);
  EXPECT_TRUE(has_warning(report, "node_domain_not_imported"));
  EXPECT_EQ(vie::find_propagated_shape(report, "seq"), nullptr);
}

TEST(OnnxAnalyzer, ExternalDataSubgraphAndDtypeCoverage) {
  const std::filesystem::path directory = vie_fixture("testdata/onnx");
  write_bytes(directory / "weights.bin", std::string(64, '\0'));
  write_bytes(directory / "external_and_subgraph.onnx", external_subgraph_model().str());

  vie::OnnxAnalyzer analyzer;
  const vie::ModelAnalysisReport report = analyzer.analyze_file((directory / "external_and_subgraph.onnx").string());

  ASSERT_EQ(report.graph_inputs.size(), 4u);
  ASSERT_EQ(report.runtime_inputs.size(), 3u);
  EXPECT_EQ(report.runtime_inputs[0].name, "x");
  EXPECT_EQ(report.runtime_inputs[0].type.element_type.runtime_type, vie::DataType::Fp16);
  EXPECT_EQ(report.runtime_inputs[1].name, "cond");
  EXPECT_TRUE(report.runtime_inputs[1].type.declared_shape.rank_known);
  EXPECT_TRUE(report.runtime_inputs[1].type.declared_shape.dimensions.empty());
  EXPECT_EQ(report.runtime_inputs[2].name, "scale64");
  EXPECT_EQ(report.runtime_inputs[2].type.element_type.name, "double");
  EXPECT_FALSE(report.runtime_inputs[2].type.element_type.runtime_type.has_value());

  const vie::InitializerInfo* bias = find_initializer(report.initializers, "bias");
  ASSERT_NE(bias, nullptr);
  EXPECT_TRUE(bias->listed_as_graph_input);
  EXPECT_EQ(find_endpoint(report.runtime_inputs, "bias"), nullptr);
  const vie::InitializerInfo* weights = find_initializer(report.initializers, "w");
  ASSERT_NE(weights, nullptr);
  EXPECT_EQ(weights->payload, vie::TensorPayloadKind::External);
  EXPECT_TRUE(weights->external_file_found);
  EXPECT_EQ(weights->external_file_bytes, 64u);
  const vie::InitializerInfo* missing = find_initializer(report.initializers, "gone");
  ASSERT_NE(missing, nullptr);
  EXPECT_FALSE(missing->external_file_found);
  EXPECT_TRUE(has_warning(report, "external_data_missing"));

  const vie::ValueEndpoint* mm = find_endpoint(report.outputs, "mm");
  ASSERT_NE(mm, nullptr);
  expect_static(mm->type.declared_shape, {2, 5});
  const vie::PropagatedShape* propagated_mm = vie::find_propagated_shape(report, "mm");
  ASSERT_NE(propagated_mm, nullptr);
  expect_static(propagated_mm->shape, {2, 4});
  const vie::ValueEndpoint* y_if = find_endpoint(report.outputs, "y_if");
  ASSERT_NE(y_if, nullptr);
  EXPECT_FALSE(y_if->type.declared_shape.rank_known);
  const vie::PropagatedShape* propagated_if = vie::find_propagated_shape(report, "y_if");
  ASSERT_NE(propagated_if, nullptr);
  expect_static(propagated_if->shape, {2, 4});
  EXPECT_EQ(propagated_if->op_type, "If");

  int subgraph_nodes = 0;
  for (const vie::OnnxNode& node : report.nodes) {
    if (node.depth == 1) {
      ++subgraph_nodes;
      EXPECT_EQ(node.domain, "ai.onnx");
      EXPECT_NE(node.graph_path.find("choose/"), std::string::npos);
    }
  }
  EXPECT_EQ(subgraph_nodes, 2);
  EXPECT_EQ(operator_count(report, "ai.onnx", "MatMul"), 1);
  EXPECT_EQ(operator_count(report, "ai.onnx", "If"), 1);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Relu"), 1);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Abs"), 1);
  EXPECT_TRUE(has_warning(report, "declared_shape_conflicts_with_propagation"));
}

TEST(OnnxAnalyzer, StaticZeroStaysStaticAndKeepsDenotation) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Identity", {"z"}, {"y"}));
  graph.message(11, value_info("z", tensor_type(1, true, {static_dim(0), static_dim(4, "WIDTH")})));
  graph.message(12, value_info("y", tensor_type(1, true, {static_dim(0), static_dim(4, "WIDTH")})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));

  const vie::ModelAnalysisReport report = analyze(model, "static-zero.onnx");
  ASSERT_EQ(report.runtime_inputs[0].type.declared_shape.dimensions.size(), 2u);
  EXPECT_EQ(report.runtime_inputs[0].type.declared_shape.dimensions[0].kind, vie::DimensionKind::Static);
  EXPECT_EQ(report.runtime_inputs[0].type.declared_shape.dimensions[0].value, 0);
  EXPECT_EQ(report.runtime_inputs[0].type.declared_shape.dimensions[1].denotation, "WIDTH");
  const vie::PropagatedShape* propagated = vie::find_propagated_shape(report, "y");
  ASSERT_NE(propagated, nullptr);
  expect_static(propagated->shape, {0, 4});
}

TEST(OnnxAnalyzer, UnpackedDimsAndReshape) {
  OnnxTestWriter shape = tensor_message("shape", 7, {2}, {6, 4}, {}, true);
  OnnxTestWriter graph;
  graph.message(1, node_message("Reshape", {"x", "shape"}, {"y"}));
  graph.message(5, shape);
  graph.message(11, value_info("x", tensor_type(1, true, {static_dim(2), static_dim(3), static_dim(4)})));
  graph.message(12, value_info("y", tensor_type(1, false, {})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));

  const vie::ModelAnalysisReport report = analyze(model, "reshape.onnx");
  const vie::InitializerInfo* encoded = find_initializer(report.initializers, "shape");
  ASSERT_NE(encoded, nullptr);
  expect_static(encoded->declared_shape, {2});
  EXPECT_FALSE(report.outputs[0].type.declared_shape.rank_known);
  const vie::PropagatedShape* propagated = vie::find_propagated_shape(report, "y");
  ASSERT_NE(propagated, nullptr);
  expect_static(propagated->shape, {6, 4});
}

TEST(OnnxAnalyzer, ConcatMatchesNonAxisDimensions) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Concat", {"a", "b"}, {"y"}, {attr_int("axis", 0)}));
  graph.message(11, value_info("a", tensor_type(1, true, {static_dim(1), static_dim(2)})));
  graph.message(11, value_info("b", tensor_type(1, true, {static_dim(1), static_dim(2)})));
  graph.message(12, value_info("y", tensor_type(1, false, {})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 13));
  const vie::ModelAnalysisReport matched = analyze(model, "concat.onnx");
  const vie::PropagatedShape* propagated = vie::find_propagated_shape(matched, "y");
  ASSERT_NE(propagated, nullptr);
  expect_static(propagated->shape, {2, 2});

  OnnxTestWriter mismatch_graph;
  mismatch_graph.message(1, node_message("Concat", {"a", "b"}, {"y"}, {attr_int("axis", 0)}));
  mismatch_graph.message(11, value_info("a", tensor_type(1, true, {static_dim(1), static_dim(3)})));
  mismatch_graph.message(11, value_info("b", tensor_type(1, true, {static_dim(2), static_dim(1)})));
  mismatch_graph.message(12, value_info("y", tensor_type(1, false, {})));
  OnnxTestWriter mismatch;
  mismatch.i64(1, 10);
  mismatch.message(7, mismatch_graph);
  mismatch.message(8, opset_import("", 13));
  const vie::ModelAnalysisReport rejected = analyze(mismatch, "concat-mismatch.onnx");
  EXPECT_EQ(vie::find_propagated_shape(rejected, "y"), nullptr);
}

TEST(OnnxAnalyzer, ValueInfoDoesNotSeedPropagation) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Relu", {"images"}, {"y"}));
  graph.message(11, value_info("images", tensor_type(1, false, {})));
  graph.message(12, value_info("y", tensor_type(1, false, {})));
  graph.message(13, value_info("images", tensor_type(1, true, {static_dim(4), static_dim(4)})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));

  const vie::ModelAnalysisReport report = analyze(model, "value-info.onnx");
  EXPECT_FALSE(report.runtime_inputs[0].type.declared_shape.rank_known);
  expect_static(report.value_infos[0].type.declared_shape, {4, 4});
  EXPECT_EQ(vie::find_propagated_shape(report, "y"), nullptr);
  EXPECT_FALSE(has_warning(report, "value_info_disagrees_with_declared"));
}

TEST(OnnxAnalyzer, CustomDomainReluIsNotShapePreserving) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Relu", {"x"}, {"y"}, {}, "com.example"));
  graph.message(11, value_info("x", tensor_type(1, true, {static_dim(2), static_dim(2)})));
  graph.message(12, value_info("y", tensor_type(1, false, {})));
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));

  const vie::ModelAnalysisReport report = analyze(model, "custom-relu.onnx");
  EXPECT_EQ(vie::find_propagated_shape(report, "y"), nullptr);
  EXPECT_EQ(operator_count(report, "com.example", "Relu"), 1);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Relu"), 0);
  EXPECT_TRUE(has_warning(report, "node_domain_not_imported"));
}

TEST(OnnxAnalyzer, FunctionAndSparseAndTrainingStayStructural) {
  OnnxTestWriter values = tensor_message("sp", 1, {2}, {}, std::string(8, '\0'), false);
  OnnxTestWriter sparse;
  sparse.message(1, values);
  sparse.packed_i64(3, {3});

  OnnxTestWriter graph;
  graph.text(2, "edges");
  graph.message(15, sparse);

  OnnxTestWriter function;
  function.text(1, "Scale");
  function.message(7, node_message("Mul", {"a", "b"}, {"c"}));

  OnnxTestWriter model;
  model.i64(1, 9);
  model.message(7, graph);
  model.message(8, opset_import("ai.onnx", 17));
  model.message(20, OnnxTestWriter{});
  model.message(25, function);

  const vie::ModelAnalysisReport report = analyze(model, "edges.onnx");
  EXPECT_TRUE(report.initializers.empty());
  ASSERT_EQ(report.sparse_initializers.size(), 1u);
  EXPECT_EQ(report.sparse_initializers[0].name, "sp");
  EXPECT_TRUE(report.sparse_initializers[0].sparse);
  expect_static(report.sparse_initializers[0].declared_shape, {3});
  ASSERT_EQ(report.functions.size(), 1u);
  EXPECT_EQ(report.functions[0].name, "Scale");
  EXPECT_EQ(report.functions[0].node_count, 1);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Mul"), 1);
  EXPECT_TRUE(report.propagated_shapes.empty());
  EXPECT_EQ(report.training_info_count, 1);
  EXPECT_NE(std::find(report.not_interpreted.begin(), report.not_interpreted.end(), "training_info"),
            report.not_interpreted.end());
  EXPECT_NE(std::find(report.not_interpreted.begin(), report.not_interpreted.end(),
                      "function body shape propagation"),
            report.not_interpreted.end());
}

TEST(OnnxAnalyzer, MalformedAndMissingFiles) {
  vie::OnnxAnalyzer analyzer;
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError, analyzer.analyze_bytes(std::string("\x0a", 1), "bad.onnx", "."));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, analyzer.analyze_file("/tmp/vie-missing-model.onnx"));
}

TEST(OnnxAnalyzer, JsonIsAViewOfTheReport) {
  const vie::ModelAnalysisReport report = analyze(fixed_shapes_model(), "fixed.onnx");
  const std::string json = vie::analysis_to_json(report);
  EXPECT_NE(json.find("\"scope\""), std::string::npos);
  EXPECT_NE(json.find("\"propagated_shapes\""), std::string::npos);
  EXPECT_NE(json.find("\"declared_shape\""), std::string::npos);
  EXPECT_EQ(json.find("\"task\":"), std::string::npos);
  EXPECT_EQ(json.find("\"decoder\":"), std::string::npos);
  EXPECT_EQ(json.find("\"normalization\":"), std::string::npos);
  EXPECT_EQ(json.find("\"nms\":"), std::string::npos);
  EXPECT_EQ(json.find("\"model_family\":"), std::string::npos);
  EXPECT_NE(json.find("\"omitted\""), std::string::npos);
}

namespace {

std::string encode_varint(std::uint64_t value) {
  std::string out;
  while (value >= 0x80u) {
    out.push_back(static_cast<char>((value & 0x7fu) | 0x80u));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
  return out;
}

std::string length_delimited(std::uint64_t field, const std::string& payload) {
  std::string out = encode_varint((field << 3) | 2);
  out += encode_varint(payload.size());
  out += payload;
  return out;
}

OnnxTestWriter model_with_initializer(const OnnxTestWriter& tensor) {
  OnnxTestWriter graph;
  graph.message(5, tensor);
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

OnnxTestWriter nested_identity(int remaining) {
  OnnxTestWriter graph;
  graph.message(1, node_message("Identity", {"x"}, {"y"}));
  if (remaining > 0) {
    graph.message(1, node_message("If", {"c"}, {"z"}, {attr_graph("then_branch", nested_identity(remaining - 1))}));
  }
  return graph;
}

OnnxTestWriter wrap_graph(const OnnxTestWriter& graph) {
  OnnxTestWriter model;
  model.i64(1, 10);
  model.message(7, graph);
  model.message(8, opset_import("", 17));
  return model;
}

}  // namespace

TEST(OnnxAnalyzer, RejectsAliasedFieldNumbersAndBadWire) {
  vie::OnnxAnalyzer analyzer;
  const std::uint64_t aliased_producer = (std::uint64_t{1} << 32) + 2;
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   analyzer.analyze_bytes(length_delimited(aliased_producer, "hijack"), "alias.onnx", "."));
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError, analyzer.analyze_bytes(encode_varint((2ull << 3) | 7), "wire.onnx", "."));
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   analyzer.analyze_bytes(length_delimited(2, "short").substr(0, 2), "truncated.onnx", "."));
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError, analyzer.analyze_bytes(std::string(10, '\x80'), "wide.onnx", "."));

  const std::string skipped = length_delimited(100, "xyz") + fixed_shapes_model().str();
  const vie::ModelAnalysisReport report = analyzer.analyze_bytes(skipped, "unknown-field.onnx", ".");
  EXPECT_EQ(report.producer_name, "vie-test");
  EXPECT_EQ(report.ir_version, 10);
}

TEST(OnnxAnalyzer, PackedFieldsStayBounded) {
  vie::OnnxAnalyzer analyzer;
  OnnxTestWriter truncated;
  truncated.bytes(1, std::string("\x80", 1));
  truncated.i64(2, 1);
  truncated.text(8, "w");
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   analyzer.analyze_bytes(model_with_initializer(truncated).str(), "packed.onnx", "."));

  const OnnxTestWriter wide = tensor_message("w", 1, std::vector<std::int64_t>(33, 1), {}, {}, false);
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   analyzer.analyze_bytes(model_with_initializer(wide).str(), "rank.onnx", "."));

  std::vector<std::int64_t> values(300, 1);
  const vie::ModelAnalysisReport report =
      analyze(model_with_initializer(tensor_message("big", 7, {300}, values, {}, false)), "many-int64.onnx");
  const vie::InitializerInfo* big = find_initializer(report.initializers, "big");
  ASSERT_NE(big, nullptr);
  expect_static(big->declared_shape, {300});
  EXPECT_EQ(big->typed_element_count, 300u);
}

TEST(OnnxAnalyzer, SubgraphDepthIsBounded) {
  vie::OnnxAnalyzer analyzer;
  const vie::ModelAnalysisReport shallow =
      analyze(wrap_graph(nested_identity(2)), "shallow-subgraph.onnx");
  EXPECT_EQ(operator_count(shallow, "ai.onnx", "If"), 2);
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   analyzer.analyze_bytes(wrap_graph(nested_identity(17)).str(), "deep-subgraph.onnx", "."));
}

TEST(OnnxAnalyzer, ExternalPathsStayInsideTheModelDirectory) {
  const std::filesystem::path root = "/tmp/vie-external-root";
  const std::filesystem::path model_dir = root / "model";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(model_dir / "nested");
  const std::string secret(123, 's');
  write_bytes(root / "secret.bin", secret);
  write_bytes(model_dir / "weights.bin", std::string(4, 'w'));
  std::filesystem::create_symlink(root / "secret.bin", model_dir / "escape.bin");

  vie::OnnxAnalyzer analyzer;
  const auto check = [&](const std::string& location, const std::string& name) {
    return analyzer.analyze_bytes(model_with_initializer(external_tensor(name, 1, {1}, location, "1")).str(),
                                  name + ".onnx", model_dir.string());
  };

  const vie::ModelAnalysisReport absolute = check("/tmp/vie-external-root/secret.bin", "absolute");
  const vie::InitializerInfo* absolute_weight = find_initializer(absolute.initializers, "absolute");
  ASSERT_NE(absolute_weight, nullptr);
  EXPECT_FALSE(absolute_weight->external_file_found);
  EXPECT_EQ(absolute_weight->external_file_bytes, 0u);
  EXPECT_TRUE(has_warning(absolute, "external_data_path_rejected"));

  const vie::ModelAnalysisReport escaped = check("../secret.bin", "escaped");
  EXPECT_FALSE(find_initializer(escaped.initializers, "escaped")->external_file_found);
  EXPECT_TRUE(has_warning(escaped, "external_data_path_rejected"));

  const vie::ModelAnalysisReport linked = check("escape.bin", "linked");
  EXPECT_FALSE(find_initializer(linked.initializers, "linked")->external_file_found);
  EXPECT_EQ(find_initializer(linked.initializers, "linked")->external_file_bytes, 0u);
  EXPECT_TRUE(has_warning(linked, "external_data_path_rejected"));

  const vie::ModelAnalysisReport normalized = check("nested/../weights.bin", "normalized");
  const vie::InitializerInfo* kept = find_initializer(normalized.initializers, "normalized");
  ASSERT_NE(kept, nullptr);
  EXPECT_TRUE(kept->external_file_found);
  EXPECT_EQ(kept->external_file_bytes, 4u);

  OnnxTestWriter bad_offset = external_tensor("bad", 1, {1}, "weights.bin", "1");
  bad_offset.message(13, string_entry("offset", "nope"));
  const vie::ModelAnalysisReport bad =
      analyzer.analyze_bytes(model_with_initializer(bad_offset).str(), "bad-offset.onnx", model_dir.string());
  EXPECT_TRUE(has_warning(bad, "external_data_bad_range"));

  OnnxTestWriter past = tensor_message("past", 1, {1}, {}, {}, false);
  past.message(13, string_entry("location", "weights.bin"));
  past.message(13, string_entry("offset", "100"));
  past.i64(14, 1);
  const vie::ModelAnalysisReport out_of_range =
      analyzer.analyze_bytes(model_with_initializer(past).str(), "past.onnx", model_dir.string());
  EXPECT_TRUE(find_initializer(out_of_range.initializers, "past")->external_file_found);
  EXPECT_TRUE(has_warning(out_of_range, "external_data_out_of_range"));
}

TEST(OnnxAnalyzer, ParsesThirdPartyYoloV8Export) {
  VIE_REQUIRE_MODEL("testdata/onnx/yolov8n.onnx");
  const std::filesystem::path path = vie_fixture("testdata/onnx/yolov8n.onnx");
  ASSERT_EQ(std::filesystem::file_size(path), 12834838u);
  vie::OnnxAnalyzer analyzer;
  const vie::ModelAnalysisReport report = analyzer.analyze_file(path.string());
  EXPECT_EQ(report.producer_name, "pytorch");
  EXPECT_EQ(report.ir_version, 9);
  ASSERT_EQ(report.opsets.size(), 1u);
  EXPECT_EQ(report.opsets[0].domain, "ai.onnx");
  EXPECT_EQ(report.opsets[0].version, 20);
  ASSERT_EQ(report.runtime_inputs.size(), 1u);
  EXPECT_EQ(report.runtime_inputs[0].name, "images");
  expect_static(report.runtime_inputs[0].type.declared_shape, {1, 3, 640, 640});
  const vie::ValueEndpoint* output = find_endpoint(report.outputs, "predictions");
  ASSERT_NE(output, nullptr);
  expect_static(output->type.declared_shape, {1, 84, 8400});
  EXPECT_GT(report.value_infos.size(), 0u);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Conv"), 64);
  EXPECT_EQ(operator_count(report, "ai.onnx", "Softmax"), 1);
  const vie::InitializerInfo* dfl = find_initializer(report.initializers, "model.22.dfl.conv.weight");
  ASSERT_NE(dfl, nullptr);
  expect_static(dfl->declared_shape, {1, 16, 1, 1});
}
