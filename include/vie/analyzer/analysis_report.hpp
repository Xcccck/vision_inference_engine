#pragma once

#include "vie/core/types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vie {

// A dimension exactly as stored, or as produced by a separate propagation pass.
// Static 0 is a real size. Unknown is an unnamed dynamic axis, not a zero.
enum class DimensionKind {
  Static,
  Symbolic,
  Unknown,
};

struct ShapeDimension {
  DimensionKind kind = DimensionKind::Unknown;
  std::int64_t value = 0;
  std::string symbol;
  std::string denotation;
};

// rank_known == false means the TypeProto omitted the shape (unknown rank).
// rank_known with an empty dimension list is a scalar.
struct TensorShape {
  bool rank_known = false;
  std::vector<ShapeDimension> dimensions;
};

enum class OnnxTypeKind {
  Absent,
  Tensor,
  Sequence,
  Map,
  Optional,
  SparseTensor,
  Opaque,
  Unknown,
};

// ONNX TensorProto.DataType. runtime_type is set only when VIE already has
// that dtype. Unmapped codes stay unmapped.
struct OnnxElementType {
  int code = 0;
  std::string name = "undefined";
  std::optional<DataType> runtime_type;
};

// One level of Sequence / Optional / Map is expanded. Deeper nesting sets
// nested_truncated and does not invent an inner tensor shape.
struct OnnxType {
  OnnxTypeKind kind = OnnxTypeKind::Absent;
  OnnxElementType element_type;
  TensorShape declared_shape;
  std::optional<OnnxElementType> map_key_type;
  OnnxTypeKind nested_kind = OnnxTypeKind::Absent;
  OnnxElementType nested_element_type;
  TensorShape nested_declared_shape;
  bool nested_truncated = false;
};

enum class TensorPayloadKind {
  Empty,
  Raw,
  TypedElements,
  External,
};

struct ExternalDataEntry {
  std::string key;
  std::string value;
};

struct InitializerInfo {
  std::string name;
  OnnxElementType element_type;
  TensorShape declared_shape;
  bool sparse = false;
  TensorPayloadKind payload = TensorPayloadKind::Empty;
  std::uint64_t payload_bytes = 0;
  std::uint64_t typed_element_count = 0;
  std::vector<ExternalDataEntry> external_data;
  std::string external_location;
  bool external_file_found = false;
  std::uint64_t external_file_bytes = 0;
  bool listed_as_graph_input = false;
};

struct ValueEndpoint {
  std::string name;
  std::string doc_string;
  OnnxType type;
};

// Shape recorded in graph.value_info. This is not copied onto input or output
// declared shapes, even when the names match.
struct RecordedValueInfo {
  std::string name;
  std::string doc_string;
  OnnxType type;
};

struct OnnxOpset {
  std::string raw_domain;
  std::string domain;
  std::int64_t version = 0;
};

struct OnnxNode {
  std::string name;
  std::string op_type;
  std::string raw_domain;
  std::string domain;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::string graph_path;
  int depth = 0;
  // Scalar integer attributes, such as Concat axis and TopK axis. Packed
  // integer lists and tensor attributes are not copied here.
  std::vector<std::pair<std::string, std::int64_t>> int_attributes;
};

// A small int64 tensor produced by a Constant node. Weights are not copied.
// The name is the Constant node's output, which is what later nodes consume.
struct IntegerConstant {
  std::string name;
  std::vector<std::int64_t> values;
};

// One finite FP32 or FP64 element. Weights are not copied. The name is the
// initializer or Constant output that later nodes consume.
struct FloatConstant {
  std::string name;
  double value = 0;
};

struct OperatorStat {
  std::string domain;
  std::string op_type;
  std::int64_t count = 0;
};

// A shape computed from declared inputs, initializer dims, and a small set of
// operators. It is never written into declared_shape.
struct PropagatedShape {
  std::string name;
  TensorShape shape;
  std::string domain;
  std::string op_type;
  std::vector<std::string> from_inputs;
  std::string graph_path;
};

struct FunctionSummary {
  std::string name;
  std::string raw_domain;
  std::string domain;
  std::string overload;
  std::int64_t node_count = 0;
  std::vector<OnnxOpset> opsets;
};

struct AnalysisWarning {
  std::string code;
  std::string message;
};

// Structural facts only. This type has no task, decoder, color, normalization,
// or NMS members, so those cannot be filled in by accident.
struct ModelAnalysisReport {
  std::string scope = "onnx_structure";
  std::string source;
  std::int64_t ir_version = 0;
  bool ir_version_present = false;
  std::string producer_name;
  std::string producer_version;
  std::string model_domain;
  std::int64_t model_version = 0;
  std::string doc_string;
  std::string graph_name;
  std::vector<std::pair<std::string, std::string>> metadata;
  std::vector<OnnxOpset> opsets;
  std::vector<ValueEndpoint> graph_inputs;
  std::vector<ValueEndpoint> runtime_inputs;
  std::vector<ValueEndpoint> outputs;
  std::vector<RecordedValueInfo> value_infos;
  std::vector<InitializerInfo> initializers;
  std::vector<InitializerInfo> sparse_initializers;
  std::vector<OnnxNode> nodes;
  std::vector<IntegerConstant> integer_constants;
  std::vector<FloatConstant> float_constants;
  std::vector<OperatorStat> operators;
  std::vector<PropagatedShape> propagated_shapes;
  std::vector<FunctionSummary> functions;
  std::int64_t training_info_count = 0;
  std::int64_t quantization_annotation_count = 0;
  std::vector<AnalysisWarning> warnings;
  std::vector<std::string> not_interpreted;
  std::vector<std::string> omitted{"model_family", "task", "decoder", "color",
                                   "normalization", "nms"};
};

OnnxElementType onnx_element_type(int code);
std::string normalize_onnx_domain(std::string_view raw_domain);
std::string format_tensor_shape(const TensorShape& shape);

const PropagatedShape* find_propagated_shape(const ModelAnalysisReport& report,
                                              std::string_view name);

std::string analysis_to_json(const ModelAnalysisReport& report);
std::string analysis_to_text(const ModelAnalysisReport& report);

}  // namespace vie
