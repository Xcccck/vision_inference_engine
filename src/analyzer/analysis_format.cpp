#include "vie/analyzer/analysis_report.hpp"

#include <nlohmann/json.hpp>
#include <sstream>

namespace vie {
namespace {

using json = nlohmann::json;

const char* dimension_kind_name(DimensionKind kind) {
  switch (kind) {
    case DimensionKind::Static:
      return "static";
    case DimensionKind::Symbolic:
      return "symbolic";
    case DimensionKind::Unknown:
      return "unknown";
  }
  return "unknown";
}

const char* type_kind_name(OnnxTypeKind kind) {
  switch (kind) {
    case OnnxTypeKind::Absent:
      return "absent";
    case OnnxTypeKind::Tensor:
      return "tensor";
    case OnnxTypeKind::Sequence:
      return "sequence";
    case OnnxTypeKind::Map:
      return "map";
    case OnnxTypeKind::Optional:
      return "optional";
    case OnnxTypeKind::SparseTensor:
      return "sparse_tensor";
    case OnnxTypeKind::Opaque:
      return "opaque";
    case OnnxTypeKind::Unknown:
      return "unknown";
  }
  return "unknown";
}

const char* payload_kind_name(TensorPayloadKind kind) {
  switch (kind) {
    case TensorPayloadKind::Empty:
      return "empty";
    case TensorPayloadKind::Raw:
      return "raw";
    case TensorPayloadKind::TypedElements:
      return "typed_elements";
    case TensorPayloadKind::External:
      return "external";
  }
  return "empty";
}

json shape_json(const TensorShape& shape) {
  json dims = json::array();
  for (const ShapeDimension& dimension : shape.dimensions) {
    json item = {{"kind", dimension_kind_name(dimension.kind)}};
    if (dimension.kind == DimensionKind::Static) item["value"] = dimension.value;
    if (dimension.kind == DimensionKind::Symbolic) item["symbol"] = dimension.symbol;
    if (!dimension.denotation.empty()) item["denotation"] = dimension.denotation;
    dims.push_back(std::move(item));
  }
  return json{{"rank_known", shape.rank_known}, {"dimensions", std::move(dims)}};
}

json element_json(const OnnxElementType& element) {
  json item = {{"code", element.code}, {"name", element.name}};
  if (element.runtime_type.has_value()) item["runtime_type"] = to_string(*element.runtime_type);
  else item["runtime_type"] = nullptr;
  return item;
}

json type_json(const OnnxType& type) {
  json item = {{"kind", type_kind_name(type.kind)},
               {"element_type", element_json(type.element_type)},
               {"declared_shape", shape_json(type.declared_shape)},
               {"nested_kind", type_kind_name(type.nested_kind)},
               {"nested_truncated", type.nested_truncated}};
  if (type.nested_kind != OnnxTypeKind::Absent) {
    item["nested_element_type"] = element_json(type.nested_element_type);
    item["nested_declared_shape"] = shape_json(type.nested_declared_shape);
  }
  if (type.map_key_type.has_value()) item["map_key_type"] = element_json(*type.map_key_type);
  return item;
}

json endpoint_json(const ValueEndpoint& endpoint) {
  return json{{"name", endpoint.name},
              {"doc_string", endpoint.doc_string},
              {"type", type_json(endpoint.type)}};
}

json initializer_json(const InitializerInfo& initializer) {
  json external = json::array();
  for (const ExternalDataEntry& entry : initializer.external_data) {
    external.push_back(json{{"key", entry.key}, {"value", entry.value}});
  }
  return json{{"name", initializer.name},
              {"element_type", element_json(initializer.element_type)},
              {"declared_shape", shape_json(initializer.declared_shape)},
              {"sparse", initializer.sparse},
              {"payload", payload_kind_name(initializer.payload)},
              {"payload_bytes", initializer.payload_bytes},
              {"typed_element_count", initializer.typed_element_count},
              {"external_data", std::move(external)},
              {"external_location", initializer.external_location},
              {"external_file_found", initializer.external_file_found},
              {"external_file_bytes", initializer.external_file_bytes},
              {"listed_as_graph_input", initializer.listed_as_graph_input}};
}

void write_endpoint_text(std::ostringstream& out, const ValueEndpoint& endpoint) {
  out << "  " << endpoint.name << "  " << type_kind_name(endpoint.type.kind) << "  "
      << endpoint.type.element_type.name << "  declared "
      << format_tensor_shape(endpoint.type.declared_shape);
  if (endpoint.type.kind == OnnxTypeKind::Sequence || endpoint.type.kind == OnnxTypeKind::Optional ||
      endpoint.type.kind == OnnxTypeKind::Map) {
    out << "  nested " << type_kind_name(endpoint.type.nested_kind) << " "
        << endpoint.type.nested_element_type.name;
    if (endpoint.type.nested_kind == OnnxTypeKind::Tensor) {
      out << " " << format_tensor_shape(endpoint.type.nested_declared_shape);
    }
  }
  out << '\n';
}

}  // namespace

std::string normalize_onnx_domain(std::string_view raw_domain) {
  if (raw_domain.empty() || raw_domain == "ai.onnx") return "ai.onnx";
  return std::string(raw_domain);
}

OnnxElementType onnx_element_type(int code) {
  OnnxElementType element;
  element.code = code;
  switch (code) {
    case 1:
      element.name = "float";
      element.runtime_type = DataType::Fp32;
      break;
    case 2:
      element.name = "uint8";
      element.runtime_type = DataType::Uint8;
      break;
    case 3:
      element.name = "int8";
      element.runtime_type = DataType::Int8;
      break;
    case 4:
      element.name = "uint16";
      break;
    case 5:
      element.name = "int16";
      break;
    case 6:
      element.name = "int32";
      element.runtime_type = DataType::Int32;
      break;
    case 7:
      element.name = "int64";
      break;
    case 8:
      element.name = "string";
      break;
    case 9:
      element.name = "bool";
      element.runtime_type = DataType::Bool;
      break;
    case 10:
      element.name = "float16";
      element.runtime_type = DataType::Fp16;
      break;
    case 11:
      element.name = "double";
      break;
    case 12:
      element.name = "uint32";
      break;
    case 13:
      element.name = "uint64";
      break;
    case 14:
      element.name = "complex64";
      break;
    case 15:
      element.name = "complex128";
      break;
    case 16:
      element.name = "bfloat16";
      break;
    case 17:
      element.name = "float8e4m3fn";
      break;
    case 18:
      element.name = "float8e4m3fnuz";
      break;
    case 19:
      element.name = "float8e5m2";
      break;
    case 20:
      element.name = "float8e5m2fnuz";
      break;
    case 21:
      element.name = "uint4";
      break;
    case 22:
      element.name = "int4";
      break;
    case 23:
      element.name = "float4e2m1";
      break;
    case 24:
      element.name = "float8e8m0";
      break;
    case 25:
      element.name = "uint2";
      break;
    case 26:
      element.name = "int2";
      break;
    case 27:
      element.name = "float6e2m3";
      break;
    case 28:
      element.name = "float6e3m2";
      break;
    default:
      element.name = "onnx_dtype_" + std::to_string(code);
      break;
  }
  return element;
}

std::string format_tensor_shape(const TensorShape& shape) {
  if (!shape.rank_known) return "rank unknown";
  if (shape.dimensions.empty()) return "scalar";
  std::string text = "[";
  for (std::size_t i = 0; i < shape.dimensions.size(); ++i) {
    if (i != 0) text += ", ";
    const ShapeDimension& dimension = shape.dimensions[i];
    if (dimension.kind == DimensionKind::Static) text += std::to_string(dimension.value);
    else if (dimension.kind == DimensionKind::Symbolic) text += dimension.symbol;
    else text += "?";
    if (!dimension.denotation.empty()) text += "{" + dimension.denotation + "}";
  }
  text += "]";
  return text;
}

const PropagatedShape* find_propagated_shape(const ModelAnalysisReport& report,
                                             std::string_view name) {
  for (const PropagatedShape& shape : report.propagated_shapes) {
    if (shape.name == name) return &shape;
  }
  return nullptr;
}

std::string analysis_to_json(const ModelAnalysisReport& report) {
  json opsets = json::array();
  for (const OnnxOpset& opset : report.opsets) {
    opsets.push_back(json{{"raw_domain", opset.raw_domain},
                          {"domain", opset.domain},
                          {"version", opset.version}});
  }
  json metadata = json::array();
  for (const auto& entry : report.metadata) {
    metadata.push_back(json{{"key", entry.first}, {"value", entry.second}});
  }
  json graph_inputs = json::array();
  for (const ValueEndpoint& endpoint : report.graph_inputs) graph_inputs.push_back(endpoint_json(endpoint));
  json runtime_inputs = json::array();
  for (const ValueEndpoint& endpoint : report.runtime_inputs) {
    runtime_inputs.push_back(endpoint_json(endpoint));
  }
  json outputs = json::array();
  for (const ValueEndpoint& endpoint : report.outputs) outputs.push_back(endpoint_json(endpoint));
  json value_infos = json::array();
  for (const RecordedValueInfo& info : report.value_infos) {
    value_infos.push_back(json{{"name", info.name},
                               {"doc_string", info.doc_string},
                               {"type", type_json(info.type)}});
  }
  json initializers = json::array();
  for (const InitializerInfo& initializer : report.initializers) {
    initializers.push_back(initializer_json(initializer));
  }
  json sparse = json::array();
  for (const InitializerInfo& initializer : report.sparse_initializers) {
    sparse.push_back(initializer_json(initializer));
  }
  json nodes = json::array();
  for (const OnnxNode& node : report.nodes) {
    json described = json{{"name", node.name},
                          {"op_type", node.op_type},
                          {"raw_domain", node.raw_domain},
                          {"domain", node.domain},
                          {"inputs", node.inputs},
                          {"outputs", node.outputs},
                          {"graph_path", node.graph_path},
                          {"depth", node.depth}};
    if (!node.int_attributes.empty()) {
      json attributes = json::array();
      for (const auto& attribute : node.int_attributes) {
        attributes.push_back(json{{"name", attribute.first}, {"value", attribute.second}});
      }
      described["int_attributes"] = std::move(attributes);
    }
    nodes.push_back(std::move(described));
  }
  json integer_constants = json::array();
  for (const IntegerConstant& constant : report.integer_constants) {
    integer_constants.push_back(json{{"name", constant.name}, {"values", constant.values}});
  }
  json operators = json::array();
  for (const OperatorStat& stat : report.operators) {
    operators.push_back(
        json{{"domain", stat.domain}, {"op_type", stat.op_type}, {"count", stat.count}});
  }
  json propagated = json::array();
  for (const PropagatedShape& shape : report.propagated_shapes) {
    propagated.push_back(json{{"name", shape.name},
                              {"shape", shape_json(shape.shape)},
                              {"domain", shape.domain},
                              {"op_type", shape.op_type},
                              {"from_inputs", shape.from_inputs},
                              {"graph_path", shape.graph_path}});
  }
  json functions = json::array();
  for (const FunctionSummary& function : report.functions) {
    json function_opsets = json::array();
    for (const OnnxOpset& opset : function.opsets) {
      function_opsets.push_back(json{{"raw_domain", opset.raw_domain},
                                    {"domain", opset.domain},
                                    {"version", opset.version}});
    }
    functions.push_back(json{{"name", function.name},
                             {"raw_domain", function.raw_domain},
                             {"domain", function.domain},
                             {"overload", function.overload},
                             {"node_count", function.node_count},
                             {"opsets", std::move(function_opsets)}});
  }
  json warnings = json::array();
  for (const AnalysisWarning& warning : report.warnings) {
    warnings.push_back(json{{"code", warning.code}, {"message", warning.message}});
  }
  const json document = {{"scope", report.scope},
                         {"source", report.source},
                         {"ir_version", report.ir_version_present ? json(report.ir_version) : json(nullptr)},
                         {"producer_name", report.producer_name},
                         {"producer_version", report.producer_version},
                         {"model_domain", report.model_domain},
                         {"model_version", report.model_version},
                         {"doc_string", report.doc_string},
                         {"graph_name", report.graph_name},
                         {"metadata", std::move(metadata)},
                         {"opsets", std::move(opsets)},
                         {"graph_inputs", std::move(graph_inputs)},
                         {"runtime_inputs", std::move(runtime_inputs)},
                         {"outputs", std::move(outputs)},
                         {"value_infos", std::move(value_infos)},
                         {"initializers", std::move(initializers)},
                         {"sparse_initializers", std::move(sparse)},
                         {"nodes", std::move(nodes)},
                         {"integer_constants", std::move(integer_constants)},
                         {"operators", std::move(operators)},
                         {"propagated_shapes", std::move(propagated)},
                         {"functions", std::move(functions)},
                         {"training_info_count", report.training_info_count},
                         {"quantization_annotation_count", report.quantization_annotation_count},
                         {"warnings", std::move(warnings)},
                         {"not_interpreted", report.not_interpreted},
                         {"omitted", report.omitted}};
  return document.dump(2);
}

std::string analysis_to_text(const ModelAnalysisReport& report) {
  std::ostringstream out;
  out << "ONNX structural analysis\n";
  out << "scope: " << report.scope << '\n';
  out << "source: " << report.source << '\n';
  if (report.ir_version_present) out << "ir_version: " << report.ir_version << '\n';
  else out << "ir_version: missing\n";
  if (!report.producer_name.empty()) out << "producer: " << report.producer_name << '\n';
  if (!report.graph_name.empty()) out << "graph: " << report.graph_name << '\n';
  out << "opsets:\n";
  if (report.opsets.empty()) out << "  (none)\n";
  for (const OnnxOpset& opset : report.opsets) {
    out << "  domain " << opset.domain << "  raw \"" << opset.raw_domain << "\"  version "
        << opset.version << '\n';
  }
  out << "runtime inputs:\n";
  if (report.runtime_inputs.empty()) out << "  (none)\n";
  for (const ValueEndpoint& endpoint : report.runtime_inputs) write_endpoint_text(out, endpoint);
  out << "graph inputs:\n";
  if (report.graph_inputs.empty()) out << "  (none)\n";
  for (const ValueEndpoint& endpoint : report.graph_inputs) write_endpoint_text(out, endpoint);
  out << "outputs:\n";
  if (report.outputs.empty()) out << "  (none)\n";
  for (const ValueEndpoint& endpoint : report.outputs) write_endpoint_text(out, endpoint);
  out << "file value_info:\n";
  if (report.value_infos.empty()) out << "  (none)\n";
  for (const RecordedValueInfo& info : report.value_infos) {
    out << "  " << info.name << "  " << type_kind_name(info.type.kind) << "  "
        << info.type.element_type.name << "  recorded "
        << format_tensor_shape(info.type.declared_shape) << '\n';
  }
  out << "initializers:\n";
  if (report.initializers.empty()) out << "  (none)\n";
  for (const InitializerInfo& initializer : report.initializers) {
    out << "  " << initializer.name << "  " << initializer.element_type.name << "  declared "
        << format_tensor_shape(initializer.declared_shape) << "  " << payload_kind_name(initializer.payload);
    if (initializer.payload == TensorPayloadKind::External) {
      out << "  location " << initializer.external_location
          << (initializer.external_file_found ? "  found" : "  missing");
    }
    if (initializer.listed_as_graph_input) out << "  also_graph_input";
    out << '\n';
  }
  if (!report.sparse_initializers.empty()) {
    out << "sparse initializers:\n";
    for (const InitializerInfo& initializer : report.sparse_initializers) {
      out << "  " << initializer.name << "  " << initializer.element_type.name << "  declared "
          << format_tensor_shape(initializer.declared_shape) << '\n';
    }
  }
  out << "operators:\n";
  if (report.operators.empty()) out << "  (none)\n";
  for (const OperatorStat& stat : report.operators) {
    out << "  " << stat.domain << "  " << stat.op_type << "  count " << stat.count << '\n';
  }
  out << "propagated shapes:\n";
  if (report.propagated_shapes.empty()) out << "  (none)\n";
  for (const PropagatedShape& shape : report.propagated_shapes) {
    out << "  " << shape.name << "  " << format_tensor_shape(shape.shape) << "  via " << shape.domain
        << "::" << shape.op_type << '\n';
  }
  out << "warnings:\n";
  if (report.warnings.empty()) out << "  (none)\n";
  for (const AnalysisWarning& warning : report.warnings) {
    out << "  " << warning.code << ": " << warning.message << '\n';
  }
  if (!report.not_interpreted.empty()) {
    out << "not interpreted:\n";
    for (const std::string& item : report.not_interpreted) out << "  " << item << '\n';
  }
  out << "omitted: model_family, task, decoder, color, normalization, nms\n";
  return out.str();
}

}  // namespace vie
