#include "vie/recognizer/classification_recognizer.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace vie {
namespace {

struct GraphIndex {
  std::map<std::string, const OnnxNode*> producers;
  std::map<std::string, const InitializerInfo*> initializers;
};

GraphIndex index_root(const ModelAnalysisReport& report) {
  GraphIndex index;
  for (const OnnxNode& node : report.nodes) {
    if (node.depth != 0 || node.domain != "ai.onnx") continue;
    for (const std::string& output : node.outputs) {
      if (!output.empty() && index.producers.find(output) == index.producers.end()) {
        index.producers.emplace(output, &node);
      }
    }
  }
  for (const InitializerInfo& initializer : report.initializers) {
    index.initializers.emplace(initializer.name, &initializer);
  }
  return index;
}

const OnnxNode* producer_of(const GraphIndex& index, const std::string& name) {
  const auto found = index.producers.find(name);
  if (found == index.producers.end()) return nullptr;
  return found->second;
}

const InitializerInfo* initializer_of(const GraphIndex& index, const std::string& name) {
  const auto found = index.initializers.find(name);
  if (found == index.initializers.end()) return nullptr;
  return found->second;
}

bool static_dims(const TensorShape& shape, std::vector<std::int64_t>& dims) {
  if (!shape.rank_known) return false;
  dims.clear();
  dims.reserve(shape.dimensions.size());
  for (const ShapeDimension& dimension : shape.dimensions) {
    if (dimension.kind != DimensionKind::Static || dimension.value < 1) return false;
    dims.push_back(dimension.value);
  }
  return true;
}

std::optional<std::int64_t> scalar_attribute(const OnnxNode& node, const std::string& name) {
  for (const auto& attribute : node.int_attributes) {
    if (attribute.first == name) return attribute.second;
  }
  return std::nullopt;
}

bool is_float_tensor(const ValueEndpoint& endpoint) {
  return endpoint.type.kind == OnnxTypeKind::Tensor &&
         endpoint.type.element_type.runtime_type == DataType::Fp32;
}

bool passthrough(const std::string& op_type) {
  return op_type == "Identity" || op_type == "Flatten" || op_type == "Reshape" || op_type == "Squeeze" ||
         op_type == "Unsqueeze" || op_type == "Dropout" || op_type == "Cast" || op_type == "Transpose";
}

bool global_pool(const std::string& op_type) {
  return op_type == "GlobalAveragePool" || op_type == "GlobalMaxPool";
}

struct ClassLayout {
  int class_axis = -1;
  int batch_axis = -1;
  int num_classes = 0;
  std::string batch_detail;
};

std::optional<ClassLayout> class_layout(const TensorShape& shape) {
  if (!shape.rank_known || shape.dimensions.size() != 2) return std::nullopt;
  ClassLayout layout;
  int class_count = 0;
  int batch_count = 0;
  for (int axis = 0; axis < 2; ++axis) {
    const ShapeDimension& dimension = shape.dimensions[static_cast<std::size_t>(axis)];
    if (dimension.kind == DimensionKind::Static && dimension.value >= 2) {
      layout.class_axis = axis;
      layout.num_classes = static_cast<int>(dimension.value);
      ++class_count;
    } else if (dimension.kind == DimensionKind::Static && dimension.value == 1) {
      layout.batch_axis = axis;
      layout.batch_detail = "axis " + std::to_string(axis) + " is static 1";
      ++batch_count;
    } else if (dimension.kind == DimensionKind::Symbolic || dimension.kind == DimensionKind::Unknown) {
      layout.batch_axis = axis;
      layout.batch_detail = dimension.kind == DimensionKind::Symbolic
                                ? "axis " + std::to_string(axis) + " is symbolic '" + dimension.symbol + "'"
                                : "axis " + std::to_string(axis) + " is dynamic";
      ++batch_count;
    }
  }
  // The weight proves Gemm/MatMul column N. A leading class axis is not accepted.
  if (class_count != 1 || batch_count != 1 || layout.class_axis != 1) return std::nullopt;
  if (layout.num_classes < 2) return std::nullopt;
  return layout;
}

bool bias_matches(const GraphIndex& index, const std::string& name, int num_classes) {
  const InitializerInfo* bias = initializer_of(index, name);
  if (bias == nullptr) return false;
  std::vector<std::int64_t> dims;
  if (!static_dims(bias->declared_shape, dims) || dims.size() > 2) return false;
  if (dims.empty()) return true;
  bool saw_classes = false;
  for (const std::int64_t dim : dims) {
    if (dim == num_classes) saw_classes = true;
    else if (dim != 1) return false;
  }
  return saw_classes || dims.size() == 1;
}

struct LinearHead {
  const OnnxNode* node = nullptr;
  std::string bias_name;
  std::vector<std::string> path;
};

std::optional<LinearHead> classification_head(const GraphIndex& index, const std::string& output_name) {
  LinearHead head;
  std::string current = output_name;
  std::set<std::string> seen;
  bool saw_add = false;
  for (int step = 0; step < 8; ++step) {
    if (!seen.insert(current).second) return std::nullopt;
    const OnnxNode* node = producer_of(index, current);
    if (node == nullptr || node->inputs.empty()) return std::nullopt;
    head.path.push_back(node->op_type);
    if (passthrough(node->op_type)) {
      current = node->inputs[0];
      continue;
    }
    if (node->op_type == "Add" && !saw_add) {
      saw_add = true;
      if (node->inputs.size() < 2) return std::nullopt;
      const bool left_bias = initializer_of(index, node->inputs[0]) != nullptr;
      const bool right_bias = initializer_of(index, node->inputs[1]) != nullptr;
      if (left_bias == right_bias) return std::nullopt;
      head.bias_name = left_bias ? node->inputs[0] : node->inputs[1];
      current = left_bias ? node->inputs[1] : node->inputs[0];
      continue;
    }
    if (node->op_type == "Gemm" || node->op_type == "MatMul") {
      head.node = node;
      return head;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

struct PoolPath {
  std::string op_type;
  std::string output_name;
  std::vector<std::string> hops;
};

std::optional<PoolPath> pool_feeding(const GraphIndex& index, const std::string& activation) {
  PoolPath path;
  std::string current = activation;
  std::set<std::string> seen;
  for (int step = 0; step < 8; ++step) {
    if (!seen.insert(current).second) return std::nullopt;
    const OnnxNode* node = producer_of(index, current);
    if (node == nullptr) return std::nullopt;
    if (global_pool(node->op_type)) {
      path.op_type = node->op_type;
      path.output_name = current;
      return path;
    }
    if (!passthrough(node->op_type) || node->inputs.empty()) return std::nullopt;
    path.hops.push_back(node->op_type);
    current = node->inputs[0];
  }
  return std::nullopt;
}

bool weight_produces_classes(const OnnxNode& node, const InitializerInfo& weight, int num_classes,
                             std::string& detail) {
  std::vector<std::int64_t> dims;
  if (!static_dims(weight.declared_shape, dims) || dims.size() != 2) return false;
  std::int64_t classes = 0;
  std::int64_t features = 0;
  if (node.op_type == "Gemm") {
    const std::int64_t trans_b = scalar_attribute(node, "transB").value_or(0);
    if (trans_b != 0 && trans_b != 1) return false;
    classes = trans_b == 1 ? dims[0] : dims[1];
    features = trans_b == 1 ? dims[1] : dims[0];
    detail = "Gemm weight '" + weight.name + "' shape [" + std::to_string(dims[0]) + ", " +
             std::to_string(dims[1]) + "] transB=" + std::to_string(trans_b);
  } else if (node.op_type == "MatMul") {
    classes = dims[1];
    features = dims[0];
    detail = "MatMul weight '" + weight.name + "' shape [" + std::to_string(dims[0]) + ", " +
             std::to_string(dims[1]) + "]";
  } else {
    return false;
  }
  return classes == num_classes && features >= 1;
}

bool has_conv(const ModelAnalysisReport& report) {
  for (const OperatorStat& stat : report.operators) {
    if (stat.domain == "ai.onnx" && stat.op_type == "Conv" && stat.count > 0) return true;
  }
  return false;
}

std::string shape_text(const TensorShape& shape) {
  if (!shape.rank_known) return "rank unknown";
  std::string text = "[";
  for (std::size_t i = 0; i < shape.dimensions.size(); ++i) {
    if (i != 0) text += ",";
    const ShapeDimension& dimension = shape.dimensions[i];
    if (dimension.kind == DimensionKind::Static) text += std::to_string(dimension.value);
    else if (dimension.kind == DimensionKind::Symbolic) text += dimension.symbol;
    else text += "?";
  }
  text += "]";
  return text;
}

}  // namespace

std::string ClassificationRecognizer::id() const { return "classification_logits"; }

std::optional<RecognitionCandidate> ClassificationRecognizer::consider(
    const ModelAnalysisReport& report) const {
  if (report.runtime_inputs.size() != 1 || report.outputs.size() != 1) return std::nullopt;
  const ValueEndpoint& input = report.runtime_inputs.front();
  const ValueEndpoint& output = report.outputs.front();
  if (!is_float_tensor(input) || !is_float_tensor(output)) return std::nullopt;
  const std::optional<ClassLayout> layout = class_layout(output.type.declared_shape);
  if (!layout.has_value()) return std::nullopt;

  const GraphIndex index = index_root(report);
  const std::optional<LinearHead> head = classification_head(index, output.name);
  if (!head.has_value() || head->node->inputs.size() < 2) return std::nullopt;
  if (head->node->op_type == "Gemm" && head->node->inputs.size() >= 3 && !head->bias_name.empty()) {
    return std::nullopt;
  }
  const std::string gemm_bias = head->node->op_type == "Gemm" && head->node->inputs.size() >= 3
                                    ? head->node->inputs[2]
                                    : "";
  const std::string bias_name = !head->bias_name.empty() ? head->bias_name : gemm_bias;
  if (!bias_name.empty() && !bias_matches(index, bias_name, layout->num_classes)) return std::nullopt;

  const InitializerInfo* weight = initializer_of(index, head->node->inputs[1]);
  if (weight == nullptr) return std::nullopt;
  std::string weight_detail;
  if (!weight_produces_classes(*head->node, *weight, layout->num_classes, weight_detail)) {
    return std::nullopt;
  }
  const std::optional<PoolPath> pool = pool_feeding(index, head->node->inputs[0]);
  if (!pool.has_value()) return std::nullopt;

  RecognitionCandidate candidate;
  candidate.recognizer_id = id();
  candidate.architecture = "single_label_classification";
  if (has_conv(report)) candidate.family = "cnn_classifier";
  candidate.evidence.push_back(
      {"single_runtime_input", "one fp32 runtime input '" + input.name + "' " +
                                   shape_text(input.type.declared_shape)});
  candidate.evidence.push_back(
      {"single_output", "one fp32 output '" + output.name + "' " + shape_text(output.type.declared_shape)});
  candidate.evidence.push_back(
      {"rank2_output", "declared output rank is 2 and shape is " + shape_text(output.type.declared_shape)});
  candidate.evidence.push_back(
      {"class_dimension", "axis " + std::to_string(layout->class_axis) + " is static " +
                              std::to_string(layout->num_classes)});
  candidate.evidence.push_back({"batch_dimension", layout->batch_detail});
  candidate.evidence.push_back({"linear_classification_head", weight_detail});
  std::string pool_detail = pool->op_type + " output '" + pool->output_name + "'";
  if (!pool->hops.empty()) {
    pool_detail += " reaches the linear input through";
    for (const std::string& hop : pool->hops) pool_detail += " " + hop;
  }
  candidate.evidence.push_back({"global_pool_path", pool_detail});
  candidate.evidence.push_back(
      {"logits_output", "the path from the linear head to '" + output.name +
                            "' has no Softmax, Sigmoid, or other output activation"});
  candidate.evidence.push_back(
      {"single_label_output", "the graph declares one output and that output is the linear class vector"});
  candidate.decoder.protocol = "classification_logits_raw";
  candidate.decoder.output_name = output.name;
  candidate.decoder.num_classes = layout->num_classes;
  candidate.decoder.class_axis = layout->class_axis;
  candidate.decoder.score_type = "logits";
  return candidate;
}

}  // namespace vie
