#include "vie/recognizer/rtdetr_recognizer.hpp"

#include <map>
#include <set>
#include <utility>

namespace vie {
namespace {

struct GraphIndex {
  std::map<std::string, const OnnxNode*> producers;
  std::map<std::string, std::vector<const OnnxNode*>> consumers;
  std::map<std::string, const InitializerInfo*> initializers;
  std::map<std::string, const IntegerConstant*> constants;
};

GraphIndex index_graph(const ModelAnalysisReport& report) {
  GraphIndex index;
  for (const OnnxNode& node : report.nodes) {
    for (const std::string& output : node.outputs) {
      if (!output.empty() && index.producers.find(output) == index.producers.end()) {
        index.producers.emplace(output, &node);
      }
    }
    for (const std::string& input : node.inputs) {
      if (!input.empty()) index.consumers[input].push_back(&node);
    }
  }
  for (const InitializerInfo& initializer : report.initializers) {
    index.initializers.emplace(initializer.name, &initializer);
  }
  for (const IntegerConstant& constant : report.integer_constants) {
    index.constants.emplace(constant.name, &constant);
  }
  return index;
}

const OnnxNode* producer_of(const GraphIndex& index, const std::string& name) {
  const auto found = index.producers.find(name);
  if (found == index.producers.end()) return nullptr;
  return found->second;
}

bool reaches(const GraphIndex& index, const std::string& start, const std::string& target) {
  std::vector<std::string> pending{start};
  std::set<std::string> seen;
  while (!pending.empty()) {
    const std::string current = pending.back();
    pending.pop_back();
    if (!seen.insert(current).second) continue;
    if (current == target) return true;
    const auto found = index.consumers.find(current);
    if (found == index.consumers.end()) continue;
    for (const OnnxNode* node : found->second) {
      for (const std::string& output : node->outputs) {
        if (!output.empty()) pending.push_back(output);
      }
    }
  }
  return false;
}

bool static_dims(const TensorShape& shape, std::vector<std::int64_t>& dims) {
  if (!shape.rank_known) return false;
  dims.clear();
  dims.reserve(shape.dimensions.size());
  for (const ShapeDimension& dimension : shape.dimensions) {
    if (dimension.kind != DimensionKind::Static) return false;
    dims.push_back(dimension.value);
  }
  return true;
}

bool has_operator(const ModelAnalysisReport& report, const std::string& op_type) {
  for (const OperatorStat& stat : report.operators) {
    if (stat.domain == "ai.onnx" && stat.op_type == op_type && stat.count > 0) return true;
  }
  return false;
}

std::optional<std::int64_t> int_attribute(const OnnxNode& node, const std::string& name) {
  for (const auto& attribute : node.int_attributes) {
    if (attribute.first == name) return attribute.second;
  }
  return std::nullopt;
}

bool is_constant_tensor(const GraphIndex& index, const std::string& name) {
  if (index.initializers.find(name) != index.initializers.end()) return true;
  if (index.constants.find(name) != index.constants.end()) return true;
  const OnnxNode* node = producer_of(index, name);
  return node != nullptr && node->op_type == "Constant";
}

std::vector<std::string> data_inputs(const GraphIndex& index, const OnnxNode& node) {
  std::vector<std::string> inputs;
  for (const std::string& input : node.inputs) {
    if (!input.empty() && !is_constant_tensor(index, input)) inputs.push_back(input);
  }
  return inputs;
}

const OnnxNode* unwrap(const GraphIndex& index, const std::string& start) {
  std::string name = start;
  std::set<std::string> seen;
  while (seen.insert(name).second) {
    const OnnxNode* node = producer_of(index, name);
    if (node == nullptr) return nullptr;
    const bool skip = node->op_type == "Identity" || node->op_type == "Squeeze" || node->op_type == "Unsqueeze" ||
                      node->op_type == "Concat";
    if (!skip) return node;
    const std::vector<std::string> inputs = data_inputs(index, *node);
    if (inputs.size() != 1) return node;
    name = inputs.front();
  }
  return nullptr;
}

struct LinearHead {
  int width = 0;
  std::string data_input;
  std::string node_name;
};

std::optional<int> matmul_width(const GraphIndex& index, const OnnxNode& node, std::string& data_input) {
  if (node.op_type != "MatMul" || node.inputs.size() != 2) return std::nullopt;
  const InitializerInfo* weight = nullptr;
  for (const std::string& input : node.inputs) {
    const auto found = index.initializers.find(input);
    if (found == index.initializers.end()) {
      data_input = input;
      continue;
    }
    weight = found->second;
  }
  if (weight == nullptr || data_input.empty()) return std::nullopt;
  std::vector<std::int64_t> dims;
  if (!static_dims(weight->declared_shape, dims) || dims.size() < 2 || dims.back() < 1) return std::nullopt;
  return static_cast<int>(dims.back());
}

std::optional<LinearHead> as_linear(const GraphIndex& index, const OnnxNode* node) {
  if (node == nullptr) return std::nullopt;
  LinearHead head;
  head.node_name = node->name.empty() ? node->op_type : node->name;
  if (node->op_type == "MatMul") {
    const std::optional<int> width = matmul_width(index, *node, head.data_input);
    if (!width.has_value()) return std::nullopt;
    head.width = *width;
    return head;
  }
  if (node->op_type != "Add" || node->inputs.size() != 2) return std::nullopt;
  const InitializerInfo* bias = nullptr;
  std::string other;
  for (const std::string& input : node->inputs) {
    const auto found = index.initializers.find(input);
    if (found == index.initializers.end()) {
      other = input;
      continue;
    }
    std::vector<std::int64_t> dims;
    if (!static_dims(found->second->declared_shape, dims) || dims.size() != 1) return std::nullopt;
    bias = found->second;
  }
  if (bias == nullptr || other.empty()) return std::nullopt;
  const OnnxNode* matmul = producer_of(index, other);
  if (matmul != nullptr && matmul->op_type == "Identity") matmul = unwrap(index, other);
  if (matmul == nullptr || matmul->op_type != "MatMul") return std::nullopt;
  std::string data_input;
  const std::optional<int> width = matmul_width(index, *matmul, data_input);
  std::vector<std::int64_t> bias_dims;
  if (!width.has_value() || !static_dims(bias->declared_shape, bias_dims) || bias_dims.size() != 1 ||
      bias_dims[0] != *width) {
    return std::nullopt;
  }
  head.width = *width;
  head.data_input = data_input;
  return head;
}

bool is_inverse_sigmoid(const GraphIndex& index, const std::string& tensor) {
  const OnnxNode* log = unwrap(index, tensor);
  if (log == nullptr || log->op_type != "Log") return false;
  const std::vector<std::string> log_inputs = data_inputs(index, *log);
  if (log_inputs.size() != 1) return false;
  const OnnxNode* div = unwrap(index, log_inputs.front());
  return div != nullptr && div->op_type == "Div" && data_inputs(index, *div).size() == 2;
}

struct Side {
  bool box = false;
  bool classes = false;
  LinearHead linear;
  std::string sigmoid_name;
};

std::optional<Side> match_side(const GraphIndex& index, const std::string& tensor) {
  const OnnxNode* sigmoid = unwrap(index, tensor);
  if (sigmoid == nullptr || sigmoid->op_type != "Sigmoid") return std::nullopt;
  const std::vector<std::string> inputs = data_inputs(index, *sigmoid);
  if (inputs.size() != 1) return std::nullopt;
  const OnnxNode* produced = unwrap(index, inputs.front());
  if (produced == nullptr) return std::nullopt;
  Side side;
  side.sigmoid_name = sigmoid->name.empty() ? sigmoid->op_type : sigmoid->name;
  if (produced->op_type == "Add" && produced->inputs.size() == 2) {
    const OnnxNode* left = producer_of(index, produced->inputs[0]);
    const OnnxNode* right = producer_of(index, produced->inputs[1]);
    const OnnxNode* linear_node = nullptr;
    std::string inverse;
    if (left != nullptr && is_inverse_sigmoid(index, produced->inputs[0])) {
      inverse = produced->inputs[0];
      linear_node = right;
    } else if (right != nullptr && is_inverse_sigmoid(index, produced->inputs[1])) {
      inverse = produced->inputs[1];
      linear_node = left;
    }
    if (!inverse.empty()) {
      const std::optional<LinearHead> linear = as_linear(index, linear_node);
      if (!linear.has_value() || linear->width != 4) return std::nullopt;
      side.box = true;
      side.linear = *linear;
      return side;
    }
  }
  const std::optional<LinearHead> linear = as_linear(index, produced);
  if (!linear.has_value() || linear->width < 1) return std::nullopt;
  side.classes = true;
  side.linear = *linear;
  return side;
}

struct HeadFacts {
  int num_classes = 0;
  int num_queries = 0;
  std::string output_name;
  std::string input_shape;
  std::string output_shape;
  std::string box_sigmoid;
  std::string class_sigmoid;
  std::string topk_name;
};

std::optional<HeadFacts> match_rtdetr_head(const ModelAnalysisReport& report) {
  if (report.runtime_inputs.size() != 1 || report.outputs.size() != 1) return std::nullopt;
  const ValueEndpoint& input = report.runtime_inputs.front();
  const ValueEndpoint& output = report.outputs.front();
  if (input.type.kind != OnnxTypeKind::Tensor || output.type.kind != OnnxTypeKind::Tensor) return std::nullopt;
  if (input.type.element_type.runtime_type != DataType::Fp32 ||
      output.type.element_type.runtime_type != DataType::Fp32) {
    return std::nullopt;
  }
  const TensorShape& input_shape = input.type.declared_shape;
  const TensorShape& output_shape = output.type.declared_shape;
  if (!input_shape.rank_known || input_shape.dimensions.size() != 4) return std::nullopt;
  if (input_shape.dimensions[1].kind != DimensionKind::Static || input_shape.dimensions[1].value != 3) {
    return std::nullopt;
  }
  if (!output_shape.rank_known || output_shape.dimensions.size() != 3) return std::nullopt;
  if (has_operator(report, "NonMaxSuppression")) return std::nullopt;

  const GraphIndex index = index_graph(report);
  const OnnxNode* final_node = producer_of(index, output.name);
  const std::vector<std::string> sides =
      final_node == nullptr ? std::vector<std::string>{} : data_inputs(index, *final_node);
  if (final_node == nullptr || final_node->op_type != "Concat" || sides.size() != 2) return std::nullopt;
  const std::optional<std::int64_t> axis = int_attribute(*final_node, "axis");
  if (!axis.has_value() || *axis != -1) return std::nullopt;

  const std::optional<Side> first = match_side(index, sides[0]);
  const std::optional<Side> second = match_side(index, sides[1]);
  if (!first.has_value() || !second.has_value()) return std::nullopt;
  const Side* box = nullptr;
  const Side* classes = nullptr;
  if (first->box && second->classes) {
    box = &*first;
    classes = &*second;
  } else if (second->box && first->classes) {
    box = &*second;
    classes = &*first;
  } else {
    return std::nullopt;
  }

  const OnnxNode* topk = nullptr;
  for (const OnnxNode& node : report.nodes) {
    if (node.op_type != "TopK" || node.domain != "ai.onnx") continue;
    if (topk != nullptr) return std::nullopt;
    topk = &node;
  }
  if (topk == nullptr || topk->inputs.size() < 2 || topk->outputs.empty()) return std::nullopt;
  const std::optional<std::int64_t> topk_axis = int_attribute(*topk, "axis");
  if (!topk_axis.has_value() || *topk_axis != 1) return std::nullopt;
  const auto constant = index.constants.find(topk->inputs[1]);
  if (constant == index.constants.end() || constant->second->values.size() != 1 || constant->second->values.front() < 1) {
    return std::nullopt;
  }
  const int queries = static_cast<int>(constant->second->values.front());
  if (!reaches(index, input.name, topk->inputs[0])) return std::nullopt;
  bool reaches_box = false;
  bool reaches_class = false;
  for (const std::string& produced : topk->outputs) {
    if (produced.empty()) continue;
    if (reaches(index, produced, box->linear.data_input)) reaches_box = true;
    if (reaches(index, produced, classes->linear.data_input)) reaches_class = true;
  }
  if (!reaches_box || !reaches_class) return std::nullopt;

  HeadFacts facts;
  facts.num_classes = classes->linear.width;
  facts.num_queries = queries;
  facts.output_name = output.name;
  facts.input_shape = format_tensor_shape(input_shape);
  facts.output_shape = format_tensor_shape(output_shape);
  facts.box_sigmoid = box->sigmoid_name;
  facts.class_sigmoid = classes->sigmoid_name;
  facts.topk_name = topk->name.empty() ? "TopK" : topk->name;
  return facts;
}

}  // namespace

std::string RtdetrRecognizer::id() const { return "rtdetr_detection"; }

std::optional<RecognitionCandidate> RtdetrRecognizer::consider(const ModelAnalysisReport& report) const {
  const std::optional<HeadFacts> facts = match_rtdetr_head(report);
  if (!facts.has_value()) return std::nullopt;

  RecognitionCandidate candidate;
  candidate.recognizer_id = id();
  candidate.architecture = "rtdetr_detection";
  candidate.family = "rtdetr";
  candidate.evidence.push_back(
      {"single_input_axis1_is_3",
       "the only runtime input is float " + facts->input_shape + " with declared axis 1 equal to 3"});
  candidate.evidence.push_back(
      {"final_concat_without_nms",
       "output '" + facts->output_name + "' is declared " + facts->output_shape +
           " and is produced by a Concat on axis -1; the graph has no NonMaxSuppression"});
  candidate.evidence.push_back(
      {"normalized_box_refinement",
       "Sigmoid '" + facts->box_sigmoid +
           "' refines a width-4 linear with Log(Div(...)), and that Sigmoid reaches one Concat input"});
  candidate.evidence.push_back(
      {"sigmoid_class_linear",
       "Sigmoid '" + facts->class_sigmoid + "' is applied to a linear of width " +
           std::to_string(facts->num_classes) + ", and that Sigmoid reaches the other Concat input"});
  candidate.evidence.push_back(
      {"query_selection_topk",
       "TopK '" + facts->topk_name + "' has axis 1 and scalar k " + std::to_string(facts->num_queries) +
           "; the runtime input reaches its data input, and one of its outputs reaches both linears"});
  candidate.decoder.protocol = "rtdetr_detection_raw";
  candidate.decoder.output_name = facts->output_name;
  candidate.decoder.box_format = "cxcywh_normalized";
  candidate.decoder.channel_order = "predictions";
  candidate.decoder.channel_axis = 2;
  candidate.decoder.box_channels = 4;
  candidate.decoder.num_classes = facts->num_classes;
  candidate.decoder.class_scores_are_probabilities = true;
  candidate.decoder.num_queries = facts->num_queries;
  return candidate;
}

}  // namespace vie
