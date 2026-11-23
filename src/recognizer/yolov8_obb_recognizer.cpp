#include "vie/recognizer/yolov8_obb_recognizer.hpp"

#include <cmath>
#include <map>
#include <set>
#include <string>

namespace vie {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

struct GraphIndex {
  std::map<std::string, const OnnxNode*> producers;
  std::map<std::string, std::vector<const OnnxNode*>> consumers;
  std::map<std::string, double> scalars;
};

GraphIndex index_graph(const ModelAnalysisReport& report) {
  GraphIndex index;
  for (const OnnxNode& node : report.nodes) {
    if (node.depth != 0 || node.domain != "ai.onnx") continue;
    for (const std::string& output : node.outputs) {
      if (!output.empty() && index.producers.find(output) == index.producers.end()) {
        index.producers.emplace(output, &node);
      }
    }
    for (const std::string& input : node.inputs) {
      if (!input.empty()) index.consumers[input].push_back(&node);
    }
  }
  for (const FloatConstant& constant : report.float_constants) {
    index.scalars.emplace(constant.name, constant.value);
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
  for (const ShapeDimension& dimension : shape.dimensions) {
    if (dimension.kind != DimensionKind::Static) return false;
    dims.push_back(dimension.value);
  }
  return true;
}

const InitializerInfo* weight_of(const ModelAnalysisReport& report, const OnnxNode& node) {
  if (node.inputs.size() < 2) return nullptr;
  for (const InitializerInfo& initializer : report.initializers) {
    if (initializer.name == node.inputs[1]) return &initializer;
  }
  return nullptr;
}

bool has_operator(const ModelAnalysisReport& report, const std::string& op_type) {
  for (const OperatorStat& stat : report.operators) {
    if (stat.domain == "ai.onnx" && stat.op_type == op_type && stat.count > 0) return true;
  }
  return false;
}

struct Linear {
  bool constant = false;
  double scale = 0;
  double bias = 0;
  std::string sigmoid;
};

std::optional<Linear> eval_linear(const GraphIndex& index, const std::string& tensor, int depth) {
  if (depth > 8 || tensor.empty()) return std::nullopt;
  const auto scalar = index.scalars.find(tensor);
  if (scalar != index.scalars.end()) {
    Linear value;
    value.constant = true;
    value.bias = scalar->second;
    return value;
  }
  const OnnxNode* node = producer_of(index, tensor);
  if (node == nullptr || node->outputs.size() != 1) return std::nullopt;
  if (node->op_type == "Sigmoid" && node->inputs.size() == 1) {
    Linear value;
    value.scale = 1;
    value.sigmoid = tensor;
    return value;
  }
  if (node->op_type == "Identity" && node->inputs.size() == 1) {
    return eval_linear(index, node->inputs[0], depth + 1);
  }
  if (node->op_type == "Neg" && node->inputs.size() == 1) {
    const std::optional<Linear> inner = eval_linear(index, node->inputs[0], depth + 1);
    if (!inner.has_value() || inner->constant) return std::nullopt;
    Linear value = *inner;
    value.scale = -value.scale;
    value.bias = -value.bias;
    return value;
  }
  if (node->inputs.size() != 2) return std::nullopt;
  if (node->op_type != "Add" && node->op_type != "Sub" && node->op_type != "Mul" && node->op_type != "Div") {
    return std::nullopt;
  }
  const std::optional<Linear> left = eval_linear(index, node->inputs[0], depth + 1);
  const std::optional<Linear> right = eval_linear(index, node->inputs[1], depth + 1);
  if (!left.has_value() || !right.has_value()) return std::nullopt;
  if (left->constant && right->constant) {
    Linear value;
    value.constant = true;
    if (node->op_type == "Add") value.bias = left->bias + right->bias;
    else if (node->op_type == "Sub") value.bias = left->bias - right->bias;
    else if (node->op_type == "Mul") value.bias = left->bias * right->bias;
    else if (right->bias == 0) return std::nullopt;
    else value.bias = left->bias / right->bias;
    return value;
  }
  const Linear* signal = nullptr;
  const Linear* constant = nullptr;
  bool constant_on_left = false;
  if (left->constant && !right->constant) {
    signal = &*right;
    constant = &*left;
    constant_on_left = true;
  } else if (right->constant && !left->constant) {
    signal = &*left;
    constant = &*right;
  } else {
    return std::nullopt;
  }
  if (!signal->sigmoid.empty() && !constant->constant) return std::nullopt;
  Linear value = *signal;
  const double number = constant->bias;
  if (node->op_type == "Add") value.bias += number;
  else if (node->op_type == "Sub") {
    if (constant_on_left) {
      value.scale = -value.scale;
      value.bias = number - value.bias;
    } else {
      value.bias -= number;
    }
  } else if (node->op_type == "Mul") {
    value.scale *= number;
    value.bias *= number;
  } else if (constant_on_left || number == 0) {
    return std::nullopt;
  } else {
    value.scale /= number;
    value.bias /= number;
  }
  return value;
}

bool close_to(double value, double target) { return std::isfinite(value) && std::fabs(value - target) <= 1e-4; }

bool is_angle_affine(const Linear& value) {
  return !value.constant && !value.sigmoid.empty() && close_to(value.scale, kPi) &&
         close_to(value.bias, -0.25 * kPi);
}

bool three_convs(const ModelAnalysisReport& report, const GraphIndex& index, const std::string& tensor,
                 int expected_channels, int& channels) {
  const OnnxNode* concat = producer_of(index, tensor);
  std::string current = tensor;
  for (int step = 0; step < 3 && concat != nullptr &&
                     (concat->op_type == "Reshape" || concat->op_type == "Identity" || concat->op_type == "Transpose");
       ++step) {
    if (concat->inputs.empty()) return false;
    current = concat->inputs[0];
    concat = producer_of(index, current);
  }
  if (concat == nullptr || concat->op_type != "Concat" || concat->inputs.size() != 3) return false;
  int found = -1;
  for (const std::string& input : concat->inputs) {
    const OnnxNode* node = producer_of(index, input);
    for (int step = 0; step < 4 && node != nullptr &&
                       (node->op_type == "Reshape" || node->op_type == "Identity" || node->op_type == "Transpose");
         ++step) {
      if (node->inputs.empty()) return false;
      node = producer_of(index, node->inputs[0]);
    }
    if (node == nullptr || node->op_type != "Conv") return false;
    const InitializerInfo* weight = weight_of(report, *node);
    std::vector<std::int64_t> dims;
    if (weight == nullptr || !static_dims(weight->declared_shape, dims) || dims.size() != 4 || dims[0] < 1) {
      return false;
    }
    if (found < 0) found = static_cast<int>(dims[0]);
    else if (found != dims[0]) return false;
  }
  if (expected_channels > 0 && found != expected_channels) return false;
  channels = found;
  return channels >= 1;
}

std::optional<int> dfl_reg_max(const ModelAnalysisReport& report, const OnnxNode& conv) {
  const InitializerInfo* weight = weight_of(report, conv);
  std::vector<std::int64_t> dims;
  if (weight == nullptr || !static_dims(weight->declared_shape, dims) || dims.size() != 4) return std::nullopt;
  if (dims[0] != 1 || dims[2] != 1 || dims[3] != 1 || dims[1] < 2) return std::nullopt;
  return static_cast<int>(dims[1]);
}

bool backward_has(const GraphIndex& index, const std::string& start, const std::string& op_type) {
  std::vector<std::string> pending{start};
  std::set<std::string> seen;
  int steps = 0;
  while (!pending.empty() && steps < 128) {
    const std::string current = pending.back();
    pending.pop_back();
    if (!seen.insert(current).second) continue;
    ++steps;
    const OnnxNode* node = producer_of(index, current);
    if (node == nullptr) continue;
    if (node->op_type == op_type) return true;
    // A convolution is a feature boundary. Cos and Sin that prove the rotated
    // box decode sit above it; walking through Conv would enter the trunk.
    if (node->op_type == "Conv") continue;
    for (const std::string& input : node->inputs) {
      if (!input.empty() && index.scalars.find(input) == index.scalars.end()) pending.push_back(input);
    }
  }
  return false;
}

struct HeadFacts {
  int num_classes = 0;
  int reg_max = 0;
  int channel_axis = -1;
  std::string output_name;
  std::string input_shape;
  std::string output_shape;
};

std::optional<HeadFacts> match_obb(const ModelAnalysisReport& report) {
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
  if (!input_shape.rank_known || input_shape.dimensions.size() != 4 || !output_shape.rank_known ||
      output_shape.dimensions.size() != 3) {
    return std::nullopt;
  }
  if (input_shape.dimensions[1].kind != DimensionKind::Static || input_shape.dimensions[1].value != 3) {
    return std::nullopt;
  }
  if (output_shape.dimensions[0].kind != DimensionKind::Static || output_shape.dimensions[0].value != 1) {
    return std::nullopt;
  }
  if (has_operator(report, "NonMaxSuppression")) return std::nullopt;
  const GraphIndex index = index_graph(report);
  const OnnxNode* final_node = producer_of(index, output.name);
  if (final_node != nullptr && final_node->op_type == "Identity" && !final_node->inputs.empty()) {
    final_node = producer_of(index, final_node->inputs[0]);
  }
  if (final_node == nullptr || final_node->op_type != "Concat") return std::nullopt;

  std::vector<std::string> branches;
  std::vector<std::string> pending = final_node->inputs;
  int guard = 0;
  while (!pending.empty() && guard++ < 16) {
    const std::string tensor = pending.back();
    pending.pop_back();
    const OnnxNode* node = producer_of(index, tensor);
    if (node != nullptr && node->op_type == "Concat" && node != final_node) {
      for (auto it = node->inputs.rbegin(); it != node->inputs.rend(); ++it) pending.push_back(*it);
    } else {
      branches.push_back(tensor);
    }
  }
  if (branches.size() != 3) return std::nullopt;

  std::string angle_tensor;
  std::string angle_sigmoid;
  for (const std::string& branch : branches) {
    const std::optional<Linear> linear = eval_linear(index, branch, 0);
    if (!linear.has_value() || !is_angle_affine(*linear)) continue;
    if (!angle_tensor.empty()) return std::nullopt;
    angle_tensor = branch;
    angle_sigmoid = linear->sigmoid;
  }
  if (angle_tensor.empty()) return std::nullopt;
  const OnnxNode* angle_sigmoid_node = producer_of(index, angle_sigmoid);
  if (angle_sigmoid_node == nullptr || angle_sigmoid_node->op_type != "Sigmoid" ||
      angle_sigmoid_node->inputs.empty()) {
    return std::nullopt;
  }
  int angle_channels = 0;
  if (!three_convs(report, index, angle_sigmoid_node->inputs[0], 1, angle_channels)) return std::nullopt;

  std::string class_tensor;
  std::string box_tensor;
  for (const std::string& branch : branches) {
    if (branch == angle_tensor) continue;
    const OnnxNode* node = producer_of(index, branch);
    if (node != nullptr && node->op_type == "Sigmoid" && branch != angle_sigmoid) {
      if (!class_tensor.empty()) return std::nullopt;
      class_tensor = branch;
    } else {
      if (!box_tensor.empty()) return std::nullopt;
      box_tensor = branch;
    }
  }
  if (class_tensor.empty() || box_tensor.empty()) return std::nullopt;
  const OnnxNode* class_sigmoid = producer_of(index, class_tensor);
  if (class_sigmoid == nullptr || class_sigmoid->inputs.empty()) return std::nullopt;
  int num_classes = 0;
  if (!three_convs(report, index, class_sigmoid->inputs[0], 0, num_classes) || num_classes < 1) return std::nullopt;

  if (!backward_has(index, box_tensor, "Cos") && !backward_has(index, box_tensor, "Sin")) return std::nullopt;
  const OnnxNode* dfl_conv = nullptr;
  int reg_max = 0;
  for (const OnnxNode& node : report.nodes) {
    if (node.depth != 0 || node.op_type != "Softmax" || node.outputs.empty() || node.inputs.empty()) continue;
    if (!reaches(index, node.outputs[0], box_tensor)) continue;
    const auto consumers = index.consumers.find(node.outputs[0]);
    if (consumers == index.consumers.end()) continue;
    for (const OnnxNode* consumer : consumers->second) {
      if (consumer->op_type != "Conv" || consumer->outputs.empty()) continue;
      const std::optional<int> width = dfl_reg_max(report, *consumer);
      if (!width.has_value() || !reaches(index, consumer->outputs[0], box_tensor)) continue;
      if (dfl_conv != nullptr) return std::nullopt;
      dfl_conv = consumer;
      reg_max = *width;
    }
  }
  if (dfl_conv == nullptr) return std::nullopt;
  int box_channels = 0;
  const OnnxNode* softmax = nullptr;
  for (const auto& entry : index.producers) {
    if (entry.second->op_type == "Softmax") {
      for (const std::string& output : entry.second->outputs) {
        const auto consumers = index.consumers.find(output);
        if (consumers == index.consumers.end()) continue;
        for (const OnnxNode* consumer : consumers->second) {
          if (consumer == dfl_conv) softmax = entry.second;
        }
      }
    }
  }
  if (softmax == nullptr || softmax->inputs.empty()) return std::nullopt;
  if (!three_convs(report, index, softmax->inputs[0], 4 * reg_max, box_channels)) return std::nullopt;

  int channel_axis = -1;
  const std::int64_t expect = static_cast<std::int64_t>(4 + num_classes + 1);
  for (std::size_t axis = 0; axis < output_shape.dimensions.size(); ++axis) {
    const ShapeDimension& dimension = output_shape.dimensions[axis];
    if (dimension.kind == DimensionKind::Static && dimension.value == expect) {
      if (channel_axis >= 0) return std::nullopt;
      channel_axis = static_cast<int>(axis);
    }
  }
  if (channel_axis < 1) return std::nullopt;

  HeadFacts facts;
  facts.num_classes = num_classes;
  facts.reg_max = reg_max;
  facts.channel_axis = channel_axis;
  facts.output_name = output.name;
  facts.input_shape = format_tensor_shape(input_shape);
  facts.output_shape = format_tensor_shape(output_shape);
  return facts;
}

}  // namespace

std::string YoloV8ObbRecognizer::id() const { return "yolov8_obb"; }

std::optional<RecognitionCandidate> YoloV8ObbRecognizer::consider(const ModelAnalysisReport& report) const {
  const std::optional<HeadFacts> facts = match_obb(report);
  if (!facts.has_value()) return std::nullopt;
  RecognitionCandidate candidate;
  candidate.recognizer_id = id();
  candidate.architecture = "yolov8_obb";
  candidate.family = "yolov8";
  candidate.evidence.push_back(
      {"single_input_axis1_is_3",
       "the only runtime input is float " + facts->input_shape + " with declared axis 1 equal to 3"});
  candidate.evidence.push_back(
      {"angle_affine_quarter_pi",
       "one output branch is (sigmoid - 1/4) * pi, and that sigmoid is fed by three convs of 1 channel"});
  candidate.evidence.push_back(
      {"class_sigmoid_three_scales",
       "another branch is a Sigmoid of three convs with " + std::to_string(facts->num_classes) + " channels"});
  candidate.evidence.push_back(
      {"dfl_and_rotated_decode",
       "the box branch reaches Cos or Sin and one DFL conv of reg_max " + std::to_string(facts->reg_max) +
           ", fed by three box convs"});
  candidate.evidence.push_back(
      {"channel_count",
       "declared output " + facts->output_shape + " has axis " + std::to_string(facts->channel_axis) +
           " equal to 4 + classes + angle"});
  candidate.evidence.push_back(
      {"no_nms_node", "the graph has no NonMaxSuppression, so this tensor is still a raw candidate list"});
  candidate.decoder.protocol = "yolov8_obb_raw";
  candidate.decoder.output_name = facts->output_name;
  candidate.decoder.box_format = "cxcywh_angle";
  candidate.decoder.channel_order = "cxcywh_class_angle";
  candidate.decoder.channel_axis = facts->channel_axis;
  candidate.decoder.box_channels = 4;
  candidate.decoder.num_classes = facts->num_classes;
  candidate.decoder.class_scores_are_probabilities = true;
  candidate.decoder.reg_max = facts->reg_max;
  candidate.decoder.score_type = "probabilities";
  return candidate;
}

}  // namespace vie
