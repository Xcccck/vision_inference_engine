#include "vie/recognizer/yolov8_detect_recognizer.hpp"

#include <map>
#include <set>
#include <utility>

namespace vie {
namespace {

struct GraphIndex {
  std::map<std::string, const OnnxNode*> producers;
  std::map<std::string, std::vector<const OnnxNode*>> consumers;
  std::map<std::string, const InitializerInfo*> initializers;
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

const InitializerInfo* weight_of(const GraphIndex& index, const OnnxNode& node) {
  if (node.inputs.size() < 2) return nullptr;
  const auto found = index.initializers.find(node.inputs[1]);
  if (found == index.initializers.end()) return nullptr;
  return found->second;
}

bool has_operator(const ModelAnalysisReport& report, const std::string& op_type) {
  for (const OperatorStat& stat : report.operators) {
    if (stat.domain == "ai.onnx" && stat.op_type == op_type && stat.count > 0) return true;
  }
  return false;
}

std::optional<int> dfl_reg_max(const GraphIndex& index, const OnnxNode& conv) {
  const InitializerInfo* weight = weight_of(index, conv);
  if (weight == nullptr) return std::nullopt;
  std::vector<std::int64_t> dims;
  if (!static_dims(weight->declared_shape, dims) || dims.size() != 4) return std::nullopt;
  if (dims[0] != 1 || dims[2] != 1 || dims[3] != 1 || dims[1] < 2) return std::nullopt;
  return static_cast<int>(dims[1]);
}

struct HeadFacts {
  int reg_max = 0;
  int num_classes = 0;
  int channel_axis = -1;
  std::string output_name;
  std::string input_shape;
  std::string output_shape;
};

std::optional<HeadFacts> match_yolov8_head(const ModelAnalysisReport& report) {
  if (report.runtime_inputs.size() != 1 || report.outputs.size() != 1) return std::nullopt;
  const ValueEndpoint& input = report.runtime_inputs.front();
  const ValueEndpoint& output = report.outputs.front();
  if (input.type.kind != OnnxTypeKind::Tensor || output.type.kind != OnnxTypeKind::Tensor) {
    return std::nullopt;
  }
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
  if (final_node == nullptr || final_node->op_type != "Concat" || final_node->inputs.size() != 2) {
    return std::nullopt;
  }
  const OnnxNode* first = producer_of(index, final_node->inputs[0]);
  const OnnxNode* second = producer_of(index, final_node->inputs[1]);
  const OnnxNode* sigmoid = nullptr;
  std::string box_tensor;
  if (first != nullptr && first->op_type == "Sigmoid") {
    sigmoid = first;
    box_tensor = final_node->inputs[1];
  } else if (second != nullptr && second->op_type == "Sigmoid") {
    sigmoid = second;
    box_tensor = final_node->inputs[0];
  } else {
    return std::nullopt;
  }
  if (sigmoid->inputs.size() != 1) return std::nullopt;
  const OnnxNode* split = producer_of(index, sigmoid->inputs[0]);
  if (split == nullptr || split->op_type != "Split" || split->outputs.size() != 2) return std::nullopt;
  std::string box_distribution;
  for (const std::string& split_output : split->outputs) {
    if (split_output != sigmoid->inputs[0]) box_distribution = split_output;
  }
  if (box_distribution.empty() || split->inputs.empty()) return std::nullopt;

  const OnnxNode* dfl_conv = nullptr;
  int reg_max = 0;
  for (const OnnxNode& node : report.nodes) {
    if (node.op_type != "Softmax" || node.outputs.empty() || node.inputs.empty()) continue;
    if (!reaches(index, box_distribution, node.inputs[0])) continue;
    const auto consumers = index.consumers.find(node.outputs[0]);
    if (consumers == index.consumers.end()) continue;
    for (const OnnxNode* consumer : consumers->second) {
      if (consumer->op_type != "Conv") continue;
      const std::optional<int> width = dfl_reg_max(index, *consumer);
      if (!width.has_value() || consumer->outputs.empty()) continue;
      if (!reaches(index, consumer->outputs[0], box_tensor)) continue;
      if (dfl_conv != nullptr) return std::nullopt;
      dfl_conv = consumer;
      reg_max = *width;
    }
  }
  if (dfl_conv == nullptr) return std::nullopt;

  const OnnxNode* scales = producer_of(index, split->inputs[0]);
  if (scales == nullptr || scales->op_type != "Concat" || scales->inputs.size() != 3) return std::nullopt;
  const int box_channels = 4 * reg_max;
  int num_classes = -1;
  for (const std::string& scale : scales->inputs) {
    const OnnxNode* reshape = producer_of(index, scale);
    if (reshape == nullptr || reshape->op_type != "Reshape" || reshape->inputs.empty()) return std::nullopt;
    const OnnxNode* branches = producer_of(index, reshape->inputs[0]);
    if (branches == nullptr || branches->op_type != "Concat" || branches->inputs.size() != 2) {
      return std::nullopt;
    }
    int class_channels = -1;
    bool saw_box = false;
    for (const std::string& branch : branches->inputs) {
      const OnnxNode* conv = producer_of(index, branch);
      if (conv == nullptr || conv->op_type != "Conv") return std::nullopt;
      const InitializerInfo* weight = weight_of(index, *conv);
      if (weight == nullptr) return std::nullopt;
      std::vector<std::int64_t> dims;
      if (!static_dims(weight->declared_shape, dims) || dims.empty() || dims[0] < 1) return std::nullopt;
      if (dims[0] == box_channels) {
        saw_box = true;
      } else if (class_channels == -1) {
        class_channels = static_cast<int>(dims[0]);
      } else if (class_channels != dims[0]) {
        return std::nullopt;
      }
    }
    if (!saw_box || class_channels < 1) return std::nullopt;
    if (num_classes == -1) num_classes = class_channels;
    else if (num_classes != class_channels) return std::nullopt;
  }

  const OnnxNode* stride_mul = producer_of(index, box_tensor);
  if (stride_mul == nullptr || stride_mul->op_type != "Mul" || stride_mul->inputs.size() != 2) {
    return std::nullopt;
  }
  const bool left_constant = index.initializers.count(stride_mul->inputs[0]) > 0;
  const bool right_constant = index.initializers.count(stride_mul->inputs[1]) > 0;
  if (left_constant == right_constant) return std::nullopt;
  const std::string decoded = left_constant ? stride_mul->inputs[1] : stride_mul->inputs[0];
  const OnnxNode* decoded_concat = producer_of(index, decoded);
  if (decoded_concat == nullptr || decoded_concat->op_type != "Concat" || decoded_concat->inputs.size() != 2) {
    return std::nullopt;
  }
  const OnnxNode* left_decode = producer_of(index, decoded_concat->inputs[0]);
  const OnnxNode* right_decode = producer_of(index, decoded_concat->inputs[1]);
  if (left_decode == nullptr || right_decode == nullptr) return std::nullopt;
  const bool center_and_size = (left_decode->op_type == "Div" && right_decode->op_type == "Sub") ||
                               (left_decode->op_type == "Sub" && right_decode->op_type == "Div");
  if (!center_and_size) return std::nullopt;

  int channel_axis = -1;
  for (std::size_t axis = 0; axis < output_shape.dimensions.size(); ++axis) {
    const ShapeDimension& dimension = output_shape.dimensions[axis];
    if (dimension.kind == DimensionKind::Static && dimension.value == num_classes + 4) {
      if (channel_axis >= 0) return std::nullopt;
      channel_axis = static_cast<int>(axis);
    }
  }
  if (channel_axis < 0) return std::nullopt;

  HeadFacts facts;
  facts.reg_max = reg_max;
  facts.num_classes = num_classes;
  facts.channel_axis = channel_axis;
  facts.output_name = output.name;
  facts.input_shape = format_tensor_shape(input_shape);
  facts.output_shape = format_tensor_shape(output_shape);
  return facts;
}

}  // namespace

std::string YoloV8DetectRecognizer::id() const { return "yolov8_detection"; }

std::optional<RecognitionCandidate> YoloV8DetectRecognizer::consider(
    const ModelAnalysisReport& report) const {
  const std::optional<HeadFacts> facts = match_yolov8_head(report);
  if (!facts.has_value()) return std::nullopt;

  RecognitionCandidate candidate;
  candidate.recognizer_id = id();
  candidate.architecture = "yolov8_detection";
  candidate.family = "yolov8";
  candidate.evidence.push_back(
      {"single_input_axis1_is_3",
       "the only runtime input is float " + facts->input_shape + " with declared axis 1 equal to 3"});
  candidate.evidence.push_back(
      {"final_concat_of_box_and_sigmoid",
       "output '" + facts->output_name + "' is a Concat of a decoded box tensor and a Sigmoid"});
  candidate.evidence.push_back(
      {"dfl_softmax_conv", "one Softmax feeds a Conv whose weight is [1, " + std::to_string(facts->reg_max) +
                               ", 1, 1], and that Conv reaches the box tensor"});
  candidate.evidence.push_back(
      {"three_scale_heads",
       "the Split feeding the Sigmoid is a Concat of three Reshape nodes, each joining a box conv and a class conv"});
  candidate.evidence.push_back(
      {"class_channels_match_output",
       "the three class conv weights agree on " + std::to_string(facts->num_classes) +
           " channels, and declared output " + facts->output_shape + " has axis " +
           std::to_string(facts->channel_axis) + " equal to 4+" + std::to_string(facts->num_classes)});
  candidate.evidence.push_back(
      {"dist2bbox_cxcywh",
       "the box tensor is a stride Mul of a Concat of Div and Sub, which is the center/size decode"});
  candidate.evidence.push_back(
      {"no_objectness_or_nms",
       "the final Concat has two inputs and the graph has no NonMaxSuppression node"});
  candidate.decoder.protocol = "yolov8_detect_raw";
  candidate.decoder.output_name = facts->output_name;
  candidate.decoder.box_format = "cxcywh";
  candidate.decoder.channel_order = "cxcywh_then_class_scores";
  candidate.decoder.channel_axis = facts->channel_axis;
  candidate.decoder.box_channels = 4;
  candidate.decoder.num_classes = facts->num_classes;
  candidate.decoder.class_scores_are_probabilities = true;
  candidate.decoder.reg_max = facts->reg_max;
  return candidate;
}

}  // namespace vie
