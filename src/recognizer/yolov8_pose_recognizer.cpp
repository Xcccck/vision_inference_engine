#include "vie/recognizer/yolov8_pose_recognizer.hpp"

#include <cmath>
#include <map>
#include <set>
#include <string>

namespace vie {
namespace {

struct GraphIndex {
  std::map<std::string, const OnnxNode*> producers;
  std::map<std::string, std::vector<const OnnxNode*>> consumers;
  std::map<std::string, const InitializerInfo*> initializers;
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
  for (const InitializerInfo& initializer : report.initializers) {
    index.initializers.emplace(initializer.name, &initializer);
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

bool close_to(double value, double target) { return std::isfinite(value) && std::fabs(value - target) <= 1e-4; }

bool is_view(const std::string& op_type) {
  return op_type == "Reshape" || op_type == "Identity" || op_type == "Transpose";
}

const OnnxNode* skip_view(const GraphIndex& index, const std::string& tensor, std::string& current) {
  current = tensor;
  const OnnxNode* node = producer_of(index, current);
  for (int step = 0; step < 4 && node != nullptr && is_view(node->op_type); ++step) {
    if (node->inputs.empty()) return nullptr;
    current = node->inputs[0];
    node = producer_of(index, current);
  }
  return node;
}

bool three_convs(const ModelAnalysisReport& report, const GraphIndex& index, const std::string& tensor,
                 int expected_channels, int& channels) {
  const OnnxNode* concat = producer_of(index, tensor);
  std::string current = tensor;
  for (int step = 0; step < 3 && concat != nullptr && is_view(concat->op_type); ++step) {
    if (concat->inputs.empty()) return false;
    current = concat->inputs[0];
    concat = producer_of(index, current);
  }
  if (concat == nullptr || concat->op_type != "Concat" || concat->inputs.size() != 3) return false;
  int found = -1;
  for (const std::string& input : concat->inputs) {
    const OnnxNode* node = producer_of(index, input);
    for (int step = 0; step < 4 && node != nullptr && is_view(node->op_type); ++step) {
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

bool walk_find(const GraphIndex& index, const std::string& start, const std::string& op_type, const OnnxNode*& found) {
  std::vector<std::string> pending{start};
  std::set<std::string> seen;
  int steps = 0;
  while (!pending.empty() && steps < 32) {
    const std::string current = pending.back();
    pending.pop_back();
    if (!seen.insert(current).second) continue;
    ++steps;
    const OnnxNode* node = producer_of(index, current);
    if (node == nullptr) continue;
    if (node->op_type == op_type) {
      found = node;
      return true;
    }
    if (node->op_type == "Conv") continue;
    for (const std::string& input : node->inputs) {
      if (!input.empty() && index.scalars.find(input) == index.scalars.end()) pending.push_back(input);
    }
  }
  return false;
}

bool mul_by_two(const GraphIndex& index, const std::string& start) {
  std::vector<std::string> pending{start};
  std::set<std::string> seen;
  int steps = 0;
  while (!pending.empty() && steps < 32) {
    const std::string current = pending.back();
    pending.pop_back();
    if (!seen.insert(current).second) continue;
    ++steps;
    const OnnxNode* node = producer_of(index, current);
    if (node == nullptr) continue;
    if (node->op_type == "Mul") {
      for (const std::string& input : node->inputs) {
        const auto scalar = index.scalars.find(input);
        if (scalar != index.scalars.end() && close_to(scalar->second, 2.0)) return true;
      }
    }
    if (node->op_type == "Conv") continue;
    for (const std::string& input : node->inputs) {
      if (!input.empty() && index.scalars.find(input) == index.scalars.end()) pending.push_back(input);
    }
  }
  return false;
}

bool slice_data(const GraphIndex& index, const std::string& start, std::string& data) {
  const OnnxNode* slice = nullptr;
  if (!walk_find(index, start, "Slice", slice) && !walk_find(index, start, "Split", slice)) return false;
  if (slice == nullptr || slice->inputs.empty()) return false;
  data = slice->inputs[0];
  return !data.empty();
}

bool is_constant_source(const GraphIndex& index, const std::string& name) {
  if (index.initializers.count(name) > 0 || index.scalars.count(name) > 0) return true;
  const OnnxNode* node = producer_of(index, name);
  return node != nullptr && node->op_type == "Constant";
}

bool decoded_box(const GraphIndex& index, const std::string& box_tensor) {
  std::string current;
  const OnnxNode* stride_mul = skip_view(index, box_tensor, current);
  if (stride_mul == nullptr || stride_mul->op_type != "Mul" || stride_mul->inputs.size() != 2) return false;
  const bool left_constant = is_constant_source(index, stride_mul->inputs[0]);
  const bool right_constant = is_constant_source(index, stride_mul->inputs[1]);
  if (left_constant == right_constant) return false;
  const std::string decoded = left_constant ? stride_mul->inputs[1] : stride_mul->inputs[0];
  std::string decoded_name;
  const OnnxNode* decoded_concat = skip_view(index, decoded, decoded_name);
  if (decoded_concat == nullptr || decoded_concat->op_type != "Concat" || decoded_concat->inputs.size() != 2) {
    return false;
  }
  const OnnxNode* left = producer_of(index, decoded_concat->inputs[0]);
  const OnnxNode* right = producer_of(index, decoded_concat->inputs[1]);
  if (left == nullptr || right == nullptr) return false;
  return (left->op_type == "Div" && right->op_type == "Sub") || (left->op_type == "Sub" && right->op_type == "Div");
}

bool dfl_box(const ModelAnalysisReport& report, const GraphIndex& index, const std::string& box_tensor, int& reg_max) {
  const OnnxNode* dfl_conv = nullptr;
  int found_reg_max = 0;
  for (const OnnxNode& node : report.nodes) {
    if (node.depth != 0 || node.op_type != "Softmax" || node.outputs.empty() || node.inputs.empty()) continue;
    if (!reaches(index, node.outputs[0], box_tensor)) continue;
    const auto consumers = index.consumers.find(node.outputs[0]);
    if (consumers == index.consumers.end()) continue;
    for (const OnnxNode* consumer : consumers->second) {
      if (consumer->op_type != "Conv" || consumer->outputs.empty()) continue;
      const std::optional<int> width = dfl_reg_max(report, *consumer);
      if (!width.has_value() || !reaches(index, consumer->outputs[0], box_tensor)) continue;
      if (dfl_conv != nullptr) return false;
      dfl_conv = consumer;
      found_reg_max = *width;
    }
  }
  if (dfl_conv == nullptr || !decoded_box(index, box_tensor)) return false;
  const OnnxNode* softmax = nullptr;
  for (const auto& entry : index.producers) {
    if (entry.second->op_type != "Softmax") continue;
    for (const std::string& output : entry.second->outputs) {
      const auto consumers = index.consumers.find(output);
      if (consumers == index.consumers.end()) continue;
      for (const OnnxNode* consumer : consumers->second) {
        if (consumer == dfl_conv) softmax = entry.second;
      }
    }
  }
  if (softmax == nullptr || softmax->inputs.empty()) return false;
  int box_channels = 0;
  if (!three_convs(report, index, softmax->inputs[0], 4 * found_reg_max, box_channels)) return false;
  reg_max = found_reg_max;
  return true;
}

bool keypoint_branch(const ModelAnalysisReport& report, const GraphIndex& index, const std::string& tensor,
                     int& keypoint_channels) {
  std::string current;
  const OnnxNode* concat = skip_view(index, tensor, current);
  if (concat == nullptr || concat->op_type != "Concat" || concat->inputs.size() != 2) return false;
  std::string visibility_data;
  std::string xy_data;
  bool saw_sigmoid = false;
  bool saw_mul = false;
  for (const std::string& input : concat->inputs) {
    std::string at;
    const OnnxNode* node = skip_view(index, input, at);
    if (node != nullptr && node->op_type == "Sigmoid" && !node->inputs.empty()) {
      if (saw_sigmoid || !slice_data(index, node->inputs[0], visibility_data)) return false;
      saw_sigmoid = true;
    } else if (mul_by_two(index, input)) {
      if (saw_mul || !slice_data(index, input, xy_data)) return false;
      saw_mul = true;
    } else {
      return false;
    }
  }
  if (!saw_sigmoid || !saw_mul || visibility_data.empty() || visibility_data != xy_data) return false;
  int channels = 0;
  if (!three_convs(report, index, visibility_data, 0, channels)) return false;
  if (channels < 3 || channels % 3 != 0) return false;
  keypoint_channels = channels;
  return true;
}

struct HeadFacts {
  int num_classes = 0;
  int reg_max = 0;
  int keypoint_count = 0;
  int channel_axis = -1;
  std::string output_name;
  std::string input_shape;
  std::string output_shape;
};

std::optional<HeadFacts> match_pose(const ModelAnalysisReport& report) {
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
  if (final_node == nullptr || final_node->op_type != "Concat" || final_node->inputs.size() != 3) return std::nullopt;

  std::string class_tensor;
  std::string box_tensor;
  std::string keypoint_tensor;
  for (const std::string& branch : final_node->inputs) {
    std::string current;
    const OnnxNode* node = skip_view(index, branch, current);
    if (node != nullptr && node->op_type == "Sigmoid") {
      if (!class_tensor.empty()) return std::nullopt;
      class_tensor = branch;
    } else if (decoded_box(index, branch)) {
      if (!box_tensor.empty()) return std::nullopt;
      box_tensor = branch;
    } else {
      if (!keypoint_tensor.empty()) return std::nullopt;
      keypoint_tensor = branch;
    }
  }
  if (class_tensor.empty() || box_tensor.empty() || keypoint_tensor.empty()) return std::nullopt;
  const OnnxNode* class_sigmoid = producer_of(index, class_tensor);
  if (class_sigmoid == nullptr || class_sigmoid->inputs.empty()) return std::nullopt;
  int num_classes = 0;
  if (!three_convs(report, index, class_sigmoid->inputs[0], 0, num_classes) || num_classes < 1) return std::nullopt;
  int reg_max = 0;
  if (!dfl_box(report, index, box_tensor, reg_max)) return std::nullopt;
  int keypoint_channels = 0;
  if (!keypoint_branch(report, index, keypoint_tensor, keypoint_channels)) return std::nullopt;

  int channel_axis = -1;
  const std::int64_t expect = static_cast<std::int64_t>(4 + num_classes + keypoint_channels);
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
  facts.keypoint_count = keypoint_channels / 3;
  facts.channel_axis = channel_axis;
  facts.output_name = output.name;
  facts.input_shape = format_tensor_shape(input_shape);
  facts.output_shape = format_tensor_shape(output_shape);
  return facts;
}

}  // namespace

std::string YoloV8PoseRecognizer::id() const { return "yolov8_pose"; }

std::optional<RecognitionCandidate> YoloV8PoseRecognizer::consider(const ModelAnalysisReport& report) const {
  const std::optional<HeadFacts> facts = match_pose(report);
  if (!facts.has_value()) return std::nullopt;
  RecognitionCandidate candidate;
  candidate.recognizer_id = id();
  candidate.architecture = "yolov8_pose";
  candidate.family = "yolov8";
  candidate.evidence.push_back(
      {"single_input_axis1_is_3",
       "the only runtime input is float " + facts->input_shape + " with declared axis 1 equal to 3"});
  candidate.evidence.push_back(
      {"person_box_dfl",
       "the output Concat has a class Sigmoid and a DFL box, and that box is a stride Mul of Div and Sub"});
  candidate.evidence.push_back(
      {"class_sigmoid_three_scales",
       "the class Sigmoid is fed by three convs with " + std::to_string(facts->num_classes) + " channels"});
  candidate.evidence.push_back(
      {"keypoint_xy_times_two",
       "the other branch concats a Mul by 2, reached from a Slice or Split, with a Sigmoid of the matching slice"});
  candidate.evidence.push_back(
      {"keypoint_third_channel",
       "three keypoint convs have " + std::to_string(facts->keypoint_count * 3) +
           " channels, divisible by 3, so each keypoint is x, y, and a sigmoid probability"});
  candidate.evidence.push_back(
      {"channel_count",
       "declared output " + facts->output_shape + " has axis " + std::to_string(facts->channel_axis) +
           " equal to 4 + classes + keypoints"});
  candidate.evidence.push_back(
      {"no_nms_node", "the graph has no NonMaxSuppression, so this tensor is still a raw candidate list"});
  candidate.decoder.protocol = "yolov8_pose_raw";
  candidate.decoder.output_name = facts->output_name;
  candidate.decoder.box_format = "cxcywh";
  candidate.decoder.channel_order = "cxcywh_class_xy_confidence";
  candidate.decoder.channel_axis = facts->channel_axis;
  candidate.decoder.box_channels = 4;
  candidate.decoder.num_classes = facts->num_classes;
  candidate.decoder.class_scores_are_probabilities = true;
  candidate.decoder.reg_max = facts->reg_max;
  candidate.decoder.keypoint_count = facts->keypoint_count;
  candidate.decoder.keypoint_dimensions = 3;
  candidate.decoder.keypoint_layout = "xy_confidence";
  candidate.decoder.score_type = "probabilities";
  return candidate;
}

}  // namespace vie
