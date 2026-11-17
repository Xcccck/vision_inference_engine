#include "vie/recognizer/yolov8_seg_recognizer.hpp"

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

std::optional<int> conv_out_channels(const GraphIndex& index, const OnnxNode& conv) {
  if (conv.op_type != "Conv") return std::nullopt;
  const InitializerInfo* weight = weight_of(index, conv);
  if (weight == nullptr) return std::nullopt;
  std::vector<std::int64_t> dims;
  if (!static_dims(weight->declared_shape, dims) || dims.empty() || dims[0] < 1) return std::nullopt;
  return static_cast<int>(dims[0]);
}

std::optional<int> three_scale_channels(const GraphIndex& index, const OnnxNode& concat) {
  if (concat.op_type != "Concat" || concat.inputs.size() != 3) return std::nullopt;
  int channels = -1;
  for (const std::string& input : concat.inputs) {
    const OnnxNode* reshape = producer_of(index, input);
    if (reshape == nullptr || reshape->op_type != "Reshape" || reshape->inputs.empty()) {
      return std::nullopt;
    }
    const OnnxNode* conv = producer_of(index, reshape->inputs[0]);
    if (conv == nullptr) return std::nullopt;
    const std::optional<int> width = conv_out_channels(index, *conv);
    if (!width.has_value()) return std::nullopt;
    if (channels == -1) channels = *width;
    else if (channels != *width) return std::nullopt;
  }
  if (channels < 1) return std::nullopt;
  return channels;
}

bool is_constant_source(const GraphIndex& index, const std::string& name) {
  if (index.initializers.find(name) != index.initializers.end()) return true;
  const OnnxNode* node = producer_of(index, name);
  return node != nullptr && node->op_type == "Constant";
}

bool is_decoded_box(const GraphIndex& index, const OnnxNode* node, std::string& box_tensor) {
  if (node == nullptr || node->op_type != "Mul" || node->inputs.size() != 2 || node->outputs.empty()) {
    return false;
  }
  const OnnxNode* data = nullptr;
  bool saw_constant = false;
  for (const std::string& input : node->inputs) {
    if (is_constant_source(index, input)) {
      saw_constant = true;
    } else if (data == nullptr) {
      data = producer_of(index, input);
    } else {
      return false;
    }
  }
  if (!saw_constant || data == nullptr || data->op_type != "Concat" || data->inputs.size() != 2) {
    return false;
  }
  const OnnxNode* left = producer_of(index, data->inputs[0]);
  const OnnxNode* right = producer_of(index, data->inputs[1]);
  if (left == nullptr || right == nullptr) return false;
  const bool dist2bbox = (left->op_type == "Div" && right->op_type == "Sub") ||
                         (left->op_type == "Sub" && right->op_type == "Div");
  if (!dist2bbox) return false;
  box_tensor = node->outputs[0];
  return true;
}

bool is_class_sigmoid(const GraphIndex& index, const OnnxNode* node, int& num_classes) {
  if (node == nullptr || node->op_type != "Sigmoid" || node->inputs.size() != 1) return false;
  const OnnxNode* concat = producer_of(index, node->inputs[0]);
  if (concat == nullptr) return false;
  const std::optional<int> channels = three_scale_channels(index, *concat);
  if (!channels.has_value()) return false;
  num_classes = *channels;
  return true;
}

bool is_raw_coefficients(const GraphIndex& index, const OnnxNode* node, int& mask_channels) {
  if (node == nullptr) return false;
  const std::optional<int> channels = three_scale_channels(index, *node);
  if (!channels.has_value()) return false;
  mask_channels = *channels;
  return true;
}

std::optional<int> dfl_reg_max_reaching(const ModelAnalysisReport& report, const GraphIndex& index,
                                       const std::string& box_tensor) {
  int found = 0;
  int reg_max = 0;
  for (const OnnxNode& node : report.nodes) {
    if (node.op_type != "Softmax" || node.outputs.empty()) continue;
    const auto consumers = index.consumers.find(node.outputs[0]);
    if (consumers == index.consumers.end()) continue;
    for (const OnnxNode* consumer : consumers->second) {
      if (consumer->op_type != "Conv" || consumer->outputs.empty()) continue;
      const InitializerInfo* weight = weight_of(index, *consumer);
      if (weight == nullptr) continue;
      std::vector<std::int64_t> dims;
      if (!static_dims(weight->declared_shape, dims) || dims.size() != 4) continue;
      if (dims[0] != 1 || dims[2] != 1 || dims[3] != 1 || dims[1] < 2) continue;
      if (!reaches(index, consumer->outputs[0], box_tensor)) continue;
      ++found;
      reg_max = static_cast<int>(dims[1]);
    }
  }
  if (found != 1) return std::nullopt;
  return reg_max;
}

bool is_prototype(const ModelAnalysisReport& report, const GraphIndex& index, const OnnxNode* node,
                  int mask_channels) {
  if (node == nullptr || node->op_type != "Mul" || node->inputs.size() != 2 || node->outputs.empty()) {
    return false;
  }
  const OnnxNode* left = producer_of(index, node->inputs[0]);
  const OnnxNode* right = producer_of(index, node->inputs[1]);
  const OnnxNode* sigmoid = nullptr;
  std::string feature;
  if (left != nullptr && left->op_type == "Sigmoid" && left->inputs.size() == 1) {
    sigmoid = left;
    feature = node->inputs[1];
  } else if (right != nullptr && right->op_type == "Sigmoid" && right->inputs.size() == 1) {
    sigmoid = right;
    feature = node->inputs[0];
  } else {
    return false;
  }
  if (sigmoid->inputs[0] != feature) return false;
  const OnnxNode* conv = producer_of(index, feature);
  if (conv == nullptr) return false;
  const std::optional<int> channels = conv_out_channels(index, *conv);
  if (!channels.has_value() || *channels != mask_channels) return false;

  bool saw_transpose = false;
  for (const OnnxNode& candidate : report.nodes) {
    if (candidate.op_type != "ConvTranspose" || candidate.outputs.empty()) continue;
    if (reaches(index, candidate.outputs[0], node->outputs[0])) saw_transpose = true;
  }
  return saw_transpose;
}

struct SegFacts {
  const ValueEndpoint* predictions = nullptr;
  const ValueEndpoint* prototypes = nullptr;
  int num_classes = 0;
  int mask_channels = 0;
  int reg_max = 0;
  int channel_axis = -1;
  int prototype_height = 0;
  int prototype_width = 0;
};

std::optional<SegFacts> match_prediction(const ModelAnalysisReport& report, const GraphIndex& index,
                                         const ValueEndpoint& predictions, const ValueEndpoint& prototypes) {
  if (predictions.type.kind != OnnxTypeKind::Tensor || prototypes.type.kind != OnnxTypeKind::Tensor) {
    return std::nullopt;
  }
  if (predictions.type.element_type.runtime_type != DataType::Fp32 ||
      prototypes.type.element_type.runtime_type != DataType::Fp32) {
    return std::nullopt;
  }
  std::vector<std::int64_t> prediction_dims;
  std::vector<std::int64_t> prototype_dims;
  if (!static_dims(predictions.type.declared_shape, prediction_dims) || prediction_dims.size() != 3) {
    return std::nullopt;
  }
  if (!static_dims(prototypes.type.declared_shape, prototype_dims) || prototype_dims.size() != 4) {
    return std::nullopt;
  }

  const OnnxNode* final_node = producer_of(index, predictions.name);
  if (final_node == nullptr || final_node->op_type != "Concat" || final_node->inputs.size() != 3) {
    return std::nullopt;
  }
  std::string box_tensor;
  int num_classes = 0;
  int mask_channels = 0;
  bool saw_box = false;
  bool saw_class = false;
  bool saw_mask = false;
  for (const std::string& input : final_node->inputs) {
    const OnnxNode* branch = producer_of(index, input);
    std::string decoded;
    int classes = 0;
    int masks = 0;
    const bool box = is_decoded_box(index, branch, decoded);
    const bool classes_ok = is_class_sigmoid(index, branch, classes);
    const bool masks_ok = is_raw_coefficients(index, branch, masks);
    if (static_cast<int>(box) + static_cast<int>(classes_ok) + static_cast<int>(masks_ok) != 1) {
      return std::nullopt;
    }
    if (box) {
      if (saw_box) return std::nullopt;
      saw_box = true;
      box_tensor = decoded;
    }
    if (classes_ok) {
      if (saw_class) return std::nullopt;
      saw_class = true;
      num_classes = classes;
    }
    if (masks_ok) {
      if (saw_mask) return std::nullopt;
      saw_mask = true;
      mask_channels = masks;
    }
  }
  if (!saw_box || !saw_class || !saw_mask) return std::nullopt;

  const std::optional<int> reg_max = dfl_reg_max_reaching(report, index, box_tensor);
  if (!reg_max.has_value()) return std::nullopt;
  const int expected = 4 + num_classes + mask_channels;
  int channel_axis = -1;
  for (std::size_t axis = 0; axis < prediction_dims.size(); ++axis) {
    if (prediction_dims[axis] != expected) continue;
    if (channel_axis != -1) return std::nullopt;
    channel_axis = static_cast<int>(axis);
  }
  if (channel_axis < 0) return std::nullopt;

  if (!is_prototype(report, index, producer_of(index, prototypes.name), mask_channels)) return std::nullopt;
  int channel_matches = 0;
  for (const std::int64_t dim : prototype_dims) {
    if (dim == mask_channels) ++channel_matches;
  }
  if (channel_matches != 1 || prototype_dims[1] != mask_channels) return std::nullopt;
  if (prototype_dims[2] < 1 || prototype_dims[3] < 1) return std::nullopt;

  SegFacts facts;
  facts.predictions = &predictions;
  facts.prototypes = &prototypes;
  facts.num_classes = num_classes;
  facts.mask_channels = mask_channels;
  facts.reg_max = *reg_max;
  facts.channel_axis = channel_axis;
  facts.prototype_height = static_cast<int>(prototype_dims[2]);
  facts.prototype_width = static_cast<int>(prototype_dims[3]);
  return facts;
}

std::optional<SegFacts> match_yolov8_seg(const ModelAnalysisReport& report) {
  if (report.runtime_inputs.size() != 1 || report.outputs.size() != 2) return std::nullopt;
  const ValueEndpoint& input = report.runtime_inputs.front();
  if (input.type.kind != OnnxTypeKind::Tensor || input.type.element_type.runtime_type != DataType::Fp32) {
    return std::nullopt;
  }
  const TensorShape& input_shape = input.type.declared_shape;
  if (!input_shape.rank_known || input_shape.dimensions.size() != 4) return std::nullopt;
  if (input_shape.dimensions[1].kind != DimensionKind::Static || input_shape.dimensions[1].value != 3) {
    return std::nullopt;
  }
  if (has_operator(report, "NonMaxSuppression")) return std::nullopt;

  const GraphIndex index = index_graph(report);
  const std::optional<SegFacts> forward =
      match_prediction(report, index, report.outputs[0], report.outputs[1]);
  const std::optional<SegFacts> swapped =
      match_prediction(report, index, report.outputs[1], report.outputs[0]);
  if (forward.has_value() == swapped.has_value()) return std::nullopt;
  return forward.has_value() ? forward : swapped;
}

}  // namespace

std::string YoloV8SegRecognizer::id() const { return "yolov8_segmentation"; }

std::optional<RecognitionCandidate> YoloV8SegRecognizer::consider(const ModelAnalysisReport& report) const {
  const std::optional<SegFacts> facts = match_yolov8_seg(report);
  if (!facts.has_value()) return std::nullopt;

  RecognitionCandidate candidate;
  candidate.recognizer_id = id();
  candidate.architecture = "yolov8_segmentation";
  candidate.family = "yolov8";
  candidate.evidence.push_back(
      {"two_outputs_prediction_and_prototype",
       "final nodes are a 3-input prediction Concat and a prototype SiLU, assigned by structure"});
  candidate.evidence.push_back(
      {"prediction_concat_box_class_coefficients",
       "prediction Concat inputs are decoded boxes, sigmoid class scores, and raw mask coefficients"});
  candidate.evidence.push_back(
      {"dfl_dist2bbox_already_decoded",
       "one DFL Softmax-Conv reaches the Div/Sub box, which is then multiplied by the stride constant"});
  candidate.evidence.push_back(
      {"class_sigmoid_without_objectness",
       "class scores are one Sigmoid over three scale convs; the prediction Concat has no objectness input"});
  candidate.evidence.push_back(
      {"mask_coefficients_without_activation",
       "mask coefficients are three scale convs concatenated directly into the prediction Concat"});
  candidate.evidence.push_back(
      {"prototype_channels_match_coefficients",
       "prototype SiLU conv channels equal the coefficient channels, and ConvTranspose reaches that output"});
  candidate.decoder.protocol = "yolov8_seg_raw";
  candidate.decoder.output_name = facts->predictions->name;
  candidate.decoder.box_format = "cxcywh";
  candidate.decoder.channel_order = "cxcywh_class_probabilities_mask_coefficients";
  candidate.decoder.channel_axis = facts->channel_axis;
  candidate.decoder.box_channels = 4;
  candidate.decoder.num_classes = facts->num_classes;
  candidate.decoder.class_scores_are_probabilities = true;
  candidate.decoder.reg_max = facts->reg_max;
  candidate.decoder.prototype_name = facts->prototypes->name;
  candidate.decoder.mask_channels = facts->mask_channels;
  candidate.decoder.prototype_channel_axis = 1;
  candidate.decoder.prototype_height = facts->prototype_height;
  candidate.decoder.prototype_width = facts->prototype_width;
  candidate.decoder.mask_coefficient_activation = "none";
  candidate.decoder.prototype_activation = "model_output_features";
  candidate.decoder.has_objectness = false;
  return candidate;
}

}  // namespace vie
