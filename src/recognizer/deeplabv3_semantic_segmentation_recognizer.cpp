#include "vie/recognizer/deeplabv3_semantic_segmentation_recognizer.hpp"

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

bool is_float_tensor(const ValueEndpoint& endpoint) {
  return endpoint.type.kind == OnnxTypeKind::Tensor &&
         endpoint.type.element_type.runtime_type == DataType::Fp32;
}

// Identity-style layout ops that never change the numeric meaning of the logits.
bool passthrough(const std::string& op_type) {
  return op_type == "Identity" || op_type == "Reshape" || op_type == "Transpose" ||
         op_type == "Squeeze" || op_type == "Unsqueeze" || op_type == "Cast";
}

// Spatial upsampling of a dense map. DeepLabV3 emits the low-resolution logits
// from a 1x1 Conv and then interpolates to the model-input grid with Resize.
bool upsample(const std::string& op_type) { return op_type == "Resize" || op_type == "Upsample"; }

// Ops that mean this is NOT a raw dense-logit head. Softmax / ArgMax post-process
// the classes; a global pool reduces the spatial map (classification head).
bool disqualifying(const std::string& op_type) {
  return op_type == "Softmax" || op_type == "LogSoftmax" || op_type == "ArgMax" ||
         op_type == "GlobalAveragePool" || op_type == "GlobalMaxPool" || op_type == "AveragePool" ||
         op_type == "MaxPool";
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

// A dense semantic-segmentation output is Float32 [N, C, H, W]: a static batch
// axis, a static class axis >= 2 at position 1, and two static spatial axes.
// Returns num_classes when the shape fits that contract, otherwise 0.
int dense_class_count(const TensorShape& shape) {
  std::vector<std::int64_t> dims;
  if (!static_dims(shape, dims) || dims.size() != 4) return 0;
  const std::int64_t num_classes = dims[1];
  if (num_classes < 2) return 0;
  if (dims[2] < 1 || dims[3] < 1) return 0;
  return static_cast<int>(num_classes);
}

struct SegmentHead {
  const OnnxNode* conv = nullptr;
  std::vector<std::string> path;
};

// Walk back from the graph output through Resize / passthrough ops only, until a
// Conv is reached. Any post-processing op aborts the walk. The Conv must be a
// 1x1 projection whose output channel count equals num_classes.
std::optional<SegmentHead> semantic_head(const GraphIndex& index, const std::string& output_name,
                                         int num_classes) {
  SegmentHead head;
  std::string current = output_name;
  std::set<std::string> seen;
  for (int step = 0; step < 16; ++step) {
    if (!seen.insert(current).second) return std::nullopt;
    const OnnxNode* node = producer_of(index, current);
    if (node == nullptr || node->inputs.empty()) return std::nullopt;
    if (disqualifying(node->op_type)) return std::nullopt;
    head.path.push_back(node->op_type);
    if (node->op_type == "Conv") {
      if (node->inputs.size() < 2) return std::nullopt;
      const InitializerInfo* weight = initializer_of(index, node->inputs[1]);
      if (weight == nullptr) return std::nullopt;
      std::vector<std::int64_t> dims;
      if (!static_dims(weight->declared_shape, dims) || dims.size() != 4) return std::nullopt;
      // [out_channels, in_channels/group, kH, kW]: 1x1 dense class projection.
      if (dims[0] != num_classes) return std::nullopt;
      if (dims[2] != 1 || dims[3] != 1) return std::nullopt;
      head.conv = node;
      return head;
    }
    if (upsample(node->op_type) || passthrough(node->op_type)) {
      current = node->inputs[0];
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace

std::string Deeplabv3SemanticSegmentationRecognizer::id() const {
  return "deeplabv3_semantic_segmentation";
}

std::optional<RecognitionCandidate> Deeplabv3SemanticSegmentationRecognizer::consider(
    const ModelAnalysisReport& report) const {
  if (report.runtime_inputs.size() != 1 || report.outputs.size() != 1) return std::nullopt;
  const ValueEndpoint& input = report.runtime_inputs.front();
  const ValueEndpoint& output = report.outputs.front();
  if (!is_float_tensor(input) || !is_float_tensor(output)) return std::nullopt;

  // Input must be a 4-channel-dim NCHW image: axis 1 is the static 3 color
  // channels. This is what separates a dense seg model from an embedding or a
  // feature-input regression network.
  std::vector<std::int64_t> input_dims;
  if (!static_dims(input.type.declared_shape, input_dims) || input_dims.size() != 4) {
    return std::nullopt;
  }
  if (input_dims[1] != 3) return std::nullopt;

  const int num_classes = dense_class_count(output.type.declared_shape);
  if (num_classes < 2) return std::nullopt;
  if (!has_conv(report)) return std::nullopt;

  const GraphIndex index = index_root(report);
  const std::optional<SegmentHead> head = semantic_head(index, output.name, num_classes);
  if (!head.has_value()) return std::nullopt;

  RecognitionCandidate candidate;
  candidate.recognizer_id = id();
  candidate.architecture = "deeplabv3_semantic_segmentation";
  candidate.family = "deeplabv3";
  candidate.evidence.push_back(
      {"single_runtime_input", "one fp32 NCHW input '" + input.name + "' " + shape_text(input.type.declared_shape) +
                                   " with 3 channels on axis 1"});
  candidate.evidence.push_back({"single_output",
                                "one fp32 rank-4 output '" + output.name + "' " +
                                    shape_text(output.type.declared_shape)});
  candidate.evidence.push_back({"dense_class_axis",
                                "output axis 1 is static class dimension " + std::to_string(num_classes) +
                                    " with two spatial axes"});
  candidate.evidence.push_back({"terminal_1x1_conv",
                                "a 1x1 Conv with " + std::to_string(num_classes) +
                                    " output channels projects the dense feature map"});
  std::string path_text;
  for (const std::string& op : head->path) {
    if (!path_text.empty()) path_text += " -> ";
    path_text += op;
  }
  candidate.evidence.push_back({"raw_logits_head",
                                "the path from the class Conv to '" + output.name + "' is " + path_text +
                                    " and runs no Softmax, ArgMax, or pooling"});
  candidate.evidence.push_back({"convolutional_backbone", "the graph contains Conv operators"});

  candidate.decoder.protocol = "deeplabv3_semantic_seg_raw";
  candidate.decoder.output_name = output.name;
  candidate.decoder.num_classes = num_classes;
  candidate.decoder.channel_axis = 1;
  return candidate;
}

}  // namespace vie
