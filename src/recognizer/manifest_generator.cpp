#include "vie/recognizer/manifest_generator.hpp"

#include <utility>

namespace vie {
namespace {

void put(ModelManifest& manifest, const std::string& path, Certainty certainty, std::string evidence) {
  FieldProvenance record;
  record.certainty = certainty;
  if (!evidence.empty()) record.evidence.push_back(std::move(evidence));
  manifest.provenance[path] = std::move(record);
}

std::vector<std::int64_t> manifest_shape(const TensorShape& shape) {
  if (!shape.rank_known || shape.dimensions.empty()) return {};
  std::vector<std::int64_t> values;
  values.reserve(shape.dimensions.size());
  for (const ShapeDimension& dimension : shape.dimensions) {
    if (dimension.kind == DimensionKind::Static && dimension.value > 0) {
      values.push_back(dimension.value);
    } else {
      values.push_back(-1);
    }
  }
  return values;
}

void describe_endpoint(const ValueEndpoint& endpoint, std::string& name, std::vector<std::int64_t>& shape,
                       DataType& dtype) {
  name = endpoint.name;
  if (endpoint.type.kind == OnnxTypeKind::Tensor || endpoint.type.kind == OnnxTypeKind::SparseTensor) {
    shape = manifest_shape(endpoint.type.declared_shape);
    if (endpoint.type.element_type.runtime_type.has_value()) {
      dtype = *endpoint.type.element_type.runtime_type;
    }
  }
}

std::string joined_evidence(const RecognitionCandidate& candidate) {
  std::string text;
  for (const RecognitionEvidence& evidence : candidate.evidence) {
    if (!text.empty()) text += "; ";
    text += evidence.code + ": " + evidence.detail;
  }
  return text;
}

std::vector<std::string> quoted_class_names(const std::string& text) {
  std::vector<std::string> names;
  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] != '\'') continue;
    const std::size_t end = text.find('\'', index + 1);
    if (end == std::string::npos) return {};
    names.push_back(text.substr(index + 1, end - index - 1));
    index = end;
  }
  return names;
}

const std::string* metadata_value(const ModelAnalysisReport& report, const std::string& key) {
  for (const auto& entry : report.metadata) {
    if (entry.first == key) return &entry.second;
  }
  return nullptr;
}

}  // namespace

ModelManifest ManifestGenerator::generate(const ModelAnalysisReport& report,
                                          const RecognitionResult& recognition,
                                          const std::string& model_path) const {
  ModelManifest manifest;
  manifest.schema_version = "1.0";
  manifest.model_path = model_path;
  if (!report.graph_name.empty()) {
    manifest.model_name = report.graph_name;
    put(manifest, "model_name", Certainty::Observed, "ONNX graph name");
  }
  if (!model_path.empty()) {
    manifest.model_format = ModelFormat::Onnx;
    put(manifest, "model_format", Certainty::Observed, "file parsed as an ONNX ModelProto");
    put(manifest, "model_path", Certainty::Observed, "path passed to the analyzer");
  }

  if (!report.runtime_inputs.empty()) {
    describe_endpoint(report.runtime_inputs.front(), manifest.input.name, manifest.input.shape,
                      manifest.input.dtype);
    if (!manifest.input.name.empty()) {
      put(manifest, "input.name", Certainty::Observed, "runtime input name");
    }
    if (!manifest.input.shape.empty()) {
      put(manifest, "input.shape", Certainty::Observed, "declared shape of the runtime input");
    }
    if (manifest.input.dtype != DataType::Unknown) {
      put(manifest, "input.dtype", Certainty::Observed, "declared element type of the runtime input");
    }
  }
  for (const ValueEndpoint& output : report.outputs) {
    if (output.type.kind != OnnxTypeKind::Tensor && output.type.kind != OnnxTypeKind::SparseTensor) {
      continue;
    }
    OutputDesc described;
    describe_endpoint(output, described.name, described.shape, described.dtype);
    const std::size_t index = manifest.outputs.size();
    const std::string prefix = "outputs." + std::to_string(index) + ".";
    manifest.outputs.push_back(std::move(described));
    if (!manifest.outputs.back().name.empty()) {
      put(manifest, prefix + "name", Certainty::Observed, "declared output name");
    }
    if (!manifest.outputs.back().shape.empty()) {
      put(manifest, prefix + "shape", Certainty::Observed, "declared output shape");
    }
    if (manifest.outputs.back().dtype != DataType::Unknown) {
      put(manifest, prefix + "dtype", Certainty::Observed, "declared output element type");
    }
  }

  const RecognitionCandidate* selected = recognition.selected();
  if (selected != nullptr && selected->architecture == "yolov8_detection") {
    const std::string because = joined_evidence(*selected);
    manifest.model_family = selected->family;
    manifest.task = TaskType::Detection;
    put(manifest, "model_family", Certainty::Inferred, because);
    put(manifest, "task", Certainty::Inferred, because);
    if (selected->decoder.num_classes.has_value()) {
      manifest.num_classes = selected->decoder.num_classes;
      put(manifest, "num_classes", Certainty::Inferred, because);
    }
    if (manifest.input.shape.size() == 4 && manifest.input.shape[1] == 3) {
      manifest.input.layout = Layout::Nchw;
      put(manifest, "input.layout", Certainty::Inferred,
          "declared input axis 1 is static 3 and the detection head consumes that tensor");
    }
    manifest.decoder.protocol = selected->decoder.protocol;
    put(manifest, "decoder.protocol", Certainty::Inferred, because);
    if (!selected->decoder.box_format.empty()) {
      manifest.decoder.box_format = selected->decoder.box_format;
      put(manifest, "decoder.box_format", Certainty::Inferred, because);
    }
    manifest.decoder.box_decode = "decoded_cxcywh";
    put(manifest, "decoder.box_decode", Certainty::Inferred, because);
    manifest.decoder.scores_are_probabilities = selected->decoder.class_scores_are_probabilities;
    put(manifest, "decoder.scores_are_probabilities", Certainty::Inferred, because);
    if (selected->decoder.box_channels > 0) {
      manifest.decoder.box_channels = selected->decoder.box_channels;
      put(manifest, "decoder.box_channels", Certainty::Inferred, because);
    }
    if (selected->decoder.channel_axis >= 0) {
      manifest.decoder.channel_axis = selected->decoder.channel_axis;
      put(manifest, "decoder.channel_axis", Certainty::Inferred, because);
    }
    if (selected->decoder.reg_max.has_value()) {
      manifest.decoder.reg_max = selected->decoder.reg_max;
      put(manifest, "decoder.reg_max", Certainty::Inferred, because);
    }
    if (!manifest.outputs.empty() && !selected->decoder.channel_order.empty()) {
      manifest.outputs.front().semantic = selected->decoder.channel_order;
      put(manifest, "outputs.0.semantic", Certainty::Inferred, because);
    }
  }
  if (selected != nullptr && selected->architecture == "yolov8_segmentation") {
    const std::string because = joined_evidence(*selected);
    manifest.model_family = selected->family;
    manifest.task = TaskType::Segmentation;
    put(manifest, "model_family", Certainty::Inferred, because);
    put(manifest, "task", Certainty::Inferred, because);
    if (selected->decoder.num_classes.has_value()) {
      manifest.num_classes = selected->decoder.num_classes;
      put(manifest, "num_classes", Certainty::Inferred, because);
    }
    if (manifest.input.shape.size() == 4 && manifest.input.shape[1] == 3) {
      manifest.input.layout = Layout::Nchw;
      put(manifest, "input.layout", Certainty::Inferred,
          "declared input axis 1 is static 3 and the segmentation head consumes that tensor");
    }
    manifest.decoder.protocol = selected->decoder.protocol;
    put(manifest, "decoder.protocol", Certainty::Inferred, because);
    if (!selected->decoder.box_format.empty()) {
      manifest.decoder.box_format = selected->decoder.box_format;
      put(manifest, "decoder.box_format", Certainty::Inferred, because);
    }
    manifest.decoder.box_decode = "decoded_cxcywh";
    put(manifest, "decoder.box_decode", Certainty::Inferred, because);
    manifest.decoder.scores_are_probabilities = selected->decoder.class_scores_are_probabilities;
    put(manifest, "decoder.scores_are_probabilities", Certainty::Inferred, because);
    if (selected->decoder.box_channels > 0) {
      manifest.decoder.box_channels = selected->decoder.box_channels;
      put(manifest, "decoder.box_channels", Certainty::Inferred, because);
    }
    if (selected->decoder.channel_axis >= 0) {
      manifest.decoder.channel_axis = selected->decoder.channel_axis;
      put(manifest, "decoder.channel_axis", Certainty::Inferred, because);
    }
    if (selected->decoder.reg_max.has_value()) {
      manifest.decoder.reg_max = selected->decoder.reg_max;
      put(manifest, "decoder.reg_max", Certainty::Inferred, because);
    }
    for (std::size_t index = 0; index < manifest.outputs.size(); ++index) {
      OutputDesc& output = manifest.outputs[index];
      const std::string prefix = "outputs." + std::to_string(index) + ".";
      if (output.name == selected->decoder.output_name && !selected->decoder.channel_order.empty()) {
        output.semantic = selected->decoder.channel_order;
        put(manifest, prefix + "semantic", Certainty::Inferred, because);
      }
      if (output.name == selected->decoder.prototype_name) {
        output.semantic = "mask_prototypes";
        output.layout = Layout::Nchw;
        put(manifest, prefix + "semantic", Certainty::Inferred, because);
        put(manifest, prefix + "layout", Certainty::Inferred,
            "prototype channel axis 1 matches the coefficient channel count");
      }
    }
    if (selected->decoder.mask_channels.has_value()) {
      manifest.segmentation.mask_channels = selected->decoder.mask_channels;
      put(manifest, "segmentation.mask_channels", Certainty::Inferred, because);
    }
    if (!selected->decoder.prototype_name.empty()) {
      manifest.segmentation.prototype_name = selected->decoder.prototype_name;
      put(manifest, "segmentation.prototype_name", Certainty::Observed,
          "declared name of the prototype output");
    }
    manifest.segmentation.prototype_layout = Layout::Nchw;
    put(manifest, "segmentation.prototype_layout", Certainty::Inferred, because);
    if (selected->decoder.prototype_channel_axis.has_value()) {
      manifest.segmentation.prototype_channel_axis = selected->decoder.prototype_channel_axis;
      put(manifest, "segmentation.prototype_channel_axis", Certainty::Inferred, because);
    }
    if (selected->decoder.prototype_height.has_value()) {
      manifest.segmentation.prototype_height = selected->decoder.prototype_height;
      put(manifest, "segmentation.prototype_height", Certainty::Observed,
          "declared spatial height of the prototype output");
    }
    if (selected->decoder.prototype_width.has_value()) {
      manifest.segmentation.prototype_width = selected->decoder.prototype_width;
      put(manifest, "segmentation.prototype_width", Certainty::Observed,
          "declared spatial width of the prototype output");
    }
    manifest.segmentation.mask_coefficient_activation = selected->decoder.mask_coefficient_activation;
    put(manifest, "segmentation.mask_coefficient_activation", Certainty::Inferred, because);
    manifest.segmentation.prototype_activation = selected->decoder.prototype_activation;
    put(manifest, "segmentation.prototype_activation", Certainty::Inferred, because);
    if (selected->decoder.has_objectness.has_value()) {
      manifest.segmentation.has_objectness = selected->decoder.has_objectness;
      put(manifest, "segmentation.has_objectness", Certainty::Inferred, because);
    }
    if (const std::string* names = metadata_value(report, "names");
        names != nullptr && manifest.num_classes.has_value()) {
      const std::vector<std::string> parsed = quoted_class_names(*names);
      if (static_cast<int>(parsed.size()) == *manifest.num_classes) {
        manifest.segmentation.class_names = parsed;
        put(manifest, "segmentation.class_names", Certainty::Metadata,
            "ONNX metadata names count matches the class channels");
      } else {
        put(manifest, "segmentation.class_names", Certainty::Unresolved,
            "ONNX metadata names count " + std::to_string(parsed.size()) +
                " does not match graph class count " + std::to_string(*manifest.num_classes));
      }
    }
  }
  if (selected != nullptr && selected->architecture == "rtdetr_detection") {
    const std::string because = joined_evidence(*selected);
    manifest.model_family = selected->family;
    manifest.task = TaskType::Detection;
    put(manifest, "model_family", Certainty::Inferred, because);
    put(manifest, "task", Certainty::Inferred, because);
    if (selected->decoder.num_classes.has_value()) {
      manifest.num_classes = selected->decoder.num_classes;
      put(manifest, "num_classes", Certainty::Inferred, because);
    }
    if (manifest.input.shape.size() == 4 && manifest.input.shape[1] == 3) {
      manifest.input.layout = Layout::Nchw;
      put(manifest, "input.layout", Certainty::Inferred,
          "declared input axis 1 is static 3 and the detection head consumes that tensor");
    }
    manifest.input.resize = "stretch";
    put(manifest, "input.resize", Certainty::Inferred,
        "this Ultralytics RT-DETR export stretches the image to the static square input and does not letterbox");
    manifest.decoder.protocol = selected->decoder.protocol;
    put(manifest, "decoder.protocol", Certainty::Inferred, because);
    if (!selected->decoder.box_format.empty()) {
      manifest.decoder.box_format = selected->decoder.box_format;
      put(manifest, "decoder.box_format", Certainty::Inferred, because);
    }
    manifest.decoder.box_decode = "decoded_normalized_cxcywh";
    put(manifest, "decoder.box_decode", Certainty::Inferred, because);
    manifest.decoder.scores_are_probabilities = selected->decoder.class_scores_are_probabilities;
    put(manifest, "decoder.scores_are_probabilities", Certainty::Inferred, because);
    if (selected->decoder.box_channels > 0) {
      manifest.decoder.box_channels = selected->decoder.box_channels;
      put(manifest, "decoder.box_channels", Certainty::Inferred, because);
    }
    if (selected->decoder.channel_axis >= 0) {
      manifest.decoder.channel_axis = selected->decoder.channel_axis;
      put(manifest, "decoder.channel_axis", Certainty::Inferred, because);
    }
    if (selected->decoder.num_queries.has_value()) {
      manifest.decoder.num_queries = selected->decoder.num_queries;
      put(manifest, "decoder.num_queries", Certainty::Inferred, because);
    }
    manifest.nms.enabled = false;
    put(manifest, "nms.enabled", Certainty::Inferred,
        "the graph has no NonMaxSuppression; the in-graph TopK selects queries and is not pre_nms_top_k");
    if (!manifest.outputs.empty() && manifest.outputs.front().name == selected->decoder.output_name) {
      manifest.outputs.front().semantic = "predictions";
      put(manifest, "outputs.0.semantic", Certainty::Inferred, because);
    }
  }
  if (selected != nullptr && selected->architecture == "yolov8_obb") {
    const std::string because = joined_evidence(*selected);
    manifest.model_family = selected->family;
    manifest.task = TaskType::Obb;
    put(manifest, "model_family", Certainty::Inferred, because);
    put(manifest, "task", Certainty::Inferred, because);
    if (selected->decoder.num_classes.has_value()) {
      manifest.num_classes = selected->decoder.num_classes;
      put(manifest, "num_classes", Certainty::Inferred, because);
    }
    if (manifest.input.shape.size() == 4 && manifest.input.shape[1] == 3) {
      manifest.input.layout = Layout::Nchw;
      put(manifest, "input.layout", Certainty::Inferred, "declared input axis 1 is static 3");
    }
    manifest.decoder.protocol = selected->decoder.protocol;
    put(manifest, "decoder.protocol", Certainty::Inferred, because);
    manifest.decoder.box_format = "cxcywh_angle";
    put(manifest, "decoder.box_format", Certainty::Inferred, because);
    manifest.decoder.box_channels = 4;
    put(manifest, "decoder.box_channels", Certainty::Inferred, because);
    manifest.decoder.scores_are_probabilities = true;
    put(manifest, "decoder.scores_are_probabilities", Certainty::Inferred, because);
    if (selected->decoder.channel_axis >= 0) {
      manifest.decoder.channel_axis = selected->decoder.channel_axis;
      put(manifest, "decoder.channel_axis", Certainty::Inferred, because);
    }
    manifest.obb.output_name = selected->decoder.output_name;
    put(manifest, "obb.output_name", Certainty::Observed, "declared output name");
    manifest.obb.channel_axis = selected->decoder.channel_axis;
    put(manifest, "obb.channel_axis", Certainty::Inferred, because);
    manifest.obb.angle_unit = "radian";
    put(manifest, "obb.angle_unit", Certainty::Inferred,
        "the angle branch is (sigmoid - 1/4) * pi, so the unit is radians");
    manifest.obb.angle_range = "[-pi/4,3pi/4]";
    put(manifest, "obb.angle_range", Certainty::Inferred,
        "sigmoid maps to (0, 1), and (sigmoid - 1/4) * pi maps to [-pi/4, 3pi/4]");
    manifest.obb.score_type = "probabilities";
    put(manifest, "obb.score_type", Certainty::Inferred, "the class branch is a Sigmoid");
    for (std::size_t index = 0; index < manifest.outputs.size(); ++index) {
      if (manifest.outputs[index].name == selected->decoder.output_name) {
        manifest.outputs[index].semantic = "oriented_boxes";
        put(manifest, "outputs." + std::to_string(index) + ".semantic", Certainty::Inferred, because);
      }
    }
  }
  if (selected != nullptr && selected->architecture == "yolov8_pose") {
    const std::string because = joined_evidence(*selected);
    manifest.model_family = selected->family;
    manifest.task = TaskType::Pose;
    put(manifest, "model_family", Certainty::Inferred, because);
    put(manifest, "task", Certainty::Inferred, because);
    if (selected->decoder.num_classes.has_value()) {
      manifest.num_classes = selected->decoder.num_classes;
      put(manifest, "num_classes", Certainty::Inferred, because);
    }
    if (manifest.input.shape.size() == 4 && manifest.input.shape[1] == 3) {
      manifest.input.layout = Layout::Nchw;
      put(manifest, "input.layout", Certainty::Inferred, "declared input axis 1 is static 3");
    }
    manifest.decoder.protocol = selected->decoder.protocol;
    put(manifest, "decoder.protocol", Certainty::Inferred, because);
    manifest.decoder.box_format = "cxcywh";
    put(manifest, "decoder.box_format", Certainty::Inferred, because);
    manifest.decoder.box_decode = "decoded_cxcywh";
    put(manifest, "decoder.box_decode", Certainty::Inferred,
        "the box branch is a stride Mul of Div and Sub, so the center and size are already model-input pixels");
    manifest.decoder.box_channels = 4;
    put(manifest, "decoder.box_channels", Certainty::Inferred, because);
    manifest.decoder.scores_are_probabilities = true;
    put(manifest, "decoder.scores_are_probabilities", Certainty::Inferred, because);
    if (selected->decoder.channel_axis >= 0) {
      manifest.decoder.channel_axis = selected->decoder.channel_axis;
      put(manifest, "decoder.channel_axis", Certainty::Inferred, because);
    }
    if (selected->decoder.reg_max.has_value()) {
      manifest.decoder.reg_max = selected->decoder.reg_max;
      put(manifest, "decoder.reg_max", Certainty::Inferred, because);
    }
    manifest.pose.output_name = selected->decoder.output_name;
    put(manifest, "pose.output_name", Certainty::Observed, "declared output name");
    if (selected->decoder.channel_axis >= 0) {
      manifest.pose.channel_axis = selected->decoder.channel_axis;
      put(manifest, "pose.channel_axis", Certainty::Inferred, because);
    }
    if (selected->decoder.keypoint_count.has_value()) {
      manifest.pose.keypoint_count = selected->decoder.keypoint_count;
      put(manifest, "pose.keypoint_count", Certainty::Inferred, because);
    }
    if (selected->decoder.keypoint_dimensions.has_value()) {
      manifest.pose.keypoint_dimensions = selected->decoder.keypoint_dimensions;
      put(manifest, "pose.keypoint_dimensions", Certainty::Inferred,
          "the keypoint branch sigmoids one sliced field and the conv width is divisible by 3");
    }
    if (!selected->decoder.keypoint_layout.empty()) {
      manifest.pose.keypoint_layout = selected->decoder.keypoint_layout;
      put(manifest, "pose.keypoint_layout", Certainty::Inferred,
          "each keypoint group is x, y, and the sigmoid of the third channel; that value is not a COCO visibility label");
    }
    manifest.pose.score_type = "probabilities";
    put(manifest, "pose.score_type", Certainty::Inferred, "the class branch is a Sigmoid");
    for (std::size_t index = 0; index < manifest.outputs.size(); ++index) {
      if (manifest.outputs[index].name == selected->decoder.output_name) {
        manifest.outputs[index].semantic = "poses";
        put(manifest, "outputs." + std::to_string(index) + ".semantic", Certainty::Inferred, because);
      }
    }
  }
  if (selected != nullptr && selected->architecture == "single_label_classification") {
    const std::string because = joined_evidence(*selected);
    if (!selected->family.empty()) {
      manifest.model_family = selected->family;
      put(manifest, "model_family", Certainty::Inferred, because);
    }
    manifest.task = TaskType::Classification;
    put(manifest, "task", Certainty::Inferred, because);
    if (selected->decoder.num_classes.has_value()) {
      manifest.num_classes = selected->decoder.num_classes;
      put(manifest, "num_classes", Certainty::Inferred, because);
    }
    if (manifest.input.shape.size() == 4 && manifest.input.shape[1] == 3) {
      manifest.input.layout = Layout::Nchw;
      put(manifest, "input.layout", Certainty::Inferred,
          "declared input axis 1 is static 3");
    }
    manifest.decoder.protocol = selected->decoder.protocol;
    put(manifest, "decoder.protocol", Certainty::Inferred, because);
    if (!selected->decoder.output_name.empty()) {
      manifest.classification.output_name = selected->decoder.output_name;
      put(manifest, "classification.output_name", Certainty::Observed, "declared output name");
    }
    if (selected->decoder.class_axis.has_value()) {
      manifest.classification.class_axis = selected->decoder.class_axis;
      put(manifest, "classification.class_axis", Certainty::Inferred, because);
    }
    if (!selected->decoder.score_type.empty()) {
      manifest.classification.score_type = selected->decoder.score_type;
      put(manifest, "classification.score_type", Certainty::Inferred, because);
    }
    for (std::size_t index = 0; index < manifest.outputs.size(); ++index) {
      if (manifest.outputs[index].name == selected->decoder.output_name) {
        manifest.outputs[index].semantic =
            selected->decoder.score_type == "probabilities" ? "probabilities" : "logits";
        put(manifest, "outputs." + std::to_string(index) + ".semantic", Certainty::Inferred, because);
      }
    }
  }
  if (selected != nullptr && selected->architecture == "deeplabv3_semantic_segmentation") {
    const std::string because = joined_evidence(*selected);
    if (!selected->family.empty()) {
      manifest.model_family = selected->family;
      put(manifest, "model_family", Certainty::Inferred, because);
    }
    manifest.task = TaskType::SemanticSegmentation;
    put(manifest, "task", Certainty::Inferred, because);
    if (selected->decoder.num_classes.has_value()) {
      manifest.num_classes = selected->decoder.num_classes;
      put(manifest, "num_classes", Certainty::Inferred, because);
    }
    if (manifest.input.shape.size() == 4 && manifest.input.shape[1] == 3) {
      manifest.input.layout = Layout::Nchw;
      put(manifest, "input.layout", Certainty::Inferred, "declared input axis 1 is static 3");
    }
    manifest.decoder.protocol = selected->decoder.protocol;
    put(manifest, "decoder.protocol", Certainty::Inferred, because);
    if (!selected->decoder.output_name.empty()) {
      manifest.semantic_segmentation.output_name = selected->decoder.output_name;
      put(manifest, "semantic_segmentation.output_name", Certainty::Observed, "declared output name");
    }
    if (selected->decoder.channel_axis >= 0) {
      manifest.semantic_segmentation.channel_axis = selected->decoder.channel_axis;
      put(manifest, "semantic_segmentation.channel_axis", Certainty::Inferred, because);
    }
    manifest.semantic_segmentation.logits_semantics = "per_pixel_class_logits";
    put(manifest, "semantic_segmentation.logits_semantics", Certainty::Inferred,
        "the head emits raw per-pixel class logits and runs no Softmax");
    manifest.semantic_segmentation.output_interpolation = "bilinear";
    put(manifest, "semantic_segmentation.output_interpolation", Certainty::Inferred,
        "continuous logits are resampled to the model-input grid with bilinear interpolation");
    manifest.semantic_segmentation.class_id_interpolation = "nearest";
    put(manifest, "semantic_segmentation.class_id_interpolation", Certainty::Inferred,
        "integer class ids are only ever resampled with nearest, never bilinear");
    for (std::size_t index = 0; index < manifest.outputs.size(); ++index) {
      if (manifest.outputs[index].name == selected->decoder.output_name) {
        manifest.outputs[index].semantic = "semantic_logits";
        put(manifest, "outputs." + std::to_string(index) + ".semantic", Certainty::Inferred, because);
      }
    }
  }

  manifest.ensure_provenance();
  return manifest;
}

}  // namespace vie
