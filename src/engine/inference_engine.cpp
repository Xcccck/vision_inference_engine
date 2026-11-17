#include "vie/engine/inference_engine.hpp"

#include "vie/backend/mock_backend.hpp"
#include "vie/decoder/detection.hpp"
#include "vie/decoder/mock_decoder.hpp"
#include "vie/decoder/rtdetr_cuda_decoder.hpp"
#include "vie/decoder/yolov8_detect_decoder.hpp"
#include "vie/decoder/yolov8_seg_decoder.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <chrono>

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/backend/onnxruntime_backend.hpp"
#endif
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
#include "vie/backend/onnxruntime_cuda_backend.hpp"
#endif

namespace vie {
namespace {

// The only place that pairs a model format and a device with a backend.
BackendKind resolve_backend(BackendKind requested, ModelFormat format, DeviceType device,
                            bool onnx_cpu, bool onnx_cuda) {
  const auto require_onnx = [&] {
    if (format != ModelFormat::Onnx) {
      throw VieError(ErrorCode::Unsupported,
                     "the ONNX Runtime backend only executes model format 'onnx'");
    }
  };
  if (requested == BackendKind::OnnxRuntimeCuda ||
      (requested == BackendKind::OnnxRuntime && device == DeviceType::Cuda)) {
    require_onnx();
    if (device != DeviceType::Cuda) {
      throw VieError(ErrorCode::Unsupported,
                     "the ONNX Runtime CUDA backend only runs on CUDA");
    }
    if (!onnx_cuda) {
      throw VieError(ErrorCode::Unsupported,
                     "CUDA was requested, and this build has no CUDA backend");
    }
    return BackendKind::OnnxRuntimeCuda;
  }
  if (requested == BackendKind::OnnxRuntime) {
    require_onnx();
    if (!onnx_cpu) {
      throw VieError(ErrorCode::Unsupported,
                     "ONNX Runtime was requested, and this build has no ONNX Runtime backend");
    }
    if (device != DeviceType::Cpu) {
      throw VieError(ErrorCode::Unsupported, "the ONNX Runtime CPU backend only runs on CPU");
    }
    return BackendKind::OnnxRuntime;
  }
  if (requested == BackendKind::TensorRT) {
    throw VieError(ErrorCode::Unsupported,
                   "TensorRT was requested, and this build has no TensorRT backend");
  }
  if (requested == BackendKind::Auto) {
    if (format == ModelFormat::Mock) {
      if (device == DeviceType::Cuda) {
        throw VieError(ErrorCode::Unsupported, "the mock backend does not run on CUDA");
      }
      return BackendKind::Mock;
    }
    if (format == ModelFormat::Onnx && device == DeviceType::Cuda) {
      if (!onnx_cuda) {
        throw VieError(ErrorCode::Unsupported,
                       "CUDA was requested, and this build has no CUDA backend");
      }
      return BackendKind::OnnxRuntimeCuda;
    }
    if (format == ModelFormat::Onnx && device == DeviceType::Cpu && onnx_cpu) {
      return BackendKind::OnnxRuntime;
    }
    throw VieError(ErrorCode::Unsupported,
                   std::string("no backend in this build can execute model format ") +
                       to_string(format));
  }
  if (requested == BackendKind::Mock && format != ModelFormat::Mock) {
    throw VieError(ErrorCode::Unsupported, "the mock backend cannot execute this model format");
  }
  if (requested == BackendKind::Mock && device == DeviceType::Cuda) {
    throw VieError(ErrorCode::Unsupported, "the mock backend does not run on CUDA");
  }
  return requested;
}

void require_model_field(bool ready, const std::string& message) {
  if (!ready) throw VieError(ErrorCode::UnresolvedField, message);
}

std::optional<std::int64_t> concrete_dim(const std::vector<std::int64_t>& shape, std::size_t axis) {
  if (axis >= shape.size() || shape[axis] < 1) return std::nullopt;
  return shape[axis];
}

const OutputDesc* output_named(const ModelManifest& manifest, const std::string& name) {
  for (const OutputDesc& output : manifest.outputs) {
    if (output.name == name) return &output;
  }
  return nullptr;
}

DecoderConfig segmentation_model_config(const ModelManifest& manifest) {
  require_model_field(manifest.decoder.box_format == "cxcywh",
                      "decoder.box_format is unresolved or is not cxcywh");
  require_model_field(manifest.decoder.box_decode == "decoded_cxcywh",
                      "decoder.box_decode is unresolved or is not decoded_cxcywh");
  require_model_field(manifest.decoder.box_channels.has_value() && *manifest.decoder.box_channels == 4,
                      "decoder.box_channels is unresolved or is not 4");
  require_model_field(manifest.decoder.channel_axis.has_value(), "decoder.channel_axis is unresolved");
  require_model_field(manifest.num_classes.has_value() && *manifest.num_classes >= 1,
                      "num_classes is unresolved");
  require_model_field(manifest.decoder.scores_are_probabilities.has_value(),
                      "decoder.scores_are_probabilities is unresolved");
  if (!*manifest.decoder.scores_are_probabilities) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_seg_raw class scores are already probabilities; refusing a second sigmoid");
  }
  require_model_field(manifest.segmentation.mask_channels.has_value() &&
                          *manifest.segmentation.mask_channels >= 1,
                      "segmentation.mask_channels is unresolved");
  require_model_field(!manifest.segmentation.prototype_name.empty(),
                      "segmentation.prototype_name is unresolved");
  require_model_field(manifest.segmentation.prototype_channel_axis.has_value(),
                      "segmentation.prototype_channel_axis is unresolved");
  if (*manifest.segmentation.prototype_channel_axis != 1) {
    throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw only reads an NCHW prototype tensor");
  }
  require_model_field(manifest.segmentation.prototype_layout == Layout::Nchw,
                      "segmentation.prototype_layout is unresolved or is not nchw");
  require_model_field(manifest.segmentation.prototype_height.has_value() &&
                          *manifest.segmentation.prototype_height >= 1,
                      "segmentation.prototype_height is unresolved");
  require_model_field(manifest.segmentation.prototype_width.has_value() &&
                          *manifest.segmentation.prototype_width >= 1,
                      "segmentation.prototype_width is unresolved");
  require_model_field(manifest.segmentation.mask_coefficient_activation == "none",
                      "segmentation.mask_coefficient_activation is unresolved or is not none");
  require_model_field(manifest.segmentation.prototype_activation == "model_output_features",
                      "segmentation.prototype_activation is unresolved or is not model_output_features");
  require_model_field(manifest.segmentation.has_objectness.has_value(),
                      "segmentation.has_objectness is unresolved");
  if (*manifest.segmentation.has_objectness) {
    throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw has no objectness score");
  }

  const OutputDesc* prototype = output_named(manifest, manifest.segmentation.prototype_name);
  require_model_field(prototype != nullptr, "segmentation.prototype_name does not match an output");
  const OutputDesc* prediction = nullptr;
  for (const OutputDesc& output : manifest.outputs) {
    if (&output == prototype) continue;
    if (prediction != nullptr) {
      throw VieError(ErrorCode::InvalidArgument,
                     "yolov8_seg_raw requires one prediction output besides the prototype");
    }
    prediction = &output;
  }
  require_model_field(prediction != nullptr && !prediction->name.empty(),
                      "prediction output name is unresolved");
  if (!prediction->semantic.empty() &&
      prediction->semantic != "cxcywh_class_probabilities_mask_coefficients") {
    throw VieError(ErrorCode::InvalidArgument, "prediction output semantic conflicts with yolov8_seg_raw");
  }
  if (!prototype->semantic.empty() && prototype->semantic != "mask_prototypes") {
    throw VieError(ErrorCode::InvalidArgument, "prototype output semantic conflicts with yolov8_seg_raw");
  }

  const int classes = *manifest.num_classes;
  const int mask_channels = *manifest.segmentation.mask_channels;
  const int channel_axis = *manifest.decoder.channel_axis;
  if (const std::optional<std::int64_t> channels = concrete_dim(prediction->shape, static_cast<std::size_t>(channel_axis))) {
    if (*channels != static_cast<std::int64_t>(4 + classes + mask_channels)) {
      throw VieError(ErrorCode::InvalidArgument,
                     "prediction shape conflicts with 4 + num_classes + mask_channels");
    }
  }
  if (const std::optional<std::int64_t> proto_channels = concrete_dim(prototype->shape, 1)) {
    if (*proto_channels != mask_channels) {
      throw VieError(ErrorCode::InvalidArgument, "prototype shape conflicts with segmentation.mask_channels");
    }
  }
  if (const std::optional<std::int64_t> height = concrete_dim(prototype->shape, 2)) {
    if (*height != *manifest.segmentation.prototype_height) {
      throw VieError(ErrorCode::InvalidArgument, "prototype shape conflicts with segmentation.prototype_height");
    }
  }
  if (const std::optional<std::int64_t> width = concrete_dim(prototype->shape, 3)) {
    if (*width != *manifest.segmentation.prototype_width) {
      throw VieError(ErrorCode::InvalidArgument, "prototype shape conflicts with segmentation.prototype_width");
    }
  }

  DecoderConfig config;
  config.output_name = prediction->name;
  config.prototype_name = prototype->name;
  config.channel_axis = channel_axis;
  config.box_channels = 4;
  config.num_classes = classes;
  config.mask_channels = mask_channels;
  config.box_decode = BoxDecode::DecodedCxcywh;
  config.scores_are_probabilities = true;
  config.prototype_channel_axis = 1;
  if (manifest.decoder.reg_max.has_value()) config.reg_max = *manifest.decoder.reg_max;
  return config;
}

void apply_segmentation_runtime_config(DecoderConfig& config, const SegmentationRuntimeConfig& runtime) {
  config.class_selection = runtime.class_selection;
  config.confidence_threshold = runtime.confidence_threshold;
  config.iou_threshold = runtime.iou_threshold;
  config.class_aware_nms = runtime.class_aware_nms;
  config.pre_nms_top_k = runtime.pre_nms_top_k;
  config.max_detections = runtime.max_detections;
  config.nms_enabled = runtime.nms_enabled;
  config.retina_masks = runtime.retina_masks;
  config.binary_mask_gt_zero = runtime.binary_mask_gt_zero;
  config.max_mask_bytes = runtime.max_mask_bytes;
  validate_decoder_config(config);
  require_model_field(!config.prototype_name.empty(), "decoder.prototype_name is unresolved");
  require_model_field(config.mask_channels >= 1, "decoder.mask_channels is unresolved");
  require_model_field(config.prototype_channel_axis.has_value(),
                      "decoder.prototype_channel_axis is unresolved");
  require_model_field(config.retina_masks.has_value(), "decoder.retina_masks is unresolved");
  require_model_field(config.binary_mask_gt_zero.has_value(), "decoder.binary_mask_gt_zero is unresolved");
  require_model_field(config.max_mask_bytes.has_value(), "decoder.max_mask_bytes is unresolved");
  if (*config.prototype_channel_axis != 1) {
    throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw only reads an NCHW prototype tensor");
  }
  if (!*config.binary_mask_gt_zero) {
    throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw only emits a binary mask from the M > 0 test");
  }
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void require_same_device(const Tensor& tensor, const DeviceContext& device, const std::string& what) {
  if (tensor.device() != device.device() || tensor.device_id() != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument, what + " device does not match the engine device");
  }
}

}  // namespace

InferenceEngine::InferenceEngine() {
  backends_.register_factory(BackendKind::Mock, [] { return std::make_unique<MockBackend>(); });
  decoders_.register_factory("mock", [] { return std::make_unique<MockDecoder>(); });
  decoders_.register_factory("yolov8_detect_raw", [] { return std::make_unique<YoloV8DetectionDecoder>(); });
  decoders_.register_factory("yolov8_seg_raw", [] { return std::make_unique<YoloV8SegmentationDecoder>(); });
  decoders_.register_factory("rtdetr_detection_raw", [] { return std::make_unique<RtdetrDetectionDecoder>(); });
#if defined(VIE_HAS_ONNXRUNTIME)
  backends_.register_factory(BackendKind::OnnxRuntime,
                             [] { return std::make_unique<OnnxRuntimeBackend>(); });
#endif
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
  backends_.register_factory(BackendKind::OnnxRuntimeCuda,
                             [] { return std::make_unique<OnnxRuntimeCudaBackend>(); });
#endif
}

void InferenceEngine::load(const ModelConfig& config) {
  if (config.device_id < 0) {
    throw VieError(ErrorCode::InvalidArgument, "device id must be >= 0");
  }
  if (!config.manifest.has_value()) {
    throw VieError(ErrorCode::Unsupported,
                   "no manifest was provided. Model analysis is not implemented in this build");
  }

  ModelManifest manifest = *config.manifest;
  if (!config.model_path.empty()) {
    manifest.model_path = config.model_path;
    manifest.provenance["model_path"] =
        FieldProvenance{Certainty::User, {"ModelConfig.model_path"}};
  }
  if (config.task != TaskType::Unknown) {
    manifest.task = config.task;
    manifest.provenance["task"] = FieldProvenance{Certainty::User, {"ModelConfig.task"}};
  }
  manifest.ensure_provenance();

  const ManifestValidation report = ManifestValidator{}.validate(manifest);
  if (!report.schema_ok()) {
    throw VieError(ErrorCode::SchemaError,
                   "manifest schema is invalid: " + format_issues(report.schema_issues));
  }
  if (!report.runnable()) {
    throw VieError(ErrorCode::UnresolvedField, "critical fields are unresolved: " +
                                                   format_issues(report.unresolved_critical));
  }

  const BackendKind kind = resolve_backend(config.backend, manifest.model_format, config.device,
                                          backends_.contains(BackendKind::OnnxRuntime),
                                          backends_.contains(BackendKind::OnnxRuntimeCuda));
  if (!decoders_.contains(manifest.decoder.protocol)) {
    throw VieError(ErrorCode::Unsupported, "no decoder is registered for protocol '" +
                                               manifest.decoder.protocol + "'");
  }

  BackendConfig backend_config;
  backend_config.model_path = manifest.model_path;
  backend_config.model_format = manifest.model_format;
  backend_config.device = config.device;
  backend_config.device_id = config.device_id;
  backend_config.input.name = manifest.input.name;
  backend_config.input.shape = manifest.input.shape;
  backend_config.input.dtype = manifest.input.dtype;
  backend_config.input.layout = manifest.input.layout;
  for (const OutputDesc& output : manifest.outputs) {
    TensorSpec spec;
    spec.name = output.name;
    spec.shape = output.shape;
    spec.dtype = output.dtype;
    spec.layout = output.layout;
    backend_config.outputs.push_back(std::move(spec));
  }

  std::unique_ptr<IInferenceBackend> backend = backends_.create(kind);
  backend->prepare(backend_config);
  std::unique_ptr<IModelDecoder> decoder = decoders_.create(manifest.decoder.protocol);

  device_ = DeviceContext(config.device, config.device_id);
  manifest_ = std::move(manifest);
  backend_ = std::move(backend);
  decoder_ = std::move(decoder);
  loaded_ = true;
}

const ModelManifest& InferenceEngine::manifest() const {
  if (!loaded_) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  return manifest_;
}

BackendOutput InferenceEngine::execute(const BackendRequest& request) {
  if (!loaded_ || backend_ == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  return backend_->execute(request);
}

PreprocessResult InferenceEngine::preprocess(const ImageInput& image) const {
  if (!loaded_) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  ImagePreprocessor preprocessor;
  preprocessor.configure(
      preprocess_config_from_manifest(manifest_, image.color, image.layout));
  return preprocessor.execute(image, device_);
}

DetectionBatch InferenceEngine::infer(const ImageInput& image, DetectionInferTiming* timing) {
  if (!loaded_) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  if (manifest_.task == TaskType::Segmentation) {
    throw VieError(ErrorCode::Unsupported,
                   "task segmentation does not produce DetectionBatch; call infer_segmentation()");
  }
  const auto total_start = std::chrono::steady_clock::now();
  const PreprocessResult prepared = preprocess(image);
  const double preprocess_ms = elapsed_ms(total_start);
  require_same_device(prepared.output, device_, "preprocess output");
  BackendRequest request;
  request.inputs.emplace(manifest_.input.name, prepared.output);
  const auto execute_start = std::chrono::steady_clock::now();
  const BackendOutput output = execute(request);
  const double execute_ms = elapsed_ms(execute_start);
  DecoderInput decoded;
  for (const auto& entry : output.tensors) {
    require_same_device(entry.second.tensor, device_, "backend output '" + entry.first + "'");
    decoded.outputs.emplace(entry.first, entry.second.tensor);
  }
  decoded.geometry = prepared.geometry;
  const auto decode_start = std::chrono::steady_clock::now();
  DetectionBatch batch =
      decoder_->decode_batch(decoded, decoder_config_from_manifest(manifest_), device_);
  if (timing != nullptr) {
    timing->preprocess_ms = preprocess_ms;
    timing->execute_ms = execute_ms;
    timing->decode_ms = elapsed_ms(decode_start);
    timing->total_ms = elapsed_ms(total_start);
    timing->includes_sync = device_.device() == DeviceType::Cuda;
    timing->includes_h2d = false;
    timing->includes_d2h = device_.device() == DeviceType::Cuda;
  }
  return batch;
}

SegmentationBatch InferenceEngine::infer_segmentation(const ImageInput& image,
                                                     const SegmentationRuntimeConfig& runtime,
                                                     SegmentationInferTiming* timing) {
  if (!loaded_ || backend_ == nullptr || decoder_ == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  if (manifest_.task != TaskType::Segmentation) {
    throw VieError(ErrorCode::Unsupported, "infer_segmentation() requires task segmentation");
  }
  if (manifest_.decoder.protocol != "yolov8_seg_raw") {
    throw VieError(ErrorCode::Unsupported,
                   "infer_segmentation() requires decoder protocol yolov8_seg_raw");
  }
  if (image.data == nullptr || image.width <= 0 || image.height <= 0) {
    throw VieError(ErrorCode::InvalidArgument, "image data is null or the image size is invalid");
  }
  DecoderConfig config = segmentation_model_config(manifest_);
  apply_segmentation_runtime_config(config, runtime);

  const auto total_start = std::chrono::steady_clock::now();
  const auto preprocess_start = total_start;
  const PreprocessResult prepared = preprocess(image);
  const double preprocess_ms = elapsed_ms(preprocess_start);
  require_same_device(prepared.output, device_, "preprocess output");

  BackendRequest request;
  request.inputs.emplace(manifest_.input.name, prepared.output);
  const auto execute_start = std::chrono::steady_clock::now();
  const BackendOutput output = execute(request);
  const double execute_ms = elapsed_ms(execute_start);

  DecoderInput decoded;
  decoded.geometry = prepared.geometry;
  for (const auto& entry : output.tensors) {
    require_same_device(entry.second.tensor, device_, "backend output '" + entry.first + "'");
    decoded.outputs.emplace(entry.first, entry.second.tensor);
  }
  if (decoded.outputs.find(config.output_name) == decoded.outputs.end() ||
      decoded.outputs.find(config.prototype_name) == decoded.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "backend outputs are missing the prediction or prototype tensor");
  }

  const auto decode_start = std::chrono::steady_clock::now();
  SegmentationBatch batch = decoder_->decode_segmentation(decoded, config, device_);
  const double decode_ms = elapsed_ms(decode_start);
  for (const InstanceSegmentation& instance : batch.instances) {
    require_same_device(instance.mask.tensor(), device_, "segmentation mask");
  }
  if (timing != nullptr) {
    timing->preprocess_ms = preprocess_ms;
    timing->execute_ms = execute_ms;
    timing->decode_ms = decode_ms;
    timing->total_ms = elapsed_ms(total_start);
    timing->includes_sync = device_.device() == DeviceType::Cuda;
    timing->includes_h2d = false;
    timing->includes_mask_d2h = false;
  }
  return batch;
}

RawPredictions InferenceEngine::decode(const BackendOutput& output) const {
  if (!loaded_ || decoder_ == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  std::map<std::string, Tensor> tensors;
  for (const auto& entry : output.tensors) {
    tensors.emplace(entry.first, entry.second.tensor);
  }
  DecoderContext context;
  context.protocol = manifest_.decoder.protocol;
  context.box_format = manifest_.decoder.box_format;
  context.num_classes = manifest_.num_classes;
  for (const OutputDesc& spec : manifest_.outputs) {
    TensorSpec tensor_spec;
    tensor_spec.name = spec.name;
    tensor_spec.shape = spec.shape;
    tensor_spec.dtype = spec.dtype;
    tensor_spec.layout = spec.layout;
    context.outputs.push_back(std::move(tensor_spec));
  }
  return decoder_->decode(tensors, context);
}

RawPredictions InferenceEngine::run(const BackendRequest& request) {
  return decode(execute(request));
}

void InferenceEngine::synchronize() const { device_.synchronize(); }

}  // namespace vie
