#include "vie/engine/inference_engine.hpp"

#include "vie/backend/mock_backend.hpp"
#include "vie/decoder/classification.hpp"
#include "vie/decoder/classification_cuda_decoder.hpp"
#include "vie/decoder/detection.hpp"
#include "vie/decoder/mock_decoder.hpp"
#include "vie/decoder/obb_cuda_decoder.hpp"
#include "vie/decoder/pose_cuda_decoder.hpp"
#include "vie/decoder/rtdetr_cuda_decoder.hpp"
#include "vie/decoder/semantic_segmentation_cuda_decoder.hpp"
#include "vie/decoder/yolov8_detect_decoder.hpp"
#include "vie/decoder/yolov8_obb_decoder.hpp"
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
  decoders_.register_factory("yolov8_obb_raw", [] { return std::make_unique<ObbDispatchDecoder>(); });
  decoders_.register_factory("yolov8_pose_raw", [] { return std::make_unique<PoseDispatchDecoder>(); });
  decoders_.register_factory("yolov8_seg_raw", [] { return std::make_unique<YoloV8SegmentationDecoder>(); });
  decoders_.register_factory("rtdetr_detection_raw", [] { return std::make_unique<RtdetrDetectionDecoder>(); });
  decoders_.register_factory("classification_logits_raw",
                             [] { return std::make_unique<ClassificationDispatchDecoder>(false); });
  decoders_.register_factory("classification_probabilities_raw",
                             [] { return std::make_unique<ClassificationDispatchDecoder>(true); });
  decoders_.register_factory("deeplabv3_semantic_seg_raw",
                             [] { return std::make_unique<SemanticSegmentationDispatchDecoder>(); });
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
    // Phase 12: the NMS thresholds are policy parameters with framework
    // defaults. A manifest that enables NMS without pinning them may still
    // load; resolve_inference_config() supplies 0.25 / 0.45. Every other
    // critical field keeps its strict check.
    std::vector<SchemaIssue> blocking;
    for (const SchemaIssue& issue : report.unresolved_critical) {
      if (issue.path == "nms.confidence_threshold" || issue.path == "nms.iou_threshold") continue;
      blocking.push_back(issue);
    }
    if (!blocking.empty()) {
      throw VieError(ErrorCode::UnresolvedField,
                     "critical fields are unresolved: " + format_issues(blocking));
    }
  }
  // Reject a model-level option block for another task, an invalid value, or a
  // resolved effective value the decoder cannot honor. This fails at load
  // instead of on the first inference.
  ::vie::resolve_inference_config(manifest.task, manifest, config.inference_options, InferenceOptions{});

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
  
  // Configure CUDA-specific options before prepare if applicable
#if defined(VIE_HAS_ONNXRUNTIME_CUDA)
  if (kind == BackendKind::OnnxRuntimeCuda) {
    auto* cuda_backend = dynamic_cast<OnnxRuntimeCudaBackend*>(backend.get());
    if (cuda_backend != nullptr) {
      if (config.cuda_cpu_ep_fallback) {
        cuda_backend->set_cpu_ep_fallback(true);
      }
      // Phase 9.3.2.1: opt-in profiling on the real inference session. Default
      // off; when off this leaves the backend exactly as before.
      if (config.ort_enable_profiling) {
        cuda_backend->set_profiling(true, config.ort_profile_prefix);
      }
    }
  }
#endif
  
  backend->prepare(backend_config);
  
  std::unique_ptr<IModelDecoder> decoder = decoders_.create(manifest.decoder.protocol);

  device_ = DeviceContext(config.device, config.device_id);
  manifest_ = std::move(manifest);
  model_options_ = config.inference_options;
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

ResolvedInferenceConfig InferenceEngine::resolve_inference_config(const InferenceOptions& request) const {
  if (!loaded_) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  return ::vie::resolve_inference_config(manifest_.task, manifest_, model_options_, request);
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
  return infer_resolved(image, resolve_inference_config(InferenceOptions{}), timing);
}

DetectionBatch InferenceEngine::infer_resolved(const ImageInput& image, const ResolvedInferenceConfig& resolved,
                                               DetectionInferTiming* timing) {
  if (!loaded_) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  if (resolved.task != manifest_.task || resolved.protocol != manifest_.decoder.protocol) {
    throw VieError(ErrorCode::InvalidArgument,
                   "resolved inference config does not belong to this engine");
  }
  if (manifest_.task == TaskType::Segmentation) {
    throw VieError(ErrorCode::Unsupported,
                   "task segmentation does not produce DetectionBatch; call infer_segmentation()");
  }
  if (manifest_.task == TaskType::Classification) {
    throw VieError(ErrorCode::Unsupported,
                   "task classification does not produce DetectionBatch; call infer_classification()");
  }
  if (manifest_.task == TaskType::Obb) {
    throw VieError(ErrorCode::Unsupported, "task obb does not produce DetectionBatch; call infer_obb()");
  }
  if (manifest_.task == TaskType::Pose) {
    throw VieError(ErrorCode::Unsupported, "task pose does not produce DetectionBatch; call infer_pose()");
  }
  if (manifest_.task == TaskType::SemanticSegmentation) {
    throw VieError(ErrorCode::Unsupported,
                   "task semantic_segmentation does not produce DetectionBatch; call infer_semantic_segmentation()");
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
      decoder_->decode_batch(decoded, decoder_config_from_resolved(manifest_, resolved), device_);
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

// Phase 9.1: Batch detection. Runs one preprocess / execute / decode pass
// for the whole batch. Only yolov8_detect_raw on CPU currently supports
// Batch>1; other decoders or CUDA reject a batch larger than one via their
// own decode_batch implementation, so this function does not silently fall
// back to looping infer().
DetectionBatch InferenceEngine::infer_batch(const std::vector<ImageInput>& images,
                                            DetectionInferTiming* timing) {
  return infer_batch_resolved(images, resolve_inference_config(InferenceOptions{}), timing);
}

DetectionBatch InferenceEngine::infer_batch_resolved(const std::vector<ImageInput>& images,
                                                     const ResolvedInferenceConfig& resolved,
                                                     DetectionInferTiming* timing) {
  if (!loaded_) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  if (resolved.task != manifest_.task || resolved.protocol != manifest_.decoder.protocol) {
    throw VieError(ErrorCode::InvalidArgument,
                   "resolved inference config does not belong to this engine");
  }
  if (images.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "infer_batch requires at least one image");
  }
  if (manifest_.task == TaskType::Segmentation) {
    throw VieError(ErrorCode::Unsupported,
                   "task segmentation does not produce DetectionBatch; call infer_segmentation()");
  }
  if (manifest_.task == TaskType::Classification) {
    throw VieError(ErrorCode::Unsupported,
                   "task classification does not produce DetectionBatch; call infer_classification()");
  }
  if (manifest_.task == TaskType::Obb) {
    throw VieError(ErrorCode::Unsupported, "task obb does not produce DetectionBatch; call infer_obb()");
  }
  if (manifest_.task == TaskType::Pose) {
    throw VieError(ErrorCode::Unsupported, "task pose does not produce DetectionBatch; call infer_pose()");
  }
  if (manifest_.task == TaskType::SemanticSegmentation) {
    throw VieError(ErrorCode::Unsupported,
                   "task semantic_segmentation does not produce DetectionBatch; call infer_semantic_segmentation()");
  }
  // Reject mixed devices so the batch Tensor stays on one execution device.
  for (std::size_t i = 1; i < images.size(); ++i) {
    if (images[i].device != images[0].device || images[i].device_id != images[0].device_id) {
      throw VieError(ErrorCode::InvalidArgument,
                     "infer_batch image[" + std::to_string(i) +
                         "] device does not match image[0] device");
    }
  }

  const auto total_start = std::chrono::steady_clock::now();
  ImagePreprocessor preprocessor;
  preprocessor.configure(
      preprocess_config_from_manifest(manifest_, images[0].color, images[0].layout));
  const BatchPreprocessResult prepared = preprocessor.execute_batch(images, device_);
  const double preprocess_ms = elapsed_ms(total_start);
  require_same_device(prepared.output, device_, "batch preprocess output");

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
  decoded.geometries = prepared.geometries;
  if (prepared.geometries.size() == 1) {
    decoded.geometry = prepared.geometries.front();
  }

  const auto decode_start = std::chrono::steady_clock::now();
  DetectionBatch batch =
      decoder_->decode_batch(decoded, decoder_config_from_resolved(manifest_, resolved), device_);
  if (batch.results.size() != images.size()) {
    throw VieError(ErrorCode::Internal,
                   "decoder returned " + std::to_string(batch.results.size()) +
                       " results for a batch of " + std::to_string(images.size()));
  }
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

SegmentationBatch InferenceEngine::infer_segmentation_options(const ImageInput& image,
                                                              const InferenceOptions& options,
                                                              SegmentationInferTiming* timing) {
  const ResolvedInferenceConfig resolved = resolve_inference_config(options);
  if (resolved.task != TaskType::Segmentation) {
    throw VieError(ErrorCode::Unsupported, "infer_segmentation() requires task segmentation");
  }
  const ResolvedSegmentationConfig& policy = resolved.require_segmentation();
  SegmentationRuntimeConfig runtime;
  runtime.confidence_threshold = policy.confidence_threshold.value;
  runtime.iou_threshold = policy.iou_threshold.value;
  // The YOLOv8 seg decoder keeps one class per candidate; segmentation has no
  // multi-label option because the mask carries the class decision.
  runtime.class_selection = ClassSelection::BestClass;
  runtime.class_aware_nms = !policy.class_agnostic_nms.value;
  runtime.pre_nms_top_k = policy.pre_nms_top_k.value;
  runtime.max_detections = policy.max_detections.value;
  runtime.nms_enabled = policy.nms_enabled;
  if (policy.retina_masks.has_value()) runtime.retina_masks = policy.retina_masks->value;
  if (policy.binary_mask_gt_zero.has_value()) {
    runtime.binary_mask_gt_zero = policy.binary_mask_gt_zero->value;
  }
  if (policy.max_mask_bytes.has_value()) runtime.max_mask_bytes = policy.max_mask_bytes->value;
  return infer_segmentation(image, runtime, timing);
}

ClassificationBatch InferenceEngine::infer_classification_options(const ImageInput& image,
                                                                   const InferenceOptions& options,
                                                                   ClassificationInferTiming* timing) {
  const ResolvedInferenceConfig resolved = resolve_inference_config(options);
  if (resolved.task != TaskType::Classification) {
    throw VieError(ErrorCode::Unsupported, "infer_classification() requires task classification");
  }
  ClassificationRuntimeConfig runtime;
  runtime.top_k = resolved.require_classification().top_k.value;
  return infer_classification(image, runtime, timing);
}

ClassificationBatch InferenceEngine::infer_classification(const ImageInput& image,
                                                         const ClassificationRuntimeConfig& runtime,
                                                         ClassificationInferTiming* timing) {
  if (!loaded_ || backend_ == nullptr || decoder_ == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  if (manifest_.task != TaskType::Classification) {
    throw VieError(ErrorCode::Unsupported, "infer_classification() requires task classification");
  }
  if (manifest_.decoder.protocol != "classification_logits_raw" &&
      manifest_.decoder.protocol != "classification_probabilities_raw") {
    throw VieError(ErrorCode::Unsupported,
                   "infer_classification() requires classification_logits_raw or "
                   "classification_probabilities_raw");
  }
  if (!runtime.top_k.has_value()) {
    throw VieError(ErrorCode::UnresolvedField, "classification top_k is unresolved");
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

  const std::string output_name = manifest_.classification.output_name;
  const auto found = output.tensors.find(output_name);
  if (found == output.tensors.end()) {
    throw VieError(ErrorCode::InvalidArgument, "classification output '" + output_name + "' is missing");
  }
  require_same_device(found->second.tensor, device_, "backend output '" + output_name + "'");
  const Tensor& scores = found->second.tensor;
  const std::size_t output_bytes = scores.nbytes();

  ClassificationDecodeRequest decoded;
  decoded.output_name = output_name;
  decoded.class_axis = manifest_.classification.class_axis.value_or(-1);
  decoded.num_classes = manifest_.num_classes.value_or(0);
  decoded.top_k = *runtime.top_k;
  decoded.score_type = manifest_.classification.score_type;
  decoded.class_names = manifest_.classification.class_names;
  DecoderInput decoder_input;
  decoder_input.outputs.emplace(output_name, scores);
  // Phase 13: the tensor stays on the engine device. On CUDA the decoder runs
  // validation, the probability sum, the sort, and the top-k cut in kernels
  // and copies only the final top_k pairs; on CPU it reads the host tensor and
  // copies nothing. The engine records whatever the decoder reports.
  const auto decode_start = std::chrono::steady_clock::now();
  ClassificationBatch batch = decoder_->decode_classification(decoder_input, decoded, device_);
  const double decode_ms = elapsed_ms(decode_start);
  const double validate_ms = batch.results.empty() ? 0.0 : batch.results[0].validate_ms;
  const double topk_ms = batch.results.empty() ? 0.0 : batch.results[0].topk_ms;
  const std::size_t d2h_bytes = batch.results.empty() ? 0 : batch.results[0].d2h_bytes;
  const double d2h_ms = batch.results.empty() ? 0.0 : batch.results[0].d2h_ms;
  if (!batch.results.empty()) {
    batch.results[0].output_bytes = output_bytes;
  }
  if (timing != nullptr) {
    timing->preprocess_ms = preprocess_ms;
    timing->execute_ms = execute_ms;
    timing->d2h_ms = d2h_ms;
    timing->validate_ms = validate_ms;
    timing->topk_ms = topk_ms;
    // CPU keeps the documented sum: validate plus sort is all the decoder does
    // and it copies nothing. CUDA reports the wall clock of the whole decoder
    // call so launch and stream-sync cost is not hidden behind GPU event time.
    timing->decode_ms = device_.device() == DeviceType::Cuda
                            ? decode_ms
                            : d2h_ms + validate_ms + topk_ms;
    timing->total_ms = elapsed_ms(total_start);
    timing->output_bytes = output_bytes;
    timing->d2h_bytes = d2h_bytes;
    timing->includes_sync = device_.device() == DeviceType::Cuda;
    timing->includes_h2d = false;
    timing->includes_d2h = d2h_bytes > 0;
  }
  return batch;
}

SemanticSegmentationResult InferenceEngine::infer_semantic_segmentation(const ImageInput& image,
                                                                        SemanticSegmentationInferTiming* timing) {
  if (!loaded_ || backend_ == nullptr || decoder_ == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  if (manifest_.task != TaskType::SemanticSegmentation) {
    throw VieError(ErrorCode::Unsupported, "infer_semantic_segmentation() requires task semantic_segmentation");
  }
  if (manifest_.decoder.protocol != "deeplabv3_semantic_seg_raw") {
    throw VieError(ErrorCode::Unsupported,
                   "infer_semantic_segmentation() requires decoder protocol deeplabv3_semantic_seg_raw");
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

  const std::string output_name = manifest_.semantic_segmentation.output_name;
  if (output_name.empty()) {
    throw VieError(ErrorCode::UnresolvedField, "semantic_segmentation.output_name is unresolved");
  }
  const auto found = output.tensors.find(output_name);
  if (found == output.tensors.end()) {
    throw VieError(ErrorCode::InvalidArgument, "semantic segmentation output '" + output_name + "' is missing");
  }
  require_same_device(found->second.tensor, device_, "backend output '" + output_name + "'");
  Tensor logits = found->second.tensor;
  const std::size_t output_bytes = logits.nbytes();

  SemanticSegmentationDecodeRequest decoded;
  decoded.output_name = output_name;
  decoded.num_classes = manifest_.num_classes.value_or(0);
  decoded.channel_axis = manifest_.semantic_segmentation.channel_axis.value_or(1);
  decoded.logits_semantics = manifest_.semantic_segmentation.logits_semantics;
  decoded.output_interpolation = manifest_.semantic_segmentation.output_interpolation;
  decoded.class_id_interpolation = manifest_.semantic_segmentation.class_id_interpolation;
  DecoderInput decoder_input;
  decoder_input.outputs.emplace(output_name, std::move(logits));
  decoder_input.geometry = prepared.geometry;
  // Phase 13: the logits stay on the engine device. On CUDA the decoder runs
  // validation, the bilinear resize, and ArgMax in kernels and copies only the
  // compact class map; on CPU it reads the host tensor and copies nothing.
  const auto decode_start = std::chrono::steady_clock::now();
  SemanticSegmentationBatch batch =
      decoder_->decode_semantic_segmentation(decoder_input, decoded, device_);
  const double decode_ms = elapsed_ms(decode_start);
  if (batch.results.empty()) {
    throw VieError(ErrorCode::Internal, "semantic segmentation decoder produced no result");
  }
  SemanticSegmentationResult result = std::move(batch.results.front());
  result.output_bytes = output_bytes;
  // The decoder reports its own compact D2H: 0 on CPU, the class map on CUDA.
  const std::size_t d2h_bytes = result.d2h_bytes;
  const double d2h_ms = result.d2h_ms;
  if (timing != nullptr) {
    timing->preprocess_ms = preprocess_ms;
    timing->execute_ms = execute_ms;
    timing->d2h_ms = d2h_ms;
    timing->validate_ms = result.validate_ms;
    timing->argmax_ms = result.argmax_ms;
    // Wall clock of the decoder call; on CUDA it contains the class-map copy,
    // so the D2H wait is not reported twice.
    timing->decode_ms = decode_ms;
    timing->total_ms = elapsed_ms(total_start);
    timing->output_bytes = output_bytes;
    timing->d2h_bytes = d2h_bytes;
    timing->includes_sync = device_.device() == DeviceType::Cuda;
    timing->includes_h2d = false;
    timing->includes_d2h = d2h_bytes > 0;
  }
  return result;
}

ObbBatch InferenceEngine::infer_obb_options(const ImageInput& image, const InferenceOptions& options,
                                         ObbInferTiming* timing) {
  const ResolvedInferenceConfig resolved = resolve_inference_config(options);
  if (resolved.task != TaskType::Obb) {
    throw VieError(ErrorCode::Unsupported, "infer_obb() requires task obb");
  }
  const ObbDecodeConfig config = obb_decode_config_from_resolved(manifest_, resolved);
  ObbRuntimeConfig runtime;
  runtime.confidence_threshold = config.confidence_threshold;
  runtime.iou_threshold = config.iou_threshold;
  runtime.class_selection = config.class_selection;
  runtime.class_aware = config.class_aware;
  runtime.pre_nms_top_k = config.pre_nms_top_k;
  runtime.max_detections = config.max_detections;
  runtime.nms_enabled = config.nms_enabled;
  return infer_obb(image, runtime, timing);
}

ObbBatch InferenceEngine::infer_obb(const ImageInput& image, const ObbRuntimeConfig& runtime,
                                  ObbInferTiming* timing) {
  if (!loaded_ || backend_ == nullptr || decoder_ == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  if (manifest_.task != TaskType::Obb) {
    throw VieError(ErrorCode::Unsupported, "infer_obb() requires task obb");
  }
  if (manifest_.decoder.protocol != "yolov8_obb_raw") {
    throw VieError(ErrorCode::Unsupported, "infer_obb() requires decoder protocol yolov8_obb_raw");
  }
  ObbDecodeConfig config;
  config.output_name = manifest_.obb.output_name;
  config.channel_axis = manifest_.obb.channel_axis.value_or(-1);
  config.num_classes = manifest_.num_classes.value_or(0);
  config.class_selection = runtime.class_selection;
  config.confidence_threshold = runtime.confidence_threshold;
  config.nms_enabled = runtime.nms_enabled;
  config.class_aware = runtime.class_aware;
  config.iou_threshold = runtime.iou_threshold;
  config.max_detections = runtime.max_detections;
  config.pre_nms_top_k = runtime.pre_nms_top_k;

  const auto total_start = std::chrono::steady_clock::now();
  const PreprocessResult prepared = preprocess(image);
  const double preprocess_ms = elapsed_ms(total_start);
  require_same_device(prepared.output, device_, "preprocess output");
  BackendRequest request;
  request.inputs.emplace(manifest_.input.name, prepared.output);
  const auto execute_start = std::chrono::steady_clock::now();
  const BackendOutput output = execute(request);
  const double execute_ms = elapsed_ms(execute_start);
  const auto found = output.tensors.find(config.output_name);
  if (found == output.tensors.end()) {
    throw VieError(ErrorCode::InvalidArgument, "obb output '" + config.output_name + "' is missing");
  }
  require_same_device(found->second.tensor, device_, "backend output '" + config.output_name + "'");
  Tensor scores = found->second.tensor;
  const std::size_t output_bytes = scores.nbytes();
  DecoderInput decoded;
  decoded.geometry = prepared.geometry;
  decoded.outputs.emplace(config.output_name, std::move(scores));
  // Phase 15: the tensor stays on the engine device. On CUDA the decoder runs
  // candidate validation, the score sort, the GeometryTransform map, Polygon
  // IoU, and rotated NMS in kernels and copies only the kept objects; on CPU it
  // reads the host tensor and copies nothing. The engine records whatever the
  // decoder reports.
  const auto decode_start = std::chrono::steady_clock::now();
  ObbBatch batch = decoder_->decode_obb(decoded, config, device_);
  const double decode_ms = elapsed_ms(decode_start);
  const std::size_t d2h_bytes = batch.results.empty() ? 0 : batch.results[0].d2h_bytes;
  const double d2h_ms = batch.results.empty() ? 0.0 : batch.results[0].d2h_ms;
  if (!batch.results.empty()) {
    batch.results[0].output_bytes = output_bytes;
    batch.results[0].preprocess_ms = preprocess_ms;
    batch.results[0].execute_ms = execute_ms;
    // CPU keeps the documented component sum. CUDA reports the wall clock so
    // launch and stream-sync cost is not hidden behind GPU event time.
    batch.results[0].decode_ms =
        device_.device() == DeviceType::Cuda
            ? decode_ms
            : d2h_ms + batch.results[0].validate_ms + batch.results[0].nms_ms;
  }
  if (timing != nullptr) {
    timing->preprocess_ms = preprocess_ms;
    timing->execute_ms = execute_ms;
    timing->d2h_ms = d2h_ms;
    timing->validate_ms = batch.results.empty() ? 0 : batch.results[0].validate_ms;
    timing->sort_ms = batch.results.empty() ? 0 : batch.results[0].sort_ms;
    timing->map_ms = batch.results.empty() ? 0 : batch.results[0].map_ms;
    timing->nms_ms = batch.results.empty() ? 0 : batch.results[0].nms_ms;
    timing->decode_ms =
        device_.device() == DeviceType::Cuda
            ? decode_ms
            : d2h_ms + timing->validate_ms + timing->nms_ms;
    timing->total_ms = elapsed_ms(total_start);
    timing->output_bytes = output_bytes;
    timing->d2h_bytes = d2h_bytes;
    timing->candidates_before_nms = batch.results.empty() ? 0 : batch.results[0].candidates_before_nms;
    timing->results_after_nms = batch.results.empty() ? 0 : static_cast<int>(batch.results[0].objects.size());
    timing->iou_calls = batch.results.empty() ? 0 : batch.results[0].iou_calls;
    timing->includes_sync = device_.device() == DeviceType::Cuda;
    timing->includes_h2d = false;
    timing->includes_d2h = d2h_bytes > 0;
  }
  return batch;
}

PoseBatch InferenceEngine::infer_pose_options(const ImageInput& image, const InferenceOptions& options,
                                            PoseInferTiming* timing) {
  const ResolvedInferenceConfig resolved = resolve_inference_config(options);
  if (resolved.task != TaskType::Pose) {
    throw VieError(ErrorCode::Unsupported, "infer_pose() requires task pose");
  }
  const PoseDecodeConfig config = pose_decode_config_from_resolved(manifest_, resolved);
  PoseRuntimeConfig runtime;
  runtime.confidence_threshold = config.confidence_threshold;
  runtime.iou_threshold = config.iou_threshold;
  runtime.class_selection = config.class_selection;
  runtime.class_aware = config.class_aware;
  runtime.pre_nms_top_k = config.pre_nms_top_k;
  runtime.max_detections = config.max_detections;
  runtime.nms_enabled = config.nms_enabled;
  return infer_pose(image, runtime, timing);
}

PoseBatch InferenceEngine::infer_pose(const ImageInput& image, const PoseRuntimeConfig& runtime,
                                      PoseInferTiming* timing) {
  if (!loaded_ || backend_ == nullptr || decoder_ == nullptr) {
    throw VieError(ErrorCode::InvalidArgument, "engine is not loaded");
  }
  if (manifest_.task != TaskType::Pose) {
    throw VieError(ErrorCode::Unsupported, "infer_pose() requires task pose");
  }
  if (manifest_.decoder.protocol != "yolov8_pose_raw") {
    throw VieError(ErrorCode::Unsupported, "infer_pose() requires decoder protocol yolov8_pose_raw");
  }
  PoseDecodeConfig config;
  config.output_name = manifest_.pose.output_name;
  config.channel_axis = manifest_.pose.channel_axis.value_or(-1);
  config.num_classes = manifest_.num_classes.value_or(0);
  config.keypoint_count = manifest_.pose.keypoint_count.value_or(0);
  config.keypoint_dimensions = manifest_.pose.keypoint_dimensions.value_or(0);
  config.class_selection = runtime.class_selection;
  config.confidence_threshold = runtime.confidence_threshold;
  config.nms_enabled = runtime.nms_enabled;
  config.class_aware = runtime.class_aware;
  config.iou_threshold = runtime.iou_threshold;
  config.max_detections = runtime.max_detections;
  config.pre_nms_top_k = runtime.pre_nms_top_k;

  const auto total_start = std::chrono::steady_clock::now();
  const PreprocessResult prepared = preprocess(image);
  const double preprocess_ms = elapsed_ms(total_start);
  require_same_device(prepared.output, device_, "preprocess output");
  BackendRequest request;
  request.inputs.emplace(manifest_.input.name, prepared.output);
  const auto execute_start = std::chrono::steady_clock::now();
  const BackendOutput output = execute(request);
  const double execute_ms = elapsed_ms(execute_start);
  const auto found = output.tensors.find(config.output_name);
  if (found == output.tensors.end()) {
    throw VieError(ErrorCode::InvalidArgument, "pose output '" + config.output_name + "' is missing");
  }
  require_same_device(found->second.tensor, device_, "backend output '" + config.output_name + "'");
  Tensor scores = found->second.tensor;
  const std::size_t output_bytes = scores.nbytes();
  DecoderInput decoded;
  decoded.geometry = prepared.geometry;
  decoded.outputs.emplace(config.output_name, scores);
  // Phase 14: the tensor stays on the engine device. On CUDA the decoder runs
  // candidate filtering, box and keypoint decode, the sort, the inverse map,
  // and NMS in kernels and copies only the kept objects; on CPU it reads the
  // host tensor and copies nothing. The engine records whatever the decoder
  // reports.
  const auto decode_start = std::chrono::steady_clock::now();
  PoseBatch batch = decoder_->decode_pose(decoded, config, device_);
  const double decode_ms = elapsed_ms(decode_start);
  const std::size_t d2h_bytes = batch.results.empty() ? 0 : batch.results[0].d2h_bytes;
  const double d2h_ms = batch.results.empty() ? 0.0 : batch.results[0].d2h_ms;
  if (!batch.results.empty()) {
    batch.results[0].output_bytes = output_bytes;
    batch.results[0].preprocess_ms = preprocess_ms;
    batch.results[0].execute_ms = execute_ms;
    // CPU keeps the documented component sum. CUDA reports the wall clock so
    // launch and stream-sync cost is not hidden behind GPU event time.
    batch.results[0].decode_ms =
        device_.device() == DeviceType::Cuda
            ? decode_ms
            : d2h_ms + batch.results[0].validate_ms + batch.results[0].pose_decode_ms +
                  batch.results[0].nms_ms;
  }
  if (timing != nullptr) {
    timing->preprocess_ms = preprocess_ms;
    timing->execute_ms = execute_ms;
    timing->d2h_ms = d2h_ms;
    timing->validate_ms = batch.results.empty() ? 0 : batch.results[0].validate_ms;
    timing->pose_decode_ms = batch.results.empty() ? 0 : batch.results[0].pose_decode_ms;
    timing->nms_ms = batch.results.empty() ? 0 : batch.results[0].nms_ms;
    timing->decode_ms = device_.device() == DeviceType::Cuda
                            ? decode_ms
                            : d2h_ms + timing->validate_ms + timing->pose_decode_ms + timing->nms_ms;
    timing->total_ms = elapsed_ms(total_start);
    timing->output_bytes = output_bytes;
    timing->d2h_bytes = d2h_bytes;
    timing->candidates_before_nms = batch.results.empty() ? 0 : batch.results[0].candidates_before_nms;
    timing->results_after_nms = batch.results.empty() ? 0 : static_cast<int>(batch.results[0].objects.size());
    timing->includes_sync = device_.device() == DeviceType::Cuda;
    timing->includes_h2d = false;
    timing->includes_d2h = d2h_bytes > 0;
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

std::string InferenceEngine::end_profiling() {
  // Forwards to the real backend's session; the default interface method returns
  // "" for backends that never profile, so this is a no-op unless the model was
  // loaded with ort_enable_profiling.
  return backend_ != nullptr ? backend_->end_profiling() : std::string();
}

}  // namespace vie
