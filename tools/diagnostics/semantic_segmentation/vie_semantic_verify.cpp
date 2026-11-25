// Phase 9.4 real-model semantic-segmentation end-to-end verifier.
//
// It recognizes the DeepLabV3 ONNX (via VIE's own analyzer + recognizer +
// manifest generator), saves and reloads a schema 1.0 manifest, loads the real
// engine on the requested device, then dumps the three artifacts the Python
// reference compares against: the model-input tensor (preprocessing check), the
// raw class-logit tensor (backend numerical check), and the decoded per-pixel
// class map (decoder coordinate check). It does not convert or edit the model
// and does not fabricate weights. CUDA uses the real CUDA backend; there is no
// silent CPU fallback (cuda_cpu_ep_fallback defaults to false).
#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/core/tensor.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"
#include "vie/runtime/cuda_input_staging.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

void write_raw(const std::string& path, const void* data, std::size_t bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

// Reads a binary P6 PPM into an RGB HWC uint8 ImageInput.
vie::ImageInput load_ppm(const std::string& path, std::vector<std::uint8_t>& pixels) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int width = 0;
  int height = 0;
  int max_value = 0;
  file >> magic >> width >> height >> max_value;
  if (magic != "P6" || width <= 0 || height <= 0 || max_value <= 0) {
    throw std::runtime_error("not a P6 PPM: " + path);
  }
  file.get();
  pixels.assign(static_cast<std::size_t>(width) * height * 3, 0);
  file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  if (!file) throw std::runtime_error("short read: " + path);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  image.device = vie::DeviceType::Cpu;
  return image;
}

struct Args {
  std::string model;
  std::string image;
  std::string prefix = "semantic";
  std::string manifest_out;
  std::string device = "cpu";
  std::string profile_prefix;
  int size = 520;
  int device_id = 0;
  bool ep_fallback = false;
};

Args parse(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    const auto next = [&] { return i + 1 < argc ? argv[++i] : std::string(); };
    if (key == "--model") args.model = next();
    else if (key == "--image") args.image = next();
    else if (key == "--prefix") args.prefix = next();
    else if (key == "--manifest-out") args.manifest_out = next();
    else if (key == "--device") args.device = next();
    else if (key == "--device-id") args.device_id = std::stoi(next());
    else if (key == "--size") args.size = std::stoi(next());
    else if (key == "--profile-prefix") args.profile_prefix = next();
    else if (key == "--ep-fallback") args.ep_fallback = true;
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  if (args.model.empty() || args.image.empty()) {
    std::cerr << "usage: vie_semantic_verify --model <onnx> --image <ppm> [--prefix P] "
                 "[--device cpu|cuda] [--size 520] [--manifest-out path] [--ep-fallback]\n";
    return 2;
  }
  try {
    // 1) Recognize and generate a manifest from the real graph.
    const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(args.model);
    const vie::RecognitionResult recognition = vie::default_recognizer_registry().recognize(report);
    if (recognition.selected() == nullptr ||
        recognition.selected()->architecture != "deeplabv3_semantic_segmentation") {
      std::cerr << "the recognizer did not identify this model as semantic segmentation\n";
      return 1;
    }
    vie::ModelManifest manifest =
        vie::ManifestGenerator{}.generate(report, recognition, args.model);
    // Fill the preprocessing contract from the torchvision DeepLabV3 convention:
    // RGB, resize to the fixed model grid, ImageNet mean/std, bilinear.
    manifest.input.color = "rgb";
    manifest.input.normalization = "imagenet";
    manifest.input.resize = "stretch";
    manifest.input.interpolation = "bilinear";
    // These come from the torchvision DeepLabV3 transform contract, recorded as
    // User provenance so the manifest becomes runnable (generate() leaves them
    // unresolved on purpose).
    const vie::FieldProvenance user_transform{
        vie::Certainty::User, {"torchvision DeepLabV3 eval transform (RGB, ImageNet mean/std)"}};
    for (const char* field : {"input.color", "input.normalization", "input.resize", "input.interpolation"}) {
      manifest.provenance[field] = user_transform;
    }
    manifest.ensure_provenance();
    if (!args.manifest_out.empty()) manifest.save_file(args.manifest_out);

    // 2) Save -> reload the manifest and load the real engine on the device.
    const std::string manifest_text = manifest.to_json_string();
    const vie::ModelManifest reloaded = vie::ModelManifest::from_json_string(manifest_text);
    vie::ModelConfig config;
    config.manifest = reloaded;
    config.model_path = args.model;
    config.backend = vie::BackendKind::Auto;
    config.device = args.device == "cuda" ? vie::DeviceType::Cuda : vie::DeviceType::Cpu;
    config.device_id = 0;
    config.cuda_cpu_ep_fallback = args.ep_fallback;
    // Optional ORT Chrome-trace profiling for the CUDA placement audit.
    // Off by default; enabling it does not change the execution path.
    if (!args.profile_prefix.empty()) {
      config.ort_enable_profiling = true;
      config.ort_profile_prefix = args.profile_prefix;
    }
    vie::InferenceEngine engine;
    engine.load(config);

    std::vector<std::uint8_t> pixels;
    vie::ImageInput image = load_ppm(args.image, pixels);
    // When the engine runs on CUDA, ImageInput must live on the same device.
    // Stage the raw pixels through the production Runtime helper rather than
    // inventing a second upload path; the staged buffer keeps the bytes alive
    // for every call below.
    std::optional<vie::CudaInputBuffer> staged;
    if (config.device == vie::DeviceType::Cuda) {
      staged = vie::CudaInputBuffer::stage(image, vie::DeviceContext(vie::DeviceType::Cuda, args.device_id));
      image = staged->image();
    }

    // 3) Preprocessing artifact: dump the model-input tensor VIE feeds the backend.
    const vie::PreprocessResult prepared = engine.preprocess(image);
    const vie::Tensor input_on_cpu =
        prepared.output.is_cuda() ? prepared.output.to(vie::DeviceContext(vie::DeviceType::Cpu, 0))
                                  : prepared.output;
    write_raw(args.prefix + "_input.f32", input_on_cpu.data_as<float>(), input_on_cpu.nbytes());

    // 4) Backend numerical artifact: run execute() and dump the raw logits.
    vie::BackendRequest request;
    request.inputs.emplace(manifest.input.name, prepared.output);
    const vie::BackendOutput output = engine.execute(request);
    const auto found = output.tensors.find(manifest.semantic_segmentation.output_name);
    if (found == output.tensors.end()) {
      std::cerr << "logits output missing: " << manifest.semantic_segmentation.output_name << "\n";
      return 1;
    }
    const vie::Tensor logits_on_cpu = found->second.tensor.is_cuda()
                                          ? found->second.tensor.to(vie::DeviceContext(vie::DeviceType::Cpu, 0))
                                          : found->second.tensor;
    write_raw(args.prefix + "_logits.f32", logits_on_cpu.data_as<float>(), logits_on_cpu.nbytes());

    // 5) Decoder artifact: the dense class map in original image coordinates.
    vie::SemanticSegmentationInferTiming timing;
    const vie::SemanticSegmentationResult result = engine.infer_semantic_segmentation(image, &timing);
    write_raw(args.prefix + "_classids.i32", result.class_ids.data(),
              result.class_ids.size() * sizeof(std::int32_t));

    std::cout << "{\"device\":\"" << (config.device == vie::DeviceType::Cuda ? "cuda" : "cpu") << "\""
              << ",\"execution_device\":\"" << (result.execution_device == vie::DeviceType::Cuda ? "cuda" : "cpu") << "\""
              << ",\"original_width\":" << result.width << ",\"original_height\":" << result.height
              << ",\"num_classes\":" << result.num_classes << ",\"logits_width\":" << result.logits_width
              << ",\"logits_height\":" << result.logits_height << ",\"class_id_count\":" << result.class_ids.size()
              << ",\"d2h_bytes\":" << result.d2h_bytes << ",\"output_bytes\":" << result.output_bytes
              << ",\"preprocess_ms\":" << timing.preprocess_ms << ",\"execute_ms\":" << timing.execute_ms
              << ",\"decode_ms\":" << timing.decode_ms << ",\"total_ms\":" << timing.total_ms
              << ",\"includes_d2h\":" << (timing.includes_d2h ? "true" : "false") << "}\n";
    if (!args.profile_prefix.empty()) {
      const std::string path = engine.end_profiling();
      std::cerr << "ort_profile: " << path << "\n";
    }
    return 0;
  } catch (const vie::VieError& error) {
    std::cerr << "VIE error [" << vie::to_string(error.code()) << "]: " << error.what() << "\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
