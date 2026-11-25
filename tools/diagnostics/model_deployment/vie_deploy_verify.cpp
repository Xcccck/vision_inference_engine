// vie_deploy_verify: run a real image through the actual VIE pipeline and emit a
// machine-readable verification record (Phase 9.3.4 model-deployment workflow).
//
// It deliberately reuses VIE itself instead of duplicating inference:
//   * model load / device selection / CUDA-vs-CPU EP via InferenceEngine::load
//   * preprocessing (letterbox, normalisation) via the manifest-driven pipeline
//   * the decoder + NMS via InferenceEngine::infer  -> real DetectionResult
//   * ONNX Runtime provider placement via the model's own ORT profiling
//     (ModelConfig::ort_enable_profiling + InferenceEngine::end_profiling)
//   * full-pipeline per-call latency (preprocess + execute + decode) via the
//     DetectionInferTiming the engine already measures.
//
// The tool is generic: the task, decoder and preprocessing all come from the
// supplied manifest, so nothing here is YOLO-specific -- yolov8n is only the
// first fixture. It performs NO model conversion and NEVER mutates the input.
//
// It writes a JSON document describing what it observed and exits 0 even when a
// strict CUDA session legitimately fails to create (that is a real result the
// caller must classify, not a tool crash). Non-zero is reserved for misuse.

#include "vie/decoder/detection.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest.hpp"
#include "vie/model/model_config.hpp"
#include "vie/preprocess/image_input.hpp"

#ifdef VIE_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

// Reads a binary P6 PPM (810x1080 bus.ppm in the repo). Returns a host RGB HWC
// ImageInput borrowing `pixels`; same reader style as the existing test fixtures.
vie::ImageInput load_ppm(const std::string& path, std::vector<std::uint8_t>& pixels, bool& ok) {
  ok = false;
  std::ifstream file(path, std::ios::binary);
  if (!file) return {};
  std::string magic;
  int width = 0, height = 0, max_value = 0;
  file >> magic >> width >> height >> max_value;
  if (magic != "P6" || width <= 0 || height <= 0 || max_value <= 0) return {};
  file.get();  // single whitespace after the header
  pixels.resize(static_cast<std::size_t>(width) * height * 3);
  file.read(reinterpret_cast<char*>(pixels.data()),
            static_cast<std::streamsize>(pixels.size()));
  if (!file) return {};
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  image.device = vie::DeviceType::Cpu;
  ok = true;
  return image;
}

double percentile(std::vector<double> sorted, double q) {
  if (sorted.empty()) return 0.0;
  if (sorted.size() == 1) return sorted.front();
  const double rank = q * (static_cast<double>(sorted.size()) - 1.0);
  const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
  const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
  const double frac = rank - static_cast<double>(lower);
  return sorted[lower] * (1.0 - frac) + sorted[upper] * frac;
}

void print_usage(const char* program) {
  std::cerr
      << "usage: " << program
      << " --model <onnx> --manifest <json> --image <ppm> [options]\n"
      << "\n"
      << "Runs one image through the real VIE engine and reports session creation,\n"
      << "detections (via VIE's own decoder/NMS), full-pipeline latency and the ORT\n"
      << "provider profile path. Does NOT convert or modify any model.\n"
      << "\n"
      << "Options:\n"
      << "  --model PATH       ONNX model to load (required)\n"
      << "  --manifest PATH    ModelManifest JSON driving decode/preprocess (required)\n"
      << "  --image PATH       P6 PPM image (required)\n"
      << "  --device cpu|cuda  execution device (default: cpu)\n"
      << "  --strict           CUDA: disable CPU EP fallback (cuda_cpu_ep_fallback=false)\n"
      << "  --warmup N         untimed warmup inference calls (default: 3)\n"
      << "  --repeat N         timed inference calls (default: 20)\n"
      << "  --profile-dir DIR  enable ORT profiling; traces written under DIR\n"
      << "  --json PATH        write the result JSON to PATH (default: stdout)\n";
}

struct Args {
  std::string model, manifest, image, json_path, profile_dir;
  vie::DeviceType device = vie::DeviceType::Cpu;
  bool strict = false;
  int warmup = 3;
  int repeat = 20;
};

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "error: " << name << " requires a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    else if (a == "--model") args.model = need("--model");
    else if (a == "--manifest") args.manifest = need("--manifest");
    else if (a == "--image") args.image = need("--image");
    else if (a == "--json") args.json_path = need("--json");
    else if (a == "--profile-dir") args.profile_dir = need("--profile-dir");
    else if (a == "--device") {
      const std::string d = need("--device");
      if (d == "cpu") args.device = vie::DeviceType::Cpu;
      else if (d == "cuda") args.device = vie::DeviceType::Cuda;
      else { std::cerr << "error: --device must be cpu or cuda\n"; return 2; }
    } else if (a == "--strict") args.strict = true;
    else if (a == "--warmup") args.warmup = std::atoi(need("--warmup"));
    else if (a == "--repeat") args.repeat = std::atoi(need("--repeat"));
    else { std::cerr << "error: unknown argument: " << a << "\n"; print_usage(argv[0]); return 2; }
  }
  if (args.model.empty() || args.manifest.empty() || args.image.empty()) {
    std::cerr << "error: --model, --manifest and --image are required\n";
    print_usage(argv[0]);
    return 2;
  }
  if (args.warmup < 0) args.warmup = 0;
  if (args.repeat < 1) args.repeat = 1;

  std::ostringstream js;
  js << std::fixed << std::setprecision(6);
  js << "{\n";
  js << "  \"schema_version\": \"1.0\",\n";
  js << "  \"tool\": \"vie_deploy_verify\",\n";
  js << "  \"model\": \"" << json_escape(args.model) << "\",\n";
  js << "  \"manifest\": \"" << json_escape(args.manifest) << "\",\n";
  js << "  \"image\": \"" << json_escape(args.image) << "\",\n";
  js << "  \"device\": \"" << (args.device == vie::DeviceType::Cuda ? "cuda" : "cpu") << "\",\n";
  js << "  \"strict_cuda\": " << (args.strict ? "true" : "false") << ",\n";

  // 1) Manifest + engine configuration.
  vie::ModelManifest manifest;
  try {
    manifest = vie::ModelManifest::load_file(args.manifest);
  } catch (const std::exception& error) {
    js << "  \"session_create\": {\"status\": \"manifest_error\", \"error\": \""
       << json_escape(error.what()) << "\"}\n}\n";
    const std::string text = js.str();
    if (args.json_path.empty()) std::cout << text; else std::ofstream(args.json_path) << text;
    return 0;
  }

  vie::ModelConfig config;
  config.model_path = args.model;
  config.task = vie::TaskType::Detection;
  config.backend = vie::BackendKind::Auto;
  config.device = args.device;
  config.device_id = 0;
  config.manifest = manifest;
  // Strict GPU-only means no silent CPU fallback. On CPU the flag is a no-op.
  config.cuda_cpu_ep_fallback = (args.device == vie::DeviceType::Cuda) && !args.strict;
  const bool profiling = !args.profile_dir.empty();
  config.ort_enable_profiling = profiling;
  if (profiling) config.ort_profile_prefix = args.profile_dir + "/deploy_profile_";

  vie::InferenceEngine engine;
  bool session_ok = false;
  std::string session_error;
  try {
    engine.load(config);
    session_ok = true;
  } catch (const std::exception& error) {
    session_error = error.what();
  }
  js << "  \"session_create\": {\"status\": \"" << (session_ok ? "PASS" : "FAIL")
     << "\", \"error\": \"" << json_escape(session_error) << "\"},\n";

  if (!session_ok) {
    js << "  \"inference\": {\"status\": \"NOT_RUN\"},\n";
    js << "  \"detections\": []\n}\n";
    const std::string text = js.str();
    if (args.json_path.empty()) std::cout << text; else std::ofstream(args.json_path) << text;
    return 0;
  }

  // 2) Load the real image and run inference.
  std::vector<std::uint8_t> pixels;
  bool image_ok = false;
  vie::ImageInput image = load_ppm(args.image, pixels, image_ok);
  if (!image_ok) {
    js << "  \"inference\": {\"status\": \"FAIL\", \"error\": \"cannot load PPM image\"},\n";
    js << "  \"detections\": []\n}\n";
    const std::string text = js.str();
    if (args.json_path.empty()) std::cout << text; else std::ofstream(args.json_path) << text;
    return 0;
  }

  // A CUDA engine rejects a host image (device-match boundary), so for the CUDA
  // path we stage the real pixels onto the device first -- exactly what the
  // Runtime's async CPU->CUDA staging does internally, but synchronous here.
#ifdef VIE_HAS_CUDA
  void* device_pixels = nullptr;
  if (args.device == vie::DeviceType::Cuda) {
    const std::size_t bytes = pixels.size();
    if (cudaMalloc(&device_pixels, bytes) != cudaSuccess) device_pixels = nullptr;
    if (device_pixels &&
        cudaMemcpy(device_pixels, pixels.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(device_pixels);
      device_pixels = nullptr;
    }
    if (!device_pixels) {
      js << "  \"inference\": {\"status\": \"FAIL\", \"error\": \"host-to-device staging failed\"},\n";
      js << "  \"detections\": []\n}\n";
      const std::string text = js.str();
      if (args.json_path.empty()) std::cout << text; else std::ofstream(args.json_path) << text;
      return 0;
    }
    image.data = device_pixels;
    image.device = vie::DeviceType::Cuda;
    image.device_id = 0;
  }
#else
  if (args.device == vie::DeviceType::Cuda) {
    js << "  \"inference\": {\"status\": \"UNSUPPORTED\", \"error\": \"built without CUDA\"},\n";
    js << "  \"detections\": []\n}\n";
    const std::string text = js.str();
    if (args.json_path.empty()) std::cout << text; else std::ofstream(args.json_path) << text;
    return 0;
  }
#endif

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(args.repeat));
  vie::DetectionBatch last;
  bool inference_ok = false;
  std::string inference_error;
  try {
    for (int i = 0; i < args.warmup; ++i) engine.infer(image);
    for (int i = 0; i < args.repeat; ++i) {
      vie::DetectionInferTiming timing;
      last = engine.infer(image, &timing);
      samples.push_back(timing.total_ms);
    }
    inference_ok = true;
  } catch (const std::exception& error) {
    inference_error = error.what();
  }

  std::string profile_path;
  if (profiling) {
    try {
      profile_path = engine.end_profiling();
    } catch (const std::exception&) {
      profile_path.clear();
    }
  }
#ifdef VIE_HAS_CUDA
  if (device_pixels) cudaFree(device_pixels);
#endif

  js << "  \"inference\": {\"status\": \"" << (inference_ok ? "PASS" : "FAIL")
     << "\", \"error\": \"" << json_escape(inference_error) << "\"},\n";
  js << "  \"profile_path\": "
     << (profile_path.empty() ? "null" : "\"" + json_escape(profile_path) + "\"") << ",\n";

  if (inference_ok && !samples.empty()) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
                        static_cast<double>(sorted.size());
    js << "  \"latency_ms\": {\n";
    js << "    \"rounds\": " << samples.size() << ",\n";
    js << "    \"mean\": " << mean << ",\n";
    js << "    \"p50\": " << percentile(sorted, 0.50) << ",\n";
    js << "    \"p95\": " << percentile(sorted, 0.95) << ",\n";
    js << "    \"max\": " << sorted.back() << "\n";
    js << "  },\n";
  } else {
    js << "  \"latency_ms\": null,\n";
  }

  // 3) Detections from VIE's own decoder + NMS (reused, not reimplemented).
  if (!last.results.empty()) {
    const vie::DetectionResult& result = last.results.front();
    js << "  \"detection_meta\": {\n";
    js << "    \"protocol\": \"" << json_escape(result.protocol) << "\",\n";
    js << "    \"original_width\": " << result.original_width << ",\n";
    js << "    \"original_height\": " << result.original_height << ",\n";
    js << "    \"coordinates_mapped\": " << (result.coordinates_mapped ? "true" : "false") << ",\n";
    js << "    \"nms_applied\": " << (result.nms_applied ? "true" : "false") << ",\n";
    js << "    \"count\": " << result.detections.size() << "\n";
    js << "  },\n";
    js << "  \"detections\": [";
    for (std::size_t i = 0; i < result.detections.size(); ++i) {
      const vie::Detection& det = result.detections[i];
      if (i) js << ", ";
      js << "{\"class_id\": " << det.class_id << ", \"confidence\": " << det.confidence
         << ", \"x1\": " << det.box.x1 << ", \"y1\": " << det.box.y1 << ", \"x2\": " << det.box.x2
         << ", \"y2\": " << det.box.y2 << "}";
    }
    js << "]\n";
  } else {
    js << "  \"detections\": []\n";
  }
  js << "}\n";

  const std::string text = js.str();
  if (args.json_path.empty()) {
    std::cout << text;
  } else {
    std::ofstream out(args.json_path);
    if (!out) {
      std::cerr << "error: cannot write " << args.json_path << "\n";
      return 1;
    }
    out << text;
  }
  return 0;
}
