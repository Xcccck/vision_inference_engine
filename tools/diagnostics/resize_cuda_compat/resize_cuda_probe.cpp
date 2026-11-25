// Phase 9.3.3 diagnostic probe (NOT part of VIE production; standalone tool).
//
// Loads an arbitrary ONNX model with ONNX Runtime 1.20.1 GPU and mirrors the
// EXACT calls the production CUDA backend makes:
//   * CUDA EP via AppendExecutionProvider_CUDA(device_id = 0)
//   * strict mode via AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1")
//   * profiling via EnableProfiling(prefix)
// It runs at ORT VERBOSE log severity so the CUDA EP "GetCapability" rejection
// reason for each node (in particular Resize) is captured to a log file -- that
// reason is the primary evidence for WHY a node lands on the CPU EP.
//
// Usage:
//   resize_cuda_probe MODEL.onnx [--cpu] [--strict] [--profile PREFIX]
//                        [--dump-opt OUT.onnx] [--opt-level N]
//                        [--log LOGPATH] [--feed] [--allow-dynamic]
// Exit code: 0 session created OK, 3 session creation failed (strict), 2 bad args.
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
std::string g_log_path;

void ProbeLogger(void*, OrtLoggingLevel level, const char* tag, const char*, const char*,
                 const char* message) {
  static const char* names[] = {"VERBOSE", "INFO", "WARNING", "ERROR", "FATAL"};
  const int li = static_cast<int>(level);
  std::ofstream ofs(g_log_path, std::ios::app);
  ofs << "[" << (li >= 0 && li < 5 ? names[li] : "?") << "] " << tag << " | " << message << "\n";
  std::fprintf(stderr, "[%s] %s\n", li >= 0 && li < 5 ? names[li] : "?", message);
}

GraphOptimizationLevel ParseLevel(const char* s) {
  std::string v(s);
  if (v == "0") return ORT_DISABLE_ALL;
  if (v == "1") return ORT_ENABLE_BASIC;
  if (v == "2") return ORT_ENABLE_EXTENDED;
  return ORT_ENABLE_ALL;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s MODEL.onnx [--cpu] [--strict] [--profile PREFIX] "
                         "[--dump-opt OUT] [--opt-level N] [--log PATH] [--feed]\n", argv[0]);
    return 2;
  }
  const std::string model = argv[1];
  bool use_cpu = false, strict = false, feed = false;
  int repeat = 1;
  std::string profile_prefix, dump_opt, log_path = "resize_probe.log";
  GraphOptimizationLevel level = ORT_ENABLE_ALL;
  const auto t_create0 = std::chrono::steady_clock::now();
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--cpu") use_cpu = true;
    else if (a == "--strict") strict = true;
    else if (a == "--feed") feed = true;
    else if (a == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
    else if (a == "--profile" && i + 1 < argc) profile_prefix = argv[++i];
    else if (a == "--dump-opt" && i + 1 < argc) dump_opt = argv[++i];
    else if (a == "--opt-level" && i + 1 < argc) level = ParseLevel(argv[++i]);
    else if (a == "--log" && i + 1 < argc) log_path = argv[++i];
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }
  g_log_path = log_path;
  std::ofstream(log_path).close();  // truncate

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "resize_cuda_probe", &ProbeLogger, nullptr);
    Ort::SessionOptions opts{};
    opts.SetGraphOptimizationLevel(level);
    opts.SetLogSeverityLevel(0);  // VERBOSE
    opts.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, strict ? "1" : "0");
    if (!dump_opt.empty()) opts.SetOptimizedModelFilePath(dump_opt.c_str());
    if (!profile_prefix.empty()) opts.EnableProfiling(profile_prefix.c_str());
    if (!use_cpu) {
      OrtCUDAProviderOptions cuda{};
      cuda.device_id = 0;
      cuda.do_copy_in_default_stream = 1;
      opts.AppendExecutionProvider_CUDA(cuda);
    }

    std::printf("provider=%s strict=%s opt_level=%d model=%s\n", use_cpu ? "CPU" : "CUDA",
                strict ? "ON(disable-fallback)" : "OFF(fallback-allowed)",
                static_cast<int>(level), model.c_str());

    Ort::Session session(env, model.c_str(), opts);  // may throw under strict
    const double load_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t_create0).count();
    std::printf("SESSION_CREATE=OK load_ms=%.2f\n", load_ms);

    if (feed) {
      Ort::AllocatorWithDefaultOptions alloc;
      const size_t n_in = session.GetInputCount();
      auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::vector<std::string> names;
      std::vector<const char*> name_ptrs;
      std::vector<std::vector<int64_t>> shapes;
      std::vector<std::vector<float>> buffers;
      std::vector<Ort::Value> inputs;
      for (size_t i = 0; i < n_in; ++i) {
        auto ti = session.GetInputTypeInfo(i);
        auto tinfo = ti.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> raw = tinfo.GetShape();
        std::vector<int64_t> dims;
        for (auto d : raw) dims.push_back(d < 0 ? 1 : d);
        std::string nm = session.GetInputNameAllocated(i, alloc).get();
        size_t cnt = 1;
        for (auto d : dims) cnt *= static_cast<size_t>(d);
        buffers.emplace_back(cnt, 0.0f);
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, buffers.back().data(),
                                                         buffers.back().size(), dims.data(), dims.size()));
        names.push_back(std::move(nm));
        shapes.push_back(std::move(dims));
      }
      for (auto& s : names) name_ptrs.push_back(s.c_str());
      const size_t n_out = session.GetOutputCount();
      std::vector<const char*> out_ptrs;
      std::vector<std::string> out_names;
      for (size_t i = 0; i < n_out; ++i) out_names.push_back(session.GetOutputNameAllocated(i, alloc).get());
      for (auto& s : out_names) out_ptrs.push_back(s.c_str());
      auto out = session.Run(Ort::RunOptions{}, name_ptrs.data(), inputs.data(), n_in, out_ptrs.data(), n_out);
      // Steady-state timing: warmup already done by the first Run above; time
      // `repeat` additional runs and report latency percentiles (profiling OFF
      // unless --profile given; production path has profiling off).
      {
        std::vector<double> ms;
        ms.reserve(repeat > 0 ? repeat : 0);
        for (int r = 0; r < repeat; ++r) {
          const auto t0 = std::chrono::steady_clock::now();
          (void)session.Run(Ort::RunOptions{}, name_ptrs.data(), inputs.data(), n_in, out_ptrs.data(), n_out);
          ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        if (!ms.empty()) {
          std::sort(ms.begin(), ms.end());
          double sum = 0;
          for (double v : ms) sum += v;
          size_t p50 = ms.size() / 2, p95 = (ms.size() * 95) / 100;
          if (p95 >= ms.size()) p95 = ms.size() - 1;
          std::printf("TIMING runs=%d mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f min_ms=%.3f\n",
                      repeat, sum / ms.size(), ms[p50], ms[p95], ms.front());
        }
      }
      for (size_t i = 0; i < out.size(); ++i) {
        auto tinfo = out[i].GetTensorTypeAndShapeInfo();
        auto shp = tinfo.GetShape();
        std::string s;
        for (auto d : shp) s += std::to_string(d) + ",";
        std::printf("OUTPUT[%zu] name=%s rank=%zu shape=[%s]\n", i, out_names[i].c_str(), shp.size(), s.c_str());
      }
      std::printf("RUN=OK\n");
    }

    if (!profile_prefix.empty()) {
      Ort::AllocatorWithDefaultOptions alloc;
      auto path = session.EndProfilingAllocated(alloc);
      std::printf("PROFILE_PATH=%s\n", path.get());
    }
    std::printf("PROBE=SUCCESS\n");
    return 0;
  } catch (const Ort::Exception& e) {
    std::printf("SESSION_CREATE=FAIL\n");
    std::printf("ORT_ERROR: %s\n", e.what());
    return 3;
  }
}
