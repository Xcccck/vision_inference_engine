// vie_ort_profile_inspect: standalone Provider Placement audit of an ORT
// Chrome-trace profiling document (§6). It only reads an already-generated
// profile file -- it never loads a model, creates a Session, or requires the
// CUDA runtime, so it works in CPU-only builds. Raw JSON parsing and provider
// attribution live in vie/diagnostics/ort_profile_parser; this tool just
// formats the statistics and reports what is confirmed vs. unknown.

#include "vie/diagnostics/ort_profile_parser.hpp"

#include <cstddef>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program) {
  std::cerr << "usage: " << program << " <profile.json> [options]\n"
            << "\n"
            << "Analyzes an ONNX Runtime profiling trace and reports the node-level\n"
            << "Execution Provider placement. Does NOT reload the model.\n"
            << "\n"
            << "Options:\n"
            << "  --onnx-nodes <N>   Report N as the original ONNX node count\n"
            << "                     (an ORT profile has no original graph size, so\n"
            << "                     this is shown as 'Not Available' when omitted)\n"
            << "  --json             Emit the placement statistics as a JSON object on\n"
            << "                     stdout for tooling (validate_model_deployment.py);\n"
            << "                     the human report above is suppressed\n"
            << "  --help, -h         Show this help\n";
}

// Machine-readable placement view for the deployment workflow (Phase 9.3.4).
// It reports exactly what the shared parser confirmed and never reinterprets
// memcpy/transfer events as CPU nodes, so a caller can apply the GPU-only rule
// without reimplementing any parsing here.
std::size_t unique_for(const vie::OrtProfileStats& stats, vie::OrtProviderClass wanted);

std::string json_escape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char c : text) {
    if (c == '"') {
      out += "\\\"";
    } else if (c == '\\') {
      out += "\\\\";
    } else {
      out += c;
    }
  }
  return out;
}

void print_json(const std::string& profile_path, const vie::OrtProfileStats& stats) {
  const bool gpu_only = !stats.incomplete && stats.total_node_events > 0 &&
                        stats.cuda_node_events == stats.total_node_events &&
                        stats.cpu_node_events == 0 && stats.unknown_provider_events == 0;
  std::string conclusion;
  if (stats.unknown_provider_events > 0) {
    conclusion = "inconclusive_unattributed";
  } else if (stats.cpu_node_events > 0) {
    conclusion = "cpu_ep_fallback";
  } else if (stats.total_node_events == 0) {
    conclusion = "unattributable";
  } else if (stats.cuda_node_events == stats.total_node_events) {
    conclusion = "all_cuda";
  } else {
    conclusion = "mixed";
  }
  std::cout << "{\n";
  std::cout << "  \"schema_version\": \"1.0\",\n";
  std::cout << "  \"profile_path\": \"" << json_escape(profile_path) << "\",\n";
  std::cout << "  \"parse_status\": \"" << (stats.incomplete ? "incomplete" : "ok") << "\",\n";
  std::cout << "  \"complete\": " << (stats.incomplete ? "false" : "true") << ",\n";
  std::cout << "  \"conclusion\": \"" << conclusion << "\",\n";
  std::cout << "  \"gpu_only\": " << (gpu_only ? "true" : "false") << ",\n";
  std::cout << "  \"total_node_events\": " << stats.total_node_events << ",\n";
  std::cout << "  \"cuda_nodes\": " << stats.cuda_node_events << ",\n";
  std::cout << "  \"cpu_nodes\": " << stats.cpu_node_events << ",\n";
  std::cout << "  \"other_nodes\": " << stats.other_provider_events << ",\n";
  std::cout << "  \"unknown_nodes\": " << stats.unknown_provider_events << ",\n";
  std::cout << "  \"memcpy_events\": " << stats.transfer_events << ",\n";
  std::cout << "  \"unique_cuda_nodes\": " << unique_for(stats, vie::OrtProviderClass::Cuda) << ",\n";
  std::cout << "  \"unique_cpu_nodes\": " << unique_for(stats, vie::OrtProviderClass::Cpu) << ",\n";
  std::cout << "  \"unique_node_names\": " << stats.unique_node_names << ",\n";
  std::cout << "  \"total_trace_events\": " << stats.total_trace_events << ",\n";
  std::cout << "  \"category_node_events\": " << stats.category_node_events << ",\n";
  std::cout << "  \"category_session_events\": " << stats.category_session_events << ",\n";
  std::cout << "  \"category_threadpool_events\": " << stats.category_threadpool_events << ",\n";
  std::cout << "  \"category_memcpy_events\": " << stats.category_memcpy_events << ",\n";
  std::cout << "  \"category_other_events\": " << stats.category_other_events << ",\n";
  std::cout << "  \"category_unknown_events\": " << stats.category_unknown_events << ",\n";
  if (stats.original_onnx_nodes_known) {
    std::cout << "  \"original_onnx_nodes\": " << stats.original_onnx_nodes << ",\n";
    std::cout << "  \"original_onnx_nodes_known\": true,\n";
  } else {
    std::cout << "  \"original_onnx_nodes\": null,\n";
    std::cout << "  \"original_onnx_nodes_known\": false,\n";
  }
  std::cout << "  \"memcpy_names\": [";
  for (std::size_t i = 0; i < stats.memcpy_names.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << "\"" << json_escape(stats.memcpy_names[i]) << "\"";
  }
  std::cout << "]\n}\n";
}

// Distinct node names for one provider class, used for the Unique Nodes rows.
std::size_t unique_for(const vie::OrtProfileStats& stats, vie::OrtProviderClass wanted) {
  std::set<std::string> names;
  for (const vie::OrtNodeEvent& node : stats.node_events) {
    if (node.provider_class == wanted && !node.name.empty()) names.insert(node.name);
  }
  return names.size();
}

void print_conclusion(const vie::OrtProfileStats& stats) {
  std::cout << "Conclusion:\n";
  if (stats.unknown_provider_events > 0) {
    std::cout << "  INCONCLUSIVE: " << stats.unknown_provider_events
              << " executed node(s) could not be attributed to any EP (Profile may be\n"
              << "  incomplete). Other nodes being CUDA/CPU does not clear this.\n";
  } else if (stats.cpu_node_events > 0) {
    std::cout << "  CPU EP placement confirmed: " << stats.cpu_node_events << " of "
              << stats.total_node_events
              << " executed node(s) run on the CPU EP (i.e. a real CPU fallback).\n";
  } else if (stats.total_node_events > 0 && stats.cuda_node_events == stats.total_node_events) {
    std::cout << "  All " << stats.total_node_events
              << " executed node(s) are placed on the CUDA EP; no CPU EP node confirmed.\n";
  } else if (stats.total_node_events == 0) {
    std::cout << "  No executed node events carried a provider; placement cannot be judged.\n";
  } else {
    std::cout << "  Mixed placement: CUDA " << stats.cuda_node_events << ", CPU "
              << stats.cpu_node_events << ", Other " << stats.other_provider_events << ".\n";
  }
  if (stats.transfer_events > 0) {
    std::cout << "  Note: " << stats.transfer_events
              << " memcpy/transfer event(s) were attributed by their provider field; the\n"
              << "  presence of memcpy does NOT by itself indicate a CPU fallback.\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string profile_path;
  bool onnx_known = false;
  bool json_output = false;
  std::size_t onnx_nodes = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
    if (arg == "--json") {
      json_output = true;
      continue;
    }
    if (arg == "--onnx-nodes") {
      if (i + 1 >= argc) {
        std::cerr << "error: --onnx-nodes requires an integer argument\n";
        return 2;
      }
      try {
        onnx_nodes = static_cast<std::size_t>(std::stoul(argv[++i]));
        onnx_known = true;
      } catch (const std::exception&) {
        std::cerr << "error: --onnx-nodes expects a non-negative integer\n";
        return 2;
      }
      continue;
    }
    if (!arg.empty() && arg[0] == '-') {
      std::cerr << "error: unknown option: " << arg << "\n";
      print_usage(argv[0]);
      return 2;
    }
    if (profile_path.empty()) {
      profile_path = arg;
      continue;
    }
    std::cerr << "error: unexpected extra argument: " << arg << "\n";
    print_usage(argv[0]);
    return 2;
  }

  if (profile_path.empty()) {
    std::cerr << "error: no profile file provided\n";
    print_usage(argv[0]);
    return 2;
  }

  vie::OrtProfileStats stats;
  try {
    stats = vie::parse_ort_profile_file(profile_path);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  stats.original_onnx_nodes = onnx_nodes;
  stats.original_onnx_nodes_known = onnx_known;

  if (json_output) {
    print_json(profile_path, stats);
    return stats.incomplete ? 1 : 0;
  }

  std::cout << "ORT Profile Summary\n";
  std::cout << "===================\n\n";

  std::cout << "Profile:\n";
  std::cout << "  Path: " << profile_path << "\n";
  std::cout << "  Parse Status: " << (stats.incomplete ? "incomplete" : "ok") << "\n";
  std::cout << "  Complete: " << (stats.incomplete ? "no" : "yes") << "\n\n";

  std::cout << "Events:\n";
  std::cout << "  Total Events: " << stats.total_trace_events << "\n";
  std::cout << "  Node Events: " << stats.category_node_events << "\n";
  std::cout << "  Session Events: " << stats.category_session_events << "\n";
  std::cout << "  ThreadPool Events: " << stats.category_threadpool_events << "\n";
  std::cout << "  Memcpy/Transfer Events: " << stats.transfer_events << "\n";
  std::cout << "  Other Events: " << stats.category_other_events << "\n";
  std::cout << "  Unknown Events: " << stats.category_unknown_events << "\n\n";

  std::cout << "Provider Placement:\n";
  std::cout << "  CUDA Execution Provider:\n";
  std::cout << "    Node Events: " << stats.cuda_node_events << "\n";
  std::cout << "    Unique Nodes: " << unique_for(stats, vie::OrtProviderClass::Cuda) << "\n";
  std::cout << "  CPU Execution Provider:\n";
  std::cout << "    Node Events: " << stats.cpu_node_events << "\n";
  std::cout << "    Unique Nodes: " << unique_for(stats, vie::OrtProviderClass::Cpu) << "\n";
  std::cout << "  Other Providers:\n";
  std::cout << "    Node Events: " << stats.other_provider_events << "\n";
  std::cout << "  Unknown Provider:\n";
  std::cout << "    Node Events: " << stats.unknown_provider_events << "\n\n";

  std::cout << "Graph Information:\n";
  if (stats.original_onnx_nodes_known) {
    std::cout << "  Original ONNX Nodes: " << stats.original_onnx_nodes << "\n";
  } else {
    std::cout << "  Original ONNX Nodes: Not Available\n";
  }
  std::cout << "  Profile Node Events: " << stats.total_node_events << "\n";
  std::cout << "  Unique Profile Nodes: " << stats.unique_node_names << "\n\n";

  print_conclusion(stats);

  // Signal an unattributable/incomplete profile with a non-zero exit so callers
  // can distinguish "all placed cleanly" from "needs investigation" (§6).
  return stats.incomplete ? 1 : 0;
}
