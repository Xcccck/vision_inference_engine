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
            << "  --help, -h         Show this help\n";
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
  std::size_t onnx_nodes = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
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
