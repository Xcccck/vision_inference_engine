#include "vie/diagnostics/ort_profile_parser.hpp"

#include "vie/core/error.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

namespace vie {
namespace {

using json = nlohmann::json;

// ASCII-lowercases a copy of the input; used so ORT spelling differences
// ("CUDA"/"cuda", "Memcpy"/"memcpy", category casing) all match uniformly.
std::string to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

// Case-insensitive substring test.
bool contains_ci(const std::string& haystack, std::string_view needle) {
  if (needle.empty()) return true;
  return to_lower(haystack).find(to_lower(std::string(needle))) != std::string::npos;
}

// Reads a string field, returning an empty string when absent or not a string.
std::string get_string(const json& obj, const char* key) {
  if (!obj.is_object()) return {};
  const auto it = obj.find(key);
  if (it == obj.end() || !it->is_string()) return {};
  return it->get<std::string>();
}

double get_number(const json& obj, const char* key) {
  if (!obj.is_object()) return 0.0;
  const auto it = obj.find(key);
  if (it == obj.end() || !it->is_number()) return 0.0;
  return it->get<double>();
}

// An event is a memcpy/transfer event when its op_name, node name or category
// carries a memcpy marker. These are tallied separately from compute nodes.
bool looks_like_memcpy(const std::string& cat, const std::string& name,
                       const std::string& op_name) {
  return contains_ci(cat, "memcpy") || contains_ci(name, "memcpy") ||
         contains_ci(op_name, "memcpy");
}

// ORT 1.20.1 logs each executed node as three Chrome-trace events whose names
// carry a timing suffix; only the *_kernel_time event holds args.provider (the
// authoritative EP placement). Stripping the suffix recovers the graph node
// name so counts and the detail table refer to real nodes, not event variants.
std::string base_node_name(const std::string& name) {
  static constexpr const char* kSuffixes[] = {"_kernel_time", "_fence_before", "_fence_after"};
  for (const char* suffix : kSuffixes) {
    const std::string s(suffix);
    if (name.size() >= s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0) {
      return name.substr(0, name.size() - s.size());
    }
  }
  return name;
}

}  // namespace

OrtProviderClass classify_ort_provider(const std::string& provider) {
  if (provider.empty()) return OrtProviderClass::Unknown;
  if (contains_ci(provider, "cuda")) return OrtProviderClass::Cuda;
  if (contains_ci(provider, "cpu")) return OrtProviderClass::Cpu;
  return OrtProviderClass::Other;
}

OrtProfileStats parse_ort_profile_string(std::string_view json_text) {
  json root;
  try {
    root = json::parse(json_text);
  } catch (const json::exception& error) {
    throw VieError(ErrorCode::InvalidArgument,
                   std::string("ort profile is not valid JSON: ") + error.what());
  }
  // ORT 1.20.1 EndProfiling writes a bare JSON array of trace events; some
  // Chrome-trace variants wrap the same list in {"traceEvents": [...]}. Accept
  // both shapes and reject a scalar root as structurally invalid.
  const json* events = nullptr;
  if (root.is_array()) {
    events = &root;
  } else if (root.is_object()) {
    const auto it = root.find("traceEvents");
    if (it == root.end()) return OrtProfileStats{};  // empty, well-formed document
    if (!it->is_array()) {
      throw VieError(ErrorCode::InvalidArgument, "ort profile 'traceEvents' is not an array");
    }
    events = &(*it);
  } else {
    throw VieError(ErrorCode::InvalidArgument, "ort profile root is neither array nor object");
  }

  OrtProfileStats stats;
  std::set<std::string> unique_names;
  std::set<std::string> unique_memcpy;

  for (const json& ev : *events) {
    if (!ev.is_object()) continue;
    ++stats.total_trace_events;
    std::string cat = get_string(ev, "cat");
    if (cat.empty()) cat = get_string(ev, "category");
    const std::string name = get_string(ev, "name");

    const auto args_it = ev.find("args");
    const bool has_args = args_it != ev.end() && args_it->is_object();
    const bool has_provider =
        has_args && args_it->find("provider") != args_it->end() && (*args_it)["provider"].is_string();

    // The only events ORT tags with args.provider are the per-node kernel_time
    // events. Every other event (Session init, the *_fence_before/_fence_after
    // markers, thread metadata) has no provider and is NOT an executed node, so
    // it is skipped rather than guessed at (§5.1, A7).
    if (!has_provider) continue;

    const std::string provider = get_string(*args_it, "provider");
    const std::string op_name = get_string(*args_it, "op_name");

    OrtNodeEvent node;
    node.name = base_node_name(name);
    node.op_type = op_name;
    node.provider = provider;
    node.provider_class = classify_ort_provider(provider);
    node.category = cat;
    node.duration_us = get_number(ev, "dur");

    ++stats.total_node_events;
    switch (node.provider_class) {
      case OrtProviderClass::Cuda: ++stats.cuda_node_events; break;
      case OrtProviderClass::Cpu: ++stats.cpu_node_events; break;
      case OrtProviderClass::Other: ++stats.other_provider_events; break;
      case OrtProviderClass::Unknown: ++stats.unknown_provider_events; break;
    }
    if (!node.name.empty()) unique_names.insert(node.name);

    // A memcpy-family kernel_time event is both a host/device transfer and an
    // executed node; count it in both views so "which EP runs it" and "data
    // crossed EPs" never contradict each other (§5.4).
    if (looks_like_memcpy(cat, name, op_name)) {
      ++stats.transfer_events;
      if (!node.name.empty()) unique_memcpy.insert(node.name);
    }

    stats.node_events.push_back(std::move(node));
  }

  stats.unique_node_names = unique_names.size();
  stats.memcpy_names.assign(unique_memcpy.begin(), unique_memcpy.end());
  std::sort(stats.memcpy_names.begin(), stats.memcpy_names.end());
  return stats;
}

OrtProfileStats parse_ort_profile_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw VieError(ErrorCode::InvalidArgument, "cannot open ort profile file: " + path);
  }
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return parse_ort_profile_string(text);
}

}  // namespace vie
