#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace vie {

// Coarse classification of an ONNX Runtime Execution Provider string as it
// appears in a profile node event's args.provider field. Unknown means the
// event carried no provider (we never silently fold it into CPU/CUDA).
enum class OrtProviderClass { Cuda, Cpu, Other, Unknown };

// One executed ONNX node observed in a profile trace. name is the graph node
// (op instance) name; op_type is args.op_name when present.
struct OrtNodeEvent {
  std::string name;
  std::string op_type;
  std::string provider;  // raw args.provider ("" when absent/unclassifiable)
  OrtProviderClass provider_class = OrtProviderClass::Unknown;
  std::string category;  // event "cat"
  double duration_us = 0.0;
};

// Aggregate node-level provider statistics from an ORT Chrome-trace profile.
// Two independent views are kept so nothing is conflated (§5.4):
//   * category_* counts RAW trace events by their "cat" field (Node/Session/
//     ThreadPool/Memcpy/Other/Unknown) -- this shows that raw Node-cat events
//     (fences + kernel markers) far exceed the number of executed nodes;
//   * the provider view counts EXECUTED nodes, i.e. the *_kernel_time events
//     that carry args.provider, which is the authoritative EP placement.
// unique_node_names counts distinct executed-node names. `incomplete` is set
// when an executed node could not be attributed to any provider (§5.5). The
// original_onnx_nodes fields are NOT derived by the parser (an ORT profile has
// no original graph size); a caller/CLI may fill them, and `known` guards the
// display so we never fabricate the original node count (§5.4, §6).
struct OrtProfileStats {
  std::size_t total_trace_events = 0;  // every traceEvents[] entry seen
  std::size_t total_node_events = 0;   // executed node events (carry a provider)
  std::size_t cuda_node_events = 0;
  std::size_t cpu_node_events = 0;
  std::size_t other_provider_events = 0;
  std::size_t unknown_provider_events = 0;  // executed node with no usable provider
  std::size_t transfer_events = 0;          // memcpy-family placed nodes
  std::size_t unique_node_names = 0;
  // Raw event classification by "cat" (independent of the provider analysis).
  std::size_t category_node_events = 0;
  std::size_t category_session_events = 0;
  std::size_t category_threadpool_events = 0;
  std::size_t category_memcpy_events = 0;
  std::size_t category_other_events = 0;
  std::size_t category_unknown_events = 0;   // no "cat" field at all
  // Completeness: true when an executed node could not be attributed (§5.5).
  bool incomplete = false;
  // Optional, caller-supplied (parser leaves known=false).
  std::size_t original_onnx_nodes = 0;
  bool original_onnx_nodes_known = false;
  std::vector<OrtNodeEvent> node_events;    // all classified node events
  std::vector<std::string> memcpy_names;    // distinct memcpy event names
};

// Classify a raw provider string. Contains "CUDA" -> Cuda, "CPU" -> Cpu, empty
// -> Unknown, anything else -> Other. Never infers a provider from absence.
OrtProviderClass classify_ort_provider(const std::string& provider);

// Parses an ONNX Runtime Chrome-trace profiling document. ORT 1.20.1 emits a
// bare top-level array of events (a {"traceEvents": [...]} wrapper is also
// accepted). Each executed node appears as *_fence_before / *_kernel_time /
// *_fence_after, and ONLY the *_kernel_time event carries args.provider: that
// tag is the authoritative EP placement, so a node event is any event with a
// string args.provider. Fence markers and non-node events (Session init, thread
// metadata) are never counted as executed nodes. A Node-category event that has
// neither a provider nor a kernel/fence suffix is kept as an Unknown-provider
// executed node and marks the profile incomplete -- we never guess its EP
// (§5.2, A7). A memcpy-family node is tallied BOTH by its provider and as a
// transfer event. Throws VieError(InvalidArgument) on non-JSON or structurally
// invalid input.
OrtProfileStats parse_ort_profile_string(std::string_view json_text);
OrtProfileStats parse_ort_profile_file(const std::string& path);

}  // namespace vie
