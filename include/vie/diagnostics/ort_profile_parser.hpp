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
// Node events are counted independently from transfer/memcpy events so the two
// are never conflated. unique_node_names counts distinct node-event names.
struct OrtProfileStats {
  std::size_t total_trace_events = 0;  // every traceEvents[] entry seen
  std::size_t total_node_events = 0;   // executed node events (carry a provider)
  std::size_t cuda_node_events = 0;
  std::size_t cpu_node_events = 0;
  std::size_t other_provider_events = 0;
  std::size_t unknown_provider_events = 0;  // node event without a provider tag
  std::size_t transfer_events = 0;          // memcpy-family placed nodes
  std::size_t unique_node_names = 0;
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
// string args.provider. Events lacking it (Session init, fences, thread metadata)
// are never counted as nodes; a memcpy-family node is tallied BOTH by its
// provider and as a transfer event. Throws VieError(InvalidArgument) on
// non-JSON or structurally invalid input.
OrtProfileStats parse_ort_profile_string(std::string_view json_text);
OrtProfileStats parse_ort_profile_file(const std::string& path);

}  // namespace vie
