// Phase 9.3.2.1: ORT profile parser unit tests.
//
// The parser is CUDA-free and ONNX-Runtime-free (it only reads a Chrome-trace
// JSON document), so every case here runs in ALL builds (A/B/C/D). The fixtures
// below mirror the REAL ONNX Runtime 1.20.1 EndProfiling output captured from a
// live CUDA session: a bare top-level JSON array, each executed node logged as
// *_fence_before / *_kernel_time / *_fence_after, and args.provider present ONLY
// on the *_kernel_time event. The CUDA integration tests in
// test_execution_scheduling.cpp then exercise the parser against real traces.

#include "vie/core/error.hpp"
#include "vie/diagnostics/ort_profile_parser.hpp"

#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

using namespace vie;

namespace {

// Builds a bare top-level JSON array of events (the real ORT 1.20.1 shape).
std::string array_trace(std::initializer_list<const char*> events) {
  std::string out = "[";
  bool first = true;
  for (const char* ev : events) {
    if (!first) out += ",";
    out += ev;
    first = false;
  }
  out += "]";
  return out;
}

// A real *_kernel_time node event: the ONLY variant ORT tags with a provider.
const char* kCudaConv =
    R"({"cat":"Node","name":"/model.0/conv/Conv_kernel_time","ph":"X","dur":235147,"args":{"op_name":"Conv","provider":"CUDAExecutionProvider","node_index":1}})";
const char* kCpuResize =
    R"({"cat":"Node","name":"/model.10/Resize_kernel_time","ph":"X","dur":9120,"args":{"op_name":"Resize","provider":"CPUExecutionProvider","node_index":140}})";
const char* kMemcpyToHost =
    R"({"cat":"Node","name":"Memcpy_token_2_kernel_time","ph":"X","dur":640,"args":{"op_name":"MemcpyToHost","provider":"CUDAExecutionProvider","node_index":180}})";
// Fence events carry op_name but NO provider -> must never be counted as nodes.
const char* kFence =
    R"({"cat":"Node","name":"/model.0/conv/Conv_fence_before","ph":"X","dur":0,"args":{"op_name":"Conv"}})";
const char* kSession =
    R"({"cat":"Session","name":"session_initialization","ph":"X","dur":111650,"args":{}})";

}  // namespace

TEST(OrtProfileParserTest, AcceptsBareArrayRoot) {
  auto stats = parse_ort_profile_string(array_trace({kCudaConv}));
  EXPECT_EQ(stats.total_node_events, 1u);
  EXPECT_EQ(stats.cuda_node_events, 1u);
}

TEST(OrtProfileParserTest, AcceptsObjectTraceEventsRoot) {
  // Older / tooling variants wrap the list; both shapes must parse identically.
  const std::string wrapped = "{\"traceEvents\":" + array_trace({kCudaConv}) + "}";
  auto stats = parse_ort_profile_string(wrapped);
  EXPECT_EQ(stats.total_node_events, 1u);
  EXPECT_EQ(stats.cuda_node_events, 1u);
}

TEST(OrtProfileParserTest, ClassifiesCudaNodeEvent) {
  auto stats = parse_ort_profile_string(array_trace({kCudaConv}));
  EXPECT_EQ(stats.total_node_events, 1u);
  EXPECT_EQ(stats.cuda_node_events, 1u);
  EXPECT_EQ(stats.cpu_node_events, 0u);
  EXPECT_EQ(stats.unknown_provider_events, 0u);
  ASSERT_EQ(stats.node_events.size(), 1u);
  EXPECT_EQ(stats.node_events[0].provider_class, OrtProviderClass::Cuda);
  EXPECT_EQ(stats.node_events[0].op_type, "Conv");
  // The *_kernel_time suffix is stripped so the name is the graph node.
  EXPECT_EQ(stats.node_events[0].name, "/model.0/conv/Conv");
  EXPECT_DOUBLE_EQ(stats.node_events[0].duration_us, 235147.0);
}

TEST(OrtProfileParserTest, ClassifiesCpuNodeEvent) {
  auto stats = parse_ort_profile_string(array_trace({kCpuResize}));
  EXPECT_EQ(stats.total_node_events, 1u);
  EXPECT_EQ(stats.cpu_node_events, 1u);
  EXPECT_EQ(stats.cuda_node_events, 0u);
  EXPECT_EQ(stats.node_events[0].name, "/model.10/Resize");
}

TEST(OrtProfileParserTest, EmptyProviderIsUnknownNotCpu) {
  // A kernel_time event whose provider is an empty string must be Unknown, never
  // silently folded into CPU (§5.2 / A7).
  auto stats = parse_ort_profile_string(array_trace({
      R"({"cat":"Node","name":"/x/Op_kernel_time","ph":"X","args":{"op_name":"SomeOp","provider":""}})",
  }));
  EXPECT_EQ(stats.total_node_events, 1u);
  EXPECT_EQ(stats.unknown_provider_events, 1u);
  EXPECT_EQ(stats.cpu_node_events, 0u);
  EXPECT_EQ(stats.cuda_node_events, 0u);
}

TEST(OrtProfileParserTest, OtherProviderClassified) {
  auto stats = parse_ort_profile_string(array_trace({
      R"({"cat":"Node","name":"/t/Op_kernel_time","ph":"X","args":{"op_name":"Foo","provider":"TensorrtExecutionProvider"}})",
  }));
  EXPECT_EQ(stats.total_node_events, 1u);
  EXPECT_EQ(stats.other_provider_events, 1u);
  EXPECT_EQ(stats.node_events[0].provider_class, OrtProviderClass::Other);
}

TEST(OrtProfileParserTest, NonNodeEventsAreIgnored) {
  // Session events and *_fence_* markers carry no args.provider and must not be
  // mistaken for executed model nodes, even though fences look node-like (§5.1).
  auto stats = parse_ort_profile_string(array_trace({kSession, kFence}));
  EXPECT_EQ(stats.total_trace_events, 2u);
  EXPECT_EQ(stats.total_node_events, 0u);
  EXPECT_EQ(stats.cuda_node_events, 0u);
}

TEST(OrtProfileParserTest, MemcpyKernelIsTransferAndCudaNode) {
  // The real ORT 1.20.1 case: Memcpy*Host kernels are tagged CUDAExecutionProvider
  // and represent host/device transfer; they count as a CUDA node AND a transfer.
  auto stats = parse_ort_profile_string(array_trace({kMemcpyToHost, kFence}));
  EXPECT_EQ(stats.total_node_events, 1u);  // the fence is not a node
  EXPECT_EQ(stats.cuda_node_events, 1u);
  EXPECT_EQ(stats.transfer_events, 1u);
  ASSERT_EQ(stats.memcpy_names.size(), 1u);
  EXPECT_EQ(stats.memcpy_names[0], "Memcpy_token_2");
}

TEST(OrtProfileParserTest, MemcpyWithoutProviderCountsNothing) {
  // A bare fence-style memcpy (no provider) is not a placed node nor a transfer.
  auto stats = parse_ort_profile_string(array_trace({
      R"({"cat":"Node","name":"Memcpy_fence_after","ph":"X","dur":0,"args":{"op_name":"MemcpyFromHost"}})",
  }));
  EXPECT_EQ(stats.total_node_events, 0u);
  EXPECT_EQ(stats.transfer_events, 0u);
}

TEST(OrtProfileParserTest, MissingArgsIsNotANode) {
  auto stats = parse_ort_profile_string(array_trace({
      R"({"cat":"Node","name":"/a/Op_kernel_time","ph":"X","dur":10})",
  }));
  EXPECT_EQ(stats.total_trace_events, 1u);
  EXPECT_EQ(stats.total_node_events, 0u);
}

TEST(OrtProfileParserTest, EmptyProfileYieldsZeroStats) {
  auto empty_array = parse_ort_profile_string("[]");
  EXPECT_EQ(empty_array.total_trace_events, 0u);
  EXPECT_EQ(empty_array.total_node_events, 0u);
  auto empty_obj = parse_ort_profile_string("{}");
  EXPECT_EQ(empty_obj.total_trace_events, 0u);
}

TEST(OrtProfileParserTest, InvalidJsonThrows) {
  EXPECT_THROW(parse_ort_profile_string("this is not json"), VieError);
  EXPECT_THROW(parse_ort_profile_string("[1, "), VieError);   // truncated array
  EXPECT_THROW(parse_ort_profile_string("42"), VieError);      // scalar root
  EXPECT_THROW(parse_ort_profile_string("{\"traceEvents\":123}"), VieError);
}

TEST(OrtProfileParserTest, RepeatedNodeCountsEventsNotUniques) {
  // The same node executed across several runs: total_node_events counts every
  // kernel_time event, unique_node_names counts the distinct node once (§5.3).
  auto stats = parse_ort_profile_string(array_trace({kCudaConv, kCudaConv, kCudaConv}));
  EXPECT_EQ(stats.total_node_events, 3u);
  EXPECT_EQ(stats.cuda_node_events, 3u);
  EXPECT_EQ(stats.unique_node_names, 1u);
}

TEST(OrtProfileParserTest, IncompleteEventFieldsAreTolerated) {
  // Events missing dur / op_name must still parse; duration defaults to zero.
  auto stats = parse_ort_profile_string(array_trace({
      R"({"cat":"Node","name":"/p/Conv_kernel_time","args":{"provider":"CUDAExecutionProvider"}})",
      R"({"cat":"Node","name":"/p/Conv_kernel_time","dur":"nan","args":{"provider":"CUDAExecutionProvider","op_name":5}})",
  }));
  EXPECT_EQ(stats.total_node_events, 2u);
  EXPECT_EQ(stats.cuda_node_events, 2u);
  EXPECT_DOUBLE_EQ(stats.node_events[0].duration_us, 0.0);
}

TEST(OrtProfileParserTest, ProviderClassificationIsCaseInsensitive) {
  EXPECT_EQ(classify_ort_provider("CUDAExecutionProvider"), OrtProviderClass::Cuda);
  EXPECT_EQ(classify_ort_provider("cuda"), OrtProviderClass::Cuda);
  EXPECT_EQ(classify_ort_provider("CPUExecutionProvider"), OrtProviderClass::Cpu);
  EXPECT_EQ(classify_ort_provider(""), OrtProviderClass::Unknown);
  EXPECT_EQ(classify_ort_provider("DnnlExecutionProvider"), OrtProviderClass::Other);
}

TEST(OrtProfileParserTest, RealisticMixedTraceMatchesYolov8Shape) {
  // A miniature of the captured YOLOv8n trace: many CUDA kernels, 4 CUDA-attributed
  // memcpy transfers, and 2 CPU Resize nodes; fences and Session excluded.
  auto stats = parse_ort_profile_string(array_trace({
      kSession,
      kFence, kCudaConv,
      kMemcpyToHost, R"({"cat":"Node","name":"Memcpy_kernel_time","ph":"X","dur":512,"args":{"op_name":"MemcpyFromHost","provider":"CUDAExecutionProvider"}})",
      kCpuResize, R"({"cat":"Node","name":"/model.13/Resize_kernel_time","ph":"X","dur":8000,"args":{"op_name":"Resize","provider":"CPUExecutionProvider"}})",
  }));
  EXPECT_EQ(stats.cuda_node_events, 3u);  // Conv + 2 memcpy
  EXPECT_EQ(stats.cpu_node_events, 2u);   // two Resize
  EXPECT_EQ(stats.transfer_events, 2u);   // two memcpy
  EXPECT_EQ(stats.unknown_provider_events, 0u);
  EXPECT_EQ(stats.total_node_events, 5u);
}

TEST(OrtProfileParserTest, ParseFileReadsFromDisk) {
  const std::string path = "/tmp/vie_ort_profile_roundtrip.json";
  {
    std::ofstream out(path);
    out << array_trace({kCudaConv});
  }
  auto stats = parse_ort_profile_file(path);
  EXPECT_EQ(stats.cuda_node_events, 1u);
  std::remove(path.c_str());
}

TEST(OrtProfileParserTest, ParseMissingFileThrows) {
  EXPECT_THROW(parse_ort_profile_file("/tmp/vie_does_not_exist_profile.json"), VieError);
}
