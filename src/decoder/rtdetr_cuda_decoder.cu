#include "rtdetr_cuda.hpp"

#include "vie/decoder/rtdetr_decoder.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <climits>

namespace vie {
namespace {

struct GpuKept {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int class_id;
  int query_index;
};

static_assert(sizeof(GpuKept) == 28, "kept-detection record gained padding");

struct KeptOrder {
  __host__ __device__ bool operator()(const GpuKept& left, const GpuKept& right) const {
    if (left.score != right.score) return left.score > right.score;
    return left.query_index < right.query_index;
  }
};

struct Counters {
  int fault;
  int passed;
  int kept;
};

void cuda_check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  throw VieError(ErrorCode::Internal, std::string(operation) + " failed: " + cudaGetErrorString(status));
}

struct DeviceBuffer {
  void* data = nullptr;
  explicit DeviceBuffer(std::size_t bytes) {
    if (bytes == 0) return;
    cuda_check(cudaMalloc(&data, bytes), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (data != nullptr) cudaFree(data);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct EventPair {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  EventPair() {
    cuda_check(cudaEventCreate(&start), "cudaEventCreate");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate");
  }
  ~EventPair() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
  }
  EventPair(const EventPair&) = delete;
  EventPair& operator=(const EventPair&) = delete;
};

__device__ float map_axis(float value, int pad, double scale) {
  return static_cast<float>((static_cast<double>(value) - static_cast<double>(pad)) / scale);
}

__device__ float clip_axis(float value, float limit) {
  if (value < 0.f) return 0.f;
  if (value > limit) return limit;
  return value;
}

// One thread per query. fault starts at INT_MAX. The first non-finite value
// in a query is stored as one signed integer, (query << 3) | reason, by a
// single atomicMin. reason is 1..5 in CPU check order: cx, cy, width, height,
// class score. Those three bits cannot spill into the query. Two separate
// atomicMin calls, one for the query and one for the field, could pair the
// earliest query with a field that belongs to a later query. This encoding
// cannot: the stored value is always one complete location, and the minimum
// is the earliest query, then the earliest field inside that query.
__global__ void decode_queries(const float* data, int queries, int classes, float threshold, float model_width,
                               float model_height, int pad_left, int pad_top, double scale_x, double scale_y,
                               float original_width, float original_height, GpuKept* kept, Counters* counters) {
  const int query = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (query >= queries) return;
  const float* row = data + static_cast<std::size_t>(query) * static_cast<std::size_t>(4 + classes);
  int reason = 0;
  if (!isfinite(row[0])) reason = 1;
  else if (!isfinite(row[1])) reason = 2;
  else if (!isfinite(row[2])) reason = 3;
  else if (!isfinite(row[3])) reason = 4;
  else {
    for (int cls = 0; cls < classes; ++cls) {
      if (!isfinite(row[4 + cls])) {
        reason = 5;
        break;
      }
    }
  }
  if (reason != 0) {
    atomicMin(&counters->fault, (query << 3) | reason);
    return;
  }
  int best_class = 0;
  float best_score = row[4];
  for (int cls = 1; cls < classes; ++cls) {
    const float score = row[4 + cls];
    if (score > best_score || (score == best_score && cls < best_class)) {
      best_score = score;
      best_class = cls;
    }
  }
  if (best_score < threshold) return;
  atomicAdd(&counters->passed, 1);
  const float cx = row[0] * model_width;
  const float cy = row[1] * model_height;
  const float width = row[2] * model_width;
  const float height = row[3] * model_height;
  GpuKept item;
  item.x1 = clip_axis(map_axis(cx - width * 0.5f, pad_left, scale_x), original_width);
  item.x2 = clip_axis(map_axis(cx + width * 0.5f, pad_left, scale_x), original_width);
  item.y1 = clip_axis(map_axis(cy - height * 0.5f, pad_top, scale_y), original_height);
  item.y2 = clip_axis(map_axis(cy + height * 0.5f, pad_top, scale_y), original_height);
  if (!(item.x2 > item.x1) || !(item.y2 > item.y1)) return;
  item.score = best_score;
  item.class_id = best_class;
  item.query_index = query;
  const int slot = atomicAdd(&counters->kept, 1);
  kept[slot] = item;
}

const char* fault_field(int reason) {
  switch (reason) {
    case 1:
      return "normalized cx";
    case 2:
      return "normalized cy";
    case 3:
      return "normalized width";
    case 4:
      return "normalized height";
    default:
      return "class score";
  }
}

}  // namespace

RtdetrCudaLaunch launch_rtdetr_cuda(const float* data, int queries, int num_classes, float threshold,
                                    int max_detections, int device_id, const GeometryTransform& geometry) {
  if (queries < 1 || queries > (INT_MAX >> 3)) {
    throw VieError(ErrorCode::InvalidArgument, "rtdetr CUDA decoder query count is out of range");
  }
  cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
  // Legacy default stream. See RtdetrCudaDecoder.
  cudaStream_t stream = nullptr;
  DeviceBuffer kept(static_cast<std::size_t>(queries) * sizeof(GpuKept));
  DeviceBuffer counters_buf(sizeof(Counters));
  Counters initial{INT_MAX, 0, 0};
  cuda_check(cudaMemcpyAsync(counters_buf.data, &initial, sizeof(Counters), cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync");

  EventPair kernel;
  EventPair sort_events;
  EventPair copy;
  cuda_check(cudaEventRecord(kernel.start, stream), "cudaEventRecord");
  const int threads = 128;
  const int blocks = (queries + threads - 1) / threads;
  decode_queries<<<blocks, threads, 0, stream>>>(
      data, queries, num_classes, threshold, static_cast<float>(geometry.output_width),
      static_cast<float>(geometry.output_height), geometry.pad_left, geometry.pad_top, geometry.scale_x,
      geometry.scale_y, static_cast<float>(geometry.original_width), static_cast<float>(geometry.original_height),
      static_cast<GpuKept*>(kept.data), static_cast<Counters*>(counters_buf.data));
  cuda_check(cudaGetLastError(), "rtdetr decode kernel");
  cuda_check(cudaEventRecord(kernel.stop, stream), "cudaEventRecord");

  Counters counters{};
  cuda_check(cudaMemcpyAsync(&counters, counters_buf.data, sizeof(Counters), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  if (counters.fault != INT_MAX) {
    throw VieError(ErrorCode::InvalidArgument,
                   rtdetr_nonfinite_message(fault_field(counters.fault & 7), counters.fault >> 3));
  }
  if (counters.kept < 0 || counters.kept > queries || counters.passed < counters.kept) {
    throw VieError(ErrorCode::Internal, "rtdetr CUDA decoder candidate count is invalid");
  }

  float sort_ms = 0.f;
  if (counters.kept > 1) {
    cuda_check(cudaEventRecord(sort_events.start, stream), "cudaEventRecord");
    thrust::sort(thrust::device_pointer_cast(static_cast<GpuKept*>(kept.data)),
                 thrust::device_pointer_cast(static_cast<GpuKept*>(kept.data) + counters.kept), KeptOrder{});
    cuda_check(cudaEventRecord(sort_events.stop, stream), "cudaEventRecord");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    cuda_check(cudaEventElapsedTime(&sort_ms, sort_events.start, sort_events.stop), "cudaEventElapsedTime");
  }

  const int copied = counters.kept < max_detections ? counters.kept : max_detections;
  std::vector<GpuKept> host(static_cast<std::size_t>(copied));
  float copy_ms = 0.f;
  if (copied > 0) {
    cuda_check(cudaEventRecord(copy.start, stream), "cudaEventRecord");
    cuda_check(cudaMemcpyAsync(host.data(), kept.data, static_cast<std::size_t>(copied) * sizeof(GpuKept),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaEventRecord(copy.stop, stream), "cudaEventRecord");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    cuda_check(cudaEventElapsedTime(&copy_ms, copy.start, copy.stop), "cudaEventElapsedTime");
  }
  float kernel_ms = 0.f;
  cuda_check(cudaEventElapsedTime(&kernel_ms, kernel.start, kernel.stop), "cudaEventElapsedTime");

  RtdetrCudaLaunch launch;
  launch.candidates_after_filter = counters.passed;
  launch.candidates_before_nms = counters.kept;
  launch.kernel_ms = kernel_ms;
  launch.sort_ms = sort_ms;
  launch.d2h_ms = copy_ms;
  launch.d2h_bytes = static_cast<std::size_t>(copied) * sizeof(GpuKept);
  launch.detections.reserve(host.size());
  for (const GpuKept& item : host) {
    Detection detection;
    detection.box = BoundingBox{item.x1, item.y1, item.x2, item.y2};
    detection.confidence = item.score;
    detection.class_id = item.class_id;
    detection.query_index = item.query_index;
    launch.detections.push_back(detection);
  }
  return launch;
}

}  // namespace vie
