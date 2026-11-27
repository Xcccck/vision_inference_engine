#include "pose_cuda.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace vie {
namespace {

constexpr float kScoreSlack = 1e-4f;
constexpr double kInsideSlack = 1e-3;
// Fault encoding: prediction in bits 63..32, kind in 31..24, channel in 23..0.
// atomicMin then names the same anchor the CPU loop would have thrown on.
constexpr unsigned long long kFaultKindShift = 24;
constexpr unsigned long long kFaultChannelMask = (1ULL << 24) - 1;
constexpr int kFaultNonFinite = 1;
constexpr int kFaultProbability = 2;
constexpr int kFaultBoxSize = 3;
constexpr int kFaultInverse = 4;
constexpr int kFaultGeometry = 5;
constexpr unsigned long long kNoFault = std::numeric_limits<unsigned long long>::max();

struct GpuPoseCandidate {
  float cx, cy, width, height;
  float score;
  int class_id;
  int anchor_index;
};

struct GpuPoseMapped {
  float x1, y1, x2, y2;
  float score;
  int class_id;
  int anchor_index;
};

struct GpuPoseKeypoint {
  float x, y, confidence;
  int inside_image;
};

struct GpuPoseObject {
  float cx, cy, width, height;
  float score;
  int class_id;
  int anchor_index;
};

static_assert(sizeof(GpuPoseCandidate) == 28, "pose candidate record gained padding");
static_assert(sizeof(GpuPoseMapped) == 28, "pose mapped record gained padding");
static_assert(sizeof(GpuPoseKeypoint) == 16, "pose keypoint record gained padding");
static_assert(sizeof(GpuPoseObject) == 28, "pose object record gained padding");

struct CandidateOrder {
  __host__ __device__ bool operator()(const GpuPoseCandidate& left, const GpuPoseCandidate& right) const {
    if (left.score != right.score) return left.score > right.score;
    if (left.anchor_index != right.anchor_index) return left.anchor_index < right.anchor_index;
    return left.class_id < right.class_id;
  }
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

__device__ float cuda_read(const float* data, int channel_axis, int channels, int predictions, int channel,
                           int prediction) {
  const int index = channel_axis == 1 ? channel * predictions + prediction : prediction * channels + channel;
  return data[index];
}

__device__ float cuda_clamp_probability(float value) {
  if (value < 0.f) return 0.f;
  if (value > 1.f) return 1.f;
  return value;
}

__device__ float cuda_map_axis(float value, int origin, double scale) {
  return static_cast<float>((static_cast<double>(value) - static_cast<double>(origin)) / scale);
}

// Mirrors detection_ops.cpp clampf: a value is clamped only when it compares
// outside the range, so the NaN path stays identical to the CPU decoder.
__device__ float cuda_clamp_coordinate(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

__device__ void record_fault(unsigned long long* fault, int index, int kind, int channel) {
  const unsigned long long key =
      (static_cast<unsigned long long>(static_cast<unsigned int>(index)) << 32) |
      (static_cast<unsigned long long>(kind) << kFaultKindShift) |
      (static_cast<unsigned long long>(static_cast<unsigned int>(channel)) & kFaultChannelMask);
  atomicMin(fault, key);
}

// One thread per anchor. It follows the CPU decoder's read order: every class
// score, then cx/cy/width/height, then the keypoints. The first defect returns
// immediately so the recorded fault is the one the CPU would throw on. Anchors
// whose best score is below the threshold stop before the box or keypoints are
// read, exactly like the CPU loop.
__global__ void filter_decode_kernel(const float* data, int channel_axis, int channels, int predictions,
                                     int num_classes, int keypoint_count, int keypoint_dimensions,
                                     int best_class, float threshold, GpuPoseCandidate* out, int* count,
                                     unsigned long long* fault) {
  const int prediction = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (prediction >= predictions) return;
  const int keypoint_base = 4 + num_classes;
  int hits = 0;
  int best_index = -1;
  float best_value = -1.f;
  for (int cls = 0; cls < num_classes; ++cls) {
    const int channel = 4 + cls;
    const float raw = cuda_read(data, channel_axis, channels, predictions, channel, prediction);
    if (!isfinite(raw)) {
      record_fault(fault, prediction, kFaultNonFinite, channel);
      return;
    }
    if (raw < -kScoreSlack || raw > 1.f + kScoreSlack) {
      record_fault(fault, prediction, kFaultProbability, channel);
      return;
    }
    const float score = cuda_clamp_probability(raw);
    if (best_class) {
      if (score > best_value || (score == best_value && cls < best_index)) {
        best_value = score;
        best_index = cls;
      }
    } else if (score >= threshold) {
      ++hits;
    }
  }
  if (best_class) {
    if (best_value >= threshold) hits = 1;
  }
  if (hits == 0) return;

  const float cx = cuda_read(data, channel_axis, channels, predictions, 0, prediction);
  const float cy = cuda_read(data, channel_axis, channels, predictions, 1, prediction);
  const float width = cuda_read(data, channel_axis, channels, predictions, 2, prediction);
  const float height = cuda_read(data, channel_axis, channels, predictions, 3, prediction);
  if (!isfinite(cx)) {
    record_fault(fault, prediction, kFaultNonFinite, 0);
    return;
  }
  if (!isfinite(cy)) {
    record_fault(fault, prediction, kFaultNonFinite, 1);
    return;
  }
  if (!isfinite(width)) {
    record_fault(fault, prediction, kFaultNonFinite, 2);
    return;
  }
  if (!isfinite(height)) {
    record_fault(fault, prediction, kFaultNonFinite, 3);
    return;
  }
  if (!(width > 0.f) || !(height > 0.f)) {
    record_fault(fault, prediction, kFaultBoxSize, 0);
    return;
  }
  for (int keypoint = 0; keypoint < keypoint_count; ++keypoint) {
    const int base = keypoint_base + keypoint * keypoint_dimensions;
    const float x = cuda_read(data, channel_axis, channels, predictions, base, prediction);
    const float y = cuda_read(data, channel_axis, channels, predictions, base + 1, prediction);
    const float confidence = cuda_read(data, channel_axis, channels, predictions, base + 2, prediction);
    if (!isfinite(x)) {
      record_fault(fault, prediction, kFaultNonFinite, base);
      return;
    }
    if (!isfinite(y)) {
      record_fault(fault, prediction, kFaultNonFinite, base + 1);
      return;
    }
    if (!isfinite(confidence)) {
      record_fault(fault, prediction, kFaultNonFinite, base + 2);
      return;
    }
    if (confidence < -kScoreSlack || confidence > 1.f + kScoreSlack) {
      record_fault(fault, prediction, kFaultProbability, base + 2);
      return;
    }
  }

  if (best_class) {
    const int slot = atomicAdd(count, 1);
    out[slot] = GpuPoseCandidate{cx, cy, width, height, best_value, best_index, prediction};
    return;
  }
  for (int cls = 0; cls < num_classes; ++cls) {
    const float raw = cuda_read(data, channel_axis, channels, predictions, 4 + cls, prediction);
    const float score = cuda_clamp_probability(raw);
    if (score >= threshold) {
      const int slot = atomicAdd(count, 1);
      out[slot] = GpuPoseCandidate{cx, cy, width, height, score, cls, prediction};
    }
  }
}

// One thread per sorted candidate. The box and keypoint inverse maps are the
// CPU decoder's double-precision expressions; a degenerate box is marked as
// dropped instead of failing. compact_kernel then preserves the sorted order
// the CPU loop would produce.
__global__ void map_kernel(const float* data, int channel_axis, int channels, int predictions,
                           int keypoint_base, int keypoint_count, int keypoint_dimensions,
                           const GpuPoseCandidate* input, int count, int origin_x, int origin_y,
                           double scale_x, double scale_y, int original_width, int original_height,
                           GpuPoseMapped* output, GpuPoseKeypoint* output_keypoints, int* keep,
                           unsigned long long* fault) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  keep[index] = 0;
  if (original_width <= 0 || original_height <= 0 || scale_x == 0.0 || scale_y == 0.0) {
    record_fault(fault, index, kFaultGeometry, 0);
    return;
  }
  const GpuPoseCandidate candidate = input[index];
  float x1 = candidate.cx - candidate.width * 0.5f;
  float y1 = candidate.cy - candidate.height * 0.5f;
  float x2 = candidate.cx + candidate.width * 0.5f;
  float y2 = candidate.cy + candidate.height * 0.5f;
  const float width = static_cast<float>(original_width);
  const float height = static_cast<float>(original_height);
  x1 = cuda_map_axis(x1, origin_x, scale_x);
  x2 = cuda_map_axis(x2, origin_x, scale_x);
  y1 = cuda_map_axis(y1, origin_y, scale_y);
  y2 = cuda_map_axis(y2, origin_y, scale_y);
  x1 = cuda_clamp_coordinate(x1, 0.f, width);
  x2 = cuda_clamp_coordinate(x2, 0.f, width);
  y1 = cuda_clamp_coordinate(y1, 0.f, height);
  y2 = cuda_clamp_coordinate(y2, 0.f, height);
  if (!(x2 > x1) || !(y2 > y1)) return;
  const std::size_t destination =
      static_cast<std::size_t>(index) * static_cast<std::size_t>(keypoint_count);
  for (int keypoint = 0; keypoint < keypoint_count; ++keypoint) {
    const int base = keypoint_base + keypoint * keypoint_dimensions;
    const int anchor = candidate.anchor_index;
    const float model_x = cuda_read(data, channel_axis, channels, predictions, base, anchor);
    const float model_y = cuda_read(data, channel_axis, channels, predictions, base + 1, anchor);
    const float confidence = cuda_read(data, channel_axis, channels, predictions, base + 2, anchor);
    const double mapped_x = (static_cast<double>(model_x) - static_cast<double>(origin_x)) / scale_x;
    const double mapped_y = (static_cast<double>(model_y) - static_cast<double>(origin_y)) / scale_y;
    if (!isfinite(mapped_x) || !isfinite(mapped_y)) {
      record_fault(fault, index, kFaultInverse, 0);
      return;
    }
    GpuPoseKeypoint point;
    point.x = static_cast<float>(mapped_x);
    point.y = static_cast<float>(mapped_y);
    point.confidence = cuda_clamp_probability(confidence);
    point.inside_image =
        mapped_x >= -kInsideSlack && mapped_x <= static_cast<double>(original_width) + kInsideSlack &&
                mapped_y >= -kInsideSlack && mapped_y <= static_cast<double>(original_height) + kInsideSlack
            ? 1
            : 0;
    output_keypoints[destination + static_cast<std::size_t>(keypoint)] = point;
  }
  GpuPoseMapped mapped;
  mapped.x1 = x1;
  mapped.y1 = y1;
  mapped.x2 = x2;
  mapped.y2 = y2;
  mapped.score = candidate.score;
  mapped.class_id = candidate.class_id;
  mapped.anchor_index = candidate.anchor_index;
  output[index] = mapped;
  keep[index] = 1;
}

// Single thread, compact only. Copies the surviving mapped records in sorted
// order, which is the order the CPU loop builds after its continue-on-degenerate
// pass. Every source byte was already written by map_kernel.
__global__ void compact_kernel(const GpuPoseMapped* input, const GpuPoseKeypoint* input_keypoints,
                               const int* keep, int count, int keypoint_count, GpuPoseMapped* output,
                               GpuPoseKeypoint* output_keypoints, int* kept) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int written = 0;
  for (int index = 0; index < count; ++index) {
    if (keep[index] == 0) continue;
    const std::size_t source =
        static_cast<std::size_t>(index) * static_cast<std::size_t>(keypoint_count);
    const std::size_t destination =
        static_cast<std::size_t>(written) * static_cast<std::size_t>(keypoint_count);
    for (int keypoint = 0; keypoint < keypoint_count; ++keypoint) {
      output_keypoints[destination + static_cast<std::size_t>(keypoint)] =
          input_keypoints[source + static_cast<std::size_t>(keypoint)];
    }
    output[written] = input[index];
    ++written;
  }
  *kept = written;
}

__device__ float cuda_iou(const GpuPoseMapped& a, const GpuPoseMapped& b) {
  const float left = fmaxf(a.x1, b.x1);
  const float top = fmaxf(a.y1, b.y1);
  const float right = fminf(a.x2, b.x2);
  const float bottom = fminf(a.y2, b.y2);
  const float width = right - left;
  const float height = bottom - top;
  if (width <= 0.f || height <= 0.f) return 0.f;
  const float intersection = width * height;
  const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  const float union_area = area_a + area_b - intersection;
  if (!(union_area > 0.f) || !isfinite(union_area)) return 0.f;
  return intersection / union_area;
}

// Same greedy loop as greedy_nms() in detection_ops.cpp: the sorted order is
// the priority, max_detections stops the loop, and IoU >= threshold suppresses.
__global__ void nms_kernel(const GpuPoseMapped* items, int count, float iou_threshold, int class_aware,
                           int max_detections, int* kept_indices, int* kept_count) {
  if (blockIdx.x != 0) return;
  extern __shared__ char suppressed[];
  __shared__ int kept;
  __shared__ int take;
  for (int i = threadIdx.x; i < count; i += blockDim.x) suppressed[i] = 0;
  if (threadIdx.x == 0) kept = 0;
  __syncthreads();
  for (int i = 0; i < count; ++i) {
    if (threadIdx.x == 0) take = suppressed[i] == 0 && kept < max_detections;
    __syncthreads();
    if (take && threadIdx.x == 0) {
      kept_indices[kept] = i;
      kept += 1;
    }
    __syncthreads();
    if (!take) continue;
    for (int j = i + 1 + threadIdx.x; j < count; j += blockDim.x) {
      if (suppressed[j] != 0) continue;
      if (class_aware && items[j].class_id != items[i].class_id) continue;
      if (cuda_iou(items[i], items[j]) >= iou_threshold) suppressed[j] = 1;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *kept_count = kept;
}

__global__ void select_prefix_kernel(int count, int max_detections, int* kept_indices, int* kept_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int limit = min(count, max_detections);
  for (int index = 0; index < limit; ++index) kept_indices[index] = index;
  *kept_count = limit;
}

// Gathers only the survivors into a contiguous payload. cx/cy/width/height use
// the same float expressions as the CPU result builder.
__global__ void pack_kernel(const GpuPoseMapped* mapped, const GpuPoseKeypoint* mapped_keypoints,
                            int keypoint_count, const int* kept_indices, int kept_count, GpuPoseObject* objects,
                            GpuPoseKeypoint* keypoints) {
  const int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (slot >= kept_count) return;
  const int index = kept_indices[slot];
  const GpuPoseMapped item = mapped[index];
  GpuPoseObject object;
  object.cx = (item.x1 + item.x2) * 0.5f;
  object.cy = (item.y1 + item.y2) * 0.5f;
  object.width = item.x2 - item.x1;
  object.height = item.y2 - item.y1;
  object.score = item.score;
  object.class_id = item.class_id;
  object.anchor_index = item.anchor_index;
  objects[slot] = object;
  const std::size_t source = static_cast<std::size_t>(index) * static_cast<std::size_t>(keypoint_count);
  const std::size_t destination = static_cast<std::size_t>(slot) * static_cast<std::size_t>(keypoint_count);
  for (int keypoint = 0; keypoint < keypoint_count; ++keypoint) {
    keypoints[destination + keypoint] = mapped_keypoints[source + keypoint];
  }
}

[[noreturn]] void throw_fault(unsigned long long fault) {
  const int kind = static_cast<int>((fault >> kFaultKindShift) & 0xFFULL);
  const long long index = static_cast<long long>(fault >> 32);
  const int channel = static_cast<int>(fault & kFaultChannelMask);
  switch (kind) {
    case kFaultNonFinite:
      throw VieError(ErrorCode::InvalidArgument,
                     "yolov8_pose_raw prediction " + std::to_string(index) + " channel " +
                         std::to_string(channel) + " is not finite");
    case kFaultProbability:
      throw VieError(ErrorCode::InvalidArgument,
                     "yolov8_pose_raw probability at prediction " + std::to_string(index) + " channel " +
                         std::to_string(channel) + " is outside [0, 1]");
    case kFaultBoxSize:
      throw VieError(ErrorCode::InvalidArgument,
                     "yolov8_pose_raw kept anchor " + std::to_string(index) + " has a non-positive box size");
    case kFaultInverse:
      throw VieError(ErrorCode::InvalidArgument, "keypoint inverse map is not finite");
    default:
      throw VieError(ErrorCode::InvalidArgument, "geometry scale or original size is invalid");
  }
}

float elapsed_ms(const EventPair& events, cudaStream_t stream) {
  cuda_check(cudaEventRecord(events.stop, stream), "cudaEventRecord");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  float elapsed = 0.f;
  cuda_check(cudaEventElapsedTime(&elapsed, events.start, events.stop), "cudaEventElapsedTime");
  return elapsed;
}

}  // namespace

PoseCudaLaunch launch_pose_cuda(const Tensor& tensor, const PoseDecodeConfig& config,
                                const GeometryTransform& geometry, int device_id) {
  const std::vector<std::int64_t>& shape = tensor.shape();
  const std::size_t axis = static_cast<std::size_t>(config.channel_axis);
  const std::int64_t channels64 = shape[axis];
  const std::int64_t predictions64 = shape[config.channel_axis == 1 ? 2 : 1];
  if (channels64 < 1 || channels64 > static_cast<std::int64_t>(kFaultChannelMask)) {
    throw VieError(ErrorCode::Unsupported,
                   "yolov8_pose_raw CUDA decoder channel count is outside the supported range");
  }
  if (predictions64 < 0 || predictions64 > 1000000) {
    throw VieError(ErrorCode::Unsupported,
                   "yolov8_pose_raw CUDA decoder prediction count is outside 0..1000000");
  }
  if (config.keypoint_count < 1 || config.keypoint_count > 10000) {
    throw VieError(ErrorCode::Unsupported,
                   "yolov8_pose_raw CUDA decoder keypoint count is outside 1..10000");
  }
  const int channels = static_cast<int>(channels64);
  const int predictions = static_cast<int>(predictions64);
  const bool best_class = *config.class_selection == ClassSelection::BestClass;
  const long long unlifted =
      best_class ? static_cast<long long>(predictions)
                 : static_cast<long long>(predictions) * static_cast<long long>(config.num_classes);
  if (unlifted > 1000000) {
    throw VieError(ErrorCode::Unsupported, "CUDA pose decoder candidate capacity exceeds 1000000");
  }
  const int capacity = static_cast<int>(unlifted);
  const int top_capacity = std::min(capacity, *config.pre_nms_top_k);
  const int keep_capacity = std::min(*config.max_detections, top_capacity);
  const std::size_t keypoint_bytes =
      static_cast<std::size_t>(std::max(top_capacity, 1)) * static_cast<std::size_t>(config.keypoint_count) *
      sizeof(GpuPoseKeypoint);
  if (keypoint_bytes > static_cast<std::size_t>(256) * 1024u * 1024u) {
    throw VieError(ErrorCode::Unsupported, "CUDA pose decoder keypoint buffer exceeds 256 MiB");
  }

  cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
  // Legacy default stream. See PoseCudaDecoder.
  cudaStream_t stream = nullptr;
  const float* data = reinterpret_cast<const float*>(tensor.device_data());

  DeviceBuffer candidates(static_cast<std::size_t>(std::max(capacity, 1)) * sizeof(GpuPoseCandidate));
  DeviceBuffer candidate_count(sizeof(int));
  DeviceBuffer fault_buffer(sizeof(unsigned long long));
  DeviceBuffer mapped(static_cast<std::size_t>(std::max(top_capacity, 1)) * sizeof(GpuPoseMapped));
  DeviceBuffer mapped_keypoints(keypoint_bytes);
  DeviceBuffer keep_flags(static_cast<std::size_t>(std::max(top_capacity, 1)) * sizeof(int));
  DeviceBuffer mapped_count(sizeof(int));
  DeviceBuffer map_fault(sizeof(unsigned long long));
  DeviceBuffer kept_indices(static_cast<std::size_t>(std::max(keep_capacity, 1)) * sizeof(int));
  DeviceBuffer kept_count(sizeof(int));
  DeviceBuffer objects(static_cast<std::size_t>(std::max(keep_capacity, 1)) * sizeof(GpuPoseObject));
  DeviceBuffer keypoints(static_cast<std::size_t>(std::max(keep_capacity, 1)) *
                         static_cast<std::size_t>(config.keypoint_count) * sizeof(GpuPoseKeypoint));

  cuda_check(cudaMemsetAsync(candidate_count.data, 0, sizeof(int), stream), "cudaMemsetAsync");
  cuda_check(cudaMemsetAsync(mapped_count.data, 0, sizeof(int), stream), "cudaMemsetAsync");
  cuda_check(cudaMemsetAsync(kept_count.data, 0, sizeof(int), stream), "cudaMemsetAsync");
  cuda_check(cudaMemcpyAsync(fault_buffer.data, &kNoFault, sizeof(unsigned long long), cudaMemcpyHostToDevice,
                             stream),
             "cudaMemcpyAsync");
  cuda_check(cudaMemcpyAsync(map_fault.data, &kNoFault, sizeof(unsigned long long), cudaMemcpyHostToDevice,
                             stream),
             "cudaMemcpyAsync");

  float validate_ms = 0.f;
  if (predictions > 0) {
    EventPair filter;
    cuda_check(cudaEventRecord(filter.start, stream), "cudaEventRecord");
    constexpr int kThreads = 128;
    const int blocks = (predictions + kThreads - 1) / kThreads;
    filter_decode_kernel<<<blocks, kThreads, 0, stream>>>(
        data, config.channel_axis, channels, predictions, config.num_classes, config.keypoint_count,
        config.keypoint_dimensions, best_class ? 1 : 0, *config.confidence_threshold,
        static_cast<GpuPoseCandidate*>(candidates.data), static_cast<int*>(candidate_count.data),
        static_cast<unsigned long long*>(fault_buffer.data));
    cuda_check(cudaGetLastError(), "pose filter kernel");
    validate_ms = elapsed_ms(filter, stream);
  }

  unsigned long long fault = kNoFault;
  int raw_count = 0;
  cuda_check(cudaMemcpyAsync(&fault, fault_buffer.data, sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                             stream),
             "cudaMemcpyAsync");
  cuda_check(cudaMemcpyAsync(&raw_count, candidate_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  if (fault != kNoFault) throw_fault(fault);
  if (raw_count < 0 || raw_count > capacity) {
    throw VieError(ErrorCode::Internal, "CUDA pose candidate count is invalid");
  }

  if (raw_count > 1) {
    EventPair sort_events;
    cuda_check(cudaEventRecord(sort_events.start, stream), "cudaEventRecord");
    thrust::sort(thrust::device_pointer_cast(static_cast<GpuPoseCandidate*>(candidates.data)),
                 thrust::device_pointer_cast(static_cast<GpuPoseCandidate*>(candidates.data) + raw_count),
                 CandidateOrder{});
    validate_ms += elapsed_ms(sort_events, stream);
  }

  const int top = std::min(raw_count, *config.pre_nms_top_k);
  int mapped_total = 0;
  float pose_decode_ms = 0.f;
  if (top > 0) {
    EventPair map_events;
    cuda_check(cudaEventRecord(map_events.start, stream), "cudaEventRecord");
    constexpr int kMapThreads = 128;
    const int map_blocks = (top + kMapThreads - 1) / kMapThreads;
    map_kernel<<<map_blocks, kMapThreads, 0, stream>>>(
        data, config.channel_axis, channels, predictions, 4 + config.num_classes, config.keypoint_count,
        config.keypoint_dimensions, static_cast<const GpuPoseCandidate*>(candidates.data), top,
        geometry_model_origin_x(geometry), geometry_model_origin_y(geometry), geometry.scale_x, geometry.scale_y,
        geometry.original_width, geometry.original_height, static_cast<GpuPoseMapped*>(mapped.data),
        static_cast<GpuPoseKeypoint*>(mapped_keypoints.data), static_cast<int*>(keep_flags.data),
        static_cast<unsigned long long*>(map_fault.data));
    cuda_check(cudaGetLastError(), "pose map kernel");
    compact_kernel<<<1, 1, 0, stream>>>(static_cast<const GpuPoseMapped*>(mapped.data),
                                        static_cast<const GpuPoseKeypoint*>(mapped_keypoints.data),
                                        static_cast<const int*>(keep_flags.data), top, config.keypoint_count,
                                        static_cast<GpuPoseMapped*>(mapped.data),
                                        static_cast<GpuPoseKeypoint*>(mapped_keypoints.data),
                                        static_cast<int*>(mapped_count.data));
    cuda_check(cudaGetLastError(), "pose compact kernel");
    pose_decode_ms = elapsed_ms(map_events, stream);
    unsigned long long map_fault_value = kNoFault;
    cuda_check(cudaMemcpyAsync(&map_fault_value, map_fault.data, sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaMemcpyAsync(&mapped_total, mapped_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    if (map_fault_value != kNoFault) throw_fault(map_fault_value);
    if (mapped_total < 0 || mapped_total > top) {
      throw VieError(ErrorCode::Internal, "CUDA pose mapped count is invalid");
    }
  }

  int kept_total = 0;
  float nms_ms = 0.f;
  if (mapped_total > 0) {
    EventPair nms_events;
    cuda_check(cudaEventRecord(nms_events.start, stream), "cudaEventRecord");
    if (*config.nms_enabled) {
      nms_kernel<<<1, 128, static_cast<std::size_t>(mapped_total), stream>>>(
          static_cast<const GpuPoseMapped*>(mapped.data), mapped_total, *config.iou_threshold,
          *config.class_aware ? 1 : 0, *config.max_detections, static_cast<int*>(kept_indices.data),
          static_cast<int*>(kept_count.data));
      cuda_check(cudaGetLastError(), "pose nms kernel");
    } else {
      select_prefix_kernel<<<1, 1, 0, stream>>>(mapped_total, *config.max_detections,
                                                static_cast<int*>(kept_indices.data),
                                                static_cast<int*>(kept_count.data));
      cuda_check(cudaGetLastError(), "pose select kernel");
    }
    nms_ms = elapsed_ms(nms_events, stream);
    cuda_check(cudaMemcpyAsync(&kept_total, kept_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    if (kept_total < 0 || kept_total > keep_capacity) {
      throw VieError(ErrorCode::Internal, "CUDA pose kept count is invalid");
    }
  }

  std::vector<GpuPoseObject> host_objects(static_cast<std::size_t>(std::max(kept_total, 0)));
  std::vector<GpuPoseKeypoint> host_keypoints(
      static_cast<std::size_t>(std::max(kept_total, 0)) * static_cast<std::size_t>(config.keypoint_count));
  float copy_ms = 0.f;
  if (kept_total > 0) {
    EventPair copy;
    cuda_check(cudaEventRecord(copy.start, stream), "cudaEventRecord");
    constexpr int kThreads = 128;
    const int blocks = (kept_total + kThreads - 1) / kThreads;
    pack_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const GpuPoseMapped*>(mapped.data), static_cast<const GpuPoseKeypoint*>(mapped_keypoints.data),
        config.keypoint_count, static_cast<const int*>(kept_indices.data), kept_total,
        static_cast<GpuPoseObject*>(objects.data), static_cast<GpuPoseKeypoint*>(keypoints.data));
    cuda_check(cudaGetLastError(), "pose pack kernel");
    cuda_check(cudaMemcpyAsync(host_objects.data(), objects.data,
                               static_cast<std::size_t>(kept_total) * sizeof(GpuPoseObject),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaMemcpyAsync(host_keypoints.data(), keypoints.data,
                               static_cast<std::size_t>(kept_total) *
                                   static_cast<std::size_t>(config.keypoint_count) * sizeof(GpuPoseKeypoint),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    copy_ms = elapsed_ms(copy, stream);
  }

  PoseCudaLaunch launch;
  launch.candidates_after_filter = raw_count;
  launch.candidates_before_nms = mapped_total;
  launch.validate_ms = static_cast<double>(validate_ms);
  launch.pose_decode_ms = static_cast<double>(pose_decode_ms);
  launch.nms_ms = static_cast<double>(nms_ms);
  launch.d2h_ms = static_cast<double>(copy_ms);
  launch.d2h_bytes = host_objects.size() * sizeof(GpuPoseObject) + host_keypoints.size() * sizeof(GpuPoseKeypoint);
  launch.objects.reserve(host_objects.size());
  for (std::size_t index = 0; index < host_objects.size(); ++index) {
    const GpuPoseObject& item = host_objects[index];
    PoseObject object;
    object.class_id = item.class_id;
    object.score = item.score;
    object.cx = item.cx;
    object.cy = item.cy;
    object.width = item.width;
    object.height = item.height;
    object.anchor_index = item.anchor_index;
    object.keypoints.reserve(static_cast<std::size_t>(config.keypoint_count));
    for (int keypoint = 0; keypoint < config.keypoint_count; ++keypoint) {
      const GpuPoseKeypoint& point =
          host_keypoints[index * static_cast<std::size_t>(config.keypoint_count) +
                         static_cast<std::size_t>(keypoint)];
      Keypoint converted;
      converted.x = point.x;
      converted.y = point.y;
      converted.confidence = point.confidence;
      converted.inside_image = point.inside_image != 0;
      object.keypoints.push_back(converted);
    }
    launch.objects.push_back(std::move(object));
  }
  return launch;
}

}  // namespace vie