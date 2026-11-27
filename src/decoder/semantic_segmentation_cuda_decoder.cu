#include "semantic_segmentation_cuda.hpp"

#include "vie/core/error.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace vie {
namespace {

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

// Exact device copy of sample_plane() in semantic_segmentation.cpp: the same
// clamp order, the same half-pixel coefficients, FP64 math with a float
// result. This translation unit is compiled with --fmad=false so the compiler
// cannot fuse the multiply-add pairs the CPU evaluates separately.
__device__ float sample_plane_device(const float* plane, int height, int width, double x, double y) {
  if (x < 0) x = 0;
  if (x > width - 1) x = width - 1;
  if (y < 0) y = 0;
  if (y > height - 1) y = height - 1;
  const int x0 = static_cast<int>(floor(x));
  const int y0 = static_cast<int>(floor(y));
  const int x1 = x0 + 1 < width ? x0 + 1 : x0;
  const int y1 = y0 + 1 < height ? y0 + 1 : y0;
  const double fx = x - x0;
  const double fy = y - y0;
  const float v00 = plane[static_cast<long long>(y0) * width + x0];
  const float v10 = plane[static_cast<long long>(y0) * width + x1];
  const float v01 = plane[static_cast<long long>(y1) * width + x0];
  const float v11 = plane[static_cast<long long>(y1) * width + x1];
  const double top = v00 * (1.0 - fx) + v10 * fx;
  const double bottom = v01 * (1.0 - fx) + v11 * fx;
  return static_cast<float>(top * (1.0 - fy) + bottom * fy);
}

// Grid-stride finite check over the raw logits. The first offending linear
// index is kept with atomicMin so the message names the same (c, y, x) the
// sequential CPU loop would have thrown on.
__global__ void validate_logits_kernel(const float* logits, long long count,
                                       unsigned long long* fault) {
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += stride) {
    if (!isfinite(logits[index])) {
      atomicMin(fault, static_cast<unsigned long long>(index));
    }
  }
}

__global__ void resize_logits_kernel(const float* src, int channels, int src_h, int src_w,
                                     float* dst, int dst_h, int dst_w) {
  const long long total = static_cast<long long>(channels) * dst_h * dst_w;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  const double sx = static_cast<double>(src_w) / dst_w;
  const double sy = static_cast<double>(src_h) / dst_h;
  for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += stride) {
    const int x = static_cast<int>(index % dst_w);
    const int y = static_cast<int>((index / dst_w) % dst_h);
    const int c = static_cast<int>(index / (static_cast<long long>(dst_w) * dst_h));
    const double source_x = (x + 0.5) * sx - 0.5;
    const double source_y = (y + 0.5) * sy - 0.5;
    const float* plane = src + static_cast<long long>(c) * src_h * src_w;
    dst[index] = sample_plane_device(plane, src_h, src_w, source_x, source_y);
  }
}

// One thread per ORIGINAL pixel. The coordinate recovery and the per-channel
// bilinear samples mirror Step B of the CPU decoder; the strict '>' keeps the
// smallest class id on a tie.
__global__ void argmax_kernel(const float* grid, int channels, int grid_h, int grid_w,
                              int original_width, int original_height, double scale_x,
                              double scale_y, int origin_x, int origin_y, int* class_ids) {
  const long long count = static_cast<long long>(original_width) * original_height;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += stride) {
    const int ox = static_cast<int>(index % original_width);
    const int oy = static_cast<int>(index / original_width);
    const double mx = ox * scale_x + origin_x;
    const double my = oy * scale_y + origin_y;
    int best_class = 0;
    float best_value = -INFINITY;
    for (int c = 0; c < channels; ++c) {
      const float* plane = grid + static_cast<long long>(c) * grid_h * grid_w;
      const float value = sample_plane_device(plane, grid_h, grid_w, mx, my);
      if (value > best_value) {
        best_value = value;
        best_class = c;
      }
    }
    class_ids[index] = best_class;
  }
}

}  // namespace

SemanticSegmentationCudaLaunch launch_semantic_segmentation_cuda(
    const float* logits, int channels, int logits_height, int logits_width,
    const GeometryTransform& geometry, const std::string& output_name, int device_id) {
  if (channels < 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation num_classes must be >= 1 in the CUDA decoder");
  }
  if (logits_height < 1 || logits_width < 1) {
    throw VieError(ErrorCode::InvalidArgument,
                   "semantic segmentation logits spatial size must be positive");
  }
  if (geometry.original_width < 1 || geometry.original_height < 1) {
    throw VieError(ErrorCode::InvalidArgument, "geometry original size must be positive");
  }
  if (geometry.output_width < 1 || geometry.output_height < 1) {
    throw VieError(ErrorCode::InvalidArgument, "geometry model-input size must be positive");
  }
  if (geometry.scale_x <= 0.0 || geometry.scale_y <= 0.0) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale must be positive");
  }
  if (static_cast<long long>(channels) * logits_height > LLONG_MAX / logits_width) {
    throw VieError(ErrorCode::InvalidArgument, "semantic segmentation logits size overflows");
  }
  const long long logits_count =
      static_cast<long long>(channels) * logits_height * logits_width;
  const long long class_count =
      static_cast<long long>(geometry.original_width) * geometry.original_height;

  (void)output_name;  // The CPU error wording for this protocol has no output name.

  cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
  // Legacy default stream. See SemanticSegmentationCudaDecoder.
  cudaStream_t stream = nullptr;

  DeviceBuffer fault_buffer(sizeof(unsigned long long));
  const unsigned long long no_fault = std::numeric_limits<unsigned long long>::max();
  cuda_check(cudaMemcpyAsync(fault_buffer.data, &no_fault, sizeof(unsigned long long),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync");

  EventPair validate_events;
  cuda_check(cudaEventRecord(validate_events.start, stream), "cudaEventRecord");
  constexpr int kThreads = 256;
  const int ctas = static_cast<int>(std::min<long long>(
      (logits_count + kThreads - 1) / kThreads, 4096));
  validate_logits_kernel<<<ctas, kThreads, 0, stream>>>(logits, logits_count,
                                                       static_cast<unsigned long long*>(fault_buffer.data));
  cuda_check(cudaGetLastError(), "semantic validate kernel");
  cuda_check(cudaEventRecord(validate_events.stop, stream), "cudaEventRecord");

  unsigned long long fault = no_fault;
  cuda_check(cudaMemcpyAsync(&fault, fault_buffer.data, sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  float validate_ms = 0.f;
  cuda_check(cudaEventElapsedTime(&validate_ms, validate_events.start, validate_events.stop),
             "cudaEventElapsedTime");
  if (fault != no_fault) {
    const long long index = static_cast<long long>(fault);
    const long long plane = static_cast<long long>(logits_height) * logits_width;
    const int c = static_cast<int>(index / plane);
    const long long rem = index % plane;
    const int y = static_cast<int>(rem / logits_width);
    const int x = static_cast<int>(rem % logits_width);
    float offending = 0.f;
    cuda_check(cudaMemcpy(&offending, logits + index, sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    const char* reason = std::isnan(offending) ? "NaN" : "infinite";
    throw VieError(ErrorCode::InvalidArgument, "semantic segmentation logit at [c=" +
                                                   std::to_string(c) + ", y=" + std::to_string(y) +
                                                   ", x=" + std::to_string(x) + "] is " + reason);
  }

  EventPair argmax_events;
  cuda_check(cudaEventRecord(argmax_events.start, stream), "cudaEventRecord");
  const int grid_h = geometry.output_height;
  const int grid_w = geometry.output_width;
  const int origin_x = geometry_model_origin_x(geometry);
  const int origin_y = geometry_model_origin_y(geometry);
  const float* grid = logits;
  // The resized plane (only when the model output is lower resolution than the
  // model input) has to stay alive until the class-map copy has completed.
  std::unique_ptr<DeviceBuffer> resized;
  if (grid_h != logits_height || grid_w != logits_width) {
    const std::size_t resized_bytes =
        static_cast<std::size_t>(channels) * grid_h * grid_w * sizeof(float);
    resized = std::make_unique<DeviceBuffer>(resized_bytes);
    const long long resized_count = static_cast<long long>(channels) * grid_h * grid_w;
    const int resize_ctas = static_cast<int>(
        std::min<long long>((resized_count + kThreads - 1) / kThreads, 4096));
    resize_logits_kernel<<<resize_ctas, kThreads, 0, stream>>>(
        logits, channels, logits_height, logits_width, static_cast<float*>(resized->data), grid_h,
        grid_w);
    cuda_check(cudaGetLastError(), "semantic resize kernel");
    grid = static_cast<const float*>(resized->data);
  }

  DeviceBuffer class_map(static_cast<std::size_t>(class_count) * sizeof(int));
  const int argmax_ctas = static_cast<int>(
      std::min<long long>((class_count + kThreads - 1) / kThreads, 4096));
  argmax_kernel<<<argmax_ctas, kThreads, 0, stream>>>(
      grid, channels, grid_h, grid_w, geometry.original_width, geometry.original_height,
      geometry.scale_x, geometry.scale_y, origin_x, origin_y, static_cast<int*>(class_map.data));
  cuda_check(cudaGetLastError(), "semantic argmax kernel");
  cuda_check(cudaEventRecord(argmax_events.stop, stream), "cudaEventRecord");

  SemanticSegmentationCudaLaunch launch;
  launch.class_ids.assign(static_cast<std::size_t>(class_count), 0);
  EventPair copy_events;
  cuda_check(cudaEventRecord(copy_events.start, stream), "cudaEventRecord");
  cuda_check(cudaMemcpyAsync(launch.class_ids.data(), class_map.data,
                             static_cast<std::size_t>(class_count) * sizeof(int),
                             cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaEventRecord(copy_events.stop, stream), "cudaEventRecord");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  float argmax_ms = 0.f;
  float copy_ms = 0.f;
  cuda_check(cudaEventElapsedTime(&argmax_ms, argmax_events.start, argmax_events.stop),
             "cudaEventElapsedTime");
  cuda_check(cudaEventElapsedTime(&copy_ms, copy_events.start, copy_events.stop),
             "cudaEventElapsedTime");
  launch.validate_ms = static_cast<double>(validate_ms);
  launch.argmax_ms = static_cast<double>(argmax_ms);
  launch.d2h_ms = static_cast<double>(copy_ms);
  launch.d2h_bytes = static_cast<std::size_t>(class_count) * sizeof(int);
  return launch;
}

}  // namespace vie