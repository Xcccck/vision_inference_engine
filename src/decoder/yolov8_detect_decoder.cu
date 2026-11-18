#include "yolov8_cuda.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace vie {
namespace {

struct GpuCandidate {
  float x1, y1, x2, y2;
  float score;
  int class_id;
  int index;
  float model_x1, model_y1, model_x2, model_y2;
  int prediction_index;
};

struct CandidateLess {
  __host__ __device__ bool operator()(const GpuCandidate& a, const GpuCandidate& b) const {
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;
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

__device__ float cuda_sigmoid(float value) {
  if (value >= 0.f) return 1.f / (1.f + expf(-value));
  const float exponent = expf(value);
  return exponent / (1.f + exponent);
}

__device__ float cuda_activate(float value, int probabilities) {
  if (!isfinite(value)) return nanf("");
  if (probabilities) return value;
  return cuda_sigmoid(value);
}

__device__ float cuda_read(const float* data, int channel_axis, int channels, int predictions, int channel,
                           int prediction) {
  const int index = channel_axis == 1 ? channel * predictions + prediction : prediction * channels + channel;
  return data[index];
}

__device__ float cuda_dfl(const float* data, int channel_axis, int channels, int predictions, int side,
                          int reg_max, int prediction) {
  float max_logit = -1e30f;
  for (int bin = 0; bin < reg_max; ++bin) {
    const float logit =
        cuda_read(data, channel_axis, channels, predictions, side * reg_max + bin, prediction);
    if (!isfinite(logit)) return nanf("");
    max_logit = fmaxf(max_logit, logit);
  }
  float sum = 0.f;
  for (int bin = 0; bin < reg_max; ++bin) {
    const float logit =
        cuda_read(data, channel_axis, channels, predictions, side * reg_max + bin, prediction);
    sum += expf(logit - max_logit);
  }
  float expectation = 0.f;
  for (int bin = 0; bin < reg_max; ++bin) {
    const float logit =
        cuda_read(data, channel_axis, channels, predictions, side * reg_max + bin, prediction);
    expectation += static_cast<float>(bin) * (expf(logit - max_logit) / sum);
  }
  return expectation;
}

__device__ float cuda_iou(const GpuCandidate& a, const GpuCandidate& b) {
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
  if (!(union_area > 0.f)) return 0.f;
  return intersection / union_area;
}

__global__ void emit_kernel(const float* data, int channel_axis, int channels, int predictions, int box_channels,
                            int num_classes, int decoded_cxcywh, int reg_max, int probabilities, int best_class,
                            float threshold, const int* level_stride, const int* level_width, const int* level_height,
                            int level_count, GpuCandidate* out, int* count, int* fault) {
  const int prediction = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (prediction >= predictions) return;
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;
  if (decoded_cxcywh) {
    const float cx = cuda_read(data, channel_axis, channels, predictions, 0, prediction);
    const float cy = cuda_read(data, channel_axis, channels, predictions, 1, prediction);
    const float width = cuda_read(data, channel_axis, channels, predictions, 2, prediction);
    const float height = cuda_read(data, channel_axis, channels, predictions, 3, prediction);
    if (!isfinite(cx) || !isfinite(cy) || !isfinite(width) || !isfinite(height)) {
      atomicExch(fault, 1);
      return;
    }
    if (width <= 0.f || height <= 0.f) return;
    x1 = cx - width * 0.5f;
    y1 = cy - height * 0.5f;
    x2 = cx + width * 0.5f;
    y2 = cy + height * 0.5f;
  } else {
    float sides[4];
    for (int side = 0; side < 4; ++side) {
      sides[side] = cuda_dfl(data, channel_axis, channels, predictions, side, reg_max, prediction);
      if (!isfinite(sides[side])) {
        atomicExch(fault, 1);
        return;
      }
    }
    int offset = 0;
    int stride = 0;
    int local = -1;
    int grid_width = 1;
    for (int level = 0; level < level_count; ++level) {
      const int cells = level_width[level] * level_height[level];
      if (prediction < offset + cells) {
        local = prediction - offset;
        stride = level_stride[level];
        grid_width = level_width[level];
        break;
      }
      offset += cells;
    }
    if (local < 0) return;
    const float anchor_x = static_cast<float>(local % grid_width) + 0.5f;
    const float anchor_y = static_cast<float>(local / grid_width) + 0.5f;
    x1 = (anchor_x - sides[0]) * static_cast<float>(stride);
    y1 = (anchor_y - sides[1]) * static_cast<float>(stride);
    x2 = (anchor_x + sides[2]) * static_cast<float>(stride);
    y2 = (anchor_y + sides[3]) * static_cast<float>(stride);
    if (!(x2 > x1) || !(y2 > y1)) return;
  }
  if (best_class) {
    int chosen = 0;
    float best = -1.f;
    for (int cls = 0; cls < num_classes; ++cls) {
      const float raw = cuda_read(data, channel_axis, channels, predictions, box_channels + cls, prediction);
      const float score = cuda_activate(raw, probabilities);
      if (!isfinite(score) || (probabilities && (raw < -1e-4f || raw > 1.f + 1e-4f))) {
        atomicExch(fault, 1);
        return;
      }
      const float clamped = probabilities ? fminf(fmaxf(score, 0.f), 1.f) : score;
      if (clamped > best || (clamped == best && cls < chosen)) {
        best = clamped;
        chosen = cls;
      }
    }
    if (best >= threshold) {
      const int slot = atomicAdd(count, 1);
      GpuCandidate item{x1, y1, x2, y2, best, chosen, prediction, x1, y1, x2, y2, prediction};
      out[slot] = item;
    }
    return;
  }
  for (int cls = 0; cls < num_classes; ++cls) {
    const float raw = cuda_read(data, channel_axis, channels, predictions, box_channels + cls, prediction);
    const float score = cuda_activate(raw, probabilities);
    if (!isfinite(score) || (probabilities && (raw < -1e-4f || raw > 1.f + 1e-4f))) {
      atomicExch(fault, 1);
      return;
    }
    const float clamped = probabilities ? fminf(fmaxf(score, 0.f), 1.f) : score;
    if (clamped >= threshold) {
      const int slot = atomicAdd(count, 1);
      GpuCandidate item{x1, y1, x2, y2, clamped, cls, prediction * num_classes + cls, x1, y1, x2, y2, prediction};
      out[slot] = item;
    }
  }
}

__device__ float map_axis(float value, int pad, double scale) {
  return static_cast<float>((static_cast<double>(value) - static_cast<double>(pad)) / scale);
}

__global__ void map_compact_kernel(GpuCandidate* input, int count, int pad_left, int pad_top, double scale_x,
                                   double scale_y, float width, float height, GpuCandidate* output, int* kept) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int written = 0;
  for (int i = 0; i < count; ++i) {
    GpuCandidate item = input[i];
    item.x1 = map_axis(item.x1, pad_left, scale_x);
    item.x2 = map_axis(item.x2, pad_left, scale_x);
    item.y1 = map_axis(item.y1, pad_top, scale_y);
    item.y2 = map_axis(item.y2, pad_top, scale_y);
    item.x1 = fminf(fmaxf(item.x1, 0.f), width);
    item.x2 = fminf(fmaxf(item.x2, 0.f), width);
    item.y1 = fminf(fmaxf(item.y1, 0.f), height);
    item.y2 = fminf(fmaxf(item.y2, 0.f), height);
    if (item.x2 > item.x1 && item.y2 > item.y1) output[written++] = item;
  }
  *kept = written;
}

__global__ void nms_kernel(const GpuCandidate* items, int count, float iou_threshold, int class_aware,
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

}  // namespace

CudaDetectionPass decode_yolov8_cuda_pass(const Tensor& tensor, const DecoderConfig& config,
                                          const GeometryTransform& geometry, const DeviceContext& device,
                                          int mask_channels) {
  cuda_check(cudaSetDevice(device.device_id()), "cudaSetDevice");
  if (tensor.rank() != 3 || tensor.shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 detection expects a rank-3 batch of 1");
  }
  const int channels = static_cast<int>(tensor.shape()[static_cast<std::size_t>(config.channel_axis)]);
  const int other_axis = config.channel_axis == 1 ? 2 : 1;
  const int predictions = static_cast<int>(tensor.shape()[static_cast<std::size_t>(other_axis)]);
  if (mask_channels < 0 || channels != config.box_channels + config.num_classes + mask_channels) {
    throw VieError(ErrorCode::InvalidArgument, "prediction channels do not match box channels plus classes");
  }
  if (predictions < 1 || predictions > 1000000) {
    throw VieError(ErrorCode::InvalidArgument, "prediction count is outside 1..1000000");
  }
  const bool best = *config.class_selection == ClassSelection::BestClass;
  const int capacity = best ? predictions : predictions * config.num_classes;
  if (capacity > 1000000) {
    throw VieError(ErrorCode::Unsupported, "CUDA decoder candidate capacity exceeds 1000000");
  }
  std::vector<int> strides;
  std::vector<int> widths;
  std::vector<int> heights;
  if (config.box_decode == BoxDecode::Dfl) {
    int cells = 0;
    for (const FeatureLevel& level : config.levels) {
      strides.push_back(level.stride);
      widths.push_back(level.grid_width);
      heights.push_back(level.grid_height);
      cells += level.grid_width * level.grid_height;
    }
    if (cells != predictions) {
      throw VieError(ErrorCode::InvalidArgument, "feature levels do not cover the prediction count");
    }
  }
  DeviceBuffer levels_stride(std::max<std::size_t>(strides.size(), 1) * sizeof(int));
  DeviceBuffer levels_width(std::max<std::size_t>(widths.size(), 1) * sizeof(int));
  DeviceBuffer levels_height(std::max<std::size_t>(heights.size(), 1) * sizeof(int));
  if (!strides.empty()) {
    cuda_check(cudaMemcpy(levels_stride.data, strides.data(), strides.size() * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy");
    cuda_check(cudaMemcpy(levels_width.data, widths.data(), widths.size() * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy");
    cuda_check(cudaMemcpy(levels_height.data, heights.data(), heights.size() * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy");
  }
  DeviceBuffer candidates(static_cast<std::size_t>(capacity) * sizeof(GpuCandidate));
  DeviceBuffer mapped(static_cast<std::size_t>(*config.pre_nms_top_k) * sizeof(GpuCandidate));
  DeviceBuffer counter(sizeof(int));
  DeviceBuffer fault(sizeof(int));
  DeviceBuffer kept_counter(sizeof(int));
  DeviceBuffer kept_indices(static_cast<std::size_t>(*config.max_detections) * sizeof(int));
  cuda_check(cudaMemset(counter.data, 0, sizeof(int)), "cudaMemset");
  cuda_check(cudaMemset(fault.data, 0, sizeof(int)), "cudaMemset");

  cudaEvent_t started = nullptr;
  cudaEvent_t stopped = nullptr;
  cuda_check(cudaEventCreate(&started), "cudaEventCreate");
  cuda_check(cudaEventCreate(&stopped), "cudaEventCreate");
  cuda_check(cudaEventRecord(started), "cudaEventRecord");
  const int threads = 128;
  const int blocks = (predictions + threads - 1) / threads;
  emit_kernel<<<blocks, threads>>>(
      reinterpret_cast<const float*>(tensor.device_data()), config.channel_axis, channels, predictions,
      config.box_channels, config.num_classes, config.box_decode == BoxDecode::DecodedCxcywh ? 1 : 0, config.reg_max,
      *config.scores_are_probabilities ? 1 : 0, best ? 1 : 0, *config.confidence_threshold,
      static_cast<int*>(levels_stride.data), static_cast<int*>(levels_width.data),
      static_cast<int*>(levels_height.data), static_cast<int>(config.levels.size()),
      static_cast<GpuCandidate*>(candidates.data), static_cast<int*>(counter.data),
      static_cast<int*>(fault.data));
  cuda_check(cudaGetLastError(), "emit kernel");
  int faulted = 0;
  cuda_check(cudaMemcpy(&faulted, fault.data, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  if (faulted != 0) {
    cudaEventDestroy(started);
    cudaEventDestroy(stopped);
    throw VieError(ErrorCode::InvalidArgument, "prediction tensor contains a non-finite value or an illegal probability");
  }
  int raw_count = 0;
  cuda_check(cudaMemcpy(&raw_count, counter.data, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  if (raw_count < 0 || raw_count > capacity) {
    cudaEventDestroy(started);
    cudaEventDestroy(stopped);
    throw VieError(ErrorCode::Internal, "CUDA candidate count is invalid");
  }
  cudaEvent_t sort_started = nullptr;
  cudaEvent_t sort_stopped = nullptr;
  cuda_check(cudaEventCreate(&sort_started), "cudaEventCreate");
  cuda_check(cudaEventCreate(&sort_stopped), "cudaEventCreate");
  if (raw_count > 1) {
    cuda_check(cudaEventRecord(sort_started), "cudaEventRecord");
    thrust::sort(thrust::device_pointer_cast(static_cast<GpuCandidate*>(candidates.data)),
                 thrust::device_pointer_cast(static_cast<GpuCandidate*>(candidates.data) + raw_count), CandidateLess{});
    cuda_check(cudaEventRecord(sort_stopped), "cudaEventRecord");
  }
  const int top = std::min(raw_count, *config.pre_nms_top_k);
  cuda_check(cudaMemset(kept_counter.data, 0, sizeof(int)), "cudaMemset");
  int mapped_count = 0;
  if (top > 0) {
    map_compact_kernel<<<1, 1>>>(static_cast<GpuCandidate*>(candidates.data), top,
                                 geometry_model_origin_x(geometry), geometry_model_origin_y(geometry),
                                 geometry.scale_x, geometry.scale_y, static_cast<float>(geometry.original_width),
                                 static_cast<float>(geometry.original_height), static_cast<GpuCandidate*>(mapped.data),
                                 static_cast<int*>(kept_counter.data));
    cuda_check(cudaGetLastError(), "map kernel");
    cuda_check(cudaMemcpy(&mapped_count, kept_counter.data, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  }
  cudaEvent_t nms_started = nullptr;
  cudaEvent_t nms_stopped = nullptr;
  cuda_check(cudaEventCreate(&nms_started), "cudaEventCreate");
  cuda_check(cudaEventCreate(&nms_stopped), "cudaEventCreate");
  int kept_count = 0;
  if (mapped_count > 0 && *config.nms_enabled) {
    cuda_check(cudaEventRecord(nms_started), "cudaEventRecord");
    nms_kernel<<<1, 128, static_cast<std::size_t>(mapped_count)>>>(
        static_cast<GpuCandidate*>(mapped.data), mapped_count, *config.iou_threshold,
        *config.class_aware_nms ? 1 : 0, *config.max_detections, static_cast<int*>(kept_indices.data),
        static_cast<int*>(kept_counter.data));
    cuda_check(cudaGetLastError(), "nms kernel");
    cuda_check(cudaEventRecord(nms_stopped), "cudaEventRecord");
    cuda_check(cudaMemcpy(&kept_count, kept_counter.data, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  } else {
    kept_count = std::min(mapped_count, *config.max_detections);
  }
  cuda_check(cudaEventRecord(stopped), "cudaEventRecord");
  cuda_check(cudaEventSynchronize(stopped), "cudaEventSynchronize");
  float elapsed = 0.f;
  cuda_check(cudaEventElapsedTime(&elapsed, started, stopped), "cudaEventElapsedTime");
  float sort_ms = 0.f;
  float nms_ms = 0.f;
  if (raw_count > 1) cuda_check(cudaEventElapsedTime(&sort_ms, sort_started, sort_stopped), "cudaEventElapsedTime");
  if (mapped_count > 0 && *config.nms_enabled) {
    cuda_check(cudaEventElapsedTime(&nms_ms, nms_started, nms_stopped), "cudaEventElapsedTime");
  }
  cudaEventDestroy(started);
  cudaEventDestroy(stopped);
  cudaEventDestroy(sort_started);
  cudaEventDestroy(sort_stopped);
  cudaEventDestroy(nms_started);
  cudaEventDestroy(nms_stopped);

  std::vector<GpuCandidate> host_boxes(static_cast<std::size_t>(std::max(mapped_count, 0)));
  std::vector<int> host_kept(static_cast<std::size_t>(std::max(kept_count, 0)));
  cudaEvent_t copy_started = nullptr;
  cudaEvent_t copy_stopped = nullptr;
  cuda_check(cudaEventCreate(&copy_started), "cudaEventCreate");
  cuda_check(cudaEventCreate(&copy_stopped), "cudaEventCreate");
  cuda_check(cudaEventRecord(copy_started), "cudaEventRecord");
  if (mapped_count > 0) {
    cuda_check(cudaMemcpy(host_boxes.data(), mapped.data, static_cast<std::size_t>(mapped_count) * sizeof(GpuCandidate),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy");
  }
  if (*config.nms_enabled && kept_count > 0) {
    cuda_check(cudaMemcpy(host_kept.data(), kept_indices.data, static_cast<std::size_t>(kept_count) * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy");
  }
  cuda_check(cudaEventRecord(copy_stopped), "cudaEventRecord");
  cuda_check(cudaEventSynchronize(copy_stopped), "cudaEventSynchronize");
  float copy_ms = 0.f;
  cuda_check(cudaEventElapsedTime(&copy_ms, copy_started, copy_stopped), "cudaEventElapsedTime");
  cudaEventDestroy(copy_started);
  cudaEventDestroy(copy_stopped);

  CudaDetectionPass pass;
  pass.candidates_after_filter = raw_count;
  pass.candidates_before_nms = mapped_count;
  pass.nms_applied = *config.nms_enabled;
  pass.gpu_execute_ms = static_cast<double>(elapsed);
  pass.d2h_ms = static_cast<double>(copy_ms);
  pass.sort_ms = static_cast<double>(sort_ms);
  pass.nms_ms = static_cast<double>(nms_ms);
  const auto push_kept = [&](const GpuCandidate& item) {
    CudaKeptPrediction kept;
    kept.model_box = BoundingBox{item.model_x1, item.model_y1, item.model_x2, item.model_y2};
    kept.original_box = BoundingBox{item.x1, item.y1, item.x2, item.y2};
    kept.score = item.score;
    kept.class_id = item.class_id;
    kept.prediction_index = item.prediction_index;
    pass.kept.push_back(kept);
  };
  if (*config.nms_enabled) {
    for (int index : host_kept) {
      if (index < 0 || index >= mapped_count) {
        throw VieError(ErrorCode::Internal, "CUDA NMS kept an invalid index");
      }
      push_kept(host_boxes[static_cast<std::size_t>(index)]);
    }
  } else {
    for (int i = 0; i < kept_count; ++i) push_kept(host_boxes[static_cast<std::size_t>(i)]);
  }
  return pass;
}

DetectionResult decode_yolov8_cuda(const Tensor& tensor, const DecoderConfig& config,
                                   const GeometryTransform& geometry, const DeviceContext& device) {
  const CudaDetectionPass pass = decode_yolov8_cuda_pass(tensor, config, geometry, device, 0);
  DetectionResult result;
  result.original_width = geometry.original_width;
  result.original_height = geometry.original_height;
  result.execution_device = DeviceType::Cuda;
  result.execution_device_id = device.device_id();
  result.protocol = "yolov8_detect_raw";
  result.nms_applied = pass.nms_applied;
  result.coordinates_mapped = true;
  result.candidates_after_filter = pass.candidates_after_filter;
  result.candidates_before_nms = pass.candidates_before_nms;
  result.gpu_execute_ms = pass.gpu_execute_ms;
  result.d2h_ms = pass.d2h_ms;
  result.detections.reserve(pass.kept.size());
  for (const CudaKeptPrediction& item : pass.kept) {
    result.detections.push_back(Detection{item.original_box, item.score, item.class_id});
  }
  return result;
}

}  // namespace vie
