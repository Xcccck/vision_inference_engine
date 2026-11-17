#include "yolov8_seg_cuda.hpp"

#include "yolov8_cuda.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <limits>
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

struct MaskJob {
  float model_x1, model_y1, model_x2, model_y2;
  float orig_x1, orig_y1, orig_x2, orig_y2;
  int prediction_index;
};

__device__ int clamp_index(int value, int limit) {
  if (value < 0) return 0;
  if (value >= limit) return limit - 1;
  return value;
}

__device__ float sample_bilinear(const float* source, int width, int height, double x, double y) {
  const int x0 = static_cast<int>(floor(x));
  const int y0 = static_cast<int>(floor(y));
  const float wx = static_cast<float>(x - static_cast<double>(x0));
  const float wy = static_cast<float>(y - static_cast<double>(y0));
  const auto at = [&](int px, int py) {
    px = clamp_index(px, width);
    py = clamp_index(py, height);
    return source[static_cast<std::size_t>(py) * static_cast<std::size_t>(width) + static_cast<std::size_t>(px)];
  };
  const float top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * wx;
  const float bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * wx;
  return top + (bottom - top) * wy;
}

__device__ bool covers_pixel(float x1, float y1, float x2, float y2, int x, int y) {
  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);
  return fx >= x1 && fx < x2 && fy >= y1 && fy < y2;
}

__global__ void scan_finite_kernel(const float* data, int count, int* fault) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count || isfinite(data[index])) return;
  atomicCAS(fault, 0, 1);
}

__global__ void gather_kernel(const float* predictions, int channel_axis, int channels, int prediction_count,
                              int coeff_base, int mask_channels, const MaskJob* jobs, int instances, float* coeffs,
                              int* fault) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = instances * mask_channels;
  if (index >= total) return;
  const int instance = index / mask_channels;
  const int channel = index - instance * mask_channels;
  const int prediction = jobs[instance].prediction_index;
  const int tensor_channel = coeff_base + channel;
  const int offset = channel_axis == 1 ? tensor_channel * prediction_count + prediction
                                       : prediction * channels + tensor_channel;
  const float value = predictions[offset];
  if (!isfinite(value)) atomicCAS(fault, 0, 2);
  coeffs[static_cast<std::size_t>(instance) * static_cast<std::size_t>(mask_channels) +
         static_cast<std::size_t>(channel)] = value;
}

__global__ void reconstruct_kernel(const float* proto, const float* coeffs, int instances, int mask_channels,
                                   int proto_h, int proto_w, float* logits, int* fault) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int pixels = proto_h * proto_w;
  if (index >= instances * pixels) return;
  const int instance = index / pixels;
  const int spatial = index - instance * pixels;
  const float* instance_coeffs = coeffs + static_cast<std::size_t>(instance) * static_cast<std::size_t>(mask_channels);
  float sum = 0.f;
  for (int channel = 0; channel < mask_channels; ++channel) {
    const float value = proto[static_cast<std::size_t>(channel) * static_cast<std::size_t>(pixels) +
                              static_cast<std::size_t>(spatial)];
    sum = sum + instance_coeffs[channel] * value;
  }
  if (!isfinite(sum)) atomicCAS(fault, 0, 3);
  logits[index] = sum;
}

__global__ void zero_proto_kernel(float* logits, const MaskJob* jobs, int instances, int proto_w, int proto_h,
                                  double proto_per_model_x, double proto_per_model_y) {
  const int pixels = proto_w * proto_h;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= instances * pixels) return;
  const int instance = index / pixels;
  const int spatial = index - instance * pixels;
  const int y = spatial / proto_w;
  const int x = spatial - y * proto_w;
  const MaskJob job = jobs[instance];
  const float x1 = static_cast<float>(static_cast<double>(job.model_x1) * proto_per_model_x);
  const float y1 = static_cast<float>(static_cast<double>(job.model_y1) * proto_per_model_y);
  const float x2 = static_cast<float>(static_cast<double>(job.model_x2) * proto_per_model_x);
  const float y2 = static_cast<float>(static_cast<double>(job.model_y2) * proto_per_model_y);
  if (!covers_pixel(x1, y1, x2, y2, x, y)) logits[index] = 0.f;
}

__global__ void zero_original_kernel(float* scaled, const MaskJob* jobs, int instances, int width, int height) {
  const int pixels = width * height;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= instances * pixels) return;
  const int instance = index / pixels;
  const int spatial = index - instance * pixels;
  const int y = spatial / width;
  const int x = spatial - y * width;
  const MaskJob job = jobs[instance];
  if (!covers_pixel(job.orig_x1, job.orig_y1, job.orig_x2, job.orig_y2, x, y)) scaled[index] = 0.f;
}

__global__ void resize_kernel(const float* source, int source_w, int source_h, float* dest, int instances, int dest_w,
                              int dest_h, int retina, int pad_left, int pad_top, double scale_x, double scale_y,
                              double proto_per_model_x, double proto_per_model_y, int* fault) {
  const int pixels = dest_w * dest_h;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= instances * pixels) return;
  const int instance = index / pixels;
  const int spatial = index - instance * pixels;
  const int y = spatial / dest_w;
  const int x = spatial - y * dest_w;
  double src_x = 0;
  double src_y = 0;
  if (retina) {
    src_x = (static_cast<double>(pad_left) + (static_cast<double>(x) + 0.5) * scale_x) * proto_per_model_x - 0.5;
    src_y = (static_cast<double>(pad_top) + (static_cast<double>(y) + 0.5) * scale_y) * proto_per_model_y - 0.5;
  } else {
    src_x = (static_cast<double>(x) + 0.5) * proto_per_model_x - 0.5;
    src_y = (static_cast<double>(y) + 0.5) * proto_per_model_y - 0.5;
  }
  const float* instance_source = source + static_cast<std::size_t>(instance) * static_cast<std::size_t>(source_w) *
                                               static_cast<std::size_t>(source_h);
  const float value = sample_bilinear(instance_source, source_w, source_h, src_x, src_y);
  if (!isfinite(value)) atomicCAS(fault, 0, 4);
  dest[index] = value;
}

__global__ void binarize_kernel(const float* scaled, std::uint8_t* binary, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  binary[index] = scaled[index] > 0.f ? 1 : 0;
}

void require_set(bool ready, const std::string& message) {
  if (!ready) throw VieError(ErrorCode::UnresolvedField, message);
}

std::size_t checked_mask_bytes(std::size_t instances, int width, int height) {
  if (width < 1 || height < 1) {
    throw VieError(ErrorCode::InvalidArgument, "mask output size is invalid");
  }
  const std::size_t wide = static_cast<std::size_t>(width);
  const std::size_t high = static_cast<std::size_t>(height);
  if (wide > std::numeric_limits<std::size_t>::max() / high) {
    throw VieError(ErrorCode::InvalidArgument, "mask width times height overflows");
  }
  const std::size_t pixels = wide * high;
  if (instances > std::numeric_limits<std::size_t>::max() / pixels) {
    throw VieError(ErrorCode::InvalidArgument, "mask allocation size overflows");
  }
  return instances * pixels;
}

const char* fault_message(int code) {
  if (code == 1) return "prototype tensor contains a non-finite value";
  if (code == 2) return "mask coefficient is not finite";
  if (code == 3) return "mask reconstruction is not finite";
  if (code == 4) return "mask resize is not finite";
  return "CUDA mask kernel reported a fault";
}

}  // namespace

SegmentationBatch decode_yolov8_seg_cuda(const DecoderInput& input, const DecoderConfig& config,
                                         const DeviceContext& device, CudaSegmentationTiming* timing) {
  const auto wall_start = std::chrono::steady_clock::now();
  if (device.device() != DeviceType::Cuda) {
    throw VieError(ErrorCode::InvalidArgument, "CUDA segmentation decoder requires a CUDA device");
  }
  cuda_check(cudaSetDevice(device.device_id()), "cudaSetDevice");
  validate_decoder_config(config);
  require_set(!config.prototype_name.empty(), "decoder.prototype_name is unresolved");
  require_set(config.mask_channels >= 1, "decoder.mask_channels is unresolved");
  require_set(config.prototype_channel_axis.has_value(), "decoder.prototype_channel_axis is unresolved");
  require_set(config.retina_masks.has_value(), "decoder.retina_masks is unresolved");
  require_set(config.binary_mask_gt_zero.has_value(), "decoder.binary_mask_gt_zero is unresolved");
  require_set(config.max_mask_bytes.has_value(), "decoder.max_mask_bytes is unresolved");
  if (config.box_decode != BoxDecode::DecodedCxcywh || config.box_channels != 4) {
    throw VieError(ErrorCode::InvalidArgument, "yolov8_seg_raw requires decoded cxcywh with 4 box channels");
  }
  if (!*config.scores_are_probabilities) {
    throw VieError(ErrorCode::InvalidArgument,
                   "yolov8_seg_raw class scores are already probabilities; refusing a second sigmoid");
  }
  if (*config.prototype_channel_axis != 1) {
    throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw only reads an NCHW prototype tensor");
  }
  if (!*config.binary_mask_gt_zero) {
    throw VieError(ErrorCode::Unsupported, "yolov8_seg_raw only emits a binary mask from the M > 0 test");
  }
  if (!input.geometry.has_value()) {
    throw VieError(ErrorCode::UnresolvedField,
                   "preprocess geometry is unresolved; model-input coordinates are not used as original-image coordinates");
  }
  const GeometryTransform& geometry = *input.geometry;
  if (geometry.output_width < 1 || geometry.output_height < 1 || geometry.original_width < 1 ||
      geometry.original_height < 1 || geometry.scale_x == 0.0 || geometry.scale_y == 0.0) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale or image size is invalid");
  }
  const auto predictions = input.outputs.find(config.output_name);
  if (predictions == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder is missing output tensor '" + config.output_name + "'");
  }
  const auto prototypes = input.outputs.find(config.prototype_name);
  if (prototypes == input.outputs.end()) {
    throw VieError(ErrorCode::InvalidArgument,
                   "decoder is missing prototype tensor '" + config.prototype_name + "'");
  }
  const Tensor& prediction_tensor = predictions->second;
  const Tensor& prototype_tensor = prototypes->second;
  if (prediction_tensor.dtype() != DataType::Fp32 || prototype_tensor.dtype() != DataType::Fp32) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 segmentation requires fp32 tensors");
  }
  if (!prediction_tensor.contiguous() || !prototype_tensor.contiguous()) {
    throw VieError(ErrorCode::Unsupported, "YOLOv8 segmentation does not read a non-contiguous tensor");
  }
  if (!prediction_tensor.is_cuda() || !prototype_tensor.is_cuda() ||
      prediction_tensor.device_id() != device.device_id() || prototype_tensor.device_id() != device.device_id()) {
    throw VieError(ErrorCode::InvalidArgument, "CUDA segmentation decoder cannot read a tensor on another device");
  }
  if (prediction_tensor.rank() != 3 || prediction_tensor.shape()[0] != 1 || prototype_tensor.rank() != 4 ||
      prototype_tensor.shape()[0] != 1) {
    throw VieError(ErrorCode::InvalidArgument, "YOLOv8 segmentation expects batch 1 prediction and prototype tensors");
  }
  const int channels = static_cast<int>(prediction_tensor.shape()[static_cast<std::size_t>(config.channel_axis)]);
  const int prediction_axis = config.channel_axis == 1 ? 2 : 1;
  const int prediction_count = static_cast<int>(prediction_tensor.shape()[static_cast<std::size_t>(prediction_axis)]);
  if (channels != 4 + config.num_classes + config.mask_channels) {
    throw VieError(ErrorCode::InvalidArgument, "prediction channels do not match 4 + classes + mask channels");
  }
  if (prototype_tensor.shape()[1] != config.mask_channels) {
    throw VieError(ErrorCode::InvalidArgument, "prototype channels do not match mask channels");
  }
  const int proto_h = static_cast<int>(prototype_tensor.shape()[2]);
  const int proto_w = static_cast<int>(prototype_tensor.shape()[3]);
  if (proto_h < 1 || proto_w < 1) {
    throw VieError(ErrorCode::InvalidArgument, "prototype spatial size is invalid");
  }
  const bool retina = *config.retina_masks;
  const int mask_w = retina ? geometry.original_width : geometry.output_width;
  const int mask_h = retina ? geometry.original_height : geometry.output_height;

  const CudaDetectionPass detected =
      decode_yolov8_cuda_pass(prediction_tensor, config, geometry, device, config.mask_channels);
  const std::size_t needed = checked_mask_bytes(detected.kept.size(), mask_w, mask_h);
  if (needed > *config.max_mask_bytes) {
    throw VieError(ErrorCode::InvalidArgument, "mask byte budget exceeded: need " + std::to_string(needed) +
                                                   ", budget " + std::to_string(*config.max_mask_bytes));
  }

  SegmentationBatch batch;
  batch.original_width = geometry.original_width;
  batch.original_height = geometry.original_height;
  batch.execution_device = DeviceType::Cuda;
  batch.execution_device_id = device.device_id();
  batch.protocol = "yolov8_seg_raw";
  if (detected.kept.empty()) {
    if (timing != nullptr) {
      timing->instances = 0;
      timing->mask_width = 0;
      timing->mask_height = 0;
      timing->prototype_channels = config.mask_channels;
      timing->prototype_width = proto_w;
      timing->prototype_height = proto_h;
      timing->detection_ms = detected.gpu_execute_ms;
      timing->sort_ms = detected.sort_ms;
      timing->nms_ms = detected.nms_ms;
      timing->d2h_ms = detected.d2h_ms;
      timing->gpu_ms = detected.gpu_execute_ms;
      timing->wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
      timing->includes_allocation = true;
      timing->includes_sync = true;
      timing->masks_copied_to_host = false;
    }
    return batch;
  }

  std::vector<MaskJob> jobs(detected.kept.size());
  for (std::size_t i = 0; i < detected.kept.size(); ++i) {
    const CudaKeptPrediction& kept = detected.kept[i];
    jobs[i] = MaskJob{kept.model_box.x1, kept.model_box.y1, kept.model_box.x2, kept.model_box.y2,
                      kept.original_box.x1, kept.original_box.y1, kept.original_box.x2, kept.original_box.y2,
                      kept.prediction_index};
  }
  const int instances = static_cast<int>(jobs.size());
  const int threads = 128;
  const std::size_t proto_pixels = static_cast<std::size_t>(proto_h) * static_cast<std::size_t>(proto_w);
  const std::size_t mask_pixels = static_cast<std::size_t>(mask_w) * static_cast<std::size_t>(mask_h);
  DeviceBuffer job_buffer(jobs.size() * sizeof(MaskJob));
  DeviceBuffer coeff_buffer(jobs.size() * static_cast<std::size_t>(config.mask_channels) * sizeof(float));
  DeviceBuffer logits(static_cast<std::size_t>(instances) * proto_pixels * sizeof(float));
  DeviceBuffer scaled(static_cast<std::size_t>(instances) * mask_pixels * sizeof(float));
  DeviceBuffer fault(sizeof(int));
  cuda_check(cudaMemcpy(job_buffer.data, jobs.data(), jobs.size() * sizeof(MaskJob), cudaMemcpyHostToDevice),
             "cudaMemcpy");
  cuda_check(cudaMemset(fault.data, 0, sizeof(int)), "cudaMemset");

  const auto alloc_start = std::chrono::steady_clock::now();
  Tensor mask_storage = Tensor::zeros(device, DataType::Uint8,
                                      {static_cast<std::int64_t>(instances), static_cast<std::int64_t>(mask_h),
                                       static_cast<std::int64_t>(mask_w)});
  const double alloc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alloc_start).count();

  cudaEvent_t gather_started = nullptr;
  cudaEvent_t gather_stopped = nullptr;
  cudaEvent_t reconstruct_started = nullptr;
  cudaEvent_t reconstruct_stopped = nullptr;
  cudaEvent_t crop_started = nullptr;
  cudaEvent_t crop_stopped = nullptr;
  cudaEvent_t resize_started = nullptr;
  cudaEvent_t resize_stopped = nullptr;
  cuda_check(cudaEventCreate(&gather_started), "cudaEventCreate");
  cuda_check(cudaEventCreate(&gather_stopped), "cudaEventCreate");
  cuda_check(cudaEventCreate(&reconstruct_started), "cudaEventCreate");
  cuda_check(cudaEventCreate(&reconstruct_stopped), "cudaEventCreate");
  cuda_check(cudaEventCreate(&crop_started), "cudaEventCreate");
  cuda_check(cudaEventCreate(&crop_stopped), "cudaEventCreate");
  cuda_check(cudaEventCreate(&resize_started), "cudaEventCreate");
  cuda_check(cudaEventCreate(&resize_stopped), "cudaEventCreate");

  const int scan_blocks = (static_cast<int>(proto_pixels * static_cast<std::size_t>(config.mask_channels)) + threads - 1) /
                          threads;
  scan_finite_kernel<<<scan_blocks, threads>>>(reinterpret_cast<const float*>(prototype_tensor.device_data()),
                                               static_cast<int>(proto_pixels * static_cast<std::size_t>(config.mask_channels)),
                                               static_cast<int*>(fault.data));
  const int gather_total = instances * config.mask_channels;
  cuda_check(cudaEventRecord(gather_started), "cudaEventRecord");
  gather_kernel<<<(gather_total + threads - 1) / threads, threads>>>(
      reinterpret_cast<const float*>(prediction_tensor.device_data()), config.channel_axis, channels, prediction_count,
      config.box_channels + config.num_classes, config.mask_channels, static_cast<const MaskJob*>(job_buffer.data),
      instances, static_cast<float*>(coeff_buffer.data), static_cast<int*>(fault.data));
  cuda_check(cudaEventRecord(gather_stopped), "cudaEventRecord");
  cuda_check(cudaGetLastError(), "gather kernel");

  const double proto_per_model_x = static_cast<double>(proto_w) / static_cast<double>(geometry.output_width);
  const double proto_per_model_y = static_cast<double>(proto_h) / static_cast<double>(geometry.output_height);
  const int proto_total = instances * static_cast<int>(proto_pixels);
  const int mask_total = instances * static_cast<int>(mask_pixels);
  const int proto_blocks = (proto_total + threads - 1) / threads;
  const int mask_blocks = (mask_total + threads - 1) / threads;
  cuda_check(cudaEventRecord(reconstruct_started), "cudaEventRecord");
  reconstruct_kernel<<<proto_blocks, threads>>>(reinterpret_cast<const float*>(prototype_tensor.device_data()),
                                                static_cast<float*>(coeff_buffer.data), instances, config.mask_channels,
                                                proto_h, proto_w, static_cast<float*>(logits.data),
                                                static_cast<int*>(fault.data));
  cuda_check(cudaEventRecord(reconstruct_stopped), "cudaEventRecord");
  if (!retina) {
    cuda_check(cudaEventRecord(crop_started), "cudaEventRecord");
    zero_proto_kernel<<<proto_blocks, threads>>>(static_cast<float*>(logits.data), static_cast<const MaskJob*>(job_buffer.data),
                                                 instances, proto_w, proto_h, proto_per_model_x, proto_per_model_y);
    cuda_check(cudaEventRecord(crop_stopped), "cudaEventRecord");
  }
  cuda_check(cudaEventRecord(resize_started), "cudaEventRecord");
  resize_kernel<<<mask_blocks, threads>>>(static_cast<float*>(logits.data), proto_w, proto_h, static_cast<float*>(scaled.data),
                                          instances, mask_w, mask_h, retina ? 1 : 0, geometry.pad_left, geometry.pad_top,
                                          geometry.scale_x, geometry.scale_y, proto_per_model_x, proto_per_model_y,
                                          static_cast<int*>(fault.data));
  cuda_check(cudaEventRecord(resize_stopped), "cudaEventRecord");
  if (retina) {
    cuda_check(cudaEventRecord(crop_started), "cudaEventRecord");
    zero_original_kernel<<<mask_blocks, threads>>>(static_cast<float*>(scaled.data),
                                                   static_cast<const MaskJob*>(job_buffer.data), instances, mask_w, mask_h);
    cuda_check(cudaEventRecord(crop_stopped), "cudaEventRecord");
  }
  binarize_kernel<<<mask_blocks, threads>>>(static_cast<float*>(scaled.data),
                                            reinterpret_cast<std::uint8_t*>(mask_storage.device_data()), mask_total);
  cuda_check(cudaGetLastError(), "mask kernel");
  cuda_check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
  int fault_code = 0;
  cuda_check(cudaMemcpy(&fault_code, fault.data, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  float gather_ms = 0.f;
  float reconstruct_ms = 0.f;
  float crop_ms = 0.f;
  float resize_ms = 0.f;
  cuda_check(cudaEventElapsedTime(&gather_ms, gather_started, gather_stopped), "cudaEventElapsedTime");
  cuda_check(cudaEventElapsedTime(&reconstruct_ms, reconstruct_started, reconstruct_stopped), "cudaEventElapsedTime");
  cuda_check(cudaEventElapsedTime(&crop_ms, crop_started, crop_stopped), "cudaEventElapsedTime");
  cuda_check(cudaEventElapsedTime(&resize_ms, resize_started, resize_stopped), "cudaEventElapsedTime");
  cudaEventDestroy(gather_started);
  cudaEventDestroy(gather_stopped);
  cudaEventDestroy(reconstruct_started);
  cudaEventDestroy(reconstruct_stopped);
  cudaEventDestroy(crop_started);
  cudaEventDestroy(crop_stopped);
  cudaEventDestroy(resize_started);
  cudaEventDestroy(resize_stopped);
  if (fault_code != 0) {
    throw VieError(ErrorCode::InvalidArgument, fault_message(fault_code));
  }

  batch.instances.reserve(static_cast<std::size_t>(instances));
  for (int instance = 0; instance < instances; ++instance) {
    const std::size_t offset = static_cast<std::size_t>(instance) * mask_pixels;
    Tensor slice = mask_storage.view({static_cast<std::int64_t>(mask_h), static_cast<std::int64_t>(mask_w)},
                                     {static_cast<std::int64_t>(mask_w), 1}, offset);
    InstanceSegmentation item;
    item.detection.box = detected.kept[static_cast<std::size_t>(instance)].original_box;
    item.detection.confidence = detected.kept[static_cast<std::size_t>(instance)].score;
    item.detection.class_id = detected.kept[static_cast<std::size_t>(instance)].class_id;
    item.mask = Mask::from_tensor(MaskKind::Binary, std::move(slice));
    batch.instances.push_back(std::move(item));
  }
  if (timing != nullptr) {
    timing->instances = instances;
    timing->mask_width = mask_w;
    timing->mask_height = mask_h;
    timing->prototype_channels = config.mask_channels;
    timing->prototype_width = proto_w;
    timing->prototype_height = proto_h;
    timing->detection_ms = detected.gpu_execute_ms;
    timing->sort_ms = detected.sort_ms;
    timing->nms_ms = detected.nms_ms;
    timing->gather_ms = gather_ms;
    timing->reconstruct_ms = reconstruct_ms;
    timing->crop_ms = crop_ms;
    timing->resize_ms = resize_ms;
    timing->gpu_ms = detected.gpu_execute_ms + static_cast<double>(gather_ms + reconstruct_ms + crop_ms + resize_ms);
    timing->d2h_ms = detected.d2h_ms;
    timing->mask_alloc_ms = alloc_ms;
    timing->wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
    timing->includes_allocation = true;
    timing->includes_sync = true;
    timing->masks_copied_to_host = false;
  }
  return batch;
}

}  // namespace vie
