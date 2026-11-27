#include "classification_cuda.hpp"

#include "vie/decoder/classification.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace vie {
namespace {

// Device-side ranked score. Kept as plain POD so the kernel does not depend on
// host-side default member initializers. The host converts to RankedScore.
struct GpuRanked {
  float score;
  int class_id;
};

static_assert(sizeof(GpuRanked) == 8, "ranked-score record gained padding");

struct GpuRankedOrder {
  __host__ __device__ bool operator()(const GpuRanked& left, const GpuRanked& right) const {
    if (left.score != right.score) return left.score > right.score;
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

// One block strided over the class axis. Every value is checked against the
// CPU contract while the packed array for the sort is filled and the class-
// order probability sum is accumulated in FP64. The first invalid class id is
// kept in fault as (class_id << 3) | reason so the message names the same
// class the CPU loop would have thrown on. reason: 1 NaN, 2 +Inf, 3 -Inf,
// 4 below 0, 5 above 1.
__global__ void validate_pack_kernel(const float* data, int classes, long long stride, int probabilities,
                                     GpuRanked* ranked, unsigned long long* fault, double* sum_out) {
  const int tid = threadIdx.x;
  double partial = 0.0;
  for (int i = tid; i < classes; i += blockDim.x) {
    const float value = data[static_cast<long long>(i) * stride];
    int reason = 0;
    if (!isfinite(value)) {
      reason = isnan(value) ? 1 : (value > 0.f ? 2 : 3);
    } else if (probabilities) {
      if (value < 0.f) {
        reason = 4;
      } else if (value > 1.f) {
        reason = 5;
      }
    }
    if (reason != 0) {
      atomicMin(fault, (static_cast<unsigned long long>(i) << 3) | static_cast<unsigned long long>(reason));
    } else {
      partial += static_cast<double>(value);
    }
    GpuRanked item;
    item.score = value;
    item.class_id = i;
    ranked[i] = item;
  }
  extern __shared__ double scratch[];
  scratch[tid] = partial;
  __syncthreads();
  for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
    if (tid < step) scratch[tid] += scratch[tid + step];
    __syncthreads();
  }
  if (tid == 0) *sum_out = scratch[0];
}

const char* fault_detail(int reason, bool probabilities) {
  switch (reason) {
    case 1:
      return probabilities ? "probability is NaN" : "logit is NaN";
    case 2:
      return probabilities ? "probability is positive infinity" : "logit is infinite";
    case 3:
      return probabilities ? "probability is negative infinity" : "logit is infinite";
    case 4:
      return "probability is below 0";
    default:
      return "probability is above 1";
  }
}

std::string probability_sum_message(const std::string& output_name, double sum) {
  const double delta = std::fabs(sum - 1.0);
  std::ostringstream message;
  message.imbue(std::locale::classic());
  message << std::setprecision(std::numeric_limits<double>::max_digits10);
  message << "classification output '" << output_name << "' probability sum " << sum
          << " differs from 1 by " << delta << ", which is outside tolerance "
          << kClassificationProbabilitySumTolerance;
  return message.str();
}

}  // namespace

ClassificationCudaLaunch launch_classification_cuda(const float* data, int num_classes,
                                                    long long stride, int top_k, bool probabilities,
                                                    const std::string& output_name, int device_id) {
  // The fault word packs the class id in the upper bits; keep the encoding
  // valid. A classification head with more than 2^28 classes is not a real
  // deployment target.
  if (num_classes < 1 || num_classes > (INT_MAX >> 3)) {
    throw VieError(ErrorCode::InvalidArgument, "classification class count is outside the CUDA decoder range");
  }
  if (top_k < 1 || top_k > num_classes) {
    throw VieError(ErrorCode::InvalidArgument, "classification top_k is outside 1..num_classes");
  }
  if (stride < 1) {
    throw VieError(ErrorCode::InvalidArgument, "classification class stride must be positive");
  }
  cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
  // Legacy default stream. See ClassificationCudaDecoder.
  cudaStream_t stream = nullptr;
  DeviceBuffer ranked(static_cast<std::size_t>(num_classes) * sizeof(GpuRanked));
  DeviceBuffer fault_buffer(sizeof(unsigned long long));
  DeviceBuffer sum_buffer(sizeof(double));
  const unsigned long long no_fault = std::numeric_limits<unsigned long long>::max();
  const double initial_sum = 0.0;
  cuda_check(cudaMemcpyAsync(fault_buffer.data, &no_fault, sizeof(unsigned long long),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaMemcpyAsync(sum_buffer.data, &initial_sum, sizeof(double), cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync");

  EventPair kernel;
  cuda_check(cudaEventRecord(kernel.start, stream), "cudaEventRecord");
  constexpr int kThreads = 256;
  validate_pack_kernel<<<1, kThreads, kThreads * sizeof(double), stream>>>(
      data, num_classes, stride, probabilities ? 1 : 0,
      static_cast<GpuRanked*>(ranked.data), static_cast<unsigned long long*>(fault_buffer.data),
      static_cast<double*>(sum_buffer.data));
  cuda_check(cudaGetLastError(), "classification validate kernel");
  cuda_check(cudaEventRecord(kernel.stop, stream), "cudaEventRecord");

  EventPair readback;
  cuda_check(cudaEventRecord(readback.start, stream), "cudaEventRecord");
  unsigned long long fault = no_fault;
  double sum = initial_sum;
  cuda_check(cudaMemcpyAsync(&fault, fault_buffer.data, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaMemcpyAsync(&sum, sum_buffer.data, sizeof(double), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaEventRecord(readback.stop, stream), "cudaEventRecord");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  float validate_ms = 0.f;
  cuda_check(cudaEventElapsedTime(&validate_ms, kernel.start, kernel.stop), "cudaEventElapsedTime");
  if (fault != no_fault) {
    const int reason = static_cast<int>(fault & 7u);
    const long long class_id = static_cast<long long>(fault >> 3);
    throw VieError(ErrorCode::InvalidArgument,
                   "classification output '" + output_name + "' class_id " + std::to_string(class_id) + " " +
                       fault_detail(reason, probabilities));
  }
  if (probabilities && std::fabs(sum - 1.0) > kClassificationProbabilitySumTolerance) {
    throw VieError(ErrorCode::InvalidArgument, probability_sum_message(output_name, sum));
  }

  float sort_ms = 0.f;
  if (num_classes > 1) {
    EventPair sort_events;
    cuda_check(cudaEventRecord(sort_events.start, stream), "cudaEventRecord");
    thrust::sort(thrust::device_pointer_cast(static_cast<GpuRanked*>(ranked.data)),
                 thrust::device_pointer_cast(static_cast<GpuRanked*>(ranked.data) + num_classes),
                 GpuRankedOrder{});
    cuda_check(cudaEventRecord(sort_events.stop, stream), "cudaEventRecord");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    cuda_check(cudaEventElapsedTime(&sort_ms, sort_events.start, sort_events.stop), "cudaEventElapsedTime");
  }

  std::vector<GpuRanked> host(static_cast<std::size_t>(top_k));
  float copy_ms = 0.f;
  EventPair copy;
  cuda_check(cudaEventRecord(copy.start, stream), "cudaEventRecord");
  cuda_check(cudaMemcpyAsync(host.data(), ranked.data, static_cast<std::size_t>(top_k) * sizeof(GpuRanked),
                             cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaEventRecord(copy.stop, stream), "cudaEventRecord");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  cuda_check(cudaEventElapsedTime(&copy_ms, copy.start, copy.stop), "cudaEventElapsedTime");

  ClassificationCudaLaunch launch;
  launch.ranked.reserve(host.size());
  for (const GpuRanked& item : host) {
    launch.ranked.push_back(RankedScore{item.class_id, item.score});
  }
  launch.validate_ms = static_cast<double>(validate_ms);
  launch.sort_ms = static_cast<double>(sort_ms);
  launch.d2h_ms = static_cast<double>(copy_ms);
  launch.d2h_bytes = static_cast<std::size_t>(top_k) * sizeof(GpuRanked);
  return launch;
}

}  // namespace vie