#include "obb_cuda.hpp"

#include "vie/core/error.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
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

constexpr float kAngleMin = -0.7853981633974483f;
constexpr float kAngleMax = 2.356194490192345f;
constexpr float kAngleSlack = 1e-3f;
constexpr float kScoreSlack = 1e-4f;
constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kInsideSlack = 1e-3;
constexpr double kIsotropicRelative = 1e-3;

// Fault encoding for the decoder: prediction in bits 63..32, the CPU read-order
// rank in bits 31..20, the kind in bits 19..16, the channel in bits 15..0.
// rank orders the faults exactly as the CPU loop reads them:
//   2 * box channel (cx, cy, width, height), 8 for the angle channel,
//   10 + 2 * cls for a class read, 11 + 2 * cls for the size/angle check an
//   AllAboveThreshold hit performs immediately, and 10 + 2 * num_classes for
//   the BestClass check after every class was read.
// atomicMin then names the same anchor the CPU loop would have thrown on.
constexpr int kFaultRankShift = 20;
constexpr int kFaultKindShift = 16;
constexpr int kFaultRankMask = 0xFFF;
constexpr int kFaultKindMask = 0xF;
constexpr int kFaultChannelMask = 0xFFFF;
constexpr int kFaultNonFinite = 1;
constexpr int kFaultProbability = 2;
constexpr int kFaultKeptBox = 3;
constexpr int kFaultFit = 5;
constexpr int kFaultMappedNonFinite = 6;
constexpr int kFaultMappedSize = 7;
constexpr unsigned long long kNoFault = std::numeric_limits<unsigned long long>::max();
// The maximum class count keeps every rank inside kFaultRankMask.
constexpr int kMaxClasses = 1000;

// Probe fault encoding: pair index in bits 63..32 and the kind in bits 15..0.
constexpr int kProbeNonFinite = 1;
constexpr int kProbeBoxSize = 2;
constexpr int kProbeArea = 3;
constexpr int kProbeIntersection = 4;
constexpr int kProbeUnion = 5;

struct GpuPt {
  double x;
  double y;
};

struct GpuObbBox {
  float cx;
  float cy;
  float width;
  float height;
  float angle;
};

struct GpuObbCandidate {
  GpuObbBox box;
  float score;
  int class_id;
  int anchor_index;
};

// One mapped candidate. corners are the corners_of() result in ccw order and
// area is polygon_area(ccw_corners()), both prepared once so the NMS loop does
// not repeat the trigonometry the CPU repeats per IoU call.
struct GpuObbMapped {
  GpuObbBox box;
  float score;
  int class_id;
  int anchor_index;
  GpuPt corners[4];
  double area;
};

struct GpuObbObject {
  int class_id;
  float score;
  GpuObbBox box;
  int anchor_index;
};

static_assert(sizeof(GpuObbBox) == 20, "obb box record gained padding");
static_assert(sizeof(GpuObbCandidate) == 32, "obb candidate record gained padding");
static_assert(sizeof(GpuObbMapped) == 104, "obb mapped record gained padding");
static_assert(sizeof(GpuObbObject) == 32, "obb object record gained padding");
static_assert(sizeof(OrientedBox) == 20, "public oriented box must stay five floats");

struct CandidateOrder {
  __host__ __device__ bool operator()(const GpuObbCandidate& left, const GpuObbCandidate& right) const {
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

float elapsed_ms(const EventPair& events, cudaStream_t stream) {
  cuda_check(cudaEventRecord(events.stop, stream), "cudaEventRecord");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  float elapsed = 0.f;
  cuda_check(cudaEventElapsedTime(&elapsed, events.start, events.stop), "cudaEventElapsedTime");
  return elapsed;
}

__device__ void record_fault(unsigned long long* fault, int prediction, int rank, int kind, int channel) {
  const unsigned long long value =
      (static_cast<unsigned long long>(static_cast<unsigned int>(prediction)) << 32) |
      (static_cast<unsigned long long>(rank & kFaultRankMask) << kFaultRankShift) |
      (static_cast<unsigned long long>(kind & kFaultKindMask) << kFaultKindShift) |
      static_cast<unsigned long long>(channel & kFaultChannelMask);
  atomicMin(fault, value);
}

__device__ GpuPt cuda_sub(GpuPt left, GpuPt right) { return {left.x - right.x, left.y - right.y}; }

__device__ double cuda_cross(GpuPt left, GpuPt right) { return left.x * right.y - left.y * right.x; }

// Mirrors corners_of() in obb_geometry.cpp.
__device__ void cuda_corners(const GpuObbBox& box, GpuPt corners[4]) {
  const double cosine = cos(static_cast<double>(box.angle));
  const double sine = sin(static_cast<double>(box.angle));
  const double half_w = static_cast<double>(box.width) * 0.5;
  const double half_h = static_cast<double>(box.height) * 0.5;
  const GpuPt along{half_w * cosine, half_w * sine};
  const GpuPt across{-half_h * sine, half_h * cosine};
  const GpuPt center{static_cast<double>(box.cx), static_cast<double>(box.cy)};
  corners[0] = {center.x + along.x + across.x, center.y + along.y + across.y};
  corners[1] = {center.x + along.x - across.x, center.y + along.y - across.y};
  corners[2] = {center.x - along.x - across.x, center.y - along.y - across.y};
  corners[3] = {center.x - along.x + across.x, center.y - along.y + across.y};
}

// Mirrors signed_area() in obb_geometry.cpp.
__device__ double cuda_signed_area(const GpuPt* points, int count) {
  double sum = 0;
  for (int index = 0; index < count; ++index) {
    sum += cuda_cross(points[index], points[(index + 1) % count]);
  }
  return sum * 0.5;
}

// Mirrors polygon_area() in obb_geometry.cpp.
__device__ double cuda_polygon_area(const GpuPt* points, int count) {
  return fabs(cuda_signed_area(points, count));
}

// Mirrors line_hit() in obb_geometry.cpp.
__device__ GpuPt cuda_line_hit(GpuPt start, GpuPt end, GpuPt edge_a, GpuPt edge_b) {
  const GpuPt se = cuda_sub(end, start);
  const GpuPt ab = cuda_sub(edge_b, edge_a);
  const double denom = cuda_cross(se, ab);
  if (fabs(denom) < 1e-18) return end;
  double t = cuda_cross(cuda_sub(edge_a, start), ab) / denom;
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  return {start.x + se.x * t, start.y + se.y * t};
}

// Convex clipping capacity. Four input points grow to at most eight after four
// half-plane clips; the CPU only removes duplicates after the last clip.
constexpr int kClipCapacity = 24;

// Mirrors clip_convex() in obb_geometry.cpp, including the order in which a
// crossing point and the kept endpoint are appended.
__device__ int cuda_clip_convex(const GpuPt* subject, int subject_count, const GpuPt* clip, int clip_count,
                                GpuPt* output) {
  GpuPt front[kClipCapacity];
  GpuPt back[kClipCapacity];
  for (int index = 0; index < subject_count; ++index) front[index] = subject[index];
  int count = subject_count;
  for (int edge = 0; edge < clip_count; ++edge) {
    if (count == 0) break;
    const GpuPt edge_a = clip[edge];
    const GpuPt edge_b = clip[(edge + 1) % clip_count];
    const int input_count = count;
    int written = 0;
    for (int index = 0; index < input_count; ++index) {
      const GpuPt start = front[index];
      const GpuPt end = front[(index + 1) % input_count];
      const bool start_in = cuda_cross(cuda_sub(edge_b, edge_a), cuda_sub(start, edge_a)) >= -1e-9;
      const bool end_in = cuda_cross(cuda_sub(edge_b, edge_a), cuda_sub(end, edge_a)) >= -1e-9;
      if (start_in && end_in) {
        back[written++] = end;
      } else if (start_in && !end_in) {
        back[written++] = cuda_line_hit(start, end, edge_a, edge_b);
      } else if (!start_in && end_in) {
        back[written++] = cuda_line_hit(start, end, edge_a, edge_b);
        back[written++] = end;
      }
    }
    for (int index = 0; index < written; ++index) front[index] = back[index];
    count = written;
  }
  int cleaned = 0;
  for (int index = 0; index < count; ++index) {
    const GpuPt point = front[index];
    if (cleaned > 0) {
      const double dx = point.x - output[cleaned - 1].x;
      const double dy = point.y - output[cleaned - 1].y;
      if (dx * dx + dy * dy <= 1e-18) continue;
    }
    output[cleaned++] = point;
  }
  if (cleaned >= 2) {
    const double dx = output[0].x - output[cleaned - 1].x;
    const double dy = output[0].y - output[cleaned - 1].y;
    if (dx * dx + dy * dy <= 1e-18) --cleaned;
  }
  return cleaned;
}

// Mirrors require_box() in obb_geometry.cpp for a mapped box.
__device__ int cuda_check_box(const GpuObbBox& box) {
  if (!isfinite(box.cx) || !isfinite(box.cy) || !isfinite(box.width) || !isfinite(box.height) ||
      !isfinite(box.angle)) {
    return kFaultMappedNonFinite;
  }
  if (!(box.width > 0.f) || !(box.height > 0.f)) return kFaultMappedSize;
  return 0;
}

// Mirrors corners_inside() in obb_geometry.cpp.
__device__ bool cuda_corners_inside(const GpuObbBox& box, int width, int height) {
  GpuPt points[4];
  cuda_corners(box, points);
  for (int index = 0; index < 4; ++index) {
    const GpuPt& point = points[index];
    if (point.x < -kInsideSlack || point.y < -kInsideSlack ||
        point.x > static_cast<double>(width) + kInsideSlack ||
        point.y > static_cast<double>(height) + kInsideSlack) {
      return false;
    }
  }
  return true;
}

// Mirrors fit_minimum_area() in obb_geometry.cpp. The tie-break comparisons use
// the stored float angle exactly like the CPU code does, including the float
// fabs of best.angle.
__device__ bool cuda_fit_minimum_area(const GpuPt points[4], GpuObbBox& best) {
  double best_area = 0;
  float best_angle = 0.f;
  bool found = false;
  for (int edge = 0; edge < 4; ++edge) {
    const GpuPt delta = cuda_sub(points[(edge + 1) % 4], points[edge]);
    const double length = hypot(delta.x, delta.y);
    if (!(length > 1e-12)) continue;
    const GpuPt axis{delta.x / length, delta.y / length};
    const GpuPt perp{-axis.y, axis.x};
    double min_u = 0;
    double max_u = 0;
    double min_v = 0;
    double max_v = 0;
    for (int index = 0; index < 4; ++index) {
      const double along = points[index].x * axis.x + points[index].y * axis.y;
      const double across = points[index].x * perp.x + points[index].y * perp.y;
      if (index == 0 || along < min_u) min_u = along;
      if (index == 0 || along > max_u) max_u = along;
      if (index == 0 || across < min_v) min_v = across;
      if (index == 0 || across > max_v) max_v = across;
    }
    double width = max_u - min_u;
    double height = max_v - min_v;
    if (!(width > 0) || !(height > 0)) continue;
    const double center_u = 0.5 * (min_u + max_u);
    const double center_v = 0.5 * (min_v + max_v);
    double angle = atan2(axis.y, axis.x);
    while (angle >= 3.0 * kPi / 4.0) {
      angle -= kPi / 2.0;
      const double swap = width;
      width = height;
      height = swap;
    }
    while (angle < -kPi / 4.0) {
      angle += kPi / 2.0;
      const double swap = width;
      width = height;
      height = swap;
    }
    const double area = width * height;
    const bool better = !found || area < best_area - 1e-9 ||
                        (fabs(area - best_area) <= 1e-9 && fabs(angle) < fabsf(best_angle) - 1e-12) ||
                        (fabs(area - best_area) <= 1e-9 &&
                         fabs(fabs(angle) - fabsf(best_angle)) <= 1e-12 && angle < static_cast<double>(best_angle));
    if (better) {
      best.cx = static_cast<float>(center_u * axis.x + center_v * perp.x);
      best.cy = static_cast<float>(center_u * axis.y + center_v * perp.y);
      best.width = static_cast<float>(width);
      best.height = static_cast<float>(height);
      best.angle = static_cast<float>(angle);
      best_angle = best.angle;
      best_area = area;
      found = true;
    }
  }
  if (!found || !isfinite(best.cx) || !(best.width > 0.f) || !(best.height > 0.f)) return false;
  return true;
}

struct GpuObbGeometry {
  double scale_x;
  double scale_y;
  double origin_x;
  double origin_y;
  double mean_scale;
  int original_width;
  int original_height;
  int isotropic;
};

// Mirrors map_obb_to_original() in obb_geometry.cpp. Returns true when the box
// survives; fault_kind is 0 unless the CPU path would have thrown.
__device__ bool cuda_map_box(const GpuObbBox& model, const GpuObbGeometry& geometry, GpuObbBox& original,
                             int& fault_kind) {
  fault_kind = 0;
  GpuPt model_points[4];
  cuda_corners(model, model_points);
  GpuPt mapped[4];
  for (int index = 0; index < 4; ++index) {
    mapped[index] = {(model_points[index].x - geometry.origin_x) / geometry.scale_x,
                     (model_points[index].y - geometry.origin_y) / geometry.scale_y};
  }
  for (int index = 0; index < 4; ++index) {
    const GpuPt& point = mapped[index];
    if (point.x < -kInsideSlack || point.y < -kInsideSlack ||
        point.x > static_cast<double>(geometry.original_width) + kInsideSlack ||
        point.y > static_cast<double>(geometry.original_height) + kInsideSlack) {
      return false;
    }
  }
  if (geometry.isotropic != 0) {
    original.cx =
        static_cast<float>((static_cast<double>(model.cx) - geometry.origin_x) / geometry.scale_x);
    original.cy =
        static_cast<float>((static_cast<double>(model.cy) - geometry.origin_y) / geometry.scale_y);
    original.width = static_cast<float>(static_cast<double>(model.width) / geometry.mean_scale);
    original.height = static_cast<float>(static_cast<double>(model.height) / geometry.mean_scale);
    original.angle = model.angle;
  } else {
    if (!cuda_fit_minimum_area(mapped, original)) {
      fault_kind = kFaultFit;
      return false;
    }
  }
  const int check = cuda_check_box(original);
  if (check != 0) {
    fault_kind = check;
    return false;
  }
  return cuda_corners_inside(original, geometry.original_width, geometry.original_height);
}

// Fills the CCW corners and the polygon area once per mapped candidate.
__device__ void cuda_prepare_mapped(GpuObbMapped& record) {
  GpuPt corners[4];
  cuda_corners(record.box, corners);
  double area = cuda_signed_area(corners, 4);
  if (area < 0) {
    const GpuPt first = corners[0];
    const GpuPt second = corners[1];
    corners[0] = corners[3];
    corners[1] = corners[2];
    corners[2] = second;
    corners[3] = first;
    area = -area;
  }
  for (int index = 0; index < 4; ++index) record.corners[index] = corners[index];
  record.area = area;
}

// Mirrors rotated_intersection_over_union() after the CCW/area preparation.
__device__ float cuda_iou_prepared(const GpuObbMapped& left, const GpuObbMapped& right) {
  GpuPt intersection_points[kClipCapacity];
  const int count = cuda_clip_convex(left.corners, 4, right.corners, 4, intersection_points);
  double intersection = cuda_polygon_area(intersection_points, count);
  if (!isfinite(intersection) || intersection < 0) return 0.f;
  if (intersection > left.area) intersection = left.area;
  if (intersection > right.area) intersection = right.area;
  const double union_area = left.area + right.area - intersection;
  if (!(union_area > 0) || !isfinite(union_area)) return 0.f;
  const double ratio = intersection / union_area;
  if (ratio <= 0) return 0.f;
  if (ratio >= 1) return 1.f;
  return static_cast<float>(ratio);
}

// Probe path: the full CPU IoU, including its validation.
__device__ int cuda_box_iou(const GpuObbBox& left, const GpuObbBox& right, float& value, int& fault_kind) {
  fault_kind = 0;
  if (!isfinite(left.cx) || !isfinite(left.cy) || !isfinite(left.width) || !isfinite(left.height) ||
      !isfinite(left.angle) || !isfinite(right.cx) || !isfinite(right.cy) || !isfinite(right.width) ||
      !isfinite(right.height) || !isfinite(right.angle)) {
    fault_kind = kProbeNonFinite;
    return 0;
  }
  if (!(left.width > 0.f) || !(left.height > 0.f) || !(right.width > 0.f) || !(right.height > 0.f)) {
    fault_kind = kProbeBoxSize;
    return 0;
  }
  GpuPt left_points[4];
  GpuPt right_points[4];
  cuda_corners(left, left_points);
  cuda_corners(right, right_points);
  const double left_area = cuda_signed_area(left_points, 4);
  const double right_area = cuda_signed_area(right_points, 4);
  const double left_size = fabs(left_area);
  const double right_size = fabs(right_area);
  if (left_area < 0) {
    const GpuPt first = left_points[0];
    const GpuPt second = left_points[1];
    left_points[0] = left_points[3];
    left_points[1] = left_points[2];
    left_points[2] = second;
    left_points[3] = first;
  }
  if (right_area < 0) {
    const GpuPt first = right_points[0];
    const GpuPt second = right_points[1];
    right_points[0] = right_points[3];
    right_points[1] = right_points[2];
    right_points[2] = second;
    right_points[3] = first;
  }
  if (!(left_size > 0) || !(right_size > 0) || !isfinite(left_size) || !isfinite(right_size)) {
    fault_kind = kProbeArea;
    return 0;
  }
  GpuPt intersection_points[kClipCapacity];
  const int count = cuda_clip_convex(left_points, 4, right_points, 4, intersection_points);
  double intersection = cuda_polygon_area(intersection_points, count);
  if (!isfinite(intersection) || intersection < 0) {
    fault_kind = kProbeIntersection;
    return 0;
  }
  if (intersection > left_size) intersection = left_size;
  if (intersection > right_size) intersection = right_size;
  const double union_area = left_size + right_size - intersection;
  if (!(union_area > 0) || !isfinite(union_area)) {
    fault_kind = kProbeUnion;
    return 0;
  }
  const double ratio = intersection / union_area;
  if (ratio <= 0) {
    value = 0.f;
  } else if (ratio >= 1) {
    value = 1.f;
  } else {
    value = static_cast<float>(ratio);
  }
  return 1;
}

__device__ bool cuda_read_finite(const float* data, int channel_axis, int channels, int predictions, int channel,
                                 int prediction, int rank, unsigned long long* fault, float& out) {
  const int index = channel_axis == 1 ? channel * predictions + prediction : prediction * channels + channel;
  const float value = data[index];
  if (!isfinite(value)) {
    record_fault(fault, prediction, rank, kFaultNonFinite, channel);
    return false;
  }
  out = value;
  return true;
}

__device__ float cuda_probability(float value) {
  if (value < 0.f) return 0.f;
  if (value > 1.f) return 1.f;
  return value;
}

// One thread per anchor. Reads the five box channels first, then the class
// scores, exactly in the CPU order, so a fault names the same anchor/channel.
__global__ void filter_kernel(const float* data, int channel_axis, int channels, int predictions, int num_classes,
                              int best_class, float threshold, GpuObbCandidate* candidates, int* candidate_count,
                              unsigned long long* fault) {
  const int prediction = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (prediction >= predictions) return;
  const int angle_channel = 4 + num_classes;
  float cx = 0.f;
  float cy = 0.f;
  float width = 0.f;
  float height = 0.f;
  float angle = 0.f;
  if (!cuda_read_finite(data, channel_axis, channels, predictions, 0, prediction, 0, fault, cx)) return;
  if (!cuda_read_finite(data, channel_axis, channels, predictions, 1, prediction, 2, fault, cy)) return;
  if (!cuda_read_finite(data, channel_axis, channels, predictions, 2, prediction, 4, fault, width)) return;
  if (!cuda_read_finite(data, channel_axis, channels, predictions, 3, prediction, 6, fault, height)) return;
  if (!cuda_read_finite(data, channel_axis, channels, predictions, angle_channel, prediction, 8, fault, angle)) {
    return;
  }
  const bool bad_box = !(width > 0.f) || !(height > 0.f) || angle < kAngleMin - kAngleSlack ||
                       angle > kAngleMax + kAngleSlack;
  if (best_class != 0) {
    int best = 0;
    float best_score = -1.f;
    for (int cls = 0; cls < num_classes; ++cls) {
      const int channel = 4 + cls;
      const int index =
          channel_axis == 1 ? channel * predictions + prediction : prediction * channels + channel;
      const float raw = data[index];
      if (!isfinite(raw)) {
        record_fault(fault, prediction, 10 + 2 * cls, kFaultNonFinite, channel);
        return;
      }
      if (raw < -kScoreSlack || raw > 1.f + kScoreSlack) {
        record_fault(fault, prediction, 10 + 2 * cls, kFaultProbability, channel);
        return;
      }
      const float score = cuda_probability(raw);
      if (score > best_score || (score == best_score && cls < best)) {
        best_score = score;
        best = cls;
      }
    }
    if (best_score < threshold) return;
    if (bad_box) {
      record_fault(fault, prediction, 10 + 2 * num_classes, kFaultKeptBox, 0);
      return;
    }
    const int slot = atomicAdd(candidate_count, 1);
    GpuObbCandidate candidate;
    candidate.box = GpuObbBox{cx, cy, width, height, angle};
    candidate.score = best_score;
    candidate.class_id = best;
    candidate.anchor_index = prediction;
    candidates[slot] = candidate;
    return;
  }
  for (int cls = 0; cls < num_classes; ++cls) {
    const int channel = 4 + cls;
    const int index = channel_axis == 1 ? channel * predictions + prediction : prediction * channels + channel;
    const float raw = data[index];
    if (!isfinite(raw)) {
      record_fault(fault, prediction, 10 + 2 * cls, kFaultNonFinite, channel);
      return;
    }
    if (raw < -kScoreSlack || raw > 1.f + kScoreSlack) {
      record_fault(fault, prediction, 10 + 2 * cls, kFaultProbability, channel);
      return;
    }
    const float score = cuda_probability(raw);
    if (score < threshold) continue;
    if (bad_box) {
      record_fault(fault, prediction, 11 + 2 * cls, kFaultKeptBox, 0);
      return;
    }
    const int slot = atomicAdd(candidate_count, 1);
    GpuObbCandidate candidate;
    candidate.box = GpuObbBox{cx, cy, width, height, angle};
    candidate.score = score;
    candidate.class_id = cls;
    candidate.anchor_index = prediction;
    candidates[slot] = candidate;
  }
}

__global__ void map_kernel(const GpuObbCandidate* candidates, int count, GpuObbGeometry geometry,
                           GpuObbMapped* mapped, int* keep_flags, unsigned long long* fault) {
  const int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (slot >= count) return;
  const GpuObbCandidate candidate = candidates[slot];
  GpuObbBox original;
  int kind = 0;
  const bool keep = cuda_map_box(candidate.box, geometry, original, kind);
  if (kind != 0) {
    record_fault(fault, slot, 0, kind, 0);
    keep_flags[slot] = 0;
    return;
  }
  if (!keep) {
    keep_flags[slot] = 0;
    return;
  }
  GpuObbMapped record;
  record.box = original;
  record.score = candidate.score;
  record.class_id = candidate.class_id;
  record.anchor_index = candidate.anchor_index;
  cuda_prepare_mapped(record);
  mapped[slot] = record;
  keep_flags[slot] = 1;
}

// Single thread, compact only. Copies the surviving mapped records in sorted
// order, which is the order the CPU loop keeps after its continue-on-drop pass.
__global__ void compact_kernel(const GpuObbMapped* input, const int* keep, int count, GpuObbMapped* output,
                               int* kept) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int written = 0;
  for (int index = 0; index < count; ++index) {
    if (keep[index] == 0) continue;
    output[written] = input[index];
    ++written;
  }
  *kept = written;
}

// Mirrors rotated_nms() in obb_geometry.cpp: the sorted order is the priority,
// max_detections stops the loop, IoU >= threshold suppresses, and the class
// filter runs before the IoU call. iou_calls counts the Polygon IoU evaluations
// the CPU loop would have performed.
__global__ void nms_kernel(const GpuObbMapped* items, int count, float iou_threshold, int class_aware,
                           int max_detections, int* kept_indices, int* kept_count,
                           unsigned long long* iou_calls) {
  if (blockIdx.x != 0) return;
  extern __shared__ char obb_suppressed[];
  char* suppressed = obb_suppressed;
  __shared__ int kept;
  __shared__ int take;
  __shared__ unsigned long long block_calls;
  unsigned long long local_calls = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) suppressed[i] = 0;
  if (threadIdx.x == 0) {
    kept = 0;
    block_calls = 0;
  }
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
      if (class_aware != 0 && items[j].class_id != items[i].class_id) continue;
      ++local_calls;
      if (cuda_iou_prepared(items[i], items[j]) >= iou_threshold) suppressed[j] = 1;
    }
    __syncthreads();
  }
  if (local_calls > 0) atomicAdd(&block_calls, local_calls);
  __syncthreads();
  if (threadIdx.x == 0) {
    *kept_count = kept;
    atomicAdd(iou_calls, block_calls);
  }
}

__global__ void select_prefix_kernel(int count, int max_detections, int* kept_indices, int* kept_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int limit = min(count, max_detections);
  for (int index = 0; index < limit; ++index) kept_indices[index] = index;
  *kept_count = limit;
}

__global__ void pack_kernel(const GpuObbMapped* mapped, const int* kept_indices, int kept_count,
                            GpuObbObject* objects) {
  const int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (slot >= kept_count) return;
  const GpuObbMapped& item = mapped[kept_indices[slot]];
  GpuObbObject object;
  object.class_id = item.class_id;
  object.score = item.score;
  object.box = item.box;
  object.anchor_index = item.anchor_index;
  objects[slot] = object;
}

__global__ void iou_probe_kernel(const GpuObbBox* left, const GpuObbBox* right, int count, float* values,
                                 unsigned long long* fault) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  float value = 0.f;
  int kind = 0;
  const int valid = cuda_box_iou(left[index], right[index], value, kind);
  if (valid == 0) {
    const unsigned long long encoded =
        (static_cast<unsigned long long>(static_cast<unsigned int>(index)) << 32) |
        static_cast<unsigned long long>(kind & kFaultChannelMask);
    atomicMin(fault, encoded);
    return;
  }
  values[index] = value;
}

[[noreturn]] void throw_fault(unsigned long long fault) {
  const int kind = static_cast<int>((fault >> kFaultKindShift) & kFaultKindMask);
  const int channel = static_cast<int>(fault & kFaultChannelMask);
  const long long index = static_cast<long long>(fault >> 32);
  switch (kind) {
    case kFaultNonFinite:
      throw VieError(ErrorCode::InvalidArgument,
                     "yolov8_obb_raw prediction " + std::to_string(index) + " channel " +
                         std::to_string(channel) + " is not finite");
    case kFaultProbability:
      throw VieError(ErrorCode::InvalidArgument,
                     "yolov8_obb_raw class score at prediction " + std::to_string(index) + " channel " +
                         std::to_string(channel) + " is outside [0, 1]");
    case kFaultKeptBox:
      throw VieError(ErrorCode::InvalidArgument,
                     "yolov8_obb_raw kept anchor " + std::to_string(index) +
                         " has a non-positive size or an angle outside [-pi/4, 3pi/4]");
    case kFaultFit:
      throw VieError(ErrorCode::InvalidArgument, "rotated rectangle fit did not produce a positive box");
    case kFaultMappedNonFinite:
      throw VieError(ErrorCode::InvalidArgument, "mapped obb has a non-finite value");
    case kFaultMappedSize:
      throw VieError(ErrorCode::InvalidArgument, "mapped obb width and height must be positive");
    default:
      throw VieError(ErrorCode::InvalidArgument, "yolov8_obb_raw CUDA decoder fault is unknown");
  }
}

[[noreturn]] void throw_probe_fault(unsigned long long fault) {
  const int kind = static_cast<int>(fault & kFaultChannelMask);
  switch (kind) {
    case kProbeNonFinite:
      throw VieError(ErrorCode::InvalidArgument, "rotated box has a non-finite value");
    case kProbeBoxSize:
      throw VieError(ErrorCode::InvalidArgument, "rotated box width and height must be positive");
    case kProbeArea:
      throw VieError(ErrorCode::InvalidArgument, "rotated box area is not positive");
    case kProbeIntersection:
      throw VieError(ErrorCode::InvalidArgument, "rotated intersection area is not finite");
    case kProbeUnion:
      throw VieError(ErrorCode::InvalidArgument, "rotated union area is not positive");
    default:
      throw VieError(ErrorCode::InvalidArgument, "rotated box probe fault is unknown");
  }
}

void check_geometry(const GeometryTransform& geometry) {
  if (geometry.original_width <= 0 || geometry.original_height <= 0 || geometry.scale_x == 0.0 ||
      geometry.scale_y == 0.0 || !std::isfinite(geometry.scale_x) || !std::isfinite(geometry.scale_y)) {
    throw VieError(ErrorCode::InvalidArgument, "geometry scale or original size is invalid");
  }
}

}  // namespace

ObbCudaLaunch launch_obb_cuda(const Tensor& tensor, const ObbDecodeConfig& config,
                              const GeometryTransform& geometry, int device_id) {
  const std::vector<std::int64_t>& shape = tensor.shape();
  const std::size_t axis = static_cast<std::size_t>(config.channel_axis);
  const std::int64_t channels64 = shape[axis];
  const std::int64_t predictions64 = shape[config.channel_axis == 1 ? 2 : 1];
  if (config.num_classes < 1 || config.num_classes > kMaxClasses) {
    throw VieError(ErrorCode::Unsupported,
                   "yolov8_obb_raw CUDA decoder class count is outside the supported range");
  }
  if (channels64 < 1 || channels64 > static_cast<std::int64_t>(kFaultChannelMask)) {
    throw VieError(ErrorCode::Unsupported,
                   "yolov8_obb_raw CUDA decoder channel count is outside the supported range");
  }
  if (predictions64 < 1 || predictions64 > 1000000) {
    throw VieError(ErrorCode::Unsupported,
                   "yolov8_obb_raw CUDA decoder prediction count is outside 1..1000000");
  }
  if (predictions64 * channels64 > static_cast<std::int64_t>(INT_MAX)) {
    throw VieError(ErrorCode::Unsupported, "yolov8_obb_raw CUDA decoder tensor is too large");
  }
  const int channels = static_cast<int>(channels64);
  const int predictions = static_cast<int>(predictions64);
  const bool best_class = *config.class_selection == ClassSelection::BestClass;
  const long long unlifted =
      best_class ? static_cast<long long>(predictions)
                 : static_cast<long long>(predictions) * static_cast<long long>(config.num_classes);
  if (unlifted > 1000000) {
    throw VieError(ErrorCode::Unsupported, "CUDA obb decoder candidate capacity exceeds 1000000");
  }
  const int capacity = static_cast<int>(unlifted);
  const int top_capacity = std::min(capacity, *config.pre_nms_top_k);
  const int keep_capacity = std::min(*config.max_detections, top_capacity);

  cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
  // Legacy default stream, the same stream the ORT CUDA EP executes on.
  cudaStream_t stream = nullptr;
  const float* data = reinterpret_cast<const float*>(tensor.device_data());

  DeviceBuffer candidates(static_cast<std::size_t>(std::max(capacity, 1)) * sizeof(GpuObbCandidate));
  DeviceBuffer candidate_count(sizeof(int));
  DeviceBuffer fault_buffer(sizeof(unsigned long long));
  DeviceBuffer mapped(static_cast<std::size_t>(std::max(top_capacity, 1)) * sizeof(GpuObbMapped));
  DeviceBuffer keep_flags(static_cast<std::size_t>(std::max(top_capacity, 1)) * sizeof(int));
  DeviceBuffer mapped_count(sizeof(int));
  DeviceBuffer map_fault(sizeof(unsigned long long));
  DeviceBuffer kept_indices(static_cast<std::size_t>(std::max(keep_capacity, 1)) * sizeof(int));
  DeviceBuffer kept_count(sizeof(int));
  DeviceBuffer iou_call_count(sizeof(unsigned long long));
  DeviceBuffer objects(static_cast<std::size_t>(std::max(keep_capacity, 1)) * sizeof(GpuObbObject));

  cuda_check(cudaMemsetAsync(candidate_count.data, 0, sizeof(int), stream), "cudaMemsetAsync");
  cuda_check(cudaMemsetAsync(mapped_count.data, 0, sizeof(int), stream), "cudaMemsetAsync");
  cuda_check(cudaMemsetAsync(kept_count.data, 0, sizeof(int), stream), "cudaMemsetAsync");
  cuda_check(cudaMemsetAsync(iou_call_count.data, 0, sizeof(unsigned long long), stream), "cudaMemsetAsync");
  cuda_check(cudaMemcpyAsync(fault_buffer.data, &kNoFault, sizeof(unsigned long long), cudaMemcpyHostToDevice,
                             stream),
             "cudaMemcpyAsync");
  cuda_check(cudaMemcpyAsync(map_fault.data, &kNoFault, sizeof(unsigned long long), cudaMemcpyHostToDevice,
                             stream),
             "cudaMemcpyAsync");

  float validate_ms = 0.f;
  {
    EventPair filter;
    cuda_check(cudaEventRecord(filter.start, stream), "cudaEventRecord");
    constexpr int kThreads = 128;
    const int blocks = (predictions + kThreads - 1) / kThreads;
    filter_kernel<<<blocks, kThreads, 0, stream>>>(
        data, config.channel_axis, channels, predictions, config.num_classes, best_class ? 1 : 0,
        *config.confidence_threshold, static_cast<GpuObbCandidate*>(candidates.data),
        static_cast<int*>(candidate_count.data), static_cast<unsigned long long*>(fault_buffer.data));
    cuda_check(cudaGetLastError(), "obb filter kernel");
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
    throw VieError(ErrorCode::Internal, "CUDA obb candidate count is invalid");
  }

  float sort_ms = 0.f;
  if (raw_count > 1) {
    EventPair sort_events;
    cuda_check(cudaEventRecord(sort_events.start, stream), "cudaEventRecord");
    thrust::sort(thrust::cuda::par.on(stream),
                 thrust::device_pointer_cast(static_cast<GpuObbCandidate*>(candidates.data)),
                 thrust::device_pointer_cast(static_cast<GpuObbCandidate*>(candidates.data) + raw_count),
                 CandidateOrder{});
    sort_ms = elapsed_ms(sort_events, stream);
  }

  const int top = std::min(raw_count, *config.pre_nms_top_k);
  int mapped_total = 0;
  float map_ms = 0.f;
  if (top > 0) {
    // The CPU decoder reaches this check inside map_obb_to_original, after the
    // filter loop and only when at least one candidate survived it.
    check_geometry(geometry);
    GpuObbGeometry gpu_geometry;
    gpu_geometry.scale_x = geometry.scale_x;
    gpu_geometry.scale_y = geometry.scale_y;
    gpu_geometry.origin_x = static_cast<double>(geometry_model_origin_x(geometry));
    gpu_geometry.origin_y = static_cast<double>(geometry_model_origin_y(geometry));
    const double max_scale = std::max(std::fabs(geometry.scale_x), std::fabs(geometry.scale_y));
    const double relative = std::fabs(geometry.scale_x - geometry.scale_y) / max_scale;
    gpu_geometry.isotropic = relative <= kIsotropicRelative ? 1 : 0;
    gpu_geometry.mean_scale = 0.5 * (geometry.scale_x + geometry.scale_y);
    gpu_geometry.original_width = geometry.original_width;
    gpu_geometry.original_height = geometry.original_height;

    EventPair map_events;
    cuda_check(cudaEventRecord(map_events.start, stream), "cudaEventRecord");
    constexpr int kMapThreads = 128;
    const int map_blocks = (top + kMapThreads - 1) / kMapThreads;
    map_kernel<<<map_blocks, kMapThreads, 0, stream>>>(
        static_cast<const GpuObbCandidate*>(candidates.data), top, gpu_geometry,
        static_cast<GpuObbMapped*>(mapped.data), static_cast<int*>(keep_flags.data),
        static_cast<unsigned long long*>(map_fault.data));
    cuda_check(cudaGetLastError(), "obb map kernel");
    compact_kernel<<<1, 1, 0, stream>>>(static_cast<const GpuObbMapped*>(mapped.data),
                                        static_cast<const int*>(keep_flags.data), top,
                                        static_cast<GpuObbMapped*>(mapped.data),
                                        static_cast<int*>(mapped_count.data));
    cuda_check(cudaGetLastError(), "obb compact kernel");
    map_ms = elapsed_ms(map_events, stream);
    unsigned long long map_fault_value = kNoFault;
    cuda_check(cudaMemcpyAsync(&map_fault_value, map_fault.data, sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaMemcpyAsync(&mapped_total, mapped_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    if (map_fault_value != kNoFault) throw_fault(map_fault_value);
    if (mapped_total < 0 || mapped_total > top) {
      throw VieError(ErrorCode::Internal, "CUDA obb mapped count is invalid");
    }
  }

  int kept_total = 0;
  float nms_ms = 0.f;
  long long iou_calls = 0;
  if (mapped_total > 0) {
    EventPair nms_events;
    cuda_check(cudaEventRecord(nms_events.start, stream), "cudaEventRecord");
    if (*config.nms_enabled) {
      nms_kernel<<<1, 128, static_cast<std::size_t>(mapped_total), stream>>>(
          static_cast<const GpuObbMapped*>(mapped.data), mapped_total, *config.iou_threshold,
          *config.class_aware ? 1 : 0, *config.max_detections, static_cast<int*>(kept_indices.data),
          static_cast<int*>(kept_count.data), static_cast<unsigned long long*>(iou_call_count.data));
      cuda_check(cudaGetLastError(), "obb nms kernel");
    } else {
      select_prefix_kernel<<<1, 1, 0, stream>>>(mapped_total, *config.max_detections,
                                                static_cast<int*>(kept_indices.data),
                                                static_cast<int*>(kept_count.data));
      cuda_check(cudaGetLastError(), "obb select kernel");
    }
    nms_ms = elapsed_ms(nms_events, stream);
    cuda_check(cudaMemcpyAsync(&kept_total, kept_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    if (*config.nms_enabled) {
      unsigned long long calls = 0;
      cuda_check(cudaMemcpyAsync(&calls, iou_call_count.data, sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync");
      cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      iou_calls = static_cast<long long>(calls);
    } else {
      cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    if (kept_total < 0 || kept_total > keep_capacity) {
      throw VieError(ErrorCode::Internal, "CUDA obb kept count is invalid");
    }
  }

  std::vector<GpuObbObject> host_objects(static_cast<std::size_t>(std::max(kept_total, 0)));
  float copy_ms = 0.f;
  if (kept_total > 0) {
    EventPair copy;
    cuda_check(cudaEventRecord(copy.start, stream), "cudaEventRecord");
    constexpr int kThreads = 128;
    const int blocks = (kept_total + kThreads - 1) / kThreads;
    pack_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<const GpuObbMapped*>(mapped.data),
                                                 static_cast<const int*>(kept_indices.data), kept_total,
                                                 static_cast<GpuObbObject*>(objects.data));
    cuda_check(cudaGetLastError(), "obb pack kernel");
    cuda_check(cudaMemcpyAsync(host_objects.data(), objects.data,
                               static_cast<std::size_t>(kept_total) * sizeof(GpuObbObject),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    copy_ms = elapsed_ms(copy, stream);
  }

  ObbCudaLaunch launch;
  launch.candidates_after_filter = raw_count;
  launch.candidates_before_nms = mapped_total;
  launch.iou_calls = iou_calls;
  launch.validate_ms = static_cast<double>(validate_ms);
  launch.sort_ms = static_cast<double>(sort_ms);
  launch.map_ms = static_cast<double>(map_ms);
  launch.nms_ms = static_cast<double>(nms_ms);
  launch.d2h_ms = static_cast<double>(copy_ms);
  launch.d2h_bytes = host_objects.size() * sizeof(GpuObbObject);
  launch.objects.reserve(host_objects.size());
  for (const GpuObbObject& item : host_objects) {
    ObbObject object;
    object.class_id = item.class_id;
    object.score = item.score;
    object.box = OrientedBox{item.box.cx, item.box.cy, item.box.width, item.box.height, item.box.angle};
    object.anchor_index = item.anchor_index;
    launch.objects.push_back(object);
  }
  return launch;
}

ObbCudaIouBatch launch_obb_iou_cuda(const std::vector<OrientedBox>& left,
                                    const std::vector<OrientedBox>& right, int device_id) {
  if (left.size() != right.size()) {
    throw VieError(ErrorCode::InvalidArgument, "obb iou probe requires equal left and right counts");
  }
  ObbCudaIouBatch batch;
  batch.values.assign(left.size(), 0.f);
  if (left.empty()) return batch;
  if (left.size() > 1000000) {
    throw VieError(ErrorCode::Unsupported, "obb iou probe pair count exceeds 1000000");
  }
  const int count = static_cast<int>(left.size());
  cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
  cudaStream_t stream = nullptr;
  DeviceBuffer left_buffer(left.size() * sizeof(GpuObbBox));
  DeviceBuffer right_buffer(right.size() * sizeof(GpuObbBox));
  DeviceBuffer values(left.size() * sizeof(float));
  DeviceBuffer fault_buffer(sizeof(unsigned long long));
  cuda_check(cudaMemcpyAsync(left_buffer.data, left.data(), left.size() * sizeof(GpuObbBox),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaMemcpyAsync(right_buffer.data, right.data(), right.size() * sizeof(GpuObbBox),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaMemcpyAsync(fault_buffer.data, &kNoFault, sizeof(unsigned long long),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync");
  constexpr int kThreads = 128;
  const int blocks = (count + kThreads - 1) / kThreads;
  iou_probe_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<const GpuObbBox*>(left_buffer.data),
                                                    static_cast<const GpuObbBox*>(right_buffer.data), count,
                                                    static_cast<float*>(values.data),
                                                    static_cast<unsigned long long*>(fault_buffer.data));
  cuda_check(cudaGetLastError(), "obb iou probe kernel");
  unsigned long long fault = kNoFault;
  cuda_check(cudaMemcpyAsync(&fault, fault_buffer.data, sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                             stream),
             "cudaMemcpyAsync");
  cuda_check(cudaMemcpyAsync(batch.values.data(), values.data, batch.values.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  if (fault != kNoFault) throw_probe_fault(fault);
  return batch;
}

}  // namespace vie