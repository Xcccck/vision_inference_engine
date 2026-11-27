// Phase 14: CPU/CUDA parity for the yolov8_pose_raw decoder.
//
// Every case feeds the same raw tensor to YoloV8PoseDecoder on the host and
// PoseCudaDecoder on the device. Discrete fields (class id, anchor, keypoint
// count, inside_image) must match exactly; floats use EXPECT_FLOAT_EQ because
// both sides run the same expression tree (the .cu is compiled with
// --fmad=false). The real-model E2E comparison lives in test_pose.cpp.
//
// Only compiled when VIE_ENABLE_CUDA is on.

#include "test_util.hpp"

#include "vie/config/inference_config.hpp"
#include "vie/decoder/pose_cuda_decoder.hpp"
#include "vie/decoder/yolov8_pose_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using vie::ClassSelection;
using vie::DataType;
using vie::DeviceContext;
using vie::DeviceType;
using vie::ErrorCode;
using vie::GeometryTransform;
using vie::Keypoint;
using vie::PoseBatch;
using vie::PoseDecodeConfig;
using vie::PoseObject;
using vie::Tensor;

constexpr int kDevice = 0;
// Payload sizes asserted by the decoder's d2h_bytes accounting.
constexpr std::size_t kObjectBytes = 28;
constexpr std::size_t kKeypointBytes = 16;

PoseDecodeConfig keep_config() {
  PoseDecodeConfig config;
  config.output_name = "output0";
  config.channel_axis = 1;
  config.num_classes = 2;
  config.keypoint_count = 2;
  config.keypoint_dimensions = 3;
  config.class_selection = ClassSelection::BestClass;
  config.confidence_threshold = 0.5f;
  config.nms_enabled = true;
  config.class_aware = true;
  config.iou_threshold = 0.5f;
  config.max_detections = 10;
  config.pre_nms_top_k = 10;
  return config;
}

GeometryTransform geometry(int width, int height, double scale_x = 1.0, double scale_y = 1.0, int pad_left = 0,
                           int pad_top = 0) {
  GeometryTransform transform;
  transform.original_width = width;
  transform.original_height = height;
  transform.output_width = static_cast<int>(static_cast<double>(width) * scale_x) + 2 * pad_left;
  transform.output_height = static_cast<int>(static_cast<double>(height) * scale_y) + 2 * pad_top;
  transform.resized_width = static_cast<int>(static_cast<double>(width) * scale_x);
  transform.resized_height = static_cast<int>(static_cast<double>(height) * scale_y);
  transform.scale_x = scale_x;
  transform.scale_y = scale_y;
  transform.uniform_scale = scale_x == scale_y ? scale_x : 0.0;
  transform.pad_left = pad_left;
  transform.pad_top = pad_top;
  transform.resize_mode = scale_x == scale_y && pad_left == 0 && pad_top == 0 ? vie::ResizeMode::Stretch
                                                                             : vie::ResizeMode::Letterbox;
  return transform;
}

void put(std::vector<float>& data, int predictions, int channel, int prediction, float value) {
  data[static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) +
      static_cast<std::size_t>(prediction)] = value;
}

Tensor make_tensor(const std::vector<float>& channel_major, int channels, int predictions, int axis,
                   DeviceType device) {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
  if (axis == 1) {
    shape = {1, static_cast<std::int64_t>(channels), static_cast<std::int64_t>(predictions)};
    values = channel_major;
  } else {
    shape = {1, static_cast<std::int64_t>(predictions), static_cast<std::int64_t>(channels)};
    values.resize(channel_major.size());
    for (int prediction = 0; prediction < predictions; ++prediction) {
      for (int channel = 0; channel < channels; ++channel) {
        values[static_cast<std::size_t>(prediction) * static_cast<std::size_t>(channels) +
               static_cast<std::size_t>(channel)] =
            channel_major[static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) +
                          static_cast<std::size_t>(prediction)];
      }
    }
  }
  const DeviceContext context(device, device == DeviceType::Cuda ? kDevice : 0);
  return Tensor::from_host(context, DataType::Fp32, shape, values.data(), values.size() * sizeof(float));
}

vie::DecoderInput decode_input(const Tensor& tensor, const GeometryTransform& transform,
                               const std::string& output_name) {
  vie::DecoderInput input;
  input.geometry = transform;
  input.outputs.emplace(output_name, tensor);
  return input;
}

PoseBatch run_cpu(const Tensor& tensor, const PoseDecodeConfig& config, const GeometryTransform& transform) {
  return vie::YoloV8PoseDecoder{}.decode_pose(decode_input(tensor, transform, config.output_name), config,
                                              DeviceContext{});
}

PoseBatch run_cuda(const Tensor& tensor, const PoseDecodeConfig& config, const GeometryTransform& transform) {
  return vie::PoseCudaDecoder{}.decode_pose(decode_input(tensor, transform, config.output_name), config,
                                            DeviceContext(DeviceType::Cuda, kDevice));
}

void expect_parity(const PoseBatch& cpu, const PoseBatch& cuda) {
  ASSERT_EQ(cpu.results.size(), 1u);
  ASSERT_EQ(cuda.results.size(), 1u);
  const vie::PoseResult& left = cpu.results[0];
  const vie::PoseResult& right = cuda.results[0];
  EXPECT_EQ(left.execution_device, DeviceType::Cpu);
  EXPECT_EQ(right.execution_device, DeviceType::Cuda);
  EXPECT_EQ(right.execution_device_id, kDevice);
  EXPECT_EQ(left.protocol, right.protocol);
  EXPECT_EQ(left.image_width, right.image_width);
  EXPECT_EQ(left.image_height, right.image_height);
  EXPECT_EQ(left.num_classes, right.num_classes);
  EXPECT_EQ(left.candidates_after_filter, right.candidates_after_filter);
  EXPECT_EQ(left.candidates_before_nms, right.candidates_before_nms);
  ASSERT_EQ(left.objects.size(), right.objects.size()) << "candidates " << left.candidates_after_filter << "/"
                                                       << right.candidates_after_filter;
  for (std::size_t index = 0; index < left.objects.size(); ++index) {
    const PoseObject& want = left.objects[index];
    const PoseObject& got = right.objects[index];
    EXPECT_EQ(want.class_id, got.class_id) << index;
    EXPECT_EQ(want.anchor_index, got.anchor_index) << index;
    EXPECT_FLOAT_EQ(want.score, got.score) << index;
    EXPECT_FLOAT_EQ(want.cx, got.cx) << index;
    EXPECT_FLOAT_EQ(want.cy, got.cy) << index;
    EXPECT_FLOAT_EQ(want.width, got.width) << index;
    EXPECT_FLOAT_EQ(want.height, got.height) << index;
    ASSERT_EQ(want.keypoints.size(), got.keypoints.size()) << index;
    for (std::size_t point = 0; point < want.keypoints.size(); ++point) {
      EXPECT_FLOAT_EQ(want.keypoints[point].x, got.keypoints[point].x) << index << " " << point;
      EXPECT_FLOAT_EQ(want.keypoints[point].y, got.keypoints[point].y) << index << " " << point;
      EXPECT_FLOAT_EQ(want.keypoints[point].confidence, got.keypoints[point].confidence) << index << " " << point;
      EXPECT_EQ(want.keypoints[point].inside_image, got.keypoints[point].inside_image) << index << " " << point;
    }
  }
  if (right.objects.empty()) {
    EXPECT_EQ(right.d2h_bytes, 0u);
  } else {
    EXPECT_EQ(right.d2h_bytes,
              right.objects.size() * kObjectBytes + right.objects.size() * right.objects[0].keypoints.size() *
                                                       kKeypointBytes);
  }
  EXPECT_GT(right.output_bytes, 0u);
}

struct CaughtError {
  bool thrown = false;
  ErrorCode code = ErrorCode::Internal;
  std::string message;
};

template <typename Callable>
CaughtError capture(Callable&& callable) {
  try {
    callable();
  } catch (const vie::VieError& error) {
    return CaughtError{true, error.code(), error.what()};
  }
  return CaughtError{};
}

template <typename CpuCallable, typename CudaCallable>
void expect_same_failure(CpuCallable&& cpu_callable, CudaCallable&& cuda_callable, bool compare_message = true) {
  const CaughtError cpu = capture(std::forward<CpuCallable>(cpu_callable));
  const CaughtError cuda = capture(std::forward<CudaCallable>(cuda_callable));
  EXPECT_TRUE(cpu.thrown) << "CPU path accepted the input";
  EXPECT_TRUE(cuda.thrown) << "CUDA path accepted the input";
  EXPECT_EQ(cpu.code, cuda.code) << "cpu: " << cpu.message << "\ncuda: " << cuda.message;
  if (compare_message) {
    EXPECT_EQ(cpu.message, cuda.message) << "cpu: " << cpu.message << "\ncuda: " << cuda.message;
  }
}

}  // namespace

TEST(PoseCudaDecoder, MatchesCpuOnFilteringNmsAndKeypoints) {
  constexpr int kChannels = 12;
  constexpr int kPredictions = 5;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  auto probability = [&](int prediction, int channel, float value) { put(values, kPredictions, channel, prediction, value); };
  auto box = [&](int prediction, float cx, float cy, float width, float height) {
    put(values, kPredictions, 0, prediction, cx);
    put(values, kPredictions, 1, prediction, cy);
    put(values, kPredictions, 2, prediction, width);
    put(values, kPredictions, 3, prediction, height);
  };
  auto points = [&](int prediction, const std::vector<Keypoint>& keypoints) {
    const int base = 6;
    for (std::size_t index = 0; index < keypoints.size(); ++index) {
      put(values, kPredictions, base + static_cast<int>(index) * 3, prediction, keypoints[index].x);
      put(values, kPredictions, base + static_cast<int>(index) * 3 + 1, prediction, keypoints[index].y);
      put(values, kPredictions, base + static_cast<int>(index) * 3 + 2, prediction, keypoints[index].confidence);
    }
  };
  box(0, 10.f, 10.f, 10.f, 10.f);
  probability(0, 4, 0.95f);
  probability(0, 5, 0.10f);
  points(0, {{5.f, 6.f, 0.4f}, {-5.f, 8.f, 0.01f}});
  box(1, 12.f, 10.f, 10.f, 10.f);
  probability(1, 4, 0.90f);
  probability(1, 5, 0.10f);
  points(1, {{1.f, 1.f, 0.2f}, {2.f, 2.f, 0.2f}});
  box(2, 12.f, 10.f, 10.f, 10.f);
  probability(2, 4, 0.10f);
  probability(2, 5, 0.85f);
  points(2, {{3.f, 4.f, 0.3f}, {4.f, 5.f, 0.3f}});
  box(3, 50.f, 50.f, 6.f, 6.f);
  probability(3, 4, 0.70f);
  probability(3, 5, 0.70f);
  points(3, {{48.f, 49.f, 0.6f}, {51.f, 52.f, 0.7f}});
  box(4, 20.f, 20.f, 4.f, 4.f);
  probability(4, 4, 0.2f);
  probability(4, 5, 0.2f);
  points(4, {{std::nanf(""), 1.f, 0.5f}, {1.f, 1.f, 0.5f}});

  const GeometryTransform transform = geometry(64, 64);
  const PoseDecodeConfig config = keep_config();
  const PoseBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
  const PoseBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
  expect_parity(cpu, cuda);
  ASSERT_EQ(cpu.results[0].objects.size(), 3u);
  EXPECT_EQ(cpu.results[0].objects[0].anchor_index, 0);
  EXPECT_EQ(cpu.results[0].objects[1].anchor_index, 2);
  EXPECT_EQ(cpu.results[0].objects[2].anchor_index, 3);
  EXPECT_EQ(cuda.results[0].candidates_after_filter, 4);
  EXPECT_EQ(cuda.results[0].candidates_before_nms, 4);

  PoseDecodeConfig top_one = config;
  top_one.pre_nms_top_k = 1;
  expect_parity(run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), top_one, transform),
                run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), top_one, transform));
}

TEST(PoseCudaDecoder, MatchesCpuOnAxis2AndSeventeenKeypoints) {
  constexpr int kChannels = 4 + 1 + 17 * 3;
  constexpr int kPredictions = 3;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  const GeometryTransform transform = geometry(100, 80, 2.5, 1.5, 10, 4);
  auto set = [&](int prediction, int channel, float value) { put(values, kPredictions, channel, prediction, value); };
  auto person = [&](int prediction, float score, float cx, float cy, float width, float height) {
    set(prediction, 0, cx);
    set(prediction, 1, cy);
    set(prediction, 2, width);
    set(prediction, 3, height);
    set(prediction, 4, score);
    for (int point = 0; point < 17; ++point) {
      const int base = 5 + point * 3;
      set(prediction, base, static_cast<float>(10.0 + 2.5 * (point * 6.0 - 10.0)));
      set(prediction, base + 1, static_cast<float>(4.0 + 1.5 * (point * 4.0 - 5.0)));
      set(prediction, base + 2, 0.05f * static_cast<float>(point));
    }
  };
  person(0, 0.9f, 50.f, 50.f, 20.f, 20.f);
  person(1, 0.2f, 30.f, 30.f, 10.f, 10.f);
  person(2, 0.8f, 20.f, 20.f, 10.f, 10.f);

  PoseDecodeConfig config = keep_config();
  config.num_classes = 1;
  config.keypoint_count = 17;
  config.channel_axis = 2;
  config.max_detections = 10;
  config.pre_nms_top_k = 10;
  const PoseBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 2, DeviceType::Cpu), config, transform);
  const PoseBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 2, DeviceType::Cuda), config, transform);
  expect_parity(cpu, cuda);
  ASSERT_EQ(cpu.results[0].objects.size(), 2u);
  EXPECT_EQ(cpu.results[0].objects[0].keypoints.size(), 17u);
  EXPECT_FALSE(cuda.results[0].objects[0].keypoints[0].inside_image);
  EXPECT_TRUE(cuda.results[0].objects[0].keypoints[10].inside_image);
  EXPECT_EQ(cuda.results[0].d2h_bytes, 2u * kObjectBytes + 2u * 17u * kKeypointBytes);
}

TEST(PoseCudaDecoder, MatchesCpuOnClassSelectionAndNmsPolicies) {
  constexpr int kChannels = 12;
  constexpr int kPredictions = 2;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  auto set = [&](int prediction, int channel, float value) { put(values, kPredictions, channel, prediction, value); };
  const std::vector<std::vector<float>> rows{
      {20.f, 20.f, 10.f, 10.f, 0.95f, 0.10f, 18.f, 22.f, 0.5f, 21.f, 19.f, 0.6f},
      {20.f, 20.f, 10.f, 10.f, 0.10f, 0.90f, 18.f, 22.f, 0.5f, 21.f, 19.f, 0.6f}};
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    for (int channel = 0; channel < kChannels; ++channel) {
      set(prediction, channel, rows[static_cast<std::size_t>(prediction)][static_cast<std::size_t>(channel)]);
    }
  }
  const GeometryTransform transform = geometry(64, 64);
  auto run_pair = [&](const PoseDecodeConfig& config) {
    const PoseBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
    const PoseBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
    expect_parity(cpu, cuda);
    return cpu.results[0].objects.size();
  };

  PoseDecodeConfig class_aware = keep_config();
  class_aware.class_aware = true;
  EXPECT_EQ(run_pair(class_aware), 2u);
  PoseDecodeConfig class_agnostic = keep_config();
  class_agnostic.class_aware = false;
  EXPECT_EQ(run_pair(class_agnostic), 1u);
  PoseDecodeConfig without_nms = keep_config();
  without_nms.nms_enabled = false;
  without_nms.class_aware.reset();
  without_nms.iou_threshold.reset();
  EXPECT_EQ(run_pair(without_nms), 2u);
  PoseDecodeConfig one = keep_config();
  one.max_detections = 1;
  EXPECT_EQ(run_pair(one), 1u);
  PoseDecodeConfig top_one = keep_config();
  top_one.pre_nms_top_k = 1;
  EXPECT_EQ(run_pair(top_one), 1u);
  PoseDecodeConfig multi = keep_config();
  multi.class_selection = ClassSelection::AllAboveThreshold;
  EXPECT_EQ(run_pair(multi), 2u);
  PoseDecodeConfig multi_agnostic = multi;
  multi_agnostic.class_aware = false;
  EXPECT_EQ(run_pair(multi_agnostic), 1u);
}

TEST(PoseCudaDecoder, MatchesCpuOnTiesAndIouBoundary) {
  constexpr int kChannels = 12;
  std::vector<float> values(static_cast<std::size_t>(kChannels * 2), 0.f);
  auto set = [&](int prediction, int channel, float value) { put(values, 2, channel, prediction, value); };
  // Two equal scores far apart: anchor order breaks the tie.
  for (int prediction = 0; prediction < 2; ++prediction) {
    set(prediction, 0, prediction == 0 ? 10.f : 40.f);
    set(prediction, 1, 10.f);
    set(prediction, 2, 4.f);
    set(prediction, 3, 4.f);
    set(prediction, 4, 0.6f);
    set(prediction, 6, 10.f);
    set(prediction, 7, 10.f);
    set(prediction, 8, 0.5f);
  }
  GeometryTransform transform = geometry(64, 64);
  PoseDecodeConfig config = keep_config();
  PoseBatch cpu = run_cpu(make_tensor(values, kChannels, 2, 1, DeviceType::Cpu), config, transform);
  PoseBatch cuda = run_cuda(make_tensor(values, kChannels, 2, 1, DeviceType::Cuda), config, transform);
  expect_parity(cpu, cuda);
  ASSERT_EQ(cpu.results[0].objects.size(), 2u);
  EXPECT_EQ(cpu.results[0].objects[0].anchor_index, 0);
  EXPECT_EQ(cpu.results[0].objects[1].anchor_index, 1);

  // Anchor 0 is [0,0,10,10]; anchor 1 is [0,0,10,5], so IoU is exactly 0.5 and
  // the ">=" rule suppresses anchor 1 at iou_threshold 0.5.
  std::vector<float> boundary(static_cast<std::size_t>(kChannels * 2), 0.f);
  auto set_boundary = [&](int prediction, int channel, float value) { put(boundary, 2, channel, prediction, value); };
  set_boundary(0, 0, 5.f);
  set_boundary(0, 1, 5.f);
  set_boundary(0, 2, 10.f);
  set_boundary(0, 3, 10.f);
  set_boundary(0, 4, 0.5f);
  set_boundary(1, 0, 5.f);
  set_boundary(1, 1, 2.5f);
  set_boundary(1, 2, 10.f);
  set_boundary(1, 3, 5.f);
  set_boundary(1, 4, 0.8f);
  cpu = run_cpu(make_tensor(boundary, kChannels, 2, 1, DeviceType::Cpu), config, transform);
  cuda = run_cuda(make_tensor(boundary, kChannels, 2, 1, DeviceType::Cuda), config, transform);
  expect_parity(cpu, cuda);
  ASSERT_EQ(cpu.results[0].objects.size(), 1u);
  EXPECT_EQ(cpu.results[0].objects[0].anchor_index, 1);
}

TEST(PoseCudaDecoder, MatchesCpuOnEmptyResultsAndDroppedAnchors) {
  constexpr int kChannels = 12;
  constexpr int kPredictions = 2;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  auto set = [&](int prediction, int channel, float value) { put(values, kPredictions, channel, prediction, value); };
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    set(prediction, 0, 20.f);
    set(prediction, 1, 20.f);
    set(prediction, 2, 8.f);
    set(prediction, 3, 8.f);
    set(prediction, 4, 0.9f);
    set(prediction, 6, 20.f);
    set(prediction, 7, 20.f);
    set(prediction, 8, 0.5f);
  }
  const GeometryTransform transform = geometry(64, 64);
  PoseDecodeConfig nothing = keep_config();
  nothing.confidence_threshold = 0.99f;
  PoseBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), nothing, transform);
  PoseBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), nothing, transform);
  expect_parity(cpu, cuda);
  EXPECT_TRUE(cpu.results[0].objects.empty());
  EXPECT_EQ(cuda.results[0].candidates_after_filter, 0);

  // A NaN box or keypoint in an anchor the score already drops is never read.
  std::vector<float> dropped = values;
  put(dropped, kPredictions, 0, 1, std::nanf(""));
  put(dropped, kPredictions, 6, 1, std::nanf(""));
  put(dropped, kPredictions, 4, 1, 0.2f);
  cpu = run_cpu(make_tensor(dropped, kChannels, kPredictions, 1, DeviceType::Cpu), keep_config(), transform);
  cuda = run_cuda(make_tensor(dropped, kChannels, kPredictions, 1, DeviceType::Cuda), keep_config(), transform);
  expect_parity(cpu, cuda);
  ASSERT_EQ(cpu.results[0].objects.size(), 1u);
}

TEST(PoseCudaDecoder, ReportsTheSameFaultAsCpu) {
  constexpr int kChannels = 12;
  constexpr int kPredictions = 1;
  std::vector<float> base(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  put(base, kPredictions, 0, 0, 10.f);
  put(base, kPredictions, 1, 0, 10.f);
  put(base, kPredictions, 2, 0, 8.f);
  put(base, kPredictions, 3, 0, 8.f);
  put(base, kPredictions, 4, 0, 0.9f);
  put(base, kPredictions, 6, 0, 5.f);
  put(base, kPredictions, 7, 0, 5.f);
  put(base, kPredictions, 8, 0, 0.5f);
  const GeometryTransform transform = geometry(64, 64);
  const PoseDecodeConfig config = keep_config();
  auto both = [&](const std::vector<float>& values) {
    expect_same_failure(
        [&] { run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform); },
        [&] { run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform); });
  };

  std::vector<float> bad = base;
  put(bad, kPredictions, 4, 0, std::nanf(""));
  both(bad);
  bad = base;
  put(bad, kPredictions, 4, 0, std::numeric_limits<float>::infinity());
  both(bad);
  bad = base;
  put(bad, kPredictions, 4, 0, 1.5f);
  both(bad);
  bad = base;
  put(bad, kPredictions, 4, 0, -0.01f);
  both(bad);
  bad = base;
  put(bad, kPredictions, 0, 0, std::nanf(""));
  both(bad);
  bad = base;
  put(bad, kPredictions, 2, 0, 0.f);
  both(bad);
  bad = base;
  put(bad, kPredictions, 6, 0, std::nanf(""));
  both(bad);
  bad = base;
  put(bad, kPredictions, 8, 0, 1.2f);
  both(bad);

  // The messages are the CPU decoder's wording, not a CUDA-specific rewrite.
  const CaughtError cpu = capture([&] {
    run_cpu(make_tensor(bad, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
  });
  EXPECT_EQ(cpu.message, "yolov8_pose_raw probability at prediction 0 channel 8 is outside [0, 1]");
}

TEST(PoseCudaDecoder, RejectsTensorAndRequestContracts) {
  constexpr int kChannels = 12;
  constexpr int kPredictions = 2;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    put(values, kPredictions, 0, prediction, 20.f);
    put(values, kPredictions, 1, prediction, 20.f);
    put(values, kPredictions, 2, prediction, 8.f);
    put(values, kPredictions, 3, prediction, 8.f);
    put(values, kPredictions, 4, prediction, 0.9f);
  }
  const GeometryTransform transform = geometry(64, 64);
  const PoseDecodeConfig config = keep_config();
  const Tensor cpu_tensor = make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu);
  const Tensor cuda_tensor = make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda);

  // Device mismatch: the host path accepts a host tensor; the CUDA path must
  // reject it instead of copying it implicitly.
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, run_cuda(cpu_tensor, config, transform));

  expect_same_failure(
      [&] {
        const Tensor wrong_rank = Tensor::from_host(DeviceContext{}, DataType::Fp32, {12, 2}, values.data(),
                                                    values.size() * sizeof(float));
        run_cpu(wrong_rank, config, transform);
      },
      [&] {
        const Tensor wrong_rank = Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32, {12, 2},
                                                    values.data(), values.size() * sizeof(float));
        run_cuda(wrong_rank, config, transform);
      });

  expect_same_failure(
      [&] {
        const std::vector<float> batched_values(2 * values.size(), 0.f);
        const Tensor batched = Tensor::from_host(DeviceContext{}, DataType::Fp32, {2, 12, 2},
                                                 batched_values.data(), batched_values.size() * sizeof(float));
        run_cpu(batched, config, transform);
      },
      [&] {
        const std::vector<float> batched_values(2 * values.size(), 0.f);
        const Tensor batched = Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32, {2, 12, 2},
                                                 batched_values.data(), batched_values.size() * sizeof(float));
        run_cuda(batched, config, transform);
      });

  expect_same_failure(
      [&] {
        const Tensor wrong_channels = Tensor::from_host(DeviceContext{}, DataType::Fp32, {1, 11, 2}, values.data(),
                                                        values.size() * sizeof(float));
        run_cpu(wrong_channels, config, transform);
      },
      [&] {
        const Tensor wrong_channels = Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32,
                                                        {1, 11, 2}, values.data(), values.size() * sizeof(float));
        run_cuda(wrong_channels, config, transform);
      });

  expect_same_failure(
      [&] {
        const std::vector<std::int32_t> ints(static_cast<std::size_t>(kChannels * kPredictions), 0);
        const Tensor wrong_dtype = Tensor::from_host(DeviceContext{}, DataType::Int32, {1, 12, 2}, ints.data(),
                                                     ints.size() * sizeof(std::int32_t));
        run_cpu(wrong_dtype, config, transform);
      },
      [&] {
        const std::vector<std::int32_t> ints(static_cast<std::size_t>(kChannels * kPredictions), 0);
        const Tensor wrong_dtype = Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Int32,
                                                     {1, 12, 2}, ints.data(), ints.size() * sizeof(std::int32_t));
        run_cuda(wrong_dtype, config, transform);
      },
      false);

  // Non-contiguous view: base [1, 12, 4] read with a stride of 2 columns.
  expect_same_failure(
      [&] {
        std::vector<float> wide(static_cast<std::size_t>(kChannels * kPredictions * 2), 0.f);
        for (int channel = 0; channel < kChannels; ++channel) {
          for (int prediction = 0; prediction < kPredictions; ++prediction) {
            wide[static_cast<std::size_t>(channel) * 4 + static_cast<std::size_t>(prediction * 2)] =
                values[static_cast<std::size_t>(channel) * kPredictions + static_cast<std::size_t>(prediction)];
          }
        }
        const Tensor base_tensor = Tensor::from_host(DeviceContext{}, DataType::Fp32, {1, 12, 4}, wide.data(),
                                                     wide.size() * sizeof(float));
        const Tensor strided = base_tensor.view({1, 12, 2}, {48, 4, 2}, 0);
        run_cpu(strided, config, transform);
      },
      [&] {
        std::vector<float> wide(static_cast<std::size_t>(kChannels * kPredictions * 2), 0.f);
        for (int channel = 0; channel < kChannels; ++channel) {
          for (int prediction = 0; prediction < kPredictions; ++prediction) {
            wide[static_cast<std::size_t>(channel) * 4 + static_cast<std::size_t>(prediction * 2)] =
                values[static_cast<std::size_t>(channel) * kPredictions + static_cast<std::size_t>(prediction)];
          }
        }
        const Tensor base_tensor = Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32,
                                                     {1, 12, 4}, wide.data(), wide.size() * sizeof(float));
        const Tensor strided = base_tensor.view({1, 12, 2}, {48, 4, 2}, 0);
        run_cuda(strided, config, transform);
      },
      false);

  vie::DecoderInput missing = decode_input(cuda_tensor, transform, config.output_name);
  missing.outputs.clear();
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument,
                   vie::PoseCudaDecoder{}.decode_pose(missing, config, DeviceContext(DeviceType::Cuda, kDevice)));
  vie::DecoderInput no_geometry = decode_input(cuda_tensor, transform, config.output_name);
  no_geometry.geometry.reset();
  VIE_EXPECT_ERROR(ErrorCode::UnresolvedField,
                   vie::PoseCudaDecoder{}.decode_pose(no_geometry, config, DeviceContext(DeviceType::Cuda, kDevice)));
  PoseDecodeConfig bad_threshold = config;
  bad_threshold.confidence_threshold = 1.5f;
  VIE_EXPECT_ERROR(ErrorCode::UnresolvedField,
                   vie::PoseCudaDecoder{}.decode_pose(decode_input(cuda_tensor, transform, config.output_name),
                                                      bad_threshold, DeviceContext(DeviceType::Cuda, kDevice)));
  VIE_EXPECT_ERROR(ErrorCode::Unsupported, vie::PoseCudaDecoder{}.decode({}, {}));
  VIE_EXPECT_ERROR(ErrorCode::Unsupported,
                   vie::PoseCudaDecoder{}.decode_pose(decode_input(cuda_tensor, transform, config.output_name),
                                                      config, DeviceContext{}));
}

TEST(PoseCudaDecoder, RunsTheResolvedPhase12Policy) {
  vie::ModelManifest manifest;
  manifest.model_name = "pose_policy";
  manifest.task = vie::TaskType::Pose;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 2;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 32, 32};
  manifest.input.dtype = DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  vie::OutputDesc output;
  output.name = "output0";
  output.shape = {1, 12, 2};
  output.dtype = DataType::Fp32;
  output.semantic = "poses";
  manifest.outputs.push_back(output);
  manifest.decoder.protocol = "yolov8_pose_raw";
  manifest.decoder.box_format = "cxcywh";
  manifest.decoder.box_decode = "decoded_cxcywh";
  manifest.decoder.box_channels = 4;
  manifest.decoder.channel_axis = 1;
  manifest.decoder.scores_are_probabilities = true;
  manifest.pose.output_name = "output0";
  manifest.pose.channel_axis = 1;
  manifest.pose.keypoint_count = 2;
  manifest.pose.keypoint_dimensions = 3;
  manifest.pose.keypoint_layout = "xy_confidence";
  manifest.pose.score_type = "probabilities";
  manifest.ensure_provenance();

  vie::InferenceOptions request;
  request.pose.confidence_threshold = 0.62f;
  request.pose.iou_threshold = 0.41f;
  request.pose.max_detections = 7;
  request.pose.pre_nms_top_k = 5;
  request.pose.class_agnostic_nms = true;
  const vie::ResolvedInferenceConfig resolved =
      vie::resolve_inference_config(manifest.task, manifest, vie::InferenceOptions{}, request);
  ASSERT_TRUE(resolved.pose.has_value());
  EXPECT_EQ(resolved.pose->confidence_threshold.value, 0.62f);
  EXPECT_EQ(resolved.pose->confidence_threshold.source, vie::ParameterSource::RequestOverride);
  EXPECT_EQ(resolved.pose->iou_threshold.value, 0.41f);
  EXPECT_EQ(resolved.pose->max_detections.value, 7);
  EXPECT_EQ(resolved.pose->pre_nms_top_k.value, 5);
  EXPECT_TRUE(resolved.pose->class_agnostic_nms.value);

  const PoseDecodeConfig config = vie::pose_decode_config_from_resolved(manifest, resolved);
  EXPECT_EQ(config.confidence_threshold, 0.62f);
  EXPECT_EQ(config.iou_threshold, 0.41f);
  EXPECT_EQ(config.max_detections, 7);
  EXPECT_EQ(config.pre_nms_top_k, 5);
  EXPECT_FALSE(*config.class_aware);

  constexpr int kChannels = 12;
  constexpr int kPredictions = 2;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    put(values, kPredictions, 0, prediction, 20.f);
    put(values, kPredictions, 1, prediction, 20.f);
    put(values, kPredictions, 2, prediction, 8.f);
    put(values, kPredictions, 3, prediction, 8.f);
    put(values, kPredictions, 4, prediction, 0.8f);
    put(values, kPredictions, 6, prediction, 20.f);
    put(values, kPredictions, 7, prediction, 20.f);
    put(values, kPredictions, 8, prediction, 0.5f);
  }
  const GeometryTransform transform = geometry(64, 64);
  expect_parity(run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform),
                run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform));
}

TEST(PoseCudaDecoder, RecordsDecoderTimings) {
  constexpr int kChannels = 4 + 1 + 17 * 3;
  constexpr int kPredictions = 8400;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    put(values, kPredictions, 0, prediction, static_cast<float>(10 + (prediction % 100) * 6));
    put(values, kPredictions, 1, prediction, static_cast<float>(10 + (prediction / 100) * 6));
    put(values, kPredictions, 2, prediction, 5.f);
    put(values, kPredictions, 3, prediction, 5.f);
    for (int point = 0; point < 17; ++point) {
      put(values, kPredictions, 5 + point * 3, prediction, 20.f);
      put(values, kPredictions, 5 + point * 3 + 1, prediction, 20.f);
      put(values, kPredictions, 5 + point * 3 + 2, prediction, 0.5f);
    }
  }
  PoseDecodeConfig config = keep_config();
  config.num_classes = 1;
  config.keypoint_count = 17;
  config.confidence_threshold = 0.25f;
  config.iou_threshold = 0.45f;
  config.max_detections = 1000;
  config.pre_nms_top_k = 4096;
  const GeometryTransform transform = geometry(640, 480);

  constexpr int kSamples = 5;
  std::ofstream record("/tmp/vie_phase14_pose.txt", std::ios::app);
  // Light, medium, and heavy candidate counts at the same 8400-anchor output.
  for (const int kept_target : {8, 120, 600}) {
    for (int prediction = 0; prediction < kPredictions; ++prediction) {
      put(values, kPredictions, 4, prediction, prediction < kept_target ? 0.9f : 0.1f);
    }
    const Tensor cpu_tensor = make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu);
    const Tensor cuda_tensor = make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda);

    const auto time_cpu = [&] {
      const auto start = std::chrono::steady_clock::now();
      const PoseBatch batch = run_cpu(cpu_tensor, config, transform);
      const double wall =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      return std::pair<PoseBatch, double>(batch, wall);
    };
    const auto time_cuda = [&] {
      const auto start = std::chrono::steady_clock::now();
      const PoseBatch batch = run_cuda(cuda_tensor, config, transform);
      const double wall =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      return std::pair<PoseBatch, double>(batch, wall);
    };
    for (int index = 0; index < 2; ++index) {
      time_cpu();
      time_cuda();
    }
    double cpu_ms = 0;
    double cuda_ms = 0;
    double validate_ms = 0;
    double pose_decode_ms = 0;
    double nms_ms = 0;
    double d2h_ms = 0;
    PoseBatch cpu_batch;
    PoseBatch cuda_batch;
    for (int index = 0; index < kSamples; ++index) {
      const auto cpu_sample = time_cpu();
      const auto cuda_sample = time_cuda();
      cpu_batch = cpu_sample.first;
      cuda_batch = cuda_sample.first;
      cpu_ms += cpu_sample.second;
      cuda_ms += cuda_sample.second;
      validate_ms += cuda_sample.first.results[0].validate_ms;
      pose_decode_ms += cuda_sample.first.results[0].pose_decode_ms;
      nms_ms += cuda_sample.first.results[0].nms_ms;
      d2h_ms += cuda_sample.first.results[0].d2h_ms;
    }
    const std::size_t kept = static_cast<std::size_t>(kept_target);
    EXPECT_EQ(cpu_batch.results[0].objects.size(), kept);
    EXPECT_EQ(cuda_batch.results[0].objects.size(), kept);
    EXPECT_EQ(cuda_batch.results[0].candidates_after_filter, kept_target);
    EXPECT_EQ(cuda_batch.results[0].candidates_before_nms, kept_target);
    EXPECT_EQ(cuda_batch.results[0].d2h_bytes, kept * kObjectBytes + kept * 17u * kKeypointBytes);
    EXPECT_LT(cuda_batch.results[0].output_bytes, 8400u * 56u * 4u + 1u);
    EXPECT_GT(validate_ms, 0.0);
    EXPECT_GT(d2h_ms, 0.0);
    EXPECT_GT(cpu_ms, 0.0);
    EXPECT_GT(cuda_ms, 0.0);

    record << "candidates " << kept_target << " predictions " << kPredictions << " keypoints 17"
           << " samples " << kSamples << " output_bytes " << cuda_batch.results[0].output_bytes
           << " d2h_bytes " << cuda_batch.results[0].d2h_bytes << "\n";
    record << "cpu_decoder_mean_ms " << cpu_ms / kSamples << " cuda_decoder_mean_ms " << cuda_ms / kSamples
           << " validate_mean_ms " << validate_ms / kSamples << " pose_decode_mean_ms " << pose_decode_ms / kSamples
           << " nms_mean_ms " << nms_ms / kSamples << " d2h_mean_ms " << d2h_ms / kSamples << "\n";
  }
}