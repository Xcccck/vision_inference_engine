// Phase 15: CPU/CUDA parity for the yolov8_obb_raw decoder.
//
// Every decode case feeds the same raw tensor to YoloV8ObbDecoder on the host
// and ObbCudaDecoder on the device. Discrete fields (class id, anchor, counts,
// kept order) must match exactly; float fields use EXPECT_FLOAT_EQ because both
// sides run the same expression tree (the .cu is compiled with --fmad=false).
// Rotated Polygon IoU is compared on its own through launch_obb_iou_cuda so a
// clipping difference cannot hide behind a matching NMS survivor set.
//
// Only compiled when VIE_ENABLE_CUDA is on.

#include "test_util.hpp"

#include "vie/config/inference_config.hpp"
#include "vie/decoder/obb_cuda_decoder.hpp"
#include "vie/decoder/yolov8_obb_decoder.hpp"
#include "vie/preprocess/preprocess.hpp"

#include "../src/decoder/obb_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
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
using vie::ObbBatch;
using vie::ObbDecodeConfig;
using vie::ObbObject;
using vie::OrientedBox;
using vie::Tensor;

constexpr int kDevice = 0;
constexpr std::size_t kObjectBytes = 32;
constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr float kAngleMin = -0.7853981633974483f;
constexpr float kAngleMax = 2.356194490192345f;

ObbDecodeConfig keep_config() {
  ObbDecodeConfig config;
  config.output_name = "output0";
  config.channel_axis = 1;
  config.num_classes = 2;
  config.class_selection = ClassSelection::BestClass;
  config.confidence_threshold = 0.5f;
  config.nms_enabled = true;
  config.class_aware = true;
  config.iou_threshold = 0.5f;
  config.max_detections = 10;
  config.pre_nms_top_k = 10;
  return config;
}

vie::ResizeMode resize_mode(double scale_x, double scale_y, int pad_left, int pad_top, int crop_x, int crop_y) {
  if (scale_x == scale_y && pad_left == 0 && pad_top == 0 && crop_x == 0 && crop_y == 0) {
    return vie::ResizeMode::Stretch;
  }
  if (crop_x != 0 || crop_y != 0) return vie::ResizeMode::ShortestSide;
  return vie::ResizeMode::Letterbox;
}

GeometryTransform geometry(int width, int height, double scale_x = 1.0, double scale_y = 1.0, int pad_left = 0,
                           int pad_top = 0, int crop_x = 0, int crop_y = 0) {
  GeometryTransform transform;
  transform.original_width = width;
  transform.original_height = height;
  transform.scale_x = scale_x;
  transform.scale_y = scale_y;
  transform.uniform_scale = scale_x == scale_y ? scale_x : 0.0;
  transform.pad_left = pad_left;
  transform.pad_top = pad_top;
  transform.crop_x = crop_x;
  transform.crop_y = crop_y;
  transform.resized_width = static_cast<int>(static_cast<double>(width) * scale_x);
  transform.resized_height = static_cast<int>(static_cast<double>(height) * scale_y);
  transform.output_width = transform.resized_width + pad_left * 2;
  transform.output_height = transform.resized_height + pad_top * 2;
  transform.resize_mode = resize_mode(scale_x, scale_y, pad_left, pad_top, crop_x, crop_y);
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

ObbBatch run_cpu(const Tensor& tensor, const ObbDecodeConfig& config, const GeometryTransform& transform) {
  return vie::YoloV8ObbDecoder{}.decode_obb(decode_input(tensor, transform, config.output_name), config,
                                            DeviceContext{});
}

ObbBatch run_cuda(const Tensor& tensor, const ObbDecodeConfig& config, const GeometryTransform& transform) {
  return vie::ObbCudaDecoder{}.decode_obb(decode_input(tensor, transform, config.output_name), config,
                                          DeviceContext(DeviceType::Cuda, kDevice));
}

void expect_parity(const ObbBatch& cpu, const ObbBatch& cuda) {
  ASSERT_EQ(cpu.results.size(), 1u);
  ASSERT_EQ(cuda.results.size(), 1u);
  const vie::ObbResult& left = cpu.results[0];
  const vie::ObbResult& right = cuda.results[0];
  EXPECT_EQ(left.image_width, right.image_width);
  EXPECT_EQ(left.image_height, right.image_height);
  EXPECT_EQ(left.num_classes, right.num_classes);
  EXPECT_EQ(left.protocol, right.protocol);
  EXPECT_EQ(right.execution_device, DeviceType::Cuda);
  EXPECT_EQ(right.execution_device_id, kDevice);
  EXPECT_EQ(left.candidates_after_filter, right.candidates_after_filter);
  EXPECT_EQ(left.candidates_before_nms, right.candidates_before_nms);
  ASSERT_EQ(left.objects.size(), right.objects.size());
  EXPECT_EQ(right.d2h_bytes, right.objects.size() * kObjectBytes);
  for (std::size_t index = 0; index < left.objects.size(); ++index) {
    const ObbObject& want = left.objects[index];
    const ObbObject& got = right.objects[index];
    EXPECT_EQ(got.class_id, want.class_id) << index;
    EXPECT_EQ(got.anchor_index, want.anchor_index) << index;
    EXPECT_FLOAT_EQ(got.score, want.score) << index;
    EXPECT_FLOAT_EQ(got.box.cx, want.box.cx) << index;
    EXPECT_FLOAT_EQ(got.box.cy, want.box.cy) << index;
    EXPECT_FLOAT_EQ(got.box.width, want.box.width) << index;
    EXPECT_FLOAT_EQ(got.box.height, want.box.height) << index;
    EXPECT_FLOAT_EQ(got.box.angle, want.box.angle) << index;
  }
}

struct CaughtError {
  bool threw = false;
  ErrorCode code = ErrorCode::Internal;
  std::string message;
};

CaughtError capture(const std::function<void()>& body) {
  CaughtError caught;
  try {
    body();
  } catch (const vie::VieError& error) {
    caught.threw = true;
    caught.code = error.code();
    caught.message = error.what();
  }
  return caught;
}

void expect_same_failure(const std::function<void()>& cpu, const std::function<void()>& cuda,
                         bool exact_message = true) {
  const CaughtError cpu_error = capture(cpu);
  const CaughtError cuda_error = capture(cuda);
  ASSERT_TRUE(cpu_error.threw);
  ASSERT_TRUE(cuda_error.threw);
  EXPECT_EQ(cuda_error.code, cpu_error.code) << "cpu: " << cpu_error.message << "\ncuda: " << cuda_error.message;
  if (exact_message) {
    EXPECT_EQ(cuda_error.message, cpu_error.message);
  }
}

double milliseconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

TEST(ObbCudaDecoder, MatchesCpuOnBestClassSortAndNms) {
  constexpr int kChannels = 4 + 2 + 1;
  constexpr int kPredictions = 9;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  auto row = [&](int prediction, float cx, float cy, float width, float height, float score0, float score1,
                 float angle) {
    put(values, kPredictions, 0, prediction, cx);
    put(values, kPredictions, 1, prediction, cy);
    put(values, kPredictions, 2, prediction, width);
    put(values, kPredictions, 3, prediction, height);
    put(values, kPredictions, 4, prediction, score0);
    put(values, kPredictions, 5, prediction, score1);
    put(values, kPredictions, 6, prediction, angle);
  };
  // Distinct scores so the sort is observable.
  row(0, 30.f, 30.f, 20.f, 10.f, 0.95f, 0.05f, 0.f);
  row(1, 31.f, 30.f, 20.f, 10.f, 0.90f, 0.04f, 0.f);
  row(2, 90.f, 70.f, 12.f, 8.f, 0.85f, 0.06f, 0.4f);
  row(3, 120.f, 40.f, 14.f, 14.f, 0.80f, 0.07f, -0.5f);
  row(4, 150.f, 90.f, 10.f, 30.f, 0.75f, 0.08f, 1.2f);
  row(5, 40.f, 120.f, 26.f, 6.f, 0.70f, 0.09f, 2.2f);
  row(6, 70.f, 140.f, 8.f, 8.f, 0.65f, 0.10f, 0.75f);
  row(7, 170.f, 150.f, 18.f, 10.f, 0.60f, 0.11f, -0.75f);
  row(8, 100.f, 100.f, 4.f, 4.f, 0.40f, 0.12f, 0.1f);  // below threshold

  const GeometryTransform transform = geometry(200, 200);
  const ObbDecodeConfig config = keep_config();
  const ObbBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
  const ObbBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
  expect_parity(cpu, cuda);
  EXPECT_EQ(cpu.results[0].candidates_after_filter, 8);
  EXPECT_GT(cpu.results[0].objects.size(), 0u);
  EXPECT_LE(cpu.results[0].objects.size(), 8u);
  EXPECT_GT(cuda.results[0].d2h_bytes, 0u);
  EXPECT_GT(cuda.results[0].d2h_ms, 0.0);
}

TEST(ObbCudaDecoder, MatchesCpuOnAxis2AllAboveThresholdAndAngleRange) {
  constexpr int kChannels = 4 + 3 + 1;
  constexpr int kPredictions = 7;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  const float angles[kPredictions] = {kAngleMin, -0.4f, 0.f, 0.3f, 1.0f, 2.0f, kAngleMax};
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    put(values, kPredictions, 0, prediction, 20.f + 15.f * static_cast<float>(prediction));
    put(values, kPredictions, 1, prediction, 40.f + 9.f * static_cast<float>(prediction));
    put(values, kPredictions, 2, prediction, 10.f + static_cast<float>(prediction));
    put(values, kPredictions, 3, prediction, 18.f - static_cast<float>(prediction));
    for (int cls = 0; cls < 3; ++cls) {
      const float score = cls == prediction % 3 ? 0.55f + 0.04f * static_cast<float>(prediction) : 0.45f;
      put(values, kPredictions, 4 + cls, prediction, score);
    }
    put(values, kPredictions, 7, prediction, angles[prediction]);
  }
  ObbDecodeConfig config = keep_config();
  config.num_classes = 3;
  config.channel_axis = 2;
  config.class_selection = ClassSelection::AllAboveThreshold;
  config.pre_nms_top_k = 21;
  config.max_detections = 21;
  const GeometryTransform transform = geometry(180, 140);
  const ObbBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 2, DeviceType::Cpu), config, transform);
  const ObbBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 2, DeviceType::Cuda), config, transform);
  expect_parity(cpu, cuda);
  EXPECT_EQ(cuda.results[0].candidates_after_filter, 7);
}

TEST(ObbCudaDecoder, MatchesCpuOnGeometryModesAndDrops) {
  constexpr int kChannels = 4 + 2 + 1;
  constexpr int kPredictions = 5;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  auto row = [&](int prediction, float cx, float cy, float width, float height, float score, float angle) {
    put(values, kPredictions, 0, prediction, cx);
    put(values, kPredictions, 1, prediction, cy);
    put(values, kPredictions, 2, prediction, width);
    put(values, kPredictions, 3, prediction, height);
    put(values, kPredictions, 4, prediction, score);
    put(values, kPredictions, 5, prediction, 0.2f);
    put(values, kPredictions, 6, prediction, angle);
  };
  row(0, 40.f, 40.f, 20.f, 12.f, 0.9f, 0.2f);
  row(1, 70.f, 30.f, 16.f, 10.f, 0.85f, 0.6f);
  row(2, 20.f, 60.f, 12.f, 12.f, 0.8f, -0.4f);
  row(3, 100.f, 60.f, 14.f, 8.f, 0.75f, 1.1f);
  row(4, 195.f, 195.f, 8.f, 8.f, 0.7f, 0.f);  // maps outside and is dropped

  const ObbDecodeConfig config = keep_config();
  const std::vector<GeometryTransform> transforms{
      geometry(200, 200),
      geometry(100, 80, 2.5, 1.5, 10, 4),
      geometry(100, 80, 2.0, 2.0, 6, 3),
      geometry(100, 80, 1.7, 1.7, 0, 0, 5, 6),
  };
  // Anchor 4 sits at 195,195 with an 8x8 footprint: inside the 200x200 image,
  // outside every smaller original frame.
  const std::size_t expected[4] = {5u, 4u, 4u, 4u};
  for (std::size_t index = 0; index < transforms.size(); ++index) {
    const GeometryTransform& transform = transforms[index];
    const ObbBatch cpu =
        run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
    const ObbBatch cuda =
        run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
    expect_parity(cpu, cuda);
    EXPECT_EQ(cpu.results[0].objects.size(), expected[index]);
  }

  // A single box that leaves the image on every mode: no objects, no copy.
  std::vector<float> outside = values;
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    put(outside, kPredictions, 0, prediction, 400.f);
  }
  const ObbBatch cpu = run_cpu(make_tensor(outside, kChannels, kPredictions, 1, DeviceType::Cpu), config,
                               geometry(200, 200));
  const ObbBatch cuda = run_cuda(make_tensor(outside, kChannels, kPredictions, 1, DeviceType::Cuda), config,
                                 geometry(200, 200));
  expect_parity(cpu, cuda);
  EXPECT_EQ(cpu.results[0].objects.size(), 0u);
  EXPECT_EQ(cuda.results[0].d2h_bytes, 0u);
  EXPECT_EQ(cuda.results[0].candidates_before_nms, 0);
}

TEST(ObbCudaDecoder, MatchesCpuOnNmsPoliciesAndTies) {
  constexpr int kChannels = 4 + 2 + 1;
  constexpr int kPredictions = 4;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  auto row = [&](int prediction, float cx, float cy, float width, float height, float score0, float score1,
                 float angle) {
    put(values, kPredictions, 0, prediction, cx);
    put(values, kPredictions, 1, prediction, cy);
    put(values, kPredictions, 2, prediction, width);
    put(values, kPredictions, 3, prediction, height);
    put(values, kPredictions, 4, prediction, score0);
    put(values, kPredictions, 5, prediction, score1);
    put(values, kPredictions, 6, prediction, angle);
  };
  // Anchors 0 and 1 tie and overlap; anchors 2 and 3 tie too but sit apart.
  row(0, 50.f, 50.f, 20.f, 12.f, 0.9f, 0.1f, 0.f);
  row(1, 50.f, 50.f, 20.f, 12.f, 0.9f, 0.1f, 0.f);
  row(2, 120.f, 40.f, 10.f, 10.f, 0.8f, 0.2f, 0.5f);
  row(3, 160.f, 120.f, 10.f, 10.f, 0.8f, 0.2f, 0.5f);
  const GeometryTransform transform = geometry(200, 200);
  auto run_pair = [&](const ObbDecodeConfig& config) {
    const ObbBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
    const ObbBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
    expect_parity(cpu, cuda);
    return cpu.results[0].objects;
  };

  ObbDecodeConfig class_aware = keep_config();
  class_aware.max_detections = 10;
  class_aware.pre_nms_top_k = 10;
  const std::vector<ObbObject> aware = run_pair(class_aware);
  ASSERT_EQ(aware.size(), 3u);
  EXPECT_EQ(aware[0].anchor_index, 0);  // equal score: smaller anchor first
  EXPECT_EQ(aware[1].anchor_index, 2);
  EXPECT_EQ(aware[2].anchor_index, 3);

  ObbDecodeConfig top_two = class_aware;
  top_two.pre_nms_top_k = 2;
  const std::vector<ObbObject> truncated = run_pair(top_two);
  // The top two are the tied, fully overlapping anchors 0 and 1, so NMS keeps
  // only the smaller anchor.
  ASSERT_EQ(truncated.size(), 1u);
  EXPECT_EQ(truncated[0].anchor_index, 0);

  ObbDecodeConfig one = class_aware;
  one.max_detections = 1;
  EXPECT_EQ(run_pair(one).size(), 1u);

  ObbDecodeConfig without_nms = class_aware;
  without_nms.nms_enabled = false;
  without_nms.class_aware.reset();
  without_nms.iou_threshold.reset();
  EXPECT_EQ(run_pair(without_nms).size(), 4u);

  // Two different classes on the same box: class aware keeps both, agnostic
  // keeps one.
  std::vector<float> classes = values;
  put(classes, kPredictions, 5, 1, 0.95f);
  put(classes, kPredictions, 4, 1, 0.1f);
  ObbDecodeConfig agnostic = class_aware;
  agnostic.class_aware = false;
  const ObbBatch agnostic_cpu =
      run_cpu(make_tensor(classes, kChannels, kPredictions, 1, DeviceType::Cpu), agnostic, transform);
  const ObbBatch agnostic_cuda =
      run_cuda(make_tensor(classes, kChannels, kPredictions, 1, DeviceType::Cuda), agnostic, transform);
  expect_parity(agnostic_cpu, agnostic_cuda);
  EXPECT_EQ(agnostic_cpu.results[0].objects.size(), 3u);
}

TEST(ObbCudaDecoder, MatchesCpuOnIouThresholdBoundary) {
  // Axis-aligned boxes make the double IoU exactly 0.5 on both sides, so the
  // >= rule can be probed at the float threshold without a numeric gray zone.
  constexpr int kChannels = 4 + 2 + 1;
  constexpr int kPredictions = 2;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  auto row = [&](int prediction, float cy, float height, float score) {
    put(values, kPredictions, 0, prediction, 5.f);
    put(values, kPredictions, 1, prediction, cy);
    put(values, kPredictions, 2, prediction, 10.f);
    put(values, kPredictions, 3, prediction, height);
    put(values, kPredictions, 4, prediction, score);
    put(values, kPredictions, 5, prediction, 0.f);
    put(values, kPredictions, 6, prediction, 0.f);
  };
  row(0, 5.f, 10.f, 0.9f);
  row(1, 2.5f, 5.f, 0.8f);
  const GeometryTransform transform = geometry(20, 20);
  auto run_pair = [&](float iou) {
    ObbDecodeConfig config = keep_config();
    config.iou_threshold = iou;
    config.max_detections = 10;
    config.pre_nms_top_k = 10;
    const ObbBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
    const ObbBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
    expect_parity(cpu, cuda);
    return cpu.results[0].objects.size();
  };
  EXPECT_EQ(run_pair(0.5f), 1u);                                              // exactly at the threshold
  EXPECT_EQ(run_pair(std::nextafter(0.5f, 0.0f)), 1u);                        // slightly below 0.5
  EXPECT_EQ(run_pair(std::nextafter(0.5f, 1.0f)), 2u);                        // slightly above 0.5
  EXPECT_EQ(run_pair(1.0f), 2u);
  EXPECT_EQ(run_pair(0.0f), 1u);
}

TEST(ObbCudaDecoder, RejectsIllegalValuesLikeCpu) {
  constexpr int kChannels = 4 + 2 + 1;
  constexpr int kPredictions = 2;
  const GeometryTransform transform = geometry(64, 64);
  const ObbDecodeConfig config = keep_config();

  auto probe = [&](const std::function<void(std::vector<float>&)>& mutate, bool exact = true) {
    std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
    for (int prediction = 0; prediction < kPredictions; ++prediction) {
      put(values, kPredictions, 0, prediction, 20.f);
      put(values, kPredictions, 1, prediction, 20.f);
      put(values, kPredictions, 2, prediction, 8.f);
      put(values, kPredictions, 3, prediction, 8.f);
      put(values, kPredictions, 4, prediction, 0.9f);
      put(values, kPredictions, 5, prediction, 0.1f);
      put(values, kPredictions, 6, prediction, 0.f);
    }
    mutate(values);
    expect_same_failure(
        [&] {
          run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
        },
        [&] {
          run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
        },
        exact);
  };

  probe([](std::vector<float>& values) { put(values, kPredictions, 0, 1, std::nanf("")); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 1, 1, std::numeric_limits<float>::infinity()); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 3, 1, -std::numeric_limits<float>::infinity()); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 4, 1, 1.5f); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 5, 1, -0.5f); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 2, 1, 0.f); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 3, 1, -2.f); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 6, 1, kAngleMax + 0.01f); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 6, 1, kAngleMin - 0.01f); });
  probe([](std::vector<float>& values) { put(values, kPredictions, 6, 1, std::nanf("")); });

  // The five box channels are read for every anchor before the score gate, so a
  // bad geometry is reported even when the anchor is below the threshold. A zero
  // size or an angle outside the range is only rejected for kept anchors.
  std::vector<float> dropped(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  put(dropped, kPredictions, 0, 0, 20.f);
  put(dropped, kPredictions, 1, 0, 20.f);
  put(dropped, kPredictions, 2, 0, 8.f);
  put(dropped, kPredictions, 3, 0, 8.f);
  put(dropped, kPredictions, 4, 0, 0.9f);
  put(dropped, kPredictions, 6, 0, 0.f);
  put(dropped, kPredictions, 0, 1, 30.f);
  put(dropped, kPredictions, 1, 1, 30.f);
  put(dropped, kPredictions, 2, 1, 0.f);  // zero width, below the threshold
  put(dropped, kPredictions, 3, 1, 0.f);
  put(dropped, kPredictions, 4, 1, 0.1f);
  put(dropped, kPredictions, 6, 1, 9.f);  // angle far outside the range
  const ObbBatch cpu = run_cpu(make_tensor(dropped, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
  const ObbBatch cuda = run_cuda(make_tensor(dropped, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
  expect_parity(cpu, cuda);
  EXPECT_EQ(cpu.results[0].objects.size(), 1u);
}

TEST(ObbCudaDecoder, RejectsTensorAndRequestContracts) {
  constexpr int kChannels = 4 + 2 + 1;
  constexpr int kPredictions = 2;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    put(values, kPredictions, 0, prediction, 20.f);
    put(values, kPredictions, 1, prediction, 20.f);
    put(values, kPredictions, 2, prediction, 8.f);
    put(values, kPredictions, 3, prediction, 8.f);
    put(values, kPredictions, 4, prediction, 0.9f);
    put(values, kPredictions, 6, prediction, 0.f);
  }
  const GeometryTransform transform = geometry(64, 64);
  const ObbDecodeConfig config = keep_config();
  const Tensor cpu_tensor = make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu);
  const Tensor cuda_tensor = make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda);

  // No implicit H2D: a CPU tensor must be rejected by the CUDA decoder.
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, run_cuda(cpu_tensor, config, transform));

  expect_same_failure(
      [&] {
        const Tensor wrong_rank = Tensor::from_host(DeviceContext{}, DataType::Fp32, {kChannels, kPredictions},
                                                    values.data(), values.size() * sizeof(float));
        run_cpu(wrong_rank, config, transform);
      },
      [&] {
        const Tensor wrong_rank =
            Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32, {kChannels, kPredictions},
                              values.data(), values.size() * sizeof(float));
        run_cuda(wrong_rank, config, transform);
      });

  expect_same_failure(
      [&] {
        const std::vector<float> batched(2 * values.size(), 0.f);
        const Tensor tensor = Tensor::from_host(DeviceContext{}, DataType::Fp32, {2, kChannels, kPredictions},
                                                batched.data(), batched.size() * sizeof(float));
        run_cpu(tensor, config, transform);
      },
      [&] {
        const std::vector<float> batched(2 * values.size(), 0.f);
        const Tensor tensor =
            Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32, {2, kChannels, kPredictions},
                              batched.data(), batched.size() * sizeof(float));
        run_cuda(tensor, config, transform);
      });

  expect_same_failure(
      [&] {
        const Tensor wrong_channels = Tensor::from_host(DeviceContext{}, DataType::Fp32,
                                                        {1, kChannels - 1, kPredictions}, values.data(),
                                                        values.size() * sizeof(float));
        run_cpu(wrong_channels, config, transform);
      },
      [&] {
        const Tensor wrong_channels =
            Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32,
                              {1, kChannels - 1, kPredictions}, values.data(), values.size() * sizeof(float));
        run_cuda(wrong_channels, config, transform);
      });

  expect_same_failure(
      [&] {
        const std::vector<float> empty;
        const Tensor tensor = Tensor::from_host(DeviceContext{}, DataType::Fp32, {1, kChannels, 0}, empty.data(), 0);
        run_cpu(tensor, config, transform);
      },
      [&] {
        const std::vector<float> empty;
        const Tensor tensor =
            Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32, {1, kChannels, 0},
                              empty.data(), 0);
        run_cuda(tensor, config, transform);
      });

  expect_same_failure(
      [&] {
        const std::vector<std::int32_t> ints(static_cast<std::size_t>(kChannels * kPredictions), 0);
        const Tensor wrong_dtype = Tensor::from_host(DeviceContext{}, DataType::Int32, {1, kChannels, kPredictions},
                                                     ints.data(), ints.size() * sizeof(std::int32_t));
        run_cpu(wrong_dtype, config, transform);
      },
      [&] {
        const std::vector<std::int32_t> ints(static_cast<std::size_t>(kChannels * kPredictions), 0);
        const Tensor wrong_dtype =
            Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Int32,
                              {1, kChannels, kPredictions}, ints.data(), ints.size() * sizeof(std::int32_t));
        run_cuda(wrong_dtype, config, transform);
      },
      false);

  // Non-contiguous view: base [1, channels, 4] read with a stride of 2 columns.
  expect_same_failure(
      [&] {
        std::vector<float> wide(static_cast<std::size_t>(kChannels * kPredictions * 2), 0.f);
        for (int channel = 0; channel < kChannels; ++channel) {
          for (int prediction = 0; prediction < kPredictions; ++prediction) {
            wide[static_cast<std::size_t>(channel) * 4 + static_cast<std::size_t>(prediction * 2)] =
                values[static_cast<std::size_t>(channel) * kPredictions + static_cast<std::size_t>(prediction)];
          }
        }
        const Tensor base = Tensor::from_host(DeviceContext{}, DataType::Fp32, {1, kChannels, 4}, wide.data(),
                                              wide.size() * sizeof(float));
        const Tensor strided = base.view({1, kChannels, kPredictions}, {kChannels * 4, 4, 2}, 0);
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
        const Tensor base =
            Tensor::from_host(DeviceContext(DeviceType::Cuda, kDevice), DataType::Fp32, {1, kChannels, 4},
                              wide.data(), wide.size() * sizeof(float));
        const Tensor strided = base.view({1, kChannels, kPredictions}, {kChannels * 4, 4, 2}, 0);
        run_cuda(strided, config, transform);
      },
      false);

  vie::DecoderInput missing = decode_input(cuda_tensor, transform, config.output_name);
  missing.outputs.clear();
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument,
                   vie::ObbCudaDecoder{}.decode_obb(missing, config, DeviceContext(DeviceType::Cuda, kDevice)));

  vie::DecoderInput no_geometry = decode_input(cuda_tensor, transform, config.output_name);
  no_geometry.geometry.reset();
  VIE_EXPECT_ERROR(ErrorCode::UnresolvedField,
                   vie::ObbCudaDecoder{}.decode_obb(no_geometry, config, DeviceContext(DeviceType::Cuda, kDevice)));

  GeometryTransform bad_geometry = transform;
  bad_geometry.original_width = 0;
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, run_cuda(cuda_tensor, config, bad_geometry));

  ObbDecodeConfig bad_threshold = config;
  bad_threshold.confidence_threshold = 1.5f;
  VIE_EXPECT_ERROR(ErrorCode::UnresolvedField,
                   vie::ObbCudaDecoder{}.decode_obb(decode_input(cuda_tensor, transform, config.output_name),
                                                    bad_threshold, DeviceContext(DeviceType::Cuda, kDevice)));

  VIE_EXPECT_ERROR(ErrorCode::Unsupported, vie::ObbCudaDecoder{}.decode({}, {}));
  VIE_EXPECT_ERROR(ErrorCode::Unsupported,
                   vie::ObbCudaDecoder{}.decode_obb(decode_input(cuda_tensor, transform, config.output_name), config,
                                                    DeviceContext{}));

  // The dispatcher routes by device and never falls back.
  const ObbBatch dispatched = vie::ObbDispatchDecoder{}.decode_obb(
      decode_input(cpu_tensor, transform, config.output_name), config, DeviceContext{});
  EXPECT_EQ(dispatched.results[0].execution_device, DeviceType::Cpu);
  const ObbBatch dispatched_cuda = vie::ObbDispatchDecoder{}.decode_obb(
      decode_input(cuda_tensor, transform, config.output_name), config, DeviceContext(DeviceType::Cuda, kDevice));
  EXPECT_EQ(dispatched_cuda.results[0].execution_device, DeviceType::Cuda);
}

TEST(ObbCudaIou, MatchesCpuOnEveryRequiredGeometry) {
  std::vector<OrientedBox> left;
  std::vector<OrientedBox> right;
  auto add = [&](OrientedBox a, OrientedBox b) {
    left.push_back(a);
    right.push_back(b);
  };
  const float quarter = kPi * 0.25f;
  // identical
  add({50.f, 50.f, 20.f, 10.f, 0.3f}, {50.f, 50.f, 20.f, 10.f, 0.3f});
  // separated
  add({10.f, 10.f, 8.f, 8.f, 0.f}, {80.f, 80.f, 8.f, 8.f, 0.f});
  // partial overlap, axis aligned
  add({0.f, 0.f, 10.f, 10.f, 0.f}, {5.f, 0.f, 10.f, 10.f, 0.f});
  // containment
  add({50.f, 50.f, 40.f, 20.f, 0.2f}, {50.f, 50.f, 6.f, 4.f, 0.2f});
  // edge touching
  add({0.f, 0.f, 10.f, 10.f, 0.f}, {10.f, 0.f, 10.f, 10.f, 0.f});
  // vertex touching
  add({0.f, 0.f, 10.f, 10.f, 0.f}, {10.f, 10.f, 10.f, 10.f, 0.f});
  // parallel edges, different angles
  add({30.f, 30.f, 20.f, 6.f, 0.6f}, {33.f, 34.f, 20.f, 6.f, 0.6f});
  // collinear edges
  add({30.f, 30.f, 20.f, 6.f, 0.6f}, {30.f, 36.f, 20.f, 6.f, 0.6f});
  // nearly parallel
  add({30.f, 30.f, 20.f, 6.f, 0.6f}, {30.f, 30.f, 20.f, 6.f, 0.6f + 1e-4f});
  // small intersection
  add({0.f, 0.f, 10.f, 10.f, 0.f}, {9.9f, 0.f, 10.f, 10.f, 0.f});
  // angle combinations
  for (float left_angle : {-0.7f, -0.2f, 0.f, 0.45f, 1.1f, 2.2f}) {
    for (float right_angle : {-0.6f, 0.1f, 0.7f, 1.7f}) {
      add({40.f, 40.f, 24.f, 9.f, left_angle}, {44.f, 43.f, 14.f, 18.f, right_angle});
    }
  }
  // angle normalization boundary
  add({40.f, 40.f, 20.f, 8.f, kAngleMax - 1e-4f}, {40.f, 40.f, 20.f, 8.f, kAngleMin});
  add({40.f, 40.f, 20.f, 8.f, kAngleMax}, {40.f, 40.f, 8.f, 20.f, kAngleMin});
  // swapped width/height is the same corner set
  add({25.f, 25.f, 20.f, 8.f, 0.1f}, {25.f, 25.f, 8.f, 20.f, 0.1f + quarter});
  // huge coordinates
  add({1e7f, 1e7f, 2000.f, 1000.f, 0.37f}, {1e7f + 500.f, 1e7f - 200.f, 1800.f, 1200.f, 0.41f});
  // tiny valid boxes
  add({0.5f, 0.5f, 1e-3f, 1e-3f, 0.2f}, {0.5005f, 0.5f, 1e-3f, 1e-3f, 0.2f});
  // degenerate rotation of a near-square
  add({10.f, 10.f, 12.f, 12.f, 0.f}, {10.f, 10.f, 12.f, 12.f, quarter});
  // highly eccentric
  add({20.f, 20.f, 200.f, 0.5f, 1.f}, {20.f, 20.f, 190.f, 0.5f, 1.1f});

  const vie::ObbCudaIouBatch cuda = vie::launch_obb_iou_cuda(left, right, kDevice);
  ASSERT_EQ(cuda.values.size(), left.size());
  double max_error = 0;
  double sum_error = 0;
  int exact = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const float want = vie::rotated_intersection_over_union(left[index], right[index]);
    const float got = cuda.values[index];
    ASSERT_GE(got, 0.f) << index;
    ASSERT_LE(got, 1.f) << index;
    const double error = std::fabs(static_cast<double>(got) - static_cast<double>(want));
    max_error = std::max(max_error, error);
    sum_error += error;
    if (got == want) ++exact;
    EXPECT_NEAR(got, want, 1e-6f) << index << " cpu " << want << " cuda " << got;
  }
  std::ofstream record("/tmp/vie_phase15_iou.txt");
  record << "pairs " << left.size() << " exact " << exact << " max_abs " << max_error << " mean_abs "
         << (sum_error / static_cast<double>(left.size())) << "\n";
  for (std::size_t index = 0; index < left.size(); ++index) {
    const float want = vie::rotated_intersection_over_union(left[index], right[index]);
    record << index << " cpu " << want << " cuda " << cuda.values[index] << " abs "
           << std::fabs(static_cast<double>(want) - static_cast<double>(cuda.values[index])) << "\n";
  }

  // Invalid boxes fail with the CPU message.
  expect_same_failure(
      [&] { vie::rotated_intersection_over_union({0.f, 0.f, std::nanf(""), 4.f, 0.f}, {1.f, 1.f, 4.f, 4.f, 0.f}); },
      [&] { vie::launch_obb_iou_cuda({{0.f, 0.f, std::nanf(""), 4.f, 0.f}}, {{1.f, 1.f, 4.f, 4.f, 0.f}}, kDevice); });
  expect_same_failure(
      [&] { vie::rotated_intersection_over_union({0.f, 0.f, 0.f, 4.f, 0.f}, {1.f, 1.f, 4.f, 4.f, 0.f}); },
      [&] { vie::launch_obb_iou_cuda({{0.f, 0.f, 0.f, 4.f, 0.f}}, {{1.f, 1.f, 4.f, 4.f, 0.f}}, kDevice); });
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument,
                   vie::launch_obb_iou_cuda({{0.f, 0.f, 4.f, 4.f, 0.f}}, {}, kDevice));
}

TEST(ObbCudaDecoder, RunsTheResolvedPhase12Policy) {
  vie::ModelManifest manifest;
  manifest.model_name = "obb_policy";
  manifest.task = vie::TaskType::Obb;
  manifest.model_format = vie::ModelFormat::Mock;
  manifest.num_classes = 2;
  manifest.input.name = "images";
  manifest.input.shape = {1, 3, 32, 32};
  manifest.input.dtype = DataType::Fp32;
  manifest.input.layout = vie::Layout::Nchw;
  manifest.input.color = "rgb";
  manifest.input.normalization = "scale_0_1";
  manifest.input.resize = "letterbox";
  manifest.input.interpolation = "bilinear";
  manifest.input.padding_value = 114.f;
  vie::OutputDesc output;
  output.name = "output0";
  output.shape = {1, 7, 2};
  output.dtype = DataType::Fp32;
  manifest.outputs.push_back(output);
  manifest.decoder.protocol = "yolov8_obb_raw";
  manifest.decoder.box_format = "cxcywh_angle";
  manifest.decoder.channel_axis = 1;
  manifest.decoder.box_channels = 4;
  manifest.decoder.scores_are_probabilities = true;
  manifest.obb.output_name = "output0";
  manifest.obb.channel_axis = 1;
  manifest.obb.angle_unit = "radian";
  manifest.obb.angle_range = "[-pi/4,3pi/4]";
  manifest.obb.score_type = "probabilities";
  manifest.ensure_provenance();

  vie::InferenceOptions options;
  options.obb.confidence_threshold = 0.35f;
  options.obb.iou_threshold = 0.4f;
  options.obb.max_detections = 7;
  options.obb.pre_nms_top_k = 9;
  options.obb.class_agnostic_nms = true;
  const vie::ResolvedInferenceConfig resolved =
      vie::resolve_inference_config(manifest.task, manifest, vie::InferenceOptions{}, options);
  const ObbDecodeConfig config = vie::obb_decode_config_from_resolved(manifest, resolved);
  const vie::ResolvedObbConfig& policy = resolved.require_obb();
  EXPECT_EQ(config.output_name, "output0");
  EXPECT_EQ(config.channel_axis, 1);
  EXPECT_EQ(config.num_classes, 2);
  ASSERT_TRUE(config.class_selection.has_value());
  EXPECT_EQ(*config.class_selection, ClassSelection::BestClass);
  EXPECT_FLOAT_EQ(*config.confidence_threshold, 0.35f);
  EXPECT_FLOAT_EQ(*config.iou_threshold, 0.4f);
  EXPECT_EQ(*config.max_detections, 7);
  EXPECT_EQ(*config.pre_nms_top_k, 9);
  EXPECT_TRUE(*config.nms_enabled);
  EXPECT_FALSE(*config.class_aware);
  EXPECT_FLOAT_EQ(policy.confidence_threshold.value, *config.confidence_threshold);
  EXPECT_FLOAT_EQ(policy.iou_threshold.value, *config.iou_threshold);

  constexpr int kChannels = 7;
  constexpr int kPredictions = 2;
  std::vector<float> values(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  for (int prediction = 0; prediction < kPredictions; ++prediction) {
    put(values, kPredictions, 0, prediction, 16.f);
    put(values, kPredictions, 1, prediction, 16.f);
    put(values, kPredictions, 2, prediction, 8.f);
    put(values, kPredictions, 3, prediction, 8.f);
    put(values, kPredictions, 4, prediction, 0.9f);
    put(values, kPredictions, 5, prediction, 0.2f);
    put(values, kPredictions, 6, prediction, 0.f);
  }
  const GeometryTransform transform = geometry(32, 32);
  const ObbBatch cpu = run_cpu(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cpu), config, transform);
  const ObbBatch cuda = run_cuda(make_tensor(values, kChannels, kPredictions, 1, DeviceType::Cuda), config, transform);
  expect_parity(cpu, cuda);
  EXPECT_EQ(cpu.results[0].objects.size(), 1u);  // class agnostic NMS is applied
}

TEST(ObbCudaPerf, Profile) {
  // Synthetic three-size profile. Timing is recorded, not asserted, because a
  // loaded machine cannot provide a stable upper bound.
  struct Case {
    const char* name;
    int predictions;
    int classes;
    int top_k;
  };
  const std::vector<Case> cases{{"small", 64, 4, 64}, {"medium", 1024, 8, 1024}, {"large", 4096, 15, 4096}};
  std::ofstream record("/tmp/vie_phase15_perf.txt");
  for (const Case& item : cases) {
    const int channels = 4 + item.classes + 1;
    std::vector<float> values(static_cast<std::size_t>(channels) * static_cast<std::size_t>(item.predictions), 0.f);
    const int angle_channel = 4 + item.classes;
    for (int prediction = 0; prediction < item.predictions; ++prediction) {
      const float step = static_cast<float>(prediction);
      const float column = std::fmod(step, 64.f);
      const float row = std::floor(step / 64.f);
      put(values, item.predictions, 0, prediction, 40.f + column * 24.f);
      put(values, item.predictions, 1, prediction, 40.f + row * 30.f);
      put(values, item.predictions, 2, prediction, 18.f + std::fmod(step * 3.f, 30.f));
      put(values, item.predictions, 3, prediction, 10.f + std::fmod(step * 2.f, 20.f));
      put(values, item.predictions, angle_channel, prediction, -0.7f + std::fmod(step * 0.017f, 2.9f));
      for (int cls = 0; cls < item.classes; ++cls) {
        const float score = 0.55f + std::fmod(step * 0.011f + static_cast<float>(cls) * 0.07f, 0.44f);
        put(values, item.predictions, 4 + cls, prediction, score);
      }
    }
    ObbDecodeConfig config = keep_config();
    config.num_classes = item.classes;
    config.confidence_threshold = 0.5f;
    config.iou_threshold = 0.45f;
    config.class_aware = true;
    config.pre_nms_top_k = item.top_k;
    config.max_detections = 100;
    GeometryTransform transform = geometry(2048, 1600);
    const Tensor cpu_tensor = make_tensor(values, channels, item.predictions, 1, DeviceType::Cpu);
    const Tensor cuda_tensor = make_tensor(values, channels, item.predictions, 1, DeviceType::Cuda);
    constexpr int kWarmup = 2;
    constexpr int kSamples = 5;
    for (int index = 0; index < kWarmup; ++index) {
      run_cpu(cpu_tensor, config, transform);
      run_cuda(cuda_tensor, config, transform);
    }
    std::vector<double> cpu_ms;
    std::vector<double> cuda_ms;
    std::vector<double> validate_ms;
    std::vector<double> sort_ms;
    std::vector<double> map_ms;
    std::vector<double> nms_ms;
    std::vector<double> d2h_ms;
    int before_nms = 0;
    int kept = 0;
    long long iou_calls = 0;
    std::size_t d2h_bytes = 0;
    for (int index = 0; index < kSamples; ++index) {
      auto start = std::chrono::steady_clock::now();
      const ObbBatch cpu = run_cpu(cpu_tensor, config, transform);
      cpu_ms.push_back(milliseconds_since(start));
      start = std::chrono::steady_clock::now();
      const ObbBatch cuda = run_cuda(cuda_tensor, config, transform);
      cuda_ms.push_back(milliseconds_since(start));
      validate_ms.push_back(cuda.results[0].validate_ms);
      sort_ms.push_back(cuda.results[0].sort_ms);
      map_ms.push_back(cuda.results[0].map_ms);
      nms_ms.push_back(cuda.results[0].nms_ms);
      d2h_ms.push_back(cuda.results[0].d2h_ms);
      before_nms = cuda.results[0].candidates_before_nms;
      kept = static_cast<int>(cuda.results[0].objects.size());
      d2h_bytes = cuda.results[0].d2h_bytes;
      iou_calls = cuda.results[0].iou_calls;
      EXPECT_EQ(cpu.results[0].objects.size(), cuda.results[0].objects.size());
    }
    const auto mean = [](const std::vector<double>& samples) {
      return std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    };
    record << "case " << item.name << " predictions " << item.predictions << " classes " << item.classes
           << " channels " << channels << " top_k " << item.top_k << " before_nms " << before_nms << " kept "
           << kept << " iou_calls " << iou_calls << " d2h_bytes " << d2h_bytes << " warmup " << kWarmup
           << " samples " << kSamples << "\n";
    record << "case " << item.name << " cpu_wall_mean " << mean(cpu_ms) << " cuda_wall_mean " << mean(cuda_ms)
           << " validate " << mean(validate_ms) << " sort " << mean(sort_ms) << " map " << mean(map_ms)
           << " nms " << mean(nms_ms) << " d2h " << mean(d2h_ms) << "\n";
  }
}