#include "test_util.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/decoder/rtdetr_cuda_decoder.hpp"
#include "vie/decoder/rtdetr_decoder.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/preprocess/preprocess.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct ReferenceDetection {
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;
  float score = 0.f;
  int class_id = -1;
  int query = -1;
};

struct ReferenceBatch {
  std::vector<ReferenceDetection> detections;
  int passed = 0;
  int kept_before_limit = 0;
};

float clip_axis(float value, float limit) {
  if (value < 0.f) return 0.f;
  if (value > limit) return limit;
  return value;
}

float map_axis(float value, int pad, double scale) {
  return static_cast<float>((static_cast<double>(value) - static_cast<double>(pad)) / scale);
}

// Independent scalar decode. It does not call a VIE decoder or map_box_to_original.
ReferenceBatch reference_decode(const std::vector<float>& data, int queries, int classes, float threshold,
                                int max_detections, const vie::GeometryTransform& geometry) {
  ReferenceBatch batch;
  const int channels = 4 + classes;
  const float model_width = static_cast<float>(geometry.output_width);
  const float model_height = static_cast<float>(geometry.output_height);
  const float original_width = static_cast<float>(geometry.original_width);
  const float original_height = static_cast<float>(geometry.original_height);
  for (int query = 0; query < queries; ++query) {
    const float* row = data.data() + static_cast<std::size_t>(query * channels);
    int best_class = 0;
    float best_score = row[4];
    for (int cls = 1; cls < classes; ++cls) {
      const float score = row[4 + cls];
      if (score > best_score || (score == best_score && cls < best_class)) {
        best_score = score;
        best_class = cls;
      }
    }
    if (best_score < threshold) continue;
    ++batch.passed;
    const float cx = row[0] * model_width;
    const float cy = row[1] * model_height;
    const float width = row[2] * model_width;
    const float height = row[3] * model_height;
    ReferenceDetection item;
    item.x1 = clip_axis(map_axis(cx - width * 0.5f, geometry.pad_left, geometry.scale_x), original_width);
    item.x2 = clip_axis(map_axis(cx + width * 0.5f, geometry.pad_left, geometry.scale_x), original_width);
    item.y1 = clip_axis(map_axis(cy - height * 0.5f, geometry.pad_top, geometry.scale_y), original_height);
    item.y2 = clip_axis(map_axis(cy + height * 0.5f, geometry.pad_top, geometry.scale_y), original_height);
    if (!(item.x2 > item.x1) || !(item.y2 > item.y1)) continue;
    item.score = best_score;
    item.class_id = best_class;
    item.query = query;
    batch.detections.push_back(item);
  }
  std::sort(batch.detections.begin(), batch.detections.end(),
            [](const ReferenceDetection& left, const ReferenceDetection& right) {
              if (left.score != right.score) return left.score > right.score;
              return left.query < right.query;
            });
  batch.kept_before_limit = static_cast<int>(batch.detections.size());
  if (batch.kept_before_limit > max_detections) {
    batch.detections.resize(static_cast<std::size_t>(max_detections));
  }
  return batch;
}

vie::DecoderConfig config_for(int queries, int classes, float threshold, int max_detections) {
  vie::DecoderConfig config;
  config.output_name = "output0";
  config.channel_axis = 2;
  config.box_channels = 4;
  config.num_classes = classes;
  config.num_queries = queries;
  config.box_decode = vie::BoxDecode::DecodedNormalizedCxcywh;
  config.scores_are_probabilities = true;
  config.class_selection = vie::ClassSelection::BestClass;
  config.confidence_threshold = threshold;
  config.nms_enabled = false;
  config.max_detections = max_detections;
  return config;
}

vie::GeometryTransform stretch(int source_width, int source_height, int model_width, int model_height) {
  vie::PreprocessConfig config;
  config.resize_mode = vie::ResizeMode::Stretch;
  config.target_width = model_width;
  config.target_height = model_height;
  return vie::make_geometry(config, source_width, source_height);
}

vie::GeometryTransform letterbox(int source_width, int source_height, int model_width, int model_height) {
  vie::PreprocessConfig config;
  config.resize_mode = vie::ResizeMode::Letterbox;
  config.target_width = model_width;
  config.target_height = model_height;
  return vie::make_geometry(config, source_width, source_height);
}

std::vector<float> blank(int queries, int classes) {
  return std::vector<float>(static_cast<std::size_t>(queries * (4 + classes)), 0.f);
}

void write_row(std::vector<float>& data, int query, int classes, float cx, float cy, float width, float height,
               const std::vector<float>& scores) {
  float* row = data.data() + static_cast<std::size_t>(query * (4 + classes));
  row[0] = cx;
  row[1] = cy;
  row[2] = width;
  row[3] = height;
  for (int cls = 0; cls < classes; ++cls) row[4 + cls] = scores[static_cast<std::size_t>(cls)];
}

vie::Tensor host_tensor(const std::vector<float>& data, int queries, int classes) {
  return vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32,
                                {1, static_cast<std::int64_t>(queries), static_cast<std::int64_t>(4 + classes)},
                                data.data(), data.size() * sizeof(float));
}

vie::DetectionBatch decode_cuda(const std::vector<float>& data, const vie::DecoderConfig& config,
                                const vie::GeometryTransform& geometry) {
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  vie::Tensor device = host_tensor(data, static_cast<int>(*config.num_queries), config.num_classes).to(gpu);
  vie::DecoderInput input;
  input.outputs.emplace(config.output_name, device);
  input.geometry = geometry;
  return vie::RtdetrCudaDecoder{}.decode_batch(input, config, gpu);
}

void expect_match(const ReferenceBatch& expected, const vie::DetectionBatch& actual) {
  ASSERT_EQ(actual.results.size(), 1u);
  const vie::DetectionResult& result = actual.results[0];
  EXPECT_EQ(result.execution_device, vie::DeviceType::Cuda);
  EXPECT_EQ(result.protocol, "rtdetr_detection_raw");
  EXPECT_FALSE(result.nms_applied);
  EXPECT_TRUE(result.coordinates_mapped);
  EXPECT_EQ(result.candidates_after_filter, expected.passed);
  EXPECT_EQ(result.candidates_before_nms, expected.kept_before_limit);
  ASSERT_EQ(result.detections.size(), expected.detections.size());
  EXPECT_EQ(result.d2h_bytes, result.detections.size() * 28u);
  for (std::size_t i = 0; i < expected.detections.size(); ++i) {
    EXPECT_EQ(result.detections[i].class_id, expected.detections[i].class_id);
    EXPECT_EQ(result.detections[i].query_index, expected.detections[i].query);
    EXPECT_FLOAT_EQ(result.detections[i].confidence, expected.detections[i].score);
    EXPECT_NEAR(result.detections[i].box.x1, expected.detections[i].x1, 1e-4f);
    EXPECT_NEAR(result.detections[i].box.y1, expected.detections[i].y1, 1e-4f);
    EXPECT_NEAR(result.detections[i].box.x2, expected.detections[i].x2, 1e-4f);
    EXPECT_NEAR(result.detections[i].box.y2, expected.detections[i].y2, 1e-4f);
  }
}

struct Summary {
  double mean = 0;
  double p50 = 0;
  double p95 = 0;
  double min = 0;
  double max = 0;
};

Summary summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  Summary out;
  double sum = 0;
  for (double value : values) sum += value;
  out.mean = sum / static_cast<double>(values.size());
  out.p50 = values[values.size() / 2];
  out.p95 = values[static_cast<std::size_t>(std::floor(0.95 * values.size()))];
  out.min = values.front();
  out.max = values.back();
  return out;
}

void write_summary(std::ostream& out, const char* name, const Summary& value) {
  out << name << " mean " << value.mean << " p50 " << value.p50 << " p95 " << value.p95 << " min " << value.min
      << " max " << value.max << "\n";
}

}  // namespace

TEST(RtdetrCudaDecoder, OneQueryMatchesTheIndependentReference) {
  const vie::GeometryTransform geometry = stretch(100, 50, 640, 640);
  EXPECT_NE(geometry.scale_x, geometry.scale_y);
  std::vector<float> data = blank(1, 3);
  write_row(data, 0, 3, 0.5f, 0.5f, 0.2f, 0.2f, {0.2f, 0.8f, 0.2f});
  const vie::DecoderConfig config = config_for(1, 3, 0.1f, 10);
  const ReferenceBatch expected = reference_decode(data, 1, 3, 0.1f, 10, geometry);
  const vie::DetectionBatch batch = decode_cuda(data, config, geometry);
  expect_match(expected, batch);
  ASSERT_EQ(batch.results[0].detections.size(), 1u);
  EXPECT_EQ(batch.results[0].detections[0].class_id, 1);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].confidence, 0.8f);
  EXPECT_NEAR(batch.results[0].detections[0].box.x1, 40.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.x2, 60.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.y1, 20.f, 1e-3f);
  EXPECT_NEAR(batch.results[0].detections[0].box.y2, 30.f, 1e-3f);
  EXPECT_EQ(batch.results[0].d2h_bytes, 28u);
}

TEST(RtdetrCudaDecoder, SeveralQueriesAndTheEmptyResultMatch) {
  const vie::GeometryTransform geometry = stretch(64, 32, 64, 32);
  std::vector<float> data = blank(4, 2);
  write_row(data, 0, 2, 0.2f, 0.3f, 0.1f, 0.2f, {0.4f, 0.1f});
  write_row(data, 1, 2, 0.4f, 0.4f, 0.2f, 0.2f, {0.7f, 0.7f});
  write_row(data, 2, 2, 0.6f, 0.5f, 0.2f, 0.2f, {0.9f, 0.1f});
  write_row(data, 3, 2, 0.8f, 0.6f, 0.1f, 0.2f, {0.6f, 0.2f});
  const vie::DecoderConfig config = config_for(4, 2, 0.5f, 10);
  expect_match(reference_decode(data, 4, 2, 0.5f, 10, geometry), decode_cuda(data, config, geometry));

  const vie::DecoderConfig capped = config_for(4, 2, 0.5f, 1);
  const vie::DetectionBatch limited = decode_cuda(data, capped, geometry);
  expect_match(reference_decode(data, 4, 2, 0.5f, 1, geometry), limited);
  ASSERT_EQ(limited.results[0].detections.size(), 1u);
  EXPECT_EQ(limited.results[0].detections[0].query_index, 2);
  EXPECT_EQ(limited.results[0].candidates_after_filter, 3);

  const vie::DetectionBatch empty = decode_cuda(data, config_for(4, 2, 0.95f, 10), geometry);
  expect_match(reference_decode(data, 4, 2, 0.95f, 10, geometry), empty);
  EXPECT_TRUE(empty.results[0].detections.empty());
  EXPECT_EQ(empty.results[0].d2h_bytes, 0u);
  EXPECT_EQ(empty.results[0].candidates_after_filter, 0);
}

TEST(RtdetrCudaDecoder, EqualScoresKeepTheSmallerClassAndEarlierQuery) {
  const vie::GeometryTransform geometry = stretch(80, 40, 80, 40);
  std::vector<float> data = blank(3, 3);
  write_row(data, 0, 3, 0.2f, 0.5f, 0.2f, 0.2f, {0.4f, 0.4f, 0.2f});
  write_row(data, 1, 3, 0.5f, 0.5f, 0.2f, 0.2f, {0.2f, 0.5f, 0.5f});
  write_row(data, 2, 3, 0.8f, 0.5f, 0.2f, 0.2f, {0.4f, 0.1f, 0.1f});
  const vie::DetectionBatch batch = decode_cuda(data, config_for(3, 3, 0.3f, 10), geometry);
  expect_match(reference_decode(data, 3, 3, 0.3f, 10, geometry), batch);
  ASSERT_EQ(batch.results[0].detections.size(), 3u);
  EXPECT_EQ(batch.results[0].detections[0].query_index, 1);
  EXPECT_EQ(batch.results[0].detections[0].class_id, 1);
  EXPECT_EQ(batch.results[0].detections[1].query_index, 0);
  EXPECT_EQ(batch.results[0].detections[1].class_id, 0);
  EXPECT_EQ(batch.results[0].detections[2].query_index, 2);
  const vie::DetectionBatch again = decode_cuda(data, config_for(3, 3, 0.3f, 10), geometry);
  ASSERT_EQ(again.results[0].detections.size(), batch.results[0].detections.size());
  for (std::size_t i = 0; i < batch.results[0].detections.size(); ++i) {
    EXPECT_EQ(again.results[0].detections[i].query_index, batch.results[0].detections[i].query_index);
    EXPECT_EQ(again.results[0].detections[i].class_id, batch.results[0].detections[i].class_id);
  }
}

TEST(RtdetrCudaDecoder, ClipsOutsideBoxesAndDropsZeroArea) {
  const vie::GeometryTransform geometry = stretch(100, 80, 100, 80);
  std::vector<float> data = blank(3, 1);
  write_row(data, 0, 1, -0.2f, 0.5f, 0.1f, 0.2f, {0.9f});
  write_row(data, 1, 1, 0.05f, 0.5f, 0.2f, 0.4f, {0.8f});
  write_row(data, 2, 1, 0.5f, 0.5f, 0.f, 0.2f, {0.95f});
  const vie::DetectionBatch batch = decode_cuda(data, config_for(3, 1, 0.1f, 10), geometry);
  expect_match(reference_decode(data, 3, 1, 0.1f, 10, geometry), batch);
  ASSERT_EQ(batch.results[0].detections.size(), 1u);
  EXPECT_EQ(batch.results[0].detections[0].query_index, 1);
  EXPECT_FLOAT_EQ(batch.results[0].detections[0].box.x1, 0.f);
  EXPECT_EQ(batch.results[0].candidates_after_filter, 3);
  EXPECT_EQ(batch.results[0].candidates_before_nms, 1);
}

TEST(RtdetrCudaDecoder, StretchLetterboxAndNonIntegerScalesMatch) {
  std::vector<float> data = blank(2, 2);
  write_row(data, 0, 2, 0.5f, 0.4f, 0.25f, 0.3f, {0.2f, 0.7f});
  write_row(data, 1, 2, 0.8f, 0.7f, 0.15f, 0.2f, {0.55f, 0.4f});
  const vie::DecoderConfig config = config_for(2, 2, 0.25f, 10);
  const vie::GeometryTransform geometries[] = {
      stretch(810, 1080, 640, 640),
      stretch(100, 50, 640, 640),
      stretch(1920, 1080, 640, 640),
      letterbox(810, 1080, 640, 640),
      letterbox(100, 400, 640, 640),
  };
  EXPECT_GT(geometries[3].pad_left + geometries[3].pad_top, 0);
  EXPECT_NE(geometries[1].scale_x, std::floor(geometries[1].scale_x));
  for (const vie::GeometryTransform& geometry : geometries) {
    expect_match(reference_decode(data, 2, 2, 0.25f, 10, geometry), decode_cuda(data, config, geometry));
  }
}

TEST(RtdetrCudaDecoder, RejectsNonFiniteValuesInsteadOfDroppingThem) {
  const vie::GeometryTransform geometry = stretch(32, 32, 32, 32);
  std::vector<float> data = blank(2, 2);
  write_row(data, 0, 2, 0.5f, 0.5f, 0.2f, 0.2f, {0.1f, 0.1f});
  write_row(data, 1, 2, 0.4f, 0.4f, 0.2f, 0.2f, {0.9f, 0.2f});
  data[2] = std::nanf("");
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decode_cuda(data, config_for(2, 2, 0.5f, 10), geometry));
  data[2] = 0.2f;
  data[5] = INFINITY;
  try {
    decode_cuda(data, config_for(2, 2, 0.5f, 10), geometry);
    FAIL() << "expected VieError";
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::InvalidArgument);
    EXPECT_NE(std::string(error.what()).find("not finite"), std::string::npos);
  }
}

TEST(RtdetrCudaDecoder, QueryIndexMatchesTheCpuDecoder) {
  const vie::GeometryTransform geometry = stretch(64, 32, 64, 32);
  std::vector<float> data = blank(5, 2);
  write_row(data, 0, 2, 0.2f, 0.5f, 0.1f, 0.2f, {0.2f, 0.1f});
  write_row(data, 1, 2, 0.35f, 0.5f, 0.2f, 0.2f, {0.55f, 0.55f});
  write_row(data, 2, 2, 0.5f, 0.5f, 0.f, 0.2f, {0.99f, 0.1f});
  write_row(data, 3, 2, 0.7f, 0.4f, 0.2f, 0.2f, {0.4f, 0.95f});
  write_row(data, 4, 2, 0.8f, 0.5f, 0.15f, 0.2f, {0.55f, 0.2f});
  const vie::DecoderConfig config = config_for(5, 2, 0.5f, 10);
  vie::DecoderInput cpu_input;
  cpu_input.outputs.emplace("output0", host_tensor(data, 5, 2));
  cpu_input.geometry = geometry;
  const vie::DetectionBatch cpu = vie::RtdetrDecoder{}.decode_batch(cpu_input, config, vie::DeviceContext{});
  const vie::DetectionBatch cuda = decode_cuda(data, config, geometry);
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  vie::DecoderInput cuda_input;
  cuda_input.outputs.emplace("output0", host_tensor(data, 5, 2).to(gpu));
  cuda_input.geometry = geometry;
  const vie::DetectionBatch routed =
      vie::RtdetrDetectionDecoder{}.decode_batch(cuda_input, config, gpu);
  const vie::DetectionBatch routed_cpu =
      vie::RtdetrDetectionDecoder{}.decode_batch(cpu_input, config, vie::DeviceContext{});

  ASSERT_EQ(cpu.results[0].detections.size(), 3u);
  ASSERT_EQ(cuda.results[0].detections.size(), cpu.results[0].detections.size());
  ASSERT_EQ(routed.results[0].detections.size(), cpu.results[0].detections.size());
  ASSERT_EQ(routed_cpu.results[0].detections.size(), cpu.results[0].detections.size());
  const int expected_queries[] = {3, 1, 4};
  for (std::size_t i = 0; i < cpu.results[0].detections.size(); ++i) {
    EXPECT_EQ(cpu.results[0].detections[i].query_index, expected_queries[i]);
    EXPECT_EQ(cuda.results[0].detections[i].query_index, cpu.results[0].detections[i].query_index);
    EXPECT_EQ(routed.results[0].detections[i].query_index, cpu.results[0].detections[i].query_index);
    EXPECT_EQ(routed_cpu.results[0].detections[i].query_index, cpu.results[0].detections[i].query_index);
    EXPECT_EQ(cuda.results[0].detections[i].class_id, cpu.results[0].detections[i].class_id);
    EXPECT_FLOAT_EQ(cuda.results[0].detections[i].confidence, cpu.results[0].detections[i].confidence);
    EXPECT_NEAR(cuda.results[0].detections[i].box.x1, cpu.results[0].detections[i].box.x1, 1e-4f);
    EXPECT_NEAR(cuda.results[0].detections[i].box.y1, cpu.results[0].detections[i].box.y1, 1e-4f);
    EXPECT_NEAR(cuda.results[0].detections[i].box.x2, cpu.results[0].detections[i].box.x2, 1e-4f);
    EXPECT_NEAR(cuda.results[0].detections[i].box.y2, cpu.results[0].detections[i].box.y2, 1e-4f);
  }
  EXPECT_EQ(cpu.results[0].detections[0].class_id, 1);
  EXPECT_EQ(cpu.results[0].detections[1].class_id, 0);
  EXPECT_EQ(cpu.results[0].d2h_bytes, 0u);
  EXPECT_EQ(cuda.results[0].d2h_bytes, 3u * 28u);
}

std::string finite_error(const std::vector<float>& data, int queries, int classes, bool cuda_device) {
  const vie::GeometryTransform geometry = stretch(32, 32, 32, 32);
  const vie::DecoderConfig config = config_for(queries, classes, 0.1f, 10);
  vie::DecoderInput input;
  vie::Tensor host = host_tensor(data, queries, classes);
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  input.outputs.emplace("output0", cuda_device ? host.to(gpu) : host);
  input.geometry = geometry;
  try {
    if (cuda_device) {
      vie::RtdetrCudaDecoder{}.decode_batch(input, config, gpu);
    } else {
      vie::RtdetrDecoder{}.decode_batch(input, config, vie::DeviceContext{});
    }
    ADD_FAILURE() << "expected VieError";
    return {};
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::InvalidArgument);
    return error.what();
  }
}

TEST(RtdetrCudaDecoder, NonFiniteLocationIsOneEncodingAndStaysPut) {
  constexpr int kQueries = 256;
  constexpr int kClasses = 2;
  constexpr int kRepeats = 32;
  auto poison = [](std::vector<float>& data, int query, int channel, float value) {
    data[static_cast<std::size_t>(query * (4 + kClasses) + channel)] = value;
  };
  struct Case {
    const char* expected;
    int queries;
    std::vector<float> data;
  };
  std::vector<float> split = blank(kQueries, kClasses);
  for (int query = 0; query < kQueries; ++query) {
    write_row(split, query, kClasses, 0.5f, 0.5f, 0.2f, 0.2f, {0.9f, 0.1f});
  }
  poison(split, 17, 5, std::nanf(""));
  poison(split, 90, 0, std::nanf(""));
  poison(split, 200, 2, INFINITY);

  std::vector<float> many_nan = blank(kQueries, kClasses);
  for (int query = 0; query < kQueries; ++query) {
    write_row(many_nan, query, kClasses, 0.4f, 0.4f, 0.2f, 0.2f, {0.8f, 0.2f});
  }
  poison(many_nan, 3, 0, std::nanf(""));
  poison(many_nan, 8, 0, std::nanf(""));
  poison(many_nan, 40, 0, std::nanf(""));

  std::vector<float> many_inf = blank(kQueries, kClasses);
  for (int query = 0; query < kQueries; ++query) {
    write_row(many_inf, query, kClasses, 0.4f, 0.4f, 0.2f, 0.2f, {0.8f, 0.2f});
  }
  poison(many_inf, 2, 3, INFINITY);
  poison(many_inf, 6, 1, -INFINITY);
  poison(many_inf, 11, 3, INFINITY);

  std::vector<float> same_query = blank(8, kClasses);
  for (int query = 0; query < 8; ++query) {
    write_row(same_query, query, kClasses, 0.5f, 0.5f, 0.2f, 0.2f, {0.7f, 0.3f});
  }
  poison(same_query, 4, 1, std::nanf(""));
  poison(same_query, 4, 2, INFINITY);
  poison(same_query, 4, 4, std::nanf(""));

  const Case cases[] = {
      {"class score is not finite at query 17", kQueries, split},
      {"normalized cx is not finite at query 3", kQueries, many_nan},
      {"normalized height is not finite at query 2", kQueries, many_inf},
      {"normalized cy is not finite at query 4", 8, same_query},
  };
  for (const Case& item : cases) {
    const std::string cpu = finite_error(item.data, item.queries, kClasses, false);
    EXPECT_EQ(cpu, item.expected);
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      EXPECT_EQ(finite_error(item.data, item.queries, kClasses, true), item.expected);
    }
  }
}

TEST(RtdetrCudaDecoder, RejectsBadInputsAndDoesNotFallBack) {
  const vie::GeometryTransform geometry = stretch(16, 16, 16, 16);
  const std::vector<float> data = blank(2, 1);
  const vie::DeviceContext cpu;
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  vie::DecoderConfig config = config_for(2, 1, 0.1f, 4);
  vie::Tensor host = host_tensor(data, 2, 1);
  vie::Tensor device = host.to(gpu);

  vie::DecoderInput cuda_input;
  cuda_input.outputs.emplace("output0", device);
  cuda_input.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, vie::RtdetrCudaDecoder{}.decode_batch(cuda_input, config, cpu));

  vie::DecoderInput cpu_tensor;
  cpu_tensor.outputs.emplace("output0", host);
  cpu_tensor.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::RtdetrCudaDecoder{}.decode_batch(cpu_tensor, config, gpu));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, vie::RtdetrDecoder{}.decode_batch(cuda_input, config, gpu));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::RtdetrDetectionDecoder{}.decode_batch(cpu_tensor, config, gpu));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::RtdetrDetectionDecoder{}.decode_batch(cuda_input, config, cpu));

  config.nms_enabled = true;
  config.iou_threshold = 0.5f;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, vie::RtdetrCudaDecoder{}.decode_batch(cuda_input, config, gpu));
  config.nms_enabled = false;

  vie::Tensor wrong_shape = vie::Tensor::zeros(gpu, vie::DataType::Fp32, {1, 4, 5});
  vie::DecoderInput shape_input;
  shape_input.outputs.emplace("output0", wrong_shape);
  shape_input.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::RtdetrCudaDecoder{}.decode_batch(shape_input, config, gpu));

  vie::Tensor wrong_type = vie::Tensor::zeros(gpu, vie::DataType::Int32, {1, 2, 5});
  vie::DecoderInput type_input;
  type_input.outputs.emplace("output0", wrong_type);
  type_input.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, vie::RtdetrCudaDecoder{}.decode_batch(type_input, config, gpu));

  vie::Tensor wide = vie::Tensor::zeros(gpu, vie::DataType::Fp32, {1, 2, 6});
  vie::Tensor strided = wide.view({1, 2, 5}, {12, 6, 1}, 0);
  ASSERT_FALSE(strided.contiguous());
  vie::DecoderInput strided_input;
  strided_input.outputs.emplace("output0", strided);
  strided_input.geometry = geometry;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, vie::RtdetrCudaDecoder{}.decode_batch(strided_input, config, gpu));

  const vie::DetectionBatch cpu_batch = vie::RtdetrDetectionDecoder{}.decode_batch(
      [&] {
        vie::DecoderInput input;
        input.outputs.emplace("output0", host);
        input.geometry = geometry;
        return input;
      }(),
      config, cpu);
  EXPECT_EQ(cpu_batch.results[0].execution_device, vie::DeviceType::Cpu);
  const vie::DetectionBatch gpu_batch = vie::RtdetrDetectionDecoder{}.decode_batch(cuda_input, config, gpu);
  EXPECT_EQ(gpu_batch.results[0].execution_device, vie::DeviceType::Cuda);
}

TEST(RtdetrCudaDecoder, BenchmarksKernelSortCopyAgainstCpu) {
  constexpr int kQueries = 300;
  constexpr int kClasses = 80;
  constexpr int kRuns = 100;
  const vie::GeometryTransform geometry = stretch(810, 1080, 640, 640);
  const vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  std::ofstream out("/tmp/vie_phase542_bench.txt");
  const int kept_targets[] = {1, 40, 300};
  const int limits[] = {1, 16, 300};
  for (int target : kept_targets) {
    for (int limit : limits) {
      std::vector<float> data = blank(kQueries, kClasses);
      for (int query = 0; query < kQueries; ++query) {
        const float score = query < target ? 0.9f - 0.001f * static_cast<float>(query) : 0.01f;
        std::vector<float> scores(static_cast<std::size_t>(kClasses), 0.001f);
        scores[query % kClasses] = score;
        write_row(data, query, kClasses, 0.2f + 0.002f * static_cast<float>(query % 50), 0.3f, 0.1f, 0.15f, scores);
      }
      const vie::DecoderConfig config = config_for(kQueries, kClasses, 0.05f, limit);
      vie::Tensor device = host_tensor(data, kQueries, kClasses).to(gpu);
      vie::DecoderInput input;
      input.outputs.emplace("output0", device);
      input.geometry = geometry;
      const auto cuda_first_start = std::chrono::steady_clock::now();
      const vie::DetectionBatch first = vie::RtdetrCudaDecoder{}.decode_batch(input, config, gpu);
      const double cuda_first_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_first_start).count();
      const int expected_kept = std::min(target, limit);
      ASSERT_EQ(static_cast<int>(first.results[0].detections.size()), expected_kept);
      EXPECT_EQ(first.results[0].d2h_bytes, static_cast<std::size_t>(expected_kept) * 28u);
      EXPECT_LT(first.results[0].d2h_bytes, static_cast<std::size_t>(kQueries * (4 + kClasses) * sizeof(float)));

      std::vector<double> wall;
      std::vector<double> kernel;
      std::vector<double> sort_ms;
      std::vector<double> d2h;
      wall.reserve(kRuns);
      for (int run = 0; run < kRuns; ++run) {
        const auto started = std::chrono::steady_clock::now();
        const vie::DetectionBatch batch = vie::RtdetrCudaDecoder{}.decode_batch(input, config, gpu);
        wall.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
        kernel.push_back(batch.results[0].gpu_execute_ms);
        sort_ms.push_back(batch.results[0].sort_ms);
        d2h.push_back(batch.results[0].d2h_ms);
      }
      vie::DecoderInput cpu_input;
      cpu_input.outputs.emplace("output0", host_tensor(data, kQueries, kClasses));
      cpu_input.geometry = geometry;
      const auto cpu_first_start = std::chrono::steady_clock::now();
      vie::RtdetrDecoder{}.decode_batch(cpu_input, config, vie::DeviceContext{});
      const double cpu_first =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_first_start).count();
      std::vector<double> cpu_wall;
      cpu_wall.reserve(kRuns);
      for (int run = 0; run < kRuns; ++run) {
        const auto started = std::chrono::steady_clock::now();
        vie::RtdetrDecoder{}.decode_batch(cpu_input, config, vie::DeviceContext{});
        cpu_wall.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
      }
      out << "kept_target " << target << " max_detections " << limit << " detections " << expected_kept << "\n";
      out << "cuda_first_wall_ms " << cuda_first_ms << "\n";
      out << "cpu_first_wall_ms " << cpu_first << "\n";
      write_summary(out, "cuda_wall", summarize(wall));
      write_summary(out, "cuda_kernel", summarize(kernel));
      write_summary(out, "cuda_sort", summarize(sort_ms));
      write_summary(out, "cuda_d2h", summarize(d2h));
      write_summary(out, "cpu_wall", summarize(cpu_wall));
    }
  }
}

#if defined(VIE_HAS_ONNXRUNTIME_CUDA)

namespace {

vie::ImageInput load_ppm(const std::string& path, std::vector<std::uint8_t>& pixels) {
  std::ifstream image_file(path, std::ios::binary);
  std::string magic;
  int width = 0;
  int height = 0;
  int max_value = 0;
  image_file >> magic >> width >> height >> max_value;
  image_file.get();
  pixels.resize(static_cast<std::size_t>(width * height * 3));
  image_file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = width;
  image.height = height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  return image;
}

vie::FieldProvenance user_field(const char* evidence) {
  return vie::FieldProvenance{vie::Certainty::User, {evidence}};
}

vie::ModelManifest runtime_manifest() {
  const std::string path = vie_fixture("testdata/onnx/rtdetr-l.onnx");
  const vie::ModelAnalysisReport report = vie::OnnxAnalyzer{}.analyze_file(path);
  const vie::RecognitionResult result = vie::default_recognizer_registry().recognize(report);
  vie::ModelManifest manifest = vie::ManifestGenerator{}.generate(report, result, path);
  manifest.input.color = "rgb";
  manifest.provenance["input.color"] = user_field("caller RGB contract");
  manifest.input.normalization = "scale_0_1";
  manifest.provenance["input.normalization"] = user_field("caller divides by 255");
  manifest.input.interpolation = "bilinear";
  manifest.provenance["input.interpolation"] = user_field("caller bilinear");
  manifest.outputs[0].shape = {1, 300, 84};
  manifest.provenance["outputs.0.shape"] = user_field("runtime profile [1,300,84]");
  manifest.decoder.class_selection = "best_class";
  manifest.provenance["decoder.class_selection"] = user_field("one class per query");
  manifest.nms.confidence_threshold = 0.25f;
  manifest.provenance["nms.confidence_threshold"] = user_field("0.25");
  manifest.nms.max_detections = 300;
  manifest.provenance["nms.max_detections"] = user_field("300");
  return manifest;
}

float box_iou(const vie::BoundingBox& a, const vie::BoundingBox& b) {
  const float width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const float height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (width <= 0.f || height <= 0.f) return 0.f;
  const float intersection = width * height;
  const float area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection;
  return area > 0.f ? intersection / area : 0.f;
}

}  // namespace

TEST(RtdetrCudaE2E, StrictCudaEpDecodesOnDeviceAndRecordsThresholds) {
  VIE_REQUIRE_MODEL("testdata/onnx/rtdetr-l.onnx");
  vie::ModelManifest manifest = runtime_manifest();
  std::vector<std::uint8_t> pixels;
  const vie::ImageInput image = load_ppm(vie_fixture("testdata/images/bus.ppm"), pixels);
  ASSERT_EQ(image.width, 810);
  ASSERT_EQ(image.height, 1080);
  void* device_pixels = nullptr;
  ASSERT_EQ(cudaMalloc(&device_pixels, pixels.size()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_pixels, pixels.data(), pixels.size(), cudaMemcpyHostToDevice), cudaSuccess);
  vie::ImageInput gpu_image = image;
  gpu_image.data = device_pixels;
  gpu_image.device = vie::DeviceType::Cuda;
  gpu_image.device_id = 0;

  vie::ModelConfig cuda_config;
  cuda_config.model_path = vie_fixture("testdata/onnx/rtdetr-l.onnx");
  cuda_config.device = vie::DeviceType::Cuda;
  cuda_config.backend = vie::BackendKind::OnnxRuntimeCuda;
  cuda_config.manifest = manifest;
  vie::InferenceEngine cuda_engine;
  cuda_engine.load(cuda_config);
  const vie::PreprocessResult prepared = cuda_engine.preprocess(gpu_image);
  ASSERT_TRUE(prepared.output.is_cuda());
  vie::BackendRequest request;
  request.inputs.emplace(cuda_engine.manifest().input.name, prepared.output);
  const vie::BackendOutput output = cuda_engine.execute(request);
  const vie::Tensor& raw = output.tensors.at("output0").tensor;
  ASSERT_TRUE(raw.is_cuda());
  ASSERT_EQ(raw.shape(), (std::vector<std::int64_t>{1, 300, 84}));
  ASSERT_EQ(raw.dtype(), vie::DataType::Fp32);

  vie::ModelConfig cpu_config = cuda_config;
  cpu_config.device = vie::DeviceType::Cpu;
  cpu_config.backend = vie::BackendKind::OnnxRuntime;
  vie::InferenceEngine cpu_engine;
  cpu_engine.load(cpu_config);
  const vie::PreprocessResult cpu_prepared = cpu_engine.preprocess(image);
  vie::BackendRequest cpu_request;
  cpu_request.inputs.emplace(cpu_engine.manifest().input.name, cpu_prepared.output);
  const vie::BackendOutput cpu_output = cpu_engine.execute(cpu_request);
  const vie::Tensor& cpu_raw = cpu_output.tensors.at("output0").tensor;
  ASSERT_TRUE(cpu_raw.is_cpu());

  std::ofstream report("/tmp/vie_phase542_e2e.txt");
  vie::DecoderConfig config = vie::decoder_config_from_manifest(manifest);
  const float thresholds[] = {0.10f, 0.25f, 0.50f};
  for (float threshold : thresholds) {
    config.confidence_threshold = threshold;
    vie::DecoderInput cuda_decoded;
    cuda_decoded.outputs.emplace("output0", raw);
    cuda_decoded.geometry = prepared.geometry;
    vie::DecoderInput cpu_decoded;
    cpu_decoded.outputs.emplace("output0", cpu_raw);
    cpu_decoded.geometry = cpu_prepared.geometry;
    const vie::DetectionBatch cuda_batch =
        vie::RtdetrCudaDecoder{}.decode_batch(cuda_decoded, config, cuda_engine.device());
    const vie::DetectionBatch cpu_batch =
        vie::RtdetrDecoder{}.decode_batch(cpu_decoded, config, vie::DeviceContext{});
    const auto& cuda_dets = cuda_batch.results[0].detections;
    const auto& cpu_dets = cpu_batch.results[0].detections;
    report << "threshold " << threshold << " cuda_count " << cuda_dets.size() << " cpu_count " << cpu_dets.size()
           << " d2h_bytes " << cuda_batch.results[0].d2h_bytes << " device "
           << vie::to_string(cuda_batch.results[0].execution_device) << "\n";
    EXPECT_EQ(cuda_batch.results[0].execution_device, vie::DeviceType::Cuda);
    EXPECT_EQ(cuda_batch.results[0].d2h_bytes, cuda_dets.size() * 28u);
    EXPECT_LT(cuda_batch.results[0].d2h_bytes, static_cast<std::size_t>(300 * 84 * sizeof(float)));
    std::vector<char> used(cpu_dets.size(), 0);
    int matched = 0;
    int class_mismatch = 0;
    float min_iou = 1.f;
    float max_corner = 0.f;
    float max_conf = 0.f;
    for (const vie::Detection& detection : cuda_dets) {
      int best = -1;
      float best_iou = -1.f;
      for (std::size_t index = 0; index < cpu_dets.size(); ++index) {
        if (used[index] != 0 || cpu_dets[index].class_id != detection.class_id) continue;
        const float overlap = box_iou(detection.box, cpu_dets[index].box);
        if (overlap > best_iou) {
          best_iou = overlap;
          best = static_cast<int>(index);
        }
      }
      report << "  cuda class " << detection.class_id << " query " << detection.query_index << " conf "
             << detection.confidence << " box " << detection.box.x1 << " " << detection.box.y1 << " "
             << detection.box.x2 << " " << detection.box.y2 << "\n";
      if (best < 0) continue;
      used[static_cast<std::size_t>(best)] = 1;
      ++matched;
      const vie::Detection& other = cpu_dets[static_cast<std::size_t>(best)];
      if (other.class_id != detection.class_id) ++class_mismatch;
      min_iou = std::min(min_iou, best_iou);
      max_corner = std::max(max_corner, std::abs(detection.box.x1 - other.box.x1));
      max_corner = std::max(max_corner, std::abs(detection.box.y1 - other.box.y1));
      max_corner = std::max(max_corner, std::abs(detection.box.x2 - other.box.x2));
      max_corner = std::max(max_corner, std::abs(detection.box.y2 - other.box.y2));
      max_conf = std::max(max_conf, std::abs(detection.confidence - other.confidence));
    }
    report << "  matched " << matched << " class_mismatch " << class_mismatch << " min_iou "
           << (matched == 0 ? 0.f : min_iou) << " max_corner " << max_corner << " max_conf " << max_conf << "\n";
    if (threshold >= 0.25f) {
      EXPECT_EQ(cuda_dets.size(), cpu_dets.size());
      EXPECT_EQ(matched, static_cast<int>(cuda_dets.size()));
      EXPECT_EQ(class_mismatch, 0);
      EXPECT_GT(min_iou, 0.99f);
      EXPECT_LT(max_corner, 1.f);
      EXPECT_LT(max_conf, 1e-2f);
    }
  }

  vie::DetectionInferTiming first;
  const auto infer_first = cuda_engine.infer(gpu_image, &first);
  report << "engine_first_preprocess " << first.preprocess_ms << " execute " << first.execute_ms << " decode "
         << first.decode_ms << " total " << first.total_ms << " count " << infer_first.results[0].detections.size()
         << " d2h_bytes " << infer_first.results[0].d2h_bytes << "\n";
  EXPECT_EQ(infer_first.results[0].execution_device, vie::DeviceType::Cuda);
  EXPECT_EQ(infer_first.results[0].detections.size(), 9u);
  constexpr int kRuns = 100;
  std::vector<double> preprocess;
  std::vector<double> execute;
  std::vector<double> decode;
  std::vector<double> total;
  std::vector<double> kernel;
  std::vector<double> sort_ms;
  std::vector<double> d2h;
  for (int run = 0; run < kRuns; ++run) {
    vie::DetectionInferTiming timing;
    const vie::DetectionBatch batch = cuda_engine.infer(gpu_image, &timing);
    EXPECT_EQ(batch.results[0].detections.size(), infer_first.results[0].detections.size());
    preprocess.push_back(timing.preprocess_ms);
    execute.push_back(timing.execute_ms);
    decode.push_back(timing.decode_ms);
    total.push_back(timing.total_ms);
    kernel.push_back(batch.results[0].gpu_execute_ms);
    sort_ms.push_back(batch.results[0].sort_ms);
    d2h.push_back(batch.results[0].d2h_ms);
  }
  write_summary(report, "engine_preprocess", summarize(preprocess));
  write_summary(report, "engine_execute", summarize(execute));
  write_summary(report, "engine_decode", summarize(decode));
  write_summary(report, "engine_total", summarize(total));
  write_summary(report, "engine_kernel", summarize(kernel));
  write_summary(report, "engine_sort", summarize(sort_ms));
  write_summary(report, "engine_d2h", summarize(d2h));
  ASSERT_EQ(cudaFree(device_pixels), cudaSuccess);
}

#endif
