// Phase 13: Classification CUDA Top-K decoder. Only compiled when
// VIE_ENABLE_CUDA is on; the CPU-only builds never see these sources.

#include "test_util.hpp"

#include "vie/decoder/classification.hpp"
#include "vie/decoder/classification_cuda_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using vie::ClassificationBatch;
using vie::ClassificationDecodeRequest;
using vie::DataType;
using vie::DeviceContext;
using vie::DeviceType;

const DeviceContext kCpu;
const DeviceContext kCuda(DeviceType::Cuda, 0);

std::vector<std::int64_t> logits_shape(int classes, int axis) {
  return axis == 0 ? std::vector<std::int64_t>{classes, 1} : std::vector<std::int64_t>{1, classes};
}

ClassificationDecodeRequest request_for(int classes, int top_k, int axis, bool probabilities) {
  ClassificationDecodeRequest request;
  request.output_name = "logits";
  request.class_axis = axis;
  request.num_classes = classes;
  request.top_k = top_k;
  request.score_type = probabilities ? "probabilities" : "logits";
  return request;
}

vie::Tensor host_tensor(const std::vector<float>& data, int classes, int axis) {
  return vie::Tensor::from_host(kCpu, DataType::Fp32, logits_shape(classes, axis), data.data(),
                                data.size() * sizeof(float));
}

ClassificationBatch decode_cpu(const std::vector<float>& data, int classes, int top_k, int axis,
                               bool probabilities) {
  vie::DecoderInput input;
  input.outputs.emplace("logits", host_tensor(data, classes, axis));
  const ClassificationDecodeRequest request = request_for(classes, top_k, axis, probabilities);
  if (probabilities) {
    return vie::ClassificationProbabilitiesDecoder{}.decode_classification(input, request, kCpu);
  }
  return vie::ClassificationDecoder{}.decode_classification(input, request, kCpu);
}

ClassificationBatch decode_cuda(const std::vector<float>& data, int classes, int top_k, int axis,
                                bool probabilities) {
  vie::Tensor device = host_tensor(data, classes, axis).to(kCuda);
  vie::DecoderInput input;
  input.outputs.emplace("logits", device);
  return vie::ClassificationCudaDecoder{probabilities}.decode_classification(
      input, request_for(classes, top_k, axis, probabilities), kCuda);
}

// Deterministic logits with both signs and a few repeated values (the sinusoid
// is not injective on this sample range, so ties occur).
std::vector<float> pattern_logits(int classes) {
  std::vector<float> data(static_cast<std::size_t>(classes));
  for (int index = 0; index < classes; ++index) {
    data[static_cast<std::size_t>(index)] =
        static_cast<float>(std::sin(static_cast<double>(index) * 0.37) * 9.5);
  }
  return data;
}

void expect_same(const ClassificationBatch& cpu, const ClassificationBatch& cuda,
                 std::size_t classes, std::size_t top_k) {
  ASSERT_EQ(cpu.results.size(), 1u);
  ASSERT_EQ(cuda.results.size(), 1u);
  const vie::ClassificationResult& expected = cpu.results.front();
  const vie::ClassificationResult& actual = cuda.results.front();
  ASSERT_EQ(expected.labels.size(), actual.labels.size());
  EXPECT_EQ(actual.labels.size(), top_k);
  for (std::size_t rank = 0; rank < expected.labels.size(); ++rank) {
    EXPECT_EQ(expected.labels[rank].class_id, actual.labels[rank].class_id) << "rank " << rank;
    EXPECT_FLOAT_EQ(expected.labels[rank].score, actual.labels[rank].score) << "rank " << rank;
  }
  EXPECT_EQ(actual.execution_device, DeviceType::Cuda);
  EXPECT_EQ(actual.d2h_bytes, top_k * 8u);
  EXPECT_EQ(expected.d2h_bytes, 0u);
  EXPECT_EQ(actual.output_bytes, classes * sizeof(float));
  EXPECT_EQ(actual.protocol, expected.protocol);
  EXPECT_EQ(actual.score_type, expected.score_type);
  EXPECT_EQ(actual.num_classes, expected.num_classes);
  EXPECT_EQ(actual.top_k, expected.top_k);
}

std::string decode_error(const std::vector<float>& data, int classes, int top_k, bool probabilities,
                         bool cuda) {
  try {
    if (cuda) {
      decode_cuda(data, classes, top_k, 1, probabilities);
    } else {
      decode_cpu(data, classes, top_k, 1, probabilities);
    }
  } catch (const vie::VieError& error) {
    EXPECT_EQ(error.code(), vie::ErrorCode::InvalidArgument);
    return error.what();
  }
  ADD_FAILURE() << "expected VieError";
  return {};
}

}  // namespace

TEST(ClassificationCudaDecoder, MatchesCpuAcrossClassCountsAndTopK) {
  for (const int classes : {1, 2, 7, 64, 1000}) {
    const std::vector<float> data = pattern_logits(classes);
    for (const int top_k : {1, std::min(5, classes), classes}) {
      const ClassificationBatch cpu = decode_cpu(data, classes, top_k, 1, false);
      const ClassificationBatch cuda = decode_cuda(data, classes, top_k, 1, false);
      expect_same(cpu, cuda, static_cast<std::size_t>(classes), static_cast<std::size_t>(top_k));
    }
  }
}

TEST(ClassificationCudaDecoder, MatchesCpuOnNegativeLogitsAndClassAxisZero) {
  constexpr int kClasses = 13;
  std::vector<float> data = pattern_logits(kClasses);
  for (int index = 0; index < kClasses; ++index) {
    data[static_cast<std::size_t>(index)] -= 40.f;
  }
  const ClassificationBatch cpu = decode_cpu(data, kClasses, 5, 1, false);
  const ClassificationBatch cuda = decode_cuda(data, kClasses, 5, 1, false);
  expect_same(cpu, cuda, kClasses, 5);
  for (const vie::Classification& label : cuda.results[0].labels) {
    EXPECT_LT(label.score, 0.f);
  }

  const ClassificationBatch axis_cpu = decode_cpu(data, kClasses, 4, 0, false);
  const ClassificationBatch axis_cuda = decode_cuda(data, kClasses, 4, 0, false);
  expect_same(axis_cpu, axis_cuda, kClasses, 4);
}

TEST(ClassificationCudaDecoder, EqualScoresKeepTheSmallerClassId) {
  constexpr int kClasses = 32;
  std::vector<float> data(kClasses, 3.5f);
  const ClassificationBatch cpu = decode_cpu(data, kClasses, 6, 1, false);
  const ClassificationBatch cuda = decode_cuda(data, kClasses, 6, 1, false);
  expect_same(cpu, cuda, kClasses, 6);
  for (int rank = 0; rank < 6; ++rank) {
    EXPECT_EQ(cuda.results[0].labels[static_cast<std::size_t>(rank)].class_id, rank);
  }
}

TEST(ClassificationCudaDecoder, RejectsNonFiniteLogitsWithTheSameFirstClass) {
  constexpr int kClasses = 50;
  struct Case {
    int poison_class;
    float value;
    const char* detail;
  };
  const Case cases[] = {
      {3, std::numeric_limits<float>::quiet_NaN(), "logit is NaN"},
      {17, std::numeric_limits<float>::infinity(), "logit is infinite"},
      {1, -std::numeric_limits<float>::infinity(), "logit is infinite"},
  };
  for (const Case& item : cases) {
    std::vector<float> data = pattern_logits(kClasses);
    data[static_cast<std::size_t>(item.poison_class)] = item.value;
    const std::string cpu = decode_error(data, kClasses, 5, false, false);
    const std::string cuda = decode_error(data, kClasses, 5, false, true);
    EXPECT_EQ(cpu, cuda);
    EXPECT_NE(cpu.find("class_id " + std::to_string(item.poison_class)), std::string::npos) << cpu;
    EXPECT_NE(cpu.find(item.detail), std::string::npos) << cpu;
  }

  // A later NaN must not mask an earlier +Inf: both paths report the lowest
  // offending class id, not the first thread to finish.
  std::vector<float> data = pattern_logits(kClasses);
  data[41] = std::numeric_limits<float>::quiet_NaN();
  data[7] = std::numeric_limits<float>::infinity();
  const std::string cpu = decode_error(data, kClasses, 5, false, false);
  const std::string cuda = decode_error(data, kClasses, 5, false, true);
  EXPECT_EQ(cpu, cuda);
  EXPECT_NE(cpu.find("class_id 7"), std::string::npos) << cpu;
}

TEST(ClassificationCudaDecoder, ProbabilityProtocolMatchesCpuContract) {
  const int classes = 6;
  const std::vector<float> probabilities{0.05f, 0.1f, 0.4f, 0.2f, 0.1f, 0.15f};
  const ClassificationBatch cpu = decode_cpu(probabilities, classes, 3, 1, true);
  const ClassificationBatch cuda = decode_cuda(probabilities, classes, 3, 1, true);
  expect_same(cpu, cuda, classes, 3);
  EXPECT_FLOAT_EQ(cuda.results[0].labels[0].score, 0.4f);

  struct Case {
    std::vector<float> data;
    const char* detail;
  };
  const Case cases[] = {
      {{0.5f, 0.4f, 0.2f, 0.f, 0.f, 0.f}, "probability sum"},
      {{0.5f, 0.5f, 1.5f, 0.f, 0.f, 0.f}, "probability is above 1"},
      {{0.5f, 0.5f, -0.5f, 0.f, 0.f, 0.f}, "probability is below 0"},
      {{0.5f, 0.5f, std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f, 0.f}, "probability is NaN"},
      {{0.5f, 0.5f, std::numeric_limits<float>::infinity(), 0.f, 0.f, 0.f},
       "probability is positive infinity"},
      {{0.5f, 0.5f, -std::numeric_limits<float>::infinity(), 0.f, 0.f, 0.f},
       "probability is negative infinity"},
  };
  for (const Case& item : cases) {
    const std::string cpu_message = decode_error(item.data, classes, 3, true, false);
    const std::string cuda_message = decode_error(item.data, classes, 3, true, true);
    EXPECT_NE(cpu_message.find(item.detail), std::string::npos) << cpu_message;
    EXPECT_NE(cuda_message.find(item.detail), std::string::npos) << cuda_message;
    // The sum diagnostic prints the accumulated value; the wording is shared
    // and only the digits may differ in the last place.
    if (std::string(item.detail) == "probability sum") {
      EXPECT_NE(cuda_message.find("differs from 1 by"), std::string::npos) << cuda_message;
      EXPECT_NE(cuda_message.find("outside tolerance"), std::string::npos) << cuda_message;
    } else {
      EXPECT_EQ(cpu_message, cuda_message);
    }
  }
}

TEST(ClassificationCudaDecoder, RejectsInvalidTensorAndRequestContracts) {
  constexpr int kClasses = 8;
  const std::vector<float> data = pattern_logits(kClasses);
  const ClassificationDecodeRequest request = request_for(kClasses, 3, 1, false);

  // CPU tensor on the CUDA decoder.
  vie::DecoderInput cpu_tensor;
  cpu_tensor.outputs.emplace("logits", host_tensor(data, kClasses, 1));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::ClassificationCudaDecoder{false}.decode_classification(cpu_tensor, request, kCuda));
  // CPU decoder with a CUDA tensor (unchanged CPU contract).
  vie::DecoderInput cuda_tensor;
  cuda_tensor.outputs.emplace("logits", host_tensor(data, kClasses, 1).to(kCuda));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::ClassificationDecoder{}.decode_classification(cuda_tensor, request, kCpu));
  // CUDA serial decoder refuses a CPU device request.
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::ClassificationCudaDecoder{false}.decode_classification(cuda_tensor, request, kCpu));

  // Device id mismatch is rejected before any device work.
  VIE_EXPECT_ERROR(
      vie::ErrorCode::InvalidArgument,
      vie::ClassificationCudaDecoder{false}.decode_classification(
          cuda_tensor, request, DeviceContext(DeviceType::Cuda, 1)));

  // Wrong dtype.
  vie::DecoderInput int_tensor;
  int_tensor.outputs.emplace("logits", vie::Tensor::zeros(kCuda, DataType::Int32, {1, kClasses}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::ClassificationCudaDecoder{false}.decode_classification(int_tensor, request, kCuda));

  // Wrong rank.
  vie::DecoderInput rank3;
  rank3.outputs.emplace("logits", vie::Tensor::zeros(kCuda, DataType::Fp32, {1, 1, kClasses}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::ClassificationCudaDecoder{false}.decode_classification(rank3, request, kCuda));

  // Batch dimension must be 1.
  vie::DecoderInput batch2;
  batch2.outputs.emplace("logits", vie::Tensor::zeros(kCuda, DataType::Fp32, {2, kClasses}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::ClassificationCudaDecoder{false}.decode_classification(batch2, request, kCuda));

  // Class axis length must match num_classes.
  vie::DecoderInput wrong_classes;
  wrong_classes.outputs.emplace("logits", vie::Tensor::zeros(kCuda, DataType::Fp32, {1, kClasses + 1}));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::ClassificationCudaDecoder{false}.decode_classification(wrong_classes, request, kCuda));

  // Non-contiguous stays refused instead of being silently packed.
  vie::Tensor wide = vie::Tensor::zeros(kCuda, DataType::Fp32, {1, 2 * kClasses});
  vie::Tensor strided = wide.view({1, kClasses}, {kClasses, 2}, 0);
  ASSERT_FALSE(strided.contiguous());
  vie::DecoderInput strided_input;
  strided_input.outputs.emplace("logits", strided);
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::ClassificationCudaDecoder{false}.decode_classification(strided_input, request, kCuda));

  // Request validation matches the CPU decoder.
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::ClassificationCudaDecoder{false}.decode_classification(
                       cuda_tensor, request_for(kClasses, kClasses + 1, 1, false), kCuda));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::ClassificationCudaDecoder{false}.decode_classification(
                       cuda_tensor, request_for(kClasses, 3, 2, false), kCuda));
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   vie::ClassificationCudaDecoder{false}.decode_classification(
                       cuda_tensor, request_for(kClasses, 3, 1, true), kCuda));
}

TEST(ClassificationCudaDecoder, DispatcherRoutesByDevice) {
  constexpr int kClasses = 10;
  const std::vector<float> data = pattern_logits(kClasses);
  vie::ClassificationDispatchDecoder logits_dispatch{false};
  vie::ClassificationDispatchDecoder probabilities_dispatch{true};
  EXPECT_EQ(logits_dispatch.protocol(), "classification_logits_raw");
  EXPECT_EQ(probabilities_dispatch.protocol(), "classification_probabilities_raw");

  vie::DecoderInput cpu_input;
  cpu_input.outputs.emplace("logits", host_tensor(data, kClasses, 1));
  const ClassificationBatch cpu_batch = logits_dispatch.decode_classification(
      cpu_input, request_for(kClasses, 3, 1, false), kCpu);
  EXPECT_EQ(cpu_batch.results[0].execution_device, DeviceType::Cpu);
  EXPECT_EQ(cpu_batch.results[0].d2h_bytes, 0u);

  vie::DecoderInput cuda_input;
  cuda_input.outputs.emplace("logits", host_tensor(data, kClasses, 1).to(kCuda));
  const ClassificationBatch cuda_batch = logits_dispatch.decode_classification(
      cuda_input, request_for(kClasses, 3, 1, false), kCuda);
  EXPECT_EQ(cuda_batch.results[0].execution_device, DeviceType::Cuda);
  EXPECT_EQ(cuda_batch.results[0].d2h_bytes, 3u * 8u);
  expect_same(cpu_batch, cuda_batch, kClasses, 3);

  const std::vector<float> probabilities(kClasses, 0.1f);
  vie::DecoderInput probability_input;
  probability_input.outputs.emplace("logits", host_tensor(probabilities, kClasses, 1).to(kCuda));
  const ClassificationBatch probability_batch = probabilities_dispatch.decode_classification(
      probability_input, request_for(kClasses, 2, 1, true), kCuda);
  EXPECT_EQ(probability_batch.results[0].protocol, "classification_probabilities_raw");
  EXPECT_EQ(probability_batch.results[0].execution_device, DeviceType::Cuda);
  EXPECT_FLOAT_EQ(probability_batch.results[0].labels[0].score, 0.1f);
}

TEST(ClassificationCudaDecoder, RecordsDecoderTimingAndPayload) {
  constexpr int kClasses = 1000;
  constexpr int kTopK = 5;
  const std::vector<float> data = pattern_logits(kClasses);
  vie::Tensor device = host_tensor(data, kClasses, 1).to(kCuda);
  vie::DecoderInput cuda_input;
  cuda_input.outputs.emplace("logits", device);
  const ClassificationDecodeRequest request = request_for(kClasses, kTopK, 1, false);
  vie::DecoderInput cpu_input;
  cpu_input.outputs.emplace("logits", host_tensor(data, kClasses, 1));
  const ClassificationDecodeRequest cpu_request = request_for(kClasses, kTopK, 1, false);

  constexpr int kWarmup = 3;
  constexpr int kSamples = 20;
  for (int index = 0; index < kWarmup; ++index) {
    vie::ClassificationCudaDecoder{false}.decode_classification(cuda_input, request, kCuda);
    vie::ClassificationDecoder{}.decode_classification(cpu_input, cpu_request, kCpu);
  }
  double cpu_validate = 0;
  double cpu_topk = 0;
  double cuda_validate = 0;
  double cuda_topk = 0;
  double cuda_d2h = 0;
  std::size_t d2h_bytes = 0;
  for (int index = 0; index < kSamples; ++index) {
    const ClassificationBatch cpu =
        vie::ClassificationDecoder{}.decode_classification(cpu_input, cpu_request, kCpu);
    const ClassificationBatch cuda =
        vie::ClassificationCudaDecoder{false}.decode_classification(cuda_input, request, kCuda);
    cpu_validate += cpu.results[0].validate_ms;
    cpu_topk += cpu.results[0].topk_ms;
    cuda_validate += cuda.results[0].validate_ms;
    cuda_topk += cuda.results[0].topk_ms;
    cuda_d2h += cuda.results[0].d2h_ms;
    d2h_bytes = cuda.results[0].d2h_bytes;
    EXPECT_EQ(cpu.results[0].labels[0].class_id, cuda.results[0].labels[0].class_id);
  }
  EXPECT_EQ(d2h_bytes, kTopK * 8u);
  EXPECT_GE(cuda_validate, 0.0);
  EXPECT_GE(cuda_topk, 0.0);
  EXPECT_GE(cuda_d2h, 0.0);

  std::ofstream record("/tmp/vie_phase13_classification_decoder.txt");
  record << "classes " << kClasses << " top_k " << kTopK << " warmup " << kWarmup << " samples "
         << kSamples << "\n";
  record << "cpu validate_ms_mean " << cpu_validate / kSamples << " topk_ms_mean "
         << cpu_topk / kSamples << " d2h_bytes 0\n";
  record << "cuda validate_ms_mean " << cuda_validate / kSamples << " topk_ms_mean "
         << cuda_topk / kSamples << " d2h_ms_mean " << cuda_d2h / kSamples << " d2h_bytes "
         << d2h_bytes << " full_logits_bytes " << kClasses * sizeof(float) << "\n";
}