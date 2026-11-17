#include "seg_scalar_reference.hpp"
#include "test_util.hpp"

#include "vie/decoder/yolov8_seg_decoder.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/preprocess/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <tuple>
#include <vector>

#if defined(VIE_HAS_ONNXRUNTIME)
#include "vie/backend/onnxruntime_backend.hpp"
#endif
#if defined(VIE_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

void put(std::vector<float>& data, int predictions, int channel, int prediction, float value) {
  data[static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) +
       static_cast<std::size_t>(prediction)] = value;
}

vie::GeometryTransform identity(int width, int height) {
  vie::GeometryTransform geometry;
  geometry.original_width = width;
  geometry.original_height = height;
  geometry.resized_width = width;
  geometry.resized_height = height;
  geometry.output_width = width;
  geometry.output_height = height;
  geometry.scale_x = 1;
  geometry.scale_y = 1;
  geometry.uniform_scale = 1;
  return geometry;
}

vie::DecoderConfig seg_config(int classes, int mask_channels) {
  vie::DecoderConfig config;
  config.output_name = "output0";
  config.prototype_name = "output1";
  config.channel_axis = 1;
  config.box_channels = 4;
  config.num_classes = classes;
  config.mask_channels = mask_channels;
  config.box_decode = vie::BoxDecode::DecodedCxcywh;
  config.scores_are_probabilities = true;
  config.class_selection = vie::ClassSelection::BestClass;
  config.confidence_threshold = 0.5f;
  config.nms_enabled = true;
  config.class_aware_nms = true;
  config.iou_threshold = 0.5f;
  config.max_detections = 10;
  config.pre_nms_top_k = 100;
  config.prototype_channel_axis = 1;
  config.retina_masks = false;
  config.binary_mask_gt_zero = true;
  config.max_mask_bytes = 1 << 20;
  return config;
}

vie::SegmentationBatch run_seg(const std::vector<float>& predictions, const std::vector<std::int64_t>& prediction_shape,
                              const std::vector<float>& proto, const std::vector<std::int64_t>& proto_shape,
                              const vie::DecoderConfig& config, const vie::GeometryTransform& geometry,
                              const vie::DeviceContext& device = vie::DeviceContext{}) {
  vie::DecoderInput input;
  input.geometry = geometry;
  input.outputs.emplace("output0", vie::Tensor::from_host(device, vie::DataType::Fp32, prediction_shape,
                                                         predictions.data(), predictions.size() * sizeof(float)));
  input.outputs.emplace("output1", vie::Tensor::from_host(device, vie::DataType::Fp32, proto_shape, proto.data(),
                                                         proto.size() * sizeof(float)));
  return vie::YoloV8SegmentationDecoder{}.decode_segmentation(input, config, device);
}

#if defined(VIE_HAS_ONNXRUNTIME)
struct IndependentInstance {
  vie::BoundingBox model_box;
  vie::BoundingBox original_box;
  float score = 0.f;
  int class_id = -1;
  int prediction_index = -1;
  std::vector<float> coefficients;
};

std::vector<IndependentInstance> independent_kept(const float* data, int predictions, int classes, int mask_channels,
                                                 const vie::DecoderConfig& config,
                                                 const vie::GeometryTransform& geometry) {
  struct Item {
    vie::BoundingBox model_box;
    vie::BoundingBox original_box;
    float score = 0.f;
    int class_id = -1;
    int sort_index = 0;
    int prediction_index = -1;
  };
  std::vector<Item> candidates;
  for (int prediction = 0; prediction < predictions; ++prediction) {
    const auto at = [&](int channel) {
      return data[static_cast<std::size_t>(channel) * static_cast<std::size_t>(predictions) +
                  static_cast<std::size_t>(prediction)];
    };
    const float width = at(2);
    const float height = at(3);
    if (width <= 0.f || height <= 0.f) continue;
    vie::BoundingBox model;
    vie::cxcywh_to_xyxy(at(0), at(1), width, height, model.x1, model.y1, model.x2, model.y2);
    int best_class = 0;
    float best_score = -1.f;
    for (int cls = 0; cls < classes; ++cls) {
      const float score = at(4 + cls);
      if (score > best_score || (score == best_score && cls < best_class)) {
        best_score = score;
        best_class = cls;
      }
    }
    if (best_score < *config.confidence_threshold) continue;
    candidates.push_back(Item{model, model, best_score, best_class, prediction, prediction});
  }
  std::sort(candidates.begin(), candidates.end(), [](const Item& a, const Item& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.sort_index < b.sort_index;
  });
  if (static_cast<int>(candidates.size()) > *config.pre_nms_top_k) {
    candidates.resize(static_cast<std::size_t>(*config.pre_nms_top_k));
  }
  std::vector<Item> mapped;
  for (Item item : candidates) {
    if (!vie::map_box_to_original(item.original_box, geometry)) continue;
    mapped.push_back(item);
  }
  std::vector<vie::NmsCandidate> nms_input;
  for (const Item& item : mapped) {
    nms_input.push_back(vie::NmsCandidate{item.original_box, item.score, item.class_id, item.sort_index});
  }
  const std::vector<int> kept =
      vie::greedy_nms(nms_input, *config.iou_threshold, *config.class_aware_nms, *config.max_detections);
  std::vector<IndependentInstance> result;
  for (const int index : kept) {
    const Item& item = mapped[static_cast<std::size_t>(index)];
    IndependentInstance instance;
    instance.model_box = item.model_box;
    instance.original_box = item.original_box;
    instance.score = item.score;
    instance.class_id = item.class_id;
    instance.prediction_index = item.prediction_index;
    instance.coefficients.resize(static_cast<std::size_t>(mask_channels));
    for (int channel = 0; channel < mask_channels; ++channel) {
      instance.coefficients[static_cast<std::size_t>(channel)] =
          data[static_cast<std::size_t>(4 + classes + channel) * static_cast<std::size_t>(predictions) +
               static_cast<std::size_t>(item.prediction_index)];
    }
    result.push_back(std::move(instance));
  }
  return result;
}
#endif

void expect_binary(const vie::Mask& mask) {
  EXPECT_EQ(mask.kind(), vie::MaskKind::Binary);
  EXPECT_EQ(mask.dtype(), vie::DataType::Uint8);
  const std::uint8_t* pixels = mask.binary_data();
  for (int i = 0; i < mask.width() * mask.height(); ++i) {
    EXPECT_TRUE(pixels[i] == 0 || pixels[i] == 1);
  }
}

}  // namespace

TEST(YoloV8SegCpu, ReconstructsHandComputedMasksAndKeepsPredictionIndex) {
  constexpr int kPredictions = 2;
  constexpr int kChannels = 4 + 2 + 2;
  std::vector<float> predictions(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  put(predictions, kPredictions, 0, 0, 1.f);
  put(predictions, kPredictions, 1, 0, 1.f);
  put(predictions, kPredictions, 2, 0, 2.f);
  put(predictions, kPredictions, 3, 0, 2.f);
  put(predictions, kPredictions, 4, 0, 0.9f);
  put(predictions, kPredictions, 6, 0, 2.f);
  put(predictions, kPredictions, 7, 0, -0.5f);
  put(predictions, kPredictions, 0, 1, 1.f);
  put(predictions, kPredictions, 1, 1, 1.f);
  put(predictions, kPredictions, 2, 1, 2.f);
  put(predictions, kPredictions, 3, 1, 2.f);
  put(predictions, kPredictions, 4, 1, 0.9f);
  put(predictions, kPredictions, 6, 1, -4.f);
  const std::vector<float> proto = {1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, -4.f};
  vie::DecoderConfig config = seg_config(2, 2);
  config.iou_threshold = 0.1f;
  const vie::SegmentationBatch batch = run_seg(predictions, {1, kChannels, kPredictions}, proto, {1, 2, 2, 2}, config,
                                               identity(2, 2));
  ASSERT_EQ(batch.instances.size(), 1u);
  EXPECT_EQ(batch.instances[0].detection.class_id, 0);
  EXPECT_FLOAT_EQ(batch.instances[0].detection.confidence, 0.9f);
  const vie::Mask& mask = batch.instances[0].mask;
  expect_binary(mask);
  ASSERT_EQ(mask.width(), 2);
  ASSERT_EQ(mask.height(), 2);
  // M = 2*P0 - 0.5*P1 = [2, -1, 0, 2]. The lower-index duplicate is suppressed.
  EXPECT_EQ(mask.binary_data()[0], 1);
  EXPECT_EQ(mask.binary_data()[1], 0);
  EXPECT_EQ(mask.binary_data()[2], 0);
  EXPECT_EQ(mask.binary_data()[3], 1);
}

TEST(YoloV8SegReference, IndependentScalarMatchesDecoderOnCropResizeAndLetterbox) {
  std::vector<float> predictions(7, 0.f);
  predictions[0] = 1.f;
  predictions[1] = 0.5f;
  predictions[2] = 1.f;
  predictions[3] = 1.f;
  predictions[4] = 0.95f;
  predictions[6] = 1.f;
  const std::vector<float> proto = {1.f, -2.f, 1.f, 0.5f, -0.25f, 3.f};
  vie::BoundingBox model;
  vie::cxcywh_to_xyxy(1.f, 0.5f, 1.f, 1.f, model.x1, model.y1, model.x2, model.y2);
  vie::GeometryTransform geometry = identity(3, 2);
  vie::BoundingBox original = model;
  ASSERT_TRUE(vie::map_box_to_original(original, geometry));
  const std::vector<std::uint8_t> expected =
      vie_ref::binary_mask(proto.data(), 1, 2, 3, {1.f}, model, original, geometry, false);
  const vie::SegmentationBatch decoded =
      run_seg(predictions, {1, 7, 1}, proto, {1, 1, 2, 3}, seg_config(2, 1), geometry);
  ASSERT_EQ(decoded.instances.size(), 1u);
  ASSERT_EQ(expected.size(), static_cast<std::size_t>(decoded.instances[0].mask.width() *
                                                      decoded.instances[0].mask.height()));
  EXPECT_EQ(0, std::memcmp(expected.data(), decoded.instances[0].mask.binary_data(), expected.size()));

  vie::GeometryTransform letterbox;
  letterbox.original_width = 3;
  letterbox.original_height = 2;
  letterbox.output_width = 5;
  letterbox.output_height = 4;
  letterbox.resized_width = 3;
  letterbox.resized_height = 2;
  letterbox.pad_left = 1;
  letterbox.pad_top = 1;
  letterbox.pad_right = 1;
  letterbox.pad_bottom = 1;
  letterbox.scale_x = 1;
  letterbox.scale_y = 1;
  letterbox.uniform_scale = 0.6;
  std::vector<float> retina_proto(40, 0.f);
  for (int i = 0; i < 40; ++i) retina_proto[static_cast<std::size_t>(i)] = (i % 5) - 2.f;
  vie::BoundingBox retina_model;
  vie::cxcywh_to_xyxy(2.5f, 2.f, 3.f, 2.f, retina_model.x1, retina_model.y1, retina_model.x2, retina_model.y2);
  vie::BoundingBox retina_original = retina_model;
  ASSERT_TRUE(vie::map_box_to_original(retina_original, letterbox));
  vie::DecoderConfig retina = seg_config(1, 2);
  retina.retina_masks = true;
  const std::vector<float> coeffs = {0.5f, -3.f};
  const std::vector<std::uint8_t> retina_expected = vie_ref::binary_mask(
      retina_proto.data(), 2, 4, 5, coeffs, retina_model, retina_original, letterbox, true);
  std::vector<float> packed(4 + 1 + 2, 0.f);
  packed[0] = 2.5f;
  packed[1] = 2.f;
  packed[2] = 3.f;
  packed[3] = 2.f;
  packed[4] = 0.8f;
  packed[5] = 0.5f;
  packed[6] = -3.f;
  const vie::SegmentationBatch retina_decoded =
      run_seg(packed, {1, 7, 1}, retina_proto, {1, 2, 4, 5}, retina, letterbox);
  ASSERT_EQ(retina_decoded.instances.size(), 1u);
  EXPECT_EQ(retina_decoded.instances[0].mask.width(), 3);
  EXPECT_EQ(retina_decoded.instances[0].mask.height(), 2);
  EXPECT_EQ(0, std::memcmp(retina_expected.data(), retina_decoded.instances[0].mask.binary_data(),
                           retina_expected.size()));
  EXPECT_NEAR(retina_decoded.instances[0].detection.box.x1, retina_original.x1, 0.f);
  EXPECT_NEAR(retina_decoded.instances[0].detection.box.y1, retina_original.y1, 0.f);
}

TEST(YoloV8SegCpu, NegativeCoefficientIsNotSigmoided) {
  std::vector<float> predictions(7, 0.f);
  predictions[0] = 1.f;
  predictions[1] = 1.f;
  predictions[2] = 2.f;
  predictions[3] = 2.f;
  predictions[4] = 0.8f;
  predictions[6] = -0.1f;
  const std::vector<float> proto = {1.f, 1.f, 1.f, 1.f};
  const vie::SegmentationBatch batch =
      run_seg(predictions, {1, 7, 1}, proto, {1, 1, 2, 2}, seg_config(2, 1), identity(2, 2));
  ASSERT_EQ(batch.instances.size(), 1u);
  expect_binary(batch.instances[0].mask);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(batch.instances[0].mask.binary_data()[i], 0);
}

TEST(YoloV8SegCpu, ClassAwareAgnosticTiesAndEmpty) {
  constexpr int kPredictions = 2;
  constexpr int kChannels = 7;
  std::vector<float> predictions(static_cast<std::size_t>(kChannels * kPredictions), 0.f);
  for (int prediction = 0; prediction < 2; ++prediction) {
    put(predictions, kPredictions, 0, prediction, prediction == 0 ? 1.f : 5.f);
    put(predictions, kPredictions, 1, prediction, 1.f);
    put(predictions, kPredictions, 2, prediction, 2.f);
    put(predictions, kPredictions, 3, prediction, 2.f);
  }
  put(predictions, kPredictions, 4, 0, 0.7f);
  put(predictions, kPredictions, 5, 1, 0.7f);
  put(predictions, kPredictions, 6, 0, 1.f);
  put(predictions, kPredictions, 6, 1, -3.f);
  const std::vector<float> proto = {1.f, 1.f, 1.f, 1.f};
  vie::DecoderConfig aware = seg_config(2, 1);
  aware.iou_threshold = 0.1f;
  const vie::SegmentationBatch separated =
      run_seg(predictions, {1, kChannels, kPredictions}, proto, {1, 1, 2, 2}, aware, identity(8, 2));
  ASSERT_EQ(separated.instances.size(), 2u);
  EXPECT_EQ(separated.instances[0].detection.class_id, 0);
  EXPECT_EQ(separated.instances[1].detection.class_id, 1);
  EXPECT_EQ(separated.instances[0].mask.binary_data()[0], 1);
  EXPECT_EQ(separated.instances[1].mask.binary_data()[0], 0);

  put(predictions, kPredictions, 0, 1, 1.f);
  vie::DecoderConfig agnostic = aware;
  agnostic.class_aware_nms = false;
  const vie::SegmentationBatch collapsed =
      run_seg(predictions, {1, kChannels, kPredictions}, proto, {1, 1, 2, 2}, agnostic, identity(2, 2));
  ASSERT_EQ(collapsed.instances.size(), 1u);
  EXPECT_EQ(collapsed.instances[0].detection.class_id, 0);
  EXPECT_EQ(collapsed.instances[0].mask.binary_data()[0], 1);

  put(predictions, kPredictions, 4, 0, 0.1f);
  put(predictions, kPredictions, 5, 1, 0.1f);
  const vie::SegmentationBatch empty =
      run_seg(predictions, {1, kChannels, kPredictions}, proto, {1, 1, 2, 2}, aware, identity(2, 2));
  EXPECT_TRUE(empty.instances.empty());
  EXPECT_EQ(empty.original_width, 2);
  EXPECT_EQ(empty.protocol, "yolov8_seg_raw");
}

TEST(YoloV8SegCpu, FloatCropBoundaryAndRetinaModes) {
  std::vector<float> predictions(7, 0.f);
  predictions[0] = 1.f;
  predictions[1] = 0.5f;
  predictions[2] = 1.f;
  predictions[3] = 1.f;
  predictions[4] = 0.95f;
  predictions[6] = 1.f;
  const std::vector<float> proto = {1.f, 1.f, 1.f};
  vie::DecoderConfig config = seg_config(2, 1);
  const vie::SegmentationBatch cropped =
      run_seg(predictions, {1, 7, 1}, proto, {1, 1, 1, 3}, config, identity(3, 1));
  ASSERT_EQ(cropped.instances.size(), 1u);
  ASSERT_EQ(cropped.instances[0].mask.width(), 3);
  EXPECT_EQ(cropped.instances[0].mask.binary_data()[0], 0);
  EXPECT_EQ(cropped.instances[0].mask.binary_data()[1], 1);
  EXPECT_EQ(cropped.instances[0].mask.binary_data()[2], 0);

  vie::GeometryTransform letterbox;
  letterbox.original_width = 2;
  letterbox.original_height = 2;
  letterbox.output_width = 4;
  letterbox.output_height = 4;
  letterbox.resized_width = 2;
  letterbox.resized_height = 2;
  letterbox.pad_left = 1;
  letterbox.pad_top = 1;
  letterbox.pad_right = 1;
  letterbox.pad_bottom = 1;
  letterbox.scale_x = 1;
  letterbox.scale_y = 1;
  letterbox.uniform_scale = 0.5;
  std::vector<float> retina_proto(16, -1.f);
  retina_proto[1 * 4 + 1] = 1.f;
  retina_proto[1 * 4 + 2] = 1.f;
  retina_proto[2 * 4 + 1] = -1.f;
  retina_proto[2 * 4 + 2] = 1.f;
  std::vector<float> retina_pred(7, 0.f);
  retina_pred[0] = 2.f;
  retina_pred[1] = 2.f;
  retina_pred[2] = 2.f;
  retina_pred[3] = 2.f;
  retina_pred[4] = 0.9f;
  retina_pred[6] = 1.f;
  vie::DecoderConfig retina = config;
  retina.retina_masks = true;
  const vie::SegmentationBatch high =
      run_seg(retina_pred, {1, 7, 1}, retina_proto, {1, 1, 4, 4}, retina, letterbox);
  ASSERT_EQ(high.instances.size(), 1u);
  EXPECT_EQ(high.instances[0].mask.width(), 2);
  EXPECT_EQ(high.instances[0].mask.height(), 2);
  EXPECT_EQ(high.instances[0].mask.binary_data()[0], 1);
  EXPECT_EQ(high.instances[0].mask.binary_data()[1], 1);
  EXPECT_EQ(high.instances[0].mask.binary_data()[2], 0);
  EXPECT_EQ(high.instances[0].mask.binary_data()[3], 1);
  EXPECT_NEAR(high.instances[0].detection.box.x1, 0.f, 1e-5f);
  EXPECT_NEAR(high.instances[0].detection.box.x2, 2.f, 1e-5f);

  vie::PreprocessConfig stretch_config;
  stretch_config.resize_mode = vie::ResizeMode::Stretch;
  stretch_config.target_width = 4;
  stretch_config.target_height = 1;
  stretch_config.interpolation = vie::Interpolation::Bilinear;
  const vie::GeometryTransform stretch = vie::make_geometry(stretch_config, 2, 1);
  EXPECT_NE(stretch.scale_x, stretch.scale_y);
  std::vector<float> stretch_pred(7, 0.f);
  stretch_pred[0] = 2.f;
  stretch_pred[1] = 0.5f;
  stretch_pred[2] = 4.f;
  stretch_pred[3] = 1.f;
  stretch_pred[4] = 0.9f;
  stretch_pred[6] = 1.f;
  const std::vector<float> stretch_proto = {0.f, 0.f, 10.f, 0.f};
  vie::DecoderConfig stretch_decode = retina;
  const vie::SegmentationBatch stretched =
      run_seg(stretch_pred, {1, 7, 1}, stretch_proto, {1, 1, 1, 4}, stretch_decode, stretch);
  ASSERT_EQ(stretched.instances.size(), 1u);
  EXPECT_EQ(stretched.instances[0].mask.width(), 2);
  EXPECT_EQ(stretched.instances[0].mask.height(), 1);
  EXPECT_EQ(stretched.instances[0].mask.binary_data()[0], 0);
  EXPECT_EQ(stretched.instances[0].mask.binary_data()[1], 1);

  vie::PreprocessConfig odd_config;
  odd_config.resize_mode = vie::ResizeMode::Letterbox;
  odd_config.target_width = 5;
  odd_config.target_height = 5;
  odd_config.interpolation = vie::Interpolation::Bilinear;
  odd_config.padding_value = 114.f;
  const vie::GeometryTransform odd = vie::make_geometry(odd_config, 2, 3);
  EXPECT_NE(odd.scale_x, odd.scale_y);
  EXPECT_EQ(odd.pad_left % 2, 1);
  EXPECT_EQ(odd.pad_right % 2, 1);
  std::vector<float> odd_proto(25, 1.f);
  std::vector<float> odd_pred(7, 0.f);
  odd_pred[0] = 2.5f;
  odd_pred[1] = 2.5f;
  odd_pred[2] = 5.f;
  odd_pred[3] = 5.f;
  odd_pred[4] = 0.9f;
  odd_pred[6] = 1.f;
  vie::DecoderConfig odd_decode = config;
  odd_decode.retina_masks = true;
  const vie::SegmentationBatch odd_batch =
      run_seg(odd_pred, {1, 7, 1}, odd_proto, {1, 1, 5, 5}, odd_decode, odd);
  ASSERT_EQ(odd_batch.instances.size(), 1u);
  EXPECT_EQ(odd_batch.instances[0].mask.width(), 2);
  EXPECT_EQ(odd_batch.instances[0].mask.height(), 3);
  expect_binary(odd_batch.instances[0].mask);
}

TEST(YoloV8SegCpu, RejectsMissingConfigBadTensorsAndBudget) {
  std::vector<float> predictions(7, 0.f);
  predictions[0] = 1.f;
  predictions[1] = 1.f;
  predictions[2] = 2.f;
  predictions[3] = 2.f;
  predictions[4] = 0.9f;
  predictions[6] = 1.f;
  const std::vector<float> proto = {1.f, 1.f, 1.f, 1.f};
  vie::DecoderConfig config = seg_config(2, 1);
  vie::DecoderInput input;
  input.geometry = identity(2, 2);
  input.outputs.emplace("output0", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 7, 1},
                                                         predictions.data(), predictions.size() * sizeof(float)));
  input.outputs.emplace("output1", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 1, 2, 2},
                                                         proto.data(), proto.size() * sizeof(float)));
  vie::YoloV8SegmentationDecoder decoder;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, decoder.decode_batch(input, config, vie::DeviceContext{}));
  vie::DecoderInput missing_proto = input;
  missing_proto.outputs.erase("output1");
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_segmentation(missing_proto, config, vie::DeviceContext{}));
  vie::DecoderInput missing_pred = input;
  missing_pred.outputs.erase("output0");
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_segmentation(missing_pred, config, vie::DeviceContext{}));
  vie::DecoderInput no_geometry = input;
  no_geometry.geometry.reset();
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   decoder.decode_segmentation(no_geometry, config, vie::DeviceContext{}));
  vie::DecoderConfig unresolved = config;
  unresolved.retina_masks.reset();
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   decoder.decode_segmentation(input, unresolved, vie::DeviceContext{}));
  unresolved = config;
  unresolved.class_aware_nms.reset();
  VIE_EXPECT_ERROR(vie::ErrorCode::UnresolvedField,
                   decoder.decode_segmentation(input, unresolved, vie::DeviceContext{}));
  vie::DecoderConfig wrong_channels = config;
  wrong_channels.mask_channels = 3;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_segmentation(input, wrong_channels, vie::DeviceContext{}));
  std::vector<std::uint8_t> bytes(7, 0);
  vie::DecoderInput wrong_type = input;
  wrong_type.outputs.erase("output0");
  wrong_type.outputs.emplace("output0", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Uint8, {1, 7, 1},
                                                              bytes.data(), bytes.size()));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_segmentation(wrong_type, config, vie::DeviceContext{}));
  std::vector<float> wide_proto(8, 1.f);
  vie::DecoderInput channel_mismatch = input;
  channel_mismatch.outputs.erase("output1");
  channel_mismatch.outputs.emplace(
      "output1", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 2, 2, 2}, wide_proto.data(),
                                        wide_proto.size() * sizeof(float)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_segmentation(channel_mismatch, config, vie::DeviceContext{}));
  std::vector<float> nan_proto = proto;
  nan_proto[0] = std::numeric_limits<float>::quiet_NaN();
  vie::DecoderInput nan_input = input;
  nan_input.outputs.erase("output1");
  nan_input.outputs.emplace("output1", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 1, 2, 2},
                                                             nan_proto.data(), nan_proto.size() * sizeof(float)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_segmentation(nan_input, config, vie::DeviceContext{}));
  const std::vector<float> interleaved = {1.f, 1.f, 2.f, 2.f, 0.8f, 0.f, 1.f};
  vie::DecoderConfig axis2 = config;
  axis2.channel_axis = 2;
  vie::DecoderInput interleaved_input;
  interleaved_input.geometry = identity(2, 2);
  interleaved_input.outputs.emplace(
      "output0", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 1, 7}, interleaved.data(),
                                        interleaved.size() * sizeof(float)));
  interleaved_input.outputs.emplace("output1", input.outputs.at("output1"));
  const vie::SegmentationBatch axis_batch =
      decoder.decode_segmentation(interleaved_input, axis2, vie::DeviceContext{});
  ASSERT_EQ(axis_batch.instances.size(), 1u);
  EXPECT_EQ(axis_batch.instances[0].mask.binary_data()[0], 1);
  predictions[6] = std::numeric_limits<float>::infinity();
  vie::DecoderInput infinite = input;
  infinite.outputs.erase("output0");
  infinite.outputs.emplace("output0", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 7, 1},
                                                            predictions.data(), predictions.size() * sizeof(float)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_segmentation(infinite, config, vie::DeviceContext{}));
  vie::DecoderConfig budget = config;
  budget.max_mask_bytes = 3;
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument, decoder.decode_segmentation(input, budget, vie::DeviceContext{}));
#if defined(VIE_HAS_CUDA)
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   decoder.decode_segmentation(input, config, vie::DeviceContext{vie::DeviceType::Cuda, 0}));
#else
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported,
                   decoder.decode_segmentation(input, config, vie::DeviceContext{vie::DeviceType::Cuda, 0}));
#endif
  vie::DecoderRegistry registry;
  VIE_EXPECT_ERROR(vie::ErrorCode::Unsupported, registry.create("yolov8_seg_raw"));
  vie::InferenceEngine engine;
  EXPECT_TRUE(engine.decoders().contains("yolov8_seg_raw"));
  EXPECT_EQ(engine.decoders().create("yolov8_seg_raw")->protocol(), "yolov8_seg_raw");
}

#if defined(VIE_HAS_CUDA)
TEST(YoloV8SegCpu, RejectsACudaTensor) {
  const std::vector<float> predictions(7, 0.f);
  const std::vector<float> proto(4, 0.f);
  vie::DeviceContext gpu(vie::DeviceType::Cuda, 0);
  vie::DecoderInput input;
  input.geometry = identity(2, 2);
  input.outputs.emplace("output0", vie::Tensor::from_host(gpu, vie::DataType::Fp32, {1, 7, 1}, predictions.data(),
                                                         predictions.size() * sizeof(float)));
  input.outputs.emplace("output1", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32, {1, 1, 2, 2},
                                                         proto.data(), proto.size() * sizeof(float)));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::YoloV8SegmentationDecoder{}.decode_segmentation(input, seg_config(2, 1), vie::DeviceContext{}));
}
#endif

#if defined(VIE_HAS_ONNXRUNTIME)
TEST(YoloV8SegCpu, RealModelUsesExplicitCpuConfig) {
  const std::string path = vie_fixture("testdata/onnx/yolov8n-seg.onnx");
  vie::PreprocessConfig prep;
  prep.resize_mode = vie::ResizeMode::Letterbox;
  prep.target_width = 640;
  prep.target_height = 640;
  prep.interpolation = vie::Interpolation::Bilinear;
  prep.input_color = vie::ColorOrder::Rgb;
  prep.output_color = vie::ColorOrder::Rgb;
  prep.normalize_enabled = true;
  prep.scale = 1.f / 255.f;
  prep.mean = {0.f, 0.f, 0.f};
  prep.std = {1.f, 1.f, 1.f};
  prep.padding_value = 114.f;
  vie::ImagePreprocessor preprocessor;
  preprocessor.configure(prep);
  std::ifstream image_file(vie_fixture("testdata/images/bus.ppm"), std::ios::binary);
  std::string magic;
  int image_width = 0;
  int image_height = 0;
  int max_value = 0;
  image_file >> magic >> image_width >> image_height >> max_value;
  image_file.get();
  ASSERT_EQ(magic, "P6");
  ASSERT_EQ(max_value, 255);
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(image_width * image_height * 3));
  image_file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  ASSERT_TRUE(image_file);
  vie::ImageInput image;
  image.data = pixels.data();
  image.width = image_width;
  image.height = image_height;
  image.channels = 3;
  image.color = vie::ColorOrder::Rgb;
  image.layout = vie::ImageLayout::Hwc;
  image.dtype = vie::DataType::Uint8;
  const vie::PreprocessResult prepared = preprocessor.execute(image, vie::DeviceContext{});
  vie::BackendConfig backend_config;
  backend_config.model_path = path;
  backend_config.model_format = vie::ModelFormat::Onnx;
  backend_config.device = vie::DeviceType::Cpu;
  backend_config.input.name = "images";
  backend_config.input.shape = {1, 3, 640, 640};
  backend_config.input.dtype = vie::DataType::Fp32;
  vie::OnnxRuntimeBackend backend;
  backend.prepare(backend_config);
  vie::BackendRequest request;
  request.inputs.emplace("images", prepared.output);
  const vie::BackendOutput output = backend.execute(request);
  ASSERT_EQ(output.tensors.count("output0"), 1u);
  ASSERT_EQ(output.tensors.count("output1"), 1u);
  const vie::Tensor& prediction = output.tensors.at("output0").tensor;
  const vie::Tensor& prototype = output.tensors.at("output1").tensor;
  ASSERT_EQ(prediction.shape().size(), 3u);
  ASSERT_EQ(prototype.shape().size(), 4u);
  const int channels = static_cast<int>(prediction.shape()[1]);
  const int mask_channels = static_cast<int>(prototype.shape()[1]);
  vie::DecoderConfig config = seg_config(channels - 4 - mask_channels, mask_channels);
  config.retina_masks = true;
  config.confidence_threshold = 0.25f;
  config.iou_threshold = 0.45f;
  config.max_detections = 20;
  config.pre_nms_top_k = 300;
  config.max_mask_bytes = 64ull << 20;
  vie::DecoderInput decoded;
  decoded.geometry = prepared.geometry;
  decoded.outputs.emplace("output0", prediction);
  decoded.outputs.emplace("output1", prototype);
  const vie::SegmentationBatch batch =
      vie::YoloV8SegmentationDecoder{}.decode_segmentation(decoded, config, vie::DeviceContext{});
  EXPECT_EQ(batch.execution_device, vie::DeviceType::Cpu);
  EXPECT_EQ(batch.original_width, image_width);
  EXPECT_EQ(batch.original_height, image_height);
  EXPECT_EQ(batch.protocol, "yolov8_seg_raw");
  ASSERT_FALSE(batch.instances.empty());
  const std::vector<IndependentInstance> reference = independent_kept(
      prediction.data_as<float>(), static_cast<int>(prediction.shape()[2]), config.num_classes, mask_channels, config,
      prepared.geometry);
  ASSERT_EQ(reference.size(), batch.instances.size());
  std::ofstream coeff_file("/tmp/vie_seg_ref_coeffs.bin", std::ios::binary);
  std::ofstream model_box_file("/tmp/vie_seg_ref_model_box.bin", std::ios::binary);
  std::ofstream orig_box_file("/tmp/vie_seg_ref_orig_box.bin", std::ios::binary);
  std::ofstream mask_file("/tmp/vie_seg_ref_mask.bin", std::ios::binary);
  std::ofstream proto_file("/tmp/vie_seg_ref_proto.bin", std::ios::binary);
  proto_file.write(reinterpret_cast<const char*>(prototype.data_as<float>()),
                   static_cast<std::streamsize>(prototype.nbytes()));
  int mismatched = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    EXPECT_EQ(batch.instances[i].detection.class_id, reference[i].class_id);
    EXPECT_FLOAT_EQ(batch.instances[i].detection.confidence, reference[i].score);
    EXPECT_FLOAT_EQ(batch.instances[i].detection.box.x1, reference[i].original_box.x1);
    EXPECT_FLOAT_EQ(batch.instances[i].detection.box.y2, reference[i].original_box.y2);
    const std::vector<std::uint8_t> mask = vie_ref::binary_mask(
        prototype.data_as<float>(), mask_channels, static_cast<int>(prototype.shape()[2]),
        static_cast<int>(prototype.shape()[3]), reference[i].coefficients, reference[i].model_box,
        reference[i].original_box, prepared.geometry, true);
    ASSERT_EQ(mask.size(), static_cast<std::size_t>(batch.instances[i].mask.width() * batch.instances[i].mask.height()));
    mismatched += std::memcmp(mask.data(), batch.instances[i].mask.binary_data(), mask.size()) == 0 ? 0 : 1;
    coeff_file.write(reinterpret_cast<const char*>(reference[i].coefficients.data()),
                     static_cast<std::streamsize>(reference[i].coefficients.size() * sizeof(float)));
    const float model_box[4] = {reference[i].model_box.x1, reference[i].model_box.y1, reference[i].model_box.x2,
                                reference[i].model_box.y2};
    const float orig_box[4] = {reference[i].original_box.x1, reference[i].original_box.y1, reference[i].original_box.x2,
                               reference[i].original_box.y2};
    model_box_file.write(reinterpret_cast<const char*>(model_box), sizeof(model_box));
    orig_box_file.write(reinterpret_cast<const char*>(orig_box), sizeof(orig_box));
    mask_file.write(reinterpret_cast<const char*>(batch.instances[i].mask.binary_data()),
                    static_cast<std::streamsize>(mask.size()));
  }
  EXPECT_EQ(mismatched, 0);
  std::ofstream meta("/tmp/vie_seg_ref_meta.txt");
  meta << "instances=" << reference.size() << " proto=" << prototype.shape()[1] << "x" << prototype.shape()[2] << "x"
       << prototype.shape()[3] << " model=640x640 original=" << image_width << "x" << image_height
       << " pad_left=" << prepared.geometry.pad_left << " pad_top=" << prepared.geometry.pad_top
       << " scale_x=" << prepared.geometry.scale_x << " scale_y=" << prepared.geometry.scale_y
       << " retina=1 mismatched=" << mismatched << "\n";
  for (const vie::InstanceSegmentation& instance : batch.instances) {
    EXPECT_GE(instance.detection.confidence, 0.25f);
    EXPECT_EQ(instance.mask.width(), image_width);
    EXPECT_EQ(instance.mask.height(), image_height);
    EXPECT_GE(instance.detection.box.x1, 0.f);
    EXPECT_GE(instance.detection.box.y1, 0.f);
    EXPECT_LE(instance.detection.box.x2, static_cast<float>(image_width));
    EXPECT_LE(instance.detection.box.y2, static_cast<float>(image_height));
    expect_binary(instance.mask);
  }
  const float* scores = prediction.data_as<float>();
  const int prediction_count = static_cast<int>(prediction.shape()[2]);
  float max_score = 0.f;
  for (int cls = 0; cls < config.num_classes; ++cls) {
    const float* row = scores + static_cast<std::size_t>(4 + cls) * static_cast<std::size_t>(prediction_count);
    for (int i = 0; i < prediction_count; ++i) max_score = std::max(max_score, row[i]);
  }
  std::ofstream out("/tmp/vie_phase433_real.txt");
  out << "device=cpu_ep\n";
  out << "output0=" << prediction.shape()[0] << "x" << prediction.shape()[1] << "x" << prediction.shape()[2] << "\n";
  out << "output1=" << prototype.shape()[0] << "x" << prototype.shape()[1] << "x" << prototype.shape()[2] << "x"
      << prototype.shape()[3] << "\n";
  out << "classes=" << config.num_classes << " mask_channels=" << mask_channels << "\n";
  out << "image=" << image_width << "x" << image_height << " max_class_score=" << max_score << "\n";
  out << "instances=" << batch.instances.size() << " mask=" << batch.instances[0].mask.width() << "x"
      << batch.instances[0].mask.height() << " binary=1 finite=1\n";
  out << "preprocess=explicit letterbox rgb scale_0_1 bilinear padding 114\n";
  out << "decoder=explicit best_class conf 0.25 iou 0.45 class_aware retina_masks true\n";
}
#endif

TEST(YoloV8SegBenchmark, CpuStages) {
  constexpr int kPredictions = 8400;
  constexpr int kClasses = 80;
  constexpr int kMasks = 32;
  constexpr int kChannels = 4 + kClasses + kMasks;
  constexpr int kRuns = 5;
  std::vector<float> predictions(static_cast<std::size_t>(kChannels) * kPredictions, 0.f);
  for (int i = 0; i < 16; ++i) {
    const int prediction = i * 20;
    put(predictions, kPredictions, 0, prediction, static_cast<float>(20 + (i % 5) * 30));
    put(predictions, kPredictions, 1, prediction, static_cast<float>(20 + (i % 7) * 20));
    put(predictions, kPredictions, 2, prediction, 12.f);
    put(predictions, kPredictions, 3, prediction, 12.f);
    put(predictions, kPredictions, 4 + (i % kClasses), prediction, 0.9f);
    for (int channel = 0; channel < kMasks; ++channel) {
      put(predictions, kPredictions, 4 + kClasses + channel, prediction, (channel % 2 == 0 ? 0.5f : -0.25f));
    }
  }
  const int proto_side = 40;
  std::vector<float> proto(static_cast<std::size_t>(kMasks * proto_side * proto_side), 0.1f);
  vie::DecoderConfig config = seg_config(kClasses, kMasks);
  config.pre_nms_top_k = 300;
  config.max_detections = 16;
  config.max_mask_bytes = 64ull << 20;
  vie::PreprocessConfig prep;
  prep.resize_mode = vie::ResizeMode::Letterbox;
  prep.target_width = 80;
  prep.target_height = 80;
  prep.interpolation = vie::Interpolation::Bilinear;
  prep.padding_value = 114.f;
  const vie::GeometryTransform geometry = vie::make_geometry(prep, 60, 40);
  vie::DecoderInput input;
  input.geometry = geometry;
  input.outputs.emplace("output0", vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32,
                                                         {1, kChannels, kPredictions}, predictions.data(),
                                                         predictions.size() * sizeof(float)));
  input.outputs.emplace("output1",
                        vie::Tensor::from_host(vie::DeviceContext{}, vie::DataType::Fp32,
                                               {1, kMasks, proto_side, proto_side}, proto.data(),
                                               proto.size() * sizeof(float)));
  vie::YoloV8SegmentationDecoder decoder;
  vie::SegmentationTiming timing;
  const vie::SegmentationBatch once = decoder.decode_timed(input, config, vie::DeviceContext{}, &timing);
  EXPECT_GT(timing.instances, 0);
  EXPECT_EQ(once.instances.size(), static_cast<std::size_t>(timing.instances));
  std::vector<vie::SegmentationTiming> samples;
  for (int run = 0; run < kRuns; ++run) {
    vie::SegmentationTiming sample;
    decoder.decode_timed(input, config, vie::DeviceContext{}, &sample);
    samples.push_back(sample);
  }
  auto write_case = [](std::ostream& out, const char* name, const vie::SegmentationTiming& once,
                      const std::vector<vie::SegmentationTiming>& rows) {
    auto summarize = [](const std::vector<vie::SegmentationTiming>& samples, double vie::SegmentationTiming::*field) {
      std::vector<double> values;
      for (const vie::SegmentationTiming& row : samples) values.push_back(row.*field);
      std::sort(values.begin(), values.end());
      double sum = 0;
      for (double value : values) sum += value;
      return std::tuple<double, double, double>{sum / static_cast<double>(values.size()), values[values.size() / 2],
                                                values[static_cast<std::size_t>(values.size() * 0.95)]};
    };
    const auto detection = summarize(rows, &vie::SegmentationTiming::detection_ms);
    const auto filter = summarize(rows, &vie::SegmentationTiming::filter_ms);
    const auto nms = summarize(rows, &vie::SegmentationTiming::nms_ms);
    const auto gather = summarize(rows, &vie::SegmentationTiming::gather_ms);
    const auto reconstruct = summarize(rows, &vie::SegmentationTiming::reconstruct_ms);
    const auto crop = summarize(rows, &vie::SegmentationTiming::crop_ms);
    const auto resize = summarize(rows, &vie::SegmentationTiming::resize_ms);
    const auto total = summarize(rows, &vie::SegmentationTiming::total_ms);
    out << "case=" << name << " instances=" << once.instances << " mask=" << once.mask_width << "x" << once.mask_height
        << " runs=" << rows.size() << " includes_allocation=1\n";
    out << "detection_mean_ms=" << std::get<0>(detection) << " p50=" << std::get<1>(detection)
        << " p95=" << std::get<2>(detection) << "\n";
    out << "filter_mean_ms=" << std::get<0>(filter) << " p50=" << std::get<1>(filter) << " p95=" << std::get<2>(filter)
        << "\n";
    out << "nms_mean_ms=" << std::get<0>(nms) << " p50=" << std::get<1>(nms) << " p95=" << std::get<2>(nms) << "\n";
    out << "gather_mean_ms=" << std::get<0>(gather) << " p50=" << std::get<1>(gather) << " p95=" << std::get<2>(gather)
        << "\n";
    out << "reconstruct_mean_ms=" << std::get<0>(reconstruct) << " p50=" << std::get<1>(reconstruct)
        << " p95=" << std::get<2>(reconstruct) << "\n";
    out << "crop_mean_ms=" << std::get<0>(crop) << " p50=" << std::get<1>(crop) << " p95=" << std::get<2>(crop) << "\n";
    out << "resize_mean_ms=" << std::get<0>(resize) << " p50=" << std::get<1>(resize) << " p95=" << std::get<2>(resize)
        << "\n";
    out << "full_mean_ms=" << std::get<0>(total) << " p50=" << std::get<1>(total) << " p95=" << std::get<2>(total)
        << "\n";
  };
  vie::DecoderConfig single = config;
  single.retina_masks = true;
  single.max_detections = 1;
  vie::PreprocessConfig large;
  large.resize_mode = vie::ResizeMode::Letterbox;
  large.target_width = 80;
  large.target_height = 80;
  large.interpolation = vie::Interpolation::Bilinear;
  large.padding_value = 114.f;
  vie::DecoderInput large_input = input;
  large_input.geometry = vie::make_geometry(large, 160, 120);
  vie::SegmentationTiming single_once;
  const vie::SegmentationBatch single_batch =
      decoder.decode_timed(large_input, single, vie::DeviceContext{}, &single_once);
  EXPECT_EQ(single_once.instances, 1);
  EXPECT_EQ(single_batch.instances.size(), 1u);
  EXPECT_EQ(single_once.mask_width, 160);
  EXPECT_EQ(single_once.mask_height, 120);
  std::vector<vie::SegmentationTiming> single_samples;
  for (int run = 0; run < kRuns; ++run) {
    vie::SegmentationTiming sample;
    decoder.decode_timed(large_input, single, vie::DeviceContext{}, &sample);
    single_samples.push_back(sample);
  }
  std::ofstream out("/tmp/vie_phase433_cpu.txt");
  out << "predictions=" << kPredictions << " classes=" << kClasses << " mask_channels=" << kMasks
      << " proto=" << proto_side << "x" << proto_side << " retina_false_model_input\n";
  write_case(out, "retina_false_80x80", timing, samples);
  out << "case_retina_true_original\n";
  write_case(out, "retina_true_160x120", single_once, single_samples);
}
