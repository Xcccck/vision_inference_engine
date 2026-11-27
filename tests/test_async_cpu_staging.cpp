// Phase 11.1: async CPU input ownership.
//
// InferenceRequest::adopt_cpu_pixels() must make a queued request independent
// of the caller's buffer and must never read past the last valid pixel. The
// tests below use std::vector buffers sized to the exact parent allocation, so
// a build with AddressSanitizer turns any strided over-read into a hard error.

#include "test_util.hpp"

#include "vie/core/error.hpp"
#include "vie/runtime/inference_request.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace vie {
namespace {

std::uint8_t pattern(int x, int y, int channel) {
  return static_cast<std::uint8_t>((x * 31 + y * 17 + channel * 7 + 3) & 0xFF);
}

ImageInput hwc_view(const void* data, int width, int height, int channels,
                    std::size_t row_stride) {
  ImageInput image;
  image.data = data;
  image.width = width;
  image.height = height;
  image.channels = channels;
  image.color = channels == 1 ? ColorOrder::Gray : ColorOrder::Bgr;
  image.layout = ImageLayout::Hwc;
  image.dtype = DataType::Uint8;
  image.row_stride_bytes = row_stride;
  image.device = DeviceType::Cpu;
  image.device_id = 0;
  image.ownership = ImageOwnership::Borrowed;
  return image;
}

// Fills a parent HWC buffer with a position-dependent pattern so a row shift or
// a padding leak is visible in the copied bytes.
std::vector<std::uint8_t> make_parent(int parent_width, int parent_height, int channels) {
  std::vector<std::uint8_t> parent(static_cast<std::size_t>(parent_width) *
                                   static_cast<std::size_t>(parent_height) *
                                   static_cast<std::size_t>(channels));
  for (int y = 0; y < parent_height; ++y) {
    for (int x = 0; x < parent_width; ++x) {
      for (int c = 0; c < channels; ++c) {
        const std::size_t offset =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(parent_width) +
             static_cast<std::size_t>(x)) *
                static_cast<std::size_t>(channels) +
            static_cast<std::size_t>(c);
        parent[offset] = pattern(x, y, c);
      }
    }
  }
  return parent;
}

// Verifies that one copied HWC image holds exactly the ROI pixels, packed.
void expect_packed_roi(const std::vector<std::uint8_t>& parent, int parent_width, int roi_x,
                       int roi_y, int roi_width, int roi_height, int channels,
                       const std::vector<std::uint8_t>& copied, std::size_t copied_stride) {
  const std::size_t packed_row =
      static_cast<std::size_t>(roi_width) * static_cast<std::size_t>(channels);
  ASSERT_EQ(copied.size(), packed_row * static_cast<std::size_t>(roi_height));
  EXPECT_EQ(copied_stride, packed_row);
  for (int y = 0; y < roi_height; ++y) {
    for (int x = 0; x < roi_width; ++x) {
      for (int c = 0; c < channels; ++c) {
        const std::size_t source_offset =
            (static_cast<std::size_t>(roi_y + y) * static_cast<std::size_t>(parent_width) +
             static_cast<std::size_t>(roi_x + x)) *
                static_cast<std::size_t>(channels) +
            static_cast<std::size_t>(c);
        const std::size_t copied_offset =
            static_cast<std::size_t>(y) * packed_row +
            static_cast<std::size_t>(x) * static_cast<std::size_t>(channels) +
            static_cast<std::size_t>(c);
        ASSERT_EQ(copied[copied_offset], parent[source_offset])
            << "row " << y << " x " << x << " channel " << c;
      }
    }
  }
}

// ------------------------------------------------------------------
// Continuous images.
// ------------------------------------------------------------------

TEST(AsyncCpuStaging, CopiesContinuousHwcImage) {
  constexpr int kWidth = 37;
  constexpr int kHeight = 13;
  constexpr int kChannels = 3;
  const std::vector<std::uint8_t> source = make_parent(kWidth, kHeight, kChannels);

  InferenceRequest request;
  request.image = hwc_view(source.data(), kWidth, kHeight, kChannels,
                           static_cast<std::size_t>(kWidth) * kChannels);
  request.adopt_cpu_pixels();

  EXPECT_EQ(request.owned_pixels.size(), source.size());
  EXPECT_EQ(request.owned_pixels, source);
  EXPECT_EQ(request.image.data, static_cast<const void*>(request.owned_pixels.data()));
  EXPECT_EQ(request.image.row_stride_bytes, static_cast<std::size_t>(kWidth) * kChannels);
  EXPECT_EQ(request.image.width, kWidth);
  EXPECT_EQ(request.image.height, kHeight);
  EXPECT_EQ(request.image.channels, kChannels);
  EXPECT_EQ(request.image.color, ColorOrder::Bgr);
  EXPECT_EQ(request.image.layout, ImageLayout::Hwc);
  EXPECT_EQ(request.image.dtype, DataType::Uint8);
  EXPECT_EQ(request.image.device, DeviceType::Cpu);
}

TEST(AsyncCpuStaging, ZeroStrideMeansPackedRow) {
  constexpr int kWidth = 9;
  constexpr int kHeight = 5;
  constexpr int kChannels = 1;
  const std::vector<std::uint8_t> source = make_parent(kWidth, kHeight, kChannels);

  InferenceRequest request;
  request.image = hwc_view(source.data(), kWidth, kHeight, kChannels, 0);
  request.adopt_cpu_pixels();

  EXPECT_EQ(request.image.row_stride_bytes, static_cast<std::size_t>(kWidth));
  EXPECT_EQ(request.owned_pixels, source);
}

// ------------------------------------------------------------------
// ROI placement: center, right edge, bottom edge, both, nested, height 1.
// Each parent buffer is allocated to its exact size, so a memcpy of
// stride*height instead of packed_row*height reads past the allocation.
// ------------------------------------------------------------------

TEST(AsyncCpuStaging, PacksRoiAtEveryParentEdge) {
  constexpr int kRoiWidth = 40;
  constexpr int kRoiHeight = 30;
  constexpr int kChannels = 3;
  struct Placement {
    const char* name;
    int left;
    int top;
    int right;
    int bottom;
  };
  const Placement placements[] = {
      {"center", 13, 9, 17, 11},
      {"right_edge", 21, 7, 0, 13},
      {"bottom_edge", 11, 17, 13, 0},
      {"bottom_right", 17, 13, 0, 0},
      {"top_left", 0, 0, 23, 19},
  };

  for (const Placement& placement : placements) {
    SCOPED_TRACE(placement.name);
    const int parent_width = placement.left + kRoiWidth + placement.right;
    const int parent_height = placement.top + kRoiHeight + placement.bottom;
    const std::vector<std::uint8_t> parent = make_parent(parent_width, parent_height, kChannels);
    const std::size_t parent_stride =
        static_cast<std::size_t>(parent_width) * static_cast<std::size_t>(kChannels);
    const std::uint8_t* roi_data =
        parent.data() + static_cast<std::size_t>(placement.top) * parent_stride +
        static_cast<std::size_t>(placement.left) * static_cast<std::size_t>(kChannels);

    InferenceRequest request;
    request.image = hwc_view(roi_data, kRoiWidth, kRoiHeight, kChannels, parent_stride);
    ASSERT_EQ(request.image.data, static_cast<const void*>(roi_data));
    request.adopt_cpu_pixels();

    expect_packed_roi(parent, parent_width, placement.left, placement.top, kRoiWidth, kRoiHeight,
                      kChannels, request.owned_pixels, request.image.row_stride_bytes);
  }
}

TEST(AsyncCpuStaging, PacksNestedRoi) {
  constexpr int kParentWidth = 100;
  constexpr int kParentHeight = 80;
  constexpr int kChannels = 3;
  const std::vector<std::uint8_t> parent = make_parent(kParentWidth, kParentHeight, kChannels);
  const std::size_t parent_stride =
      static_cast<std::size_t>(kParentWidth) * static_cast<std::size_t>(kChannels);

  // outer ROI (10, 5, 70, 60), then inner ROI (13, 9, 40, 30) inside it.
  const int roi_x = 10 + 13;
  const int roi_y = 5 + 9;
  constexpr int kRoiWidth = 40;
  constexpr int kRoiHeight = 30;
  const std::uint8_t* roi_data =
      parent.data() + static_cast<std::size_t>(roi_y) * parent_stride +
      static_cast<std::size_t>(roi_x) * static_cast<std::size_t>(kChannels);

  InferenceRequest request;
  request.image = hwc_view(roi_data, kRoiWidth, kRoiHeight, kChannels, parent_stride);
  request.adopt_cpu_pixels();

  expect_packed_roi(parent, kParentWidth, roi_x, roi_y, kRoiWidth, kRoiHeight, kChannels,
                    request.owned_pixels, request.image.row_stride_bytes);
}

TEST(AsyncCpuStaging, PacksSingleRowRoiEndingAtAllocationBoundary) {
  constexpr int kParentWidth = 50;
  constexpr int kParentHeight = 40;
  constexpr int kChannels = 3;
  constexpr int kRoiX = 35;
  constexpr int kRoiWidth = 15;
  const std::vector<std::uint8_t> parent = make_parent(kParentWidth, kParentHeight, kChannels);
  const std::size_t parent_stride =
      static_cast<std::size_t>(kParentWidth) * static_cast<std::size_t>(kChannels);

  // Last parent row, right edge: the ROI's last valid byte is parent.back().
  const std::uint8_t* roi_data =
      parent.data() + static_cast<std::size_t>(kParentHeight - 1) * parent_stride +
      static_cast<std::size_t>(kRoiX) * static_cast<std::size_t>(kChannels);
  ASSERT_EQ(roi_data + static_cast<std::size_t>(kRoiWidth) * kChannels, parent.data() + parent.size());

  InferenceRequest request;
  request.image = hwc_view(roi_data, kRoiWidth, 1, kChannels, parent_stride);
  request.adopt_cpu_pixels();

  expect_packed_roi(parent, kParentWidth, kRoiX, kParentHeight - 1, kRoiWidth, 1, kChannels,
                    request.owned_pixels, request.image.row_stride_bytes);
}

TEST(AsyncCpuStaging, PacksGrayRoiWithPadding) {
  constexpr int kParentWidth = 51;
  constexpr int kParentHeight = 33;
  constexpr int kChannels = 1;
  constexpr int kRoiX = 17;
  constexpr int kRoiY = 11;
  constexpr int kRoiWidth = 21;
  constexpr int kRoiHeight = 17;
  const std::vector<std::uint8_t> parent = make_parent(kParentWidth, kParentHeight, kChannels);
  const std::size_t parent_stride = static_cast<std::size_t>(kParentWidth);

  InferenceRequest request;
  request.image = hwc_view(parent.data() + static_cast<std::size_t>(kRoiY) * parent_stride + kRoiX,
                           kRoiWidth, kRoiHeight, kChannels, parent_stride);
  request.adopt_cpu_pixels();

  expect_packed_roi(parent, kParentWidth, kRoiX, kRoiY, kRoiWidth, kRoiHeight, kChannels,
                    request.owned_pixels, request.image.row_stride_bytes);
}

TEST(AsyncCpuStaging, PacksFp32RoiWithPadding) {
  constexpr int kParentWidth = 30;
  constexpr int kParentHeight = 20;
  constexpr int kChannels = 3;
  constexpr int kRoiX = 5;
  constexpr int kRoiY = 4;
  constexpr int kRoiWidth = 10;
  constexpr int kRoiHeight = 6;
  std::vector<float> parent(static_cast<std::size_t>(kParentWidth) * kParentHeight * kChannels);
  for (std::size_t i = 0; i < parent.size(); ++i) {
    parent[i] = static_cast<float>(i) * 0.25f;
  }
  const std::size_t parent_stride =
      static_cast<std::size_t>(kParentWidth) * kChannels * sizeof(float);

  ImageInput image = hwc_view(parent.data(), kRoiWidth, kRoiHeight, kChannels, parent_stride);
  image.dtype = DataType::Fp32;
  image.data = parent.data() + static_cast<std::size_t>(kRoiY) * kParentWidth * kChannels +
               static_cast<std::size_t>(kRoiX) * kChannels;

  InferenceRequest request;
  request.image = image;
  request.adopt_cpu_pixels();

  const std::size_t packed_row =
      static_cast<std::size_t>(kRoiWidth) * kChannels * sizeof(float);
  ASSERT_EQ(request.owned_pixels.size(), packed_row * kRoiHeight);
  EXPECT_EQ(request.image.row_stride_bytes, packed_row);
  const auto* copied = reinterpret_cast<const float*>(request.owned_pixels.data());
  for (int y = 0; y < kRoiHeight; ++y) {
    for (int x = 0; x < kRoiWidth; ++x) {
      for (int c = 0; c < kChannels; ++c) {
        const std::size_t expected_index =
            (static_cast<std::size_t>(kRoiY + y) * kParentWidth +
             static_cast<std::size_t>(kRoiX + x)) *
                kChannels +
            static_cast<std::size_t>(c);
        const std::size_t copied_index =
            (static_cast<std::size_t>(y) * kRoiWidth + static_cast<std::size_t>(x)) * kChannels +
            static_cast<std::size_t>(c);
        EXPECT_FLOAT_EQ(copied[copied_index], parent[expected_index]);
      }
    }
  }
}

// ------------------------------------------------------------------
// CHW: row_stride_bytes is the row stride inside one plane and the planes
// are contiguous. The copy must pack all planes, not just the first one.
// ------------------------------------------------------------------

TEST(AsyncCpuStaging, PacksAllChwPlanesWithPadding) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 3;
  constexpr int kChannels = 3;
  constexpr std::size_t kSourceStride = 8;  // > width*4 to leave per-row padding
  std::vector<std::uint8_t> source(kSourceStride * kHeight * kChannels, 0xEE);
  for (int p = 0; p < kChannels; ++p) {
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        source[static_cast<std::size_t>(p) * kHeight * kSourceStride +
               static_cast<std::size_t>(y) * kSourceStride + static_cast<std::size_t>(x)] =
            pattern(x, y, p);
      }
    }
  }

  ImageInput image = hwc_view(source.data(), kWidth, kHeight, kChannels, kSourceStride);
  image.layout = ImageLayout::Chw;

  InferenceRequest request;
  request.image = image;
  request.adopt_cpu_pixels();

  const std::size_t packed_row = static_cast<std::size_t>(kWidth);
  const std::size_t packed_plane = packed_row * kHeight;
  ASSERT_EQ(request.owned_pixels.size(), packed_plane * kChannels);
  EXPECT_EQ(request.image.row_stride_bytes, packed_row);
  EXPECT_EQ(request.image.layout, ImageLayout::Chw);
  for (int p = 0; p < kChannels; ++p) {
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        EXPECT_EQ(request.owned_pixels[static_cast<std::size_t>(p) * packed_plane +
                                       static_cast<std::size_t>(y) * packed_row + x],
                  pattern(x, y, p));
      }
    }
  }
}

TEST(AsyncCpuStaging, ChwZeroStrideMeansPackedPlanes) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 2;
  constexpr int kChannels = 3;
  std::vector<std::uint8_t> source(kWidth * kHeight * kChannels);
  for (std::size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<std::uint8_t>(i + 1);
  }

  ImageInput image = hwc_view(source.data(), kWidth, kHeight, kChannels, 0);
  image.layout = ImageLayout::Chw;

  InferenceRequest request;
  request.image = image;
  request.adopt_cpu_pixels();

  EXPECT_EQ(request.owned_pixels.size(), source.size());
  EXPECT_EQ(request.owned_pixels, source);
  EXPECT_EQ(request.image.row_stride_bytes, static_cast<std::size_t>(kWidth));
}

// ------------------------------------------------------------------
// Rejections and overflow.
// ------------------------------------------------------------------

TEST(AsyncCpuStaging, RejectsStrideSmallerThanPackedRow) {
  std::vector<std::uint8_t> source(16, 1);
  InferenceRequest request;
  request.image = hwc_view(source.data(), 4, 1, 3, 11);  // packed row is 12
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, request.adopt_cpu_pixels());
}

TEST(AsyncCpuStaging, RejectsMisalignedFp32Stride) {
  std::vector<std::uint8_t> source(64, 1);
  ImageInput image = hwc_view(source.data(), 4, 1, 3, 50);  // packed 48, 50 % 4 != 0
  image.dtype = DataType::Fp32;
  InferenceRequest request;
  request.image = image;
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, request.adopt_cpu_pixels());
}

TEST(AsyncCpuStaging, LeavesNonPositiveGeometryForDownstreamValidation) {
  // Runtime::submit() keeps its established contract: invalid geometry is
  // rejected by staging/preprocess inside the worker (through the future)
  // before any pixel is read, so adopt_cpu_pixels() must not copy or throw.
  std::vector<std::uint8_t> source(8, 1);
  InferenceRequest zero_width;
  zero_width.image = hwc_view(source.data(), 0, 2, 1, 0);
  zero_width.adopt_cpu_pixels();
  EXPECT_TRUE(zero_width.owned_pixels.empty());
  EXPECT_EQ(zero_width.image.data, static_cast<const void*>(source.data()));

  InferenceRequest negative_height;
  negative_height.image = hwc_view(source.data(), 2, -1, 1, 0);
  negative_height.adopt_cpu_pixels();
  EXPECT_TRUE(negative_height.owned_pixels.empty());

  InferenceRequest zero_channels;
  zero_channels.image = hwc_view(source.data(), 2, 2, 0, 0);
  zero_channels.adopt_cpu_pixels();
  EXPECT_TRUE(zero_channels.owned_pixels.empty());
}

TEST(AsyncCpuStaging, RejectsUnknownDtype) {
  std::vector<std::uint8_t> source(8, 1);
  ImageInput image = hwc_view(source.data(), 2, 2, 1, 0);
  image.dtype = DataType::Unknown;
  InferenceRequest request;
  request.image = image;
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, request.adopt_cpu_pixels());
}

TEST(AsyncCpuStaging, RejectsRowStrideTimesHeightOverflow) {
  std::vector<std::uint8_t> source(8, 1);
  InferenceRequest request;
  request.image = hwc_view(source.data(), 4, 2, 1, std::numeric_limits<std::size_t>::max() - 1);
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, request.adopt_cpu_pixels());
}

TEST(AsyncCpuStaging, RejectsPackedRowOverflow) {
  std::vector<std::uint8_t> source(8, 1);
  ImageInput image =
      hwc_view(source.data(), std::numeric_limits<int>::max(), 1,
               std::numeric_limits<int>::max(), 0);
  image.dtype = DataType::Int64;  // width*channels*8 crosses SIZE_MAX
  InferenceRequest request;
  request.image = image;
  VIE_EXPECT_ERROR(ErrorCode::InvalidArgument, request.adopt_cpu_pixels());
}

// ------------------------------------------------------------------
// Ownership semantics.
// ------------------------------------------------------------------

TEST(AsyncCpuStaging, AdoptIsIdempotent) {
  constexpr int kWidth = 8;
  constexpr int kHeight = 4;
  const std::vector<std::uint8_t> source = make_parent(kWidth, kHeight, 1);
  InferenceRequest request;
  request.image = hwc_view(source.data(), kWidth, kHeight, 1, kWidth);
  request.adopt_cpu_pixels();
  const void* first_data = request.image.data;
  const std::vector<std::uint8_t> first_copy = request.owned_pixels;

  request.adopt_cpu_pixels();

  EXPECT_EQ(request.image.data, first_data);
  EXPECT_EQ(request.owned_pixels, first_copy);
  EXPECT_EQ(request.image.row_stride_bytes, static_cast<std::size_t>(kWidth));
}

TEST(AsyncCpuStaging, SourceMutationAfterAdoptDoesNotAffectCopy) {
  constexpr int kParentWidth = 12;
  constexpr int kParentHeight = 6;
  std::vector<std::uint8_t> parent = make_parent(kParentWidth, kParentHeight, 1);
  InferenceRequest request;
  request.image = hwc_view(parent.data() + 2 * kParentWidth + 3, 4, 3, 1, kParentWidth);
  request.adopt_cpu_pixels();
  const std::vector<std::uint8_t> copied = request.owned_pixels;

  std::memset(parent.data(), 0xA5, parent.size());
  parent.clear();
  parent.shrink_to_fit();

  EXPECT_EQ(request.owned_pixels, copied);
  const auto* data = static_cast<const std::uint8_t*>(request.image.data);
  for (std::size_t i = 0; i < copied.size(); ++i) {
    EXPECT_EQ(data[i], copied[i]);
  }
}

TEST(AsyncCpuStaging, NullDataIsIgnored) {
  // Defensive path: Runtime::submit() already rejects non-CPU images, and a
  // null CPU buffer is left untouched (preprocess reports the error later).
  InferenceRequest request;
  request.image.data = nullptr;
  request.image.width = 4;
  request.image.height = 4;
  request.image.channels = 1;
  request.adopt_cpu_pixels();
  EXPECT_TRUE(request.owned_pixels.empty());
}

}  // namespace
}  // namespace vie