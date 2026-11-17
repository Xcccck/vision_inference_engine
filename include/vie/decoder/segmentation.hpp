#pragma once

#include "vie/decoder/detection.hpp"

#include <cstdint>
#include <vector>

namespace vie {

// Row-major height x width. This is the only mask layout.
enum class MaskLayout { Hw };

// Binary is uint8 0 or 1. Probability is float32 in [0, 1]. There is no logits
// kind: a value outside [0, 1] cannot be stored as Probability.
enum class MaskKind { Binary, Probability };

// Owns one contiguous HxW tensor. Copies share that storage. data() on the
// tensor follows Tensor rules: a CUDA mask has no host pointer.
class Mask {
 public:
  Mask() = default;

  static Mask binary(const DeviceContext& device, int width, int height);
  static Mask binary(const DeviceContext& device, int width, int height, const std::uint8_t* pixels,
                     std::size_t count);
  static Mask probability(const DeviceContext& device, int width, int height);
  static Mask probability(const DeviceContext& device, int width, int height, const float* pixels,
                          std::size_t count);
  // Uses the tensor's existing storage. The tensor must be contiguous HxW and
  // the dtype required by kind. CPU probability values are checked here. A CUDA
  // tensor is not copied to the host to perform that check.
  static Mask from_tensor(MaskKind kind, Tensor tensor);

  bool empty() const noexcept { return tensor_.empty(); }
  int width() const;
  int height() const;
  MaskKind kind() const;
  MaskLayout layout() const;
  DataType dtype() const;
  DeviceType device() const;
  int device_id() const;
  const std::vector<std::int64_t>& shape() const;
  const Tensor& tensor() const;

  // Host pointer. CUDA storage throws Unsupported, the same as Tensor::data().
  const std::uint8_t* binary_data() const;
  const float* probability_data() const;

  // CPU result. A CUDA mask is copied here and nowhere else.
  Mask to_cpu() const;

 private:
  Mask(MaskKind kind, Tensor tensor);

  MaskKind kind_ = MaskKind::Binary;
  Tensor tensor_;
};

struct InstanceSegmentation {
  Detection detection;
  Mask mask;
};

// One image. Another image is another SegmentationBatch; this step does not
// schedule batches. instances[i].mask belongs to instances[i].detection.
struct SegmentationBatch {
  std::vector<InstanceSegmentation> instances;
  int original_width = 0;
  int original_height = 0;
  DeviceType execution_device = DeviceType::Cpu;
  int execution_device_id = 0;
  std::string protocol;
};

}  // namespace vie
