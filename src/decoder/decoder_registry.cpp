#include "vie/decoder/decoder_registry.hpp"

#include <mutex>

namespace vie {

DetectionBatch IModelDecoder::decode_batch(const DecoderInput&, const DecoderConfig&, const DeviceContext&) const {
  throw VieError(ErrorCode::Unsupported, "this decoder does not produce detections");
}

SegmentationBatch IModelDecoder::decode_segmentation(const DecoderInput&, const DecoderConfig&,
                                                     const DeviceContext&) const {
  throw VieError(ErrorCode::Unsupported, "this decoder does not produce instance segmentation");
}

void DecoderRegistry::register_factory(std::string protocol, Factory factory) {
  if (protocol.empty()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder protocol is empty");
  }
  if (!factory) {
    throw VieError(ErrorCode::InvalidArgument, "decoder factory is empty");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (factories_.find(protocol) != factories_.end()) {
    throw VieError(ErrorCode::InvalidArgument, "decoder protocol is already registered: " + protocol);
  }
  factories_.emplace(std::move(protocol), std::move(factory));
}

bool DecoderRegistry::contains(const std::string& protocol) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return factories_.find(protocol) != factories_.end();
}

std::unique_ptr<IModelDecoder> DecoderRegistry::create(const std::string& protocol) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = factories_.find(protocol);
  if (found == factories_.end()) {
    throw VieError(ErrorCode::Unsupported, "no decoder is registered for protocol '" + protocol + "'");
  }
  return found->second();
}

}  // namespace vie
