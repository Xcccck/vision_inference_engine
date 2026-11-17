#pragma once

#include "vie/decoder/model_decoder.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace vie {

class DecoderRegistry {
 public:
  using Factory = std::function<std::unique_ptr<IModelDecoder>()>;

  void register_factory(std::string protocol, Factory factory);
  bool contains(const std::string& protocol) const;
  std::unique_ptr<IModelDecoder> create(const std::string& protocol) const;

 private:
  mutable std::mutex mutex_;
  std::map<std::string, Factory> factories_;
};

}  // namespace vie
