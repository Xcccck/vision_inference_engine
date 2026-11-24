#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace vie {

enum class ErrorCode {
  InvalidArgument,
  UnresolvedField,
  SchemaError,
  Unsupported,
  Internal,
  // Phase 9.2.1: Async request errors
  RuntimeNotStarted,
  RuntimeShuttingDown,
  RequestQueueFull,
  InvalidModelId,
};

// Project-wide error policy: callers handle VieError. Allocation failure
// still surfaces as std::bad_alloc.
class VieError : public std::runtime_error {
 public:
  VieError(ErrorCode code, std::string message)
      : std::runtime_error(std::move(message)), code_(code) {}

  ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

inline const char* to_string(ErrorCode code) {
  switch (code) {
    case ErrorCode::InvalidArgument:
      return "invalid_argument";
    case ErrorCode::UnresolvedField:
      return "unresolved_field";
    case ErrorCode::SchemaError:
      return "schema_error";
    case ErrorCode::Unsupported:
      return "unsupported";
    case ErrorCode::Internal:
      return "internal";
    case ErrorCode::RuntimeNotStarted:
      return "runtime_not_started";
    case ErrorCode::RuntimeShuttingDown:
      return "runtime_shutting_down";
    case ErrorCode::RequestQueueFull:
      return "request_queue_full";
    case ErrorCode::InvalidModelId:
      return "invalid_model_id";
  }
  return "internal";
}

}  // namespace vie
