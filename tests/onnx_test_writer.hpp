#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Minimal protobuf writer for the ONNX messages the analyzer tests build.
// It is not a general protobuf implementation and is not part of the library.
class OnnxTestWriter {
 public:
  void varint(std::uint32_t field, std::uint64_t value) {
    key(field, 0);
    append_varint(data_, value);
  }

  void i64(std::uint32_t field, std::int64_t value) {
    varint(field, static_cast<std::uint64_t>(value));
  }

  void text(std::uint32_t field, std::string_view value) { bytes(field, value); }

  void bytes(std::uint32_t field, std::string_view value) {
    key(field, 2);
    append_varint(data_, value.size());
    data_.append(value.data(), value.size());
  }

  void message(std::uint32_t field, const OnnxTestWriter& child) { bytes(field, child.data_); }

  void packed_i64(std::uint32_t field, const std::vector<std::int64_t>& values) {
    std::string packed;
    for (const std::int64_t value : values) append_varint(packed, static_cast<std::uint64_t>(value));
    bytes(field, packed);
  }

  const std::string& str() const { return data_; }

 private:
  std::string data_;

  void key(std::uint32_t field, std::uint32_t wire) {
    append_varint(data_, (static_cast<std::uint64_t>(field) << 3) | wire);
  }

  static void append_varint(std::string& out, std::uint64_t value) {
    while (value >= 0x80u) {
      out.push_back(static_cast<char>((value & 0x7fu) | 0x80u));
      value >>= 7;
    }
    out.push_back(static_cast<char>(value));
  }
};

enum class TestDimKind { Static, Symbolic, Unknown };

struct TestDim {
  TestDimKind kind = TestDimKind::Static;
  std::int64_t value = 0;
  std::string symbol;
  std::string denotation;
};

inline TestDim static_dim(std::int64_t value, std::string denotation = {}) {
  return TestDim{TestDimKind::Static, value, {}, std::move(denotation)};
}

inline TestDim symbol_dim(std::string symbol) {
  return TestDim{TestDimKind::Symbolic, 0, std::move(symbol), {}};
}

inline TestDim unknown_dim() { return TestDim{TestDimKind::Unknown, 0, {}, {}}; }

inline OnnxTestWriter dim_message(const TestDim& dimension) {
  OnnxTestWriter message;
  if (dimension.kind == TestDimKind::Static) message.i64(1, dimension.value);
  if (dimension.kind == TestDimKind::Symbolic) message.text(2, dimension.symbol);
  if (!dimension.denotation.empty()) message.text(3, dimension.denotation);
  return message;
}

inline OnnxTestWriter shape_message(const std::vector<TestDim>& dimensions) {
  OnnxTestWriter message;
  for (const TestDim& dimension : dimensions) message.message(1, dim_message(dimension));
  return message;
}

inline OnnxTestWriter tensor_type(int element_type, bool rank_known, const std::vector<TestDim>& dimensions) {
  OnnxTestWriter tensor;
  tensor.i64(1, element_type);
  if (rank_known) tensor.message(2, shape_message(dimensions));
  OnnxTestWriter type;
  type.message(1, tensor);
  return type;
}

inline OnnxTestWriter sequence_of_float() {
  OnnxTestWriter tensor;
  tensor.i64(1, 1);
  OnnxTestWriter inner;
  inner.message(1, tensor);
  OnnxTestWriter sequence;
  sequence.message(1, inner);
  OnnxTestWriter type;
  type.message(4, sequence);
  return type;
}

inline OnnxTestWriter value_info(std::string_view name, const OnnxTestWriter& type) {
  OnnxTestWriter message;
  message.text(1, name);
  message.message(2, type);
  return message;
}

inline OnnxTestWriter opset_import(std::string_view raw_domain, std::int64_t version) {
  OnnxTestWriter message;
  if (!raw_domain.empty()) message.text(1, raw_domain);
  message.i64(2, version);
  return message;
}

inline OnnxTestWriter string_entry(std::string_view key, std::string_view value) {
  OnnxTestWriter message;
  message.text(1, key);
  message.text(2, value);
  return message;
}

inline OnnxTestWriter node_message(std::string_view op_type, const std::vector<std::string>& inputs,
                                   const std::vector<std::string>& outputs,
                                   const std::vector<OnnxTestWriter>& attributes = {},
                                   std::string_view raw_domain = {}, std::string_view name = {}) {
  OnnxTestWriter message;
  for (const std::string& input : inputs) message.text(1, input);
  for (const std::string& output : outputs) message.text(2, output);
  if (!name.empty()) message.text(3, name);
  message.text(4, op_type);
  for (const OnnxTestWriter& attribute : attributes) message.message(5, attribute);
  if (!raw_domain.empty()) message.text(7, raw_domain);
  return message;
}

inline OnnxTestWriter attr_int(std::string_view name, std::int64_t value) {
  OnnxTestWriter message;
  message.text(1, name);
  message.i64(3, value);
  return message;
}

inline OnnxTestWriter attr_graph(std::string_view name, const OnnxTestWriter& graph) {
  OnnxTestWriter message;
  message.text(1, name);
  message.message(6, graph);
  return message;
}

inline OnnxTestWriter tensor_message(std::string_view name, int data_type, const std::vector<std::int64_t>& dims,
                                     const std::vector<std::int64_t>& int64_values, std::string_view raw,
                                     bool unpacked_dims) {
  OnnxTestWriter message;
  if (unpacked_dims) {
    for (const std::int64_t dim : dims) message.i64(1, dim);
  } else if (!dims.empty()) {
    message.packed_i64(1, dims);
  }
  message.i64(2, data_type);
  for (const std::int64_t value : int64_values) message.i64(7, value);
  message.text(8, name);
  if (!raw.empty()) message.bytes(9, raw);
  return message;
}
