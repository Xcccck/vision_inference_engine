#include "vie/analyzer/onnx_analyzer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <utility>

namespace vie {
namespace {

[[noreturn]] void malformed(const std::string& source, const std::string& message) {
  throw VieError(ErrorCode::SchemaError, "malformed ONNX (" + source + "): " + message);
}

class Cursor {
 public:
  Cursor(std::string_view bytes, std::string source)
      : begin_(bytes.data()), p_(bytes.data()), end_(bytes.data() + bytes.size()), source_(std::move(source)) {}

  bool done() const { return p_ >= end_; }

  std::uint64_t varint() {
    std::uint64_t result = 0;
    for (int shift = 0; shift <= 63; shift += 7) {
      if (p_ >= end_) malformed(source_, "truncated varint");
      const auto byte = static_cast<unsigned char>(*p_++);
      if (shift == 63 && byte > 1) malformed(source_, "varint is too wide");
      result |= std::uint64_t(byte & 0x7fu) << shift;
      if ((byte & 0x80u) == 0) return result;
    }
    malformed(source_, "varint is too wide");
  }

  std::string_view take(std::size_t size) {
    if (static_cast<std::size_t>(end_ - p_) < size) malformed(source_, "truncated length-delimited field");
    std::string_view view(p_, size);
    p_ += size;
    return view;
  }

  Cursor slice(std::size_t size) { return Cursor(take(size), source_); }

  const std::string& source() const { return source_; }

 private:
  const char* begin_;
  const char* p_;
  const char* end_;
  std::string source_;
};

struct Tag {
  int field = 0;
  int wire = 0;
  std::uint64_t varint = 0;
  std::string_view payload;
};

constexpr std::uint64_t kMaxFieldNumber = 536870911ull;
constexpr std::size_t kMaxTensorRank = 32;
constexpr std::size_t kMaxAttributeInts = 64;
constexpr std::size_t kMaxStoredInt64 = 256;

bool next_tag(Cursor& cursor, Tag& tag) {
  if (cursor.done()) return false;
  const std::uint64_t key = cursor.varint();
  const std::uint64_t field_number = key >> 3;
  if (field_number == 0 || field_number > kMaxFieldNumber) {
    malformed(cursor.source(), "field number is outside 1..536870911");
  }
  tag.field = static_cast<int>(field_number);
  tag.wire = static_cast<int>(key & 7u);
  if (tag.wire == 0) {
    tag.varint = cursor.varint();
  } else if (tag.wire == 1) {
    tag.payload = cursor.take(8);
  } else if (tag.wire == 2) {
    const std::uint64_t length = cursor.varint();
    if (length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      malformed(cursor.source(), "length is too large");
    }
    tag.payload = cursor.take(static_cast<std::size_t>(length));
  } else if (tag.wire == 5) {
    tag.payload = cursor.take(4);
  } else {
    malformed(cursor.source(), "unsupported protobuf wire type");
  }
  return true;
}

std::int64_t as_int64(std::uint64_t value) { return static_cast<std::int64_t>(value); }

std::vector<std::int64_t> packed_int64s(std::string_view payload, const std::string& source,
                                        std::size_t max_count) {
  Cursor cursor(payload, source);
  std::vector<std::int64_t> values;
  while (!cursor.done()) {
    if (values.size() >= max_count) {
      malformed(source, "packed repeated field exceeds " + std::to_string(max_count) + " values");
    }
    values.push_back(as_int64(cursor.varint()));
  }
  return values;
}

std::size_t count_packed_varints(std::string_view payload, const std::string& source) {
  Cursor cursor(payload, source);
  std::size_t count = 0;
  while (!cursor.done()) {
    cursor.varint();
    ++count;
  }
  return count;
}

std::uint64_t static_numel(const std::vector<std::int64_t>& dims) {
  std::uint64_t count = 1;
  for (const std::int64_t dim : dims) {
    if (dim < 0) return std::numeric_limits<std::uint64_t>::max();
    if (dim == 0) return 0;
    const auto factor = static_cast<std::uint64_t>(dim);
    if (count > std::numeric_limits<std::uint64_t>::max() / factor) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    count *= factor;
  }
  return count;
}

TensorShape shape_from_static_dims(const std::vector<std::int64_t>& dims) {
  TensorShape shape;
  shape.rank_known = true;
  shape.dimensions.reserve(dims.size());
  for (const std::int64_t dim : dims) {
    ShapeDimension dimension;
    dimension.kind = DimensionKind::Static;
    dimension.value = dim;
    shape.dimensions.push_back(std::move(dimension));
  }
  return shape;
}

enum class ShapeRelation { Equal, Compatible, Conflict, Incomparable };

ShapeRelation relate(const TensorShape& left, const TensorShape& right);

struct ParsedGraph;

struct ParsedAttribute {
  std::string name;
  bool has_int = false;
  std::int64_t int_value = 0;
  std::vector<std::int64_t> ints;
  std::vector<std::int64_t> tensor_int64;
  bool tensor_int64_ready = false;
  bool has_float = false;
  float float_value = 0.f;
  bool tensor_f32_ready = false;
  double tensor_f32 = 0;
  std::vector<std::unique_ptr<ParsedGraph>> graphs;

  ParsedAttribute();
  ~ParsedAttribute();
  ParsedAttribute(const ParsedAttribute&) = delete;
  ParsedAttribute& operator=(const ParsedAttribute&) = delete;
  ParsedAttribute(ParsedAttribute&&) noexcept = default;
};

struct ParsedNode {
  OnnxNode node;
  std::vector<ParsedAttribute> attributes;
};

struct ParsedGraph {
  std::string path;
  std::vector<ParsedNode> nodes;
  std::vector<std::string> outputs;
  std::vector<InitializerInfo> initializers;
  std::map<std::string, std::vector<std::int64_t>> int64_constants;
  std::map<std::string, double> float_scalars;
};

ParsedAttribute::ParsedAttribute() = default;
ParsedAttribute::~ParsedAttribute() = default;

struct ParsedTensor {
  InitializerInfo info;
  std::vector<std::int64_t> int64_values;
  bool int64_ready = false;
  bool scalar_ready = false;
  double scalar_value = 0;
};

float decode_f32(std::string_view bytes, const std::string& source) {
  if (bytes.size() != 4) malformed(source, "float32 payload is not 4 bytes");
  std::uint32_t bits = 0;
  for (int index = 0; index < 4; ++index) {
    bits |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[static_cast<std::size_t>(index)]))
            << (8 * index);
  }
  float value = 0.f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

double decode_f64(std::string_view bytes, const std::string& source) {
  if (bytes.size() != 8) malformed(source, "float64 payload is not 8 bytes");
  std::uint64_t bits = 0;
  for (int index = 0; index < 8; ++index) {
    bits |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[static_cast<std::size_t>(index)]))
            << (8 * index);
  }
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool contained_external_file(const std::filesystem::path& base_dir, const std::string& location,
                             std::filesystem::path& resolved, std::string& rejection) {
  if (location.empty() || location.find('\0') != std::string::npos) {
    rejection = "location is empty";
    return false;
  }
  const std::filesystem::path raw(location);
  if (raw.is_absolute() || raw.has_root_name() || raw.has_root_directory()) {
    rejection = "location is absolute";
    return false;
  }
  const std::filesystem::path normal = raw.lexically_normal();
  for (const std::filesystem::path& part : normal) {
    if (part == "..") {
      rejection = "location escapes the model directory";
      return false;
    }
  }
  std::error_code error;
  const std::filesystem::path base = std::filesystem::weakly_canonical(base_dir, error);
  if (error) {
    rejection = "model directory cannot be resolved";
    return false;
  }
  const std::filesystem::path candidate = std::filesystem::weakly_canonical(base / normal, error);
  if (error) {
    rejection = "location cannot be resolved";
    return false;
  }
  const std::filesystem::path relative = candidate.lexically_relative(base);
  if (relative.empty() || *relative.begin() == "..") {
    rejection = "location escapes the model directory";
    return false;
  }
  resolved = candidate;
  return true;
}

class Parser {
 public:
  Parser(std::string_view bytes, std::string source, std::filesystem::path external_dir)
      : cursor_(bytes, std::move(source)), external_dir_(std::move(external_dir)) {}

  ModelAnalysisReport parse() {
    Tag tag;
    while (next_tag(cursor_, tag)) {
      if (tag.wire != 0 && tag.wire != 2) continue;
      switch (tag.field) {
        case 1:
          if (tag.wire == 0) {
            report_.ir_version = as_int64(tag.varint);
            report_.ir_version_present = true;
          }
          break;
        case 2:
          if (tag.wire == 2) report_.producer_name = std::string(tag.payload);
          break;
        case 3:
          if (tag.wire == 2) report_.producer_version = std::string(tag.payload);
          break;
        case 4:
          if (tag.wire == 2) report_.model_domain = std::string(tag.payload);
          break;
        case 5:
          if (tag.wire == 0) report_.model_version = as_int64(tag.varint);
          break;
        case 6:
          if (tag.wire == 2) report_.doc_string = std::string(tag.payload);
          break;
        case 7:
          if (tag.wire == 2) main_ = parse_graph(tag.payload, "graph", 0);
          break;
        case 8:
          if (tag.wire == 2) report_.opsets.push_back(parse_opset(tag.payload));
          break;
        case 14:
          if (tag.wire == 2) report_.metadata.push_back(parse_entry(tag.payload));
          break;
        case 20:
          ++report_.training_info_count;
          break;
        case 25:
          if (tag.wire == 2) parse_function(tag.payload);
          break;
        default:
          break;
      }
    }
    finish();
    return report_;
  }

  const ParsedGraph& main_graph() const { return main_; }

 private:
  Cursor cursor_;
  std::filesystem::path external_dir_;
  ModelAnalysisReport report_;
  ParsedGraph main_;

  Cursor nested(std::string_view bytes) const { return Cursor(bytes, cursor_.source()); }

  std::pair<std::string, std::string> parse_entry(std::string_view bytes) const {
    Cursor cursor = nested(bytes);
    std::string key;
    std::string value;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.wire == 2 && tag.field == 1) key = std::string(tag.payload);
      if (tag.wire == 2 && tag.field == 2) value = std::string(tag.payload);
    }
    return {key, value};
  }

  OnnxOpset parse_opset(std::string_view bytes) {
    Cursor cursor = nested(bytes);
    OnnxOpset opset;
    bool saw_version = false;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 2) opset.raw_domain = std::string(tag.payload);
      if (tag.field == 2 && tag.wire == 0) {
        opset.version = as_int64(tag.varint);
        saw_version = true;
      }
    }
    opset.domain = normalize_onnx_domain(opset.raw_domain);
    if (!saw_version) {
      report_.warnings.push_back(
          {"opset_version_missing", "an opset import omitted its version for domain " + opset.domain});
    }
    return opset;
  }

  ShapeDimension parse_dimension(std::string_view bytes) const {
    Cursor cursor = nested(bytes);
    bool saw_value = false;
    bool saw_param = false;
    ShapeDimension dimension;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 0) {
        saw_value = true;
        dimension.value = as_int64(tag.varint);
      } else if (tag.field == 2 && tag.wire == 2) {
        saw_param = true;
        dimension.symbol = std::string(tag.payload);
      } else if (tag.field == 3 && tag.wire == 2) {
        dimension.denotation = std::string(tag.payload);
      }
    }
    if (saw_value && saw_param) malformed(cursor_.source(), "dimension sets both dim_value and dim_param");
    if (saw_value) dimension.kind = DimensionKind::Static;
    else if (saw_param) dimension.kind = DimensionKind::Symbolic;
    else dimension.kind = DimensionKind::Unknown;
    return dimension;
  }

  TensorShape parse_shape(std::string_view bytes) const {
    Cursor cursor = nested(bytes);
    TensorShape shape;
    shape.rank_known = true;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 2) shape.dimensions.push_back(parse_dimension(tag.payload));
    }
    return shape;
  }

  void parse_tensor_body(std::string_view bytes, OnnxElementType& element, TensorShape& shape) const {
    Cursor cursor = nested(bytes);
    bool saw_shape = false;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 0) element = onnx_element_type(static_cast<int>(as_int64(tag.varint)));
      if (tag.field == 2 && tag.wire == 2) {
        shape = parse_shape(tag.payload);
        saw_shape = true;
      }
    }
    if (!saw_shape) {
      shape.rank_known = false;
      shape.dimensions.clear();
    }
  }

  OnnxType parse_type(std::string_view bytes, int depth) {
    Cursor cursor = nested(bytes);
    OnnxType type;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.wire != 2 || tag.field == 6) continue;
      if (tag.field == 1) {
        type.kind = OnnxTypeKind::Tensor;
        parse_tensor_body(tag.payload, type.element_type, type.declared_shape);
      } else if (tag.field == 8) {
        type.kind = OnnxTypeKind::SparseTensor;
        parse_tensor_body(tag.payload, type.element_type, type.declared_shape);
      } else if (tag.field == 7) {
        type.kind = OnnxTypeKind::Opaque;
      } else if (tag.field == 4 || tag.field == 9) {
        type.kind = tag.field == 4 ? OnnxTypeKind::Sequence : OnnxTypeKind::Optional;
        absorb_wrapper(tag.payload, type, depth);
      } else if (tag.field == 5) {
        type.kind = OnnxTypeKind::Map;
        absorb_map(tag.payload, type, depth);
      }
    }
    if (type.kind == OnnxTypeKind::Absent) type.kind = OnnxTypeKind::Unknown;
    note_element(type.element_type);
    return type;
  }

  void absorb_wrapper(std::string_view bytes, OnnxType& type, int depth) {
    Cursor cursor = nested(bytes);
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 2) {
        if (depth >= 1) {
          type.nested_truncated = true;
          type.nested_kind = OnnxTypeKind::Unknown;
          return;
        }
        const OnnxType inner = parse_type(tag.payload, depth + 1);
        type.nested_kind = inner.kind;
        type.nested_element_type = inner.element_type;
        type.nested_declared_shape = inner.declared_shape;
        type.nested_truncated = inner.nested_truncated || inner.kind == OnnxTypeKind::Sequence ||
                                inner.kind == OnnxTypeKind::Map || inner.kind == OnnxTypeKind::Optional;
      }
    }
  }

  void absorb_map(std::string_view bytes, OnnxType& type, int depth) {
    Cursor cursor = nested(bytes);
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 0) {
        type.map_key_type = onnx_element_type(static_cast<int>(as_int64(tag.varint)));
      } else if (tag.field == 2 && tag.wire == 2) {
        if (depth >= 1) {
          type.nested_truncated = true;
          return;
        }
        const OnnxType inner = parse_type(tag.payload, depth + 1);
        type.nested_kind = inner.kind;
        type.nested_element_type = inner.element_type;
        type.nested_declared_shape = inner.declared_shape;
      }
    }
  }

  void note_element(const OnnxElementType& element) {
    if (element.code == 0) return;
    if (element.name.rfind("onnx_dtype_", 0) == 0) {
      report_.warnings.push_back(
          {"unrecognized_element_type", "unrecognized ONNX element type code " + std::to_string(element.code)});
    }
  }

  ValueEndpoint parse_value(std::string_view bytes) {
    Cursor cursor = nested(bytes);
    ValueEndpoint endpoint;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.wire != 2) continue;
      if (tag.field == 1) endpoint.name = std::string(tag.payload);
      else if (tag.field == 2) endpoint.type = parse_type(tag.payload, 0);
      else if (tag.field == 3) endpoint.doc_string = std::string(tag.payload);
    }
    return endpoint;
  }

  ParsedTensor parse_tensor(std::string_view bytes) {
    Cursor cursor = nested(bytes);
    ParsedTensor parsed;
    std::vector<std::int64_t> dims;
    int data_type = 0;
    std::uint64_t typed_count = 0;
    std::size_t int64_count = 0;
    bool saw_typed = false;
    bool saw_raw = false;
    std::string raw;
    bool saw_location = false;
    int location = 0;
    int f32_seen = 0;
    float f32_first = 0.f;
    const auto note_f32 = [&](float value) {
      if (f32_seen == 0) f32_first = value;
      ++f32_seen;
    };
    const auto push_dim = [&](std::int64_t dim) {
      if (dims.size() >= kMaxTensorRank) malformed(cursor_.source(), "tensor rank exceeds 32");
      dims.push_back(dim);
    };
    const auto remember_int64 = [&](std::int64_t value) {
      ++int64_count;
      if (parsed.int64_values.size() < kMaxStoredInt64) parsed.int64_values.push_back(value);
    };
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1) {
        if (tag.wire == 0) push_dim(as_int64(tag.varint));
        else if (tag.wire == 2) {
          for (const std::int64_t dim : packed_int64s(tag.payload, cursor_.source(), kMaxTensorRank)) {
            push_dim(dim);
          }
        }
      } else if (tag.field == 2 && tag.wire == 0) {
        data_type = static_cast<int>(as_int64(tag.varint));
      } else if (tag.field == 4) {
        saw_typed = true;
        if (tag.wire == 5) {
          ++typed_count;
          note_f32(decode_f32(tag.payload, cursor_.source()));
        } else if (tag.wire == 2) {
          if (tag.payload.size() % 4 != 0) malformed(cursor_.source(), "float_data is not aligned");
          typed_count += tag.payload.size() / 4;
          for (std::size_t offset = 0; offset < tag.payload.size(); offset += 4) {
            note_f32(decode_f32(tag.payload.substr(offset, 4), cursor_.source()));
          }
        }
      } else if (tag.field == 5) {
        saw_typed = true;
        if (tag.wire == 0) ++typed_count;
        else if (tag.wire == 2) typed_count += count_packed_varints(tag.payload, cursor_.source());
      } else if (tag.field == 7) {
        saw_typed = true;
        if (tag.wire == 0) {
          remember_int64(as_int64(tag.varint));
        } else if (tag.wire == 2) {
          Cursor packed(tag.payload, cursor_.source());
          while (!packed.done()) remember_int64(as_int64(packed.varint()));
        }
      } else if (tag.field == 8 && tag.wire == 2) {
        parsed.info.name = std::string(tag.payload);
      } else if (tag.field == 9 && tag.wire == 2) {
        saw_raw = true;
        parsed.info.payload_bytes = tag.payload.size();
        if (tag.payload.size() <= kMaxStoredInt64 * 8) raw = std::string(tag.payload);
      } else if (tag.field == 10 && tag.wire == 2) {
        saw_typed = true;
        if (tag.payload.size() % 8 != 0) malformed(cursor_.source(), "double_data is not aligned");
        typed_count += tag.payload.size() / 8;
      } else if (tag.field == 11) {
        saw_typed = true;
        if (tag.wire == 0) ++typed_count;
        else if (tag.wire == 1) ++typed_count;
        else if (tag.wire == 2) typed_count += tag.payload.size() / 8;
      } else if (tag.field == 13 && tag.wire == 2) {
        const auto entry = parse_entry(tag.payload);
        parsed.info.external_data.push_back(ExternalDataEntry{entry.first, entry.second});
        if (entry.first == "location") parsed.info.external_location = entry.second;
      } else if (tag.field == 14 && tag.wire == 0) {
        saw_location = true;
        location = static_cast<int>(as_int64(tag.varint));
      }
    }
    parsed.info.element_type = onnx_element_type(data_type);
    parsed.info.declared_shape = shape_from_static_dims(dims);
    note_element(parsed.info.element_type);
    if (saw_location && location == 1) {
      parsed.info.payload = TensorPayloadKind::External;
    } else if (saw_raw) {
      parsed.info.payload = TensorPayloadKind::Raw;
    } else if (saw_typed || int64_count > 0) {
      parsed.info.payload = TensorPayloadKind::TypedElements;
      parsed.info.typed_element_count = int64_count > 0 ? int64_count : typed_count;
    }
    if (data_type == 7 && int64_count == 0 && saw_raw && raw.size() % 8 == 0 &&
        raw.size() / 8 <= kMaxStoredInt64) {
      for (std::size_t offset = 0; offset < raw.size(); offset += 8) {
        std::uint64_t value = 0;
        for (int byte = 0; byte < 8; ++byte) {
          value |= std::uint64_t(static_cast<unsigned char>(raw[offset + static_cast<std::size_t>(byte)]))
                   << (8 * byte);
        }
        parsed.int64_values.push_back(as_int64(value));
      }
      int64_count = parsed.int64_values.size();
    }
    const std::uint64_t count = static_numel(dims);
    parsed.int64_ready = data_type == 7 && count != std::numeric_limits<std::uint64_t>::max() &&
                         parsed.int64_values.size() == static_cast<std::size_t>(count);
    if (count == 1 && data_type == 1 && !saw_raw && f32_seen == 1) {
      parsed.scalar_ready = std::isfinite(f32_first);
      parsed.scalar_value = f32_first;
    } else if (count == 1 && data_type == 1 && f32_seen == 0 && saw_raw && raw.size() == 4) {
      const float value = decode_f32(raw, cursor_.source());
      parsed.scalar_ready = std::isfinite(value);
      parsed.scalar_value = value;
    } else if (count == 1 && data_type == 11 && f32_seen == 0 && saw_raw && raw.size() == 8) {
      const double value = decode_f64(raw, cursor_.source());
      parsed.scalar_ready = std::isfinite(value);
      parsed.scalar_value = value;
    }
    return parsed;
  }

  ParsedAttribute parse_attribute(std::string_view bytes, int depth) {
    Cursor cursor = nested(bytes);
    ParsedAttribute attribute;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 2) attribute.name = std::string(tag.payload);
      else if (tag.field == 2 && tag.wire == 5) {
        attribute.has_float = true;
        attribute.float_value = decode_f32(tag.payload, cursor_.source());
      } else if (tag.field == 3 && tag.wire == 0) {
        attribute.has_int = true;
        attribute.int_value = as_int64(tag.varint);
      } else if (tag.field == 5 && tag.wire == 2) {
        const ParsedTensor tensor = parse_tensor(tag.payload);
        if (tensor.int64_ready && !tensor.int64_values.empty() && tensor.int64_values.size() <= 8) {
          attribute.tensor_int64 = tensor.int64_values;
          attribute.tensor_int64_ready = true;
        }
        if (tensor.scalar_ready) {
          attribute.tensor_f32_ready = true;
          attribute.tensor_f32 = tensor.scalar_value;
        }
      } else if (tag.field == 6 && tag.wire == 2) {
        if (depth >= 16) malformed(cursor_.source(), "subgraph nesting exceeds 16");
        attribute.graphs.push_back(std::make_unique<ParsedGraph>(parse_graph(tag.payload, "", depth + 1)));
      } else if (tag.field == 8) {
        const auto push_int = [&](std::int64_t value) {
          if (attribute.ints.size() >= kMaxAttributeInts) {
            malformed(cursor_.source(), "attribute ints exceed 64 values");
          }
          attribute.ints.push_back(value);
        };
        if (tag.wire == 0) push_int(as_int64(tag.varint));
        else if (tag.wire == 2) {
          for (const std::int64_t value : packed_int64s(tag.payload, cursor_.source(), kMaxAttributeInts)) {
            push_int(value);
          }
        }
      } else if (tag.field == 11 && tag.wire == 2) {
        if (depth >= 16) malformed(cursor_.source(), "subgraph nesting exceeds 16");
        attribute.graphs.push_back(std::make_unique<ParsedGraph>(parse_graph(tag.payload, "", depth + 1)));
      }
    }
    return attribute;
  }

  ParsedNode parse_node(std::string_view bytes, int depth) {
    Cursor cursor = nested(bytes);
    ParsedNode node;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 2) node.node.inputs.emplace_back(tag.payload);
      else if (tag.field == 2 && tag.wire == 2) node.node.outputs.emplace_back(tag.payload);
      else if (tag.field == 3 && tag.wire == 2) node.node.name = std::string(tag.payload);
      else if (tag.field == 4 && tag.wire == 2) node.node.op_type = std::string(tag.payload);
      else if (tag.field == 5 && tag.wire == 2) node.attributes.push_back(parse_attribute(tag.payload, depth));
      else if (tag.field == 7 && tag.wire == 2) node.node.raw_domain = std::string(tag.payload);
    }
    node.node.domain = normalize_onnx_domain(node.node.raw_domain);
    node.node.depth = depth;
    return node;
  }

  ParsedGraph parse_graph(std::string_view bytes, const std::string& fallback_path, int depth) {
    Cursor cursor = nested(bytes);
    ParsedGraph graph;
    graph.path = fallback_path;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 2) graph.nodes.push_back(parse_node(tag.payload, depth));
      else if (tag.field == 2 && tag.wire == 2) graph.path = std::string(tag.payload);
      else if (tag.field == 5 && tag.wire == 2) {
        const ParsedTensor tensor = parse_tensor(tag.payload);
        graph.initializers.push_back(tensor.info);
        if (tensor.int64_ready) graph.int64_constants[tensor.info.name] = tensor.int64_values;
        if (tensor.scalar_ready && !tensor.info.name.empty()) {
          graph.float_scalars.emplace(tensor.info.name, tensor.scalar_value);
        }
      } else if (tag.field == 11 && tag.wire == 2) {
        if (depth == 0) report_.graph_inputs.push_back(parse_value(tag.payload));
      } else if (tag.field == 12 && tag.wire == 2) {
        if (depth == 0 && fallback_path == "graph") {
          report_.outputs.push_back(parse_value(tag.payload));
        } else {
          graph.outputs.push_back(parse_value(tag.payload).name);
        }
      } else if (tag.field == 13 && tag.wire == 2 && depth == 0) {
        const ValueEndpoint value = parse_value(tag.payload);
        report_.value_infos.push_back(RecordedValueInfo{value.name, value.doc_string, value.type});
      } else if (tag.field == 14 && tag.wire == 2 && depth == 0) {
        ++report_.quantization_annotation_count;
      } else if (tag.field == 15 && tag.wire == 2) {
        ParsedTensor sparse = parse_sparse(tag.payload);
        if (depth == 0) report_.sparse_initializers.push_back(std::move(sparse.info));
        else graph.initializers.push_back(std::move(sparse.info));
      }
    }
    if (depth == 0 && fallback_path == "graph") {
      report_.graph_name = graph.path == "graph" ? std::string() : graph.path;
      if (!report_.graph_name.empty()) graph.path = report_.graph_name;
      else graph.path = "graph";
      report_.initializers = graph.initializers;
      for (const auto& entry : graph.float_scalars) {
        report_.float_constants.push_back(FloatConstant{entry.first, entry.second});
      }
    }
    return graph;
  }

  ParsedTensor parse_sparse(std::string_view bytes) {
    Cursor cursor = nested(bytes);
    ParsedTensor values;
    std::vector<std::int64_t> dims;
    bool saw_values = false;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.field == 1 && tag.wire == 2) {
        values = parse_tensor(tag.payload);
        saw_values = true;
      } else if (tag.field == 3) {
        if (tag.wire == 0) {
          if (dims.size() >= kMaxTensorRank) malformed(cursor_.source(), "tensor rank exceeds 32");
          dims.push_back(as_int64(tag.varint));
        } else if (tag.wire == 2) {
          for (const std::int64_t dim : packed_int64s(tag.payload, cursor_.source(), kMaxTensorRank)) {
            if (dims.size() >= kMaxTensorRank) malformed(cursor_.source(), "tensor rank exceeds 32");
            dims.push_back(dim);
          }
        }
      }
    }
    if (!saw_values) values.info.element_type = onnx_element_type(0);
    values.info.sparse = true;
    values.info.declared_shape = shape_from_static_dims(dims);
    values.int64_ready = false;
    return values;
  }

  void parse_function(std::string_view bytes) {
    Cursor cursor = nested(bytes);
    FunctionSummary summary;
    ParsedGraph body;
    Tag tag;
    while (next_tag(cursor, tag)) {
      if (tag.wire != 2 && !(tag.field == 10 && tag.wire == 2)) {
        if (tag.field == 1 && tag.wire == 2) summary.name = std::string(tag.payload);
        else if (tag.field == 7 && tag.wire == 2) body.nodes.push_back(parse_node(tag.payload, 0));
        else if (tag.field == 9 && tag.wire == 2) summary.opsets.push_back(parse_opset(tag.payload));
        else if (tag.field == 10 && tag.wire == 2) summary.raw_domain = std::string(tag.payload);
        else if (tag.field == 13 && tag.wire == 2) summary.overload = std::string(tag.payload);
        continue;
      }
      if (tag.field == 1) summary.name = std::string(tag.payload);
      else if (tag.field == 7) body.nodes.push_back(parse_node(tag.payload, 0));
      else if (tag.field == 9) summary.opsets.push_back(parse_opset(tag.payload));
      else if (tag.field == 10) summary.raw_domain = std::string(tag.payload);
      else if (tag.field == 13) summary.overload = std::string(tag.payload);
    }
    summary.domain = normalize_onnx_domain(summary.raw_domain);
    body.path = "function/" + summary.name;
    assign_paths(body, 0);
    const std::vector<OnnxOpset>& opsets = summary.opsets.empty() ? report_.opsets : summary.opsets;
    append_nodes(body, opsets);
    summary.node_count = static_cast<std::int64_t>(body.nodes.size());
    report_.functions.push_back(std::move(summary));
    report_.not_interpreted.push_back("function body shape propagation");
  }

  void finish() {
    if (!report_.ir_version_present) {
      report_.warnings.push_back({"missing_ir_version", "ModelProto.ir_version is absent"});
    }
    if (report_.opsets.empty()) {
      report_.warnings.push_back({"missing_opset", "ModelProto.opset_import is empty"});
    }
    if (report_.training_info_count > 0) {
      report_.not_interpreted.push_back("training_info");
    }
    if (report_.quantization_annotation_count > 0) {
      report_.not_interpreted.push_back("quantization_annotation");
    }
    if (main_.path.empty()) main_.path = "graph";
    assign_paths(main_, 0);
    append_nodes(main_, report_.opsets);
    link_initializers();
    resolve_external_data();
    count_operators();
    warn_duplicates();
  }

  void assign_paths(ParsedGraph& graph, int depth) const {
    int index = 0;
    for (ParsedNode& node : graph.nodes) {
      const std::string label = node.node.name.empty()
                                    ? node.node.op_type + "#" + std::to_string(index)
                                    : node.node.name;
      node.node.graph_path = graph.path;
      node.node.depth = depth;
      for (ParsedAttribute& attribute : node.attributes) {
        int graph_index = 0;
        for (const std::unique_ptr<ParsedGraph>& child : attribute.graphs) {
          if (!child) continue;
          const std::string suffix =
              attribute.name.empty() ? "graph" + std::to_string(graph_index) : attribute.name;
          child->path = graph.path + "/" + label + "/" + suffix;
          assign_paths(*child, depth + 1);
          ++graph_index;
        }
      }
      ++index;
    }
  }

  void append_nodes(const ParsedGraph& graph, const std::vector<OnnxOpset>& opsets) {
    for (const ParsedNode& node : graph.nodes) {
      OnnxNode published = node.node;
      for (const ParsedAttribute& attribute : node.attributes) {
        if (attribute.has_int) published.int_attributes.emplace_back(attribute.name, attribute.int_value);
      }
      if (published.op_type == "Constant" && published.outputs.size() == 1) {
        const std::string& produced = published.outputs.front();
        bool stored_tensor = false;
        for (const ParsedAttribute& attribute : node.attributes) {
          if (attribute.name == "value" && attribute.tensor_int64_ready) {
            report_.integer_constants.push_back(IntegerConstant{produced, attribute.tensor_int64});
            stored_tensor = true;
          }
        }
        if (!stored_tensor) {
          for (const ParsedAttribute& attribute : node.attributes) {
            if (attribute.has_int && (attribute.name == "value" || attribute.name == "value_int")) {
              report_.integer_constants.push_back(IntegerConstant{produced, {attribute.int_value}});
            }
          }
        }
        for (const ParsedAttribute& attribute : node.attributes) {
          if (attribute.name == "value" && attribute.tensor_f32_ready && std::isfinite(attribute.tensor_f32)) {
            report_.float_constants.push_back(FloatConstant{produced, attribute.tensor_f32});
          } else if (attribute.name == "value_float" && attribute.has_float &&
                     std::isfinite(attribute.float_value)) {
            report_.float_constants.push_back(FloatConstant{produced, attribute.float_value});
          }
        }
      }
      report_.nodes.push_back(std::move(published));
      if (node.node.op_type.empty()) {
        report_.warnings.push_back({"empty_op_type", "a node in " + graph.path + " has an empty op_type"});
      }
      if (!opsets.empty() && !domain_imported(opsets, node.node.domain)) {
        report_.warnings.push_back(
            {"node_domain_not_imported", "node '" + node.node.name + "' op " + node.node.op_type +
                                             " uses domain " + node.node.domain +
                                             " (raw \"" + node.node.raw_domain + "\"), which is not imported"});
      }
      for (const ParsedAttribute& attribute : node.attributes) {
        for (const std::unique_ptr<ParsedGraph>& child : attribute.graphs) {
          if (child) append_nodes(*child, opsets);
        }
      }
    }
  }

  static bool domain_imported(const std::vector<OnnxOpset>& opsets, const std::string& domain) {
    for (const OnnxOpset& opset : opsets) {
      if (opset.domain == domain) return true;
    }
    return false;
  }

  void link_initializers() {
    std::set<std::string> input_names;
    for (const ValueEndpoint& input : report_.graph_inputs) input_names.insert(input.name);
    std::set<std::string> initializer_names;
    auto link = [&](InitializerInfo& initializer) {
      initializer.listed_as_graph_input = input_names.count(initializer.name) > 0;
      initializer_names.insert(initializer.name);
      if (!initializer.listed_as_graph_input) return;
      for (const ValueEndpoint& input : report_.graph_inputs) {
        if (input.name != initializer.name) continue;
        if (input.type.kind != OnnxTypeKind::Tensor && input.type.kind != OnnxTypeKind::SparseTensor) break;
        if (relate(input.type.declared_shape, initializer.declared_shape) == ShapeRelation::Conflict) {
          report_.warnings.push_back(
              {"initializer_shape_disagrees_with_input",
               "initializer '" + initializer.name + "' dims " + format_tensor_shape(initializer.declared_shape) +
                   " disagree with graph input declared shape " +
                   format_tensor_shape(input.type.declared_shape)});
        }
      }
    };
    for (InitializerInfo& initializer : report_.initializers) link(initializer);
    for (InitializerInfo& initializer : report_.sparse_initializers) link(initializer);
    for (const ValueEndpoint& input : report_.graph_inputs) {
      if (!initializer_names.count(input.name)) report_.runtime_inputs.push_back(input);
    }
  }

  void resolve_external_data() {
    auto resolve = [&](InitializerInfo& initializer) {
      if (initializer.payload != TensorPayloadKind::External) return;
      if (initializer.external_location.empty()) {
        report_.warnings.push_back(
            {"external_data_without_location", "initializer '" + initializer.name + "' is external but has no location"});
        return;
      }
      std::filesystem::path full;
      std::string rejection;
      if (!contained_external_file(external_dir_, initializer.external_location, full, rejection)) {
        report_.warnings.push_back(
            {"external_data_path_rejected",
             "external data for '" + initializer.name + "' was rejected: " + rejection});
        return;
      }
      std::error_code error;
      if (!std::filesystem::exists(full, error) || error) {
        report_.warnings.push_back(
            {"external_data_missing", "external data for '" + initializer.name + "' was not found at " + full.string()});
        return;
      }
      if (!std::filesystem::is_regular_file(full, error) || error) {
        report_.warnings.push_back(
            {"external_data_not_a_file", "external data for '" + initializer.name + "' is not a regular file"});
        return;
      }
      const auto bytes = std::filesystem::file_size(full, error);
      if (error) {
        report_.warnings.push_back(
            {"external_data_missing", "external data for '" + initializer.name + "' could not be sized"});
        return;
      }
      initializer.external_file_found = true;
      initializer.external_file_bytes = static_cast<std::uint64_t>(bytes);
      std::uint64_t offset = 0;
      std::uint64_t length = 0;
      bool saw_offset = false;
      bool saw_length = false;
      bool offset_ok = true;
      bool length_ok = true;
      for (const ExternalDataEntry& entry : initializer.external_data) {
        if (entry.key == "offset") {
          saw_offset = true;
          offset_ok = parse_u64(entry.value, offset);
        } else if (entry.key == "length") {
          saw_length = true;
          length_ok = parse_u64(entry.value, length);
        } else if (entry.key == "checksum") {
          report_.not_interpreted.push_back("external data checksum for " + initializer.name);
        }
      }
      if ((saw_offset && !offset_ok) || (saw_length && !length_ok)) {
        report_.warnings.push_back(
            {"external_data_bad_range",
             "external data offset or length for '" + initializer.name + "' is not an unsigned integer"});
        return;
      }
      if (offset > initializer.external_file_bytes ||
          (saw_length && length > initializer.external_file_bytes - offset)) {
        report_.warnings.push_back(
            {"external_data_out_of_range",
             "external data range for '" + initializer.name + "' is outside the file"});
      }
    };
    for (InitializerInfo& initializer : report_.initializers) resolve(initializer);
    for (InitializerInfo& initializer : report_.sparse_initializers) resolve(initializer);
  }

  static bool parse_u64(const std::string& text, std::uint64_t& value) {
    if (text.empty()) return false;
    try {
      std::size_t used = 0;
      const unsigned long long parsed = std::stoull(text, &used, 10);
      if (used != text.size()) return false;
      value = static_cast<std::uint64_t>(parsed);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

  void count_operators() {
    std::map<std::pair<std::string, std::string>, std::int64_t> counts;
    for (const OnnxNode& node : report_.nodes) {
      if (node.op_type.empty()) continue;
      ++counts[{node.domain, node.op_type}];
    }
    for (const auto& entry : counts) {
      OperatorStat stat;
      stat.domain = entry.first.first;
      stat.op_type = entry.first.second;
      stat.count = entry.second;
      report_.operators.push_back(std::move(stat));
    }
  }

  void warn_duplicates() {
    std::set<std::string> names;
    for (std::size_t i = 0; i < report_.outputs.size(); ++i) {
      if (!names.insert(report_.outputs[i].name).second) {
        report_.warnings.push_back(
            {"duplicate_output_name", "duplicate output '" + report_.outputs[i].name + "'"});
      }
    }
    names.clear();
    for (const RecordedValueInfo& info : report_.value_infos) {
      if (!names.insert(info.name).second) {
        report_.warnings.push_back({"duplicate_value_info", "duplicate value_info '" + info.name + "'"});
      }
    }
  }
};

ShapeRelation relate(const TensorShape& left, const TensorShape& right) {
  if (!left.rank_known || !right.rank_known) return ShapeRelation::Incomparable;
  if (left.dimensions.size() != right.dimensions.size()) return ShapeRelation::Conflict;
  bool softer = false;
  for (std::size_t i = 0; i < left.dimensions.size(); ++i) {
    const ShapeDimension& a = left.dimensions[i];
    const ShapeDimension& b = right.dimensions[i];
    if (a.kind == DimensionKind::Unknown || b.kind == DimensionKind::Unknown) {
      softer = true;
      continue;
    }
    if (a.kind != b.kind) return ShapeRelation::Conflict;
    if (a.kind == DimensionKind::Static && a.value != b.value) return ShapeRelation::Conflict;
    if (a.kind == DimensionKind::Symbolic && a.symbol != b.symbol) return ShapeRelation::Conflict;
  }
  return softer ? ShapeRelation::Compatible : ShapeRelation::Equal;
}

ShapeDimension static_dim(std::int64_t value) {
  ShapeDimension dimension;
  dimension.kind = DimensionKind::Static;
  dimension.value = value;
  return dimension;
}

bool merge_dim(const ShapeDimension& left, const ShapeDimension& right, ShapeDimension& out) {
  if (left.kind == DimensionKind::Static && left.value == 1) {
    out = right;
    return true;
  }
  if (right.kind == DimensionKind::Static && right.value == 1) {
    out = left;
    return true;
  }
  if (left.kind == DimensionKind::Unknown) {
    out = right;
    return true;
  }
  if (right.kind == DimensionKind::Unknown) {
    out = left;
    return true;
  }
  if (left.kind == DimensionKind::Static && right.kind == DimensionKind::Static && left.value == right.value) {
    out = left;
    return true;
  }
  if (left.kind == DimensionKind::Symbolic && right.kind == DimensionKind::Symbolic && left.symbol == right.symbol) {
    out = left;
    return true;
  }
  return false;
}

bool broadcast_shapes(const TensorShape& left, const TensorShape& right, TensorShape& out) {
  if (!left.rank_known || !right.rank_known) return false;
  const std::size_t rank = std::max(left.dimensions.size(), right.dimensions.size());
  out.rank_known = true;
  out.dimensions.assign(rank, static_dim(1));
  for (std::size_t from_right = 0; from_right < rank; ++from_right) {
    const bool has_left = from_right < left.dimensions.size();
    const bool has_right = from_right < right.dimensions.size();
    const ShapeDimension a = has_left ? left.dimensions[left.dimensions.size() - 1 - from_right] : static_dim(1);
    const ShapeDimension b = has_right ? right.dimensions[right.dimensions.size() - 1 - from_right] : static_dim(1);
    ShapeDimension merged;
    if (!merge_dim(a, b, merged)) return false;
    out.dimensions[rank - 1 - from_right] = merged;
  }
  return true;
}

bool contracted_compatible(const ShapeDimension& left, const ShapeDimension& right) {
  if (left.kind == DimensionKind::Unknown || right.kind == DimensionKind::Unknown) return true;
  if (left.kind == DimensionKind::Static && right.kind == DimensionKind::Static) return left.value == right.value;
  if (left.kind == DimensionKind::Symbolic && right.kind == DimensionKind::Symbolic) return left.symbol == right.symbol;
  return false;
}

bool matmul_shapes(const TensorShape& left, const TensorShape& right, TensorShape& out) {
  if (!left.rank_known || !right.rank_known) return false;
  const std::size_t ar = left.dimensions.size();
  const std::size_t br = right.dimensions.size();
  if (ar == 1 && br == 1) {
    if (!contracted_compatible(left.dimensions[0], right.dimensions[0])) return false;
    out.rank_known = true;
    out.dimensions.clear();
    return true;
  }
  if (ar == 1 && br >= 2) {
    if (!contracted_compatible(left.dimensions[0], right.dimensions[br - 2])) return false;
    out = right;
    out.dimensions.erase(out.dimensions.end() - 2);
    return true;
  }
  if (br == 1 && ar >= 2) {
    if (!contracted_compatible(left.dimensions[ar - 1], right.dimensions[0])) return false;
    out = left;
    out.dimensions.pop_back();
    return true;
  }
  if (ar < 2 || br < 2) return false;
  if (!contracted_compatible(left.dimensions[ar - 1], right.dimensions[br - 2])) return false;
  TensorShape batch_left = left;
  TensorShape batch_right = right;
  batch_left.dimensions.resize(ar - 2);
  batch_right.dimensions.resize(br - 2);
  if (!broadcast_shapes(batch_left, batch_right, out)) return false;
  out.dimensions.push_back(left.dimensions[ar - 2]);
  out.dimensions.push_back(right.dimensions[br - 1]);
  return true;
}

bool preserves_shape(const std::string& op) {
  return op == "Abs" || op == "Acos" || op == "Asin" || op == "Atan" || op == "Cast" || op == "Ceil" ||
         op == "Clip" || op == "Cos" || op == "Cosh" || op == "Dropout" || op == "Elu" || op == "Erf" ||
         op == "Exp" || op == "Floor" || op == "Gelu" || op == "HardSigmoid" || op == "HardSwish" ||
         op == "Identity" || op == "IsInf" || op == "IsNaN" || op == "LeakyRelu" || op == "Log" ||
         op == "Mish" || op == "Neg" || op == "Not" || op == "Reciprocal" || op == "Relu" || op == "Round" ||
         op == "Selu" || op == "Sigmoid" || op == "Sign" || op == "Sin" || op == "Sinh" || op == "Softmax" ||
         op == "Softplus" || op == "Softsign" || op == "Sqrt" || op == "Tan" || op == "Tanh";
}

bool binary_broadcast(const std::string& op) {
  return op == "Add" || op == "Sub" || op == "Mul" || op == "Div" || op == "Max" || op == "Min" || op == "Pow";
}

const ParsedAttribute* find_attr(const ParsedNode& node, const std::string& name) {
  for (const ParsedAttribute& attribute : node.attributes) {
    if (attribute.name == name) return &attribute;
  }
  return nullptr;
}

const ParsedGraph* find_branch(const ParsedNode& node, const std::string& name) {
  const ParsedAttribute* attribute = find_attr(node, name);
  if (attribute == nullptr || attribute->graphs.empty() || !attribute->graphs.front()) return nullptr;
  return attribute->graphs.front().get();
}

using ShapeMap = std::map<std::string, TensorShape>;
using ConstantMap = std::map<std::string, std::vector<std::int64_t>>;

struct Propagator {
  const ModelAnalysisReport* report = nullptr;
  std::vector<PropagatedShape>* published = nullptr;
  std::set<std::string> warned;

  const TensorShape* find(const ShapeMap& shapes, const std::string& name) const {
    const auto found = shapes.find(name);
    if (found == shapes.end()) return nullptr;
    return &found->second;
  }

  bool publish(ShapeMap& shapes, const std::string& raw_name, const TensorShape& shape, const OnnxNode& node,
               const std::vector<std::string>& inputs, const std::string& graph_path, bool root) {
    if (raw_name.empty() || !shape.rank_known) return false;
    if (shapes.find(raw_name) != shapes.end()) {
      if (warned.insert(graph_path + ":" + raw_name).second && report != nullptr) {
        const_cast<ModelAnalysisReport*>(report)->warnings.push_back(
            {"tensor_has_multiple_producers", "tensor '" + raw_name + "' already has a shape source"});
      }
      return false;
    }
    shapes.emplace(raw_name, shape);
    PropagatedShape item;
    item.name = root ? raw_name : graph_path + ":" + raw_name;
    item.shape = shape;
    item.domain = node.domain;
    item.op_type = node.op_type;
    item.from_inputs = inputs;
    item.graph_path = graph_path;
    published->push_back(std::move(item));
    return true;
  }
};

std::int64_t domain_version(const std::vector<OnnxOpset>& opsets, const std::string& domain) {
  std::int64_t version = -1;
  for (const OnnxOpset& opset : opsets) {
    if (opset.domain == domain) version = std::max(version, opset.version);
  }
  return version;
}

bool dims_equal(const ShapeDimension& left, const ShapeDimension& right, ShapeDimension& out) {
  if (left.kind == DimensionKind::Unknown) {
    out = right;
    return true;
  }
  if (right.kind == DimensionKind::Unknown) {
    out = left;
    return true;
  }
  if (left.kind == DimensionKind::Static && right.kind == DimensionKind::Static && left.value == right.value) {
    out = left;
    return true;
  }
  if (left.kind == DimensionKind::Symbolic && right.kind == DimensionKind::Symbolic && left.symbol == right.symbol) {
    out = left;
    return true;
  }
  return false;
}

bool same_shape(const TensorShape& left, const TensorShape& right, TensorShape& out) {
  if (!left.rank_known && right.rank_known) {
    out = right;
    return true;
  }
  if (left.rank_known && !right.rank_known) {
    out = left;
    return true;
  }
  if (!left.rank_known || left.dimensions.size() != right.dimensions.size()) return false;
  out.rank_known = true;
  out.dimensions.resize(left.dimensions.size());
  for (std::size_t i = 0; i < left.dimensions.size(); ++i) {
    if (!dims_equal(left.dimensions[i], right.dimensions[i], out.dimensions[i])) return false;
  }
  return true;
}

bool concat_shapes(const std::vector<TensorShape>& inputs, std::int64_t axis, TensorShape& out) {
  if (inputs.empty() || !inputs.front().rank_known) return false;
  const std::size_t rank = inputs.front().dimensions.size();
  for (const TensorShape& input : inputs) {
    if (!input.rank_known || input.dimensions.size() != rank) return false;
  }
  if (axis < 0) axis += static_cast<std::int64_t>(rank);
  if (axis < 0 || static_cast<std::size_t>(axis) >= rank) return false;
  out = inputs.front();
  for (std::size_t dim = 0; dim < rank; ++dim) {
    if (dim == static_cast<std::size_t>(axis)) continue;
    for (const TensorShape& input : inputs) {
      ShapeDimension matched;
      if (!dims_equal(out.dimensions[dim], input.dimensions[dim], matched)) return false;
      out.dimensions[dim] = matched;
    }
  }
  std::int64_t sum = 0;
  for (const TensorShape& input : inputs) {
    const ShapeDimension& dimension = input.dimensions[static_cast<std::size_t>(axis)];
    if (dimension.kind != DimensionKind::Static) {
      out.dimensions[static_cast<std::size_t>(axis)].kind = DimensionKind::Unknown;
      out.dimensions[static_cast<std::size_t>(axis)].symbol.clear();
      return true;
    }
    if (sum > std::numeric_limits<std::int64_t>::max() - dimension.value) return false;
    sum += dimension.value;
  }
  out.dimensions[static_cast<std::size_t>(axis)] = static_dim(sum);
  return true;
}

bool flatten_shape(const TensorShape& input, std::int64_t axis, TensorShape& out) {
  if (!input.rank_known) return false;
  const auto rank = static_cast<std::int64_t>(input.dimensions.size());
  if (axis < -rank || axis > rank) return false;
  if (axis < 0) axis += rank;
  auto product = [](const std::vector<ShapeDimension>& dims, std::size_t begin, std::size_t end, ShapeDimension& slot) {
    std::int64_t value = 1;
    for (std::size_t i = begin; i < end; ++i) {
      if (dims[i].kind != DimensionKind::Static) {
        slot.kind = DimensionKind::Unknown;
        return;
      }
      if (value > std::numeric_limits<std::int64_t>::max() / std::max<std::int64_t>(dims[i].value, 1)) {
        slot.kind = DimensionKind::Unknown;
        return;
      }
      value *= dims[i].value;
    }
    slot = static_dim(value);
  };
  out.rank_known = true;
  out.dimensions.assign(2, ShapeDimension{});
  product(input.dimensions, 0, static_cast<std::size_t>(axis), out.dimensions[0]);
  product(input.dimensions, static_cast<std::size_t>(axis), input.dimensions.size(), out.dimensions[1]);
  return true;
}

bool transpose_shape(const TensorShape& input, const std::vector<std::int64_t>* perm, TensorShape& out) {
  if (!input.rank_known) return false;
  out.rank_known = true;
  out.dimensions.resize(input.dimensions.size());
  if (perm == nullptr) {
    std::reverse_copy(input.dimensions.begin(), input.dimensions.end(), out.dimensions.begin());
    return true;
  }
  if (perm->size() != input.dimensions.size()) return false;
  std::vector<bool> seen(perm->size(), false);
  for (std::size_t i = 0; i < perm->size(); ++i) {
    const std::int64_t axis = (*perm)[i];
    if (axis < 0 || static_cast<std::size_t>(axis) >= perm->size() || seen[static_cast<std::size_t>(axis)]) {
      return false;
    }
    seen[static_cast<std::size_t>(axis)] = true;
    out.dimensions[i] = input.dimensions[static_cast<std::size_t>(axis)];
  }
  return true;
}

bool reshape_shape(const TensorShape& input, const std::vector<std::int64_t>& encoded, bool allow_zero,
                   TensorShape& out) {
  if (!input.rank_known) return false;
  out.rank_known = true;
  out.dimensions.resize(encoded.size());
  int unknown = -1;
  std::int64_t known = 1;
  for (std::size_t i = 0; i < encoded.size(); ++i) {
    const std::int64_t dim = encoded[i];
    if (dim == -1) {
      if (unknown >= 0) return false;
      unknown = static_cast<int>(i);
      out.dimensions[i].kind = DimensionKind::Unknown;
      continue;
    }
    if (dim == 0 && !allow_zero) {
      if (i >= input.dimensions.size()) return false;
      out.dimensions[i] = input.dimensions[i];
    } else if (dim < 0) {
      return false;
    } else {
      out.dimensions[i] = static_dim(dim);
    }
    if (out.dimensions[i].kind == DimensionKind::Static) {
      if (known > std::numeric_limits<std::int64_t>::max() / std::max<std::int64_t>(out.dimensions[i].value, 1)) {
        return false;
      }
      known *= out.dimensions[i].value;
    }
  }
  if (unknown < 0) return true;
  bool input_static = true;
  std::int64_t input_count = 1;
  for (const ShapeDimension& dimension : input.dimensions) {
    if (dimension.kind != DimensionKind::Static) {
      input_static = false;
      break;
    }
    input_count *= dimension.value;
  }
  if (!input_static || known == 0 || input_count % known != 0) return true;
  out.dimensions[static_cast<std::size_t>(unknown)] = static_dim(input_count / known);
  return true;
}

enum class Step { Wait, Done, Fail };

struct GraphPropagation {
  Step step = Step::Done;
  ShapeMap shapes;
};

GraphPropagation propagate_graph(const ParsedGraph& graph, ShapeMap shapes, ConstantMap constants,
                                 const std::vector<OnnxOpset>& opsets, Propagator& propagator, bool root);

Step propagate_node(const ParsedNode& node, ShapeMap& shapes, const ConstantMap& constants,
                    const std::vector<OnnxOpset>& opsets, Propagator& propagator, bool root) {
  for (const std::string& output : node.node.outputs) {
    if (!output.empty() && shapes.find(output) != shapes.end()) return Step::Done;
  }
  const std::string& op = node.node.op_type;
  if (node.node.domain != "ai.onnx") return Step::Fail;
  auto require = [&](std::size_t index) -> const TensorShape* {
    if (index >= node.node.inputs.size()) return nullptr;
    return propagator.find(shapes, node.node.inputs[index]);
  };
  TensorShape result;
  std::vector<std::string> used;
  bool ready = false;
  if (op == "If") {
    const ParsedGraph* then_branch = find_branch(node, "then_branch");
    const ParsedGraph* else_branch = find_branch(node, "else_branch");
    if (then_branch == nullptr || else_branch == nullptr) return Step::Fail;
    if (then_branch->outputs.size() != node.node.outputs.size() ||
        else_branch->outputs.size() != node.node.outputs.size()) {
      return Step::Fail;
    }
    GraphPropagation then_result = propagate_graph(*then_branch, shapes, constants, opsets, propagator, false);
    GraphPropagation else_result = propagate_graph(*else_branch, shapes, constants, opsets, propagator, false);
    if (then_result.step == Step::Wait || else_result.step == Step::Wait) return Step::Wait;
    for (std::size_t i = 0; i < node.node.outputs.size(); ++i) {
      const TensorShape* left = propagator.find(then_result.shapes, then_branch->outputs[i]);
      const TensorShape* right = propagator.find(else_result.shapes, else_branch->outputs[i]);
      if (left == nullptr || right == nullptr) return Step::Fail;
      TensorShape merged;
      if (!same_shape(*left, *right, merged)) return Step::Fail;
      propagator.publish(shapes, node.node.outputs[i], merged, node.node,
                         {then_branch->outputs[i], else_branch->outputs[i]}, node.node.graph_path, root);
    }
    return Step::Done;
  }
  if (preserves_shape(op)) {
    const TensorShape* first = require(0);
    if (node.node.inputs.empty()) return Step::Fail;
    if (first == nullptr) return Step::Wait;
    result = *first;
    used.push_back(node.node.inputs[0]);
    ready = true;
  } else if (binary_broadcast(op)) {
    const TensorShape* left = require(0);
    const TensorShape* right = require(1);
    if (node.node.inputs.size() < 2) return Step::Fail;
    if (left == nullptr || right == nullptr) return Step::Wait;
    if (!broadcast_shapes(*left, *right, result)) return Step::Fail;
    used = {node.node.inputs[0], node.node.inputs[1]};
    ready = true;
  } else if (op == "Concat") {
    const ParsedAttribute* axis_attr = find_attr(node, "axis");
    if (axis_attr == nullptr || !axis_attr->has_int) return Step::Fail;
    std::vector<TensorShape> parts;
    for (const std::string& input : node.node.inputs) {
      if (input.empty()) continue;
      const TensorShape* shape = propagator.find(shapes, input);
      if (shape == nullptr) return Step::Wait;
      parts.push_back(*shape);
      used.push_back(input);
    }
    if (!concat_shapes(parts, axis_attr->int_value, result)) return Step::Fail;
    ready = true;
  } else if (op == "Flatten") {
    const TensorShape* first = require(0);
    if (first == nullptr) return node.node.inputs.empty() ? Step::Fail : Step::Wait;
    const ParsedAttribute* axis_attr = find_attr(node, "axis");
    const std::int64_t axis = axis_attr != nullptr && axis_attr->has_int ? axis_attr->int_value : 1;
    if (!flatten_shape(*first, axis, result)) return Step::Fail;
    used.push_back(node.node.inputs[0]);
    ready = true;
  } else if (op == "Transpose") {
    const TensorShape* first = require(0);
    if (first == nullptr) return node.node.inputs.empty() ? Step::Fail : Step::Wait;
    const ParsedAttribute* perm = find_attr(node, "perm");
    if (!transpose_shape(*first, perm == nullptr ? nullptr : &perm->ints, result)) return Step::Fail;
    used.push_back(node.node.inputs[0]);
    ready = true;
  } else if (op == "Reshape") {
    const TensorShape* first = require(0);
    if (node.node.inputs.size() < 2) return Step::Fail;
    if (first == nullptr) return Step::Wait;
    const auto constant = constants.find(node.node.inputs[1]);
    if (constant == constants.end()) return Step::Fail;
    const ParsedAttribute* allow = find_attr(node, "allowzero");
    const std::int64_t version = domain_version(opsets, node.node.domain);
    const bool allow_zero = allow != nullptr && allow->has_int && allow->int_value == 1 &&
                            (version < 0 || version >= 14);
    if (!reshape_shape(*first, constant->second, allow_zero, result)) return Step::Fail;
    used = {node.node.inputs[0], node.node.inputs[1]};
    ready = true;
  } else if (op == "MatMul") {
    const TensorShape* left = require(0);
    const TensorShape* right = require(1);
    if (node.node.inputs.size() < 2) return Step::Fail;
    if (left == nullptr || right == nullptr) return Step::Wait;
    if (!matmul_shapes(*left, *right, result)) return Step::Fail;
    used = {node.node.inputs[0], node.node.inputs[1]};
    ready = true;
  } else {
    return Step::Fail;
  }
  if (!ready) return Step::Fail;
  for (const std::string& output : node.node.outputs) {
    propagator.publish(shapes, output, result, node.node, used, node.node.graph_path, root);
  }
  return Step::Done;
}

GraphPropagation propagate_graph(const ParsedGraph& graph, ShapeMap shapes, ConstantMap constants,
                                 const std::vector<OnnxOpset>& opsets, Propagator& propagator, bool root) {
  GraphPropagation result;
  result.shapes = std::move(shapes);
  for (const InitializerInfo& initializer : graph.initializers) {
    if (initializer.declared_shape.rank_known && result.shapes.find(initializer.name) == result.shapes.end()) {
      result.shapes.emplace(initializer.name, initializer.declared_shape);
    }
  }
  for (const auto& entry : graph.int64_constants) constants.insert(entry);
  std::vector<Step> state(graph.nodes.size(), Step::Wait);
  bool progress = true;
  int guard = static_cast<int>(graph.nodes.size()) + 2;
  while (progress && guard-- > 0) {
    progress = false;
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
      if (state[i] != Step::Wait) continue;
      const Step step = propagate_node(graph.nodes[i], result.shapes, constants, opsets, propagator, root);
      state[i] = step;
      if (step == Step::Done) progress = true;
    }
  }
  result.step = Step::Done;
  for (const Step step : state) {
    if (step == Step::Wait) result.step = Step::Wait;
  }
  return result;
}

void seed_and_propagate(ModelAnalysisReport& report, const ParsedGraph& graph) {
  ShapeMap shapes;
  auto consider = [&](const std::string& name, const TensorShape& shape) {
    if (name.empty() || !shape.rank_known) return;
    const auto found = shapes.find(name);
    if (found == shapes.end()) {
      shapes.emplace(name, shape);
      return;
    }
    if (relate(found->second, shape) == ShapeRelation::Conflict) shapes.erase(found);
  };
  for (const ValueEndpoint& input : report.runtime_inputs) {
    if (input.type.kind == OnnxTypeKind::Tensor || input.type.kind == OnnxTypeKind::SparseTensor) {
      consider(input.name, input.type.declared_shape);
    }
  }
  for (const InitializerInfo& initializer : report.initializers) {
    consider(initializer.name, initializer.declared_shape);
  }
  for (const InitializerInfo& initializer : report.sparse_initializers) {
    consider(initializer.name, initializer.declared_shape);
  }
  Propagator propagator;
  propagator.report = &report;
  propagator.published = &report.propagated_shapes;
  ConstantMap constants = graph.int64_constants;
  propagate_graph(graph, shapes, constants, report.opsets, propagator, true);
  for (const ValueEndpoint& output : report.outputs) {
    const PropagatedShape* propagated = find_propagated_shape(report, output.name);
    if (propagated == nullptr) continue;
    if (output.type.kind != OnnxTypeKind::Tensor && output.type.kind != OnnxTypeKind::SparseTensor) continue;
    if (relate(output.type.declared_shape, propagated->shape) == ShapeRelation::Conflict) {
      report.warnings.push_back(
          {"declared_shape_conflicts_with_propagation",
           "output '" + output.name + "' declared " + format_tensor_shape(output.type.declared_shape) +
               " conflicts with propagated " + format_tensor_shape(propagated->shape)});
    }
  }
  for (const RecordedValueInfo& info : report.value_infos) {
    const ValueEndpoint* endpoint = nullptr;
    for (const ValueEndpoint& output : report.outputs) {
      if (output.name == info.name) endpoint = &output;
    }
    for (const ValueEndpoint& input : report.graph_inputs) {
      if (input.name == info.name) endpoint = &input;
    }
    if (endpoint == nullptr) continue;
    if (relate(endpoint->type.declared_shape, info.type.declared_shape) == ShapeRelation::Conflict) {
      report.warnings.push_back(
          {"value_info_disagrees_with_declared",
           "value_info '" + info.name + "' records " + format_tensor_shape(info.type.declared_shape) +
               " which disagrees with declared " + format_tensor_shape(endpoint->type.declared_shape)});
    }
  }
}

}  // namespace

ModelAnalysisReport OnnxAnalyzer::analyze_bytes(std::string_view bytes, const std::string& source_name,
                                                const std::string& external_directory) const {
  Parser parser(bytes, source_name, std::filesystem::path(external_directory));
  ModelAnalysisReport report = parser.parse();
  report.source = source_name;
  seed_and_propagate(report, parser.main_graph());
  return report;
}

ModelAnalysisReport OnnxAnalyzer::analyze_file(const std::string& path) const {
  const std::filesystem::path file(path);
  std::ifstream input(file, std::ios::binary);
  if (!input) throw VieError(ErrorCode::InvalidArgument, "cannot open ONNX file: " + path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string directory = file.has_parent_path() ? file.parent_path().string() : std::string(".");
  return analyze_bytes(buffer.str(), path, directory);
}

}  // namespace vie
