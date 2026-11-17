#pragma once

#include "vie/analyzer/analysis_report.hpp"

#include <string>
#include <string_view>

namespace vie {

// Reads an ONNX ModelProto from the file bytes. It does not load ONNX Runtime,
// TensorRT, or CUDA. Declared shapes stay as they are in the file.
// propagated_shapes is a separate table produced from declared inputs,
// initializer dims, and a limited operator set (shape-preserving unary ops,
// broadcast of Add/Sub/Mul/Div/Max/Min/Pow, Concat, Flatten, Transpose,
// Reshape, MatMul, and If).
class OnnxAnalyzer {
 public:
  ModelAnalysisReport analyze_file(const std::string& path) const;
  ModelAnalysisReport analyze_bytes(std::string_view bytes, const std::string& source_name,
                                    const std::string& external_directory) const;
};

}  // namespace vie
