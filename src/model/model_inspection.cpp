#include "vie/model/model_inspection.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/core/error.hpp"
#include "vie/engine/inference_engine.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/preprocess/image_input.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace vie {

namespace {

std::string read_file_hash(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return "";

  std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
  if (buffer.empty()) return "";

  // Simple FNV-1a hash for identification - not cryptographic
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (char byte : buffer) {
    h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(byte));
    h *= 0x100000001b3ULL;
  }

  std::ostringstream oss;
  oss << std::hex << h;
  return oss.str();
}

std::size_t file_size(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  return file ? static_cast<std::size_t>(file.tellg()) : 0;
}

}  // namespace

const char* to_string(InspectionStatus status) {
  switch (status) {
    case InspectionStatus::NotRun:
      return "not_run";
    case InspectionStatus::Success:
      return "success";
    case InspectionStatus::Failed:
      return "failed";
  }
  return "not_run";
}

const char* to_string(ExecutionStatus status) {
  switch (status) {
    case ExecutionStatus::NotTested:
      return "not_tested";
    case ExecutionStatus::Passed:
      return "passed";
    case ExecutionStatus::Failed:
      return "failed";
  }
  return "not_tested";
}

std::string ModelInspectionReport::diagnostic_status() const {
  if (analyzer.status == InspectionStatus::Failed) return "analyzer_failed";
  if (recognition.status == InspectionStatus::Failed) return "recognition_failed";
  if (manifest.generated == InspectionStatus::Failed) return "manifest_generation_failed";
  if (!manifest.validated) return "manifest_invalid";
  if (user_manifest.loaded && user_manifest.validated == InspectionStatus::Failed) {
    return "user_manifest_invalid";
  }
  if (execution.cpu == ExecutionStatus::Failed) return "cpu_execution_failed";
  if (execution.cuda == ExecutionStatus::Failed) return "cuda_execution_failed";

  if (execution.cpu == ExecutionStatus::Passed && execution.cuda == ExecutionStatus::Passed) {
    return "cpu_and_cuda_tested";
  }
  if (execution.cpu == ExecutionStatus::Passed) return "cpu_tested";
  if (execution.cuda == ExecutionStatus::Passed) return "cuda_tested";
  if (recognition.result.status == RecognitionStatus::Recognized) return "recognized";
  if (recognition.result.status == RecognitionStatus::Ambiguous) return "ambiguous";
  return "unknown";
}

ModelInspectionReport inspect_model(const std::string& model_path, const InspectionOptions& options) {
  ModelInspectionReport report;
  report.model_path = model_path;
  report.model_bytes = file_size(model_path);
  report.model_sha256 = read_file_hash(model_path);

  // Phase 1: Analyzer
  try {
    OnnxAnalyzer analyzer;
    report.analyzer.report = analyzer.analyze_file(model_path);
    report.analyzer.status = InspectionStatus::Success;
  } catch (const VieError& error) {
    report.analyzer.status = InspectionStatus::Failed;
    report.analyzer.error = error.what();
    return report;
  } catch (const std::exception& error) {
    report.analyzer.status = InspectionStatus::Failed;
    report.analyzer.error = error.what();
    return report;
  }

  // Phase 2: Recognizer
  try {
    ModelRecognizerRegistry registry = default_recognizer_registry();
    report.recognition.result = registry.recognize(report.analyzer.report);
    report.recognition.status = InspectionStatus::Success;
  } catch (const VieError& error) {
    report.recognition.status = InspectionStatus::Failed;
    report.recognition.error = error.what();
    return report;
  } catch (const std::exception& error) {
    report.recognition.status = InspectionStatus::Failed;
    report.recognition.error = error.what();
    return report;
  }

  // Phase 3: Manifest Generator
  try {
    ManifestGenerator generator;
    report.manifest.manifest = generator.generate(report.analyzer.report, report.recognition.result, model_path);
    report.manifest.generated = InspectionStatus::Success;
  } catch (const VieError& error) {
    report.manifest.generated = InspectionStatus::Failed;
    report.manifest.generate_error = error.what();
    return report;
  } catch (const std::exception& error) {
    report.manifest.generated = InspectionStatus::Failed;
    report.manifest.generate_error = error.what();
    return report;
  }

  // Phase 4: Manifest Validator
  try {
    ManifestValidator validator;
    report.manifest.validation = validator.validate(report.manifest.manifest);
    report.manifest.validated = report.manifest.validation.runnable();
  } catch (const VieError& error) {
    report.manifest.validated = false;
    report.manifest.generate_error = error.what();
  } catch (const std::exception& error) {
    report.manifest.validated = false;
    report.manifest.generate_error = error.what();
  }

  // Phase 5: User manifest validation (optional)
  if (options.validate_user_manifest && !options.user_manifest_path.empty()) {
    report.user_manifest.loaded = true;
    try {
      ModelManifest user_manifest = ModelManifest::load_file(options.user_manifest_path);
      ManifestValidator validator;
      report.user_manifest.validation = validator.validate(user_manifest);
      report.user_manifest.validated =
          report.user_manifest.validation.runnable() ? InspectionStatus::Success : InspectionStatus::Failed;
    } catch (const VieError& error) {
      report.user_manifest.validated = InspectionStatus::Failed;
      report.user_manifest.load_error = error.what();
    } catch (const std::exception& error) {
      report.user_manifest.validated = InspectionStatus::Failed;
      report.user_manifest.load_error = error.what();
    }
  }

  // Phase 6: Execution testing (optional)
  if (options.test_cpu_execution && report.manifest.validated) {
    try {
      InferenceEngine engine;
      ModelConfig config;
      config.model_path = model_path;
      config.manifest = report.manifest.manifest;
      config.backend = BackendKind::Auto;
      config.device = DeviceType::Cpu;

      auto start = std::chrono::steady_clock::now();
      engine.load(config);
      report.execution.load_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

      // Create a dummy input tensor
      const auto& input_spec = report.manifest.manifest.input;
      std::vector<std::int64_t> shape(input_spec.shape.begin(), input_spec.shape.end());
      Tensor input_tensor = Tensor::zeros(DeviceContext(DeviceType::Cpu, 0), DataType::Fp32, shape);

      BackendRequest request;
      request.inputs.emplace(input_spec.name, std::move(input_tensor));

      start = std::chrono::steady_clock::now();
      BackendOutput output = engine.execute(request);
      report.execution.infer_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

      for (const auto& [name, named_tensor] : output.tensors) {
        report.execution.output_bytes += named_tensor.tensor.nbytes();
      }

      report.execution.cpu = ExecutionStatus::Passed;
    } catch (const VieError& error) {
      report.execution.cpu = ExecutionStatus::Failed;
      report.execution.cpu_error = error.what();
    } catch (const std::exception& error) {
      report.execution.cpu = ExecutionStatus::Failed;
      report.execution.cpu_error = error.what();
    }
  }

  if (options.test_cuda_execution && report.manifest.validated) {
    try {
      InferenceEngine engine;
      ModelConfig config;
      config.model_path = model_path;
      config.manifest = report.manifest.manifest;
      config.backend = BackendKind::Auto;
      config.device = DeviceType::Cuda;
      config.device_id = 0;

      engine.load(config);

      const auto& input_spec = report.manifest.manifest.input;
      std::vector<std::int64_t> shape(input_spec.shape.begin(), input_spec.shape.end());

      // Create zeros on CPU then copy to CUDA
      Tensor cpu_tensor = Tensor::zeros(DeviceContext(DeviceType::Cpu, 0), DataType::Fp32, shape);
      Tensor cuda_tensor = cpu_tensor.to(DeviceContext(DeviceType::Cuda, 0));

      BackendRequest request;
      request.inputs.emplace(input_spec.name, std::move(cuda_tensor));

      auto start = std::chrono::steady_clock::now();
      BackendOutput output = engine.execute(request);
      engine.synchronize();
      report.execution.infer_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

      for (const auto& [name, named_tensor] : output.tensors) {
        report.execution.output_bytes += named_tensor.tensor.nbytes();
      }

      report.execution.cuda = ExecutionStatus::Passed;
    } catch (const VieError& error) {
      report.execution.cuda = ExecutionStatus::Failed;
      report.execution.cuda_error = error.what();
    } catch (const std::exception& error) {
      report.execution.cuda = ExecutionStatus::Failed;
      report.execution.cuda_error = error.what();
    }
  }

  return report;
}

std::string inspection_report_to_text(const ModelInspectionReport& report) {
  std::ostringstream oss;
  oss << "VIE Model Inspection\n";
  oss << "====================\n\n";

  oss << "Model:\n";
  oss << "  File: " << report.model_path << "\n";
  oss << "  Size: " << report.model_bytes << " bytes\n";
  if (!report.model_sha256.empty()) {
    oss << "  Hash: " << report.model_sha256 << "\n";
  }
  oss << "\n";

  oss << "Analyzer:\n";
  oss << "  Status: " << to_string(report.analyzer.status) << "\n";
  if (report.analyzer.status == InspectionStatus::Success) {
    oss << "  IR Version: " << report.analyzer.report.ir_version << "\n";
    oss << "  Opset: ";
    for (std::size_t i = 0; i < report.analyzer.report.opsets.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << report.analyzer.report.opsets[i].version;
    }
    oss << "\n";
    oss << "  Input Count: " << report.analyzer.report.graph_inputs.size() << "\n";
    oss << "  Output Count: " << report.analyzer.report.outputs.size() << "\n";
    oss << "  Node Count: " << report.analyzer.report.nodes.size() << "\n";
  } else if (!report.analyzer.error.empty()) {
    oss << "  Error: " << report.analyzer.error << "\n";
  }
  oss << "\n";

  oss << "Recognition:\n";
  oss << "  Status: " << to_string(report.recognition.result.status) << "\n";
  if (report.recognition.result.status == RecognitionStatus::Recognized) {
    const RecognitionCandidate* selected = report.recognition.result.selected();
    if (selected) {
      oss << "  Recognizer: " << selected->recognizer_id << "\n";
      oss << "  Architecture: " << selected->architecture << "\n";
      oss << "  Family: " << selected->family << "\n";
      oss << "  Protocol: " << selected->decoder.protocol << "\n";
      oss << "  Evidence:\n";
      for (const auto& evidence : selected->evidence) {
        oss << "    - " << evidence.code << ": " << evidence.detail << "\n";
      }
    }
  } else if (!report.recognition.result.explanation.empty()) {
    oss << "  Explanation: " << report.recognition.result.explanation << "\n";
  }
  if (!report.recognition.error.empty()) {
    oss << "  Error: " << report.recognition.error << "\n";
  }
  oss << "\n";

  oss << "Manifest:\n";
  oss << "  Generated: " << (report.manifest.generated == InspectionStatus::Success ? "Yes" : "No") << "\n";
  if (report.manifest.generated == InspectionStatus::Success) {
    oss << "  Validated: " << (report.manifest.validated ? "Yes" : "No") << "\n";
    if (!report.manifest.validation.schema_issues.empty()) {
      oss << "  Schema Issues:\n";
      for (const auto& issue : report.manifest.validation.schema_issues) {
        oss << "    - " << issue.path << ": " << issue.message << "\n";
      }
    }
    if (!report.manifest.validation.unresolved_critical.empty()) {
      oss << "  Unresolved Critical:\n";
      for (const auto& issue : report.manifest.validation.unresolved_critical) {
        oss << "    - " << issue.path << ": " << issue.message << "\n";
      }
    }
  } else if (!report.manifest.generate_error.empty()) {
    oss << "  Error: " << report.manifest.generate_error << "\n";
  }
  oss << "\n";

  if (report.user_manifest.loaded) {
    oss << "User Manifest:\n";
    oss << "  Validated: " << to_string(report.user_manifest.validated) << "\n";
    if (!report.user_manifest.load_error.empty()) {
      oss << "  Load Error: " << report.user_manifest.load_error << "\n";
    }
    if (!report.user_manifest.validation.schema_issues.empty()) {
      oss << "  Schema Issues:\n";
      for (const auto& issue : report.user_manifest.validation.schema_issues) {
        oss << "    - " << issue.path << ": " << issue.message << "\n";
      }
    }
    oss << "\n";
  }

  oss << "Execution:\n";
  oss << "  CPU EP: " << to_string(report.execution.cpu) << "\n";
  if (report.execution.cpu == ExecutionStatus::Passed) {
    oss << "    Load: " << report.execution.load_ms << " ms\n";
    oss << "    Infer: " << report.execution.infer_ms << " ms\n";
    oss << "    Output: " << report.execution.output_bytes << " bytes\n";
  } else if (!report.execution.cpu_error.empty()) {
    oss << "    Error: " << report.execution.cpu_error << "\n";
  }
  oss << "  CUDA EP: " << to_string(report.execution.cuda) << "\n";
  if (!report.execution.cuda_error.empty()) {
    oss << "    Error: " << report.execution.cuda_error << "\n";
  }
  oss << "\n";

  oss << "Diagnostic: " << report.diagnostic_status() << "\n";

  return oss.str();
}

std::string inspection_report_to_json(const ModelInspectionReport& report) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"model\": {\n";
  oss << "    \"path\": \"" << report.model_path << "\",\n";
  oss << "    \"size_bytes\": " << report.model_bytes << ",\n";
  oss << "    \"hash\": \"" << report.model_sha256 << "\"\n";
  oss << "  },\n";

  oss << "  \"analyzer\": {\n";
  oss << "    \"status\": \"" << to_string(report.analyzer.status) << "\"";
  if (report.analyzer.status == InspectionStatus::Success) {
    oss << ",\n    \"ir_version\": " << report.analyzer.report.ir_version;
    oss << ",\n    \"opsets\": [";
    for (std::size_t i = 0; i < report.analyzer.report.opsets.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << report.analyzer.report.opsets[i].version;
    }
    oss << "]";
    oss << ",\n    \"input_count\": " << report.analyzer.report.graph_inputs.size();
    oss << ",\n    \"output_count\": " << report.analyzer.report.outputs.size();
    oss << ",\n    \"node_count\": " << report.analyzer.report.nodes.size();
  } else if (!report.analyzer.error.empty()) {
    oss << ",\n    \"error\": \"" << report.analyzer.error << "\"";
  }
  oss << "\n  },\n";

  oss << "  \"recognition\": {\n";
  oss << "    \"status\": \"" << to_string(report.recognition.result.status) << "\"";
  if (report.recognition.result.status == RecognitionStatus::Recognized) {
    const RecognitionCandidate* selected = report.recognition.result.selected();
    if (selected) {
      oss << ",\n    \"recognizer\": \"" << selected->recognizer_id << "\"";
      oss << ",\n    \"architecture\": \"" << selected->architecture << "\"";
      oss << ",\n    \"family\": \"" << selected->family << "\"";
      oss << ",\n    \"protocol\": \"" << selected->decoder.protocol << "\"";
      oss << ",\n    \"evidence\": [";
      for (std::size_t i = 0; i < selected->evidence.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "{\"code\": \"" << selected->evidence[i].code << "\", \"detail\": \""
            << selected->evidence[i].detail << "\"}";
      }
      oss << "]";
    }
  } else if (!report.recognition.result.explanation.empty()) {
    oss << ",\n    \"explanation\": \"" << report.recognition.result.explanation << "\"";
  }
  if (!report.recognition.error.empty()) {
    oss << ",\n    \"error\": \"" << report.recognition.error << "\"";
  }
  oss << "\n  },\n";

  oss << "  \"manifest\": {\n";
  oss << "    \"generated\": " << (report.manifest.generated == InspectionStatus::Success ? "true" : "false")
      << ",\n";
  oss << "    \"validated\": " << (report.manifest.validated ? "true" : "false");
  if (report.manifest.generated == InspectionStatus::Success) {
    if (!report.manifest.validation.schema_issues.empty()) {
      oss << ",\n    \"schema_issues\": [";
      for (std::size_t i = 0; i < report.manifest.validation.schema_issues.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "{\"path\": \"" << report.manifest.validation.schema_issues[i].path
            << "\", \"message\": \"" << report.manifest.validation.schema_issues[i].message << "\"}";
      }
      oss << "]";
    }
    if (!report.manifest.validation.unresolved_critical.empty()) {
      oss << ",\n    \"unresolved_critical\": [";
      for (std::size_t i = 0; i < report.manifest.validation.unresolved_critical.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "{\"path\": \"" << report.manifest.validation.unresolved_critical[i].path
            << "\", \"message\": \"" << report.manifest.validation.unresolved_critical[i].message << "\"}";
      }
      oss << "]";
    }
  } else if (!report.manifest.generate_error.empty()) {
    oss << ",\n    \"error\": \"" << report.manifest.generate_error << "\"";
  }
  oss << "\n  },\n";

  if (report.user_manifest.loaded) {
    oss << "  \"user_manifest\": {\n";
    oss << "    \"loaded\": true,\n";
    oss << "    \"validated\": \"" << to_string(report.user_manifest.validated) << "\"";
    if (!report.user_manifest.load_error.empty()) {
      oss << ",\n    \"load_error\": \"" << report.user_manifest.load_error << "\"";
    }
    if (!report.user_manifest.validation.schema_issues.empty()) {
      oss << ",\n    \"schema_issues\": [";
      for (std::size_t i = 0; i < report.user_manifest.validation.schema_issues.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "{\"path\": \"" << report.user_manifest.validation.schema_issues[i].path
            << "\", \"message\": \"" << report.user_manifest.validation.schema_issues[i].message << "\"}";
      }
      oss << "]";
    }
    oss << "\n  },\n";
  }

  oss << "  \"execution\": {\n";
  oss << "    \"cpu\": \"" << to_string(report.execution.cpu) << "\"";
  if (report.execution.cpu == ExecutionStatus::Passed) {
    oss << ",\n    \"cpu_load_ms\": " << report.execution.load_ms;
    oss << ",\n    \"cpu_infer_ms\": " << report.execution.infer_ms;
    oss << ",\n    \"output_bytes\": " << report.execution.output_bytes;
  } else if (!report.execution.cpu_error.empty()) {
    oss << ",\n    \"cpu_error\": \"" << report.execution.cpu_error << "\"";
  }
  oss << ",\n    \"cuda\": \"" << to_string(report.execution.cuda) << "\"";
  if (!report.execution.cuda_error.empty()) {
    oss << ",\n    \"cuda_error\": \"" << report.execution.cuda_error << "\"";
  }
  oss << "\n  },\n";

  oss << "  \"diagnostic\": \"" << report.diagnostic_status() << "\"\n";
  oss << "}\n";

  return oss.str();
}

}  // namespace vie
