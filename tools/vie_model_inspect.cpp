#include "vie/model/model_inspection.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* program) {
  std::cerr << "usage: " << program << " model.onnx [options]\n"
            << "\n"
            << "Options:\n"
            << "  --manifest <path>   Validate a user-provided manifest JSON\n"
            << "  --report <path>     Write JSON report to file\n"
            << "  --cpu               Test CPU EP execution\n"
            << "  --cuda              Test CUDA EP execution\n"
            << "  --json              Output JSON to stdout instead of text\n"
            << "  --help              Show this help\n";
}

bool write_file(const std::string& path, const std::string& content) {
  std::ofstream file(path);
  if (!file) {
    std::cerr << "error: cannot write to " << path << "\n";
    return false;
  }
  file << content;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 2;
  }

  std::string model_path;
  vie::InspectionOptions options;
  bool json_stdout = false;
  std::string report_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
    if (arg == "--manifest" && i + 1 < argc) {
      options.user_manifest_path = argv[++i];
      options.validate_user_manifest = true;
      continue;
    }
    if (arg == "--report" && i + 1 < argc) {
      report_path = argv[++i];
      continue;
    }
    if (arg == "--cpu") {
      options.test_cpu_execution = true;
      continue;
    }
    if (arg == "--cuda") {
      options.test_cuda_execution = true;
      continue;
    }
    if (arg == "--json") {
      json_stdout = true;
      continue;
    }
    if (model_path.empty()) {
      model_path = arg;
      continue;
    }
    std::cerr << "error: unknown argument: " << arg << "\n";
    print_usage(argv[0]);
    return 2;
  }

  if (model_path.empty()) {
    std::cerr << "error: no model path provided\n";
    print_usage(argv[0]);
    return 2;
  }

  try {
    vie::ModelInspectionReport report = vie::inspect_model(model_path, options);

    if (json_stdout) {
      std::cout << vie::inspection_report_to_json(report);
    } else {
      std::cout << vie::inspection_report_to_text(report);
    }

    if (!report_path.empty()) {
      if (!write_file(report_path, vie::inspection_report_to_json(report))) {
        return 1;
      }
      std::cerr << "Report written to " << report_path << "\n";
    }

    // Return non-zero if the model is not runnable
    if (report.diagnostic_status() == "analyzer_failed" ||
        report.diagnostic_status() == "recognition_failed" ||
        report.diagnostic_status() == "manifest_generation_failed" ||
        report.diagnostic_status() == "manifest_invalid") {
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
