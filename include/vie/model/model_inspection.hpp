#pragma once

#include "vie/analyzer/analysis_report.hpp"
#include "vie/model/manifest.hpp"
#include "vie/model/manifest_validator.hpp"
#include "vie/recognizer/recognition.hpp"

#include <string>
#include <vector>

namespace vie {

enum class InspectionStatus {
  NotRun,
  Success,
  Failed,
};

enum class ExecutionStatus {
  NotTested,
  Passed,
  Failed,
};

struct AnalyzerInspection {
  InspectionStatus status = InspectionStatus::NotRun;
  ModelAnalysisReport report;
  std::string error;
};

struct RecognitionInspection {
  InspectionStatus status = InspectionStatus::NotRun;
  RecognitionResult result;
  std::string error;
};

struct ManifestInspection {
  InspectionStatus generated = InspectionStatus::NotRun;
  bool validated = false;
  ModelManifest manifest;
  ManifestValidation validation;
  std::string generate_error;
};

struct UserManifestInspection {
  bool loaded = false;
  InspectionStatus validated = InspectionStatus::NotRun;
  ManifestValidation validation;
  std::string load_error;
  std::string validate_error;
};

struct ExecutionInspection {
  ExecutionStatus cpu = ExecutionStatus::NotTested;
  ExecutionStatus cuda = ExecutionStatus::NotTested;
  std::string cpu_error;
  std::string cuda_error;
  double load_ms = 0;
  double infer_ms = 0;
  std::size_t output_bytes = 0;
};

struct ModelInspectionReport {
  std::string model_path;
  std::string model_sha256;
  std::size_t model_bytes = 0;

  AnalyzerInspection analyzer;
  RecognitionInspection recognition;
  ManifestInspection manifest;
  UserManifestInspection user_manifest;
  ExecutionInspection execution;

  std::string diagnostic_status() const;
};

struct InspectionOptions {
  bool validate_user_manifest = false;
  bool test_cpu_execution = false;
  bool test_cuda_execution = false;
  std::string user_manifest_path;
  std::string dummy_image_path;
};

ModelInspectionReport inspect_model(const std::string& model_path, const InspectionOptions& options);

std::string inspection_report_to_text(const ModelInspectionReport& report);
std::string inspection_report_to_json(const ModelInspectionReport& report);

const char* to_string(InspectionStatus status);
const char* to_string(ExecutionStatus status);

}  // namespace vie
