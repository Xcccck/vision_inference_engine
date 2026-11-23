#include "vie/model/model_inspection.hpp"

#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/core/error.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string test_data_dir() {
  return std::string(VIE_SOURCE_DIR) + "/testdata/onnx/";
}

bool file_exists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}

}  // namespace

// Test that the inspection tool correctly analyzes a YOLOv8n detection model
TEST(ModelInspection, AnalyzesYolov8nDetection) {
  const std::string model_path = test_data_dir() + "yolov8n.onnx";
  if (!file_exists(model_path)) {
    GTEST_SKIP() << "yolov8n.onnx not found at " << model_path;
  }

  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model(model_path, options);

  EXPECT_EQ(report.analyzer.status, vie::InspectionStatus::Success);
  EXPECT_EQ(report.recognition.result.status, vie::RecognitionStatus::Recognized);
  EXPECT_EQ(report.manifest.generated, vie::InspectionStatus::Success);
  EXPECT_FALSE(report.manifest.validated);  // Unresolved critical fields

  const vie::RecognitionCandidate* selected = report.recognition.result.selected();
  ASSERT_NE(selected, nullptr);
  EXPECT_EQ(selected->recognizer_id, "yolov8_detection");
  EXPECT_EQ(selected->decoder.protocol, "yolov8_detect_raw");
  EXPECT_EQ(selected->family, "yolov8");
}

// Test that the inspection tool correctly analyzes a YOLOv8n-seg model
TEST(ModelInspection, AnalyzesYolov8nSegmentation) {
  const std::string model_path = test_data_dir() + "yolov8n-seg.onnx";
  if (!file_exists(model_path)) {
    GTEST_SKIP() << "yolov8n-seg.onnx not found at " << model_path;
  }

  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model(model_path, options);

  EXPECT_EQ(report.analyzer.status, vie::InspectionStatus::Success);
  EXPECT_EQ(report.recognition.result.status, vie::RecognitionStatus::Recognized);
  EXPECT_EQ(report.manifest.generated, vie::InspectionStatus::Success);

  const vie::RecognitionCandidate* selected = report.recognition.result.selected();
  ASSERT_NE(selected, nullptr);
  EXPECT_EQ(selected->recognizer_id, "yolov8_segmentation");
  EXPECT_EQ(selected->decoder.protocol, "yolov8_seg_raw");
}

// Test that the inspection tool correctly analyzes an RT-DETR model
TEST(ModelInspection, AnalyzesRtdetrL) {
  const std::string model_path = test_data_dir() + "rtdetr-l.onnx";
  if (!file_exists(model_path)) {
    GTEST_SKIP() << "rtdetr-l.onnx not found at " << model_path;
  }

  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model(model_path, options);

  EXPECT_EQ(report.analyzer.status, vie::InspectionStatus::Success);
  EXPECT_EQ(report.recognition.result.status, vie::RecognitionStatus::Recognized);
  EXPECT_EQ(report.manifest.generated, vie::InspectionStatus::Success);

  const vie::RecognitionCandidate* selected = report.recognition.result.selected();
  ASSERT_NE(selected, nullptr);
  EXPECT_EQ(selected->recognizer_id, "rtdetr_detection");
  EXPECT_EQ(selected->decoder.protocol, "rtdetr_detection_raw");
}

// Test that the inspection tool handles non-existent files
TEST(ModelInspection, HandlesNonExistentFile) {
  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model("/nonexistent/model.onnx", options);

  EXPECT_EQ(report.analyzer.status, vie::InspectionStatus::Failed);
  EXPECT_FALSE(report.analyzer.error.empty());
  EXPECT_EQ(report.diagnostic_status(), "analyzer_failed");
}

// Test that the inspection tool handles invalid ONNX files
TEST(ModelInspection, HandlesInvalidOnnx) {
  const std::string model_path = test_data_dir() + "weights.bin";
  if (!file_exists(model_path)) {
    GTEST_SKIP() << "weights.bin not found at " << model_path;
  }

  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model(model_path, options);

  EXPECT_EQ(report.analyzer.status, vie::InspectionStatus::Failed);
  EXPECT_FALSE(report.analyzer.error.empty());
}

// Test text output format
TEST(ModelInspection, TextOutputContainsExpectedSections) {
  const std::string model_path = test_data_dir() + "yolov8n.onnx";
  if (!file_exists(model_path)) {
    GTEST_SKIP() << "yolov8n.onnx not found at " << model_path;
  }

  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model(model_path, options);
  std::string text = vie::inspection_report_to_text(report);

  EXPECT_NE(text.find("VIE Model Inspection"), std::string::npos);
  EXPECT_NE(text.find("Model:"), std::string::npos);
  EXPECT_NE(text.find("Analyzer:"), std::string::npos);
  EXPECT_NE(text.find("Recognition:"), std::string::npos);
  EXPECT_NE(text.find("Manifest:"), std::string::npos);
  EXPECT_NE(text.find("Execution:"), std::string::npos);
  EXPECT_NE(text.find("Diagnostic:"), std::string::npos);
}

// Test JSON output format
TEST(ModelInspection, JsonOutputContainsExpectedFields) {
  const std::string model_path = test_data_dir() + "yolov8n.onnx";
  if (!file_exists(model_path)) {
    GTEST_SKIP() << "yolov8n.onnx not found at " << model_path;
  }

  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model(model_path, options);
  std::string json = vie::inspection_report_to_json(report);

  EXPECT_NE(json.find("\"model\""), std::string::npos);
  EXPECT_NE(json.find("\"analyzer\""), std::string::npos);
  EXPECT_NE(json.find("\"recognition\""), std::string::npos);
  EXPECT_NE(json.find("\"manifest\""), std::string::npos);
  EXPECT_NE(json.find("\"execution\""), std::string::npos);
  EXPECT_NE(json.find("\"diagnostic\""), std::string::npos);
  EXPECT_NE(json.find("\"status\""), std::string::npos);
}

// Test diagnostic status values
TEST(ModelInspection, DiagnosticStatusValues) {
  // Analyzer failed
  {
    vie::ModelInspectionReport report;
    report.analyzer.status = vie::InspectionStatus::Failed;
    EXPECT_EQ(report.diagnostic_status(), "analyzer_failed");
  }

  // Recognition failed
  {
    vie::ModelInspectionReport report;
    report.analyzer.status = vie::InspectionStatus::Success;
    report.recognition.status = vie::InspectionStatus::Failed;
    EXPECT_EQ(report.diagnostic_status(), "recognition_failed");
  }

  // Manifest generation failed
  {
    vie::ModelInspectionReport report;
    report.analyzer.status = vie::InspectionStatus::Success;
    report.recognition.status = vie::InspectionStatus::Success;
    report.manifest.generated = vie::InspectionStatus::Failed;
    EXPECT_EQ(report.diagnostic_status(), "manifest_generation_failed");
  }

  // Manifest not validated
  {
    vie::ModelInspectionReport report;
    report.analyzer.status = vie::InspectionStatus::Success;
    report.recognition.status = vie::InspectionStatus::Success;
    report.manifest.generated = vie::InspectionStatus::Success;
    report.manifest.validated = false;
    EXPECT_EQ(report.diagnostic_status(), "manifest_invalid");
  }

  // Recognized but not runnable
  {
    vie::ModelInspectionReport report;
    report.analyzer.status = vie::InspectionStatus::Success;
    report.recognition.status = vie::InspectionStatus::Success;
    report.recognition.result.status = vie::RecognitionStatus::Recognized;
    report.manifest.generated = vie::InspectionStatus::Success;
    report.manifest.validated = true;
    EXPECT_EQ(report.diagnostic_status(), "recognized");
  }
}

// Test inspection status to_string
TEST(ModelInspection, StatusToString) {
  EXPECT_STREQ(vie::to_string(vie::InspectionStatus::NotRun), "not_run");
  EXPECT_STREQ(vie::to_string(vie::InspectionStatus::Success), "success");
  EXPECT_STREQ(vie::to_string(vie::InspectionStatus::Failed), "failed");

  EXPECT_STREQ(vie::to_string(vie::ExecutionStatus::NotTested), "not_tested");
  EXPECT_STREQ(vie::to_string(vie::ExecutionStatus::Passed), "passed");
  EXPECT_STREQ(vie::to_string(vie::ExecutionStatus::Failed), "failed");
}

// Test that model hash is computed
TEST(ModelInspection, ModelHashIsComputed) {
  const std::string model_path = test_data_dir() + "yolov8n.onnx";
  if (!file_exists(model_path)) {
    GTEST_SKIP() << "yolov8n.onnx not found at " << model_path;
  }

  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model(model_path, options);

  EXPECT_FALSE(report.model_sha256.empty());
  EXPECT_GT(report.model_bytes, 0u);
}

// Test that evidence is included in the report
TEST(ModelInspection, EvidenceIsIncluded) {
  const std::string model_path = test_data_dir() + "yolov8n.onnx";
  if (!file_exists(model_path)) {
    GTEST_SKIP() << "yolov8n.onnx not found at " << model_path;
  }

  vie::InspectionOptions options;
  vie::ModelInspectionReport report = vie::inspect_model(model_path, options);

  const vie::RecognitionCandidate* selected = report.recognition.result.selected();
  ASSERT_NE(selected, nullptr);
  EXPECT_FALSE(selected->evidence.empty());

  for (const auto& evidence : selected->evidence) {
    EXPECT_FALSE(evidence.code.empty());
    EXPECT_FALSE(evidence.detail.empty());
  }
}
