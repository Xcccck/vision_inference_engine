#include "test_util.hpp"

#include "vie/model/manifest.hpp"
#include "vie/model/manifest_validator.hpp"

#include <filesystem>

namespace {

vie::ModelManifest load_mock() {
  return vie::ModelManifest::load_file(vie_fixture("configs/mock_detection.json"));
}

bool has_issue(const std::vector<vie::SchemaIssue>& issues, const std::string& path) {
  for (const vie::SchemaIssue& issue : issues) {
    if (issue.path == path) return true;
  }
  return false;
}

}  // namespace

TEST(ManifestTest, FixtureIsRunnableAndLeavesNoncriticalFieldsOpen) {
  const vie::ModelManifest manifest = load_mock();
  const vie::ManifestValidation report = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(report.schema_ok()) << vie::format_issues(report.schema_issues);
  EXPECT_TRUE(report.runnable()) << vie::format_issues(report.unresolved_critical);

  const vie::FieldProvenance* family = manifest.find_provenance("model_family");
  ASSERT_NE(family, nullptr);
  EXPECT_EQ(family->certainty, vie::Certainty::Unresolved);
  EXPECT_FALSE(manifest.nms.enabled.has_value());
  EXPECT_TRUE(manifest.outputs.at(0).semantic.empty());
  EXPECT_EQ(manifest.outputs.at(0).layout, vie::Layout::Unknown);

  const vie::FieldProvenance* color = manifest.find_provenance("input.color");
  ASSERT_NE(color, nullptr);
  EXPECT_EQ(color->certainty, vie::Certainty::User);
  ASSERT_EQ(color->evidence.size(), 1u);
  EXPECT_EQ(color->evidence[0], "manifest value without provenance");
}

TEST(ManifestTest, PartialManifestDoesNotInventCriticalValues) {
  vie::ModelManifest manifest;
  manifest.schema_version = "1.0";
  manifest.model_name = "partial";
  const vie::ManifestValidation report = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(report.schema_ok()) << vie::format_issues(report.schema_issues);
  EXPECT_FALSE(report.runnable());
  EXPECT_TRUE(has_issue(report.unresolved_critical, "input.color"));
  EXPECT_TRUE(has_issue(report.unresolved_critical, "decoder.protocol"));
  EXPECT_TRUE(has_issue(report.unresolved_critical, "task"));
  EXPECT_TRUE(manifest.input.color.empty());
  EXPECT_EQ(manifest.task, vie::TaskType::Unknown);
}

TEST(ManifestTest, DynamicShapeStaysObservedAndBlocksExecution) {
  constexpr char kJson[] = R"({
    "schema_version": "1.0",
    "input": {"shape": [1, 3, -1, -1]},
    "provenance": {
      "input.shape": {"certainty": "observed", "evidence": ["graph input"]}
    }
  })";
  const vie::ModelManifest manifest = vie::ModelManifest::from_json_string(kJson);
  const vie::FieldProvenance* shape = manifest.find_provenance("input.shape");
  ASSERT_NE(shape, nullptr);
  EXPECT_EQ(shape->certainty, vie::Certainty::Observed);
  ASSERT_EQ(shape->evidence.size(), 1u);
  EXPECT_EQ(shape->evidence[0], "graph input");

  const vie::ManifestValidation report = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(report.schema_ok()) << vie::format_issues(report.schema_issues);
  EXPECT_FALSE(report.runnable());
  EXPECT_TRUE(has_issue(report.unresolved_critical, "input.shape"));
}

TEST(ManifestTest, RoundTripPreservesExplicitProvenance) {
  vie::ModelManifest manifest = load_mock();
  manifest.provenance["input.shape"] =
      vie::FieldProvenance{vie::Certainty::Observed, {"graph input"}};
  const auto path = std::filesystem::temp_directory_path() / "vie_phase1_roundtrip.json";
  manifest.save_file(path.string());
  const vie::ModelManifest loaded = vie::ModelManifest::load_file(path.string());
  std::filesystem::remove(path);

  EXPECT_EQ(loaded.model_name, "mock_det");
  EXPECT_EQ(loaded.task, vie::TaskType::Detection);
  EXPECT_EQ(loaded.model_format, vie::ModelFormat::Mock);
  EXPECT_EQ(loaded.input.color, "rgb");
  EXPECT_EQ(loaded.input.shape, (std::vector<std::int64_t>{1, 3, 8, 8}));
  EXPECT_EQ(loaded.decoder.protocol, "mock");
  EXPECT_EQ(loaded.num_classes, 1);
  const vie::FieldProvenance* shape = loaded.find_provenance("input.shape");
  ASSERT_NE(shape, nullptr);
  EXPECT_EQ(shape->certainty, vie::Certainty::Observed);
  ASSERT_EQ(shape->evidence.size(), 1u);
  EXPECT_EQ(shape->evidence[0], "graph input");
}

TEST(ManifestTest, RejectsBadDocuments) {
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError, vie::ModelManifest::from_json_string("{"));
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   vie::ModelManifest::from_json_string(R"({"schema_version":"9.0"})"));
  VIE_EXPECT_ERROR(
      vie::ErrorCode::SchemaError,
      vie::ModelManifest::from_json_string(
          R"({"schema_version":"1.0","input":{"color":"rgb"},"provenance":{"input.color":{"certainty":"unresolved","evidence":[]}}})"));
  VIE_EXPECT_ERROR(vie::ErrorCode::SchemaError,
                   vie::ModelManifest::from_json_string(
                       R"({"schema_version":"1.0","input":{"color":"rgba"}})"));
  VIE_EXPECT_ERROR(vie::ErrorCode::InvalidArgument,
                   vie::ModelManifest::load_file(vie_fixture("configs/missing.json")));
}

TEST(ManifestTest, DuplicateOutputNamesAndEnabledNmsAreReported) {
  vie::ModelManifest manifest = load_mock();
  manifest.outputs.push_back(manifest.outputs.front());
  vie::ManifestValidation report = vie::ManifestValidator{}.validate(manifest);
  EXPECT_FALSE(report.schema_ok());
  EXPECT_TRUE(has_issue(report.schema_issues, "outputs.1.name"));

  manifest = load_mock();
  manifest.nms.enabled = true;
  manifest.provenance["nms.enabled"] = vie::FieldProvenance{vie::Certainty::User, {"test"}};
  report = vie::ManifestValidator{}.validate(manifest);
  EXPECT_TRUE(report.schema_ok()) << vie::format_issues(report.schema_issues);
  EXPECT_FALSE(report.runnable());
  EXPECT_TRUE(has_issue(report.unresolved_critical, "nms.confidence_threshold"));
  EXPECT_TRUE(has_issue(report.unresolved_critical, "nms.iou_threshold"));
}
