#pragma once

#include "vie/model/manifest.hpp"

#include <string>
#include <vector>

namespace vie {

struct SchemaIssue {
  std::string path;
  std::string message;
};

// schema_issues: the document is ill-formed.
// unresolved_critical: the document is well-formed and a correctness-critical
// field is still open, or a dynamic axis has no concrete profile.
struct ManifestValidation {
  std::vector<SchemaIssue> schema_issues;
  std::vector<SchemaIssue> unresolved_critical;

  bool schema_ok() const { return schema_issues.empty(); }
  bool runnable() const { return schema_issues.empty() && unresolved_critical.empty(); }
};

class ManifestValidator {
 public:
  ManifestValidation validate(const ModelManifest& manifest) const;
};

std::string format_issues(const std::vector<SchemaIssue>& issues);

}  // namespace vie
