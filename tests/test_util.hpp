#pragma once

#include "vie/core/error.hpp"

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <utility>

#define VIE_EXPECT_ERROR(expected_code, statement)                              \
  do {                                                                          \
    try {                                                                       \
      statement;                                                               \
      FAIL() << "expected VieError";                                            \
    } catch (const vie::VieError& vie_error) {                                  \
      EXPECT_EQ(vie_error.code(), expected_code) << vie_error.what();          \
    }                                                                           \
  } while (0)

inline std::string vie_fixture(const char* relative) {
  return std::string(VIE_SOURCE_DIR) + "/" + relative;
}

// The Ultralytics-derived fixtures are not distributed with this repository; see
// .gitignore and tools/export_test_models.py. A test that reads one of those ONNX
// files skips when the file is absent instead of failing, so a fresh clone still
// has a readable ctest run.
inline bool vie_model_available(const char* relative) {
  std::ifstream handle(vie_fixture(relative));
  return handle.good();
}

#define VIE_REQUIRE_MODEL(...)                                           \
  do {                                                                   \
    for (const char* vie_model : {__VA_ARGS__}) {                        \
      if (!vie_model_available(vie_model)) {                             \
        GTEST_SKIP() << vie_model << " is not present; rebuild it with " \
                     "'python3 tools/export_test_models.py'";            \
      }                                                                  \
    }                                                                    \
  } while (false)
