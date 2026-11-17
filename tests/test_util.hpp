#pragma once

#include "vie/core/error.hpp"

#include <gtest/gtest.h>
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
