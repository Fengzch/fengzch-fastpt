// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch_mocker::test {

class TestFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class TestSkipped final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct TestRegistration {
  const char* selector;
  const char* name;
  void (*entry)();
};

std::vector<TestRegistration>& test_registry();

class TestRegistrar final {
 public:
  TestRegistrar(const char* selector, const char* name, void (*entry)()) {
    test_registry().push_back({selector, name, entry});
  }
};

inline void fail(
    const char* expression,
    const char* file,
    int line,
    const std::string& detail = {}) {
  std::ostringstream message;
  message << file << ':' << line << ": requirement failed: " << expression;
  if (!detail.empty()) {
    message << " (" << detail << ')';
  }
  throw TestFailure(message.str());
}

inline void require_cuda(
    cudaError_t status,
    const char* expression,
    const char* file,
    int line) {
  if (status != cudaSuccess) {
    fail(expression, file, line, cudaGetErrorString(status));
  }
}

} // namespace torch_mocker::test

#define TM_CONCAT_INNER(left, right) left##right
#define TM_CONCAT(left, right) TM_CONCAT_INNER(left, right)
#define TM_TEST(suite, name) TM_TEST_IMPL(suite, name, __COUNTER__)
#define TM_TEST_IMPL(suite, name, index)                                    \
  static_assert(                                                             \
      sizeof(TORCH_MOCKER_TEST_SELECTOR) > 1,                               \
      "TORCH_MOCKER_TEST_SELECTOR must be defined for every test source"); \
  static void TM_CONCAT(torch_mocker_test_, index)();                        \
  namespace {                                                                 \
  const ::torch_mocker::test::TestRegistrar                                  \
      TM_CONCAT(torch_mocker_test_registration_, index)(                      \
          TORCH_MOCKER_TEST_SELECTOR,                                        \
          #suite "." #name,                                                  \
          &TM_CONCAT(torch_mocker_test_, index));                             \
  }                                                                            \
  static void TM_CONCAT(torch_mocker_test_, index)()

#define TM_REQUIRE(condition)                                                \
  do {                                                                       \
    if (!(condition)) {                                                      \
      ::torch_mocker::test::fail(#condition, __FILE__, __LINE__);           \
    }                                                                        \
  } while (false)

#define TM_REQUIRE_EQ(actual, expected)                                      \
  do {                                                                       \
    const auto& tm_actual = (actual);                                        \
    const auto& tm_expected = (expected);                                    \
    if (!(tm_actual == tm_expected)) {                                       \
      ::torch_mocker::test::fail(                                            \
          #actual " == " #expected, __FILE__, __LINE__);                    \
    }                                                                        \
  } while (false)

#define TM_REQUIRE_NEAR(actual, expected, tolerance)                         \
  do {                                                                       \
    const auto tm_actual = (actual);                                         \
    const auto tm_expected = (expected);                                     \
    const auto tm_tolerance = (tolerance);                                   \
    if ((tm_actual < tm_expected - tm_tolerance) ||                          \
        (tm_actual > tm_expected + tm_tolerance)) {                          \
      ::torch_mocker::test::fail(                                            \
          #actual " ~= " #expected, __FILE__, __LINE__);                    \
    }                                                                        \
  } while (false)

#define TM_CUDA_REQUIRE(expression)                                          \
  ::torch_mocker::test::require_cuda((expression), #expression, __FILE__, __LINE__)

#define TM_SKIP(reason) throw ::torch_mocker::test::TestSkipped(reason)

#define TM_REQUIRE_THROWS(expression)                                         \
  do {                                                                         \
    bool tm_threw = false;                                                     \
    try {                                                                      \
      (void)(expression);                                                      \
    } catch (...) {                                                            \
      tm_threw = true;                                                         \
    }                                                                          \
    if (!tm_threw) {                                                           \
      ::torch_mocker::test::fail(                                              \
          "expected exception: " #expression, __FILE__, __LINE__);            \
    }                                                                          \
  } while (false)

#define TM_REQUIRE_NOTHROW(expression)                                        \
  do {                                                                         \
    try {                                                                      \
      (void)(expression);                                                      \
    } catch (const std::exception& tm_error) {                                \
      ::torch_mocker::test::fail(                                              \
          "unexpected exception: " #expression, __FILE__, __LINE__,           \
          tm_error.what());                                                    \
    }                                                                          \
  } while (false)
