// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "TestHarness.h"

#include <exception>
#include <iostream>

namespace torch_mocker::test {

std::vector<TestRegistration>& test_registry() {
  static std::vector<TestRegistration> registrations;
  return registrations;
}

} // namespace torch_mocker::test

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "expected one test selector\n";
      return 2;
    }

    const std::string selector(argv[1]);
    const auto& registrations = torch_mocker::test::test_registry();
    for (const auto& registration : registrations) {
      if (selector == registration.selector) {
        registration.entry();
        const auto last_error = cudaGetLastError();
        if (last_error != cudaSuccess) {
          std::cerr << "unreported CUDA error: "
                    << cudaGetErrorString(last_error) << '\n';
          return 1;
        }
        return 0;
      }
    }

    std::cerr << "unknown test selector: " << selector << '\n';
    return 2;
  } catch (const torch_mocker::test::TestSkipped& error) {
    std::cerr << "skipped: " << error.what() << '\n';
    return 77;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
  } catch (...) {
    std::cerr << "unknown test failure\n";
  }
  return 1;
}
