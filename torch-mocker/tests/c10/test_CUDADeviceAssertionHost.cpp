// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDADeviceAssertionHost.h>

#include <algorithm>
#include <string>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDADeviceAssertionHost) {
  auto& registry = c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref();
  const uint32_t launch_id = registry.insert(
      "test_file.cu", "test_function", 42, "test_kernel", 0);
  const auto snapshot = registry.snapshot();
  if (registry.enabled_at_runtime) {
    const auto launch = std::find_if(
        snapshot.second.begin(),
        snapshot.second.end(),
        [launch_id](const auto& candidate) {
          return candidate.generation_number == launch_id;
        });
    TM_REQUIRE(launch != snapshot.second.end());
    TM_REQUIRE(launch->launch_filename != nullptr);
    TM_REQUIRE(launch->launch_function != nullptr);
    TM_REQUIRE(launch->kernel_name != nullptr);
    TM_REQUIRE_EQ(std::string(launch->launch_filename), std::string("test_file.cu"));
    TM_REQUIRE_EQ(std::string(launch->launch_function), std::string("test_function"));
    TM_REQUIRE_EQ(launch->launch_linenum, 42U);
    TM_REQUIRE_EQ(std::string(launch->kernel_name), std::string("test_kernel"));
    TM_REQUIRE_EQ(launch->device, 0);
  } else {
    TM_REQUIRE_EQ(launch_id, 0U);
    TM_REQUIRE(std::all_of(
        snapshot.second.begin(),
        snapshot.second.end(),
        [](const auto& launch) { return launch.launch_filename == nullptr; }));
  }
  TM_REQUIRE(!registry.has_failed());

  const std::string report = c10::cuda::c10_retrieve_device_side_assertion_info();
  TM_REQUIRE(!report.empty());
}
