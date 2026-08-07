// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/csrc/profiler/stubs/base.h>

#include "../support/TestHarness.h"

TM_TEST(Csrc, ProfilerStubs) {
  c10::DeviceIndex device = 0;
  int64_t first_timestamp = 0;
  int64_t second_timestamp = 0;
  torch::profiler::impl::ProfilerVoidEventStub first_event = nullptr;
  torch::profiler::impl::ProfilerVoidEventStub second_event = nullptr;
  torch::profiler::impl::cudaStubs()->record(&device, &first_event, &first_timestamp);
  torch::profiler::impl::cudaStubs()->record(&device, &second_event, &second_timestamp);
  TM_REQUIRE(second_timestamp >= first_timestamp);
  TM_REQUIRE(torch::profiler::impl::cudaStubs()->elapsed(&first_event, &second_event) >= 0.0f);
}
