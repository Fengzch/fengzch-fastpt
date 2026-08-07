// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/Sleep.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, Sleep) {
  for (int iteration = 0; iteration < 16; ++iteration) {
    at::cuda::flush_icache();
  }
  TM_CUDA_REQUIRE(cudaGetLastError());
}
