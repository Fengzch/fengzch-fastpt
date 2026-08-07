// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/impl/CUDATest.h>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDATest) {
  const auto device_count = c10::cuda::device_count_ensure_non_zero();
  const int current_device = c10::cuda::impl::c10_cuda_test();
  int runtime_device = -1;
  TM_CUDA_REQUIRE(cudaGetDevice(&runtime_device));
  TM_REQUIRE(current_device >= 0);
  TM_REQUIRE(current_device < device_count);
  TM_REQUIRE_EQ(current_device, runtime_device);
}
