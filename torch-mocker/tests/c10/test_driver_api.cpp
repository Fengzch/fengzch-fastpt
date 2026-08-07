// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/driver_api.h>

#include "../support/TestHarness.h"

TM_TEST(C10, DriverApi) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  C10_CUDA_DRIVER_CHECK(cuInit(0));
  TM_REQUIRE(c10::cuda::device_count_ensure_non_zero() > 0);
#else
  TM_SKIP("C10 CUDA driver API is not enabled for this build");
#endif
}
