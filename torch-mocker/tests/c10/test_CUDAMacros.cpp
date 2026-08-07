// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAMacros.h>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAMacros) {
  TM_REQUIRE(C10_COMPILE_TIME_MAX_GPUS > 0);
}
