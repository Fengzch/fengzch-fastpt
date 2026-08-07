// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/native/cuda/jit_utils.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenNative, JitUtils) {
  at::cuda::jit::NvrtcFunction function;
  TM_REQUIRE(function.function == nullptr);
}
