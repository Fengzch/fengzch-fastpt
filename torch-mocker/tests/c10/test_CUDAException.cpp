// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <string>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAException) {
  c10::cuda::c10_cuda_check_implementation(
      cudaSuccess, __FILE__, __func__, __LINE__, false);

  bool threw = false;
  try {
    c10::cuda::c10_cuda_check_implementation(
        cudaErrorInvalidValue, __FILE__, __func__, __LINE__, false);
  } catch (const c10::Error& error) {
    threw = true;
    const std::string message = error.what();
    TM_REQUIRE(message.find("CUDA error") != std::string::npos);
    TM_REQUIRE(
        message.find(cudaGetErrorString(cudaErrorInvalidValue)) !=
        std::string::npos);
    TM_REQUIRE(
        message.find("Device-side assertions were explicitly omitted") !=
        std::string::npos);
  }
  TM_REQUIRE(threw);

  C10_CUDA_CLEAR_ERROR();
  C10_CUDA_IGNORE_ERROR(cudaErrorInvalidValue);
  C10_CUDA_CLEAR_ERROR();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
