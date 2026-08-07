// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAMiscFunctions.h>
#include <mutex>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAMiscFunctions) {
  const char* suffix = c10::cuda::get_cuda_check_suffix();
  TM_REQUIRE(suffix != nullptr);
  TM_REQUIRE(
      c10::cuda::get_cuda_error_help(cudaErrorInvalidValue).find(
          cudaGetErrorName(cudaErrorInvalidValue)) != std::string::npos);
  TM_REQUIRE(
      c10::cuda::get_cuda_error_help(cudaErrorInvalidDevice).find("out of range") !=
      std::string::npos);
  std::mutex* free_mutex = c10::cuda::getFreeMutex();
  TM_REQUIRE(free_mutex != nullptr);
  TM_REQUIRE_EQ(c10::cuda::getFreeMutex(), free_mutex);
  std::unique_lock<std::mutex> lock(*free_mutex);
  TM_REQUIRE(lock.owns_lock());
}
