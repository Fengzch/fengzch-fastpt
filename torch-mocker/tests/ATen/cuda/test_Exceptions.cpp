// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/ThrustAllocator.h>

#include <cublas_v2.h>
#include <cusolver_common.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, Exceptions) {
  TM_REQUIRE_EQ(
      at::cuda::blas::_cublasGetErrorEnum(CUBLAS_STATUS_SUCCESS),
      std::string("CUBLAS_STATUS_SUCCESS"));
  TM_REQUIRE_EQ(
      at::cuda::solver::cusolverGetErrorMessage(CUSOLVER_STATUS_SUCCESS),
      std::string("CUSOLVER_STATUS_SUCCESS"));
  TORCH_CUSOLVER_CHECK(CUSOLVER_STATUS_SUCCESS);
  TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_SUCCESS);

  at::cuda::CUDACachingAllocator::init(1);
  at::cuda::ThrustAllocator allocator;
  char* allocation = allocator.allocate(1024);
  TM_REQUIRE(allocation != nullptr);
  allocator.deallocate(allocation, 1024);
}
