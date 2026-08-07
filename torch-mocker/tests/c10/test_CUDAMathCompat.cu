// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "../support/TestHarness.h"

__global__ void gpu_math_compat(
    float* result_float,
    double* result_double,
    float* positive_zero) {
  *result_float = c10::cuda::compat::copysign(1.0f, -1.0f);
  *result_double = c10::cuda::compat::copysign(3.14, -1.0);
  *positive_zero = c10::cuda::compat::copysign(0.0f, 1.0f);
}

TM_TEST(C10, CUDAMathCompat) {
  float* result_float = nullptr;
  double* result_double = nullptr;
  float* positive_zero = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&result_float, sizeof(float)));
  TM_CUDA_REQUIRE(cudaMalloc(&result_double, sizeof(double)));
  TM_CUDA_REQUIRE(cudaMalloc(&positive_zero, sizeof(float)));

  gpu_math_compat<<<1, 1>>>(result_float, result_double, positive_zero);
  TM_CUDA_REQUIRE(cudaGetLastError());
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());

  float host_float = 0.0f;
  double host_double = 0.0;
  float host_positive_zero = -1.0f;
  TM_CUDA_REQUIRE(cudaMemcpy(&host_float, result_float, sizeof(float), cudaMemcpyDeviceToHost));
  TM_CUDA_REQUIRE(cudaMemcpy(&host_double, result_double, sizeof(double), cudaMemcpyDeviceToHost));
  TM_CUDA_REQUIRE(cudaMemcpy(
      &host_positive_zero,
      positive_zero,
      sizeof(float),
      cudaMemcpyDeviceToHost));
  TM_REQUIRE_NEAR(host_float, -1.0f, 1e-6f);
  TM_REQUIRE_NEAR(host_double, -3.14, 1e-12);
  TM_REQUIRE_EQ(host_positive_zero, 0.0f);
  TM_CUDA_REQUIRE(cudaFree(result_float));
  TM_CUDA_REQUIRE(cudaFree(result_double));
  TM_CUDA_REQUIRE(cudaFree(positive_zero));
}
