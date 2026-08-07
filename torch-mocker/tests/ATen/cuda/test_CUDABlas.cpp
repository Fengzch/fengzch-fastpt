// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDABlas.h>
#include <ATen/Functions.h>

#include <array>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDABlas) {
  // cuBLAS uses column-major storage.  These buffers represent
  // A=[[1,2],[3,4]] and B=[[5,6],[7,8]].
  const std::array<float, 4> host_a = {1.0f, 3.0f, 2.0f, 4.0f};
  const std::array<float, 4> host_b = {5.0f, 7.0f, 6.0f, 8.0f};
  const auto initializer = at::zeros({1}, at::device(at::kCUDA));
  (void)initializer;
  float *a = nullptr, *b = nullptr, *c = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&a, sizeof(host_a)));
  TM_CUDA_REQUIRE(cudaMalloc(&b, sizeof(host_b)));
  TM_CUDA_REQUIRE(cudaMalloc(&c, sizeof(host_a)));
  TM_CUDA_REQUIRE(cudaMemcpy(a, host_a.data(), sizeof(host_a), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(b, host_b.data(), sizeof(host_b), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemset(c, 0, sizeof(host_a)));
  at::cuda::blas::gemm<float>('n', 'n', 2, 2, 2, 1.0f, a, 2, b, 2, 0.0f, c, 2);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  std::array<float, 4> host_c{};
  TM_CUDA_REQUIRE(cudaMemcpy(host_c.data(), c, sizeof(host_c), cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(host_c, (std::array<float, 4>{19.0f, 43.0f, 22.0f, 50.0f}));
  at::cuda::blas::gemm<float>('n', 'n', 2, 2, 2, 0.0f, a, 2, b, 2, 2.0f, c, 2);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_CUDA_REQUIRE(cudaMemcpy(host_c.data(), c, sizeof(host_c), cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(host_c, (std::array<float, 4>{38.0f, 86.0f, 44.0f, 100.0f}));
  TM_CUDA_REQUIRE(cudaFree(a));
  TM_CUDA_REQUIRE(cudaFree(b));
  TM_CUDA_REQUIRE(cudaFree(c));
}
