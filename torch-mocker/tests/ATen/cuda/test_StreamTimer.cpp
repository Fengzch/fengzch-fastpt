// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/tunable/StreamTimer.h>
#include <c10/cuda/CUDAStream.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, StreamTimer) {
  float* allocation = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&allocation, 1024 * sizeof(float)));
  at::cuda::tunable::StreamTimer timer;
  timer.Start();
  TM_CUDA_REQUIRE(cudaMemsetAsync(
      allocation,
      0,
      1024 * sizeof(float),
      c10::cuda::getCurrentCUDAStream().stream()));
  timer.End();
  const auto duration = timer.Duration();
  TM_REQUIRE(duration >= 0.0f);
  TM_REQUIRE_NEAR(timer.Duration(), duration, 1e-6f);
  TM_CUDA_REQUIRE(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream()));
  TM_CUDA_REQUIRE(cudaFree(allocation));
}
