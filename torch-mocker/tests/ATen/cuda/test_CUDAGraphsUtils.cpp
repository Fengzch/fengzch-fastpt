// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDAGraphsUtils) {
  cudaStream_t stream = nullptr;
  TM_CUDA_REQUIRE(cudaStreamCreate(&stream));
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  TM_CUDA_REQUIRE(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  TM_CUDA_REQUIRE(cudaStreamGetCaptureInfo(stream, &status, nullptr));
  TM_REQUIRE_EQ(status, cudaStreamCaptureStatusActive);
  cudaGraph_t graph = nullptr;
  TM_CUDA_REQUIRE(cudaStreamEndCapture(stream, &graph));
  TM_REQUIRE(graph != nullptr);
  TM_CUDA_REQUIRE(cudaGraphDestroy(graph));
  TM_CUDA_REQUIRE(cudaStreamDestroy(stream));
}
