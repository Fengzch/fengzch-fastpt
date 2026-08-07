// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <sstream>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAGraphsC10Utils) {
  const auto status = c10::cuda::currentStreamCaptureStatusMayInitCtx();
  TM_REQUIRE(
      status == c10::cuda::CaptureStatus::None ||
      status == c10::cuda::CaptureStatus::Active ||
      status == c10::cuda::CaptureStatus::Invalidated);
  TM_REQUIRE(!c10::cuda::currentStreamCaptureIdMayInitCtx().has_value());

  c10::cuda::CUDAStreamCaptureModeGuard capture_mode(cudaStreamCaptureModeRelaxed);
  cudaStream_t stream = nullptr;
  TM_CUDA_REQUIRE(cudaStreamCreate(&stream));
  TM_REQUIRE_EQ(c10::cuda::captureStatusMayInitCtx(stream), c10::cuda::CaptureStatus::None);
  TM_REQUIRE(!c10::cuda::captureIdMayInitCtx(stream).has_value());
  TM_CUDA_REQUIRE(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  TM_REQUIRE_EQ(c10::cuda::captureStatusMayInitCtx(stream), c10::cuda::CaptureStatus::Active);
  TM_REQUIRE(c10::cuda::isStreamCapturingMayInitCtx(stream));
  TM_REQUIRE(c10::cuda::captureIdMayInitCtx(stream).has_value());
  cudaGraph_t graph = nullptr;
  TM_CUDA_REQUIRE(cudaStreamEndCapture(stream, &graph));
  TM_REQUIRE(graph != nullptr);
  TM_CUDA_REQUIRE(cudaGraphDestroy(graph));
  TM_CUDA_REQUIRE(cudaStreamDestroy(stream));

  std::ostringstream output;
  output << c10::cuda::CaptureStatus::None;
  TM_REQUIRE_EQ(output.str(), std::string("cudaStreamCaptureStatusNone"));
}
