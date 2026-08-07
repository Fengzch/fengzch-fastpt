// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cudnn/Handle.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCuDNN, Handle) {
  auto handle = at::native::getCudnnHandle();
  TM_REQUIRE(handle != nullptr);

  const auto stream = at::cuda::getCurrentCUDAStream();
  TM_REQUIRE_EQ(cudnnSetStream(handle, stream.stream()), CUDNN_STATUS_SUCCESS);
  cudaStream_t bound_stream = nullptr;
  TM_REQUIRE_EQ(cudnnGetStream(handle, &bound_stream), CUDNN_STATUS_SUCCESS);
  TM_REQUIRE_EQ(bound_stream, stream.stream());
}
