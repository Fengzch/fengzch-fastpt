// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDAEvent) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAEvent start(cudaEventDefault);
  at::cuda::CUDAEvent stop(cudaEventDefault);
  TM_REQUIRE(start.event() == nullptr);
  TM_REQUIRE_EQ(start.device_index(), -1);
  start.recordOnce(stream);
  TM_REQUIRE(start.event() != nullptr);
  TM_REQUIRE_EQ(start.device_index(), stream.device_index());
  start.recordOnce(stream);
  stop.record(stream);
  stop.synchronize();
  TM_REQUIRE(stop.query());
  TM_REQUIRE_EQ(stop.device_index(), stream.device_index());
  TM_REQUIRE(start.elapsed_time(stop) >= 0.0f);
  TM_REQUIRE(stop.event() != nullptr);
  stop.block(stream);
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream.stream()));
}
