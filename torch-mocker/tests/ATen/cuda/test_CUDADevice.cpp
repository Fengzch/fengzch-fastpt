// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDADevice) {
  const auto original_device = at::cuda::current_device();
  at::cuda::set_device(0);
  void* allocation = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&allocation, 1024));
  const auto allocation_device = at::cuda::getDeviceFromPtr(allocation);
  TM_REQUIRE_EQ(allocation_device.type(), c10::DeviceType::CUDA);
  TM_REQUIRE_EQ(allocation_device.index(), 0);
  TM_CUDA_REQUIRE(cudaFree(allocation));
  int host_value = 0;
  TM_REQUIRE_THROWS(at::cuda::getDeviceFromPtr(&host_value));
  at::cuda::set_device(original_device);
}
