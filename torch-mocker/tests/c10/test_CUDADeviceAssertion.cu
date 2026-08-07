// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>
#include <c10/cuda/CUDADeviceAssertion.h>

#include "../support/TestHarness.h"

namespace c10::cuda {
__global__ void kernel_assertion_test(DeviceAssertionsData* assertions_data) {
  if ((threadIdx.x + blockIdx.x * blockDim.x) % 3 == 0) {
    dsa_add_new_assertion_failure(
        assertions_data,
        "test assertion failure",
        __FILE__,
        __func__,
        __LINE__,
        0xDEADBEEF,
        blockIdx,
        threadIdx);
  }
}
} // namespace c10::cuda

TM_TEST(C10, CUDADeviceAssertion) {
  c10::cuda::DeviceAssertionsData* device_assertions = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_assertions, sizeof(*device_assertions)));
  TM_CUDA_REQUIRE(cudaMemset(device_assertions, 0, sizeof(*device_assertions)));
  c10::cuda::kernel_assertion_test<<<4, 4>>>(device_assertions);
  TM_CUDA_REQUIRE(cudaGetLastError());
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());

  c10::cuda::DeviceAssertionsData host_assertions{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      &host_assertions,
      device_assertions,
      sizeof(host_assertions),
      cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(host_assertions.assertion_count, 6);
  TM_CUDA_REQUIRE(cudaFree(device_assertions));
}
