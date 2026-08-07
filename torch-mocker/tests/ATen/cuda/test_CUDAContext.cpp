// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAFunctions.h>

#include <array>
#include <thread>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDAContext) {
  const auto initializer = at::zeros({1}, at::device(at::kCUDA));
  (void)initializer;
  const auto device_count = at::cuda::getNumGPUs();
  TM_REQUIRE(device_count > 0);
  TM_REQUIRE(at::cuda::is_available());
  const auto* current_properties = at::cuda::getCurrentDeviceProperties();
  const auto* first_properties = at::cuda::getDeviceProperties(0);
  TM_REQUIRE(current_properties != nullptr);
  TM_REQUIRE(first_properties != nullptr);
  TM_REQUIRE(current_properties->multiProcessorCount > 0);
  TM_REQUIRE(current_properties->totalGlobalMem > 0);
  TM_REQUIRE_EQ(
      at::cuda::getCurrentCUDAStream().device_index(),
      c10::cuda::current_device());

  TM_REQUIRE(at::cuda::getCurrentCUDABlasHandle() != nullptr);
  TM_REQUIRE(at::cuda::getCurrentCUDASparseHandle() != nullptr);
  TM_REQUIRE(at::cuda::getCurrentCUDASolverDnHandle() != nullptr);

  std::array<cusolverDnHandle_t, 2> solver_handles{};
  std::array<cudaError_t, 2> thread_status{};
  std::array<std::thread, 2> threads;
  for (size_t index = 0; index < threads.size(); ++index) {
    threads[index] = std::thread([&solver_handles, &thread_status, index] {
      thread_status[index] = cudaSetDevice(0);
      if (thread_status[index] == cudaSuccess) {
        solver_handles[index] = at::cuda::getCurrentCUDASolverDnHandle();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (size_t index = 0; index < solver_handles.size(); ++index) {
    TM_CUDA_REQUIRE(thread_status[index]);
    TM_REQUIRE(solver_handles[index] != nullptr);
  }

  if (device_count >= 2) {
    (void)at::cuda::canDeviceAccessPeer(0, 1);
    (void)at::cuda::canDeviceAccessPeer(1, 0);
  }
}
