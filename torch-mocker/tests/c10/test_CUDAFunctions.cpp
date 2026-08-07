// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAMiscFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <vector>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAFunctions) {
  const auto device_count = c10::cuda::device_count_ensure_non_zero();
  TM_REQUIRE(device_count > 0);
  const auto original_device = c10::cuda::current_device();
  c10::cuda::set_device(0);
  TM_REQUIRE_EQ(c10::cuda::current_device(), 0);
  int runtime_device_count = 0;
  c10::cuda::GetDeviceCount(&runtime_device_count);
  TM_REQUIRE_EQ(runtime_device_count, device_count);
  c10::DeviceIndex runtime_device = -1;
  c10::cuda::GetDevice(&runtime_device);
  TM_REQUIRE_EQ(runtime_device, 0);
  (void)c10::cuda::hasPrimaryContext(runtime_device);

  const std::vector<int> input = {1, 2, 3, 4, 5};
  std::vector<int> output(input.size(), 0);
  int* device_data = nullptr;
  cudaStream_t stream = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_data, input.size() * sizeof(int)));
  TM_CUDA_REQUIRE(cudaStreamCreate(&stream));
  c10::cuda::memcpy_and_sync(device_data, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
  c10::cuda::memcpy_and_sync(output.data(), device_data, output.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream));
  TM_REQUIRE_EQ(output, input);
  c10::cuda::stream_synchronize(stream);
  TM_CUDA_REQUIRE(cudaFree(device_data));
  TM_CUDA_REQUIRE(cudaStreamDestroy(stream));
  const auto primary_context_device = c10::cuda::getDeviceIndexWithPrimaryContext();
  TM_REQUIRE(primary_context_device >= 0);
  TM_REQUIRE(primary_context_device < device_count);
  c10::cuda::device_synchronize();
  c10::cuda::set_device(original_device);
}
