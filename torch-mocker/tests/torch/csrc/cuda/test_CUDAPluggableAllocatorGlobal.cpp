// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include <cstddef>
#include <memory>

#include "../../../support/TestHarness.h"

namespace {

struct CallbackState {
  size_t allocation_calls = 0;
  size_t release_calls = 0;
  size_t last_size = 0;
  int last_device = -1;
  cudaStream_t last_stream = nullptr;
};

CallbackState state;

void* allocate(size_t size, int device, cudaStream_t stream) {
  ++state.allocation_calls;
  state.last_size = size;
  state.last_device = device;
  state.last_stream = stream;
  void* pointer = nullptr;
  if (cudaMalloc(&pointer, size) != cudaSuccess) {
    return nullptr;
  }
  return pointer;
}

void release(void* pointer, size_t size, int device, cudaStream_t stream) {
  ++state.release_calls;
  state.last_size = size;
  state.last_device = device;
  state.last_stream = stream;
  (void)cudaFree(pointer);
}

} // namespace

// Each CTest selector runs in a separate process, so this test can own the
// process-global allocator without affecting the normal allocator tests.
TM_TEST(TorchCUDA, CUDAPluggableAllocatorGlobal) {
  constexpr size_t kAllocationSize = 8192;
  state = {};

  TM_REQUIRE_EQ(
      torch::cuda::CUDAPluggableAllocator::getCurrentAllocator(), nullptr);
  TM_REQUIRE(!c10::cuda::CUDACachingAllocator::get()->initialized());

  auto allocator = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
      allocate, release);
  auto custom = std::dynamic_pointer_cast<
      torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(allocator);
  TM_REQUIRE(custom != nullptr);

  torch::cuda::CUDAPluggableAllocator::changeCurrentAllocator(allocator);
  TM_REQUIRE_EQ(
      torch::cuda::CUDAPluggableAllocator::getCurrentAllocator(), allocator);
  TM_REQUIRE_EQ(c10::cuda::CUDACachingAllocator::get(), custom.get());

  {
    auto allocation = c10::cuda::CUDACachingAllocator::get()->allocate(
        kAllocationSize);
    TM_REQUIRE(allocation.get() != nullptr);
    TM_REQUIRE_EQ(state.allocation_calls, 1U);
    TM_REQUIRE_EQ(state.last_size, kAllocationSize);
    TM_REQUIRE_EQ(state.last_device, 0);
    TM_REQUIRE_EQ(
        state.last_stream,
        c10::cuda::getCurrentCUDAStream(state.last_device).stream());
  }

  TM_REQUIRE_EQ(state.release_calls, 1U);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
}
