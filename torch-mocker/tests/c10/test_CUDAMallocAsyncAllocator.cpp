// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include "../support/TestHarness.h"

TM_TEST(C10, CudaMallocAsyncAllocator) {
  namespace allocator_ns = c10::cuda::CUDACachingAllocator;

  auto* allocator = c10::cuda::CUDACachingAllocator::get();
  TM_REQUIRE(allocator != nullptr);
  TM_REQUIRE_EQ(std::string(allocator->name()), std::string("cudaMallocAsync"));

  allocator->init(c10::cuda::device_count_ensure_non_zero());
  const auto memory_fraction = allocator_ns::getMemoryFraction(0);
  TM_REQUIRE(memory_fraction > 0.0);
  allocator_ns::setMemoryFraction(1.0, 0);
  TM_REQUIRE_NEAR(allocator_ns::getMemoryFraction(0), 1.0, 1e-12);

  const auto memory_info = allocator->getMemoryInfo(0);
  TM_REQUIRE(memory_info.second > 0);
  TM_REQUIRE(memory_info.first <= memory_info.second);

  constexpr size_t kAllocationSize = 4096;
  constexpr size_t kCopySize = 256;
  auto stream = c10::cuda::getStreamFromPool(false, 0);
  void* ptr = allocator_ns::raw_alloc_with_stream(
      kAllocationSize, stream.stream());
  TM_REQUIRE(ptr != nullptr);
  size_t base_allocation_size = 0;
  TM_REQUIRE(allocator_ns::getBaseAllocation(ptr, &base_allocation_size) == ptr);
  TM_REQUIRE(base_allocation_size >= kAllocationSize);
  TM_CUDA_REQUIRE(cudaMemsetAsync(ptr, 0, kAllocationSize, stream.stream()));
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream.stream()));
  allocator_ns::raw_delete(ptr);

  auto recorded_pointer = allocator->allocate(kAllocationSize);
  TM_REQUIRE(recorded_pointer.get() != nullptr);
  allocator_ns::recordStream(recorded_pointer, stream);
  TM_CUDA_REQUIRE(cudaMemsetAsync(
      recorded_pointer.get(), 0, kAllocationSize, stream.stream()));
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream.stream()));

  void* copy_source = allocator_ns::raw_alloc_with_stream(
      kCopySize, stream.stream());
  void* copy_destination = allocator_ns::raw_alloc_with_stream(
      kCopySize, stream.stream());
  TM_REQUIRE(copy_source != nullptr);
  TM_REQUIRE(copy_destination != nullptr);
  TM_CUDA_REQUIRE(cudaMemsetAsync(copy_source, 0xA5, kCopySize, stream.stream()));
  TM_CUDA_REQUIRE(allocator_ns::memcpyAsync(
      copy_destination,
      0,
      copy_source,
      0,
      kCopySize,
      stream.stream(),
      false));
  std::array<std::uint8_t, kCopySize> host_copy{};
  TM_CUDA_REQUIRE(cudaMemcpyAsync(
      host_copy.data(),
      copy_destination,
      host_copy.size(),
      cudaMemcpyDeviceToHost,
      stream.stream()));
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream.stream()));
  TM_REQUIRE(std::all_of(
      host_copy.begin(),
      host_copy.end(),
      [](std::uint8_t value) { return value == 0xA5; }));
  allocator_ns::raw_delete(copy_destination);
  allocator_ns::raw_delete(copy_source);

  const auto stats = allocator_ns::getDeviceStats(0);
  TM_REQUIRE(stats.allocated_bytes[0].peak >= stats.allocated_bytes[0].current);
  allocator_ns::resetPeakStats(0);
  allocator_ns::resetAccumulatedStats(0);

  size_t largest_block = 0;
  allocator_ns::cacheInfo(0, &largest_block);
  TM_REQUIRE(largest_block <= memory_info.second);
  TM_REQUIRE_THROWS(allocator_ns::getExpandableSegmentSizes(0));
  TM_REQUIRE_THROWS(allocator_ns::snapshot({}, false));

  recorded_pointer = c10::DataPtr();
  c10::cuda::device_synchronize();
  allocator_ns::emptyCache();
}
