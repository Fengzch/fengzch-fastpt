// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include <array>

#include "../../../support/TestHarness.h"

namespace {

struct CallbackState {
  size_t allocation_calls = 0;
  size_t release_calls = 0;
  size_t reset_calls = 0;
  size_t init_device_count = 0;
  size_t last_allocation_size = 0;
  size_t last_release_size = 0;
  int last_device = -1;
  cudaStream_t last_stream = nullptr;
  double memory_fraction = 0.0;
  void* last_recorded_ptr = nullptr;
  cudaStream_t last_recorded_stream = nullptr;
  c10::cuda::MempoolId_t pool_id{0, 0};
  int pool_begin_calls = 0;
  int pool_end_calls = 0;
  int pool_release_calls = 0;
  bool pool_filter_result = false;
};

CallbackState state;

constexpr size_t kAllocationSize = 4096;

void* allocate(size_t size, int device, cudaStream_t stream) {
  ++state.allocation_calls;
  state.last_allocation_size = size;
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
  state.last_release_size = size;
  state.last_device = device;
  state.last_stream = stream;
  (void)cudaFree(pointer);
}

} // namespace

TM_TEST(TorchCUDA, CUDAPluggableAllocator) {
  state = {};
  torch::cuda::CUDAPluggableAllocator::_AllocationMetadata default_metadata;
  TM_REQUIRE_EQ(default_metadata.size, 0U);
  TM_REQUIRE_EQ(default_metadata.device_idx, -1);

  auto stream = c10::cuda::getStreamFromPool(false, 0);
  torch::cuda::CUDAPluggableAllocator::_AllocationMetadata metadata(
      kAllocationSize, 0, stream.stream());
  TM_REQUIRE_EQ(metadata.size, kAllocationSize);
  TM_REQUIRE_EQ(metadata.device_idx, 0);
  TM_REQUIRE_EQ(metadata.stream, stream.stream());

  auto allocator = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
      allocate, release);
  auto custom = std::dynamic_pointer_cast<
      torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(allocator);
  TM_REQUIRE(custom != nullptr);
  TM_REQUIRE_EQ(custom->name(), std::string("pluggable"));
  TM_REQUIRE(custom->isEnabled());
  TM_REQUIRE(custom->initialized());

  custom->set_init_fn([](int count) { state.init_device_count = count; });
  custom->init(1);
  TM_REQUIRE_EQ(state.init_device_count, 1U);

  custom->set_reset_fn([]() { ++state.reset_calls; });
  custom->emptyCache();
  TM_REQUIRE_EQ(state.reset_calls, 1U);

  custom->set_memory_fraction_fn(
      [](double fraction, int device) {
        state.memory_fraction = fraction;
        state.last_device = device;
      });
  custom->setMemoryFraction(0.5, 0);
  TM_REQUIRE_NEAR(state.memory_fraction, 0.5, 1e-12);
  TM_REQUIRE_EQ(state.last_device, 0);

  void* allocation = custom->raw_alloc(kAllocationSize);
  TM_REQUIRE(allocation != nullptr);
  TM_REQUIRE_EQ(state.allocation_calls, 1U);
  TM_REQUIRE_EQ(state.last_allocation_size, kAllocationSize);
  TM_REQUIRE_EQ(state.last_device, 0);

  size_t base_size = 123;
  TM_REQUIRE_EQ(custom->getBaseAllocation(allocation, &base_size), allocation);
  TM_REQUIRE_EQ(base_size, 123U);
  custom->set_base_alloc_fn([](void* pointer, size_t* size) {
    *size = kAllocationSize;
    return pointer;
  });
  TM_REQUIRE_EQ(custom->getBaseAllocation(allocation, &base_size), allocation);
  TM_REQUIRE_EQ(base_size, kAllocationSize);

  custom->set_record_stream_fn([](void* pointer, cudaStream_t recorded_stream) {
    state.last_recorded_ptr = pointer;
    state.last_recorded_stream = recorded_stream;
  });
  c10::DataPtr data_ptr(
      allocation, nullptr, nullptr, c10::Device(c10::kCUDA, 0));
  custom->recordStream(data_ptr, stream);
  TM_REQUIRE_EQ(state.last_recorded_ptr, allocation);
  TM_REQUIRE_EQ(state.last_recorded_stream, stream.stream());

  void* stream_allocation = custom->raw_alloc_with_stream(kAllocationSize, stream.stream());
  TM_REQUIRE(stream_allocation != nullptr);
  TM_REQUIRE_EQ(state.allocation_calls, 2U);
  TM_REQUIRE_EQ(state.last_stream, stream.stream());

  std::array<int, 4> host_source = {1, 2, 3, 4};
  std::array<int, 4> host_destination = {};
  void* copy_source = custom->raw_alloc(sizeof(host_source));
  void* copy_destination = custom->raw_alloc(sizeof(host_destination));
  TM_CUDA_REQUIRE(cudaMemcpy(
      copy_source,
      host_source.data(),
      sizeof(host_source),
      cudaMemcpyHostToDevice));
  custom->copy_data(copy_destination, copy_source, sizeof(host_source));
  TM_CUDA_REQUIRE(cudaMemcpy(
      host_destination.data(),
      copy_destination,
      sizeof(host_destination),
      cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(host_destination, host_source);
  TM_CUDA_REQUIRE(cudaMemset(copy_destination, 0, sizeof(host_destination)));
  TM_CUDA_REQUIRE(custom->memcpyAsync(
      copy_destination,
      0,
      copy_source,
      0,
      sizeof(host_source),
      stream.stream(),
      false));
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream.stream()));
  TM_CUDA_REQUIRE(cudaMemcpy(
      host_destination.data(),
      copy_destination,
      sizeof(host_destination),
      cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(host_destination, host_source);

  custom->set_begin_allocate_to_pool(
      [](int device, c10::cuda::MempoolId_t pool, std::function<bool(cudaStream_t)> filter) {
        ++state.pool_begin_calls;
        state.last_device = device;
        state.pool_id = pool;
        state.pool_filter_result = filter(nullptr);
      });
  custom->set_end_allocate_to_pool_fn([](int device, c10::cuda::MempoolId_t pool) {
    ++state.pool_end_calls;
    state.last_device = device;
    state.pool_id = pool;
  });
  custom->set_release_pool([](int device, c10::cuda::MempoolId_t pool) {
    ++state.pool_release_calls;
    state.last_device = device;
    state.pool_id = pool;
  });
  const c10::cuda::MempoolId_t pool_id{11, 22};
  custom->beginAllocateToPool(0, pool_id, [](cudaStream_t) { return true; });
  custom->endAllocateToPool(0, pool_id);
  custom->releasePool(0, pool_id);
  TM_REQUIRE_EQ(state.pool_begin_calls, 1);
  TM_REQUIRE_EQ(state.pool_end_calls, 1);
  TM_REQUIRE_EQ(state.pool_release_calls, 1);
  TM_REQUIRE(state.pool_filter_result);
  TM_REQUIRE_EQ(state.pool_id, pool_id);

  custom->enable(false);
  TM_REQUIRE(custom->isEnabled());
  TM_REQUIRE_THROWS(custom->getMemoryFraction(0));
  TM_REQUIRE_THROWS(custom->getExpandableSegmentSizes(0));
  TM_REQUIRE_THROWS(custom->cacheInfo(0, &base_size));
  TM_REQUIRE_THROWS(custom->getDeviceStats(0));
  TM_REQUIRE_THROWS(custom->resetAccumulatedStats(0));
  TM_REQUIRE_THROWS(custom->resetPeakStats(0));
  TM_REQUIRE_THROWS(custom->snapshot({}, false));
  TM_REQUIRE_THROWS(custom->getIpcDevPtr("unsupported"));
  TM_REQUIRE_THROWS(custom->shareIpcHandle(allocation));
  TM_REQUIRE_THROWS(custom->recordHistory(
      false,
      nullptr,
      0,
      c10::CachingDeviceAllocator::RecordContext::NEVER,
      false,
      {}));
  TM_REQUIRE_THROWS(custom->attachOutOfMemoryObserver(
      [](int64_t, size_t, size_t, size_t) {}));
  TM_REQUIRE_THROWS(custom->attachOomRejectionObserver(
      [](int64_t, size_t, size_t, size_t) {}));
  TM_REQUIRE_THROWS(custom->attachAllocatorTraceTracker(
      [](const c10::CachingDeviceAllocator::TraceEntry&) {}));
  TM_REQUIRE_THROWS(custom->getCheckpointState(0, pool_id));
  TM_REQUIRE_THROWS(custom->setCheckpointPoolState(0, nullptr));

  custom->raw_delete(allocation);
  custom->raw_delete(stream_allocation);
  custom->raw_delete(copy_source);
  custom->raw_delete(copy_destination);
  TM_REQUIRE_EQ(state.release_calls, 4U);
  TM_REQUIRE_EQ(state.last_release_size, sizeof(host_destination));
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
}
