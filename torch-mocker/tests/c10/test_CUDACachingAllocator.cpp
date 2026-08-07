// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../support/TestHarness.h"

namespace {

constexpr size_t kAllocationSize = 4096;
constexpr size_t kCopySize = 256;

class TestGatheredContext final : public c10::GatheredContext {};

std::shared_ptr<c10::GatheredContext> makeTestGatheredContext() {
  return std::make_shared<TestGatheredContext>();
}

class HistoryReset final {
 public:
  ~HistoryReset() noexcept {
    try {
      c10::cuda::CUDACachingAllocator::recordHistory(
          false,
          nullptr,
          0,
          c10::cuda::CUDACachingAllocator::RecordContext::NEVER,
          true,
          {});
    } catch (...) {
      // Do not mask the test result while restoring allocator-wide state.
    }
  }
};

} // namespace

TM_TEST(C10, CUDACachingAllocator) {
  namespace allocator_ns = c10::cuda::CUDACachingAllocator;
  using c10::CachingAllocator::StatType;

  const auto device_count = c10::cuda::device_count_ensure_non_zero();
  auto* allocator = allocator_ns::get();
  TM_REQUIRE(allocator != nullptr);
  allocator->init(device_count);

  const auto allocator_name = allocator_ns::name();
  TM_REQUIRE(!allocator_name.empty());

  const bool allocator_enabled = allocator_ns::isEnabled();
  allocator_ns::enable(allocator_enabled);
  TM_REQUIRE_EQ(allocator_ns::isEnabled(), allocator_enabled);

  const auto memory_fraction = allocator_ns::getMemoryFraction(0);
  TM_REQUIRE(memory_fraction > 0.0);
  TM_REQUIRE(memory_fraction <= 1.0);
  allocator_ns::setMemoryFraction(memory_fraction, 0);
  TM_REQUIRE_NEAR(allocator_ns::getMemoryFraction(0), memory_fraction, 1e-12);

  const auto memory_info = allocator->getMemoryInfo(0);
  TM_REQUIRE(memory_info.second > 0);
  TM_REQUIRE(memory_info.first <= memory_info.second);

  allocator_ns::setUserMetadata("torch-mocker-allocator-ctest");
  TM_REQUIRE_EQ(
      allocator_ns::getUserMetadata(), std::string("torch-mocker-allocator-ctest"));
  allocator_ns::setUserMetadata("");

  void* pointer = allocator->raw_alloc(kAllocationSize);
  TM_REQUIRE(pointer != nullptr);

  size_t base_allocation_size = 0;
  void* base_pointer = allocator_ns::getBaseAllocation(pointer, &base_allocation_size);
  TM_REQUIRE(base_pointer != nullptr);
  TM_REQUIRE(base_allocation_size >= kAllocationSize);

  const bool native_allocator = allocator_name == "native";
  if (native_allocator && allocator_enabled) {
    const auto stats = allocator_ns::getDeviceStats(0);
    const auto& allocated_bytes =
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)];
    TM_REQUIRE(allocated_bytes.current >= static_cast<int64_t>(kAllocationSize));
    TM_REQUIRE(allocated_bytes.peak >= allocated_bytes.current);

    const auto snapshot = allocator_ns::snapshot({}, false);
    const auto pointer_address = reinterpret_cast<size_t>(pointer);
    const auto segment = std::find_if(
        snapshot.segments.begin(),
        snapshot.segments.end(),
        [pointer_address](const auto& candidate) {
          return candidate.device == 0 && candidate.total_size > 0 &&
              candidate.address <= pointer_address &&
              pointer_address < candidate.address + candidate.total_size;
        });
    TM_REQUIRE(segment != snapshot.segments.end());
  }

  auto stream = c10::cuda::getStreamFromPool(false, 0);
  void* stream_pointer = allocator_ns::raw_alloc_with_stream(
      kAllocationSize, stream.stream());
  TM_REQUIRE(stream_pointer != nullptr);
  TM_CUDA_REQUIRE(cudaMemsetAsync(
      stream_pointer, 0, kAllocationSize, stream.stream()));
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream.stream()));
  allocator_ns::raw_delete(stream_pointer);

  auto recorded_pointer = allocator->allocate(kAllocationSize);
  TM_REQUIRE(recorded_pointer.get() != nullptr);
  allocator_ns::recordStream(recorded_pointer, stream);
  TM_CUDA_REQUIRE(cudaMemsetAsync(
      recorded_pointer.get(), 0, kAllocationSize, stream.stream()));
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream.stream()));

  void* copy_source = allocator->raw_alloc(kCopySize);
  void* copy_destination = allocator->raw_alloc(kCopySize);
  TM_REQUIRE(copy_source != nullptr);
  TM_REQUIRE(copy_destination != nullptr);
  TM_CUDA_REQUIRE(cudaMemsetAsync(copy_source, 0x5A, kCopySize, stream.stream()));
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
      [](std::uint8_t value) { return value == 0x5A; }));
  allocator_ns::raw_delete(copy_destination);
  allocator_ns::raw_delete(copy_source);

  {
    HistoryReset reset_history;
    allocator_ns::recordHistory(
        true,
        makeTestGatheredContext,
        8,
        allocator_ns::RecordContext::STATE,
        true,
        {});
    TM_REQUIRE(allocator_ns::isHistoryEnabled());
    auto context_pointer = allocator->allocate(kAllocationSize);
    TM_REQUIRE(context_pointer.get() != nullptr);
    TM_REQUIRE(allocator_ns::getContextForPointer(context_pointer.get()) != nullptr);

    size_t trace_entries = 0;
    allocator_ns::attachAllocatorTraceTracker(
        [&trace_entries](const c10::CachingDeviceAllocator::TraceEntry&) {
          ++trace_entries;
        });
    allocator_ns::recordAnnotation(
        {{"test", "torch-mocker-allocator-annotation"}});
    std::string compile_context = "torch-mocker-allocator-compile-context";
    allocator_ns::pushCompileContext(compile_context);
    auto traced_pointer = allocator->allocate(kAllocationSize);
    allocator_ns::popCompileContext();
    TM_REQUIRE(traced_pointer.get() != nullptr);
    TM_REQUIRE(trace_entries > 0);

    if (native_allocator && allocator_enabled) {
      const auto traced_snapshot = allocator_ns::snapshot({}, true);
      const auto annotation = std::find_if(
          traced_snapshot.external_annotations.begin(),
          traced_snapshot.external_annotations.end(),
          [](const auto& entry) {
            const auto it = entry.metadata_.find("test");
            return it != entry.metadata_.end() &&
                it->second == "torch-mocker-allocator-annotation";
          });
      TM_REQUIRE(annotation != traced_snapshot.external_annotations.end());
    }
  }

  recorded_pointer = c10::DataPtr();
  allocator_ns::raw_delete(pointer);
  c10::cuda::device_synchronize();

  size_t largest_block = 0;
  allocator_ns::cacheInfo(0, &largest_block);
  TM_REQUIRE(largest_block <= memory_info.second);
  for (const auto& segment : allocator_ns::getExpandableSegmentSizes(0)) {
    TM_REQUIRE(segment.total_size > 0);
  }

  if (native_allocator) {
    allocator_ns::resetPeakStats(0);
    const auto peak_reset_stats = allocator_ns::getDeviceStats(0);
    const auto& peak_reset_bytes = peak_reset_stats
        .allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)];
    TM_REQUIRE_EQ(peak_reset_bytes.peak, peak_reset_bytes.current);

    allocator_ns::resetAccumulatedStats(0);
    const auto accumulated_reset_stats = allocator_ns::getDeviceStats(0);
    const auto& accumulated_reset_bytes = accumulated_reset_stats
        .allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)];
    TM_REQUIRE_EQ(accumulated_reset_bytes.allocated, 0);
    TM_REQUIRE_EQ(accumulated_reset_bytes.freed, 0);
  }

  allocator_ns::emptyCache();
  size_t largest_block_after_empty = 0;
  allocator_ns::cacheInfo(0, &largest_block_after_empty);
  TM_REQUIRE(largest_block_after_empty <= memory_info.second);
}
