// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/util/Exception.h>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAAllocatorConfig) {
  auto& config = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::instance();
  TM_REQUIRE(config.roundup_power2_divisions(128) >= 0);
  TM_REQUIRE(config.max_non_split_rounding_size() > 0);

  config.parseArgs("max_split_size_mb:1024");
  config.parseArgs("garbage_collection_threshold:0.2");
  config.parseArgs("roundup_power2_divisions:16");
  config.parseArgs("backend:native");
  config.parseArgs("pinned_use_hip_host_register:True");
  config.parseArgs("pinned_num_register_threads:128");
  config.parseArgs("pinned_use_background_threads:True");
  config.parseArgs("graph_capture_record_stream_reuse:True");
  config.parseArgs("pinned_reserve_segment_size_mb:16");
  config.parseArgs("per_process_memory_fraction:0.75");
  config.parseArgs("pinned_free_catch_all:True");
  config.parseArgs("throw_on_cudamalloc_oom:True");
  config.parseArgs("release_lock_on_hipmalloc:True");
  config.set_expandable_segments_handle_type(
      c10::cuda::CUDACachingAllocator::Expandable_Segments_Handle_Type::
          FABRIC_HANDLE);
  TM_REQUIRE(config.pinned_use_cuda_host_register());
  TM_REQUIRE_EQ(config.pinned_num_register_threads(), 128U);
  TM_REQUIRE_EQ(config.pinned_reserve_segment_size_mb(), 16U);
  TM_REQUIRE_NEAR(config.per_process_memory_fraction(), 0.75, 1e-12);
  TM_REQUIRE(config.graph_capture_record_stream_reuse());
  TM_REQUIRE(config.pinned_free_catch_all());
  TM_REQUIRE(config.throw_on_cudamalloc_oom());
  TM_REQUIRE(config.release_lock_on_cudamalloc());
  TM_REQUIRE_EQ(
      config.expandable_segments_handle_type(),
      c10::cuda::CUDACachingAllocator::Expandable_Segments_Handle_Type::
          FABRIC_HANDLE);
  TM_REQUIRE(
      c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::getKeys().count(
          "backend") == 1);
  bool rejected_invalid_thread_count = false;
  try {
    config.parseArgs("pinned_num_register_threads:3");
  } catch (const c10::Error&) {
    rejected_invalid_thread_count = true;
  }
  TM_REQUIRE(rejected_invalid_thread_count);
  TM_REQUIRE_THROWS(config.parseArgs("per_process_memory_fraction:1.5"));
  TM_REQUIRE_THROWS(config.parseArgs("pinned_reserve_segment_size_mb:0"));
  TM_REQUIRE_THROWS(config.parseArgs("unknown_allocator_option:1"));
}
