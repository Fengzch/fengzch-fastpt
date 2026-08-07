// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>

#include <type_traits>

#include "../support/TestHarness.h"

namespace {

using c10d::symmetric_memory::CUDASymmetricMemory;
using c10d::symmetric_memory::CUDASymmetricMemoryAllocator;
using c10d::symmetric_memory::HandleType;
using c10d::symmetric_memory::SymmetricMemory;
using c10d::symmetric_memory::SymmetricMemoryAllocator;

static_assert(std::is_base_of_v<SymmetricMemory, CUDASymmetricMemory>);
static_assert(
    std::is_base_of_v<SymmetricMemoryAllocator, CUDASymmetricMemoryAllocator>);
static_assert(sizeof(HandleType) > 0);

} // namespace

TM_TEST(Csrc, CUDASymmetricMemoryHeaders) {
  TM_REQUIRE(c10d::symmetric_memory::max_cuda_p2p_domain_size > 0);
  TM_REQUIRE(c10d::symmetric_memory::symm_max_nblocks > 0);
  TM_REQUIRE(c10d::symmetric_memory::default_signal_pad_size > 0);
}
