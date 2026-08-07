// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/cuda/Randperm.cuh>

#include <limits>

#include "../../support/TestHarness.h"

TM_TEST(ATenNative, Randperm) {
  constexpr int64_t size = 128;
  const auto keys_cpu = at::full(
      {size}, 7, at::TensorOptions().dtype(at::kInt));
  auto result = at::arange(
      size, keys_cpu.options().device(at::kCUDA));
  const auto keys = keys_cpu.to(at::kCUDA);
  c10::optional<at::Generator> generator;
  randperm_handle_duplicate_keys(
      keys.data_ptr<int>(), result.data_ptr<int>(), 32, size, generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto host_result = result.cpu();
  TM_REQUIRE_EQ(host_result.numel(), size);
  TM_REQUIRE(at::equal(
      std::get<0>(at::sort(host_result)),
      at::arange(size, host_result.options())));
}
