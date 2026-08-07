// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/native/cuda/CuFFTPlanCache.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenNative, CuFFTPlanCache) {
  at::native::detail::CuFFTHandle handle;
  TM_REQUIRE(handle.get() != 0);

  const at::IntArrayRef strides = {1, 1, 1};
  const at::IntArrayRef signal_sizes = {1, 1, 1};
  const at::native::detail::CuFFTParams parameters(
      strides,
      strides,
      signal_sizes,
      at::native::detail::CuFFTTransformType::R2C,
      c10::ScalarType::Float);
  at::native::detail::CuFFTConfig config(parameters);
  TM_REQUIRE(config.plan() != 0);

  at::native::detail::CuFFTParamsLRUCache cache(1);
  const auto& first = cache.lookup(parameters);
  TM_REQUIRE(first.plan() != 0);
  TM_REQUIRE_EQ(cache.size(), 1U);
  const auto& second = cache.lookup(parameters);
  TM_REQUIRE_EQ(second.plan(), first.plan());
  cache.resize(0);
  TM_REQUIRE_EQ(cache.size(), 0U);
  TM_REQUIRE_THROWS(cache.resize(-1));
}
