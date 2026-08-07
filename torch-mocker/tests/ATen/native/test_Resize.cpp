// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/native/cuda/Resize.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenNative, Resize) {
  auto tensor = at::randn({3, 3}, at::device(at::kCUDA));
  const std::vector<int64_t> size = {9};
  at::native::resize_impl_cuda_(tensor.unsafeGetTensorImpl(), size, c10::nullopt);
  TM_REQUIRE_EQ(tensor.numel(), 9);
  TM_REQUIRE_EQ(tensor.dim(), 1);
  TM_REQUIRE_EQ(tensor.size(0), 9);

  const std::vector<int64_t> expanded_size = {3, 4};
  at::native::resize_impl_cuda_(
      tensor.unsafeGetTensorImpl(), expanded_size, c10::nullopt);
  TM_REQUIRE_EQ(tensor.numel(), 12);
  TM_REQUIRE_EQ(tensor.dim(), 2);
  TM_REQUIRE_EQ(tensor.size(0), 3);
  TM_REQUIRE_EQ(tensor.size(1), 4);
}
