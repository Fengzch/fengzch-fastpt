// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/util/Exception.h>
#include <ATen/native/cuda/CuFFTUtils.h>

#include <string>

#include "../../support/TestHarness.h"

TM_TEST(ATenNative, CuFFTUtils) {
  TM_REQUIRE_EQ(at::native::_cudaGetErrorEnum(CUFFT_SUCCESS), std::string("CUFFT_SUCCESS"));
  TM_REQUIRE_EQ(
      at::native::_cudaGetErrorEnum(CUFFT_INVALID_VALUE),
      std::string("CUFFT_INVALID_VALUE"));
  TM_REQUIRE_EQ(
      at::native::_cudaGetErrorEnum(CUFFT_NOT_SUPPORTED),
      std::string("CUFFT_NOT_SUPPORTED"));
  TM_REQUIRE(
      at::native::_cudaGetErrorEnum(static_cast<cufftResult>(-1)).find("unknown error") !=
      std::string::npos);
  at::native::CUFFT_CHECK(CUFFT_SUCCESS);
  TM_REQUIRE_THROWS(at::native::CUFFT_CHECK(CUFFT_INVALID_VALUE));
}
