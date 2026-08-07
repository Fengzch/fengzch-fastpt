// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDADataType.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDADataType) {
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<at::Half>(), CUDA_R_16F);
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<float>(), CUDA_R_32F);
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<double>(), CUDA_R_64F);
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<uint8_t>(), CUDA_R_8U);
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<int8_t>(), CUDA_R_8I);
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<int16_t>(), CUDA_R_16I);
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<int>(), CUDA_R_32I);
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<int64_t>(), CUDA_R_64I);
  TM_REQUIRE_EQ(at::cuda::getCudaDataType<at::BFloat16>(), CUDA_R_16BF);
  TM_REQUIRE_EQ(
      at::cuda::getCudaDataType<c10::complex<at::Half>>(), CUDA_C_16F);
  TM_REQUIRE_EQ(
      at::cuda::getCudaDataType<c10::complex<float>>(), CUDA_C_32F);
  TM_REQUIRE_EQ(
      at::cuda::getCudaDataType<c10::complex<double>>(), CUDA_C_64F);
  TM_REQUIRE_EQ(
      at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Half), CUDA_R_16F);
  TM_REQUIRE_EQ(
      at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Float), CUDA_R_32F);
  TM_REQUIRE_EQ(
      at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Double), CUDA_R_64F);
  TM_REQUIRE_EQ(
      at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Long), CUDA_R_64I);
  TM_REQUIRE_EQ(
      at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::ComplexDouble),
      CUDA_C_64F);
  TM_REQUIRE_EQ(
      at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::BFloat16),
      CUDA_R_16BF);
  TM_REQUIRE_THROWS(
      at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Undefined));
}
