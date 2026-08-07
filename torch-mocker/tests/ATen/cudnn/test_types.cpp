// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/cudnn/Types.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCuDNN, Types) {
  const std::vector<std::pair<at::ScalarType, cudnnDataType_t>> types = {
      {at::kFloat, CUDNN_DATA_FLOAT},
      {at::kHalf, CUDNN_DATA_HALF},
      {at::kBFloat16, CUDNN_DATA_BFLOAT16},
      {at::kDouble, CUDNN_DATA_DOUBLE}};
  for (const auto& [scalar_type, cudnn_type] : types) {
    const auto tensor = at::randn({2, 2}, at::device(at::kCUDA).dtype(scalar_type));
    TM_REQUIRE_EQ(at::native::getCudnnDataTypeFromScalarType(scalar_type), cudnn_type);
    TM_REQUIRE_EQ(at::native::getCudnnDataType(tensor), cudnn_type);
  }
  TM_REQUIRE_EQ(at::native::cudnn_version(), static_cast<int64_t>(CUDNN_VERSION));
  TM_REQUIRE_THROWS(
      at::native::getCudnnDataTypeFromScalarType(at::kBool));
}
