// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDASparseDescriptors.h>
#include <ATen/Functions.h>
#include <ATen/ops/sparse_csr_tensor.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDASparseDescriptors) {
  at::cuda::sparse::CuSparseSpGEMMDescriptor spgemm;
  TM_REQUIRE(spgemm.descriptor() != nullptr);

  at::cuda::sparse::CuSparseMatDescriptor matrix;
  at::cuda::sparse::CuSparseMatDescriptor lower_triangular(true, true);
  TM_REQUIRE(matrix.descriptor() != nullptr);
  TM_REQUIRE(lower_triangular.descriptor() != nullptr);

  at::cuda::sparse::CuSparseBsrsv2Info bsrsv2;
  at::cuda::sparse::CuSparseBsrsm2Info bsrsm2;
  TM_REQUIRE(bsrsv2.descriptor() != nullptr);
  TM_REQUIRE(bsrsm2.descriptor() != nullptr);

  const auto values = at::tensor({1.0f, 2.0f});
  const auto columns = at::tensor({0, 2}, at::kInt);
  const auto rows = at::tensor({0, 2, 2, 2}, at::kInt);
  const auto csr = at::sparse_csr_tensor(
      rows, columns, values, {3, 3}, at::TensorOptions().device(at::kCPU));
  at::cuda::sparse::CuSparseSpMatCsrDescriptor csr_descriptor(csr);
  const auto [rows_count, columns_count, nonzero_count] = csr_descriptor.get_size();
  TM_REQUIRE_EQ(rows_count, 3);
  TM_REQUIRE_EQ(columns_count, 3);
  TM_REQUIRE_EQ(nonzero_count, 2);
}
