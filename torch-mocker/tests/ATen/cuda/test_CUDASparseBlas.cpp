// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDASparseBlas.h>
#include <c10/util/complex.h>

#include <array>

#include "../../support/TestHarness.h"

namespace {

template <typename scalar_t>
void verify_bsrmm_identity_rhs(
    cusparseHandle_t handle,
    cusparseMatDescr_t descriptor,
    const std::array<scalar_t, 4>& expected) {
  constexpr int kBlockDim = 2;
  constexpr int kBlockRows = 1;
  constexpr int kBlockColumns = 1;
  constexpr int kNonzeroBlocks = 1;
  constexpr int kColumns = 2;
  const std::array<scalar_t, 4> matrix = {
      scalar_t{1}, scalar_t{2}, scalar_t{3}, scalar_t{4}};
  const std::array<scalar_t, 4> identity = {
      scalar_t{1}, scalar_t{0}, scalar_t{0}, scalar_t{1}};
  const std::array<int, 2> row_offsets = {0, 1};
  const std::array<int, 1> column_indices = {0};
  const scalar_t alpha{1};
  const scalar_t beta{0};

  scalar_t* device_matrix = nullptr;
  scalar_t* device_rhs = nullptr;
  scalar_t* device_output = nullptr;
  int* device_rows = nullptr;
  int* device_columns = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_matrix, sizeof(matrix)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_rhs, sizeof(identity)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_output, sizeof(expected)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_rows, sizeof(row_offsets)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_columns, sizeof(column_indices)));
  TM_CUDA_REQUIRE(cudaMemcpy(
      device_matrix, matrix.data(), sizeof(matrix), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(
      device_rhs, identity.data(), sizeof(identity), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(
      device_rows, row_offsets.data(), sizeof(row_offsets), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(
      device_columns,
      column_indices.data(),
      sizeof(column_indices),
      cudaMemcpyHostToDevice));

  at::cuda::sparse::bsrmm<scalar_t>(
      handle,
      CUSPARSE_DIRECTION_ROW,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      kBlockRows,
      kColumns,
      kBlockColumns,
      kNonzeroBlocks,
      &alpha,
      descriptor,
      device_matrix,
      device_rows,
      device_columns,
      kBlockDim,
      device_rhs,
      kColumns,
      &beta,
      device_output,
      kColumns);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  std::array<scalar_t, 4> actual{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(actual, expected);

  TM_CUDA_REQUIRE(cudaFree(device_columns));
  TM_CUDA_REQUIRE(cudaFree(device_rows));
  TM_CUDA_REQUIRE(cudaFree(device_output));
  TM_CUDA_REQUIRE(cudaFree(device_rhs));
  TM_CUDA_REQUIRE(cudaFree(device_matrix));
}

void verify_bsrmv(cusparseHandle_t handle, cusparseMatDescr_t descriptor) {
  const std::array<float, 4> matrix = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<int, 2> row_offsets = {0, 1};
  const std::array<int, 1> column_indices = {0};
  const std::array<float, 2> input = {1.0f, 1.0f};
  const std::array<float, 2> expected = {3.0f, 7.0f};
  const float alpha = 1.0f;
  const float beta = 0.0f;
  float *device_matrix = nullptr, *device_input = nullptr, *device_output = nullptr;
  int *device_rows = nullptr, *device_columns = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_matrix, sizeof(matrix)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_input, sizeof(input)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_output, sizeof(expected)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_rows, sizeof(row_offsets)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_columns, sizeof(column_indices)));
  TM_CUDA_REQUIRE(cudaMemcpy(device_matrix, matrix.data(), sizeof(matrix), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(device_input, input.data(), sizeof(input), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(device_rows, row_offsets.data(), sizeof(row_offsets), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(device_columns, column_indices.data(), sizeof(column_indices), cudaMemcpyHostToDevice));
  at::cuda::sparse::bsrmv<float>(
      handle,
      CUSPARSE_DIRECTION_ROW,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      1,
      1,
      1,
      &alpha,
      descriptor,
      device_matrix,
      device_rows,
      device_columns,
      2,
      device_input,
      &beta,
      device_output);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  std::array<float, 2> actual{};
  TM_CUDA_REQUIRE(cudaMemcpy(actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(actual, expected);
  TM_CUDA_REQUIRE(cudaFree(device_columns));
  TM_CUDA_REQUIRE(cudaFree(device_rows));
  TM_CUDA_REQUIRE(cudaFree(device_output));
  TM_CUDA_REQUIRE(cudaFree(device_input));
  TM_CUDA_REQUIRE(cudaFree(device_matrix));
}

void verify_bsr_solve_paths(cusparseHandle_t handle, cusparseMatDescr_t descriptor) {
  // A is the lower triangular matrix [[2, 0], [1, 3]] in 1x1 BSR blocks.
  const std::array<float, 3> matrix = {2.0f, 1.0f, 3.0f};
  const std::array<int, 3> row_offsets = {0, 1, 3};
  const std::array<int, 3> column_indices = {0, 0, 1};
  const std::array<float, 2> vector_rhs = {2.0f, 7.0f};
  const std::array<float, 2> vector_expected = {1.0f, 2.0f};
  // Dense right-hand sides are column-major for the cuSPARSE-compatible API.
  const std::array<float, 4> matrix_rhs = {2.0f, 7.0f, 4.0f, 10.0f};
  const std::array<float, 4> matrix_expected = {1.0f, 2.0f, 2.0f, 8.0f / 3.0f};
  const float alpha = 1.0f;
  float *device_matrix = nullptr, *device_vector_rhs = nullptr;
  float *device_vector_output = nullptr, *device_matrix_rhs = nullptr;
  float* device_matrix_output = nullptr;
  int *device_rows = nullptr, *device_columns = nullptr;
  void* buffer = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_matrix, sizeof(matrix)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_vector_rhs, sizeof(vector_rhs)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_vector_output, sizeof(vector_expected)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_matrix_rhs, sizeof(matrix_rhs)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_matrix_output, sizeof(matrix_expected)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_rows, sizeof(row_offsets)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_columns, sizeof(column_indices)));
  TM_CUDA_REQUIRE(cudaMemcpy(device_matrix, matrix.data(), sizeof(matrix), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(device_vector_rhs, vector_rhs.data(), sizeof(vector_rhs), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(device_matrix_rhs, matrix_rhs.data(), sizeof(matrix_rhs), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(device_rows, row_offsets.data(), sizeof(row_offsets), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(device_columns, column_indices.data(), sizeof(column_indices), cudaMemcpyHostToDevice));

  bsrsv2Info_t vector_info = nullptr;
  TM_REQUIRE_EQ(cusparseCreateBsrsv2Info(&vector_info), CUSPARSE_STATUS_SUCCESS);
  int vector_buffer_size = 0;
  at::cuda::sparse::bsrsv2_bufferSize<float>(
      handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, 2, 3,
      descriptor, device_matrix, device_rows, device_columns, 1, vector_info,
      &vector_buffer_size);
  TM_REQUIRE(vector_buffer_size > 0);
  TM_CUDA_REQUIRE(cudaMalloc(&buffer, vector_buffer_size));
  at::cuda::sparse::bsrsv2_analysis<float>(
      handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, 2, 3,
      descriptor, device_matrix, device_rows, device_columns, 1, vector_info,
      CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer);
  at::cuda::sparse::bsrsv2_solve<float>(
      handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, 2, 3,
      &alpha, descriptor, device_matrix, device_rows, device_columns, 1,
      vector_info, device_vector_rhs, device_vector_output,
      CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  std::array<float, 2> vector_actual{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      vector_actual.data(), device_vector_output, sizeof(vector_actual), cudaMemcpyDeviceToHost));
  TM_REQUIRE_NEAR(vector_actual[0], vector_expected[0], 1e-6f);
  TM_REQUIRE_NEAR(vector_actual[1], vector_expected[1], 1e-6f);
  TM_CUDA_REQUIRE(cudaFree(buffer));
  TM_REQUIRE_EQ(cusparseDestroyBsrsv2Info(vector_info), CUSPARSE_STATUS_SUCCESS);

  bsrsm2Info_t matrix_info = nullptr;
  TM_REQUIRE_EQ(cusparseCreateBsrsm2Info(&matrix_info), CUSPARSE_STATUS_SUCCESS);
  int matrix_buffer_size = 0;
  at::cuda::sparse::bsrsm2_bufferSize<float>(
      handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, 2, 2, 3, descriptor, device_matrix,
      device_rows, device_columns, 1, matrix_info, &matrix_buffer_size);
  TM_REQUIRE(matrix_buffer_size > 0);
  TM_CUDA_REQUIRE(cudaMalloc(&buffer, matrix_buffer_size));
  at::cuda::sparse::bsrsm2_analysis<float>(
      handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, 2, 2, 3, descriptor, device_matrix,
      device_rows, device_columns, 1, matrix_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
      buffer);
  at::cuda::sparse::bsrsm2_solve<float>(
      handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, 2, 2, 3, &alpha, descriptor,
      device_matrix, device_rows, device_columns, 1, matrix_info,
      device_matrix_rhs, 2, device_matrix_output, 2, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
      buffer);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  std::array<float, 4> matrix_actual{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      matrix_actual.data(), device_matrix_output, sizeof(matrix_actual), cudaMemcpyDeviceToHost));
  for (size_t index = 0; index < matrix_actual.size(); ++index) {
    TM_REQUIRE_NEAR(matrix_actual[index], matrix_expected[index], 1e-6f);
  }
  TM_CUDA_REQUIRE(cudaFree(buffer));
  TM_REQUIRE_EQ(cusparseDestroyBsrsm2Info(matrix_info), CUSPARSE_STATUS_SUCCESS);

  TM_CUDA_REQUIRE(cudaFree(device_columns));
  TM_CUDA_REQUIRE(cudaFree(device_rows));
  TM_CUDA_REQUIRE(cudaFree(device_matrix_output));
  TM_CUDA_REQUIRE(cudaFree(device_matrix_rhs));
  TM_CUDA_REQUIRE(cudaFree(device_vector_output));
  TM_CUDA_REQUIRE(cudaFree(device_vector_rhs));
  TM_CUDA_REQUIRE(cudaFree(device_matrix));
}

} // namespace

TM_TEST(ATenCUDA, CUDASparseBlas) {
  constexpr int rows = 3;
  constexpr int columns = 3;
  constexpr int a_nnz = 2;
  constexpr int b_nnz = 1;
  const std::array<float, a_nnz> host_a_values = {1.0f, 2.0f};
  const std::array<int, rows + 1> host_a_rows = {0, 2, 2, 2};
  const std::array<int, a_nnz> host_a_columns = {0, 2};
  const std::array<float, b_nnz> host_b_values = {3.0f};
  const std::array<int, rows + 1> host_b_rows = {0, 1, 1, 1};
  const std::array<int, b_nnz> host_b_columns = {1};
  const float alpha = 1.0f;
  const float beta = 1.0f;

  cusparseMatDescr_t a_descriptor = nullptr;
  cusparseMatDescr_t b_descriptor = nullptr;
  cusparseMatDescr_t c_descriptor = nullptr;
  TM_CUDA_REQUIRE(cudaFree(0));
  TM_REQUIRE_EQ(cusparseCreateMatDescr(&a_descriptor), CUSPARSE_STATUS_SUCCESS);
  TM_REQUIRE_EQ(cusparseCreateMatDescr(&b_descriptor), CUSPARSE_STATUS_SUCCESS);
  TM_REQUIRE_EQ(cusparseCreateMatDescr(&c_descriptor), CUSPARSE_STATUS_SUCCESS);
  for (const auto descriptor : {a_descriptor, b_descriptor, c_descriptor}) {
    TM_REQUIRE_EQ(cusparseSetMatType(descriptor, CUSPARSE_MATRIX_TYPE_GENERAL), CUSPARSE_STATUS_SUCCESS);
    TM_REQUIRE_EQ(cusparseSetMatIndexBase(descriptor, CUSPARSE_INDEX_BASE_ZERO), CUSPARSE_STATUS_SUCCESS);
  }

  float *a_values = nullptr, *b_values = nullptr, *c_values = nullptr;
  int *a_rows = nullptr, *a_columns = nullptr, *b_rows = nullptr, *b_columns = nullptr;
  int *c_rows = nullptr, *c_columns = nullptr, *c_nnz = nullptr;
  void* buffer = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&a_values, sizeof(host_a_values)));
  TM_CUDA_REQUIRE(cudaMalloc(&b_values, sizeof(host_b_values)));
  TM_CUDA_REQUIRE(cudaMalloc(&a_rows, sizeof(host_a_rows)));
  TM_CUDA_REQUIRE(cudaMalloc(&a_columns, sizeof(host_a_columns)));
  TM_CUDA_REQUIRE(cudaMalloc(&b_rows, sizeof(host_b_rows)));
  TM_CUDA_REQUIRE(cudaMalloc(&b_columns, sizeof(host_b_columns)));
  TM_CUDA_REQUIRE(cudaMemcpy(a_values, host_a_values.data(), sizeof(host_a_values), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(b_values, host_b_values.data(), sizeof(host_b_values), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(a_rows, host_a_rows.data(), sizeof(host_a_rows), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(a_columns, host_a_columns.data(), sizeof(host_a_columns), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(b_rows, host_b_rows.data(), sizeof(host_b_rows), cudaMemcpyHostToDevice));
  TM_CUDA_REQUIRE(cudaMemcpy(b_columns, host_b_columns.data(), sizeof(host_b_columns), cudaMemcpyHostToDevice));

  const auto handle = at::cuda::getCurrentCUDASparseHandle();
  size_t buffer_size = 0;
  at::cuda::sparse::csrgeam2_bufferSizeExt<float>(
      handle, rows, columns, &alpha, a_descriptor, a_nnz, a_values, a_rows, a_columns,
      &beta, b_descriptor, b_nnz, b_values, b_rows, b_columns, c_descriptor,
      nullptr, nullptr, nullptr, &buffer_size);
  TM_REQUIRE(buffer_size > 0);
  TM_CUDA_REQUIRE(cudaMalloc(&buffer, buffer_size));
  TM_CUDA_REQUIRE(cudaMalloc(&c_rows, sizeof(host_a_rows)));
  TM_CUDA_REQUIRE(cudaMalloc(&c_nnz, sizeof(int)));
  at::cuda::sparse::csrgeam2Nnz<float>(
      handle, rows, columns, a_descriptor, a_nnz, a_rows, a_columns,
      b_descriptor, b_nnz, b_rows, b_columns, c_descriptor, c_rows, c_nnz, buffer);
  int host_c_nnz = 0;
  TM_CUDA_REQUIRE(cudaMemcpy(&host_c_nnz, c_nnz, sizeof(host_c_nnz), cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(host_c_nnz, 3);
  TM_CUDA_REQUIRE(cudaMalloc(&c_values, host_c_nnz * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMalloc(&c_columns, host_c_nnz * sizeof(int)));
  at::cuda::sparse::csrgeam2<float>(
      handle, rows, columns, &alpha, a_descriptor, a_nnz, a_values, a_rows, a_columns,
      &beta, b_descriptor, b_nnz, b_values, b_rows, b_columns, c_descriptor,
      c_values, c_rows, c_columns, buffer);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  std::array<float, 3> actual_values{};
  std::array<int, 3> actual_columns{};
  std::array<int, rows + 1> actual_rows{};
  TM_CUDA_REQUIRE(cudaMemcpy(actual_values.data(), c_values, sizeof(actual_values), cudaMemcpyDeviceToHost));
  TM_CUDA_REQUIRE(cudaMemcpy(actual_columns.data(), c_columns, sizeof(actual_columns), cudaMemcpyDeviceToHost));
  TM_CUDA_REQUIRE(cudaMemcpy(actual_rows.data(), c_rows, sizeof(actual_rows), cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(actual_values, (std::array<float, 3>{1.0f, 3.0f, 2.0f}));
  TM_REQUIRE_EQ(actual_columns, (std::array<int, 3>{0, 1, 2}));
  TM_REQUIRE_EQ(actual_rows, (std::array<int, 4>{0, 3, 3, 3}));

  verify_bsrmm_identity_rhs<float>(
      handle, a_descriptor, {1.0f, 3.0f, 2.0f, 4.0f});
  verify_bsrmm_identity_rhs<c10::complex<double>>(
      handle,
      a_descriptor,
      {c10::complex<double>{1.0, 0.0},
       c10::complex<double>{3.0, 0.0},
       c10::complex<double>{2.0, 0.0},
       c10::complex<double>{4.0, 0.0}});
  verify_bsrmv(handle, a_descriptor);
  verify_bsr_solve_paths(handle, a_descriptor);

  for (void* pointer : {static_cast<void*>(a_values), static_cast<void*>(b_values), static_cast<void*>(c_values), buffer,
                        static_cast<void*>(a_rows), static_cast<void*>(a_columns), static_cast<void*>(b_rows),
                        static_cast<void*>(b_columns), static_cast<void*>(c_rows), static_cast<void*>(c_columns),
                        static_cast<void*>(c_nnz)}) {
    TM_CUDA_REQUIRE(cudaFree(pointer));
  }
  TM_REQUIRE_EQ(cusparseDestroyMatDescr(a_descriptor), CUSPARSE_STATUS_SUCCESS);
  TM_REQUIRE_EQ(cusparseDestroyMatDescr(b_descriptor), CUSPARSE_STATUS_SUCCESS);
  TM_REQUIRE_EQ(cusparseDestroyMatDescr(c_descriptor), CUSPARSE_STATUS_SUCCESS);
}
