// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/native/cuda/ScanUtils.cuh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "../../support/TestHarness.h"

namespace {

struct MaxOp {
  template <typename T>
  __host__ __device__ bool operator()(const T& left, const T& right) const {
    return left > right;
  }
};

struct AddOp {
  template <typename T>
  __host__ __device__ T operator()(const T& left, const T& right) const {
    return left + right;
  }
};

at::Tensor cuda_float_tensor(
    const std::vector<float>& values,
    at::IntArrayRef sizes) {
  return at::tensor(values, at::TensorOptions().dtype(at::kFloat))
      .reshape(sizes)
      .to(at::kCUDA);
}

void require_scan_result(
    const at::Tensor& values,
    const std::vector<float>& expected_values,
    const at::Tensor& indices,
    const std::vector<int64_t>& expected_indices) {
  const auto actual_values = values.cpu().contiguous();
  const auto actual_indices = indices.cpu().contiguous();
  const auto* value_data = actual_values.data_ptr<float>();
  const auto* index_data = actual_indices.data_ptr<int64_t>();
  TM_REQUIRE_EQ(actual_values.numel(), static_cast<int64_t>(expected_values.size()));
  TM_REQUIRE_EQ(actual_indices.numel(), static_cast<int64_t>(expected_indices.size()));
  for (size_t index = 0; index < expected_values.size(); ++index) {
    const float expected = expected_values[index];
    const float actual = value_data[index];
    TM_REQUIRE((std::isnan(expected) && std::isnan(actual)) ||
               std::abs(actual - expected) <= 1e-6f);
    TM_REQUIRE_EQ(index_data[index], expected_indices[index]);
  }
}

void inclusive_max_reference(
    const std::vector<float>& input,
    int64_t rows,
    int64_t row_size,
    std::vector<float>* values,
    std::vector<int64_t>* indices) {
  values->resize(input.size());
  indices->resize(input.size());
  for (int64_t row = 0; row < rows; ++row) {
    float maximum = -std::numeric_limits<float>::infinity();
    int64_t maximum_index = 0;
    for (int64_t column = 0; column < row_size; ++column) {
      const auto offset = row * row_size + column;
      const float value = input[offset];
      if (!std::isnan(value) && (std::isnan(maximum) || value > maximum)) {
        maximum = value;
        maximum_index = column;
      }
      (*values)[offset] = maximum;
      (*indices)[offset] = maximum_index;
    }
  }
}

} // namespace

TM_TEST(ATenNative, ScanUtilsHelpers) {
  TM_REQUIRE_EQ(at::native::ceil_div<int64_t>(0, 8), 0);
  TM_REQUIRE_EQ(at::native::ceil_div<int64_t>(1, 8), 1);
  TM_REQUIRE_EQ(at::native::ceil_div<int64_t>(17, 8), 3);
  TM_REQUIRE_EQ(at::native::get_log_num_threads_x_inner_scan<uint32_t>(1, 1), 4U);
  TM_REQUIRE_EQ(at::native::get_log_num_threads_x_inner_scan<uint32_t>(1, 512), 9U);
  TM_REQUIRE_EQ(at::native::get_log_num_threads_x_inner_scan<uint32_t>(512, 1), 4U);
  TM_REQUIRE_NOTHROW(
      at::native::check_fits_in_unsigned(std::numeric_limits<uint32_t>::max(), "limit"));
  TM_REQUIRE_THROWS(at::native::check_fits_in_unsigned(-1, "negative"));
  TM_REQUIRE_THROWS(at::native::check_fits_in_unsigned(
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1, "overflow"));
}

TM_TEST(ATenNative, ScanUtilsInnermostWithIndices) {
  constexpr int64_t rows = 3;
  constexpr int64_t row_size = 513;
  std::vector<float> input(rows * row_size);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < row_size; ++column) {
      input[row * row_size + column] =
          static_cast<float>((column * 17 + row * 29) % 101) - 50.0f;
    }
  }
  input[100] = 200.0f;
  input[400] = 200.0f; // Tie retains the first index.
  input[row_size + 37] = std::numeric_limits<float>::quiet_NaN();
  input[2 * row_size + 512] = 300.0f;

  std::vector<float> expected_values;
  std::vector<int64_t> expected_indices;
  inclusive_max_reference(input, rows, row_size, &expected_values, &expected_indices);

  const auto self = cuda_float_tensor(input, {rows, row_size});
  auto values = at::empty_like(self);
  auto indices = at::empty(
      self.sizes(), self.options().dtype(at::kLong));
  at::native::scan_dim_with_indices<float>(
      self,
      values,
      indices,
      1,
      -std::numeric_limits<float>::infinity(),
      MaxOp{});
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  require_scan_result(values, expected_values, indices, expected_indices);
}

TM_TEST(ATenNative, ScanUtilsOuterWithIndices) {
  constexpr int64_t outer_rows = 2;
  constexpr int64_t row_size = 4;
  constexpr int64_t inner_rows = 3;
  std::vector<float> input(outer_rows * row_size * inner_rows);
  for (int64_t outer = 0; outer < outer_rows; ++outer) {
    for (int64_t column = 0; column < row_size; ++column) {
      for (int64_t inner = 0; inner < inner_rows; ++inner) {
        const auto offset = outer * row_size * inner_rows + column * inner_rows + inner;
        input[offset] = static_cast<float>(outer * 20 + inner * 5 + column);
      }
    }
  }
  input[1 * inner_rows + 1] = 30.0f;
  input[3 * inner_rows + 1] = 30.0f; // Tie retains the first index.
  input[row_size * inner_rows + 2 * inner_rows] = std::numeric_limits<float>::quiet_NaN();

  std::vector<float> expected_values(input.size());
  std::vector<int64_t> expected_indices(input.size());
  for (int64_t outer = 0; outer < outer_rows; ++outer) {
    for (int64_t inner = 0; inner < inner_rows; ++inner) {
      float maximum = -std::numeric_limits<float>::infinity();
      int64_t maximum_index = 0;
      for (int64_t column = 0; column < row_size; ++column) {
        const auto offset = outer * row_size * inner_rows + column * inner_rows + inner;
        const float value = input[offset];
        if (!std::isnan(value) && (std::isnan(maximum) || value > maximum)) {
          maximum = value;
          maximum_index = column;
        }
        expected_values[offset] = maximum;
        expected_indices[offset] = maximum_index;
      }
    }
  }

  const auto self = cuda_float_tensor(input, {outer_rows, row_size, inner_rows});
  auto values = at::empty_like(self);
  auto indices = at::empty(
      self.sizes(), self.options().dtype(at::kLong));
  at::native::scan_dim_with_indices<float>(
      self,
      values,
      indices,
      1,
      -std::numeric_limits<float>::infinity(),
      MaxOp{});
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  require_scan_result(values, expected_values, indices, expected_indices);
}

TM_TEST(ATenNative, ScanUtilsValueScanDispatch) {
  constexpr int64_t inner_rows = 3;
  constexpr int64_t inner_width = 513;
  std::vector<float> inner_input(inner_rows * inner_width);
  for (int64_t row = 0; row < inner_rows; ++row) {
    for (int64_t column = 0; column < inner_width; ++column) {
      inner_input[row * inner_width + column] = static_cast<float>((column % 7) - 3);
    }
  }
  const auto inner_self = cuda_float_tensor(inner_input, {inner_rows, inner_width});
  auto inner_result = at::empty_like(inner_self);
  at::native::scan_dim<float>(inner_self, inner_result, 1, 0.0f, AddOp{});
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto inner_actual = inner_result.cpu().contiguous();
  const auto* inner_data = inner_actual.data_ptr<float>();
  for (int64_t row = 0; row < inner_rows; ++row) {
    float sum = 0.0f;
    for (int64_t column = 0; column < inner_width; ++column) {
      sum += inner_input[row * inner_width + column];
      TM_REQUIRE_NEAR(inner_data[row * inner_width + column], sum, 1e-5f);
    }
  }

  const std::vector<float> outer_input = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      2.0f, 4.0f, 6.0f,
      1.0f, 3.0f, 5.0f,
      8.0f, 7.0f, 9.0f,
  };
  const auto outer_self = cuda_float_tensor(outer_input, {2, 3, 3});
  auto outer_result = at::empty_like(outer_self);
  at::native::scan_dim<float>(outer_self, outer_result, 1, 0.0f, AddOp{});
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto outer_actual = outer_result.cpu().contiguous();
  const auto* outer_data = outer_actual.data_ptr<float>();
  for (int64_t outer = 0; outer < 2; ++outer) {
    for (int64_t inner = 0; inner < 3; ++inner) {
      float sum = 0.0f;
      for (int64_t column = 0; column < 3; ++column) {
        const auto offset = outer * 9 + column * 3 + inner;
        sum += outer_input[offset];
        TM_REQUIRE_NEAR(outer_data[offset], sum, 1e-6f);
      }
    }
  }

  std::vector<float> flat_input(1025);
  std::iota(flat_input.begin(), flat_input.end(), 1.0f);
  const auto flat_self = cuda_float_tensor(flat_input, {1, 1025});
  auto flat_result = at::empty_like(flat_self);
  at::native::scan_dim<float>(flat_self, flat_result, 1, 0.0f, AddOp{});
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto flat_actual = flat_result.cpu().contiguous();
  const auto* flat_data = flat_actual.data_ptr<float>();
  float flat_sum = 0.0f;
  for (int64_t index = 0; index < static_cast<int64_t>(flat_input.size()); ++index) {
    flat_sum += flat_input[index];
    TM_REQUIRE_NEAR(flat_data[index], flat_sum, 1e-3f);
  }
}
