// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/cub.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <array>
#include <vector>

#include "../../support/TestHarness.h"

namespace {

struct Plus {
  __device__ int operator()(const int& left, const int& right) const {
    return left + right;
  }
};

template <size_t N>
void copy_to_device(const std::array<int, N>& input, int** output) {
  TM_CUDA_REQUIRE(cudaMalloc(output, N * sizeof(int)));
  TM_CUDA_REQUIRE(cudaMemcpy(
      *output, input.data(), N * sizeof(int), cudaMemcpyHostToDevice));
}

template <size_t N>
std::array<int, N> copy_to_host(const int* input) {
  std::array<int, N> output{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      output.data(), input, N * sizeof(int), cudaMemcpyDeviceToHost));
  return output;
}

} // namespace

TM_TEST(ATenCUDA, Cub) {
  c10::cuda::CUDACachingAllocator::init(
      c10::cuda::device_count_ensure_non_zero());

  const std::array<int, 8> sort_keys = {4, 2, 7, 1, 8, 6, 3, 5};
  const std::array<int, 8> sort_values = {40, 20, 70, 10, 80, 60, 30, 50};
  const std::array<int, 3> offsets = {0, 4, 8};
  int *keys_input = nullptr, *keys_output = nullptr, *values_input = nullptr,
      *values_output = nullptr, *segment_offsets = nullptr;
  copy_to_device(sort_keys, &keys_input);
  copy_to_device(sort_values, &values_input);
  copy_to_device(offsets, &segment_offsets);
  TM_CUDA_REQUIRE(cudaMalloc(&keys_output, sort_keys.size() * sizeof(int)));
  TM_CUDA_REQUIRE(cudaMalloc(&values_output, sort_values.size() * sizeof(int)));
  at::cuda::cub::segmented_sort_pairs(
      keys_input,
      keys_output,
      values_input,
      values_output,
      8,
      2,
      segment_offsets,
      segment_offsets + 1,
      false);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_EQ(
      copy_to_host<8>(keys_output),
      (std::array<int, 8>{1, 2, 4, 7, 3, 5, 6, 8}));
  TM_REQUIRE_EQ(
      copy_to_host<8>(values_output),
      (std::array<int, 8>{10, 20, 40, 70, 30, 50, 60, 80}));
  TM_CUDA_REQUIRE(cudaFree(keys_input));
  TM_CUDA_REQUIRE(cudaFree(keys_output));
  TM_CUDA_REQUIRE(cudaFree(values_input));
  TM_CUDA_REQUIRE(cudaFree(values_output));
  TM_CUDA_REQUIRE(cudaFree(segment_offsets));

  const std::array<int, 10> scan_input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int *input = nullptr, *output = nullptr;
  copy_to_device(scan_input, &input);
  TM_CUDA_REQUIRE(cudaMalloc(&output, scan_input.size() * sizeof(int)));
  at::cuda::cub::inclusive_scan(input, output, Plus(), 10);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_EQ(
      copy_to_host<10>(output),
      (std::array<int, 10>{1, 3, 6, 10, 15, 21, 28, 36, 45, 55}));
  at::cuda::cub::exclusive_scan(input, output, Plus(), 0, 10);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_EQ(
      copy_to_host<10>(output),
      (std::array<int, 10>{0, 1, 3, 6, 10, 15, 21, 28, 36, 45}));
  at::cuda::cub::reduce(input, output, 10, Plus(), 0);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_EQ(copy_to_host<1>(output)[0], 55);
  TM_CUDA_REQUIRE(cudaFree(input));
  TM_CUDA_REQUIRE(cudaFree(output));

  const std::array<int, 10> scan_keys = {1, 1, 1, 2, 2, 3, 4, 4, 4, 5};
  const std::array<int, 10> scan_values = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int *scan_keys_input = nullptr, *scan_values_input = nullptr,
      *scan_by_key_output = nullptr;
  copy_to_device(scan_keys, &scan_keys_input);
  copy_to_device(scan_values, &scan_values_input);
  TM_CUDA_REQUIRE(cudaMalloc(
      &scan_by_key_output, scan_values.size() * sizeof(int)));
  at::cuda::cub::inclusive_sum_by_key(
      scan_keys_input,
      scan_values_input,
      scan_by_key_output,
      scan_values.size());
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_EQ(
      copy_to_host<10>(scan_by_key_output),
      (std::array<int, 10>{2, 5, 9, 5, 11, 7, 8, 17, 27, 11}));
  TM_CUDA_REQUIRE(cudaFree(scan_keys_input));
  TM_CUDA_REQUIRE(cudaFree(scan_values_input));
  TM_CUDA_REQUIRE(cudaFree(scan_by_key_output));

  const std::array<int, 10> runs = {1, 1, 1, 2, 2, 3, 3, 3, 3, 4};
  int *runs_input = nullptr, *unique_values = nullptr, *counts = nullptr,
      *run_count = nullptr;
  copy_to_device(runs, &runs_input);
  TM_CUDA_REQUIRE(cudaMalloc(&unique_values, runs.size() * sizeof(int)));
  TM_CUDA_REQUIRE(cudaMalloc(&counts, runs.size() * sizeof(int)));
  TM_CUDA_REQUIRE(cudaMalloc(&run_count, sizeof(int)));
  at::cuda::cub::run_length_encode(
      runs_input, unique_values, counts, run_count, 10);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_EQ(copy_to_host<1>(run_count)[0], 4);
  TM_REQUIRE_EQ(
      copy_to_host<10>(unique_values)[0], 1);
  TM_REQUIRE_EQ(copy_to_host<10>(unique_values)[1], 2);
  TM_REQUIRE_EQ(copy_to_host<10>(counts)[0], 3);
  TM_REQUIRE_EQ(copy_to_host<10>(counts)[1], 2);
  TM_CUDA_REQUIRE(cudaFree(runs_input));
  TM_CUDA_REQUIRE(cudaFree(unique_values));
  TM_CUDA_REQUIRE(cudaFree(counts));
  TM_CUDA_REQUIRE(cudaFree(run_count));
}
