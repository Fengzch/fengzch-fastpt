// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/native/cuda/vol2col.cuh>

#include <array>

#include "../../support/TestHarness.h"

TM_TEST(ATenNative, Vol2col) {
  constexpr int64_t channels = 1;
  constexpr int64_t depth = 2;
  constexpr int64_t height = 2;
  constexpr int64_t width = 3;
  constexpr size_t count = channels * depth * height * width;
  const std::array<float, count> input = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  float* device_volume = nullptr;
  float* device_column = nullptr;
  float* device_output = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_volume, count * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_column, count * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_output, count * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMemcpy(
      device_volume, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream = nullptr;
  TM_CUDA_REQUIRE(cudaStreamCreate(&stream));
  at::native::vol2col<float>(
      stream,
      device_volume,
      channels,
      depth,
      height,
      width,
      depth,
      height,
      width,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      device_column);
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream));
  std::array<float, count> column{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      column.data(), device_column, count * sizeof(float), cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(column, input);

  at::native::col2vol<float, float>(
      stream,
      device_column,
      channels,
      depth,
      height,
      width,
      depth,
      height,
      width,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      device_output);
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream));
  std::array<float, count> output{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      output.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(output, input);
  TM_CUDA_REQUIRE(cudaStreamDestroy(stream));
  TM_CUDA_REQUIRE(cudaFree(device_volume));
  TM_CUDA_REQUIRE(cudaFree(device_column));
  TM_CUDA_REQUIRE(cudaFree(device_output));
}
