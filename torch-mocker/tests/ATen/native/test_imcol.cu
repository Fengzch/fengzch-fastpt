// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/ops/col2im.h>
#include <ATen/ops/im2col.h>
#include <ATen/native/cuda/im2col.cuh>

#include <array>
#include <vector>

#include "../../support/TestHarness.h"

namespace {

int64_t output_extent(
    int64_t input,
    int64_t kernel,
    int64_t padding,
    int64_t stride,
    int64_t dilation) {
  return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

std::vector<float> reference_im2col(
    const std::vector<float>& input,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t padding_height,
    int64_t padding_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t dilation_height,
    int64_t dilation_width) {
  const auto height_col = output_extent(
      height, kernel_height, padding_height, stride_height, dilation_height);
  const auto width_col = output_extent(
      width, kernel_width, padding_width, stride_width, dilation_width);
  std::vector<float> column(
      static_cast<size_t>(channels * kernel_height * kernel_width * height_col * width_col));
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t h_out = 0; h_out < height_col; ++h_out) {
      for (int64_t w_out = 0; w_out < width_col; ++w_out) {
        for (int64_t kh = 0; kh < kernel_height; ++kh) {
          for (int64_t kw = 0; kw < kernel_width; ++kw) {
            const auto h = h_out * stride_height - padding_height + kh * dilation_height;
            const auto w = w_out * stride_width - padding_width + kw * dilation_width;
            const auto column_index =
                (((channel * kernel_height + kh) * kernel_width + kw) * height_col + h_out) *
                    width_col +
                w_out;
            if (h >= 0 && h < height && w >= 0 && w < width) {
              column[static_cast<size_t>(column_index)] =
                  input[static_cast<size_t>((channel * height + h) * width + w)];
            }
          }
        }
      }
    }
  }
  return column;
}

std::vector<float> reference_col2im(
    const std::vector<float>& column,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t padding_height,
    int64_t padding_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t dilation_height,
    int64_t dilation_width) {
  const auto height_col = output_extent(
      height, kernel_height, padding_height, stride_height, dilation_height);
  const auto width_col = output_extent(
      width, kernel_width, padding_width, stride_width, dilation_width);
  std::vector<float> image(static_cast<size_t>(channels * height * width), 0.0f);
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t h_out = 0; h_out < height_col; ++h_out) {
      for (int64_t w_out = 0; w_out < width_col; ++w_out) {
        for (int64_t kh = 0; kh < kernel_height; ++kh) {
          for (int64_t kw = 0; kw < kernel_width; ++kw) {
            const auto h = h_out * stride_height - padding_height + kh * dilation_height;
            const auto w = w_out * stride_width - padding_width + kw * dilation_width;
            if (h >= 0 && h < height && w >= 0 && w < width) {
              const auto column_index =
                  (((channel * kernel_height + kh) * kernel_width + kw) * height_col + h_out) *
                      width_col +
                  w_out;
              image[static_cast<size_t>((channel * height + h) * width + w)] +=
                  column[static_cast<size_t>(column_index)];
            }
          }
        }
      }
    }
  }
  return image;
}

void require_equal(const std::vector<float>& actual, const std::vector<float>& expected) {
  TM_REQUIRE_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    TM_REQUIRE_NEAR(actual[index], expected[index], 1e-5f);
  }
}

std::vector<float> copy_from_device(const float* device, size_t count) {
  std::vector<float> host(count);
  TM_CUDA_REQUIRE(cudaMemcpy(
      host.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost));
  return host;
}

} // namespace

TM_TEST(ATenNative, ImcolRawKernels) {
  constexpr int64_t channels = 2;
  constexpr int64_t height = 3;
  constexpr int64_t width = 4;
  constexpr int64_t kernel_height = 2;
  constexpr int64_t kernel_width = 3;
  constexpr int64_t padding_height = 1;
  constexpr int64_t padding_width = 1;
  constexpr int64_t stride_height = 1;
  constexpr int64_t stride_width = 2;
  constexpr int64_t dilation_height = 1;
  constexpr int64_t dilation_width = 1;
  const auto height_col = output_extent(
      height, kernel_height, padding_height, stride_height, dilation_height);
  const auto width_col = output_extent(
      width, kernel_width, padding_width, stride_width, dilation_width);
  const std::vector<float> input = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  const auto expected_column = reference_im2col(
      input, channels, height, width, kernel_height, kernel_width, padding_height,
      padding_width, stride_height, stride_width, dilation_height, dilation_width);
  const auto expected_image = reference_col2im(
      expected_column, channels, height, width, kernel_height, kernel_width,
      padding_height, padding_width, stride_height, stride_width, dilation_height,
      dilation_width);

  float *device_input = nullptr, *device_column = nullptr, *device_image = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_input, input.size() * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_column, expected_column.size() * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_image, expected_image.size() * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMemcpy(
      device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaStream_t stream = nullptr;
  TM_CUDA_REQUIRE(cudaStreamCreate(&stream));

  at::native::im2col<float>(
      stream, device_input, channels, height, width, height_col, width_col,
      kernel_height, kernel_width, padding_height, padding_width, stride_height,
      stride_width, dilation_height, dilation_width, device_column);
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream));
  require_equal(copy_from_device(device_column, expected_column.size()), expected_column);

  at::native::col2im<float, float>(
      stream, device_column, channels, height, width, height_col, width_col,
      kernel_height, kernel_width, padding_height, padding_width, stride_height,
      stride_width, dilation_height, dilation_width, device_image);
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream));
  require_equal(copy_from_device(device_image, expected_image.size()), expected_image);

  TM_CUDA_REQUIRE(cudaStreamDestroy(stream));
  TM_CUDA_REQUIRE(cudaFree(device_input));
  TM_CUDA_REQUIRE(cudaFree(device_column));
  TM_CUDA_REQUIRE(cudaFree(device_image));
}

TM_TEST(ATenNative, ImcolBatchedRawKernel) {
  constexpr int64_t batch = 2;
  constexpr int64_t channels = 1;
  constexpr int64_t height = 3;
  constexpr int64_t width = 3;
  constexpr int64_t kernel = 2;
  constexpr int64_t height_col = 2;
  constexpr int64_t width_col = 2;
  const std::vector<float> first_column = {
      1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9};
  std::vector<float> batched_column = first_column;
  for (auto value : first_column) {
    batched_column.push_back(value * 2.0f);
  }
  const auto first_image = reference_col2im(
      first_column, channels, height, width, kernel, kernel, 0, 0, 1, 1, 1, 1);
  std::vector<float> expected_image = first_image;
  for (auto value : first_image) {
    expected_image.push_back(value * 2.0f);
  }

  float *device_column = nullptr, *device_image = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_column, batched_column.size() * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_image, expected_image.size() * sizeof(float)));
  TM_CUDA_REQUIRE(cudaMemcpy(
      device_column, batched_column.data(), batched_column.size() * sizeof(float),
      cudaMemcpyHostToDevice));
  cudaStream_t stream = nullptr;
  TM_CUDA_REQUIRE(cudaStreamCreate(&stream));
  at::native::col2im_batched<float>(
      stream, device_column, static_cast<int64_t>(first_column.size()), batch, channels,
      height, width, height_col, width_col, kernel, kernel, 0, 0, 1, 1, 1, 1,
      device_image, static_cast<int64_t>(first_image.size()));
  TM_CUDA_REQUIRE(cudaStreamSynchronize(stream));
  require_equal(copy_from_device(device_image, expected_image.size()), expected_image);
  TM_CUDA_REQUIRE(cudaStreamDestroy(stream));
  TM_CUDA_REQUIRE(cudaFree(device_column));
  TM_CUDA_REQUIRE(cudaFree(device_image));
}

TM_TEST(ATenNative, ImcolPublicOperators) {
  const auto cpu_input = at::arange(1, 49, at::TensorOptions().dtype(at::kFloat))
                             .reshape({2, 1, 4, 6});
  const auto gpu_input = cpu_input.to(at::kCUDA);
  const std::array<int64_t, 2> kernel = {2, 3};
  const std::array<int64_t, 2> dilation = {1, 2};
  const std::array<int64_t, 2> padding = {1, 2};
  const std::array<int64_t, 2> stride = {2, 2};

  const auto expected_column = at::im2col(cpu_input, kernel, dilation, padding, stride);
  const auto actual_column = at::im2col(gpu_input, kernel, dilation, padding, stride);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(actual_column.cpu(), expected_column, 1e-5, 1e-5));

  auto column_out = at::empty({1}, gpu_input.options());
  at::im2col_outf(gpu_input, kernel, dilation, padding, stride, column_out);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(column_out.cpu(), expected_column, 1e-5, 1e-5));

  const std::array<int64_t, 2> output_size = {4, 6};
  const auto expected_image = at::col2im(expected_column, output_size, kernel, dilation, padding, stride);
  const auto actual_image = at::col2im(actual_column, output_size, kernel, dilation, padding, stride);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(actual_image.cpu(), expected_image, 1e-5, 1e-5));

  auto image_out = at::empty({1}, gpu_input.options());
  at::col2im_outf(actual_column, output_size, kernel, dilation, padding, stride, image_out);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(image_out.cpu(), expected_image, 1e-5, 1e-5));

  const auto unbatched_input = gpu_input.select(0, 0).transpose(1, 2);
  const auto unbatched_expected = at::im2col(
      unbatched_input.cpu(), kernel, dilation, padding, stride);
  const auto unbatched_actual = at::im2col(unbatched_input, kernel, dilation, padding, stride);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(unbatched_actual.cpu(), unbatched_expected, 1e-5, 1e-5));
}

TM_TEST(ATenNative, ImcolValidation) {
  const auto input = at::ones({1, 1, 4, 4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  TM_REQUIRE_THROWS(at::im2col(input, {2}, {1, 1}, {0, 0}, {1, 1}));
  TM_REQUIRE_THROWS(at::im2col(input, {2, 2}, {1, 1}, {0, 0}, {0, 1}));
  TM_REQUIRE_THROWS(at::col2im(input, {4, 4}, {2, 2}, {1, 1}, {0, 0}, {1, 1}));
}
