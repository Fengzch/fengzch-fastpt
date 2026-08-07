// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/Normalization.cuh>

#include <array>
#include <limits>
#include <tuple>
#include <vector>

#include "../../../support/TestHarness.h"

struct VarianceTransform {
  __device__ float operator()(float variance, double epsilon) const {
    return rsqrtf(variance + static_cast<float>(epsilon));
  }
};

TM_TEST(ATenNative, Normalization) {
  std::array<float, 8> input_values{2.0f, 2.0f, 3.0f, 3.0f,
                                    6.0f, 6.0f, 7.0f, 7.0f};
  const auto input = at::from_blob(
                         input_values.data(),
                         {2, 2, 2, 1},
                         at::TensorOptions().dtype(at::kFloat))
                         .clone()
                         .to(at::kCUDA);
  auto mean = at::empty({2}, input.options());
  auto invstd = at::empty({2}, input.options());
  at::native::batch_norm_stats_cuda_template<float, int32_t, VarianceTransform>(
      mean, invstd, input, 0.0);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(mean.cpu(), at::tensor({4.0f, 5.0f})));
  TM_REQUIRE(at::allclose(invstd.cpu(), at::tensor({0.5f, 0.5f}), 1e-5, 1e-5));

  auto output = at::empty_like(input);
  const auto weight = at::tensor({2.0f, 3.0f}, input.options());
  const auto bias = at::tensor({1.0f, -1.0f}, input.options());
  at::native::batch_norm_elemt_cuda_template<float, float, int32_t>(
      output, input, weight, bias, mean, invstd);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto expected = (input - mean.view({1, 2, 1, 1})) *
          invstd.view({1, 2, 1, 1}) * weight.view({1, 2, 1, 1}) +
      bias.view({1, 2, 1, 1});
  TM_REQUIRE(at::allclose(output.cpu(), expected.cpu(), 1e-5, 1e-5));
}

TM_TEST(ATenNative, NormalizationBackwardAndDefaultParameters) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t channels = 2;
  constexpr int64_t features = 2;
  constexpr int64_t channel_elements = batch_size * features;
  const std::vector<float> input_values = {
      2.0f, 2.0f, 3.0f, 3.0f,
      6.0f, 6.0f, 7.0f, 7.0f,
  };
  const std::vector<float> grad_values = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
  };
  const auto input = at::tensor(input_values, at::TensorOptions().dtype(at::kFloat))
                         .reshape({batch_size, channels, features, 1})
                         .to(at::kCUDA);
  const auto grad_output = at::tensor(grad_values, input.options())
                               .reshape(input.sizes());
  auto mean = at::empty({channels}, input.options());
  auto invstd = at::empty({channels}, input.options());
  at::native::batch_norm_stats_cuda_template<float, int32_t, at::native::InvStd>(
      mean, invstd, input, 0.0);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(mean.cpu(), at::tensor({4.0f, 5.0f})));
  TM_REQUIRE(at::allclose(invstd.cpu(), at::tensor({0.5f, 0.5f}), 1e-5, 1e-5));

  const auto weight = at::tensor({2.0f, 3.0f}, input.options());
  const auto reduced = at::native::batch_norm_backward_reduce_cuda_template<float, float, int32_t>(
      grad_output, input, mean, invstd, weight, true, true, true);
  const auto& sum_dy = std::get<0>(reduced);
  const auto& sum_dy_xmu = std::get<1>(reduced);
  const auto& grad_weight = std::get<2>(reduced);
  const auto& grad_bias = std::get<3>(reduced);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());

  std::vector<float> expected_sum_dy(channels, 0.0f);
  std::vector<float> expected_sum_dy_xmu(channels, 0.0f);
  std::vector<float> expected_grad_weight(channels, 0.0f);
  std::vector<float> expected_grad_input(input_values.size());
  constexpr std::array<float, channels> expected_mean = {4.0f, 5.0f};
  constexpr std::array<float, channels> expected_invstd = {0.5f, 0.5f};
  constexpr std::array<float, channels> expected_weight = {2.0f, 3.0f};
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      for (int64_t feature = 0; feature < features; ++feature) {
        const auto offset = (batch * channels + channel) * features + feature;
        expected_sum_dy[channel] += grad_values[offset];
        expected_sum_dy_xmu[channel] +=
            grad_values[offset] * (input_values[offset] - expected_mean[channel]);
      }
    }
  }
  for (int64_t channel = 0; channel < channels; ++channel) {
    expected_grad_weight[channel] = expected_sum_dy_xmu[channel] * expected_invstd[channel];
  }
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      for (int64_t feature = 0; feature < features; ++feature) {
        const auto offset = (batch * channels + channel) * features + feature;
        const float norm = 1.0f / static_cast<float>(channel_elements);
        expected_grad_input[offset] =
            (grad_values[offset] - expected_sum_dy[channel] * norm -
             (input_values[offset] - expected_mean[channel]) *
                 expected_invstd[channel] * expected_invstd[channel] *
                 expected_sum_dy_xmu[channel] * norm) *
            expected_weight[channel] * expected_invstd[channel];
      }
    }
  }
  TM_REQUIRE(at::allclose(
      sum_dy.cpu(), at::tensor(expected_sum_dy), 1e-5, 1e-5));
  TM_REQUIRE(at::allclose(
      sum_dy_xmu.cpu(), at::tensor(expected_sum_dy_xmu), 1e-5, 1e-5));
  TM_REQUIRE(at::allclose(
      grad_weight.cpu(), at::tensor(expected_grad_weight), 1e-5, 1e-5));
  TM_REQUIRE(at::allclose(
      grad_bias.cpu(), at::tensor(expected_sum_dy), 1e-5, 1e-5));

  const auto grad_input = at::native::batch_norm_backward_elemt_cuda_template<float, float, int32_t>(
      grad_output, input, mean, invstd, weight, sum_dy, sum_dy_xmu);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto expected_grad_input_tensor = at::tensor(expected_grad_input)
                                              .reshape(input.sizes());
  TM_REQUIRE(at::allclose(grad_input.cpu(), expected_grad_input_tensor, 1e-5, 1e-5));

  const auto train_backward = at::native::batch_norm_backward_cuda_template<float, float, int32_t>(
      grad_output,
      input,
      weight,
      at::Tensor(),
      at::Tensor(),
      mean,
      invstd,
      true,
      0.0,
      {true, true, true});
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(
      std::get<0>(train_backward).cpu(), expected_grad_input_tensor, 1e-5, 1e-5));
  TM_REQUIRE(at::allclose(
      std::get<1>(train_backward).cpu(), at::tensor(expected_grad_weight), 1e-5, 1e-5));
  TM_REQUIRE(at::allclose(
      std::get<2>(train_backward).cpu(), at::tensor(expected_sum_dy), 1e-5, 1e-5));

  const auto running_mean = at::tensor({1.0f, -2.0f}, input.options());
  const auto running_var = at::tensor({4.0f, 9.0f}, input.options());
  const auto eval_backward = at::native::batch_norm_backward_cuda_template<float, float, int32_t>(
      grad_output,
      input,
      at::Tensor(),
      running_mean,
      running_var,
      at::Tensor(),
      at::Tensor(),
      false,
      0.0,
      {true, false, false});
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  std::vector<float> expected_eval_grad(input_values.size());
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      const float inverse_std = channel == 0 ? 0.5f : 1.0f / 3.0f;
      for (int64_t feature = 0; feature < features; ++feature) {
        const auto offset = (batch * channels + channel) * features + feature;
        expected_eval_grad[offset] = grad_values[offset] * inverse_std;
      }
    }
  }
  TM_REQUIRE(at::allclose(
      std::get<0>(eval_backward).cpu(),
      at::tensor(expected_eval_grad).reshape(input.sizes()),
      1e-5,
      1e-5));
  TM_REQUIRE(!std::get<1>(eval_backward).defined());
  TM_REQUIRE(!std::get<2>(eval_backward).defined());

  const auto constant = at::full({2, channels, features, 1}, 3.0f, input.options());
  auto constant_mean = at::empty({channels}, input.options());
  auto constant_invstd = at::empty({channels}, input.options());
  at::native::batch_norm_stats_cuda_template<float, int32_t, at::native::InvStd>(
      constant_mean, constant_invstd, constant, 0.0);
  auto default_output = at::empty_like(constant);
  at::native::batch_norm_elemt_cuda_template<float, float, int32_t>(
      default_output,
      constant,
      at::Tensor(),
      at::Tensor(),
      constant_mean,
      constant_invstd);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(constant_mean.cpu(), at::tensor({3.0f, 3.0f})));
  TM_REQUIRE(at::equal(constant_invstd.cpu(), at::zeros({channels}, at::kFloat)));
  TM_REQUIRE(at::equal(default_output.cpu(), at::zeros_like(default_output.cpu())));
}

TM_TEST(ATenNative, NormalizationChannelsLastFusedForward) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t channels = 3;
  constexpr int64_t height = 2;
  constexpr int64_t width = 2;
  std::vector<float> input_values(batch_size * channels * height * width);
  std::vector<float> z_values(input_values.size());
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      for (int64_t feature = 0; feature < height * width; ++feature) {
        const auto offset = (batch * channels + channel) * height * width + feature;
        input_values[offset] = static_cast<float>(batch * 7 + channel * 3 + feature - 4);
        z_values[offset] = static_cast<float>((feature % 3) - 1);
      }
    }
  }
  const auto cpu_input = at::tensor(input_values, at::TensorOptions().dtype(at::kFloat))
                             .reshape({batch_size, channels, height, width});
  const auto input = cpu_input.to(at::kCUDA).contiguous(at::MemoryFormat::ChannelsLast);
  const auto z = at::tensor(z_values, at::TensorOptions().dtype(at::kFloat))
                     .reshape(input.sizes())
                     .to(at::kCUDA)
                     .contiguous(at::MemoryFormat::ChannelsLast);
  auto mean = at::empty({channels}, input.options());
  auto invstd = at::empty({channels}, input.options());
  constexpr double epsilon = 0.25;
  at::native::batch_norm_stats_channels_last_cuda_template<float, at::native::InvStd>(
      mean, invstd, input, epsilon);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());

  const auto weight = at::tensor({1.0f, 2.0f, -1.0f}, input.options());
  const auto bias = at::tensor({0.0f, -1.0f, 2.0f}, input.options());
  auto output = at::empty(
      input.sizes(), input.options().memory_format(at::MemoryFormat::ChannelsLast));
  at::native::batch_norm_elemt_channels_last_cuda_template(
      output, input, weight, bias, mean, invstd, z, true);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());

  std::vector<float> expected_mean(channels, 0.0f);
  std::vector<float> expected_invstd(channels, 0.0f);
  std::vector<float> expected_output(input_values.size());
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      for (int64_t feature = 0; feature < height * width; ++feature) {
        const auto offset = (batch * channels + channel) * height * width + feature;
        expected_mean[channel] += input_values[offset];
      }
    }
    expected_mean[channel] /= static_cast<float>(batch_size * height * width);
    float variance = 0.0f;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      for (int64_t feature = 0; feature < height * width; ++feature) {
        const auto offset = (batch * channels + channel) * height * width + feature;
        const float delta = input_values[offset] - expected_mean[channel];
        variance += delta * delta;
      }
    }
    variance /= static_cast<float>(batch_size * height * width);
    expected_invstd[channel] = 1.0f / std::sqrt(variance + static_cast<float>(epsilon));
  }
  constexpr std::array<float, channels> host_weight = {1.0f, 2.0f, -1.0f};
  constexpr std::array<float, channels> host_bias = {0.0f, -1.0f, 2.0f};
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      for (int64_t feature = 0; feature < height * width; ++feature) {
        const auto offset = (batch * channels + channel) * height * width + feature;
        const float normalized = host_weight[channel] *
                (input_values[offset] - expected_mean[channel]) * expected_invstd[channel] +
            host_bias[channel] + z_values[offset];
        expected_output[offset] = std::max(0.0f, normalized);
      }
    }
  }
  TM_REQUIRE(at::allclose(mean.cpu(), at::tensor(expected_mean), 1e-5, 1e-5));
  TM_REQUIRE(at::allclose(invstd.cpu(), at::tensor(expected_invstd), 1e-5, 1e-5));
  TM_REQUIRE(at::allclose(
      output.cpu().contiguous(),
      at::tensor(expected_output).reshape(input.sizes()),
      1e-5,
      1e-5));
}
