// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/SharedReduceOps.h>

#include "../../support/TestHarness.h"

namespace {

at::TensorIterator reduction_iterator(
    at::Tensor& output,
    const at::Tensor& input) {
  return at::TensorIteratorConfig()
      .add_output(output)
      .add_input(input)
      .is_reduction(true)
      .resize_outputs(false)
      .build();
}

} // namespace

TM_TEST(ATenNative, Reduce) {
  const auto input = at::arange(
      1, 17, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto output = at::empty({}, input.options());
  auto iterator = reduction_iterator(output, input);
  const auto sum_combine = [] GPU_LAMBDA(float left, float right) {
    return left + right;
  };
  at::native::gpu_reduce_kernel<float, float>(
      iterator, at::native::func_wrapper<float>(sum_combine));
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_NEAR(output.cpu().item<float>(), 136.0f, 1e-5f);

  auto product_output = at::empty({}, input.options());
  auto product_iterator = reduction_iterator(product_output, input.slice(0, 0, 4));
  const auto product_combine = [] GPU_LAMBDA(float left, float right) {
    return left * right;
  };
  at::native::gpu_reduce_kernel<float, float>(
      product_iterator,
      at::native::func_wrapper<float>(product_combine),
      1.0f);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_NEAR(product_output.cpu().item<float>(), 24.0f, 1e-5f);

  auto large_input = at::ones(
      {1 << 20}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto large_output = at::empty({}, large_input.options());
  auto large_iterator = reduction_iterator(large_output, large_input);
  at::native::gpu_reduce_kernel<float, float>(
      large_iterator, at::native::func_wrapper<float>(sum_combine));
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_NEAR(
      large_output.cpu().item<float>(), static_cast<float>(large_input.numel()), 1e-3f);

  auto non_contiguous_input = at::arange(
      1, 1025, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA))
                                  .reshape({32, 32})
                                  .transpose(0, 1);
  auto non_contiguous_output = at::empty({}, input.options());
  auto non_contiguous_iterator =
      reduction_iterator(non_contiguous_output, non_contiguous_input);
  at::native::gpu_reduce_kernel<float, float>(
      non_contiguous_iterator, at::native::func_wrapper<float>(sum_combine));
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_NEAR(non_contiguous_output.cpu().item<float>(), 524800.0f, 1e-3f);

  auto bool_values = at::ones(
      {1024}, at::TensorOptions().dtype(at::kBool).device(at::kCUDA));
  auto bool_output = at::empty({}, bool_values.options());
  auto all_iterator = reduction_iterator(bool_output, bool_values);
  const auto and_combine = [] GPU_LAMBDA(bool left, bool right) {
    return left && right;
  };
  at::native::gpu_reduce_kernel<bool, bool>(
      all_iterator, at::native::func_wrapper<bool>(and_combine), true);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(bool_output.cpu().item<bool>());

  bool_values.zero_();
  auto any_iterator = reduction_iterator(bool_output, bool_values);
  const auto or_combine = [] GPU_LAMBDA(bool left, bool right) {
    return left || right;
  };
  at::native::gpu_reduce_kernel<bool, bool>(
      any_iterator, at::native::func_wrapper<bool>(or_combine), false);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(!bool_output.cpu().item<bool>());
}
