// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/native/cuda/DistributionTemplates.h>

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <mutex>

#include "../../support/TestHarness.h"

TM_TEST(ATenNative, DistributionTemplates) {
  constexpr int64_t size = 1024;
  auto values = at::empty({size}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto iterator = at::TensorIterator::borrowing_nullary_op(values);
  c10::optional<at::Generator> generator;
  const auto cuda_generator = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      generator, at::cuda::detail::getDefaultCUDAGenerator(0));
  at::native::templates::cuda::random_from_to_kernel(
      iterator, 7, 11, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto host_values = values.cpu();
  TM_REQUIRE(host_values.ge(11).all().item<bool>());
  TM_REQUIRE(host_values.lt(18).all().item<bool>());

  auto probabilities = at::full({size}, 1.0f, values.options());
  at::PhiloxCudaState state;
  {
    std::lock_guard<std::mutex> lock(cuda_generator->mutex_);
    state = cuda_generator->philox_cuda_state(4);
  }
  at::native::templates::cuda::bernoulli_tensor_cuda_kernel<float, float>(
      values, probabilities, state);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(values.eq(1.0f).all().item<bool>());

  auto float_iterator = at::TensorIterator::borrowing_nullary_op(values);
  at::native::templates::cuda::uniform_kernel(
      float_iterator, -2.0, 3.0, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(values.ge(-2.0f).all().item<bool>());
  TM_REQUIRE(values.lt(3.0f).all().item<bool>());

  at::native::templates::cuda::normal_kernel(
      values, 1.25, 0.0, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(values.eq(1.25f).all().item<bool>());

  float_iterator = at::TensorIterator::borrowing_nullary_op(values);
  at::native::templates::cuda::log_normal_kernel(
      float_iterator, 0.0, 0.0, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(values.eq(1.0f).all().item<bool>());

  float_iterator = at::TensorIterator::borrowing_nullary_op(values);
  at::native::templates::cuda::exponential_kernel(
      float_iterator, 1.0, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(values.ge(0.0f).all().item<bool>());
  TM_REQUIRE(at::isfinite(values).all().item<bool>());

  float_iterator = at::TensorIterator::borrowing_nullary_op(values);
  at::native::templates::cuda::cauchy_kernel(
      float_iterator, 0.0, 1.0, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::isfinite(values).all().item<bool>());

  auto integer_values = at::empty(
      {size}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  auto integer_iterator = at::TensorIterator::borrowing_nullary_op(integer_values);
  at::native::templates::cuda::geometric_kernel(
      integer_iterator, 0.5, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(integer_values.ge(1).all().item<bool>());

  integer_values.zero_();
  integer_iterator = at::TensorIterator::borrowing_nullary_op(integer_values);
  at::native::templates::cuda::random_full_64_bits_range_kernel(
      integer_iterator, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(integer_values.ne(0).any().item<bool>());

  integer_values.zero_();
  integer_iterator = at::TensorIterator::borrowing_nullary_op(integer_values);
  at::native::templates::cuda::random_kernel(integer_iterator, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(integer_values.ne(0).any().item<bool>());

  float_iterator = at::TensorIterator::borrowing_nullary_op(values);
  at::native::templates::cuda::bernoulli_kernel(
      float_iterator, 0.0, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(values.eq(0.0f).all().item<bool>());
  float_iterator = at::TensorIterator::borrowing_nullary_op(values);
  at::native::templates::cuda::bernoulli_kernel(
      float_iterator, 1.0, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(values.eq(1.0f).all().item<bool>());

  auto tensor_probabilities = at::full({size}, 0.0f, values.options());
  at::native::templates::cuda::bernoulli_kernel(
      values, tensor_probabilities, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(values.eq(0.0f).all().item<bool>());

  auto non_contiguous_values = at::empty(
      {32, 33}, values.options()).transpose(0, 1);
  auto non_contiguous_iterator =
      at::TensorIterator::borrowing_nullary_op(non_contiguous_values);
  at::native::templates::cuda::uniform_kernel(
      non_contiguous_iterator, 4.0, 5.0, cuda_generator);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(non_contiguous_values.ge(4.0f).all().item<bool>());
  TM_REQUIRE(non_contiguous_values.lt(5.0f).all().item<bool>());
}
