// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/cuda/PersistentSoftmax.cuh>

#include <limits>
#include <vector>

#include "../../support/TestHarness.h"

TM_TEST(ATenNative, PersistentSoftmaxForwardBackward) {
  constexpr int64_t batch_count = 3;
  constexpr int64_t element_count = 7;
  const auto input = at::randn(
      {batch_count, element_count}, at::device(at::kCUDA).dtype(at::kFloat));
  auto output = at::empty_like(input);
  dispatch_softmax_forward<float, float, float, false, false>(
      output.data_ptr<float>(),
      input.data_ptr<float>(),
      element_count,
      element_count,
      batch_count);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto expected = at::softmax(input.cpu(), 1);
  TM_REQUIRE(at::allclose(output.cpu(), expected, 1e-5, 1e-5));

  const auto grad_output = at::randn({batch_count, element_count}, input.options());
  auto grad_input = at::empty_like(input);
  dispatch_softmax_backward<float, float, float, false, false>(
      grad_input.data_ptr<float>(),
      grad_output.data_ptr<float>(),
      output.data_ptr<float>(),
      element_count,
      element_count,
      batch_count);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto grad_cpu = grad_output.cpu();
  const auto expected_grad = grad_cpu - expected * grad_cpu.sum(1, true);
  TM_REQUIRE(at::allclose(grad_input.cpu(), expected_grad, 1e-5, 1e-5));
}

TM_TEST(ATenNative, PersistentSoftmaxStridedAndLogHalf) {
  constexpr int64_t batch_count = 5;
  constexpr int64_t element_count = 13;
  constexpr int64_t stride = 17;
  const auto input = at::randn(
      {batch_count, stride}, at::device(at::kCUDA).dtype(at::kFloat));
  auto output = at::zeros_like(input);
  dispatch_softmax_forward<float, float, float, false, false>(
      output.data_ptr<float>(),
      input.data_ptr<float>(),
      element_count,
      stride,
      batch_count);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto expected = at::softmax(input.cpu().narrow(1, 0, element_count), 1);
  TM_REQUIRE(at::allclose(
      output.cpu().narrow(1, 0, element_count), expected, 1e-5, 1e-5));

  const auto half_input = at::randn(
      {3, 129}, at::device(at::kCUDA).dtype(at::kHalf));
  auto half_output = at::empty_like(half_input);
  dispatch_softmax_forward<at::Half, at::Half, float, true, false>(
      half_output.data_ptr<at::Half>(),
      half_input.data_ptr<at::Half>(),
      129,
      129,
      3);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto half_expected = at::log_softmax(half_input.cpu().to(at::kFloat), 1).to(at::kHalf);
  TM_REQUIRE(at::allclose(half_output.cpu(), half_expected, 2e-3, 2e-3));

  // A stride larger than the reduction width is used by the dispatch API even
  // though the regular SoftMax caller normally provides a contiguous width.
  constexpr int64_t backward_elements = 129;
  constexpr int64_t backward_stride = 137;
  const auto backward_input = at::randn(
      {3, backward_stride}, at::device(at::kCUDA).dtype(at::kFloat));
  auto backward_output = at::zeros_like(backward_input);
  dispatch_softmax_forward<float, float, float, false, false>(
      backward_output.data_ptr<float>(),
      backward_input.data_ptr<float>(),
      backward_elements,
      backward_stride,
      3);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto backward_grad = at::randn_like(backward_input);
  auto weighted_grad = at::zeros_like(backward_input);
  weighted_grad.narrow(1, 0, backward_elements)
      .copy_(backward_grad.narrow(1, 0, backward_elements) *
             backward_output.narrow(1, 0, backward_elements));
  auto backward_result = at::zeros_like(backward_input);
  dispatch_softmax_backward<float, float, float, false, false>(
      backward_result.data_ptr<float>(),
      weighted_grad.data_ptr<float>(),
      backward_output.data_ptr<float>(),
      backward_elements,
      backward_stride,
      3);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto backward_output_cpu = backward_output.cpu().narrow(1, 0, backward_elements);
  const auto backward_grad_cpu = backward_grad.cpu().narrow(1, 0, backward_elements);
  const auto weighted_grad_cpu = weighted_grad.cpu().narrow(1, 0, backward_elements);
  const auto expected_backward = backward_output_cpu *
      (backward_grad_cpu - weighted_grad_cpu.sum(1, true));
  TM_REQUIRE(at::allclose(
      backward_result.cpu().narrow(1, 0, backward_elements),
      expected_backward,
      1e-5,
      1e-5));
}

TM_TEST(ATenNative, PersistentSoftmaxLogBackwardAndMixedPrecision) {
  constexpr int64_t batch_count = 3;
  constexpr int64_t element_count = 129;
  const auto input = at::randn(
      {batch_count, element_count}, at::device(at::kCUDA).dtype(at::kHalf));
  auto output = at::empty(
      {batch_count, element_count}, at::device(at::kCUDA).dtype(at::kFloat));
  dispatch_softmax_forward<at::Half, float, float, true, false>(
      output.data_ptr<float>(),
      input.data_ptr<at::Half>(),
      element_count,
      element_count,
      batch_count);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto expected = at::log_softmax(input.cpu().to(at::kFloat), 1);
  TM_REQUIRE(at::allclose(output.cpu(), expected, 1e-5, 1e-5));

  // This is the half-to-float route used by host_softmax_backward: grad and
  // output are float while the destination gradient is the original Half type.
  const auto grad = at::randn_like(output);
  auto grad_input = at::empty_like(input);
  dispatch_softmax_backward<float, at::Half, float, true, false>(
      grad_input.data_ptr<at::Half>(),
      grad.data_ptr<float>(),
      output.data_ptr<float>(),
      element_count,
      element_count,
      batch_count);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto grad_cpu = grad.cpu();
  const auto expected_grad =
      (grad_cpu - at::exp(expected) * grad_cpu.sum(1, true)).to(at::kHalf);
  TM_REQUIRE(at::allclose(grad_input.cpu(), expected_grad, 2e-3, 2e-3));
}

TM_TEST(ATenNative, PersistentSoftmaxMaskedForwardBackward) {
  constexpr int64_t batch_count = 3;
  constexpr int64_t element_count = 8;
  const auto input = at::randn(
      {batch_count, element_count}, at::device(at::kCUDA).dtype(at::kFloat));
  const std::vector<uint8_t> mask_values = {
      0, 0, 1, 0, 0, 1, 0, 0,
      1, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 1, 0,
  };
  const auto mask = at::tensor(mask_values, at::TensorOptions().dtype(at::kByte))
                        .reshape({batch_count, element_count})
                        .to(at::kBool)
                        .to(at::kCUDA);
  auto output = at::empty_like(input);
  dispatch_softmax_forward<float, float, float, false, true>(
      output.data_ptr<float>(),
      input.data_ptr<float>(),
      element_count,
      element_count,
      batch_count,
      mask.data_ptr<bool>());
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto mask_cpu = mask.cpu();
  const auto expected = at::softmax(
      input.cpu().masked_fill(mask_cpu, -std::numeric_limits<float>::infinity()), 1);
  TM_REQUIRE(at::allclose(output.cpu(), expected, 1e-5, 1e-5));

  const auto grad_output = at::randn({batch_count, element_count}, input.options());
  auto grad_input = at::empty_like(input);
  dispatch_softmax_backward<float, float, float, false, true>(
      grad_input.data_ptr<float>(),
      grad_output.data_ptr<float>(),
      output.data_ptr<float>(),
      element_count,
      element_count,
      batch_count,
      mask.data_ptr<bool>());
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto grad_cpu = grad_output.cpu();
  const auto unmasked_grad = grad_cpu.masked_fill(mask_cpu, 0.0f);
  const auto expected_grad = (grad_cpu - expected * unmasked_grad.sum(1, true))
                                 .masked_fill(mask_cpu, 0.0f);
  TM_REQUIRE(at::allclose(grad_input.cpu(), expected_grad, 1e-5, 1e-5));

  const auto fully_masked = at::ones(
      {1, element_count}, at::device(at::kCUDA).dtype(at::kBool));
  const auto fully_masked_input = at::randn(
      {1, element_count}, at::device(at::kCUDA).dtype(at::kFloat));
  auto fully_masked_output = at::empty_like(fully_masked_input);
  dispatch_softmax_forward<float, float, float, false, true>(
      fully_masked_output.data_ptr<float>(),
      fully_masked_input.data_ptr<float>(),
      element_count,
      element_count,
      1,
      fully_masked.data_ptr<bool>());
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::isnan(fully_masked_output).all().item<bool>());

  const auto fully_masked_grad = at::randn_like(fully_masked_input);
  auto fully_masked_grad_input = at::empty_like(fully_masked_input);
  dispatch_softmax_backward<float, float, float, false, true>(
      fully_masked_grad_input.data_ptr<float>(),
      fully_masked_grad.data_ptr<float>(),
      fully_masked_output.data_ptr<float>(),
      element_count,
      element_count,
      1,
      fully_masked.data_ptr<bool>());
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::equal(fully_masked_grad_input, at::zeros_like(fully_masked_grad_input)));
}

TM_TEST(ATenNative, PersistentSoftmaxBoundaryAndZeroLength) {
  const auto input = at::randn(
      {1, 2048}, at::device(at::kCUDA).dtype(at::kFloat));
  auto output = at::empty_like(input);
  dispatch_softmax_forward<float, float, float, false, false>(
      output.data_ptr<float>(), input.data_ptr<float>(), 2048, 2048, 1);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(output.cpu(), at::softmax(input.cpu(), 1), 2e-5, 2e-5));

  const auto backward_input = at::randn(
      {1, 1024}, at::device(at::kCUDA).dtype(at::kFloat));
  const auto backward_output = at::softmax(backward_input, 1);
  const auto backward_grad = at::randn_like(backward_output);
  const auto weighted_grad = backward_grad * backward_output;
  auto backward_result = at::empty_like(backward_output);
  dispatch_softmax_backward<float, float, float, false, false>(
      backward_result.data_ptr<float>(),
      weighted_grad.data_ptr<float>(),
      backward_output.data_ptr<float>(),
      1024,
      1024,
      1);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto expected_backward = backward_output.cpu() *
      (backward_grad.cpu() - weighted_grad.cpu().sum(1, true));
  TM_REQUIRE(at::allclose(backward_result.cpu(), expected_backward, 2e-5, 2e-5));

  const auto zero_input = at::zeros(
      {2, 1}, at::device(at::kCUDA).dtype(at::kFloat));
  auto zero_output = at::full_like(zero_input, 42.0f);
  dispatch_softmax_forward<float, float, float, false, false>(
      zero_output.data_ptr<float>(), zero_input.data_ptr<float>(), 0, 0, 2);
  TM_REQUIRE(at::equal(zero_output, at::full_like(zero_output, 42.0f)));

  auto zero_grad_input = at::full_like(zero_input, -17.0f);
  dispatch_softmax_backward<float, float, float, false, false>(
      zero_grad_input.data_ptr<float>(), zero_input.data_ptr<float>(),
      zero_input.data_ptr<float>(), 0, 0, 2);
  TM_REQUIRE(at::equal(zero_grad_input, at::full_like(zero_grad_input, -17.0f)));
}

TM_TEST(ATenNative, PersistentSoftmaxTransformerMask) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t num_heads = 2;
  constexpr int64_t sequence_length = 8;
  const auto input = at::randn(
      {batch_size, num_heads, sequence_length, sequence_length},
      at::device(at::kCUDA).dtype(at::kFloat));
  const std::vector<uint8_t> mask_values = {
      0, 0, 1, 0, 0, 0, 1, 0,
      1, 0, 0, 0, 1, 0, 0, 0,
  };
  const auto mask = at::tensor(mask_values, at::TensorOptions().dtype(at::kByte))
                        .reshape({batch_size, sequence_length})
                        .to(at::kBool)
                        .to(at::kCUDA);
  auto output = at::empty_like(input);
  dispatch_softmax_forward<float, float, float, false, true>(
      output.data_ptr<float>(),
      input.data_ptr<float>(),
      sequence_length,
      sequence_length,
      batch_size * num_heads * sequence_length,
      mask.data_ptr<bool>(),
      num_heads * sequence_length * sequence_length,
      true);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  const auto expanded_mask = mask.cpu()
      .view({batch_size, 1, 1, sequence_length})
      .expand({batch_size, num_heads, sequence_length, sequence_length});
  const auto expected = at::softmax(
      input.cpu().masked_fill(expanded_mask, -std::numeric_limits<float>::infinity()),
      3);
  TM_REQUIRE(at::allclose(output.cpu(), expected, 1e-5, 1e-5));
}
