// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/tunable/GemmCommon.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <array>

#include "../support/TestHarness.h"

TM_TEST(C10, GemmCommon) {
  c10::cuda::CUDACachingAllocator::init(
      c10::cuda::device_count_ensure_non_zero());

  at::cuda::tunable::GemmParams<float> parameters;
  parameters.m = parameters.n = parameters.k = 8;
  parameters.lda = parameters.ldb = parameters.ldc = 8;
  parameters.a = static_cast<const float*>(
      c10::cuda::CUDACachingAllocator::raw_alloc(64 * sizeof(float)));
  parameters.b = static_cast<const float*>(
      c10::cuda::CUDACachingAllocator::raw_alloc(64 * sizeof(float)));
  parameters.c = static_cast<float*>(
      c10::cuda::CUDACachingAllocator::raw_alloc(64 * sizeof(float)));

  std::array<float, 64> expected{};
  for (size_t index = 0; index < expected.size(); ++index) {
    expected[index] = static_cast<float>(index) * 0.25f;
  }
  TM_CUDA_REQUIRE(cudaMemcpy(
      parameters.c,
      expected.data(),
      expected.size() * sizeof(float),
      cudaMemcpyHostToDevice));

  auto* copy = parameters.DeepCopy(true);
  TM_REQUIRE(copy != nullptr);
  TM_REQUIRE(copy->a != nullptr);
  TM_REQUIRE(copy->b != nullptr);
  TM_REQUIRE(copy->c != nullptr);
  TM_REQUIRE(copy->a != parameters.a);
  TM_REQUIRE(copy->b != parameters.b);
  TM_REQUIRE(copy->c != parameters.c);
  std::array<float, 64> copied{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      copied.data(),
      copy->c,
      copied.size() * sizeof(float),
      cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(copied, expected);
  copy->Delete();
  delete copy;

  auto* output_only_copy = parameters.DeepCopy(false);
  TM_REQUIRE(output_only_copy != nullptr);
  TM_REQUIRE_EQ(output_only_copy->a, parameters.a);
  TM_REQUIRE_EQ(output_only_copy->b, parameters.b);
  TM_REQUIRE(output_only_copy->c != parameters.c);
  std::array<float, 64> output_only_copied{};
  TM_CUDA_REQUIRE(cudaMemcpy(
      output_only_copied.data(),
      output_only_copy->c,
      output_only_copied.size() * sizeof(float),
      cudaMemcpyDeviceToHost));
  TM_REQUIRE_EQ(output_only_copied, expected);
  output_only_copy->Delete();
  delete output_only_copy;

  c10::cuda::CUDACachingAllocator::raw_delete(const_cast<float*>(parameters.a));
  c10::cuda::CUDACachingAllocator::raw_delete(const_cast<float*>(parameters.b));
  c10::cuda::CUDACachingAllocator::raw_delete(parameters.c);
}
