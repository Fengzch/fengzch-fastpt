// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "../../support/TestHarness.h"

struct AddOp {
  template <typename scalar_t>
  __device__ __forceinline__ void operator()(scalar_t& left, scalar_t right) const {
    left += right;
  }
};

struct AddStep4Op {
  __device__ void operator()(
      int valid,
      float& left0,
      float& left1,
      float& left2,
      float& left3,
      const float& right0,
      const float& right1,
      const float& right2,
      const float& right3) const {
    if (valid > 0) left0 += right0;
    if (valid > 1) left1 += right1;
    if (valid > 2) left2 += right2;
    if (valid > 3) left3 += right3;
  }
};

TM_TEST(ATenCUDA, CUDAApplyUtilsContiguousDefaultStep) {
  auto left = at::randn({1024}, at::device(at::kCUDA).dtype(at::kFloat));
  const auto right = at::randn({1024}, left.options());
  const auto expected = (left + right).cpu();
  TM_REQUIRE((at::cuda::CUDA_tensor_apply2<float, float>(
      left,
      right,
      AddOp(),
      at::cuda::TensorArgType::ReadWrite,
      at::cuda::TensorArgType::ReadOnly)));
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(left.cpu(), expected));
}

TM_TEST(ATenCUDA, CUDAApplyUtilsStepTailAndDouble) {
  constexpr int64_t count = 1027;
  auto left = at::arange(count, at::device(at::kCUDA).dtype(at::kFloat));
  const auto right = at::full({count}, 0.5f, left.options());
  const auto expected = (left + right).cpu();
  TM_REQUIRE((at::cuda::CUDA_tensor_apply2<float, float, 4>(
      left,
      right,
      AddStep4Op(),
      at::cuda::TensorArgType::ReadWrite,
      at::cuda::TensorArgType::ReadOnly)));
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(left.cpu(), expected));

  auto double_left = at::arange(257, at::device(at::kCUDA).dtype(at::kDouble));
  const auto double_right = at::full({257}, -1.25, double_left.options());
  const auto double_expected = (double_left + double_right).cpu();
  TM_REQUIRE((at::cuda::CUDA_tensor_apply2<double, double>(
      double_left,
      double_right,
      AddOp(),
      at::cuda::TensorArgType::ReadWrite,
      at::cuda::TensorArgType::ReadOnly)));
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(double_left.cpu(), double_expected, 1e-12, 1e-12));
}

TM_TEST(ATenCUDA, CUDAApplyUtilsNonContiguous) {
  auto left = at::arange(30, at::device(at::kCUDA).dtype(at::kFloat))
                  .reshape({5, 6})
                  .transpose(0, 1);
  const auto right = at::full({6, 5}, 3.0f, left.options());
  const auto expected = (left + right).cpu();
  TM_REQUIRE(!left.is_contiguous());
  TM_REQUIRE((at::cuda::CUDA_tensor_apply2<float, float>(
      left,
      right,
      AddOp(),
      at::cuda::TensorArgType::ReadWrite,
      at::cuda::TensorArgType::ReadOnly)));
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(left.cpu(), expected));
}

TM_TEST(ATenCUDA, CUDAApplyUtilsEmptyAndValidation) {
  const auto options = at::device(at::kCUDA).dtype(at::kFloat);
  auto empty_left = at::empty({0}, options);
  const auto empty_right = at::empty({0}, options);
  TM_REQUIRE((at::cuda::CUDA_tensor_apply2<float, float>(
      empty_left,
      empty_right,
      AddOp(),
      at::cuda::TensorArgType::ReadWrite,
      at::cuda::TensorArgType::ReadOnly)));

  auto left = at::ones({4}, options);
  const auto mismatched = at::ones({5}, options);
  const auto mismatched_result = at::cuda::CUDA_tensor_apply2<float, float>(
      left,
      mismatched,
      AddOp(),
      at::cuda::TensorArgType::ReadWrite,
      at::cuda::TensorArgType::ReadOnly);
  TM_REQUIRE(!mismatched_result);
  bool cpu_inputs_throw = false;
  try {
    (void)at::cuda::CUDA_tensor_apply2<float, float>(
        at::ones({1}),
        at::ones({1}),
        AddOp(),
        at::cuda::TensorArgType::ReadWrite,
        at::cuda::TensorArgType::ReadOnly);
  } catch (...) {
    cpu_inputs_throw = true;
  }
  TM_REQUIRE(cpu_inputs_throw);
}
