// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>

#include <vector>

#include "../../support/TestHarness.h"

struct ScaleTensor {
  __device__ void operator()(
      int64_t chunk_size,
      const at::native::TensorListMetadata<1>& metadata,
      float scale) const {
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    const int chunk_index = metadata.block_to_chunk[blockIdx.x];
    auto* data = static_cast<float*>(
        const_cast<void*>(metadata.addresses[0][tensor_index]));
    const int64_t offset = chunk_index * chunk_size;
    const int64_t remaining = metadata.numel_for_tensor[tensor_index] - offset;
    const int64_t count = remaining < chunk_size ? remaining : chunk_size;
    for (int64_t index = threadIdx.x; index < count; index += blockDim.x) {
      data[offset + index] *= scale;
    }
  }
};

struct AddTensorLists {
  __device__ void operator()(
      int64_t chunk_size,
      const at::native::TensorListMetadata<2>& metadata,
      float scale) const {
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    const int chunk_index = metadata.block_to_chunk[blockIdx.x];
    auto* output = static_cast<float*>(
        const_cast<void*>(metadata.addresses[0][tensor_index]));
    const auto* input = static_cast<const float*>(metadata.addresses[1][tensor_index]);
    const int64_t offset = chunk_index * chunk_size;
    const int64_t remaining = metadata.numel_for_tensor[tensor_index] - offset;
    const int64_t count = remaining < chunk_size ? remaining : chunk_size;
    for (int64_t index = threadIdx.x; index < count; index += blockDim.x) {
      output[offset + index] += input[offset + index] * scale;
    }
  }
};

struct ScaleByTensorScalar {
  using opmath_t = float;

  __device__ void operator()(
      int64_t chunk_size,
      const at::native::TensorListScalarListMetadata<float, 1>& metadata) const {
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    const int chunk_index = metadata.block_to_chunk[blockIdx.x];
    auto* data = static_cast<float*>(
        const_cast<void*>(metadata.addresses[0][tensor_index]));
    const int64_t offset = chunk_index * chunk_size;
    const int64_t remaining = metadata.numel_for_tensor[tensor_index] - offset;
    const int64_t count = remaining < chunk_size ? remaining : chunk_size;
    const float scale = metadata.scalar_vals[tensor_index];
    for (int64_t index = threadIdx.x; index < count; index += blockDim.x) {
      data[offset + index] *= scale;
    }
  }
};

TM_TEST(ATenNative, MultiTensorApplyChunkingAndEmptyTensors) {
  constexpr int64_t kChunkSize = 65536;
  const auto options = at::device(at::kCUDA).dtype(at::kFloat);
  auto first = at::ones({1024}, options);
  auto empty = at::empty({0}, options);
  auto chunked = at::ones({kChunkSize + 17}, options);
  std::vector<std::vector<at::Tensor>> tensors = {{first, empty, chunked}};
  at::native::multi_tensor_apply<1>(tensors, ScaleTensor{}, 2.0f);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(first.cpu(), at::full({1024}, 2.0f)));
  TM_REQUIRE_EQ(empty.numel(), 0);
  TM_REQUIRE(at::allclose(chunked.cpu(), at::full({kChunkSize + 17}, 2.0f)));
}

TM_TEST(ATenNative, MultiTensorApplyMetadataBatching) {
  constexpr int64_t kTensorCount = 111;
  const auto options = at::device(at::kCUDA).dtype(at::kFloat);
  std::vector<at::Tensor> values;
  values.reserve(kTensorCount);
  for (int64_t index = 0; index < kTensorCount; ++index) {
    values.push_back(at::full({3}, static_cast<float>(index + 1), options));
  }
  std::vector<std::vector<at::Tensor>> tensors = {values};
  at::native::multi_tensor_apply<1>(tensors, ScaleTensor{}, 3.0f);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  for (int64_t index = 0; index < kTensorCount; ++index) {
    TM_REQUIRE(at::allclose(
        values[static_cast<size_t>(index)].cpu(),
        at::full({3}, static_cast<float>((index + 1) * 3))));
  }
}

TM_TEST(ATenNative, MultiTensorApplyDepthTwo) {
  const auto options = at::device(at::kCUDA).dtype(at::kFloat);
  auto output_first = at::ones({257}, options);
  auto output_second = at::full({33}, 4.0f, options);
  auto input_first = at::full({257}, 3.0f, options);
  auto input_second = at::full({33}, -2.0f, options);
  std::vector<std::vector<at::Tensor>> tensors = {
      {output_first, output_second}, {input_first, input_second}};
  at::native::multi_tensor_apply<2>(tensors, AddTensorLists{}, 2.0f);
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(output_first.cpu(), at::full({257}, 7.0f)));
  TM_REQUIRE(at::allclose(output_second.cpu(), at::zeros({33})));
}

TM_TEST(ATenNative, MultiTensorApplyScalarList) {
  const auto options = at::device(at::kCUDA).dtype(at::kFloat);
  auto first = at::ones({8}, options);
  auto second = at::full({5}, 2.0f, options);
  std::vector<std::vector<at::Tensor>> tensors = {{first, second}};
  const std::vector<at::Scalar> scales = {2.5, -3.0};
  at::native::multi_tensor_apply<1, float>(tensors, scales, ScaleByTensorScalar{});
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE(at::allclose(first.cpu(), at::full({8}, 2.5f)));
  TM_REQUIRE(at::allclose(second.cpu(), at::full({5}, -6.0f)));
}

TM_TEST(ATenNative, MultiTensorApplyValidation) {
  const auto input = at::ones({1}, at::device(at::kCUDA).dtype(at::kFloat));
  std::vector<std::vector<at::Tensor>> tensors = {{input}};
  TM_REQUIRE_THROWS(at::native::multi_tensor_apply<2>(tensors, AddTensorLists{}, 1.0f));
}
