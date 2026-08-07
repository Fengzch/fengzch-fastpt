// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/csrc/cuda/comm.h>

#include <ATen/ATen.h>
#include <ATen/Functions.h>

#include "../../../support/TestHarness.h"

TM_TEST(TorchCUDA, Comm) {
  const auto input = at::randn({4, 8}, at::device(at::kCUDA).dtype(at::kFloat));
  std::vector<at::Tensor> outputs = {at::empty_like(input)};
  const std::vector<c10::optional<at::cuda::CUDAStream>> streams = {
      at::cuda::getCurrentCUDAStream(0)};
  torch::cuda::scatter_out(input, outputs, 0, streams);
  TM_REQUIRE(at::allclose(outputs[0].cpu(), input.cpu()));

  const std::vector<int64_t> devices = {0};
  const std::vector<int64_t> chunk_sizes = {4};
  const auto scattered = torch::cuda::scatter(
      input,
      devices,
      chunk_sizes,
      0,
      c10::nullopt);
  TM_REQUIRE_EQ(scattered.size(), static_cast<size_t>(1));
  TM_REQUIRE(at::allclose(scattered[0].cpu(), input.cpu()));
}
