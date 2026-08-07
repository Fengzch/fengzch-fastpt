// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>

#include "../../../include/csrc_test_autograd.h"
#include "../../../support/TestHarness.h"

TM_TEST(Csrc, AutogradScatter) {
  if (c10::cuda::device_count() < 2) {
    TM_SKIP("autograd Scatter requires two CUDA devices");
  }
  at::manual_seed(0);
  const auto input = at::randn(
      {4, 4}, at::device(c10::Device(c10::DeviceType::CUDA, 0)));
  const std::vector<at::Device> devices = {
      at::Device(at::kCUDA, 0), at::Device(at::kCUDA, 1)};
  const std::vector<c10::optional<at::cuda::CUDAStream>> streams = {
      at::cuda::getCurrentCUDAStream(0), at::cuda::getCurrentCUDAStream(1)};
  torch::autograd::Scatter scatter(devices, c10::nullopt, 0, streams);
  const auto outputs = scatter.apply({input});
  TM_REQUIRE_EQ(outputs.size(), static_cast<size_t>(2));
  TM_REQUIRE(at::allclose(outputs[0].cpu(), input.slice(0, 0, 2).cpu()));
  TM_REQUIRE(at::allclose(outputs[1].cpu(), input.slice(0, 2, 4).cpu()));
}
