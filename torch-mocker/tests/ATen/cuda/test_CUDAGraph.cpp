// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Functions.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDAGraph) {
  auto captured = at::zeros({1}, at::device(at::kCUDA));
  const auto increment = at::ones({1}, at::device(at::kCUDA));
  const auto original = at::cuda::getCurrentCUDAStream();
  const auto capture_stream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(capture_stream);
  auto generator = at::cuda::detail::createCUDAGenerator();
  auto* generator_impl = static_cast<at::CUDAGeneratorImpl*>(
      generator.unsafeGetGeneratorImpl());
  TM_REQUIRE(generator_impl != nullptr);
  generator_impl->set_current_seed(1234);
  generator_impl->set_philox_offset_per_thread(0);
  at::cuda::CUDAGraph graph;
  graph.register_generator_state(generator);
  graph.capture_begin();
  (void)generator_impl->philox_cuda_state(4);
  captured.add_(increment);
  graph.capture_end();
  TM_REQUIRE_EQ(generator_impl->philox_offset_per_thread(), 0ULL);
  graph.replay();
  TM_CUDA_REQUIRE(cudaDeviceSynchronize());
  TM_REQUIRE_EQ(captured.cpu().item<int>(), 1);
  TM_REQUIRE_EQ(generator_impl->philox_offset_per_thread(), 4ULL);
  at::cuda::setCurrentCUDAStream(original);
}
