// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <c10/util/irange.h>

#include "../support/TestHarness.h"

namespace torch::jit {
namespace {

using namespace torch::jit::tensorexpr;

template <typename scalar_t>
void run_vector_add_codegen(
    int64_t outer_extent,
    int64_t block_extent,
    int64_t thread_extent) {
  const auto dtype = ToDtype<scalar_t>();
  BufHandle input_a("input_a", {outer_extent, block_extent, thread_extent}, dtype);
  BufHandle input_b("input_b", {outer_extent, block_extent, thread_extent}, dtype);
  Tensor output = Compute(
      "output",
      {outer_extent, block_extent, thread_extent},
      [&](const VarHandle& outer, const VarHandle& block, const VarHandle& thread) {
        return input_a.load(outer, block, thread) + input_b.load(outer, block, thread);
      });

  LoopNest loop_nest({output});
  auto loops = loop_nest.getLoopStmtsFor(output);
  loops[1]->set_gpu_block_index(0);
  loops[2]->set_gpu_thread_index(0);
  loop_nest.prepareForCodegen();
  CudaCodeGen codegen(loop_nest.root_stmt(), output, input_a, input_b);

  const auto generated = codegen.getCodeText();
  TM_REQUIRE(generated.find("blockIdx.x") != std::string::npos);
  TM_REQUIRE(generated.find("threadIdx.x") != std::string::npos);
  TM_REQUIRE(!codegen.gpu_block_extents().empty());
  TM_REQUIRE(!codegen.gpu_thread_extents().empty());

  const auto numel = outer_extent * block_extent * thread_extent;
  std::vector<scalar_t> host_a(static_cast<size_t>(numel));
  std::vector<scalar_t> host_b(static_cast<size_t>(numel));
  std::vector<scalar_t> host_output(static_cast<size_t>(numel));
  scalar_t* device_a = nullptr;
  scalar_t* device_b = nullptr;
  scalar_t* device_output = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&device_a, static_cast<size_t>(numel) * sizeof(scalar_t)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_b, static_cast<size_t>(numel) * sizeof(scalar_t)));
  TM_CUDA_REQUIRE(cudaMalloc(&device_output, static_cast<size_t>(numel) * sizeof(scalar_t)));

  for (int pass = 0; pass < 2; ++pass) {
    for (const auto index : c10::irange(numel)) {
      host_a[static_cast<size_t>(index)] = static_cast<scalar_t>(index + pass * 3);
      host_b[static_cast<size_t>(index)] = static_cast<scalar_t>(index * 2 - pass * 5);
    }
    TM_CUDA_REQUIRE(cudaMemcpy(
        device_a,
        host_a.data(),
        static_cast<size_t>(numel) * sizeof(scalar_t),
        cudaMemcpyHostToDevice));
    TM_CUDA_REQUIRE(cudaMemcpy(
        device_b,
        host_b.data(),
        static_cast<size_t>(numel) * sizeof(scalar_t),
        cudaMemcpyHostToDevice));

    codegen(device_output, device_a, device_b);
    TM_CUDA_REQUIRE(cudaDeviceSynchronize());
    TM_CUDA_REQUIRE(cudaMemcpy(
        host_output.data(),
        device_output,
        static_cast<size_t>(numel) * sizeof(scalar_t),
        cudaMemcpyDeviceToHost));
    for (const auto index : c10::irange(numel)) {
      const auto expected = host_a[static_cast<size_t>(index)] + host_b[static_cast<size_t>(index)];
      TM_REQUIRE_NEAR(
          static_cast<double>(host_output[static_cast<size_t>(index)]),
          static_cast<double>(expected),
          1e-10);
    }
  }

  TM_CUDA_REQUIRE(cudaFree(device_a));
  TM_CUDA_REQUIRE(cudaFree(device_b));
  TM_CUDA_REQUIRE(cudaFree(device_output));
}

} // namespace

TM_TEST(Csrc, CudaCodeGenFloat) {
  run_vector_add_codegen<float>(3, 16, 128);
}

TM_TEST(Csrc, CudaCodeGenMaskedThreadExtent) {
  // 257 is intentionally not a conventional power-of-two thread extent.
  run_vector_add_codegen<float>(1, 3, 257);
}

TM_TEST(Csrc, CudaCodeGenDoubleAndReuse) {
  run_vector_add_codegen<double>(2, 7, 65);
}

} // namespace torch::jit
