// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>

#ifndef USE_ROCM
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#else
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace torch::jit::fuser::cuda {

// query codegen output arch and target
TORCH_CUDA_CU_API void codegenOutputQuery(
#ifndef USE_ROCM
    const cudaDeviceProp* const prop,
#else
    const hipDeviceProp_t_v2* const prop,
#endif
    int& major,
    int& minor,
    bool& compile_to_sass);

// A class holding metadata for an actual CUDA function.
// Note: CUDA functions are per device.
#ifndef USE_ROCM
struct TORCH_CUDA_CU_API FusedKernelCUDA
#else
struct TORCH_HIP_API FusedKernelCUDA
#endif
    : public ::torch::jit::fuser::FusedKernel {
  FusedKernelCUDA(
      at::DeviceIndex device,
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random);

  ~FusedKernelCUDA() override;

  void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const override;

  at::Backend backend() const override {
    return at::Backend::CUDA;
  }

 private:
  static constexpr auto kBlockSize = 128;

  // Note: per device to store device properties and compute launch heuristics
  //  Acquiring these values at launch time would be too slow
  at::DeviceIndex device_;
  int maxBlocks_{};
#ifndef USE_ROCM
  cudaDeviceProp* prop_{};
#else
  hipDeviceProp_t_v2* prop_{};
#endif
  std::vector<char> ptx_;
#ifndef USE_ROCM
  CUmodule module_{};
  CUfunction function_{};
#else
  hipModule_t module_{};
  hipFunction_t function_{};
#endif
};

} // namespace torch::jit::fuser::cuda
