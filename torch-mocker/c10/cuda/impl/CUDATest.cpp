// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

// Just a little test file to make sure that the CUDA library works
#include "hip/hip_runtime.h"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/impl/CUDATest.h>
#include <utils/mocker_logging.h>
#include <cuda_runtime.h>

namespace c10::cuda::impl {

static bool has_cuda_gpu() {
  if (Vlog()) LOG_INFO();
  int count = 0;
  C10_CUDA_IGNORE_ERROR(cudaGetDeviceCount(&count));

  return count != 0;
}

int c10_cuda_test() {
  if (Vlog()) LOG_INFO();
  int r = 0;
  if (has_cuda_gpu()) {
    C10_CUDA_CHECK(cudaGetDevice(&r));
  }
  return r;
}

} // namespace c10::cuda::impl
