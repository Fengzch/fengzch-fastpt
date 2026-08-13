// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>

namespace c10::cuda::impl {

C10_REGISTER_GUARD_IMPL(CUDA, CUDAGuardImpl)

} // namespace c10::cuda::impl
