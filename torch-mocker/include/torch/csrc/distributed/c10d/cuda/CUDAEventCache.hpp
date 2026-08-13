// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

#pragma once

#include <array>
#include <deque>
#include <memory>
#include <mutex>

#ifndef USE_ROCM
#include <ATen/cuda/CUDAEvent.h>
#else
#include <ATen/hip/HIPEvent.h>
#endif
#include <c10/macros/Export.h>

namespace c10d {

class TORCH_API CUDAEventCache
    : public std::enable_shared_from_this<CUDAEventCache> {
 public:
  CUDAEventCache();
  std::shared_ptr<at::cuda::CUDAEvent> create(bool timing);
  static std::shared_ptr<CUDAEventCache> get(at::DeviceIndex device);

 private:
  std::mutex cacheMutex_;
  // NOTE: We intentionally store raw pointers so that
  // we do not attempt to destroy the event objects on process exit,
  // because cuda may be gone.
  std::array<std::deque<at::cuda::CUDAEvent*>, 2>
      eventsArray_; // 0 for timing=false, 1 for timing=true
};

} // namespace c10d
