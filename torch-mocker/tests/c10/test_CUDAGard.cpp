// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAGuard) {
  const auto device_index = c10::cuda::current_device();
  const auto device = c10::Device(c10::DeviceType::CUDA, device_index);
  const auto original_stream = c10::cuda::getCurrentCUDAStream(device_index);
  const auto pooled_stream = c10::cuda::getStreamFromPool(false, device_index);

  {
    c10::cuda::CUDAGuard device_guard(device);
    TM_REQUIRE_EQ(device_guard.current_device(), device);
    TM_REQUIRE_EQ(device_guard.original_device(), device);

    c10::cuda::CUDAStreamGuard stream_guard(pooled_stream);
    TM_REQUIRE_EQ(stream_guard.current_stream(), pooled_stream);
    TM_REQUIRE_EQ(stream_guard.current_device(), device);
    TM_REQUIRE_EQ(stream_guard.original_stream(), original_stream);
    stream_guard.reset_stream(original_stream);
    TM_REQUIRE_EQ(stream_guard.current_stream(), original_stream);
  }

  {
    c10::cuda::OptionalCUDAGuard optional_device_guard;
    TM_REQUIRE(!optional_device_guard.current_device().has_value());
    optional_device_guard.set_device(device);
    TM_REQUIRE_EQ(optional_device_guard.current_device().value(), device);
    optional_device_guard.reset();
    TM_REQUIRE(!optional_device_guard.current_device().has_value());

    c10::cuda::OptionalCUDAStreamGuard optional_stream_guard;
    TM_REQUIRE(!optional_stream_guard.current_stream().has_value());
    optional_stream_guard.reset_stream(pooled_stream);
    TM_REQUIRE_EQ(optional_stream_guard.current_stream().value(), pooled_stream);
    optional_stream_guard.reset();
    TM_REQUIRE(!optional_stream_guard.current_stream().has_value());
  }

  TM_REQUIRE_EQ(c10::cuda::current_device(), device_index);
  TM_REQUIRE_EQ(c10::cuda::getCurrentCUDAStream(device_index), original_stream);
}
