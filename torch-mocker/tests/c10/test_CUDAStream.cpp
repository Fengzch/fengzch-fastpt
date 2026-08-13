// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <functional>
#include <sstream>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAStream) {
  const auto device = c10::cuda::current_device();
  const auto original = c10::cuda::getCurrentCUDAStream(device);
  const auto default_stream = c10::cuda::getDefaultCUDAStream(device);
  const auto pooled_stream = c10::cuda::getStreamFromPool(false, device);
  const auto high_priority_stream = c10::cuda::getStreamFromPool(true, device);
  const auto explicit_priority_stream = c10::cuda::getStreamFromPool(-1, device);
  const auto external_stream = c10::cuda::getStreamFromExternal(
      pooled_stream.stream(), device);

  TM_REQUIRE_EQ(default_stream.device_index(), device);
  TM_REQUIRE_EQ(default_stream.device(), c10::Device(c10::DeviceType::CUDA, device));
  TM_REQUIRE_EQ(pooled_stream.device_index(), device);
  TM_REQUIRE(pooled_stream.stream() != nullptr);
  TM_REQUIRE_EQ(external_stream.stream(), pooled_stream.stream());
  TM_REQUIRE_EQ(explicit_priority_stream.device_index(), device);
  TM_REQUIRE(explicit_priority_stream.stream() != nullptr);
  TM_REQUIRE(!pooled_stream.is_capturing());
  TM_REQUIRE(pooled_stream.priority() <= 0);
  TM_REQUIRE(high_priority_stream.priority() <= pooled_stream.priority());

  c10::cuda::setCurrentCUDAStream(pooled_stream);
  TM_REQUIRE_EQ(c10::cuda::getCurrentCUDAStream(device), pooled_stream);

  const auto packed = pooled_stream.pack3();
  const auto unpacked = c10::cuda::CUDAStream::unpack3(
      packed.stream_id,
      packed.device_index,
      packed.device_type);
  TM_REQUIRE_EQ(unpacked, pooled_stream);
  TM_REQUIRE_EQ(
      std::hash<c10::cuda::CUDAStream>{}(unpacked),
      std::hash<c10::cuda::CUDAStream>{}(pooled_stream));
  std::ostringstream description;
  description << pooled_stream;
  TM_REQUIRE(!description.str().empty());
  TM_REQUIRE(pooled_stream.query());
  pooled_stream.synchronize();
  TM_REQUIRE(pooled_stream.query());

  c10::cuda::setCurrentCUDAStream(original);
  TM_REQUIRE_EQ(c10::cuda::getCurrentCUDAStream(device), original);
}
