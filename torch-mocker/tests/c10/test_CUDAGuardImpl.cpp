// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>

#include "../support/TestHarness.h"

TM_TEST(C10, CUDAGuardImpl) {
  c10::cuda::impl::CUDAGuardImpl guard;
  const auto device_index = c10::cuda::current_device();
  const auto device = c10::Device(c10::DeviceType::CUDA, device_index);

  TM_REQUIRE_EQ(guard.type(), c10::DeviceType::CUDA);
  TM_REQUIRE_EQ(guard.getDevice(), device);
  TM_REQUIRE_EQ(guard.uncheckedGetDevice().value(), device);
  TM_REQUIRE(guard.deviceCount() > 0);
  TM_REQUIRE_EQ(guard.getDefaultStream(device).device_index(), device_index);
  TM_REQUIRE_EQ(guard.getStreamFromGlobalPool(device, false).device_index(), device_index);

  const auto original_stream = guard.getStream(device);
  const auto new_stream = guard.getNewStream(device);
  TM_REQUIRE_EQ(new_stream.device_index(), device_index);
  TM_REQUIRE(guard.getStreamNativeHandle(new_stream) != nullptr);
  TM_REQUIRE(!guard.isStreamCapturing(new_stream));
  TM_REQUIRE(guard.queryStream(new_stream));
  const auto exchanged_stream = guard.exchangeStream(new_stream);
  TM_REQUIRE_EQ(exchanged_stream, original_stream);
  TM_REQUIRE_EQ(guard.getStream(device), new_stream);
  TM_REQUIRE_EQ(guard.exchangeStream(original_stream), new_stream);
  TM_REQUIRE_EQ(guard.getStream(device), original_stream);
  guard.synchronizeStream(new_stream);

  cudaEvent_t event = nullptr;
  TM_REQUIRE(guard.queryEvent(event));
  guard.createEvent(&event, c10::EventFlag::PYTORCH_DEFAULT);
  TM_REQUIRE(event != nullptr);
  TM_REQUIRE(guard.queryEvent(event));
  guard.destroyEvent(event, device_index);

  event = nullptr;
  guard.record(
      reinterpret_cast<void**>(&event),
      new_stream,
      device_index,
      c10::EventFlag::BACKEND_DEFAULT);
  TM_REQUIRE(event != nullptr);
  guard.block(event, original_stream);
  guard.synchronizeEvent(event);
  TM_REQUIRE(guard.queryEvent(event));
  TM_REQUIRE(guard.elapsedTime(event, event, device_index) >= 0.0);
  guard.destroyEvent(event, device_index);
  guard.synchronizeDevice(device_index);
}

TM_TEST(C10, CUDAGuardImplDeviceAndStreamContracts) {
  c10::cuda::impl::CUDAGuardImpl guard;
  const auto device_index = c10::cuda::current_device();
  const auto device = c10::Device(c10::DeviceType::CUDA, device_index);
  const auto original_device = guard.getDevice();
  const auto original_stream = guard.getStream(device);

  TM_REQUIRE_THROWS(c10::cuda::impl::CUDAGuardImpl(c10::DeviceType::CPU));
  TM_REQUIRE_THROWS(guard.setDevice(c10::Device(c10::DeviceType::CPU, 0)));
  TM_REQUIRE_THROWS(guard.exchangeDevice(c10::Device(c10::DeviceType::CPU, 0)));

  const auto exchanged_device = guard.exchangeDevice(device);
  TM_REQUIRE_EQ(exchanged_device, original_device);
  TM_REQUIRE_EQ(guard.getDevice(), device);
  guard.uncheckedSetDevice(device);
  TM_REQUIRE_EQ(guard.uncheckedGetDevice().value(), device);

  const auto default_stream = guard.getDefaultStream(device);
  const auto low_priority_stream = guard.getNewStream(device, 0);
  const auto high_priority_stream = guard.getNewStream(device, -1);
  const auto global_low_stream = guard.getStreamFromGlobalPool(device, false);
  const auto global_high_stream = guard.getStreamFromGlobalPool(device, true);
  TM_REQUIRE_EQ(default_stream.device(), device);
  TM_REQUIRE_EQ(low_priority_stream.device(), device);
  TM_REQUIRE_EQ(high_priority_stream.device(), device);
  TM_REQUIRE_EQ(global_low_stream.device(), device);
  TM_REQUIRE_EQ(global_high_stream.device(), device);
  TM_REQUIRE(guard.getStreamNativeHandle(default_stream) != nullptr);
  TM_REQUIRE(guard.getStreamNativeHandle(high_priority_stream) != nullptr);
  TM_REQUIRE(
      c10::cuda::CUDAStream(high_priority_stream).priority() <=
      c10::cuda::CUDAStream(low_priority_stream).priority());
  TM_REQUIRE(
      c10::cuda::CUDAStream(global_high_stream).priority() <=
      c10::cuda::CUDAStream(global_low_stream).priority());

  const auto previous_stream = guard.exchangeStream(low_priority_stream);
  TM_REQUIRE_EQ(previous_stream, original_stream);
  TM_REQUIRE_EQ(guard.getStream(device), low_priority_stream);
  TM_REQUIRE_EQ(guard.exchangeStream(original_stream), low_priority_stream);
  TM_REQUIRE_EQ(guard.getStream(device), original_stream);
}

TM_TEST(C10, CUDAGuardImplEventAndAllocatorContracts) {
  c10::cuda::impl::CUDAGuardImpl guard;
  const auto device_index = c10::cuda::current_device();
  const auto device = c10::Device(c10::DeviceType::CUDA, device_index);
  const auto stream = guard.getNewStream(device);

  TM_REQUIRE(guard.queryEvent(nullptr));
  guard.block(nullptr, stream);
  guard.synchronizeEvent(nullptr);
  guard.destroyEvent(nullptr, device_index);
  cudaEvent_t invalid_event = nullptr;
  TM_REQUIRE_THROWS(
      guard.createEvent(&invalid_event, static_cast<c10::EventFlag>(-1)));
  TM_REQUIRE_THROWS(guard.elapsedTime(nullptr, nullptr, device_index));

  void* first_event = nullptr;
  void* second_event = nullptr;
  guard.record(
      &first_event,
      stream,
      -1,
      c10::EventFlag::BACKEND_DEFAULT);
  TM_REQUIRE(first_event != nullptr);
  const auto first_event_address = first_event;
  guard.record(
      &first_event,
      stream,
      device_index,
      c10::EventFlag::BACKEND_DEFAULT);
  TM_REQUIRE_EQ(first_event, first_event_address);
  guard.record(
      &second_event,
      stream,
      device_index,
      c10::EventFlag::BACKEND_DEFAULT);
  TM_REQUIRE(second_event != nullptr);
  guard.block(first_event, stream);
  guard.synchronizeEvent(second_event);
  TM_REQUIRE(guard.queryEvent(first_event));
  TM_REQUIRE(guard.queryEvent(second_event));
  TM_REQUIRE(guard.elapsedTime(first_event, second_event, device_index) >= 0.0);
  guard.destroyEvent(first_event, device_index);
  guard.destroyEvent(second_event, device_index);

  auto allocation = c10::cuda::CUDACachingAllocator::get()->allocate(4096);
  TM_REQUIRE(allocation.get() != nullptr);
  guard.recordDataPtrOnStream(allocation, stream);
  guard.synchronizeStream(stream);
}
