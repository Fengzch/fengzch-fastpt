// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/csrc/CudaIPCTypes.h>
#include <cuda_runtime.h>

#include <array>
#include <memory>
#include <string>

#include "../support/TestHarness.h"

TM_TEST(Csrc, CudaIPCTypes) {
  const c10::Device device(c10::DeviceType::CUDA, 0);
  auto received_storage = std::make_shared<int>(42);
  torch::CudaIPCReceivedData received_data(received_storage);
  TM_REQUIRE(received_data.shared_ptr_ != nullptr);
  TM_REQUIRE_EQ(
      *static_cast<int*>(received_data.shared_ptr_.get()), *received_storage);
  received_storage.reset();
  TM_REQUIRE_EQ(*static_cast<int*>(received_data.shared_ptr_.get()), 42);

  float* allocation = nullptr;
  TM_CUDA_REQUIRE(cudaMalloc(&allocation, 10 * sizeof(float)));
  uint64_t counter = 1;
  const std::string handle = "torch-mocker-ipc-test";
  torch::CudaIPCSentData sent_data(handle, 17, &counter, device);
  sent_data.set_original_ptr(at::DataPtr(allocation, nullptr, nullptr, device));
  TM_REQUIRE_EQ(sent_data.handle(), handle);
  TM_REQUIRE_EQ(sent_data.offset(), 17U);
  TM_REQUIRE_EQ(sent_data.counter_value(), 1);
  TM_REQUIRE_EQ(sent_data.device_, device);
  TM_REQUIRE_EQ(sent_data.original_ptr_.get(), allocation);
#ifdef USE_ROCM
  TM_REQUIRE(!sent_data.event_sync_required_);
#endif
  TM_CUDA_REQUIRE(cudaFree(allocation));
}
