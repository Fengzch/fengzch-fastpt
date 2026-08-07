// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include "../../support/TestHarness.h"

TM_TEST(ATenCUDA, CUDAGeneratorImpl) {
  const auto& default_generator = at::cuda::detail::getDefaultCUDAGenerator();
  TM_REQUIRE_EQ(default_generator.device().type(), c10::DeviceType::CUDA);

  auto generator = at::cuda::detail::createCUDAGenerator();
  TM_REQUIRE_EQ(generator.device().type(), c10::DeviceType::CUDA);
  auto* implementation = static_cast<at::CUDAGeneratorImpl*>(
      generator.unsafeGetGeneratorImpl());
  TM_REQUIRE(implementation != nullptr);
  implementation->set_current_seed(12345);
  TM_REQUIRE_EQ(implementation->current_seed(), 12345ULL);
  implementation->set_offset(16);
  TM_REQUIRE_EQ(implementation->get_offset(), 16ULL);
  implementation->set_philox_offset_per_thread(8);
  TM_REQUIRE_EQ(implementation->philox_offset_per_thread(), 8ULL);
  TM_REQUIRE(implementation->reset_rnn_state());
  TM_REQUIRE(!implementation->reset_rnn_state());
  const auto inputs = implementation->philox_engine_inputs(4);
  TM_REQUIRE_EQ(inputs.first, 12345ULL);
  auto clone = implementation->clone();
  TM_REQUIRE(clone != nullptr);
  TM_REQUIRE_EQ(clone->current_seed(), implementation->current_seed());
  TM_REQUIRE_EQ(clone->get_offset(), implementation->get_offset());
  implementation->set_current_seed(67890);
  TM_REQUIRE_EQ(clone->current_seed(), 12345ULL);

  auto state = c10::make_intrusive<at::CUDAGeneratorState>(123, 8);
  state->increase(5);
  TM_REQUIRE_EQ(state->philox_offset_per_thread_, 16ULL);
  auto state_clone = state->clone();
  TM_REQUIRE_EQ(state_clone->seed_, 123ULL);
  TM_REQUIRE_EQ(state_clone->philox_offset_per_thread_, 16ULL);

  constexpr at::CaptureId_t kCaptureId = 17;
  TM_REQUIRE(state->get_capture_state(kCaptureId) == nullptr);
  state->init_capture_state(kCaptureId);
  TM_REQUIRE(state->get_capture_state(kCaptureId) != nullptr);
  state->init_capture_state(kCaptureId);
  TM_REQUIRE(state->get_capture_state(kCaptureId) != nullptr);
  TM_REQUIRE_EQ(state->capture_epilogue(kCaptureId), 0ULL);
  state->replay_prologue(kCaptureId, 0);
  state->remove_capture_state(kCaptureId);
  TM_REQUIRE(state->get_capture_state(kCaptureId) == nullptr);
}
