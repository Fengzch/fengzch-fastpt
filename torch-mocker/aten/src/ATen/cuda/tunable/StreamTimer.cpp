// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/tunable/StreamTimer.h>
#include <c10/cuda/CUDAStream.h>

#include <utils/mocker_logging.h>
namespace at::cuda::tunable {

StreamTimer::StreamTimer() {
  if (Vlog()) LOG_INFO();
  AT_CUDA_CHECK(cudaEventCreate(&start_));
  AT_CUDA_CHECK(cudaEventCreate(&end_));
}

StreamTimer::~StreamTimer() = default;

void StreamTimer::Start() {
  if (Vlog()) LOG_INFO();
  AT_CUDA_CHECK(cudaEventSynchronize(start_));
  AT_CUDA_CHECK(cudaEventRecord(start_, at::cuda::getCurrentCUDAStream()));
}

void StreamTimer::End() {
  if (Vlog()) LOG_INFO();
  AT_CUDA_CHECK(cudaEventRecord(end_, at::cuda::getCurrentCUDAStream()));
  AT_CUDA_CHECK(cudaEventSynchronize(end_));
}

float StreamTimer::Duration() {
  if (Vlog()) LOG_INFO();
  auto time = std::numeric_limits<float>::quiet_NaN();
  // time is in ms with a resolution of 1 us
  AT_CUDA_CHECK(cudaEventElapsedTime(&time, start_, end_));
  return time;
}

StreamTimerNoSync::StreamTimerNoSync() {
  if (Vlog()) LOG_INFO();
  AT_CUDA_CHECK(cudaEventCreate(&start_));
  AT_CUDA_CHECK(cudaEventCreate(&end_));
}

StreamTimerNoSync::~StreamTimerNoSync() = default;

void StreamTimerNoSync::Start() {
  if (Vlog()) LOG_INFO();
  AT_CUDA_CHECK(cudaEventRecord(start_, at::cuda::getCurrentCUDAStream()));
}

void StreamTimerNoSync::End() {
  if (Vlog()) LOG_INFO();
  AT_CUDA_CHECK(cudaEventRecord(end_, at::cuda::getCurrentCUDAStream()));
}

float StreamTimerNoSync::Duration() {
  if (Vlog()) LOG_INFO();
  auto time = std::numeric_limits<float>::quiet_NaN();
  AT_CUDA_CHECK(cudaEventSynchronize(end_));
  // time is in ms with a resolution of 1 us
  AT_CUDA_CHECK(cudaEventElapsedTime(&time, start_, end_));
  return time;
}

} // namespace at::cuda::tunable
