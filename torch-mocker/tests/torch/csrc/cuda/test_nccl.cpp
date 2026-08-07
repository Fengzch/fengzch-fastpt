// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>

#include <array>
#include <string>

#include "../../../support/TestHarness.h"

namespace {

void require_nccl(ncclResult_t result, const char* expression, const char* file, int line) {
  if (result != ncclSuccess) {
    ::torch_mocker::test::fail(expression, file, line, ncclGetErrorString(result));
  }
}

} // namespace

#define TM_NCCL_REQUIRE(expression) require_nccl((expression), #expression, __FILE__, __LINE__)

TM_TEST(TorchCUDA, NCCL) {
  if (c10::cuda::device_count() < 2) {
    TM_SKIP("NCCL collective test requires at least two CUDA devices");
  }

  const std::array<int, 2> devices = {0, 1};
  std::array<ncclComm_t, 2> communicators{};
  TM_NCCL_REQUIRE(ncclCommInitAll(communicators.data(), 2, devices.data()));

  std::array<float*, 2> buffers{};
  std::array<cudaStream_t, 2> streams{};
  for (int rank = 0; rank < 2; ++rank) {
    TM_CUDA_REQUIRE(cudaSetDevice(devices[rank]));
    TM_CUDA_REQUIRE(cudaMalloc(&buffers[rank], sizeof(float)));
    const float value = static_cast<float>(rank + 1);
    TM_CUDA_REQUIRE(cudaMemcpy(
        buffers[rank], &value, sizeof(value), cudaMemcpyHostToDevice));
    TM_CUDA_REQUIRE(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
  }

  TM_NCCL_REQUIRE(ncclGroupStart());
  for (int rank = 0; rank < 2; ++rank) {
    TM_NCCL_REQUIRE(ncclAllReduce(
        buffers[rank], buffers[rank], 1, ncclFloat, ncclSum, communicators[rank], streams[rank]));
  }
  TM_NCCL_REQUIRE(ncclGroupEnd());

  for (int rank = 0; rank < 2; ++rank) {
    TM_CUDA_REQUIRE(cudaSetDevice(devices[rank]));
    TM_CUDA_REQUIRE(cudaStreamSynchronize(streams[rank]));
    float value = 0.0f;
    TM_CUDA_REQUIRE(cudaMemcpy(
        &value, buffers[rank], sizeof(value), cudaMemcpyDeviceToHost));
    TM_REQUIRE_NEAR(value, 3.0f, 1e-6f);
  }

  constexpr float kPointToPointValue = 17.0f;
  TM_CUDA_REQUIRE(cudaSetDevice(devices[0]));
  TM_CUDA_REQUIRE(cudaMemcpy(
      buffers[0], &kPointToPointValue, sizeof(kPointToPointValue), cudaMemcpyHostToDevice));
  TM_NCCL_REQUIRE(ncclGroupStart());
  TM_NCCL_REQUIRE(ncclSend(
      buffers[0], 1, ncclFloat, 1, communicators[0], streams[0]));
  TM_NCCL_REQUIRE(ncclRecv(
      buffers[1], 1, ncclFloat, 0, communicators[1], streams[1]));
  TM_NCCL_REQUIRE(ncclGroupEnd());
  TM_CUDA_REQUIRE(cudaSetDevice(devices[1]));
  TM_CUDA_REQUIRE(cudaStreamSynchronize(streams[1]));
  float received = 0.0f;
  TM_CUDA_REQUIRE(cudaMemcpy(
      &received, buffers[1], sizeof(received), cudaMemcpyDeviceToHost));
  TM_REQUIRE_NEAR(received, kPointToPointValue, 1e-6f);

  for (int rank = 0; rank < 2; ++rank) {
    TM_CUDA_REQUIRE(cudaSetDevice(devices[rank]));
    TM_CUDA_REQUIRE(cudaFree(buffers[rank]));
    TM_CUDA_REQUIRE(cudaStreamDestroy(streams[rank]));
    TM_NCCL_REQUIRE(ncclCommDestroy(communicators[rank]));
  }
}
