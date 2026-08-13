// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/CallOnce.h>
#include <utils/mocker_logging.h>
// #include <ATen/cuda/CUDAConfig.h>
#include <deque>
#include <vector>
#ifdef USE_ROCM
#include <rocm-core/rocm_version.h>
#endif

namespace at::cuda {

namespace {

DeviceIndex num_gpus = -1;
std::deque<c10::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initCUDAContextVectors() {
  if (Vlog()) LOG_INFO();
  static bool init_flag [[maybe_unused]] = []() {
    num_gpus = c10::cuda::device_count();
    device_flags.resize(num_gpus);
    device_properties.resize(num_gpus);
    return true;
  }();
}

void initDeviceProperty(DeviceIndex device_index) {
  if (Vlog()) LOG_INFO();
  cudaDeviceProp device_prop{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda(_cpp) on Windows.
// If you need to modify this function, please specify a new function and apply
// the changes according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  if (Vlog()) LOG_INFO();
  return getCurrentDeviceProperties()->warpSize;
}

template <typename T>
typename std::enable_if<true, cudaDeviceProp*>::type getCurrentDeviceProperties(){
    if (Vlog()) LOG_INFO();
    auto device = c10::cuda::current_device();
    return getDeviceProperties(device);
}
template TORCH_CUDA_CPP_API typename std::enable_if<true, cudaDeviceProp*>::type getCurrentDeviceProperties<cudaDeviceProp*>();

template <typename T>
typename std::enable_if<true, cudaDeviceProp*>::type getDeviceProperties(c10::DeviceIndex device){
    if (Vlog()) LOG_INFO();
  initCUDAContextVectors();
  if (device == -1)
    device = c10::cuda::current_device();
  AT_ASSERT(
      device >= 0 && device < num_gpus,
      "device=",
      static_cast<int>(device),
      ", num_gpus=",
      static_cast<int>(num_gpus));
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}
template TORCH_CUDA_CPP_API typename std::enable_if<true, cudaDeviceProp*>::type getDeviceProperties<cudaDeviceProp*>(c10::DeviceIndex device);

bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device) {
  if (Vlog()) LOG_INFO();
  initCUDAContextVectors();
  if (device == -1)
    device = c10::cuda::current_device();
  AT_ASSERT(
      device >= 0 && device < num_gpus,
      "device=",
      static_cast<int>(device),
      ", num_gpus=",
      static_cast<int>(num_gpus));
  AT_ASSERT(
      peer_device >= 0 && peer_device < num_gpus,
      "peer_device=",
      static_cast<int>(peer_device),
      ", num_gpus=",
      static_cast<int>(num_gpus));
  int can_access = 0;
  AT_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getCUDADeviceAllocator() {
  if (Vlog()) LOG_INFO();
  return c10::cuda::CUDACachingAllocator::get();
}

bool getCurrentDeviceEcoInfo(int device){
  if (Vlog()) LOG_INFO();
#if defined(USE_ROCM) && ROCM_VERSION_ECO==1
  int eco_info;
  hipDeviceAttribute_t attr = hipDeviceAttributeEcoInfo;
  hipError_t result = hipGetDevice(&device);
  result = hipDeviceGetAttribute(&eco_info, attr, device);
  return eco_info ==1;
#else
TORCH_WARN("Please compile pytorch with dtk24.04.3 or later, the current version does not support eco info capture.");
  return false;
#endif
}

} // namespace at::cuda
