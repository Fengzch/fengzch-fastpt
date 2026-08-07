#define USE_ROCM
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <utils/mocker_logging.h>
#include <iostream>
#include <dlfcn.h>

namespace at::cuda {

NVRTC* hip_nvrtc() {
  if (Vlog()) LOG_INFO();
  auto self = new NVRTC();
#define CREATE_ASSIGN(name) self->name = name;
  AT_FORALL_NVRTC(CREATE_ASSIGN)
  return self;
}
#undef USE_ROCM
NVRTC* load_nvrtc() {
    if (Vlog()) LOG_INFO();
    return hip_nvrtc();
}

} // at::cuda
