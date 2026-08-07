#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <utils/mocker_logging.h>
#include <iostream>

namespace at { namespace cuda {

NVRTC* cuda_nvrtc() {
  if (Vlog()) LOG_INFO();
  auto self = new NVRTC();
#define CREATE_ASSIGN(name) self->name = name;
  AT_FORALL_NVRTC(CREATE_ASSIGN)
  return self;
}

}} // at::cuda

