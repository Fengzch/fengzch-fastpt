// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

#ifndef THCP_EVENT_INC
#define THCP_EVENT_INC

#ifndef USE_ROCM
#include <ATen/cuda/CUDAEvent.h>
#else
#include <ATen/hip/HIPEvent.h>
#endif
#include <torch/csrc/Event.h>
#include <torch/csrc/python_headers.h>

struct THCPEvent : THPEvent {
  at::cuda::CUDAEvent cuda_event;
};
extern PyObject* THCPEventClass;

void THCPEvent_init(PyObject* module);

inline bool THCPEvent_Check(PyObject* obj) {
  return THCPEventClass && PyObject_IsInstance(obj, THCPEventClass);
}

#endif // THCP_EVENT_INC
