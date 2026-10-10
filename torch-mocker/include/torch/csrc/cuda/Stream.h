// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

#ifndef THCP_STREAM_INC
#define THCP_STREAM_INC

#ifndef USE_ROCM
#include <c10/cuda/CUDAStream.h>
#else
#include <c10/hip/HIPStream.h>
#endif
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THCPStream : THPStream {
  at::cuda::CUDAStream cuda_stream;
};
extern PyObject* THCPStreamClass;

void THCPStream_init(PyObject* module);

inline bool THCPStream_Check(PyObject* obj) {
  return THCPStreamClass && PyObject_IsInstance(obj, THCPStreamClass);
}

#endif // THCP_STREAM_INC
