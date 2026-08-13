// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

#pragma once

#ifndef USE_ROCM
#include <c10/cuda/CUDAStream.h>
#else
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#endif
#include <torch/csrc/utils/python_numbers.h>

#include <vector>

#ifndef USE_ROCM
std::vector<std::optional<at::cuda::CUDAStream>>
#else
std::vector<std::optional<at::hip::HIPStreamMasqueradingAsCUDA>>
#endif
THPUtils_PySequence_to_CUDAStreamList(PyObject* obj);
