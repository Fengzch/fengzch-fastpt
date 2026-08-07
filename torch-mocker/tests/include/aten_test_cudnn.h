// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ATen/cudnn/Descriptors.h>
#include <ATen/ATen.h>
#include <iostream>
/**********test_cudnn_types ***************/
void test_cudnn_version();
void test_getCudnnDataType(const at::ScalarType dtype);
void test_getCudnnDataTypeFromScalarType(const at::Tensor& tensor);
void test_cudnn_types();
/******test_cudnn_discription ************/
void test_cudnnTypeToString();
void test_cudnn_dataSize();
void test_cudnn_fixSizeOneDimStride();
void test_cudnn_TensorDescriptor();
void test_cudnn_FilterDescriptor();
void test_cudnn_ConvolutionDescriptor();
void test_cudnn_SpatialTransformerDescriptor();
void test_cudnn_DropoutDescriptor();
void test_cudnn_RNNDescriptor();
void test_cudnn_CTCLossDescriptor();
void test_cudnn_ActivationDescriptor();
void test_cudnn_Constant();
void test_cudnn_discription();

