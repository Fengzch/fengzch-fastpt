// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Descriptors.h>

#include <array>
#include <string>

#include "../../support/TestHarness.h"

TM_TEST(ATenCuDNN, Descriptors) {
  TM_REQUIRE_EQ(
      at::native::cudnnTypeToString(CUDNN_DATA_FLOAT),
      std::string("CUDNN_DATA_FLOAT"));
  TM_REQUIRE_EQ(at::native::dataSize(CUDNN_DATA_FLOAT), 4);
  TM_REQUIRE_EQ(at::native::dataSize(CUDNN_DATA_HALF), 2);
  TM_REQUIRE_EQ(at::native::dataSize(CUDNN_DATA_BFLOAT16), 2);
  TM_REQUIRE_EQ(at::native::dataSize(CUDNN_DATA_DOUBLE), 8);

  int sizes[] = {2, 1, 3, 4};
  int strides[] = {12, 0, 4, 1};
  at::native::fixSizeOneDimStride(4, sizes, strides, true);
  TM_REQUIRE_EQ(strides[0], 12);
  TM_REQUIRE_EQ(strides[1], 1);
  TM_REQUIRE_EQ(strides[2], 4);
  TM_REQUIRE_EQ(strides[3], 1);

  int unchanged_sizes[] = {2, 1, 3, 4};
  int unchanged_strides[] = {12, 0, 4, 1};
  at::native::fixSizeOneDimStride(4, unchanged_sizes, unchanged_strides, false);
  TM_REQUIRE_EQ(unchanged_strides[0], 12);
  TM_REQUIRE_EQ(unchanged_strides[1], 12);
  TM_REQUIRE_EQ(unchanged_strides[2], 4);
  TM_REQUIRE_EQ(unchanged_strides[3], 1);

  const auto tensor = at::randn({2, 3, 4, 5}, at::device(at::kCUDA));
  at::native::TensorDescriptor descriptor(tensor, 0);
  TM_REQUIRE(descriptor.desc() != nullptr);
  TM_REQUIRE(descriptor.mut_desc() != nullptr);
  descriptor.set(tensor, at::MemoryFormat::Contiguous, 6);
  TM_REQUIRE(descriptor.desc() != nullptr);
  descriptor.set(
      CUDNN_DATA_FLOAT, tensor.sizes(), tensor.strides(), 0);
  TM_REQUIRE(descriptor.desc() != nullptr);

  at::native::FilterDescriptor filter_descriptor;
  filter_descriptor.set(tensor, at::MemoryFormat::Contiguous, 0);
  TM_REQUIRE(filter_descriptor.desc() != nullptr);

  int convolution_pad[] = {1, 1};
  int convolution_stride[] = {1, 1};
  int convolution_dilation[] = {1, 1};
  at::native::ConvolutionDescriptor convolution_descriptor;
  convolution_descriptor.set(
      CUDNN_DATA_FLOAT,
      2,
      convolution_pad,
      convolution_stride,
      convolution_dilation,
      1,
      false);
  TM_REQUIRE(convolution_descriptor.desc() != nullptr);

  int spatial_sizes[] = {2, 3, 4, 5};
  at::native::SpatialTransformerDescriptor spatial_descriptor;
  // DTK cuDNN currently exposes this legacy descriptor API but reports it as
  // unsupported. Keep the call covered and make that contract explicit.
  TM_REQUIRE_THROWS(
      spatial_descriptor.set(CUDNN_DATA_FLOAT, 4, spatial_sizes));

  const auto handle = at::native::getCudnnHandle();
  TM_REQUIRE(handle != nullptr);
  at::native::DropoutDescriptor no_dropout_descriptor;
  no_dropout_descriptor.set_no_dropout(handle);
  TM_REQUIRE(no_dropout_descriptor.desc() != nullptr);

  at::native::DropoutDescriptor initialized_dropout_descriptor;
  initialized_dropout_descriptor.initialize_rng(
      handle, 0.25f, 1234, at::device(at::kCUDA).dtype(at::kByte));
  TM_REQUIRE(initialized_dropout_descriptor.desc() != nullptr);
  TM_REQUIRE(initialized_dropout_descriptor.state.defined());
  TM_REQUIRE(initialized_dropout_descriptor.state.numel() > 0);
  at::native::DropoutDescriptor restored_dropout_descriptor;
  restored_dropout_descriptor.set(
      handle, 0.25f, initialized_dropout_descriptor.state);
  TM_REQUIRE(restored_dropout_descriptor.desc() != nullptr);

  at::native::RNNDescriptor rnn_descriptor;
  at::native::DropoutDescriptor rnn_dropout_descriptor;
  rnn_dropout_descriptor.set_no_dropout(handle);
#ifdef USE_CUDNN_RNN_V8_API
  rnn_descriptor.set(
      handle,
      3,
      false,
      4,
      0,
      1,
      std::move(rnn_dropout_descriptor),
      CUDNN_LINEAR_INPUT,
      CUDNN_UNIDIRECTIONAL,
      CUDNN_LSTM,
      CUDNN_DATA_FLOAT,
      CUDNN_DATA_FLOAT,
      CUDNN_RNN_ALGO_STANDARD,
      false);
#else
  rnn_descriptor.set(
      handle,
      4,
      0,
      1,
      std::move(rnn_dropout_descriptor),
      CUDNN_LINEAR_INPUT,
      CUDNN_UNIDIRECTIONAL,
      CUDNN_LSTM,
      CUDNN_DATA_FLOAT,
      CUDNN_DATA_FLOAT,
      CUDNN_RNN_ALGO_STANDARD,
      false);
#endif
  TM_REQUIRE(rnn_descriptor.desc() != nullptr);

  at::native::CTCLossDescriptor ctc_descriptor;
  ctc_descriptor.set(CUDNN_DATA_FLOAT);
  ctc_descriptor.setEx(
      CUDNN_DATA_FLOAT,
      CUDNN_LOSS_NORMALIZATION_NONE,
      CUDNN_NOT_PROPAGATE_NAN);
  ctc_descriptor.set_v8_v9(
      CUDNN_DATA_FLOAT,
      CUDNN_LOSS_NORMALIZATION_NONE,
      CUDNN_NOT_PROPAGATE_NAN,
      4);
  TM_REQUIRE(ctc_descriptor.desc() != nullptr);

  at::native::ActivationDescriptor activation_descriptor;
  activation_descriptor.set(CUDNN_ACTIVATION_RELU);
  TM_REQUIRE(activation_descriptor.desc() != nullptr);

  at::native::Constant float_constant(CUDNN_DATA_FLOAT, 12.5);
  at::native::Constant double_constant(CUDNN_DATA_DOUBLE, 12.5);
  TM_REQUIRE_NEAR(float_constant.f, 12.5f, 1e-6f);
  TM_REQUIRE_NEAR(double_constant.d, 12.5, 1e-12);
}
