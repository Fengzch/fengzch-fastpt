// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include <ATen/cudnn/Types.h>


#include <c10/util/Exception.h>
#include <utils/mocker_logging.h>

namespace at::native {

/**
 * cudnnDataType_t类型需要cudamocker支持
 * miopen目前只能识别以下三种数据类型
 * 此处调用miopen接口仍需要进行一次数据转换，因此，参考原生cudnn中的方式直接实现
 */
cudnnDataType_t getCudnnDataTypeFromScalarType(const at::ScalarType dtype) {
  if (Vlog()) LOG_INFO();
  if (dtype == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (dtype == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (dtype == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (dtype == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  }// else if (dtype == at::kInt) {
  //   return CUDNN_DATA_INT32;
  // } else if (dtype == at::kByte) {
  //   return CUDNN_DATA_UINT8;
  // } else if (dtype == at::kChar) {
  //   return CUDNN_DATA_INT8;
  // }
  TORCH_CHECK(false,
    "getCudnnDataTypeFromScalarType() not supported for ",
    toString(dtype)
  );
}

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
  if (Vlog()) LOG_INFO();
  return getCudnnDataTypeFromScalarType(tensor.scalar_type());
}

int64_t cudnn_version() {
  if (Vlog()) LOG_INFO();
  return CUDNN_VERSION;
}

} // namespace at::native
