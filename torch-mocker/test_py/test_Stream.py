# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import sys
sys.path.append('../build/test_py/')

import test_THCP
# 验证 `_CudaStreamBase` 类是否成功注册
if hasattr(test_THCP, "_CudaStreamBase"):
    print("Successfully initialized _CudaStreamBase")
    try:
        cuda_stream = test_THCP._CudaStreamBase()
        print("Instance of _CudaStreamBase created successfully")
    except Exception as e:
        print("Error creating instance:", e)

    # 测试方法调用
    if cuda_stream:
        try:
            print("Calling query method:", cuda_stream.query())
            cuda_stream.synchronize()
            print("synchronize() method called successfully")
            # priority = cuda_stream.priority_range()
            # print("priority_range() called, priority:", priority)
        except AttributeError as e:
            print("Method not found:", e)
else:
    print("_CudaStreamBase not found in test_THCPStream")

