# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import sys
sys.path.append('../build/test_py/')

#必须后引用torch
import test_THCP
import torch

stream1 = torch.cuda.Stream(priority=0)
stream2 = torch.cuda.Stream(priority=-1)
stream_list = [stream1, None, stream2]

cuda_streams = test_THCP.to_cuda_stream_list(stream_list)
print("CUDA Stream List:", cuda_streams)  # 输出: [指针地址, None, 指针地址]

for i, stream in enumerate(cuda_streams):
    if stream is None:
        print(f"Stream {i}: None")
    else:
        print(f"Stream {i}: {stream}")
