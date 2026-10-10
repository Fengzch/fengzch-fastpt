# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# cuda_add_executable(test_THCP "./test_stream.cpp")
file(GLOB_RECURSE TEST_PYTHON_SRCS "*.cpp")
add_library(test_THCP MODULE ${TEST_PYTHON_SRCS})
target_compile_definitions(test_THCP PRIVATE "-DUSE_CUDA")
target_link_libraries(test_THCP ${Python3_LIBRARIES})
target_link_libraries(test_THCP ${TORCH_PATH}/lib/libc10.so)
target_link_libraries(test_THCP ${TORCH_PATH}/lib/libtorch_cpu.so)
target_link_libraries(test_THCP ${TORCH_PATH}/lib/libtorch.so)
target_link_libraries(test_THCP torch_cuda c10_cuda torch_mocker)
target_link_libraries(test_THCP cufft cusparse cudnn cudnn_cnn_infer cudnn_adv_infer)
target_link_libraries(test_THCP cublas cublasLt)
set_target_properties(test_THCP PROPERTIES PREFIX "" SUFFIX ".so")
