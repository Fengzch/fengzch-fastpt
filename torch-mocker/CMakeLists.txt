# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.10)
project(Torch-Mocker-Test LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TORCH_PATH "" CACHE PATH "Path to the installed torch package")
if(NOT TORCH_PATH)
    if(DEFINED ENV{TORCH_PATH} AND NOT "$ENV{TORCH_PATH}" STREQUAL "")
        set(TORCH_PATH "$ENV{TORCH_PATH}")
    else()
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                    "import os, torch; print(os.path.dirname(torch.__file__))"
            RESULT_VARIABLE torch_path_result
            OUTPUT_VARIABLE TORCH_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE torch_path_error
        )
        if(NOT torch_path_result EQUAL 0)
            message(FATAL_ERROR
                "Unable to determine the installed torch path: ${torch_path_error}")
        endif()
    endif()
endif()

get_filename_component(TORCH_PATH "${TORCH_PATH}" ABSOLUTE)
set(TORCH_PATH "${TORCH_PATH}" CACHE PATH "Path to the installed torch package" FORCE)
if(NOT EXISTS "${TORCH_PATH}/include")
    message(FATAL_ERROR
        "Invalid TORCH_PATH '${TORCH_PATH}': torch headers were not found")
endif()
set(ENV{TORCH_PATH} "${TORCH_PATH}")
message(STATUS "Using torch package: ${TORCH_PATH}")

# fmt is a transitive dependency of torch_cuda.  Configure the vendored
# project before this legacy build sets nvcc as the host C++ compiler.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt EXCLUDE_FROM_ALL)
set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_COMPILER nvcc)

execute_process(
    COMMAND python3 tools/generate_torch_version.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "Python script failed: ${error}")
else()
    message(STATUS "Python output: ${output}")
endif()

option(USE_TEST "Use single test" OFF)
option(USE_NCCL "Enable NCCL support" ON)
option(USE_C10D_NCCL "Enable C10D_NCCL support" ON)
if(USE_NCCL)
    add_definitions(-DUSE_NCCL)
endif()

if(USE_C10D_NCCL)
    add_definitions(-DUSE_C10D_NCCL)
endif()

set(Torch_CUDA_SRCS)
set(C10_CUDA_SRCS)
set(Torch_MOCKER_SRCS)
set(Torch_MOCKER_VLOG)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

include(GNUInstallDirs)

find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})
file(GLOB_RECURSE Torch_CUDA_SRCS "aten/src/ATen/*.cpp")
list(FILTER Torch_CUDA_SRCS EXCLUDE REGEX "aten/src/ATen/cuda/nvrtc_stub/*ATenNVRTC.cpp")
file(GLOB NVRTC_SRCS "aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp")
file(GLOB_RECURSE Torch_CUDA_SRCS2 "torch/csrc/*.cpp" "torch/csrc/*.cu")
list(APPEND Torch_CUDA_SRCS ${Torch_CUDA_SRCS} ${Torch_CUDA_SRCS2})

file(GLOB C10_CUDA_SRCS1 "c10/cuda/*.cpp")
file(GLOB C10_CUDA_SRCS2 "c10/cuda/impl/*.cpp")
file(GLOB C10_CUDA_SRCS3 "c10/util/*.cpp")
list(APPEND C10_CUDA_SRCS ${C10_CUDA_SRCS1} ${C10_CUDA_SRCS2} ${C10_CUDA_SRCS3})
file(GLOB Torch_MOCKER_SRCS "torch_mocker/*.cpp")
file(GLOB Torch_MOCKER_VLOG "torch_mocker/utils/*.cpp")

# build libcaffe2_nvrtc.so
#add_library(caffe2_nvrtc SHARED ${NVRTC_SRCS})
#target_link_libraries(caffe2_nvrtc PRIVATE hiprtc)
#target_compile_definitions(caffe2_nvrtc PRIVATE "-D__HIP_PLATFORM_AMD__")

add_library(torch_mocker  SHARED ${Torch_MOCKER_SRCS})
add_library(torch_cuda SHARED ${Torch_CUDA_SRCS})
add_library(c10_cuda  SHARED ${C10_CUDA_SRCS} ${Torch_MOCKER_VLOG})

# Encode DTK MAJOR.MINOR.PATCH as MAJOR * 10000 + MINOR * 100 + PATCH.
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(FASTPT_ROCM_PATH "$ENV{ROCM_PATH}")
elseif(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
    set(FASTPT_ROCM_PATH "$ENV{ROCM_HOME}")
else()
    message(FATAL_ERROR "ROCM_PATH or ROCM_HOME is required")
endif()

set(FASTPT_DTK_VERSION_FILE "${FASTPT_ROCM_PATH}/.dtk_version")
if(NOT EXISTS "${FASTPT_DTK_VERSION_FILE}")
    message(FATAL_ERROR
        "DTK version file does not exist: ${FASTPT_DTK_VERSION_FILE}")
endif()

file(STRINGS "${FASTPT_DTK_VERSION_FILE}"
    FASTPT_DTK_VERSION_LINE LIMIT_COUNT 1)
string(REGEX MATCH
    "^DTK[-_]([0-9]+)[.]([0-9][0-9])([.]([0-9]+))?([-_]|$)"
    FASTPT_DTK_VERSION_MATCH
    "${FASTPT_DTK_VERSION_LINE}")

if(NOT FASTPT_DTK_VERSION_MATCH)
    message(FATAL_ERROR
        "Invalid DTK version in ${FASTPT_DTK_VERSION_FILE}: ${FASTPT_DTK_VERSION_LINE}")
endif()

if(CMAKE_MATCH_4)
    set(FASTPT_DTK_PATCH "${CMAKE_MATCH_4}")
else()
    set(FASTPT_DTK_PATCH 0)
endif()

math(EXPR PYTORCH_DTK_VERSION
    "${CMAKE_MATCH_1} * 10000 + ${CMAKE_MATCH_2} * 100 + ${FASTPT_DTK_PATCH}")
message(STATUS "Detected DTK version: ${PYTORCH_DTK_VERSION}")
target_compile_definitions(c10_cuda PRIVATE
    PYTORCH_DTK_VERSION=${PYTORCH_DTK_VERSION})

add_dependencies(torch_cuda torch_mocker c10_cuda)
#优先查找torchmocker下的头文件
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include/aten/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/aten/src)
include_directories(SYSTEM ${TORCH_PATH}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/torch_mocker)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn-frontend/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn-frontend/include/cudnn_frontend/thirdparty)
include_directories(SYSTEM ${TORCH_PATH}/include/torch/csrc/api/include)

add_definitions(-DTORCH_USE_CUDA_DSA)
add_definitions(-DPYTORCH_C10_DRIVER_API_SUPPORTED)
#add_definitions("-D__CUDA_HIP_PLATFORM_AMD__")
#add_definitions("-D__HIP_PLATFORM_AMD__")
#add_definitions("-DC10_USING_CUSTOM_GENERATED_MACROS")
target_compile_definitions(torch_mocker PRIVATE "-D__CUDA_HIP_PLATFORM_AMD__")
target_compile_definitions(torch_mocker PRIVATE "-D__HIP_PLATFORM_AMD__")
target_link_libraries(torch_mocker PUBLIC ${TORCH_PATH}/lib/libtorch_hip.so)
target_compile_definitions(torch_cuda PUBLIC "-DC10_USING_CUSTOM_GENERATED_MACROS")
#target_include_directories(torch_cuda PRIVATE $ENV{TORCH_PATH}/include)
target_compile_definitions(c10_cuda PRIVATE "-D__CUDA_HIP_PLATFORM_AMD__")
target_compile_definitions(c10_cuda PRIVATE "-D__HIP_PLATFORM_AMD__")
target_compile_definitions(c10_cuda PUBLIC "-DC10_USING_CUSTOM_GENERATED_MACROS")

target_link_libraries(torch_cuda PUBLIC torch_mocker)
message("${TORCH_PATH}/lib/libc10_hip.so")
target_link_libraries(c10_cuda PUBLIC ${TORCH_PATH}/lib/libc10_hip.so cublasLt torch_mocker)

target_compile_definitions(torch_cuda PRIVATE "-D__HIP_PLATFORM_AMD__")
target_compile_definitions(torch_cuda PUBLIC "-DC10_USING_CUSTOM_GENERATED_MACROS")
if(USE_GLOG)
    target_compile_definitions(torch_cuda PUBLIC "-DC10_USE_GLOG")
    set(TORCH_LIBS_DIR ${TORCH_PATH}/../torch.libs)
    if(EXISTS ${TORCH_LIBS_DIR})
        file(GLOB GLOG_LIB "${TORCH_LIBS_DIR}/libglog-*.so*")
        if(NOT GLOG_LIB)
            message(FATAL_ERROR "libglog not found in ${TORCH_LIBS_DIR}")
        else()
            list(GET GLOG_LIB 0 GLOG_LIB_PATH)
            target_link_libraries(torch_cuda PUBLIC ${GLOG_LIB_PATH})
        endif()
    else()
        target_link_libraries(torch_cuda PUBLIC glog)
    endif()
endif()
target_link_libraries(torch_cuda PUBLIC torch_mocker)
target_link_libraries(torch_cuda PUBLIC ${TORCH_PATH}/lib/libtorch_hip.so)
# target_link_libraries(torch_cuda PUBLIC ${TORCH_PATH}/lib/libtorch_python.so)
target_link_directories(torch_cuda PUBLIC ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
target_link_libraries(torch_cuda PUBLIC cuda nvrtc cudnn_cnn_infer cublas cublasLt cufft cusparse cudnn cudnn_cnn_infer cudnn_adv_infer nvToolsExt fmt)
#target_link_libraries(torch_cuda PUBLIC cuda nvrtc cudnn_cnn_infer cublas cublasLt cufft cusparse cudnn cudnn_cnn_infer cudnn_adv_infer)
target_link_libraries(torch_cuda PUBLIC c10_cuda)
# target_link_libraries(torch_cuda PUBLIC ${Python3_LIBRARIES})
message("${TORCH_PATH}/lib/libtorch_cuda.so")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++20")
include(CTest)
if(USE_TEST AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_py)
