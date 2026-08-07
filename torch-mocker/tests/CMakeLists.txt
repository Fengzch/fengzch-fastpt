# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# CTest registration and test-only build settings.
# C++ tests share one runner per domain. CUDA test translation units retain an
# isolated runner because device-template linkage can affect kernel behavior.

set(TORCH_MOCKER_TEST_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(torch_mocker_add_test_runner target)
    cuda_add_executable(${target} ${TORCH_MOCKER_TEST_SOURCE_DIR}/support/TestMain.cpp)
    target_link_libraries(${target}
        $ENV{TORCH_PATH}/lib/libc10.so
        $ENV{TORCH_PATH}/lib/libtorch_cpu.so
        $ENV{TORCH_PATH}/lib/libtorch.so
        $ENV{TORCH_PATH}/lib/libcaffe2_nvrtc.so
        torch_cuda c10_cuda torch_mocker
        cufft cusparse cudnn cudnn_cnn_infer cudnn_adv_infer numa
        cublas nccl cublasLt pthread fmt)
    target_compile_options(${target} PRIVATE -O3)
    target_compile_definitions(${target} PRIVATE
        CUB_WRAPPED_NAMESPACE=at_cuda_detail
        __HIP_PLATFORM_AMD__)
endfunction()

foreach(group IN ITEMS c10 aten_cuda aten_cudnn aten_native csrc torch_cuda)
    torch_mocker_add_test_runner(torch_mocker_tests_${group})
endforeach()

function(torch_mocker_add_test_source group test_name source)
    get_filename_component(extension "${source}" LAST_EXT)
    if(extension STREQUAL ".cu")
        string(REPLACE "." "_" target_suffix "${test_name}")
        set(target torch_mocker_test_${target_suffix})
        torch_mocker_add_test_runner(${target})
    else()
        set(target torch_mocker_tests_${group})
    endif()
    target_sources(${target} PRIVATE ${source})
    set_property(SOURCE ${source} APPEND PROPERTY
        COMPILE_DEFINITIONS "TORCH_MOCKER_TEST_SELECTOR=\"${test_name}\"")
    add_test(
        NAME torch_mocker.${test_name}
        COMMAND $<TARGET_FILE:${target}> ${test_name})
    set_tests_properties(torch_mocker.${test_name} PROPERTIES
        LABELS "c10;gpu"
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>)
endfunction()

torch_mocker_add_test_source(c10 c10.cuda_functions ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAFunctions.cpp)
torch_mocker_add_test_source(c10 c10.cuda_stream ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAStream.cpp)
torch_mocker_add_test_source(c10 c10.cuda_guard ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAGard.cpp)
torch_mocker_add_test_source(c10 c10.cuda_guard_impl ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAGuardImpl.cpp)
torch_mocker_add_test_source(c10 c10.cuda_graph_utils ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAGraphsC10Utils.cpp)
torch_mocker_add_test_source(c10 c10.cuda_math_compat ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAMathCompat.cu)
torch_mocker_add_test_source(c10 c10.cuda_exception ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAException.cpp)
torch_mocker_add_test_source(c10 c10.cuda_misc ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAMiscFunctions.cpp)
torch_mocker_add_test_source(c10 c10.cuda_smoke ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDATest.cpp)
torch_mocker_add_test_source(c10 c10.cuda_macros ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAMacros.cpp)
torch_mocker_add_test_source(c10 c10.driver_api ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_driver_api.cpp)
torch_mocker_add_test_source(c10 c10.cuda_malloc_async_allocator ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAMallocAsyncAllocator.cpp)
torch_mocker_add_test_source(c10 c10.cuda_allocator_config ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDAAllocatorConfig.cpp)
torch_mocker_add_test_source(c10 c10.cuda_caching_allocator ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDACachingAllocator.cpp)
torch_mocker_add_test_source(c10 c10.cuda_device_assertion ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDADeviceAssertion.cu)
torch_mocker_add_test_source(c10 c10.cuda_device_assertion_host ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_CUDADeviceAssertionHost.cpp)
torch_mocker_add_test_source(c10 c10.gemm_common ${TORCH_MOCKER_TEST_SOURCE_DIR}/c10/test_GemmCommon.cpp)
set_tests_properties(torch_mocker.c10.cuda_malloc_async_allocator PROPERTIES
    ENVIRONMENT "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
    LABELS "c10;allocator;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_data_type ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDADataType.cpp)
torch_mocker_add_test_source(aten_cuda aten.cuda_device ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDADevice.cpp)
torch_mocker_add_test_source(aten_cuda aten.cuda_event ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDAEvent.cpp)
torch_mocker_add_test_source(aten_cuda aten.cuda_graph ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDAGraph.cpp)
torch_mocker_add_test_source(aten_cuda aten.cuda_graph_utils ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDAGraphsUtils.cpp)
set_tests_properties(
    torch_mocker.aten.cuda_data_type
    torch_mocker.aten.cuda_device
    torch_mocker.aten.cuda_event
    torch_mocker.aten.cuda_graph
    torch_mocker.aten.cuda_graph_utils
    PROPERTIES LABELS "aten;cuda;gpu")

torch_mocker_add_test_source(csrc csrc.profiler_stubs ${TORCH_MOCKER_TEST_SOURCE_DIR}/csrc/test_base.cpp)
torch_mocker_add_test_source(csrc csrc.cuda_ipc_types ${TORCH_MOCKER_TEST_SOURCE_DIR}/csrc/test_CudaIPCTypes.cpp)
torch_mocker_add_test_source(csrc csrc.cuda_symmetric_memory_headers ${TORCH_MOCKER_TEST_SOURCE_DIR}/csrc/test_CUDASymmetricMemoryHeaders.cpp)
torch_mocker_add_test_source(csrc csrc.autograd_scatter ${TORCH_MOCKER_TEST_SOURCE_DIR}/csrc/autograd/functions/test_comm.cpp)
set_tests_properties(torch_mocker.csrc.autograd_scatter PROPERTIES
    LABELS "csrc;autograd;multi-gpu"
    RUN_SERIAL TRUE)

torch_mocker_add_test_source(csrc csrc.cuda_codegen ${TORCH_MOCKER_TEST_SOURCE_DIR}/csrc/test_codegen.cpp)
set_tests_properties(torch_mocker.csrc.cuda_codegen PROPERTIES
    LABELS "csrc;codegen;gpu")

torch_mocker_add_test_source(aten_cudnn aten.cudnn_handle ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cudnn/test_handle.cpp)
torch_mocker_add_test_source(aten_cudnn aten.cudnn_types ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cudnn/test_types.cpp)
set_tests_properties(
    torch_mocker.aten.cudnn_handle
    torch_mocker.aten.cudnn_types
    PROPERTIES LABELS "aten;cudnn;gpu")

torch_mocker_add_test_source(aten_cudnn aten.cudnn_descriptors ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cudnn/test_discription.cpp)
set_tests_properties(torch_mocker.aten.cudnn_descriptors PROPERTIES
    LABELS "aten;cudnn;gpu")

torch_mocker_add_test_source(torch_cuda torch.cuda_pluggable_allocator ${TORCH_MOCKER_TEST_SOURCE_DIR}/torch/csrc/cuda/test_CUDAPluggableAllocator.cpp)
set_tests_properties(torch_mocker.torch.cuda_pluggable_allocator PROPERTIES
    LABELS "torch;allocator;gpu")

torch_mocker_add_test_source(torch_cuda torch.cuda_pluggable_allocator_global ${TORCH_MOCKER_TEST_SOURCE_DIR}/torch/csrc/cuda/test_CUDAPluggableAllocatorGlobal.cpp)
set_tests_properties(torch_mocker.torch.cuda_pluggable_allocator_global PROPERTIES
    LABELS "torch;allocator;gpu"
    RUN_SERIAL TRUE)

torch_mocker_add_test_source(torch_cuda torch.cuda_comm ${TORCH_MOCKER_TEST_SOURCE_DIR}/torch/csrc/cuda/torch_test_comm.cpp)
set_tests_properties(torch_mocker.torch.cuda_comm PROPERTIES
    LABELS "torch;cuda;gpu")

torch_mocker_add_test_source(torch_cuda torch.cuda_nccl ${TORCH_MOCKER_TEST_SOURCE_DIR}/torch/csrc/cuda/test_nccl.cpp)
set_tests_properties(torch_mocker.torch.cuda_nccl PROPERTIES
    LABELS "torch;nccl;multi-gpu"
    RUN_SERIAL TRUE)

torch_mocker_add_test_source(aten_native aten.native_cufft_utils ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_CuFFTUtils.cpp)
torch_mocker_add_test_source(aten_native aten.native_jit_utils ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_jit_utils.cpp)
torch_mocker_add_test_source(aten_native aten.native_resize ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_Resize.cpp)
set_tests_properties(
    torch_mocker.aten.native_cufft_utils
    torch_mocker.aten.native_jit_utils
    torch_mocker.aten.native_resize
    PROPERTIES LABELS "aten;native;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_apply_utils ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDAApplyUtils.cu)
torch_mocker_add_test_source(aten_cuda aten.cuda_exceptions ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_Exceptions.cpp)
torch_mocker_add_test_source(aten_cuda aten.cuda_sleep ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_sleep.cu)
torch_mocker_add_test_source(aten_cuda aten.cuda_stream_timer ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_StreamTimer.cpp)
set_tests_properties(
    torch_mocker.aten.cuda_apply_utils
    torch_mocker.aten.cuda_exceptions
    torch_mocker.aten.cuda_sleep
    torch_mocker.aten.cuda_stream_timer
    PROPERTIES LABELS "aten;cuda;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_context ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDAContext.cpp)
set_tests_properties(torch_mocker.aten.cuda_context PROPERTIES
    LABELS "aten;cuda;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_generator ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDAGeneratorImpl.cpp)
set_tests_properties(torch_mocker.aten.cuda_generator PROPERTIES
    LABELS "aten;cuda;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_loops ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDALoops.cu)
set_tests_properties(torch_mocker.aten.cuda_loops PROPERTIES
    LABELS "aten;cuda;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_sparse_descriptors ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDASparseDescriptors.cpp)
set_tests_properties(torch_mocker.aten.cuda_sparse_descriptors PROPERTIES
    LABELS "aten;cuda;sparse;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_cub ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_cub.cu)
set_tests_properties(torch_mocker.aten.cuda_cub PROPERTIES
    LABELS "aten;cuda;cub;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_blas ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDABlas.cpp)
set_tests_properties(torch_mocker.aten.cuda_blas PROPERTIES
    LABELS "aten;cuda;blas;gpu")

torch_mocker_add_test_source(aten_cuda aten.cuda_sparse_blas ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/cuda/test_CUDASparseBlas.cpp)
set_tests_properties(torch_mocker.aten.cuda_sparse_blas PROPERTIES
    LABELS "aten;cuda;sparse;gpu")

torch_mocker_add_test_source(aten_native aten.native_imcol ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_imcol.cu)
torch_mocker_add_test_source(aten_native aten.native_vol2col ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_vol2col.cu)
set_tests_properties(
    torch_mocker.aten.native_imcol
    torch_mocker.aten.native_vol2col
    PROPERTIES LABELS "aten;native;gpu")

torch_mocker_add_test_source(aten_native aten.native_scan_utils ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_ScanUtils.cu)
set_tests_properties(torch_mocker.aten.native_scan_utils PROPERTIES
    LABELS "aten;native;gpu")

torch_mocker_add_test_source(aten_native aten.native_normalization ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/cuda/test_Normalization.cu)
set_tests_properties(torch_mocker.aten.native_normalization PROPERTIES
    LABELS "aten;native;gpu")

torch_mocker_add_test_source(aten_native aten.native_reduce ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_Reduce.cu)
torch_mocker_add_test_source(aten_native aten.native_distribution_templates ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_DistributionTemplates.cu)
set_tests_properties(
    torch_mocker.aten.native_reduce
    torch_mocker.aten.native_distribution_templates
    PROPERTIES LABELS "aten;native;gpu")

torch_mocker_add_test_source(aten_native aten.native_persistent_softmax ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_PersistentSoftmax.cu)
torch_mocker_add_test_source(aten_native aten.native_multi_tensor_apply ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_MultiTensorApply.cu)
set_tests_properties(
    torch_mocker.aten.native_persistent_softmax
    torch_mocker.aten.native_multi_tensor_apply
    PROPERTIES LABELS "aten;native;gpu")

torch_mocker_add_test_source(aten_native aten.native_cufft_plan_cache ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_CuFFTPlanCache.cpp)
torch_mocker_add_test_source(aten_native aten.native_randperm ${TORCH_MOCKER_TEST_SOURCE_DIR}/ATen/native/test_Randperm.cu)
set_tests_properties(
    torch_mocker.aten.native_cufft_plan_cache
    torch_mocker.aten.native_randperm
    PROPERTIES LABELS "aten;native;gpu")
