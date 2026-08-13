// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Modified by Hygon Information Technology Co., Ltd., 2026.

#ifndef USE_ROCM
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#else
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPGuard.h>
#endif
#include <mutex>
#include <utility>
#include <utils/mocker_logging.h>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

namespace torch::cuda::CUDAPluggableAllocator {

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata() : size(0), device_idx(-1) { if (Vlog()) LOG_INFO(); }

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    c10::DeviceIndex device_idx,
#ifndef USE_ROCM
    cudaStream_t stream)
#else
    hipStream_t stream)
#endif
    : size(size), device_idx(device_idx), stream(stream) { if (Vlog()) LOG_INFO(); }

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
CUDAPluggableAllocator::CUDAPluggableAllocator(
#ifndef USE_ROCM
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn)
#else
    std::function<void*(size_t, int, hipStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, hipStream_t)> free_fn)
#endif
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) { if (Vlog()) LOG_INFO(); }

CUDAPluggableAllocator::CUDAPluggableAllocator(CUDAPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) { if (Vlog()) LOG_INFO(); }

void CUDAPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  if (Vlog()) LOG_INFO();
  init_fn_ = std::move(init_fn);
}

void CUDAPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  if (Vlog()) LOG_INFO();
  reset_fn_ = std::move(reset_fn);
}

void CUDAPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn) {
  if (Vlog()) LOG_INFO();
  memory_fraction_fn_ = std::move(memory_fraction_fn);
}

void CUDAPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn) {
  if (Vlog()) LOG_INFO();
  base_alloc_fn_ = std::move(base_alloc_fn);
}

void CUDAPluggableAllocator::set_record_stream_fn(
#ifndef USE_ROCM
    std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn) {
#else
    std::function<void(void* ptr, hipStream_t stream)> record_stream_fn) {
#endif
  if (Vlog()) LOG_INFO();
  record_stream_fn_ = std::move(record_stream_fn);
}

void CUDAPluggableAllocator::set_begin_allocate_to_pool(
#ifndef USE_ROCM
    std::function<
        void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
        capture_begin_fn) {
#else
    std::function<
        void(int, c10::cuda::MempoolId_t, std::function<bool(hipStream_t)>)>
        capture_begin_fn) {
#endif
  if (Vlog()) LOG_INFO();
  begin_allocate_to_pool_fn_ = std::move(capture_begin_fn);
}

void CUDAPluggableAllocator::set_end_allocate_to_pool_fn(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_about_to_end_fn) {
  if (Vlog()) LOG_INFO();
  end_allocate_to_pool_fn_ = std::move(capture_about_to_end_fn);
}

void CUDAPluggableAllocator::set_release_pool(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn) {
  if (Vlog()) LOG_INFO();
  relase_pool_fn_ = std::move(capture_destroy_fn);
}

void* CUDAPluggableAllocator::malloc(
    size_t size,
    c10::DeviceIndex device,
#ifndef USE_ROCM
    cudaStream_t stream) {
#else
    hipStream_t stream) {
#endif
  if (Vlog()) LOG_INFO();
  void* r = alloc_fn_(size, device, stream);
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  return r;
}

c10::DataPtr CUDAPluggableAllocator::allocate(size_t size) {
  if (Vlog()) LOG_INFO();
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
#ifndef USE_ROCM
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
#else
  hipStream_t stream = c10::cuda::getCurrentCUDAStream(device);
#endif
  void* r = this->malloc(size, device, stream);
  c10::DataPtr data_ptr = {
      r, r, raw_deleter(), c10::Device(c10::DeviceType::CUDA, device)};
  return data_ptr;
}

c10::DeleterFnPtr CUDAPluggableAllocator::raw_deleter() const {
  if (Vlog()) LOG_INFO();
  return &custom_raw_deleter;
}

void* CUDAPluggableAllocator::raw_alloc(size_t nbytes) {
  if (Vlog()) LOG_INFO();
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
#ifndef USE_ROCM
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
#else
  hipStream_t stream = c10::cuda::getCurrentCUDAStream(device);
#endif
  return malloc(nbytes, device, stream);
}

void* CUDAPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
#ifndef USE_ROCM
    cudaStream_t stream) {
#else
    hipStream_t stream) {
#endif
  if (Vlog()) LOG_INFO();
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  return malloc(nbytes, device, stream);
}

void CUDAPluggableAllocator::raw_delete(void* ptr) {
  if (Vlog()) LOG_INFO();
#ifndef USE_ROCM
  cudaStream_t stream{};
#else
  hipStream_t stream{};
#endif
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, stream);
}

void CUDAPluggableAllocator::init(int device_count) {
  if (Vlog()) LOG_INFO();
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool CUDAPluggableAllocator::initialized() {
  if (Vlog()) LOG_INFO();
  return initialized_;
}

double CUDAPluggableAllocator::getMemoryFraction(c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getMemoryFraction. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::setMemoryFraction(
    double fraction,
    c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

std::vector<c10::cuda::CUDACachingAllocator::StreamSegmentSize>
CUDAPluggableAllocator::getExpandableSegmentSizes(c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAMallocAsyncAllocator does not yet support getExpandableSegmentSizes.");
}

void CUDAPluggableAllocator::emptyCache(
    /*unused*/ c10::cuda::MempoolId_t mempool_id) {
  if (Vlog()) LOG_INFO();
  if (reset_fn_) {
    return reset_fn_();
  }
}

void CUDAPluggableAllocator::cacheInfo(
    c10::DeviceIndex device,
    size_t* largestBlock) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* CUDAPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (Vlog()) LOG_INFO();
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

void CUDAPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    c10::cuda::CUDAStream stream) {
  if (Vlog()) LOG_INFO();
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

c10::CachingDeviceAllocator::DeviceStats CUDAPluggableAllocator::getDeviceStats(
    c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::SnapshotInfo CUDAPluggableAllocator::snapshot(
    c10::cuda::MempoolId_t mempool_id,
    bool include_traces) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::ShareableHandle CUDAPluggableAllocator::
    shareIpcHandle(void* ptr) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support shareIPcHandle. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<void> CUDAPluggableAllocator::getIpcDevPtr(std::string handle) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// CUDAGraph interactions
void CUDAPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id,
#ifndef USE_ROCM
    std::function<bool(cudaStream_t)> filter) {
#else
    std::function<bool(hipStream_t)> filter) {
#endif
  if (Vlog()) LOG_INFO();
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

void CUDAPluggableAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  if (Vlog()) LOG_INFO();
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

void CUDAPluggableAllocator::releasePool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  if (Vlog()) LOG_INFO();
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

void CUDAPluggableAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10::cuda::CUDACachingAllocator::RecordContext when,
    bool clearHistory,
    const std::vector<std::string>& skip_actions) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachOomRejectionObserver(
    c10::cuda::CUDACachingAllocator::OomRejectionObserver observer) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "CUDAPluggableAllocator does not yet support attachOomRejectionObserver. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachAllocatorTraceTracker(
    c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not support attachAllocatorTraceTracker. "
      "attachAllocatorTraceTracker is only used inside Pytorch.");
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
CUDAPluggableAllocator::getCheckpointState(
    c10::DeviceIndex device,
    at::cuda::MempoolId_t id) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getCheckpointState. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::CheckpointDelta CUDAPluggableAllocator::
    setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  if (Vlog()) LOG_INFO();
  c10::cuda::CUDAGuard device_guard(dev);
#ifndef USE_ROCM
  cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    // ignore and clear the error if access was already enabled
    (void)cudaGetLastError();
#else
  hipError_t err = hipDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == hipErrorPeerAccessAlreadyEnabled) {
    // ignore and clear the error if access was already enabled
    (void)hipGetLastError();
#endif
  } else {
    C10_CUDA_CHECK(err);
  }
}

#ifndef USE_ROCM
cudaError_t CUDAPluggableAllocator::memcpyAsync(
#else
hipError_t CUDAPluggableAllocator::memcpyAsync(
#endif
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
#ifndef USE_ROCM
    cudaStream_t stream,
#else
    hipStream_t stream,
#endif
    bool p2p_enabled) {
  if (Vlog()) LOG_INFO();
#ifndef USE_ROCM
  return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
#else
  return hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream);
#endif
}

std::string CUDAPluggableAllocator::name() {
  if (Vlog()) LOG_INFO();
  return "pluggable";
}

void CUDAPluggableAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  if (Vlog()) LOG_INFO();
#ifndef USE_ROCM
  C10_CUDA_CHECK(
      cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
#else
  C10_CUDA_CHECK(
      hipMemcpy(dest, src, count, hipMemcpyKind::hipMemcpyDeviceToDevice));
#endif
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
    current_custom_allocator;

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
getCurrentAllocator() {
  if (Vlog()) LOG_INFO();
  return current_custom_allocator;
}

// TODO: add more functions in the argument
std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomAllocator(
#ifndef USE_ROCM
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn) {
#else
    std::function<void*(size_t, int, hipStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, hipStream_t)> free_fn) {
#endif
  if (Vlog()) LOG_INFO();
  std::shared_ptr<CUDAPluggableAllocator> allocator(
      new CUDAPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
  allocator->init(device_count);
  return allocator;
}

void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      !c10::cuda::CUDACachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  c10::cuda::CUDACachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr) {
  if (Vlog()) LOG_INFO();
  current_custom_allocator->raw_delete(ptr);
}

} // namespace torch::cuda::CUDAPluggableAllocator
