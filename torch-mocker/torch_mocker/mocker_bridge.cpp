// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// #include <vector_types.h>
#include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>

#include <dlfcn.h>
#include <atomic>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <utils/mocker_logging.h>

namespace c10::cuda::CUDACachingAllocator {
class CUDAAllocator;
}

namespace {

void* resolve_c10_hip_symbol(void* handle, const char* name) {
  dlerror();
  void* symbol = dlsym(handle, name);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    throw std::runtime_error(
        std::string("FastPT failed to resolve libc10_hip.so symbol ") + name +
        ": " + (error == nullptr ? "null symbol" : error));
  }
  return symbol;
}

using CUDAAllocatorType = c10::cuda::CUDACachingAllocator::CUDAAllocator;
using AllocatorAtomic = std::atomic<CUDAAllocatorType*>;

AllocatorAtomic& resolve_allocator_atomic(void* handle) {
  return *reinterpret_cast<AllocatorAtomic*>(
      resolve_c10_hip_symbol(handle,
          "_ZN3c104cuda20CUDACachingAllocator9allocatorE"));
}

void* open_libc10_hip() {
  void* h = dlopen("libc10_hip.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (!h) {
    h = dlopen("libc10_hip.so", RTLD_NOW | RTLD_LOCAL);
  }
  if (!h) {
    throw std::runtime_error(
        std::string("FastPT failed to open libc10_hip.so for CUDAStream bridge: ") +
        dlerror());
  }
  return h;
}

template <typename Fn>
Fn resolve_owner(void* handle, const char* name) {
  return reinterpret_cast<Fn>(resolve_c10_hip_symbol(handle, name));
}

using StreamFn = cudaStream_t (*)(const c10::cuda::CUDAStream*);
using QueryFn = bool (*)(const c10::cuda::CUDAStream*);
using SynchronizeFn = void (*)(const c10::cuda::CUDAStream*);
using GetPoolIntFn = c10::cuda::CUDAStream (*)(int, c10::DeviceIndex);
using GetPoolBoolFn = c10::cuda::CUDAStream (*)(bool, c10::DeviceIndex);
using GetExternalFn = c10::cuda::CUDAStream (*)(cudaStream_t, c10::DeviceIndex);
using GetDefaultFn = c10::cuda::CUDAStream (*)(c10::DeviceIndex);
using GetCurrentFn = c10::cuda::CUDAStream (*)(c10::DeviceIndex);
using SetCurrentFn = void (*)(c10::cuda::CUDAStream);
using StreamOutFn = std::ostream& (*)(std::ostream&, const c10::cuda::CUDAStream&);

struct OwnerTable {
  void* handle;
  AllocatorAtomic& allocator_atomic;
  StreamFn stream;
  QueryFn query;
  SynchronizeFn synchronize;
  GetPoolIntFn get_pool_int;
  GetPoolBoolFn get_pool_bool;
  GetExternalFn get_external;
  GetDefaultFn get_default;
  GetCurrentFn get_current;
  SetCurrentFn set_current;
  StreamOutFn stream_out;
};

OwnerTable make_owner_table() {
  void* handle = open_libc10_hip();
  return OwnerTable{
      handle,
      resolve_allocator_atomic(handle),
      resolve_owner<StreamFn>(handle, "_ZNK3c104cuda10CUDAStream6streamEv"),
      resolve_owner<QueryFn>(handle, "_ZNK3c104cuda10CUDAStream5queryEv"),
      resolve_owner<SynchronizeFn>(
          handle, "_ZNK3c104cuda10CUDAStream11synchronizeEv"),
      resolve_owner<GetPoolIntFn>(handle, "_ZN3c104cuda17getStreamFromPoolEia"),
      resolve_owner<GetPoolBoolFn>(handle, "_ZN3c104cuda17getStreamFromPoolEba"),
      resolve_owner<GetExternalFn>(
          handle, "_ZN3c104cuda21getStreamFromExternalEP12ihipStream_ta"),
      resolve_owner<GetDefaultFn>(handle, "_ZN3c104cuda20getDefaultCUDAStreamEa"),
      resolve_owner<GetCurrentFn>(handle, "_ZN3c104cuda20getCurrentCUDAStreamEa"),
      resolve_owner<SetCurrentFn>(
          handle, "_ZN3c104cuda20setCurrentCUDAStreamENS0_10CUDAStreamE"),
      resolve_owner<StreamOutFn>(handle, "_ZN3c104cudalsERSoRKNS0_10CUDAStreamE")};
}

const OwnerTable& owner_table() {
  static const OwnerTable table = make_owner_table();
  return table;
}

} // namespace

extern "C" cudaStream_t fastpt_cuda_stream_stream(
    const c10::cuda::CUDAStream* stream) {
  if (Vlog()) LOG_INFO();
  return owner_table().stream(stream);
}

extern "C" bool fastpt_cuda_stream_query(
    const c10::cuda::CUDAStream* stream) {
  if (Vlog()) LOG_INFO();
  return owner_table().query(stream);
}

extern "C" void fastpt_cuda_stream_synchronize(
    const c10::cuda::CUDAStream* stream) {
  if (Vlog()) LOG_INFO();
  owner_table().synchronize(stream);
}

extern "C" c10::cuda::CUDAStream fastpt_cuda_get_stream_from_pool_int(
    int priority,
    c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  return owner_table().get_pool_int(priority, device);
}

extern "C" c10::cuda::CUDAStream fastpt_cuda_get_stream_from_pool_bool(
    bool is_high_priority,
    c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  return owner_table().get_pool_bool(is_high_priority, device);
}

extern "C" c10::cuda::CUDAStream fastpt_cuda_get_stream_from_external(
    cudaStream_t stream,
    c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  return owner_table().get_external(stream, device);
}

extern "C" c10::cuda::CUDAStream fastpt_cuda_get_default_stream(
    c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  return owner_table().get_default(device);
}

extern "C" c10::cuda::CUDAStream fastpt_cuda_get_current_stream(
    c10::DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  return owner_table().get_current(device);
}

extern "C" void fastpt_cuda_set_current_stream(
    c10::cuda::CUDAStream stream) {
  if (Vlog()) LOG_INFO();
  owner_table().set_current(stream);
}

extern "C" std::ostream& fastpt_cuda_stream_out(
    std::ostream& out,
    const c10::cuda::CUDAStream* stream) {
  if (Vlog()) LOG_INFO();
  return owner_table().stream_out(out, *stream);
}

extern "C" CUDAAllocatorType* fastpt_cuda_caching_allocator_get() {
  if (Vlog()) LOG_INFO();
  CUDAAllocatorType* alloc = owner_table().allocator_atomic.load(std::memory_order_acquire);
  // return owner_table().allocator_atomic.load(std::memory_order_acquire);
  if(nullptr == alloc){
    std::cout << "fastpt_cuda_caching_allocator_get: alloc is nullptr, please check if libc10_hip.so is loaded correctly" << std::endl;
  }
  return alloc;
}