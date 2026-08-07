#include "hip/hip_runtime.h"
#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <utils/mocker_logging.h>
#include <array>
#include <atomic>
#include <cstdint>

namespace c10::cuda {

namespace {

// Global stream state and constants
DeviceIndex num_gpus = -1;
constexpr int kStreamsPerPoolBits = 5;
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;
constexpr int kStreamTypeBits = 4;

int max_stream_priorities;

// Non-default streams
// Note: the number of CUDA devices is determined at run time,
// and the low and high priority pools are lazily initialized
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next stream
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in CUDAStream.h).
// The streams are "leaked": they are created but never destroyed because the
// destruction of global variables could happen after the CUDA runtime has
// already been destroyed and thus invoking cudaStreamDestroy could lead to a
// crash. It's likely an issue in CUDA, but to be safe - let's just "forget"
// the destruction.
#if !defined(USE_ROCM)
// CUDA-only: used to initializes the stream pools (once)
std::array<c10::once_flag, C10_COMPILE_TIME_MAX_GPUS> device_flags;
#endif
std::array<
    std::array<std::atomic<uint32_t>, C10_COMPILE_TIME_MAX_GPUS>,
    c10::cuda::max_compile_time_stream_priorities>
    priority_counters;

std::array<
    std::array<
        std::array<cudaStream_t, kStreamsPerPool>,
        C10_COMPILE_TIME_MAX_GPUS>,
    c10::cuda::max_compile_time_stream_priorities>
    streams;
#ifdef USE_ROCM
static c10::once_flag
    stream_flags[c10::cuda::max_compile_time_stream_priorities]
                [C10_COMPILE_TIME_MAX_GPUS][kStreamsPerPool];
#endif

// Note [HIP Lazy Streams]
// ~~~~~~~~~~~~~~~~~~~~~~~
// For ROCm/HIP, each stream is lazily initialized rather than creating all
// streams when the first stream is requested. HIP streams are not as
// lightweight as CUDA streams; the pooling strategy can affect performance.
// Rather than changing the pooling implementation, ROCm/HIP will lazy init
// each stream when it is first requested.

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
// zeros          stream id index  StreamIdType     Ext/native stream
//                ignored for ext   ignored for ext
// for external stream, StreamID is a cudaStream_t pointer
// this means that last bit will always be 0
// so when constructing StreamId for a native stream we set last bit to 1
// to distinguish between native and external streams
//
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream, so this is one exception to
// "last bit = 1 for native streams". However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
//
// Note that it is really important that the MSB is zero; StreamId is a
// *signed* integer, and unsigned to signed conversion outside of the
// bounds of signed integer representation is undefined behavior.  You
// could work around this with something like
// https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
// but it seems a bit overkill for this.
//
// Also, external managed stream pointers (cudaStream_t) can be directly stored
// in the Id field so in this case, we need to check the stream alignment.

class StreamIdType {
  // StreamIdType encodes whether this stream is DEFAULT, EXTernal or
  // for all other native streams, the stream priority (higher value is higher
  // priority)
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

 public:
  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {
    if (Vlog()) LOG_INFO();
  }

  bool isExt() const {
    if (Vlog()) LOG_INFO();
    return EXT == stream_type;
  }

  bool isDefault() const {
    if (Vlog()) LOG_INFO();
    return DEFAULT == stream_type;
  }

  uint8_t getStreamType() const {
    if (Vlog()) LOG_INFO();
    return stream_type;
  }
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (Vlog()) LOG_INFO();
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << static_cast<int>(s.getStreamType());
  }
  return stream;
}

// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on streamIdIndex and streamIdType being non-negative;
// see Note [Hazard when concatenating signed integers]

inline StreamIdType streamIdType(StreamId s) {
  if (Vlog()) LOG_INFO();
  // Externally allocated streams have their id being the cudaStream_ptr
  // so the last bit will be 0
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }
  // last bit is external/internal stream, the mask should start from second
  // rightmost bit
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto val = (s >> 1) & mask_for_type;
  TORCH_CHECK(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

inline size_t streamIdIndex(StreamId s) {
  if (Vlog()) LOG_INFO();
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  if (Vlog()) LOG_INFO();
  if (st.isDefault()) {
    return static_cast<StreamId>(0);
  }
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      static_cast<StreamId>(st.getStreamType() << 1) | 1;
}

// Thread-local current streams
// NOLINTNEXTLINE(*-arrays)
thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// Populates global values.
// Warning: this function must only be called once!
void initGlobalStreamState() {
  if (Vlog()) LOG_INFO();
  num_gpus = device_count();
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  TORCH_CHECK(
      num_gpus <= C10_COMPILE_TIME_MAX_GPUS,
      "Number of CUDA devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      C10_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile.");
  int leastPriority = -1, greatestPriority = -1;
  C10_CUDA_CHECK(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  // Note [HIP stream priorities]
  // HIP stream priorities are 1=low, 0=default, -1=high which differs from CUDA
  // which is 0=default, -1=high, -2=higher etc.
  // Clamp leastPriority to 0 for HIP.
#ifdef USE_ROCM
  leastPriority = 0;
#endif
  // greatestPriority is negative
  auto range = leastPriority - greatestPriority + 1;
  max_stream_priorities = range >= c10::cuda::max_compile_time_stream_priorities
      ? c10::cuda::max_compile_time_stream_priorities
      : range;
}

// Init a single CUDA or HIP stream
// See Note [HIP Lazy Streams]
void initSingleStream(int p, DeviceIndex device_index, int i) {
  if (Vlog()) LOG_INFO();
  CUDAGuard device_guard(device_index);
  auto& stream = streams[p][device_index][i];
  auto pri = -p; // lower number is higher priority

  C10_CUDA_CHECK(cudaStreamCreateWithPriority(&stream, kDefaultFlags, pri));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_creation(
        c10::kCUDA, reinterpret_cast<uintptr_t>(stream));
    priority_counters[p][device_index] = 0;
  }
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
void initDeviceStreamState(DeviceIndex device_index) {
  if (Vlog()) LOG_INFO();
  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_stream_priorities)) {
      initSingleStream(p, device_index, i);
    }
  }
}

// Init front-end to ensure initialization only occurs once
void initCUDAStreamsOnce() {
  if (Vlog()) LOG_INFO();
  // Inits default streams (once, globally)
  auto static init_flag [[maybe_unused]] = [] {
    initGlobalStreamState();
    return true;
  }();

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  // NOLINTNEXTLINE(*-arrays)
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

// Helper to verify the GPU index is valid
inline void check_gpu(DeviceIndex device_index) {
  if (Vlog()) LOG_INFO();
  TORCH_CHECK(
      device_index >= 0 && device_index < num_gpus,
      "Device index value ",
      static_cast<int>(device_index),
      " is out of index range [0, ",
      static_cast<int>(num_gpus),
      ")");
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in CUDAStream.h)
uint32_t get_idx(std::atomic<uint32_t>& counter) {
  if (Vlog()) LOG_INFO();
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

CUDAStream CUDAStreamForId(DeviceIndex device_index, StreamId stream_id) {
  if (Vlog()) LOG_INFO();
  return CUDAStream(
      CUDAStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::CUDA, device_index),
          stream_id));
}

} // anonymous namespace

extern "C" cudaStream_t fastpt_cuda_stream_stream(const CUDAStream* stream);
extern "C" bool fastpt_cuda_stream_query(const CUDAStream* stream);
extern "C" void fastpt_cuda_stream_synchronize(const CUDAStream* stream);
extern "C" CUDAStream fastpt_cuda_get_stream_from_pool_int(int priority, DeviceIndex device);
extern "C" CUDAStream fastpt_cuda_get_stream_from_pool_bool(bool is_high_priority, DeviceIndex device);
extern "C" CUDAStream fastpt_cuda_get_stream_from_external(cudaStream_t stream, DeviceIndex device);
extern "C" CUDAStream fastpt_cuda_get_default_stream(DeviceIndex device);
extern "C" CUDAStream fastpt_cuda_get_current_stream(DeviceIndex device);
extern "C" void fastpt_cuda_set_current_stream(CUDAStream stream);
extern "C" std::ostream& fastpt_cuda_stream_out(std::ostream& out, const CUDAStream* stream);

bool CUDAStream::query() const {
  if (Vlog()) LOG_INFO();
  return fastpt_cuda_stream_query(this);
}

void CUDAStream::synchronize() const {
  if (Vlog()) LOG_INFO();
  fastpt_cuda_stream_synchronize(this);
}

// See Note [StreamId assignment]
cudaStream_t CUDAStream::stream() const {
  if (Vlog()) LOG_INFO();
  return fastpt_cuda_stream_stream(this);
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
CUDAStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  if (Vlog()) LOG_INFO();
  return fastpt_cuda_get_stream_from_pool_int(priority, device_index);
}

CUDAStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  if (Vlog()) LOG_INFO();
  return fastpt_cuda_get_stream_from_pool_bool(isHighPriority, device);
}

CUDAStream getStreamFromExternal(
    cudaStream_t ext_stream,
    DeviceIndex device_index) {
  if (Vlog()) LOG_INFO();
  return fastpt_cuda_get_stream_from_external(ext_stream, device_index);
}

CUDAStream getDefaultCUDAStream(DeviceIndex device_index) {
  if (Vlog()) LOG_INFO();
  return fastpt_cuda_get_default_stream(device_index);
}

CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  if (Vlog()) LOG_INFO();
  return fastpt_cuda_get_current_stream(device_index);
}

void setCurrentCUDAStream(CUDAStream stream) {
  if (Vlog()) LOG_INFO();
  fastpt_cuda_set_current_stream(stream);
}

std::ostream& operator<<(std::ostream& stream, const CUDAStream& s) {
  if (Vlog()) LOG_INFO();
  return fastpt_cuda_stream_out(stream, &s);
}

} // namespace c10::cuda
