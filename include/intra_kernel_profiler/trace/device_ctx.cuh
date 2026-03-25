#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include "intra_kernel_profiler/trace/event.cuh"

namespace intra_kernel_profiler::trace {

// Per-warp circular buffer recorder. Intended usage: only the warp leader (lane 0)
// calls init/record/flush to keep overhead low.
//
// CAP must be a power of two. If a warp records more than CAP events, only the last CAP are kept.
template <uint32_t CAP, uint32_t WARPS_PER_BLOCK>
struct WarpContext {
  static_assert((CAP & (CAP - 1u)) == 0u, "CAP must be power of 2");
  static constexpr uint32_t kMask = CAP - 1u;

  uint32_t base = 0;  // slot * CAP
  uint32_t cnt = 0;   // local counter
  uint32_t slot = 0;  // slot index
  uint32_t info = 0;  // packed: block(lo16) | (warp|smid)(hi16)

  __device__ __forceinline__ void init() {
    const uint32_t w = uint32_t(threadIdx.x >> 5);
    const uint32_t b = uint32_t(blockIdx.x);
#if defined(__CUDA_ARCH__)
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
#else
    uint32_t sm = 0;
#endif
    slot = b * WARPS_PER_BLOCK + w;
    base = slot * CAP;
    cnt = 0;
    const uint16_t packed = pack_warp_smid(uint16_t(w), uint16_t(sm));
    info = b | (uint32_t(packed) << 16);
  }

  // Streaming store (bypass caches), branch-free except for "profiling disabled".
  __device__ __forceinline__ void record(GlobalBuffer& buf, uint16_t id, uint16_t type) {
    if (!buf.events) return;
    uint64_t ts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
    const uint32_t idx = base + (cnt & kMask);
    const uint32_t w0 = uint32_t(ts);
    const uint32_t w1 = uint32_t(ts >> 32);
    const uint32_t w2 = uint32_t(id) | (uint32_t(type) << 16);
    const uint32_t w3 = info;
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(&buf.events[idx]), "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                 : "memory");
    ++cnt;
  }

  __device__ __forceinline__ void flush(GlobalBuffer& buf) {
    if (!buf.counters) return;
    buf.counters[slot] = cnt;
  }
};

}  // namespace intra_kernel_profiler::trace

