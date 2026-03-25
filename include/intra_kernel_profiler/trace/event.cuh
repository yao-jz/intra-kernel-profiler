#pragma once

#include <cstdint>

namespace intra_kernel_profiler::trace {

// Pack (warp, smid) into 16 bits:
// - warp:  6 bits  (0..63)
// - smid: 10 bits  (0..1023)
//
// This is used to expose SMID in traces without growing the on-device event record.
__host__ __device__ __forceinline__ uint16_t pack_warp_smid(uint16_t warp, uint16_t smid) {
  return uint16_t((warp & 0x3Fu) | ((smid & 0x3FFu) << 6));
}
__host__ __device__ __forceinline__ uint16_t unpack_warp(uint16_t packed_warp_smid) {
  return uint16_t(packed_warp_smid & 0x3Fu);
}
__host__ __device__ __forceinline__ uint16_t unpack_smid(uint16_t packed_warp_smid) {
  return uint16_t(packed_warp_smid >> 6);
}

// Event is written via a 16B vector store (st.global.*.v4.u32). Keep object size AND alignment at 16B.
struct alignas(16) Event {
  uint64_t ts;      // globaltimer (raw)
  uint16_t id;      // region id
  uint16_t type;    // 0=begin, 1=end, 2=instant(mark)
  uint16_t block;   // CTA id (blockIdx.x)
  uint16_t warp;    // packed (warp_in_cta, smid)
};
static_assert(sizeof(Event) == 16, "Event must be exactly 16 bytes");
static_assert(alignof(Event) == 16, "Event must be 16-byte aligned");

struct GlobalBuffer {
  Event* events = nullptr;
  uint32_t* counters = nullptr;  // per-(block,warp) final count (for circular buffer)
};

}  // namespace intra_kernel_profiler::trace

