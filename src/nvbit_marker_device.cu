#include <stdint.h>

// Keep these as independent device functions (separate TU) so nvcc does not internalize them
// as local subroutines inside the kernel. NVBit relies on these being visible as CUfunctions.

#define IKP_DEF_MARKER(ID) \
  extern "C" __device__ __noinline__ __attribute__((used)) \
  void ikp_nvbit_region_push_##ID() { \
    uint32_t tmp = 0; \
    asm volatile("mov.u32 %0, %0;" : "+r"(tmp)); \
  } \
  extern "C" __device__ __noinline__ __attribute__((used)) \
  void ikp_nvbit_region_pop_##ID() { \
    uint32_t tmp = 0; \
    asm volatile("mov.u32 %0, %0;" : "+r"(tmp)); \
  }

IKP_DEF_MARKER(0)
IKP_DEF_MARKER(1)
IKP_DEF_MARKER(2)
IKP_DEF_MARKER(3)
IKP_DEF_MARKER(4)
IKP_DEF_MARKER(5)
IKP_DEF_MARKER(6)

#undef IKP_DEF_MARKER

