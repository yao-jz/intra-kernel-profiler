#pragma once

#include <stdint.h>

// NVBit region markers.
//
// IMPORTANT (performance):
// These markers compile into extern device calls (CALL), which can force ptxas to
// serialize some pipelines (notably WGMMA). Therefore they are disabled by default and
// should only be enabled for NVBit runs by defining IKP_ENABLE_NVBIT_MARKERS and linking
// `intra_kernel_profiler/src/ikp_nvbit_marker_device.cu` into the binary (usually with -rdc=true).
//
// Normal (no-NVBit) builds should NOT define IKP_ENABLE_NVBIT_MARKERS.

#if defined(IKP_ENABLE_NVBIT_MARKERS)

// Marker functions are defined in a separate TU (`ikp_nvbit_marker_device.cu`) so nvcc
// does not internalize them as local subroutines inside kernels.
//
// NOTE: region 0 is reserved as "unmarked/root" (same as empty stack).
extern "C" __device__ __noinline__ void ikp_nvbit_region_push_0();
extern "C" __device__ __noinline__ void ikp_nvbit_region_pop_0();
extern "C" __device__ __noinline__ void ikp_nvbit_region_push_1();
extern "C" __device__ __noinline__ void ikp_nvbit_region_pop_1();
extern "C" __device__ __noinline__ void ikp_nvbit_region_push_2();
extern "C" __device__ __noinline__ void ikp_nvbit_region_pop_2();
extern "C" __device__ __noinline__ void ikp_nvbit_region_push_3();
extern "C" __device__ __noinline__ void ikp_nvbit_region_pop_3();
extern "C" __device__ __noinline__ void ikp_nvbit_region_push_4();
extern "C" __device__ __noinline__ void ikp_nvbit_region_pop_4();
extern "C" __device__ __noinline__ void ikp_nvbit_region_push_5();
extern "C" __device__ __noinline__ void ikp_nvbit_region_pop_5();
extern "C" __device__ __noinline__ void ikp_nvbit_region_push_6();
extern "C" __device__ __noinline__ void ikp_nvbit_region_pop_6();

#define IKP_NVBIT_LANEID() \
  (static_cast<uint32_t>(((((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x) & 31u)))

#define IKP_NVBIT_LEADER_LANE() (__ffs((int)__activemask()) - 1)

#define IKP_NVBIT_BEGIN(id) \
  do { \
    if ((int)IKP_NVBIT_LANEID() == IKP_NVBIT_LEADER_LANE()) { \
      switch (static_cast<uint32_t>(id)) { \
        case 0: break; \
        case 1: ikp_nvbit_region_push_1(); break; \
        case 2: ikp_nvbit_region_push_2(); break; \
        case 3: ikp_nvbit_region_push_3(); break; \
        case 4: ikp_nvbit_region_push_4(); break; \
        case 5: ikp_nvbit_region_push_5(); break; \
        case 6: ikp_nvbit_region_push_6(); break; \
        default: break; \
      } \
    } \
  } while (0)

#define IKP_NVBIT_END(id) \
  do { \
    if ((int)IKP_NVBIT_LANEID() == IKP_NVBIT_LEADER_LANE()) { \
      switch (static_cast<uint32_t>(id)) { \
        case 0: break; \
        case 1: ikp_nvbit_region_pop_1(); break; \
        case 2: ikp_nvbit_region_pop_2(); break; \
        case 3: ikp_nvbit_region_pop_3(); break; \
        case 4: ikp_nvbit_region_pop_4(); break; \
        case 5: ikp_nvbit_region_pop_5(); break; \
        case 6: ikp_nvbit_region_pop_6(); break; \
        default: break; \
      } \
    } \
  } while (0)

#else

// Markers disabled: compile to nothing, avoiding extern calls.
#define IKP_NVBIT_BEGIN(id) do { (void)(id); } while (0)
#define IKP_NVBIT_END(id)   do { (void)(id); } while (0)

#endif

