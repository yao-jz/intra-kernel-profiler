#pragma once

#include "intra_kernel_profiler/trace/device_ctx.cuh"
#include "intra_kernel_profiler/trace/event.cuh"

// Device-side convenience macros.
//
// Design:
// - Only warp leader (lane 0) records to keep overhead low.
// - Use IKP_* prefix to avoid macro name collisions.

#define IKP_TRACE_CTX_TYPE(cap, warps_per_block) ::intra_kernel_profiler::trace::WarpContext<(cap), (warps_per_block)>

#define IKP_TRACE_CTX_INIT(ctx) \
  do { \
    if ((threadIdx.x & 31) == 0) { \
      (ctx).init(); \
    } \
  } while (0)

#define IKP_TRACE_CTX_FLUSH(ctx, gbuf) \
  do { \
    if ((threadIdx.x & 31) == 0) { \
      (ctx).flush((gbuf)); \
    } \
  } while (0)

#define IKP_TRACE_REC(ctx, gbuf, id, type) \
  do { \
    if ((threadIdx.x & 31) == 0) { \
      (ctx).record((gbuf), (uint16_t)(id), (uint16_t)(type)); \
    } \
  } while (0)

#define IKP_TRACE_REC_B(ctx, gbuf, id) IKP_TRACE_REC((ctx), (gbuf), (id), 0)
#define IKP_TRACE_REC_E(ctx, gbuf, id) IKP_TRACE_REC((ctx), (gbuf), (id), 1)
#define IKP_TRACE_REC_M(ctx, gbuf, id) IKP_TRACE_REC((ctx), (gbuf), (id), 2)

#define IKP_TRACE_REC_IF(ctx, gbuf, id, type, cond) \
  do { \
    if ((threadIdx.x & 31) == 0 && (cond)) { \
      (ctx).record((gbuf), (uint16_t)(id), (uint16_t)(type)); \
    } \
  } while (0)


