#pragma once

// Public entry for the intra-kernel timeline trace recorder:
// - device: warp-leader timestamped begin/end/mark events into per-warp circular buffers
// - host:   reconstruct ordered events, emit Chrome Trace JSON + summary JSON

#include "intra_kernel_profiler/trace/device_ctx.cuh"
#include "intra_kernel_profiler/trace/event.cuh"
#include "intra_kernel_profiler/trace/host_session.hpp"
#include "intra_kernel_profiler/trace/macros.cuh"

