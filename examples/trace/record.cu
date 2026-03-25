#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <intra_kernel_profiler/intra_kernel_profiler.hpp>

#ifndef PROFILE_PER_WARP_CAP
#define PROFILE_PER_WARP_CAP 32768
#endif
static_assert((PROFILE_PER_WARP_CAP & (PROFILE_PER_WARP_CAP - 1u)) == 0u,
              "PROFILE_PER_WARP_CAP must be power of 2");

constexpr int kBlockThreads = 32;
constexpr uint32_t kWarpsPerBlock = (kBlockThreads + 31) / 32;

__global__ void RecordKernel(int iters, intra_kernel_profiler::trace::GlobalBuffer prof) {
  IKP_TRACE_CTX_TYPE(PROFILE_PER_WARP_CAP, kWarpsPerBlock) ctx;
  IKP_TRACE_CTX_INIT(ctx);

  IKP_TRACE_REC_B(ctx, prof, 1);
  for (int i = 0; i < iters; ++i) {
    IKP_TRACE_REC_B(ctx, prof, 0);
    IKP_TRACE_REC_E(ctx, prof, 0);
  }
  IKP_TRACE_REC_E(ctx, prof, 1);

  IKP_TRACE_CTX_FLUSH(ctx, prof);
}

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  int iters = 100000;
  std::string out_path = "record_kernel_trace.json";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!a) continue;
    if (std::strncmp(a, "--iters=", 8) == 0) {
      iters = std::atoi(a + 8);
    } else if (std::strncmp(a, "--out=", 6) == 0) {
      out_path = std::string(a + 6);
    } else if (i == 1 && a[0] != '-') {
      // Backward compatible positional: ./ikp_trace_record <iters>
      iters = std::atoi(a);
    }
  }
  if (iters < 0) iters = 0;

  const int grid = 1;
  dim3 block(kBlockThreads, 1, 1);
  dim3 grid_dim(grid, 1, 1);

  intra_kernel_profiler::trace::HostSession sess;
  sess.set_region_names({"loop_body", "kernel_total"});
  sess.init(uint32_t(PROFILE_PER_WARP_CAP), uint32_t(grid_dim.x), uint32_t(block.x));
  sess.reset();

  const uint32_t expected_events_per_warp = 2u + 2u * uint32_t(iters);
  std::printf("record: cap=%u grid=%u block=%u warps=%u iters=%d expected_events_per_warp=%u\n",
              uint32_t(PROFILE_PER_WARP_CAP), uint32_t(grid_dim.x), uint32_t(block.x),
              kWarpsPerBlock, iters, expected_events_per_warp);

  RecordKernel<<<grid_dim, block>>>(iters, sess.global_buffer());
  ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  intra_kernel_profiler::trace::TraceWriteOptions opt;
  opt.scale = 1e-3;
  sess.write_trace(out_path, opt);

  return 0;
}

