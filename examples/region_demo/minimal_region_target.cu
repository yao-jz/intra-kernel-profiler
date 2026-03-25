#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "intra_kernel_profiler/nvbit/markers.cuh"

// A small kernel with two marked regions. This binary is intended to be used with:
// - NVBit region profiler (to generate pc2region_*.json)
// - CUPTI SASS metrics (to generate per-PC metrics)
// and then joined to get per-region metrics.
extern "C" __global__ void region_demo_kernel(float* out, int n, int inner) {
  const int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (idx >= n) return;

  float x = float(idx) * 0.001f;

  // Region 1: arithmetic-heavy loop.
  IKP_NVBIT_BEGIN(1);
#pragma unroll 1
  for (int i = 0; i < inner; ++i) {
    x = x * 1.00001f + 0.00001f;
  }
  IKP_NVBIT_END(1);

  // Region 2: write-out.
  IKP_NVBIT_BEGIN(2);
  out[idx] = x;
  IKP_NVBIT_END(2);
}

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", what, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
}

int main(int argc, char** argv) {
  int iters = 20;
  int inner = 4096;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!a) continue;
    if (std::strncmp(a, "--iters=", 7) == 0) {
      iters = std::atoi(a + 7);
    } else if (std::strncmp(a, "--inner=", 8) == 0) {
      inner = std::atoi(a + 8);
    }
  }
  if (iters < 1) iters = 1;
  if (inner < 1) inner = 1;

  constexpr int n = 1 << 20;
  float* d_out = nullptr;
  check(cudaMalloc(&d_out, sizeof(float) * n), "cudaMalloc");

  dim3 block(256);
  dim3 grid((n + int(block.x) - 1) / int(block.x));

  for (int it = 0; it < iters; ++it) {
    region_demo_kernel<<<grid, block>>>(d_out, n, inner);
    check(cudaGetLastError(), "kernel launch");
  }
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<float> h(n);
  check(cudaMemcpy(h.data(), d_out, sizeof(float) * n, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  std::printf("out[0]=%.6f out[n-1]=%.6f iters=%d inner=%d\n",
              double(h[0]), double(h[n - 1]), iters, inner);

  check(cudaFree(d_out), "cudaFree");
  return 0;
}

