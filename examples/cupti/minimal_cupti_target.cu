#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void cupti_target_kernel(float* out, int n, int inner) {
  const int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (idx >= n) return;
  float x = float(idx) * 0.001f;
#pragma unroll 1
  for (int i = 0; i < inner; ++i) {
    x = x * 1.00001f + 0.00001f;
  }
  out[idx] = x;
}

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", what, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
}

int main(int argc, char** argv) {
  int iters = 50;
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
    cupti_target_kernel<<<grid, block>>>(d_out, n, inner);
    check(cudaGetLastError(), "kernel launch");
  }
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<float> h(n);
  check(cudaMemcpy(h.data(), d_out, sizeof(float) * n, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  std::printf("out[0]=%.6f out[n-1]=%.6f\n", double(h[0]), double(h[n - 1]));

  check(cudaFree(d_out), "cudaFree");
  return 0;
}

