#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "intra_kernel_profiler/nvbit/markers.cuh"

extern "C" __global__ void nvbit_marked_kernel(float* out, int n) {
  const int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (idx >= n) return;

  float x = float(idx) * 0.001f;

  IKP_NVBIT_BEGIN(1);
#pragma unroll 1
  for (int i = 0; i < 256; ++i) {
    x = x * 1.0001f + 0.00001f;
  }
  IKP_NVBIT_END(1);

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

int main() {
  constexpr int n = 1 << 20;
  float* d_out = nullptr;
  check(cudaMalloc(&d_out, sizeof(float) * n), "cudaMalloc");

  dim3 block(256);
  dim3 grid((n + int(block.x) - 1) / int(block.x));
  nvbit_marked_kernel<<<grid, block>>>(d_out, n);
  check(cudaGetLastError(), "kernel launch");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<float> h(n);
  check(cudaMemcpy(h.data(), d_out, sizeof(float) * n, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  std::printf("out[0]=%.6f out[n-1]=%.6f\n", double(h[0]), double(h[n - 1]));

  check(cudaFree(d_out), "cudaFree");
  return 0;
}

