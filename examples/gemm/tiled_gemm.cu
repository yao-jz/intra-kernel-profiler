// tiled_gemm.cu — Shared-memory tiled GEMM with comprehensive intra-kernel profiling.
//
// Instrumented with BOTH trace markers (timing) and NVBit region markers
// (instruction-level attribution), plus compilable with -lineinfo for CUPTI.
//
// Build (trace + CUPTI mode — NVBit markers are no-ops):
//   nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo -I ../../include tiled_gemm.cu -o tiled_gemm
//
// Build (NVBit mode — NVBit markers active):
//   nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo -rdc=true -DIKP_ENABLE_NVBIT_MARKERS \
//     -I ../../include tiled_gemm.cu ../../src/nvbit_marker_device.cu -o tiled_gemm_nvbit
//
// Run (trace):
//   ./tiled_gemm --m=2048 --n=2048 --k=2048 --out=gemm_trace.json
//
// Run (CUPTI):
//   CUDA_INJECTION64_PATH=ikp_cupti_sassmetrics.so ./tiled_gemm --iters=20
//
// Run (NVBit):
//   LD_PRELOAD=region_profiler.so ./tiled_gemm_nvbit --iters=1
//
// Profiling regions (5 regions, IDs 1-5):
//   1: total      — entire tile computation envelope
//   2: load_A     — global → shared: load A tile
//   3: load_B     — global → shared: load B tile
//   4: compute    — shared → registers: multiply-accumulate
//   5: store      — registers → global: write-back C

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include <intra_kernel_profiler/intra_kernel_profiler.hpp>
#include <intra_kernel_profiler/nvbit/markers.cuh>

// ---------------------------------------------------------------------------
// Tile configuration
// ---------------------------------------------------------------------------
constexpr int BM = 64;    // tile rows
constexpr int BN = 64;    // tile cols
constexpr int BK = 16;    // tile K-depth
constexpr int TM = 4;     // each thread computes TM rows
constexpr int TN = 4;     // each thread computes TN cols
// Thread block: (BN/TN) x (BM/TM) = 16x16 = 256 threads
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256

// Per-warp profiling ring-buffer capacity (must be power of 2).
#ifndef PROFILE_CAP
#define PROFILE_CAP 8192
#endif
static_assert((PROFILE_CAP & (PROFILE_CAP - 1)) == 0, "PROFILE_CAP must be power of 2");
constexpr uint32_t kWarpsPerBlock = (THREADS + 31) / 32;

// Profiling region IDs — each names a phase of the GEMM inner loop.
// NVBit reserves region 0 as "outside/unmarked", so we start from 1.
enum Region : uint16_t {
  kTotal   = 1,  // entire tile computation
  kLoadA   = 2,  // global → shared: load A tile
  kLoadB   = 3,  // global → shared: load B tile
  kCompute = 4,  // shared → registers: multiply-accumulate
  kStore   = 5,  // registers → global: write-back
};

// ---------------------------------------------------------------------------
// Kernel: C[M,N] += A[M,K] * B[K,N],  row-major, 1D grid
// ---------------------------------------------------------------------------
__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    int tiles_n,
    intra_kernel_profiler::trace::GlobalBuffer prof) {

  // ---- profiler context (per-warp ring buffer) ----------------------------
  IKP_TRACE_CTX_TYPE(PROFILE_CAP, kWarpsPerBlock) ctx;
  IKP_TRACE_CTX_INIT(ctx);

  // Linearize 1D blockIdx into 2D tile coords.
  const int tile_row = blockIdx.x / tiles_n;
  const int tile_col = blockIdx.x % tiles_n;

  // Thread position within the block: used for loading tiles.
  const int tx = threadIdx.x % (BN / TN);  // 0..15 (column index in thread grid)
  const int ty = threadIdx.x / (BN / TN);  // 0..15 (row index in thread grid)

  __shared__ float sA[BM][BK];
  __shared__ float sB[BK][BN];

  // Accumulator: each thread computes a TM x TN sub-tile.
  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  // Global row/col base for this thread's output sub-tile.
  const int row_base = tile_row * BM + ty * TM;
  const int col_base = tile_col * BN + tx * TN;

  IKP_NVBIT_BEGIN(kTotal);
  IKP_TRACE_REC_B(ctx, prof, kTotal);

  const int num_k_tiles = (K + BK - 1) / BK;
  for (int t = 0; t < num_k_tiles; ++t) {

    // ---- Load A tile into shared memory -----------------------------------
    IKP_NVBIT_BEGIN(kLoadA);
    IKP_TRACE_REC_B(ctx, prof, kLoadA);

    // A tile: BM x BK = 64 x 16 = 1024 floats, 256 threads → 4 loads each
    #pragma unroll
    for (int load = 0; load < (BM * BK) / THREADS; ++load) {
      int linear = load * THREADS + threadIdx.x;
      int sr = linear / BK;
      int sc = linear % BK;
      int gr = tile_row * BM + sr;
      int gc = t * BK + sc;
      sA[sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }

    IKP_TRACE_REC_E(ctx, prof, kLoadA);
    IKP_NVBIT_END(kLoadA);

    // ---- Load B tile into shared memory -----------------------------------
    IKP_NVBIT_BEGIN(kLoadB);
    IKP_TRACE_REC_B(ctx, prof, kLoadB);

    // B tile: BK x BN = 16 x 64 = 1024 floats, 256 threads → 4 loads each
    #pragma unroll
    for (int load = 0; load < (BK * BN) / THREADS; ++load) {
      int linear = load * THREADS + threadIdx.x;
      int sr = linear / BN;
      int sc = linear % BN;
      int gr = t * BK + sr;
      int gc = tile_col * BN + sc;
      sB[sr][sc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }

    IKP_TRACE_REC_E(ctx, prof, kLoadB);
    IKP_NVBIT_END(kLoadB);

    __syncthreads();

    // ---- Compute: multiply-accumulate from shared memory ------------------
    IKP_NVBIT_BEGIN(kCompute);
    IKP_TRACE_REC_B(ctx, prof, kCompute);
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += sA[ty * TM + i][k] * sB[k][tx * TN + j];
        }
      }
    }
    IKP_TRACE_REC_E(ctx, prof, kCompute);
    IKP_NVBIT_END(kCompute);

    __syncthreads();
  }

  // ---- Store result to global memory --------------------------------------
  IKP_NVBIT_BEGIN(kStore);
  IKP_TRACE_REC_B(ctx, prof, kStore);
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      int r = row_base + i;
      int c = col_base + j;
      if (r < M && c < N)
        C[r * N + c] = acc[i][j];
    }
  }
  IKP_TRACE_REC_E(ctx, prof, kStore);
  IKP_NVBIT_END(kStore);

  IKP_TRACE_REC_E(ctx, prof, kTotal);
  IKP_NVBIT_END(kTotal);

  // ---- Flush profiler ring buffer -----------------------------------------
  IKP_TRACE_CTX_FLUSH(ctx, prof);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); std::exit(1); }
}
static int get_int(int argc, char** argv, const char* key, int def) {
  std::string pfx = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], pfx.c_str(), pfx.size()) == 0)
      return std::atoi(argv[i] + pfx.size());
  return def;
}
static const char* get_str(int argc, char** argv, const char* key, const char* def) {
  std::string pfx = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], pfx.c_str(), pfx.size()) == 0)
      return argv[i] + pfx.size();
  return def;
}

int main(int argc, char** argv) {
  const int M     = get_int(argc, argv, "m", 1024);
  const int N     = get_int(argc, argv, "n", 1024);
  const int K     = get_int(argc, argv, "k", 1024);
  const int iters = get_int(argc, argv, "iters", 3);
  const char* out = get_str(argc, argv, "out", "gemm_trace.json");

  const int tiles_m = (M + BM - 1) / BM;
  const int tiles_n = (N + BN - 1) / BN;
  const int total_blocks = tiles_m * tiles_n;

  std::printf("=== Intra-Kernel Profiler: Tiled GEMM Demo ===\n");
  std::printf("Problem:  M=%d N=%d K=%d\n", M, N, K);
  std::printf("Tiling:   BM=%d BN=%d BK=%d  TM=%d TN=%d  threads=%d\n",
              BM, BN, BK, TM, TN, THREADS);
  std::printf("Grid:     %d blocks (%d x %d tiles)\n", total_blocks, tiles_m, tiles_n);

  // Allocate
  size_t szA = size_t(M) * K, szB = size_t(K) * N, szC = size_t(M) * N;
  std::vector<float> hA(szA), hB(szB), hC(szC, 0.f);
  for (size_t i = 0; i < szA; ++i) hA[i] = float((i * 17) % 101) / 101.f;
  for (size_t i = 0; i < szB; ++i) hB[i] = float((i * 29) % 103) / 103.f;

  float *dA, *dB, *dC;
  ck(cudaMalloc(&dA, szA * sizeof(float)), "malloc A");
  ck(cudaMalloc(&dB, szB * sizeof(float)), "malloc B");
  ck(cudaMalloc(&dC, szC * sizeof(float)), "malloc C");
  ck(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice), "cpy A");
  ck(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice), "cpy B");

  dim3 grid(total_blocks);
  dim3 block(THREADS);

  // ---- Set up profiler ---------------------------------------------------
  intra_kernel_profiler::trace::HostSession sess;
  sess.set_region_names({"_outside", "total", "load_A", "load_B", "compute", "store"});
  // Trace only the first few blocks to keep output manageable.
  sess.set_block_filter({0, 1, 2, 3});
  sess.init(PROFILE_CAP, total_blocks, THREADS);

  std::printf("Profiling: blocks 0-3 (of %d), cap=%d events/warp\n\n", total_blocks, PROFILE_CAP);

  // Warmup
  {
    intra_kernel_profiler::trace::GlobalBuffer null_prof{};
    tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K, tiles_n, null_prof);
    ck(cudaDeviceSynchronize(), "warmup");
  }

  // Profiled run
  ck(cudaMemset(dC, 0, szC * sizeof(float)), "clear C");
  sess.reset();
  tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K, tiles_n, sess.global_buffer());
  ck(cudaDeviceSynchronize(), "profile sync");

  intra_kernel_profiler::trace::TraceWriteOptions opt;
  opt.scale               = 1.0;   // globaltimer ≈ ns
  opt.group_by_smid       = true;
  opt.emit_summary_json   = true;
  opt.emit_complete_events = true;
  sess.write_trace(out, opt);

  // Benchmark (profiling disabled)
  cudaEvent_t ev0, ev1;
  ck(cudaEventCreate(&ev0), "ev0");
  ck(cudaEventCreate(&ev1), "ev1");
  ck(cudaEventRecord(ev0), "rec0");
  for (int i = 0; i < iters; ++i) {
    intra_kernel_profiler::trace::GlobalBuffer null_prof{};
    tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K, tiles_n, null_prof);
  }
  ck(cudaEventRecord(ev1), "rec1");
  ck(cudaEventSynchronize(ev1), "sync");
  float ms = 0;
  ck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
  double gflops = 2.0 * M * N * K * iters / (ms * 1e6);
  std::printf("Benchmark: %d iters, %.3f ms total, %.3f ms/iter, %.1f GFLOP/s\n",
              iters, ms, ms / iters, gflops);

  // Quick verify
  ck(cudaMemcpy(hC.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost), "cpy C");
  double ref = 0;
  for (int k = 0; k < K; ++k) ref += double(hA[k]) * double(hB[k * N]);
  double err = std::fabs(double(hC[0]) - ref);
  std::printf("Verify:    C[0,0]=%.4f ref=%.4f err=%.2e %s\n",
              hC[0], ref, err, (err < 0.1) ? "PASS" : "FAIL");

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cudaEventDestroy(ev0); cudaEventDestroy(ev1);
  return 0;
}
