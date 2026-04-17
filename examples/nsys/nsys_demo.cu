// nsys_demo.cu — Minimal tiled GEMM for NSys + IKP trace demo.
//
// Single kernel launch, no warmup, no benchmark — produces a clean
// trace where every NSys event maps directly to the IKP-profiled kernel.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo -I ../../include \
//     nsys_demo.cu -o nsys_demo
//
// Run (under nsys):
//   nsys profile --trace=cuda ./nsys_demo --out=trace.json

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <intra_kernel_profiler/intra_kernel_profiler.hpp>

constexpr int BM = 64, BN = 64, BK = 16;
constexpr int TM = 4,  TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);
constexpr uint32_t kWarpsPerBlock = (THREADS + 31) / 32;
constexpr uint32_t PROFILE_CAP = 4096;

enum Region : uint16_t {
    kTotal   = 1,
    kLoadA   = 2,
    kLoadB   = 3,
    kCompute = 4,
    kStore   = 5,
};

__global__ void gemm_kernel(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K, int tiles_n,
    intra_kernel_profiler::trace::GlobalBuffer prof)
{
    IKP_TRACE_CTX_TYPE(PROFILE_CAP, kWarpsPerBlock) ctx;
    IKP_TRACE_CTX_INIT(ctx);

    const int bx = blockIdx.x % tiles_n, by = blockIdx.x / tiles_n;
    const int row0 = by * BM, col0 = bx * BN;
    const int tx = threadIdx.x % (BN / TN), ty = threadIdx.x / (BN / TN);

    __shared__ float sA[BM][BK], sB[BK][BN];
    float acc[TM][TN] = {};

    IKP_TRACE_REC_B(ctx, prof, kTotal);

    for (int t = 0; t < (K + BK - 1) / BK; ++t) {
        IKP_TRACE_REC_B(ctx, prof, kLoadA);
        for (int i = threadIdx.x; i < BM * BK; i += THREADS) {
            int r = i / BK, c = i % BK;
            int gr = row0 + r, gc = t * BK + c;
            sA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        IKP_TRACE_REC_E(ctx, prof, kLoadA);

        IKP_TRACE_REC_B(ctx, prof, kLoadB);
        for (int i = threadIdx.x; i < BK * BN; i += THREADS) {
            int r = i / BN, c = i % BN;
            int gr = t * BK + r, gc = col0 + c;
            sB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        IKP_TRACE_REC_E(ctx, prof, kLoadB);

        __syncthreads();

        IKP_TRACE_REC_B(ctx, prof, kCompute);
        for (int k = 0; k < BK; ++k)
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += sA[ty * TM + i][k] * sB[k][tx * TN + j];
        IKP_TRACE_REC_E(ctx, prof, kCompute);
        __syncthreads();
    }

    IKP_TRACE_REC_B(ctx, prof, kStore);
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int gr = row0 + ty * TM + i, gc = col0 + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
    IKP_TRACE_REC_E(ctx, prof, kStore);

    IKP_TRACE_REC_E(ctx, prof, kTotal);
    IKP_TRACE_CTX_FLUSH(ctx, prof);
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
    const int M = get_int(argc, argv, "m", 2048);
    const int N = get_int(argc, argv, "n", 2048);
    const int K = get_int(argc, argv, "k", 2048);
    const char* out = get_str(argc, argv, "out", "trace.json");

    const int tiles_m = (M + BM - 1) / BM;
    const int tiles_n = (N + BN - 1) / BN;
    const int nblocks = tiles_m * tiles_n;

    // Allocate & init
    size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
    std::vector<float> hA(szA), hB(szB);
    for (size_t i = 0; i < szA; ++i) hA[i] = float(i % 101) / 101.f;
    for (size_t i = 0; i < szB; ++i) hB[i] = float(i % 103) / 103.f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA * sizeof(float));
    cudaMalloc(&dB, szB * sizeof(float));
    cudaMalloc(&dC, szC * sizeof(float));
    cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, szC * sizeof(float));

    // IKP profiler
    intra_kernel_profiler::trace::HostSession sess;
    sess.set_region_names({"_outside", "total", "load_A", "load_B", "compute", "store"});
    sess.set_block_filter({0u, 1u, 2u, 3u});
    sess.init(PROFILE_CAP, nblocks, THREADS);

    // Single profiled kernel launch — this is what NSys and IKP both capture.
    gemm_kernel<<<nblocks, THREADS>>>(dA, dB, dC, M, N, K, tiles_n, sess.global_buffer());
    cudaDeviceSynchronize();

    // Write IKP trace
    intra_kernel_profiler::trace::TraceWriteOptions opt;
    opt.emit_summary_json = true;
    sess.write_trace(out, opt);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    printf("Done: %s\n", out);
    return 0;
}
