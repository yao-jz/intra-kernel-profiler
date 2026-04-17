// nccl_demo.cu — Minimal multi-GPU NCCL demo for NSys profiling.
//
// Runs AllReduce, AllGather, and ReduceScatter on 2+ GPUs so that
// NSys captures each collective type.  No complex compute — the
// goal is to show NCCL operations in a merged IKP + NSys trace.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_90a \
//     -I $NCCL_HOME/include -L $NCCL_HOME/lib -lnccl \
//     nccl_demo.cu -o nccl_demo
//
// Run:
//   CUDA_VISIBLE_DEVICES=0,1 ./nccl_demo

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>

#define CK_CUDA(e) do { if ((e) != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CK_NCCL(e) do { if ((e) != ncclSuccess) { \
    fprintf(stderr, "NCCL %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); exit(1); } } while(0)

int main(int argc, char** argv) {
    int ngpus = 2;
    if (argc > 1) ngpus = atoi(argv[1]);

    int dev_count = 0;
    CK_CUDA(cudaGetDeviceCount(&dev_count));
    if (ngpus > dev_count) { fprintf(stderr, "Only %d GPUs available\n", dev_count); return 1; }

    const size_t N = 1 << 20;  // 1M floats = 4 MB per GPU

    printf("NCCL demo: %d GPUs, %zu floats (%.1f MB) per GPU\n", ngpus, N, N * 4.0 / 1e6);

    // Init NCCL
    std::vector<int> devs(ngpus);
    for (int i = 0; i < ngpus; i++) devs[i] = i;
    std::vector<ncclComm_t> comms(ngpus);
    CK_NCCL(ncclCommInitAll(comms.data(), ngpus, devs.data()));

    // Allocate per-GPU buffers
    struct Gpu { float *send, *recv; cudaStream_t stream; };
    std::vector<Gpu> gpus(ngpus);
    for (int g = 0; g < ngpus; g++) {
        CK_CUDA(cudaSetDevice(g));
        CK_CUDA(cudaMalloc(&gpus[g].send, N * sizeof(float)));
        CK_CUDA(cudaMalloc(&gpus[g].recv, N * ngpus * sizeof(float)));
        CK_CUDA(cudaMemset(gpus[g].send, 1, N * sizeof(float)));
        CK_CUDA(cudaStreamCreate(&gpus[g].stream));
    }

    // Run each collective type (all GPUs must participate together)
    auto run_all = [&](const char* name, auto fn) {
        printf("  %-20s", name);
        std::vector<std::thread> threads;
        for (int g = 0; g < ngpus; g++) {
            threads.emplace_back([&, g]() {
                CK_CUDA(cudaSetDevice(g));
                fn(g);
                CK_CUDA(cudaStreamSynchronize(gpus[g].stream));
            });
        }
        for (auto& t : threads) t.join();
        printf("done\n");
    };

    // Warmup
    run_all("warmup", [&](int g) {
        CK_NCCL(ncclAllReduce(gpus[g].send, gpus[g].send, N, ncclFloat, ncclSum, comms[g], gpus[g].stream));
    });

    // --- The collectives we want NSys to capture ---

    run_all("AllReduce", [&](int g) {
        CK_NCCL(ncclAllReduce(gpus[g].send, gpus[g].send, N, ncclFloat, ncclSum, comms[g], gpus[g].stream));
    });

    run_all("AllGather", [&](int g) {
        CK_NCCL(ncclAllGather(gpus[g].send, gpus[g].recv, N, ncclFloat, comms[g], gpus[g].stream));
    });

    run_all("ReduceScatter", [&](int g) {
        CK_NCCL(ncclReduceScatter(gpus[g].send, gpus[g].send, N / ngpus, ncclFloat, ncclSum, comms[g], gpus[g].stream));
    });

    run_all("Broadcast", [&](int g) {
        CK_NCCL(ncclBroadcast(gpus[g].send, gpus[g].send, N, ncclFloat, 0, comms[g], gpus[g].stream));
    });

    run_all("Reduce", [&](int g) {
        CK_NCCL(ncclReduce(gpus[g].send, gpus[g].send, N, ncclFloat, ncclSum, 0, comms[g], gpus[g].stream));
    });

    // Cleanup
    for (int g = 0; g < ngpus; g++) {
        CK_CUDA(cudaSetDevice(g));
        cudaFree(gpus[g].send);
        cudaFree(gpus[g].recv);
        cudaStreamDestroy(gpus[g].stream);
        ncclCommDestroy(comms[g]);
    }

    printf("Done.\n");
    return 0;
}
