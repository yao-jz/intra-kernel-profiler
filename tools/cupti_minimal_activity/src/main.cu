#include <cupti.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>

static void die(const char* msg) {
  std::fprintf(stderr, "FATAL: %s\n", msg);
  std::fflush(stderr);
  std::exit(1);
}

static void cupti_die(CUptiResult res, const char* call, const char* file, int line) {
  const char* errstr = nullptr;
  (void)cuptiGetResultString(res, &errstr);
  std::fprintf(stderr, "CUPTI error: %s failed at %s:%d: %s (%d)\n", call, file, line,
               errstr ? errstr : "<unknown>", (int)res);
  std::fflush(stderr);
  std::exit(1);
}

static void cudart_die(cudaError_t err, const char* call, const char* file, int line) {
  std::fprintf(stderr, "CUDA runtime error: %s failed at %s:%d: %s (%d)\n", call, file, line,
               cudaGetErrorString(err), (int)err);
  std::fflush(stderr);
  std::exit(1);
}

#define CUPTI_CALL(call)                                                     \
  do {                                                                       \
    CUptiResult _res = (call);                                               \
    if (_res != CUPTI_SUCCESS) {                                             \
      cupti_die(_res, #call, __FILE__, __LINE__);                            \
    }                                                                        \
  } while (0)

#define CUDART_CALL(call)                                                    \
  do {                                                                       \
    cudaError_t _err = (call);                                               \
    if (_err != cudaSuccess) {                                               \
      cudart_die(_err, #call, __FILE__, __LINE__);                           \
    }                                                                        \
  } while (0)

static constexpr size_t kActivityBufferSize = 16 * 1024;  // bytes

static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *maxNumRecords = 0;
  *size = kActivityBufferSize;

  void* ptr = nullptr;
  constexpr size_t kAlign = 8;  // CUPTI requires at least 8-byte alignment.
  if (posix_memalign(&ptr, kAlign, *size) != 0 || ptr == nullptr) {
    die("posix_memalign failed");
  }
  *buffer = reinterpret_cast<uint8_t*>(ptr);
}

static void printMemcpy(const CUpti_ActivityMemcpy5* m) {
  std::printf(
      "[MEMCPY] bytes=%" PRIu64 " start=%" PRIu64 " end=%" PRIu64
      " copyKind=%u srcKind=%u dstKind=%u device=%u context=%u stream=%u\n",
      (uint64_t)m->bytes, (uint64_t)m->start, (uint64_t)m->end, (unsigned)m->copyKind,
      (unsigned)m->srcKind, (unsigned)m->dstKind, (unsigned)m->deviceId, (unsigned)m->contextId,
      (unsigned)m->streamId);
}

static void printKernel(const CUpti_ActivityKernel9* k) {
  std::printf(
      "[KERNEL] name=%s grid=(%d,%d,%d) block=(%d,%d,%d) start=%" PRIu64 " end=%" PRIu64
      " device=%u context=%u stream=%u\n",
      k->name ? k->name : "<null>", (int)k->gridX, (int)k->gridY, (int)k->gridZ, (int)k->blockX,
      (int)k->blockY, (int)k->blockZ, (uint64_t)k->start, (uint64_t)k->end, (unsigned)k->deviceId,
      (unsigned)k->contextId, (unsigned)k->streamId);
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
                                     size_t validSize) {
  (void)size;

  CUpti_Activity* record = nullptr;
  size_t numRecords = 0;

  while (true) {
    CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      ++numRecords;
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_MEMCPY:
          printMemcpy(reinterpret_cast<const CUpti_ActivityMemcpy5*>(record));
          break;
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
          printKernel(reinterpret_cast<const CUpti_ActivityKernel9*>(record));
          break;
        default:
          break;
      }
      continue;
    }

    if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
    cupti_die(status, "cuptiActivityGetNextRecord", __FILE__, __LINE__);
  }

  size_t dropped = 0;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
  if (dropped != 0) {
    std::fprintf(stderr, "WARNING: dropped records: %zu (ctx=%p stream=%u)\n", dropped, (void*)ctx,
                 streamId);
  }

  std::fprintf(stderr, "CUPTI: bufferCompleted validSize=%zu records=%zu\n", validSize, numRecords);
  std::free(buffer);
}

__global__ void saxpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  // Basic driver init (avoid hard-failing on non-GPU/login nodes).
  {
    CUresult initRes = cuInit(0);
    if (initRes != CUDA_SUCCESS) {
      const char* name = nullptr;
      const char* str = nullptr;
      (void)cuGetErrorName(initRes, &name);
      (void)cuGetErrorString(initRes, &str);
      std::fprintf(stderr,
                   "CUDA driver init failed: %s: %s (%d)\n"
                   "If you're on a login node, please run on a GPU node.\n",
                   name ? name : "<unknown>", str ? str : "<no description>", (int)initRes);
      return 2;
    }

    int devCount = 0;
    CUresult cntRes = cuDeviceGetCount(&devCount);
    if (cntRes != CUDA_SUCCESS || devCount <= 0) {
      const char* name = nullptr;
      const char* str = nullptr;
      (void)cuGetErrorName(cntRes, &name);
      (void)cuGetErrorString(cntRes, &str);
      std::fprintf(stderr,
                   "No CUDA devices available: %s: %s (%d), count=%d\n"
                   "If you're on a login node, please run on a GPU node.\n",
                   name ? name : "<unknown>", str ? str : "<no description>", (int)cntRes, devCount);
      return 2;
    }
  }

  // Register CUPTI activity callbacks and enable activity kinds.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  // Avoid CUPTI_ACTIVITY_KIND_KERNEL (may serialize kernels); CONCURRENT_KERNEL preserves concurrency.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));

  // Tiny CUDA workload: MEMCPY + KERNEL activity.
  constexpr int n = 1 << 20;
  constexpr size_t bytes = (size_t)n * sizeof(float);

  float* hX = (float*)std::malloc(bytes);
  float* hY = (float*)std::malloc(bytes);
  if (!hX || !hY) die("host malloc failed");
  for (int i = 0; i < n; ++i) {
    hX[i] = 1.0f;
    hY[i] = 2.0f;
  }

  float* dX = nullptr;
  float* dY = nullptr;
  CUDART_CALL(cudaSetDevice(0));
  CUDART_CALL(cudaMalloc((void**)&dX, bytes));
  CUDART_CALL(cudaMalloc((void**)&dY, bytes));
  CUDART_CALL(cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dY, hY, bytes, cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((n + (int)block.x - 1) / (int)block.x);
  saxpy<<<grid, block>>>(dY, dX, 3.0f, n);
  CUDART_CALL(cudaGetLastError());

  CUDART_CALL(cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost));
  CUDART_CALL(cudaDeviceSynchronize());

  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

  CUDART_CALL(cudaFree(dX));
  CUDART_CALL(cudaFree(dY));
  std::free(hX);
  std::free(hY);

  std::printf("DONE\n");
  return 0;
}

