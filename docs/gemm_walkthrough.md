# Profiling a GEMM Kernel with Intra-Kernel Profiler

This guide walks through instrumenting, running, and analyzing a shared-memory
tiled GEMM kernel using Intra-Kernel Profiler. The full source is at
[`examples/gemm/tiled_gemm.cu`](../examples/gemm/tiled_gemm.cu).

---

## 1. Instrumenting the kernel

A tiled GEMM has four natural phases per tile iteration:

| Region ID | Name | What happens |
|-----------|------|--------------|
| 0 | `total` | Entire tile computation (outer) |
| 1 | `load_tile` | Global → shared memory: load A and B tiles |
| 2 | `compute` | Shared memory → registers: multiply-accumulate |
| 3 | `store` | Registers → global memory: write back C |

### Step 1: Include the header and declare the profiler context

```cpp
#include <intra_kernel_profiler/intra_kernel_profiler.hpp>

#define PROFILE_CAP 8192   // events per warp (power of 2)
constexpr uint32_t kWarpsPerBlock = (THREADS + 31) / 32;
```

### Step 2: Initialize the context at kernel start

```cpp
__global__ void tiled_gemm_kernel(..., intra_kernel_profiler::trace::GlobalBuffer prof) {
  IKP_TRACE_CTX_TYPE(PROFILE_CAP, kWarpsPerBlock) ctx;
  IKP_TRACE_CTX_INIT(ctx);
```

### Step 3: Bracket each phase with begin/end

```cpp
  IKP_TRACE_REC_B(ctx, prof, kTotal);

  for (int t = 0; t < num_k_tiles; ++t) {
    IKP_TRACE_REC_B(ctx, prof, kLoadTile);
    // ... load A and B tiles into shared memory ...
    IKP_TRACE_REC_E(ctx, prof, kLoadTile);

    __syncthreads();

    IKP_TRACE_REC_B(ctx, prof, kCompute);
    // ... multiply-accumulate from shared memory ...
    IKP_TRACE_REC_E(ctx, prof, kCompute);

    __syncthreads();
  }

  IKP_TRACE_REC_B(ctx, prof, kStore);
  // ... write back C ...
  IKP_TRACE_REC_E(ctx, prof, kStore);

  IKP_TRACE_REC_E(ctx, prof, kTotal);
  IKP_TRACE_CTX_FLUSH(ctx, prof);
}
```

### Step 4: Set up the host session

```cpp
intra_kernel_profiler::trace::HostSession sess;
sess.set_region_names({"total", "load_tile", "compute", "store"});
sess.set_block_filter({0, 1, 2, 3});  // only trace 4 blocks
sess.init(PROFILE_CAP, total_blocks, THREADS);
sess.reset();

tiled_gemm_kernel<<<grid, block>>>(..., sess.global_buffer());
cudaDeviceSynchronize();

sess.write_trace("gemm_trace.json");
```

**Overhead:** Profiling adds ~5 `globaltimer` reads + streaming stores per
region per iteration. For a GEMM with 64 K-tiles, that's ~640 events per warp.
With `PROFILE_CAP=8192`, overflow is not a concern.

---

## 2. Build and run

```bash
# From intra_kernel_profiler/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ikp_gemm_demo --m=1024 --n=1024 --k=1024
```

Output:
```
=== Intra-Kernel Profiler: Tiled GEMM Demo ===
Problem:  M=1024 N=1024 K=1024
Tiling:   BM=64 BN=64 BK=16  TM=4 TN=4  threads=256
Grid:     256 blocks (16 x 16 tiles)
Profiling: blocks 0-3 (of 256), cap=8192 events/warp

intra_kernel_profiler::trace: summary -> gemm_trace_summary.json
intra_kernel_profiler::trace: 8320 events -> gemm_trace.json
Benchmark: 3 iters, 0.274 ms total, 0.091 ms/iter, 23451.5 GFLOP/s
Verify:    C[0,0]=250.5690 ref=250.5691 err=2.16e-05 PASS
```

Two output files are generated:
- `gemm_trace.json` — Chrome Trace JSON (4196 trace events)
- `gemm_trace_summary.json` — per-region aggregate statistics

---

## 3. Viewing the trace in Perfetto

1. Open [ui.perfetto.dev](https://ui.perfetto.dev)
2. Drag and drop `gemm_trace.json`
3. You'll see a timeline like this:

```
Process: sm 124                          (= SM ID)
├── block 0 warp 0  ║ total ═══════════════════════════════════════════ ║
│                    ║ load ║ compute ║ load ║ compute ║ ... ║ store ║  │
├── block 0 warp 1  ║ total ═══════════════════════════════════════════ ║
│                    ║ load ║ compute ║ load ║ compute ║ ... ║ store ║  │
...
Process: sm 125
├── block 1 warp 0  ║ ...                                              ║
```

- **Each row** = one warp's timeline
- **Each colored bar** = one region instance (load_tile, compute, or store)
- **Gaps** between bars = `__syncthreads()` or memory latency
- **Click any bar** to see exact duration, SM, block, warp in the details panel

### What to look for

- **Compute vs. Load ratio**: If `load_tile` dominates, the kernel is memory-bound.
  The example shows `compute` takes ~2.2x longer than `load_tile`, indicating it's
  compute-bound at this problem size.

- **Warp-to-warp variation**: All warps in a block should have similar timelines.
  Large differences indicate load imbalance or divergence.

- **First-tile effects**: The first tile often takes longer due to cold caches.
  Compare first vs. steady-state iterations.

---

## 4. Analyzing the summary JSON

The summary JSON provides aggregate statistics without needing the viewer:

```json
{
  "unmatched_begin": 0,
  "unmatched_end": 0,
  "regions": [
    {
      "name": "total",
      "count": 32,
      "mean_dur": 85426.0,
      "cv_dur": 0.004,
      "percentiles": {"p50": 85537, "p95": 85760},
      "min_dur": 84832,
      "max_dur": 85792
    },
    {
      "name": "load_tile",
      "count": 2048,
      "mean_dur": 338.0,
      "cv_dur": 0.15,
      "percentiles": {"p50": 349, "p95": 388}
    },
    {
      "name": "compute",
      "count": 2048,
      "mean_dur": 758.0,
      "cv_dur": 0.12,
      "percentiles": {"p50": 767, "p95": 830}
    },
    {
      "name": "store",
      "count": 32,
      "mean_dur": 505.0,
      "cv_dur": 0.23,
      "percentiles": {"p50": 514, "p95": 701}
    }
  ]
}
```

### Key metrics

| Metric | Meaning |
|--------|---------|
| `count` | Total paired begin/end events across all profiled warps |
| `mean_dur` | Average duration (in display units, default = globaltimer ticks ≈ ns) |
| `cv_dur` | Coefficient of variation = std/mean. Low = consistent, high = variable |
| `p50/p95` | Median and 95th percentile duration |
| `unmatched_*` | Should be 0. Non-zero means instrumentation error |

### Time breakdown

From the summary above:

```
Per-tile:  64 × load_tile (338 ns) + 64 × compute (758 ns) + 1 × store (505 ns)
           = 21,632 ns (load) + 48,512 ns (compute) + 505 ns (store)
           ≈ 70,649 ns of traced work

Total measured: 85,426 ns
Overhead:       14,777 ns → __syncthreads() + profiler overhead + scheduling
```

This tells us compute is ~69% of traced time, loads are ~31%.

---

## 5. Capacity sizing

Each warp records `2 * (num_regions_per_iteration) + 2` events per K-tile,
plus 2 events for `total` and 2 for `store`.

For our GEMM with 64 K-tiles and 3 inner regions:
```
events_per_warp = 2 * (load + compute) * 64 + 2 * (total + store)
               = 2 * 2 * 64 + 2 * 2 = 260
```

`PROFILE_CAP=8192` is plenty. If you see `unmatched_begin > 0` in the summary,
the ring buffer overflowed — increase `PROFILE_CAP`.

---

## 6. Tips for production kernels

- **Filter blocks**: `sess.set_block_filter({0, 1})` keeps traces small and fast to load.

- **Sample in hot loops**: For kernels with thousands of iterations, use conditional recording:
  ```cpp
  IKP_TRACE_REC_IF(ctx, prof, kCompute, 0, (iter & 0xFF) == 0);  // every 256 iters
  ```

- **Disable profiling for benchmarking**: Pass a null `GlobalBuffer` to eliminate all overhead:
  ```cpp
  intra_kernel_profiler::trace::GlobalBuffer null_prof{};
  my_kernel<<<grid, block>>>(..., null_prof);  // no profiling
  ```

- **1D grids only**: The profiler uses `blockIdx.x` for slot indexing. For 2D/3D grids,
  linearize the block index and use a 1D launch.

- **Scale factor**: `opt.scale = 1.0` means globaltimer ticks (≈ nanoseconds on modern GPUs).
  Use `opt.scale = 1e-3` for microseconds, `opt.scale = 1e-6` for milliseconds.

---

## 7. Full pipeline: Explorer

This guide covered the **trace** instrumentation. For a complete profiling
pipeline that also collects NVBit instruction attribution, CUPTI hardware
counters, memory locality analysis, and generates the interactive **IKP Explorer**
dashboard, run:

```bash
bash examples/gemm/run.sh --nvbit-path=$NVBIT_PATH
```

This runs every profiling stage and produces `examples/gemm/_out/explorer.html`
— a single-page interactive dashboard with 7 tabs covering all collected data.

See [`tutorial.md`](tutorial.md) for the step-by-step breakdown of each stage.
