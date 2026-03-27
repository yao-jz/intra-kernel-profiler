## Integrating into an existing kernel

This document shows how to integrate the standalone, namespaced API into your
CUDA kernels.

For a runnable reference in this standalone repo, start with:

- `examples/trace/record.cu`

For "how to build and run everything", see [`tutorial.md`](../docs/tutorial.md).

---

## Using the IKP_* macros

1) Include the header:

```cpp
#include <intra_kernel_profiler/intra_kernel_profiler.hpp>
```

2) Device-side macros:

- `IKP_TRACE_CTX_TYPE` -- declare per-warp ring-buffer context
- `IKP_TRACE_CTX_INIT` -- initialize context (call once at kernel start)
- `IKP_TRACE_REC_B/E/M/IF` -- record begin / end / mark / conditional events
- `IKP_TRACE_CTX_FLUSH` -- flush ring buffer to host (call once at kernel end)

---

## Where to place regions in a real GEMM/TMA kernel

In many kernels, you will already have region ids (e.g. an enum) plus a name table. Example regions:

- `kProfileKernel` (entire kernel)
- `kProfileTileLoop` (persistent tile loop)
- `kProfileTileLoop` (persistent tile loop)
- `kProfileNSlice` (N-slice)
- `kProfileTmaIssue` (producer issues TMA)
- `kProfileWgmma` (consumer WGMMA)
- `kProfileEpilogue` / `kProfileTmaStore`

When you start, pick **a small number of high-value** regions (avoid recording B/E on every tight-loop iteration, which can overflow CAP):

- overall kernel: `kProfileKernel`
- per tile / per n-slice: `kProfileTileLoop` / `kProfileNSlice` (sample if needed)
- compute: `kProfileWgmma`
- TMA: `kProfileTmaIssue`, `kProfileTmaStore`

On the host, keep using the "region id -> readable name" mapping:

```cpp
sess.set_region_names({"kernel","tile_loop","n_slice","tma_issue","wgmma","epilogue","tma_store"});
```

---

## Relationship between NVBit markers and trace

- **NVBit markers** (`IKP_NVBIT_BEGIN/END`) are for the NVBit tool's region stack, producing `pc2region_*.json` and `region_stats_*.json`.
- **Intra-kernel trace** (`IKP_TRACE_REC_*`) produces a timeline (Chrome Trace) and can later be used to slice time windows by regions.

They can coexist, but note:

- markers introduce extern device calls which may perturb some pipelines (notably WGMMA). Build a separate "NVBit-only binary" for NVBit runs.
