# NSys Integration Guide

IKP can import NVIDIA Nsight Systems (nsys) profiling data and merge it
with intra-kernel traces into a **unified Chrome Trace JSON**.  The result
is a single Perfetto view that shows system-level events (kernel launches,
memory copies, NCCL communication, CUDA API calls) alongside IKP's
per-SM/warp region timing — all on the same GPU-clock-aligned timeline.

---

## Try it now — no GPU required

Pre-generated traces are checked in — open them in
[Perfetto](https://ui.perfetto.dev) with no GPU required:

- [`examples/nsys/merged_trace.json`](../examples/nsys/merged_trace.json) —
  single GEMM kernel with IKP region timing + NSys system-level events
- [`examples/nsys_nccl/gemm_nccl_trace.json`](../examples/nsys_nccl/gemm_nccl_trace.json) —
  multi-GPU GEMM + NCCL AllGather with IKP intra-kernel trace
- [`examples/nsys/explorer.html`](../examples/nsys/explorer.html) —
  IKP Explorer with the **System** tab (serve locally with
  `python3 -m http.server --directory examples/nsys 8080`)

---

## Prerequisites

- `nsys` CLI on PATH (bundled with CUDA Toolkit 11.1+)
- Python 3.8+ (stdlib only — no extra packages)

---

## Tutorial: Merging NSys with IKP Trace

This walkthrough uses the tiled GEMM example.  The same steps apply to any
IKP-instrumented kernel.

### Step 1 — Collect an IKP trace

```bash
# Build (if not already done)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Run the GEMM demo to get an IKP trace
./build/ikp_gemm_demo --m=2048 --n=2048 --k=2048 --out=_out/trace/gemm_trace.json
```

This produces `gemm_trace.json` (per-warp region timing) and
`gemm_trace_summary.json` (aggregate statistics).

### Step 2 — Profile the same binary with nsys

```bash
mkdir -p _out/nsys

nsys profile \
  --output=_out/nsys/report \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  ./build/ikp_gemm_demo --m=2048 --n=2048 --k=2048
```

This produces `report.nsys-rep` containing kernel launches, memory copies,
CUDA API calls, and NVTX ranges.

### Step 3 — Import nsys data into IKP JSON

```bash
python3 scripts/ikp_nsys_import.py \
  --nsys-rep _out/nsys/report.nsys-rep \
  --out-dir _out/nsys/ \
  --kernel-regex "tiled_gemm_kernel"
```

This converts the `.nsys-rep` to SQLite (via `nsys export`), queries it,
and writes two JSON files:

| File | Contents |
|------|----------|
| `nsys_events.json` | All system-level events: kernel launches, memcpy, memset, CUDA API calls, NVTX ranges, synchronization, NCCL operations |
| `nsys_kernels.json` | Filtered kernel launches (matching `--kernel-regex`) |

### Step 4 — Merge into a unified trace

```bash
python3 scripts/ikp_nsys_merge.py \
  --nsys-events _out/nsys/nsys_events.json \
  --ikp-trace _out/trace/gemm_trace.json \
  --nsys-kernels _out/nsys/nsys_kernels.json \
  --kernel-regex "tiled_gemm_kernel" \
  --out _out/trace/merged_trace.json
```

Open `merged_trace.json` in [Perfetto](https://ui.perfetto.dev) or
`chrome://tracing`.  You will see:

| Process group | What it shows |
|---------------|---------------|
| `[NSys] CUDA API` | Host-side API calls (`cudaLaunchKernel`, `cudaMemcpy`, `cudaMalloc`, ...) with flow arrows to GPU kernels |
| `[NSys] GPU Kernels` | Kernel execution intervals per stream, with duration and grid/block info |
| `[NSys] Memory Ops` | HtoD/DtoH/DtoD transfers with size and bandwidth annotations |
| `[NSys] NCCL` | NCCL collective operations (AllGather, AllReduce, ...) if present |
| `[NSys] NVTX` | User-defined NVTX annotation ranges |
| `sm N` | IKP per-warp region timing (load_A, load_B, compute, store, ...) |

NSys rows appear at the top; IKP SM rows appear below.  Both share the
same nanosecond timeline anchored to the kernel launch.

### Step 5 — Generate the Explorer

```bash
python3 scripts/generate_explorer.py \
  --demo-dir _out \
  --source examples/gemm/tiled_gemm.cu \
  --output _out/explorer.html
```

The Explorer's **System** tab shows kernel launch tables, memory operation
breakdowns, NCCL timing charts, and NVTX ranges — all from the imported
nsys data.

### One-command version

```bash
# Single-GPU: GEMM with IKP trace + NSys merge
bash examples/nsys/run.sh
bash examples/nsys/run.sh --gpu=3        # use a specific GPU

# Multi-GPU: NCCL collectives profiling
bash examples/nsys_nccl/run.sh --ngpus=2 --nccl=$NCCL_HOME
```

---

## What You Can Do With This

### See the full picture

A single GEMM kernel is never just compute.  The merged trace reveals:
- How long the host spends in `cudaMalloc` and `cudaMemcpy` before the
  kernel even launches
- The gap between `cudaLaunchKernel` returning and the kernel starting
  on the GPU (launch latency)
- Whether memory transfers and kernel execution overlap on different
  streams

### Extend to your own workflows

The nsys integration is designed to be composable:

- **Import is separate from merge.** You can import nsys data once and
  merge it with different IKP traces, or use `nsys_events.json` directly
  in your own analysis scripts.
- **The JSON schema is stable.** `nsys_events.json` uses a flat, readable
  structure. Parse it with any language — Python, C++, or a Jupyter notebook.
- **NCCL auto-detection.** The importer auto-detects NCCL kernels by name
  pattern. If your workload doesn't use NCCL, the `nccl` section is simply
  empty.

---

## Script Reference

| Script | Purpose |
|--------|---------|
| `ikp_nsys_import.py` | Convert `.nsys-rep` → SQLite → IKP JSON |
| `ikp_nsys_merge.py` | Merge NSys + IKP trace into unified Chrome Trace |

### `ikp_nsys_import.py`

```
python3 scripts/ikp_nsys_import.py \
    --nsys-rep report.nsys-rep \
    --out-dir _out/nsys/ \
    [--kernel-regex "kernel_name"] \
    [--skip-export]
```

| Flag | Description |
|------|-------------|
| `--nsys-rep` | Path to `.nsys-rep` file |
| `--out-dir` | Output directory for JSON files |
| `--kernel-regex` | Regex to filter kernel names in `nsys_kernels.json` |
| `--skip-export` | Skip `nsys export` (reuse existing `.sqlite` file) |

### `ikp_nsys_merge.py`

```
python3 scripts/ikp_nsys_merge.py \
    --nsys-events _out/nsys/nsys_events.json \
    --ikp-trace _out/trace/trace.json \
    [--nsys-kernels _out/nsys/nsys_kernels.json] \
    [--kernel-regex "kernel_name"] \
    --out merged_trace.json
```

**Time alignment:** Both nsys GPU timestamps and IKP `globaltimer` come
from the same hardware clock.  The merge anchors on the first matching
kernel launch — nsys events are rebased relative to that kernel's start
time, matching IKP's t=0 convention.

**Merged trace features:**
- Flow arrows connecting `cudaLaunchKernel` → GPU kernel execution
- Duration labels on events (e.g., `tiled_gemm_kernel [667us]`)
- Bandwidth annotations on memory transfers (e.g., `HtoD 16MB 17.3 GB/s`)
- Color-coded categories (green=kernel, blue=DtoH, red=sync, orange=NCCL)
- Process sort order: NSys rows on top, IKP SM rows below

---

## Explorer System Tab

When `nsys_events.json` is present in the `nsys/` subdirectory of
`--demo-dir`, the Explorer shows a **System** tab with:

- **Summary cards** — GPU kernel count, memory op count, NCCL kernel count
- **Kernel launches table** — sortable by duration, with grid/block/stream
- **Kernel duration distribution** — bar chart of top kernels
- **Memory operations** — pie chart by direction (HtoD/DtoH/DtoD)
- **NCCL communication** — per-collective timing chart (if NCCL present)
- **NVTX ranges** — user annotation table

---

## Version Compatibility

Tested with:

| Component | Version |
|-----------|---------|
| NVIDIA Nsight Systems | 2023.4+ |
| CUDA Toolkit | 12.4+ |
| GPU | H100 (SM 9.0) |
| Python | 3.8+ |

### NSys SQLite schema differences

The importer handles schema variations across nsys versions transparently:

| Feature | Older nsys | Newer nsys (2023.4+) | Handling |
|---------|-----------|---------------------|----------|
| Kernel name | `shortName` (string) | `shortName` (int → `StringIds`) | Auto-resolved |
| Kernel table | `CUPTI_ACTIVITY_KIND_KERNEL` | `..._CONCURRENT_KERNEL` | Fallback |
| Grid dims | `gridX/Y/Z` | `gridDimX/Y/Z` | Both checked |
| Runtime API | `cbid` | `nameId` → `StringIds` | Both supported |
| NVTX text | `text` (string) | `text` (int → `StringIds`) | Auto-resolved |
| Thread ID | `threadId` | `globalTid` | Both checked |

All column resolution uses `PRAGMA table_info()`, so future nsys versions
with new columns will not break existing queries.

---

## Troubleshooting

**`nsys` not found:**
Add it to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

**Large `.nsys-rep` files:**
Use `--kernel-regex` to filter during import.  The Explorer auto-subsamples
events to keep the embedded HTML manageable.

**`nsys export` fails:**
Some HPC environments restrict nsys.  Verify access with
`nsys profile --stats=true ./your_app` first.

**Time misalignment in merged trace:**
Ensure `--kernel-regex` selects the correct anchor kernel.  Both nsys and
IKP must profile the same kernel launch for alignment to work.
