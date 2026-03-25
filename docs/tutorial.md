# Intra-Kernel Profiler Tutorial

This tutorial walks through collecting profiler data and generating the
**IKP Explorer** -- a Compiler-Explorer-style interactive dashboard that
consolidates every profiling signal (timing, instructions, memory, stalls,
hardware counters) into a single HTML page.

The running example throughout is a **shared-memory tiled GEMM**
(`examples/gemm/tiled_gemm.cu`) with five named regions:

| Region ID | Name | What it covers |
|:---------:|------|----------------|
| 1 | `total` | Entire tile computation envelope |
| 2 | `load_A` | Global -> shared: load A tile |
| 3 | `load_B` | Global -> shared: load B tile |
| 4 | `compute` | Shared -> registers: multiply-accumulate |
| 5 | `store` | Registers -> global: write-back C |

All commands assume you are working from the `intra_kernel_profiler/`
directory root.

---

## Quick Start: 5 Commands to Explorer

For the impatient -- build everything, run all profilers on the GEMM example,
and open the Explorer.  Requires `NVBIT_PATH` to be set (see Prerequisites).

```bash
# 1. Build trace examples (CMake)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 2. Build CUPTI + NVBit tools
make -C tools/cupti_region_profiler -j
make -C tools/nvbit_region_profiler NVBIT_PATH=$NVBIT_PATH ARCH=90a -j

# 3. Run everything end-to-end (all profiler modes + post-processing)
bash scripts/run_all_examples.sh --out=_demo_out --nvbit-path=$NVBIT_PATH

# 4. Generate the Explorer
python3 scripts/generate_explorer.py \
  --demo-dir _demo_out \
  --source examples/gemm/tiled_gemm.cu \
  --output explorer.html

# 5. Open it
python3 -m http.server 8080 -d _demo_out
# Then visit: http://localhost:8080/explorer.html
```

The Explorer uses Monaco Editor (loaded from CDN), so it must be served over
HTTP -- opening the `.html` file directly via `file://` will not work.

If you want to understand what each step produces and how it maps to Explorer
tabs, read on.

---

## Prerequisites

| Dependency | Required for | Notes |
|-----------|-------------|-------|
| NVIDIA GPU + CUDA driver | Everything | |
| CUDA Toolkit (`nvcc`, CUPTI) | Everything | >= 11.0, tested on 12.x |
| CMake >= 3.20 | Trace examples | `pip install cmake` if needed |
| NVBit 1.7+ | `tools/nvbit_region_profiler` | Architecture-specific download |
| Python 3 >= 3.8 | Explorer + analysis scripts | Standard library only for core scripts |
| NumPy + Matplotlib | `generate_gallery.py` (optional) | Only for publication-quality static charts |

See [`docs/install.md`](docs/install.md) for detailed installation
instructions for each dependency.

---

## Step 0: Build Everything

Three separate build targets: the CMake trace examples, the CUPTI injection
libraries, and the NVBit region profiler tool.

### 0a. Trace examples (CMake)

Builds four example binaries, including `ikp_gemm_demo` (our GEMM):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Produces:
- `build/ikp_gemm_demo` -- tiled GEMM with trace instrumentation
- `build/ikp_trace_record`, `build/ikp_trace_block_filter`, `build/ikp_trace_sampled_loop`

### 0b. CUPTI injection collectors

Builds four `.so` libraries that are loaded via `CUDA_INJECTION64_PATH`:

```bash
make -C tools/cupti_region_profiler -j
```

Produces:
- `tools/cupti_region_profiler/ikp_cupti_pcsamp.so` -- PC sampling (stall reasons)
- `tools/cupti_region_profiler/ikp_cupti_sassmetrics.so` -- SASS metrics (5 profiles)
- `tools/cupti_region_profiler/ikp_cupti_instrexec.so` -- instruction execution counts
- `tools/cupti_region_profiler/ikp_cupti_pmsamp.so` -- PM sampling

### 0c. NVBit region profiler tool

Requires `NVBIT_PATH` pointing to the NVBit root (must contain `core/libnvbit.a`):

```bash
make -C tools/nvbit_region_profiler \
  NVBIT_PATH=$NVBIT_PATH \
  ARCH=90a \
  -j
```

Produces: `tools/nvbit_region_profiler/region_profiler.so`

### 0d. NVBit-enabled GEMM binary

NVBit requires a separate binary built with `-rdc=true` and
`-DIKP_ENABLE_NVBIT_MARKERS`.  The markers are no-ops in the CMake build,
so we compile a second binary manually:

```bash
cd examples/gemm
nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo -rdc=true \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I ../../include \
  tiled_gemm.cu ../../src/nvbit_marker_device.cu \
  -o tiled_gemm_nvbit
cd ../..
```

Produces: `examples/gemm/tiled_gemm_nvbit`

---

## Step 1: Intra-Kernel Trace

**What this does:** Records nanosecond-resolution per-warp begin/end
timestamps for each profiling region using `globaltimer`.  Zero runtime
cost on warps outside the block filter.

**Adds to Explorer:** Trace tab (timing distributions, percentiles,
per-block/per-warp heatmap).

### Run

```bash
mkdir -p _demo_out/trace

./build/ikp_gemm_demo \
  --m=1024 --n=1024 --k=1024 \
  --out=_demo_out/trace/gemm_trace.json
```

### Output

| File | Description |
|------|-------------|
| `_demo_out/trace/gemm_trace.json` | Chrome Trace JSON -- open in [Perfetto](https://ui.perfetto.dev) |
| `_demo_out/trace/gemm_trace_summary.json` | Per-region statistics: count, mean, p50/p95/p99, histograms |

The summary JSON is what the Explorer's **Trace** tab reads.  It shows:
- Per-region duration distributions (histograms + percentile tables)
- Coefficient of variation (CV) to flag inconsistent regions
- Per-block-per-warp breakdown (shows load imbalance across CTAs)

### Verify

Open the trace in Perfetto to confirm your instrumentation recorded all five
regions:

```
https://ui.perfetto.dev   # drag-and-drop gemm_trace.json
```

---

## Step 2: NVBit Binary Instrumentation

**What this does:** NVBit intercepts the kernel at the SASS level.  The
`region_profiler.so` tool uses device-call markers (`IKP_NVBIT_BEGIN` /
`IKP_NVBIT_END`) to maintain a per-warp region stack and attribute every
executed instruction to a named region.

**Adds to Explorer:** Overview (region summaries, 40+ metrics), Regions tab
(per-region detail with derived metrics), Execution tab (pipeline attribution,
basic-block hotspots, branch analysis), Memory tab (access patterns, locality,
reuse distance).

All NVBit runs use the same pattern:

```bash
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=tiled_gemm_kernel \
IKP_NVBIT_MODE=<mode> \
IKP_NVBIT_TRACE_PATH=<output_dir> \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./examples/gemm/tiled_gemm_nvbit --iters=1
```

We run five modes.  Each adds data to different Explorer panels.

### 2a. pcmap -- PC-to-region mapping + region stats

The foundation.  Maps every SASS PC offset to its dominant region and
computes per-region instruction counts, global memory operations, sector
counts, and cache line estimates.

```bash
mkdir -p _demo_out/nvbit/pcmap

IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=tiled_gemm_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_TRACE_PATH=_demo_out/nvbit/pcmap \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./examples/gemm/tiled_gemm_nvbit --iters=1
```

**Output:**
- `pc2region_*.json` -- PC offset -> region ID mapping (used by CUPTI join)
- `region_stats_*.json` -- per-region instruction counts, memory stats
- `sass_all_*.sass` -- full SASS listing with region annotations
- `summary_*.txt` -- human-readable summary

### 2b. all -- + memory trace + instruction mix

Adds per-warp, per-lane memory address traces and complete instruction class
breakdown.

```bash
mkdir -p _demo_out/nvbit/all

IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=tiled_gemm_kernel \
IKP_NVBIT_MODE=all \
IKP_NVBIT_TRACE_CAP=4096 \
IKP_NVBIT_TRACE_PATH=_demo_out/nvbit/all \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./examples/gemm/tiled_gemm_nvbit --iters=1
```

**Output adds:** `mem_trace_*.jsonl` -- per-warp, per-lane memory addresses.
The Explorer's **Memory** tab uses this for access pattern visualization,
per-PC locality analysis, and cache line reuse distance histograms.

### 2c. inst_pipe -- pipeline attribution (16 categories)

Attributes each instruction to one of 16 hardware pipelines (fp32, fp16,
fp64, int, tensor, ld, st, sfu, etc.).

```bash
mkdir -p _demo_out/nvbit/inst_pipe

IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=tiled_gemm_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_ENABLE_INST_PIPE=1 \
IKP_NVBIT_TRACE_PATH=_demo_out/nvbit/inst_pipe \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./examples/gemm/tiled_gemm_nvbit --iters=1
```

**Output adds:** `inst_pipe` field in `region_stats_*.json` with per-pipeline
counts.  The Explorer's **Execution** tab renders this as a stacked bar chart
showing which functional units each region exercises.

### 2d. bb_hot -- basic-block hotspots + branch sites

Identifies the most-executed basic blocks and analyzes branch
taken/fallthrough rates.

```bash
mkdir -p _demo_out/nvbit/bb_hot

IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=tiled_gemm_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_ENABLE_BB_HOT=1 \
IKP_NVBIT_ENABLE_BRANCH_SITES=1 \
IKP_NVBIT_TRACE_PATH=_demo_out/nvbit/bb_hot \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./examples/gemm/tiled_gemm_nvbit --iters=1
```

**Output adds:** `hotspots_*.json` with per-basic-block execution counts
and per-branch-site taken/fallthrough analysis.  The Explorer uses this in the
**Execution** tab to highlight the hottest code paths and show branch divergence.

### 2e. nvdisasm -- high-quality SASS for source panel

Dumps nvdisasm output with source line mappings, register metadata, and
barrier annotations.  This gives the Explorer's SASS panel richer data than
the default NVBit-quality SASS listing.

```bash
mkdir -p _demo_out/nvbit/nvdisasm

IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=tiled_gemm_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_DUMP_NVDISASM_SASS=1 \
IKP_NVBIT_DUMP_SASS_META=1 \
IKP_NVBIT_DUMP_SASS_LINEINFO=1 \
IKP_NVBIT_KEEP_CUBIN=1 \
IKP_NVBIT_TRACE_PATH=_demo_out/nvbit/nvdisasm \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./examples/gemm/tiled_gemm_nvbit --iters=1
```

**Output adds:** High-quality `sass_all_*.sass` with `//## File "...", line N`
comments, `cubin_*.cubin` for external analysis.  The Explorer uses these
for source-SASS-PTX cross-linking in the three-panel code view.

---

## Step 3: CUPTI Hardware Counters

**What this does:** CUPTI collects real hardware counter values at per-PC
granularity via injection libraries.  These complement NVBit's instruction-level
view with actual performance counter data from the GPU.

**Adds to Explorer:** Stalls tab (PC-sampling stall reasons per region),
Line tab (per-source-line CUPTI metrics), Overview (cross-validation of
NVBit x CUPTI region attribution).

CUPTI works on the **CMake-built** binary (no NVBit markers needed).
The binary must be compiled with `-lineinfo` for source mapping; the CMake
build already ensures this for `ikp_gemm_demo`.

All CUPTI runs follow the pattern:

```bash
CUDA_INJECTION64_PATH=tools/cupti_region_profiler/<collector>.so \
<env vars> \
./build/ikp_gemm_demo --m=1024 --n=1024 --k=1024 --iters=20
```

### 3a. SASS Metrics (5 profiles)

Each profile collects a different set of hardware counters.  The GPU can only
collect a limited number of counters per pass, so we run five separate
profiles:

```bash
mkdir -p _demo_out/cupti

for profile in core divergence memory instruction_mix branch; do
  CUDA_INJECTION64_PATH=tools/cupti_region_profiler/ikp_cupti_sassmetrics.so \
  IKP_CUPTI_SASS_OUT=_demo_out/cupti/sassmetrics_${profile}.json \
  IKP_CUPTI_SASS_PROFILE=${profile} \
  IKP_CUPTI_SASS_LAZY_PATCHING=1 \
  IKP_CUPTI_SASS_ENABLE_SOURCE=1 \
  ./build/ikp_gemm_demo --m=1024 --n=1024 --k=1024 --iters=20
done
```

**Output:** `sassmetrics_core.json`, `sassmetrics_divergence.json`,
`sassmetrics_memory.json`, `sassmetrics_instruction_mix.json`,
`sassmetrics_branch.json`.

Each contains per-PC hardware counter values.  The Explorer aggregates
these across profiles and maps them to source lines (in the **Line** tab)
and to NVBit regions (in the **Regions** tab) using the pc2region join.

### 3b. PC Sampling (stall reasons)

Statistically samples the warp scheduler to determine what each warp is
stalled on at each PC.

```bash
CUDA_INJECTION64_PATH=tools/cupti_region_profiler/ikp_cupti_pcsamp.so \
IKP_CUPTI_PCSAMP_OUT=_demo_out/cupti/pcsampling_raw.json \
IKP_CUPTI_PCSAMP_COLLECTION_MODE=serialized \
IKP_CUPTI_PCSAMP_KERNEL_REGEX=tiled_gemm_kernel \
IKP_CUPTI_PCSAMP_PERIOD=5 \
IKP_CUPTI_PCSAMP_MAX_PCS=10000 \
IKP_CUPTI_PCSAMP_VERBOSE=1 \
./build/ikp_gemm_demo --m=1024 --n=1024 --k=1024 --iters=20
```

**Output:** `pcsampling_raw.json` -- per-PC stall reason samples.

The Explorer's **Stalls** tab visualizes this as a stacked bar chart showing
the distribution of stall reasons (memory dependency, scoreboard, barrier,
etc.) for the overall kernel and per region.

> **Note:** PC sampling requires unrestricted profiling permissions.  On
> managed HPC clusters it may return empty results.  SASS metrics generally
> works even on restricted nodes.

### 3c. Instruction Execution

Counts the number of threads executing each PC, including predication info.

```bash
CUDA_INJECTION64_PATH=tools/cupti_region_profiler/ikp_cupti_instrexec.so \
IKP_CUPTI_INSTREXEC_OUT=_demo_out/cupti/instrexec_raw.json \
IKP_CUPTI_INSTREXEC_KERNEL_REGEX=tiled_gemm_kernel \
IKP_CUPTI_INSTREXEC_MAX_RECORDS=0 \
./build/ikp_gemm_demo --m=1024 --n=1024 --k=1024 --iters=20
```

**Output:** `instrexec_raw.json` -- per-PC thread execution counts and
predication breakdown.  The Explorer uses this for cross-validation with
NVBit instruction counts and to compute thread-level occupancy efficiency.

---

## Step 4: Generate the Explorer

With all profiling data in `_demo_out/`, generate the single-page Explorer:

```bash
python3 scripts/generate_explorer.py \
  --demo-dir _demo_out \
  --source examples/gemm/tiled_gemm.cu \
  --output explorer.html
```

The Explorer is a self-contained HTML file that embeds all profiling data
as JSON.  It uses Monaco Editor (CDN), ECharts, and Split.js.

### Serve locally

Monaco requires HTTP, so serve the output directory:

```bash
python3 -m http.server 8080 -d _demo_out
# visit http://localhost:8080/explorer.html
```

Or use the built-in `--serve` flag:

```bash
python3 scripts/generate_explorer.py \
  --demo-dir _demo_out \
  --source examples/gemm/tiled_gemm.cu \
  --output explorer.html \
  --serve
# Starts server on http://localhost:8080/explorer.html
```

### What each tab shows

The Explorer has a three-panel code view (CUDA Source, PTX, SASS) on the
left and a tabbed metrics panel on the right with seven tabs:

| Tab | What it shows | Data sources |
|-----|---------------|-------------|
| **Overview** | Kernel-level summary: total instructions, memory ops, pipeline utilization, region count, CUPTI profile coverage | NVBit region_stats, CUPTI SASS metrics |
| **Line** | Per-source-line metrics -- click any line in the source panel to see its CUPTI hardware counters, region attribution, and per-PC breakdown | CUPTI SASS metrics (with `-lineinfo`) |
| **Regions** | Per-region detail cards with 40+ metrics: instruction counts, memory stats, derived ratios (coalescing efficiency, wasted bandwidth), CUPTI per-region aggregation | NVBit region_stats + CUPTI pc2region join |
| **Execution** | Pipeline attribution (stacked bar by region), basic-block hotspot table, branch site analysis (taken/fallthrough rates) | NVBit inst_pipe, bb_hot, branch_sites |
| **Memory** | Per-region memory access patterns, per-PC locality analysis, cache line reuse distance, sectors-per-instruction histogram | NVBit mem_trace, region_stats |
| **Stalls** | PC-sampling stall reason distribution (overall + per-region), dominant stall identification | CUPTI pcsampling |
| **Trace** | Timing distributions (histograms + percentile tables), per-block/per-warp heatmap, CV analysis | Intra-kernel trace summary |

The source, PTX, and SASS panels are cross-linked: clicking a source line
highlights the corresponding PTX `.loc` range and SASS instructions.  Region
colors are consistent across all panels.

---

## Step 5: (Optional) Additional Visualizations

The Explorer is the primary destination, but several other visualization
scripts exist for specific use cases.

### Matplotlib gallery (publication-quality charts)

Generates 19 static PNG charts suitable for papers and presentations.
Requires `numpy` and `matplotlib`.

```bash
python3 scripts/generate_gallery.py \
  --demo-dir _demo_out \
  --out-dir _demo_out/gallery
```

### Source annotation

Annotates the CUDA source file with per-line SASS metrics and region
attribution, producing a standalone HTML file:

```bash
python3 scripts/annotate_source.py \
  --sass _demo_out/cupti/sassmetrics_core.json \
        _demo_out/cupti/sassmetrics_divergence.json \
  --pc2region _demo_out/nvbit/pcmap/pc2region_*.json \
  --source examples/gemm/tiled_gemm.cu \
  --labels "0:outside,1:total,2:load_A,3:load_B,4:compute,5:store" \
  --html _demo_out/annotated_source.html
```

### Perfetto trace viewer

The Chrome Trace JSON from Step 1 can be opened directly in Perfetto for
a timeline view:

```
https://ui.perfetto.dev   # drag-and-drop _demo_out/trace/gemm_trace.json
```

---

## One-Command Pipeline

`scripts/run_all_examples.sh` runs all build + profiling + post-processing
steps in one shot.  It accepts the same conventions used throughout this
tutorial:

```bash
bash scripts/run_all_examples.sh \
  --out=_demo_out \
  --nvbit-path=$NVBIT_PATH \
  --arch=90a \
  --sm=sm_90a
```

This produces:
- `_demo_out/trace/` -- Chrome Trace JSON + summary statistics
- `_demo_out/cupti/` -- PC sampling, SASS metrics (5 profiles), instrexec
- `_demo_out/nvbit/` -- 6 NVBit modes (pcmap, all, inst_pipe, bb_hot, nvdisasm, ptx)
- `_demo_out/join/` -- NVBit + CUPTI join analysis
- `_demo_out/gallery/` -- 19 auto-generated charts (matplotlib)
- `_demo_out/explorer.html` -- the Explorer
- `_demo_out/dashboard.html` -- interactive Plotly dashboard
- `_demo_out/report.html` -- Plotly HTML report
- `_demo_out/annotated_source.html` -- annotated source

---

## Environment Variable Reference

### NVBit Region Profiler

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_NVBIT_ENABLE` | `0` | Master enable |
| `IKP_NVBIT_MODE` | `pcmap` | `pcmap`, `instmix`, `memtrace`, `all` |
| `IKP_NVBIT_KERNEL_REGEX` | `.*` | Kernel name filter (regex) |
| `IKP_NVBIT_TRACE_PATH` | `.` | Output directory |
| `IKP_NVBIT_TRACE_CAP` | `0` (unlimited) | Max mem_trace records |
| `IKP_NVBIT_ENABLE_INST_PIPE` | `0` | Per-pipeline instruction counts |
| `IKP_NVBIT_ENABLE_BB_HOT` | `0` | Basic-block hotspot analysis |
| `IKP_NVBIT_ENABLE_BRANCH_SITES` | `0` | Per-branch taken/fallthrough analysis |
| `IKP_NVBIT_DUMP_SASS` | `1` | Dump NVBit SASS listing |
| `IKP_NVBIT_DUMP_SASS_BY_REGION` | `1` | Per-region SASS slices |
| `IKP_NVBIT_DUMP_NVDISASM_SASS` | `0` | High-quality nvdisasm output |
| `IKP_NVBIT_DUMP_SASS_META` | `0` | SASS metadata (register usage, barriers) |
| `IKP_NVBIT_DUMP_SASS_LINEINFO` | `0` | Source file:line in SASS comments |
| `IKP_NVBIT_DUMP_PTX` | `0` | PTX listing dump |
| `IKP_NVBIT_DUMP_PTX_BY_REGION` | `0` | Per-region PTX slices |
| `IKP_NVBIT_KEEP_CUBIN` | `0` | Keep extracted cubin files |

### CUPTI PC Sampling

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_CUPTI_PCSAMP_OUT` | `pcsampling.json` | Output file |
| `IKP_CUPTI_PCSAMP_COLLECTION_MODE` | `serialized` | Collection mode |
| `IKP_CUPTI_PCSAMP_KERNEL_REGEX` | `.*` | Kernel name filter |
| `IKP_CUPTI_PCSAMP_PERIOD` | `5` | Sampling period |
| `IKP_CUPTI_PCSAMP_MAX_PCS` | `10000` | Max PC buffer records |
| `IKP_CUPTI_PCSAMP_MAX_RECORDS` | `0` | Max total records (0=unlimited) |
| `IKP_CUPTI_PCSAMP_VERBOSE` | `0` | Verbosity level |

### CUPTI SASS Metrics

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_CUPTI_SASS_OUT` | `sassmetrics.json` | Output file |
| `IKP_CUPTI_SASS_PROFILE` | `core` | Profile: `core`, `divergence`, `memory`, `instruction_mix`, `branch` |
| `IKP_CUPTI_SASS_LAZY_PATCHING` | `0` | Use lazy patching (reduces overhead) |
| `IKP_CUPTI_SASS_ENABLE_SOURCE` | `0` | Include source file:line (requires `-lineinfo`) |
| `IKP_CUPTI_SASS_LIST` | `0` | List available metrics and exit |
| `IKP_CUPTI_SASS_LIST_OUT` | (stdout) | Output file for metric list |

### CUPTI Instruction Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_CUPTI_INSTREXEC_OUT` | `instrexec.json` | Output file |
| `IKP_CUPTI_INSTREXEC_KERNEL_REGEX` | `.*` | Kernel name filter |
| `IKP_CUPTI_INSTREXEC_MAX_RECORDS` | `0` | Max records (0=unlimited) |

---

## Data Directory Layout

The `generate_explorer.py` script expects profiling outputs organized
in the following directory structure.  This is the layout produced by
`run_all_examples.sh` and the manual steps in this tutorial.

```
_demo_out/
  trace/
    gemm_trace.json              # Chrome Trace JSON (Step 1)
    gemm_trace_summary.json      # Per-region statistics (Step 1)

  cupti/
    sassmetrics_core.json         # SASS metrics: core profile (Step 3a)
    sassmetrics_divergence.json   #   divergence profile
    sassmetrics_memory.json       #   memory profile
    sassmetrics_instruction_mix.json  #   instruction_mix profile
    sassmetrics_branch.json       #   branch profile
    sassmetrics_source.json       # (optional) with source mapping
    pcsampling_raw.json           # PC sampling stall reasons (Step 3b)
    instrexec_raw.json            # Instruction execution (Step 3c)

  nvbit/
    pcmap/
      pc2region_<kernel>_0.json   # PC -> region mapping (Step 2a)
      region_stats_<kernel>_0.json  # Per-region stats
      sass_all_<kernel>_0.sass    # SASS listing
      summary_<kernel>_0.txt      # Human-readable summary
    all/
      mem_trace_<kernel>_0.jsonl  # Memory address traces (Step 2b)
      pc2region_<kernel>_0.json
      region_stats_<kernel>_0.json
      locality_analysis.json      # (post-processing)
    inst_pipe/
      region_stats_<kernel>_0.json  # Has inst_pipe field (Step 2c)
      pc2region_<kernel>_0.json
    bb_hot/
      hotspots_<kernel>_0.json    # BB hotspots + branches (Step 2d)
      region_stats_<kernel>_0.json
      pc2region_<kernel>_0.json
    nvdisasm/
      sass_all_<kernel>_0.sass    # nvdisasm-quality SASS (Step 2e)
      cubin_<kernel>_0.cubin      # Extracted cubin

  explorer.html                    # The Explorer (Step 4)
```

The `<kernel>` placeholder expands to the mangled kernel name, e.g.:

```
tiled_gemm_kernel_float_const___float_const___float___int__int__int__int__intra_kernel_profiler__trace__GlobalBuffer_
```

The Explorer scans `nvbit/*/pc2region_*.json` and `nvbit/*/region_stats_*.json`
using glob patterns, so the exact subdirectory names and kernel name mangling
do not matter -- it finds them automatically.
