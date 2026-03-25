# NVBit Region Profiler Guide

The NVBit region profiler instruments your CUDA binary at the SASS level to
attribute every instruction to a **region** (defined by `IKP_NVBIT_BEGIN/END`
markers in your kernel source). It produces per-region instruction counts,
memory statistics, and PC-to-region mappings.

---

## Overview

```
Your Kernel Source          NVBit Tool (LD_PRELOAD)           Output
┌──────────────────┐       ┌─────────────────────────┐       ┌──────────────────────┐
│ IKP_NVBIT_BEGIN(1)│──────▶│ Intercepts marker calls │──────▶│ pc2region_*.json     │
│ // compute...    │       │ Maintains region stack  │       │ region_stats_*.json  │
│ IKP_NVBIT_END(1) │       │ Counts instructions     │       │ sass_all_*.sass      │
│ IKP_NVBIT_BEGIN(2)│       │ per region per PC       │       │ sass_regions_*/      │
│ // store...      │       │                         │       │ summary_*.txt        │
│ IKP_NVBIT_END(2) │       │                         │       │ mem_trace_*.jsonl    │
└──────────────────┘       └─────────────────────────┘       └──────────────────────┘
```

---

## Step 1: Add markers to your kernel

```cpp
#include <intra_kernel_profiler/nvbit/markers.cuh>

__global__ void my_kernel(float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  IKP_NVBIT_BEGIN(1);       // region 1: compute
  float x = float(idx);
  for (int i = 0; i < 256; ++i)
    x = x * 1.0001f + 0.00001f;
  IKP_NVBIT_END(1);

  IKP_NVBIT_BEGIN(2);       // region 2: store
  if (idx < n) out[idx] = x;
  IKP_NVBIT_END(2);
}
```

Markers support up to **7 nested regions** (IDs 0–6, region 0 is "outside all markers").

## Step 2: Build with markers enabled

```bash
nvcc -O3 -std=c++17 -arch=sm_90a -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I path/to/intra_kernel_profiler/include \
  my_kernel.cu path/to/intra_kernel_profiler/src/nvbit_marker_device.cu \
  -o my_kernel_nvbit
```

Key flags:
- `-DIKP_ENABLE_NVBIT_MARKERS` — enables the marker device calls
- `-rdc=true` — required for cross-file device linking
- `-lineinfo` — enables source-line correlation in SASS dumps

**Important:** Only enable markers for NVBit profiling runs. The extern device
calls can perturb performance. Build a separate binary without markers for
benchmarking.

## Step 3: Build the NVBit tool

```bash
make -C tools/nvbit_region_profiler \
  NVBIT_PATH=/path/to/nvbit_release \
  ARCH=90a -j
```

This produces `tools/nvbit_region_profiler/region_profiler.so`.

## Step 4: Run with injection

```bash
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=my_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_TRACE_PATH=./nvbit_out \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./my_kernel_nvbit
```

---

## Modes

| Mode | What it collects | Output files | Overhead |
|------|-----------------|--------------|----------|
| `pcmap` | PC → region mapping + instruction counts | `pc2region_*.json`, `region_stats_*.json`, SASS | Low |
| `instmix` | Instruction mix + memory instruction stats | `region_stats_*.json` | Low |
| `memtrace` | Per-address memory trace | `mem_trace_*.jsonl` | **High** |
| `all` | Everything above | All files | High |

### pcmap (recommended default)

Produces the PC-to-region mapping needed to join with CUPTI per-PC data:

```bash
IKP_NVBIT_MODE=pcmap
```

### instmix

Collects instruction class breakdown (ALU, memory, tensor, branch, etc.):

```bash
IKP_NVBIT_MODE=instmix
```

### memtrace

Traces every global/shared/local memory access with addresses:

```bash
IKP_NVBIT_MODE=memtrace
IKP_NVBIT_TRACE_CAP=4096        # limit trace entries per kernel
IKP_NVBIT_SAMPLE_MEM_EVERY_N=8  # sample 1-in-8 memory ops
```

### all

Enables everything:

```bash
IKP_NVBIT_MODE=all
IKP_NVBIT_TRACE_CAP=4096
```

---

## Output Files

### `region_stats_*.json`

Per-region instruction and memory statistics:

```json
{
  "kernel": "my_kernel",
  "regions": [
    {
      "region": 0,
      "inst_total": 1146880,
      "inst_pred_off": 32768,
      "gmem_load": 0, "gmem_store": 0, "gmem_bytes": 0,
      "smem_load": 0, "smem_store": 0, "smem_bytes": 0,
      "inst_class": {
        "alu_fp32": 0, "alu_int": 229376,
        "ld_global": 65536, "st_global": 0,
        "branch": 65536, "call": 65536, "other": 688128
      },
      "bb_exec": 229376
    },
    {
      "region": 1,
      "inst_total": 34078720,
      "inst_class": {
        "alu_fp32": 8454144, "alu_int": 16809984,
        "branch": 8388608
      }
    },
    {
      "region": 2,
      "inst_total": 491520,
      "gmem_store": 32768, "gmem_bytes": 4194304,
      "inst_class": {
        "st_global": 32768, "alu_int": 65536
      }
    }
  ]
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `region` | Region ID (0 = outside all markers) |
| `inst_total` | Total dynamic instructions executed in this region |
| `inst_pred_off` | Instructions skipped due to predication |
| `gmem_load/store` | Number of global memory load/store operations |
| `gmem_bytes` | Total global memory bytes accessed |
| `smem_load/store` | Shared memory operations |
| `inst_class` | Breakdown by instruction type |
| `bb_exec` | Basic block execution count |
| `branch_div_hist` | Histogram of branch divergence (warps with N active threads) |

**inst_class categories:**

| Category | SASS instructions |
|----------|-------------------|
| `alu_fp32` | FADD, FMUL, FFMA, etc. |
| `alu_int` | IADD, IMAD, ISETP, etc. |
| `tensor_wgmma` | WGMMA (Tensor Core) |
| `ld_global` / `st_global` | LDG, STG |
| `ld_shared` / `st_shared` | LDS, STS |
| `barrier` | BAR.SYNC |
| `branch` | BRA, JMP, etc. |
| `call` / `ret` | CAL, RET |

### `pc2region_*.json`

Maps each program counter offset to a region:

```json
{
  "kernel": "my_kernel",
  "pc2region": [
    {
      "function_id": 4,
      "function_name": "my_kernel",
      "pc_offset": 256,
      "dominant_region": 1,
      "executed_count": 1048576,
      "ambiguity": "none"
    }
  ]
}
```

This file is the **join key** for combining NVBit regions with CUPTI per-PC data.

### `sass_all_*.sass`

Full SASS disassembly of the instrumented kernel (from NVBit's internal representation).

### `sass_regions_*/region_N.sass`

Per-region SASS slices — only the instructions belonging to each region.

### `mem_trace_*.jsonl` (memtrace/all mode only)

JSON Lines format, one entry per memory operation:

```json
{"pc":256,"region":1,"addr":"0x7f1234000","bytes":4,"type":"ld_global","warp":0,"cta":0}
```

### `summary_*.txt`

Human-readable text summary with paths to all output files and configuration.

---

## Environment Variables Reference

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_NVBIT_ENABLE` | `0` | Must be `1` to activate |
| `IKP_NVBIT_MODE` | `pcmap` | `pcmap`, `instmix`, `memtrace`, or `all` |
| `IKP_NVBIT_KERNEL_REGEX` | `.*` | Regex to filter kernel names |
| `IKP_NVBIT_TRACE_PATH` | `.` | Output directory |
| `IKP_NVBIT_MAX_REGIONS` | `128` | Max tracked regions |

### Memory trace

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_NVBIT_TRACE_CAP` | `65536` | Max trace entries per kernel |
| `IKP_NVBIT_SAMPLE_MEM_EVERY_N` | `1` | Sample rate for memory ops |
| `IKP_NVBIT_SAMPLE_CTA` | all | Only trace this CTA ID |
| `IKP_NVBIT_SAMPLE_WARP` | all | Only trace this warp ID |

### SASS dump

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_NVBIT_DUMP_SASS` | `1` | Dump full SASS listing |
| `IKP_NVBIT_DUMP_SASS_BY_REGION` | `1` | Dump per-region SASS slices |
| `IKP_NVBIT_DUMP_NVDISASM_SASS` | `0` | Use `nvdisasm` for SASS (higher quality, requires binary in PATH) |
| `IKP_NVBIT_DUMP_PTX` | `0` | Dump PTX |
| `IKP_NVBIT_DUMP_PTX_BY_REGION` | `0` | Dump per-region PTX slices |
| `IKP_NVBIT_KEEP_CUBIN` | `0` | Keep extracted CUBIN files |
| `IKP_NVBIT_DUMP_SASS_META` | `0` | Include per-instruction metadata (opcode, predicate, memory space, ld/st flags) |
| `IKP_NVBIT_DUMP_SASS_LINEINFO` | `0` | Include source file/line correlation in SASS dumps |
| `IKP_NVBIT_DUMP_PTX_LINEINFO` | `1` | Include file/line info in PTX dumps |

### Advanced

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_NVBIT_TARGET_REGION` | all | Only instrument this region ID |
| `IKP_NVBIT_ITER_BEGIN/END` | `0`/max | Kernel invocation range to profile |
| `IKP_NVBIT_ENABLE_INST_CLASS` | `1` | Track instruction class breakdown |
| `IKP_NVBIT_ENABLE_INST_PIPE` | `0` | Track per-instruction pipeline attribution (16 categories) |
| `IKP_NVBIT_ENABLE_BB` | `1` | Track basic block execution |
| `IKP_NVBIT_ENABLE_BRANCH_DIV` | `1` | Track branch divergence histogram |
| `IKP_NVBIT_ENABLE_BB_HOT` | `0` | Track basic block hotspot counts and output `hotspots_*.json` |
| `IKP_NVBIT_ENABLE_BRANCH_SITES` | `0` | Track per-branch taken/fallthrough counts in `hotspots_*.json` |
| `IKP_NVBIT_ENABLE_MEM_PATTERN` | `0` | Track memory access patterns |
| `IKP_NVBIT_INSTRUMENT_RELATED` | `0` | Also instrument callee/related functions |
| `IKP_NVBIT_REWEIGHT_MEM_EXEC` | `1` | Reweight memory instruction execution counts |
| `IKP_NVBIT_REWEIGHT_MEM_PATTERN` | `1` | Reweight memory pattern stats |
| `IKP_NVBIT_GMEM_SET_BINS` | `0` | Number of global memory working set histogram bins |
| `IKP_NVBIT_VERBOSE` | `0` | Verbosity level |

### Capacity tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_NVBIT_MAX_DEPTH` | `16` | Max region nesting depth |
| `IKP_NVBIT_PCMAP_CAP` | `1048576` | Max entries in PC→region map |
| `IKP_NVBIT_MAX_WARPS` | `262144` | Max warps to track |
| `IKP_NVBIT_BB_CAP` | `65536` | Max basic block entries (when `ENABLE_BB_HOT=1`) |
| `IKP_NVBIT_BRANCH_SITE_CAP` | `65536` | Max branch site entries (when `ENABLE_BRANCH_SITES=1`) |
| `IKP_NVBIT_TRACE_ADDR` | `0` | Legacy alias for memtrace mode |

---

## Interpreting Results

### Instruction mix analysis

From the `region_stats` output, you can compute:

```
Region 1 (compute loop):
  Total instructions: 34,078,720
  FP32 ALU:          8,454,144  (24.8%)
  INT ALU:          16,809,984  (49.3%)
  Branch:            8,388,608  (24.6%)

→ This loop is dominated by integer address computation and branch overhead.
  The actual FP32 compute is only 25% of instructions.
  Optimization: increase loop unrolling to reduce branch/integer overhead.
```

### Memory analysis

```
Region 2 (store):
  Total instructions: 491,520
  Global stores:      32,768 (32K threads × 4B = 128 KB)

→ 4 bytes per thread, fully coalesced (32 threads × 4B = 128B per transaction).
```

### Branch divergence

The `branch_div_hist` array shows how many branch instances had N active threads:
- `branch_div_hist[32]` = fully converged (all 32 threads active)
- `branch_div_hist[1]` = maximally diverged (1 thread active)
- High counts at low indices indicate divergence problems.

---

## Instruction Pipeline Attribution (`inst_pipe`)

When `IKP_NVBIT_ENABLE_INST_PIPE=1`, each region's stats include an `inst_pipe` field
with per-pipeline instruction counts. This provides finer-grained classification than
`inst_class`.

### Run

```bash
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_ENABLE_INST_PIPE=1 \
IKP_NVBIT_KERNEL_REGEX=my_kernel \
IKP_NVBIT_TRACE_PATH=./nvbit_out \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./my_kernel_nvbit
```

### Output

The `inst_pipe` field in `region_stats_*.json` provides a per-pipeline breakdown:

```json
{
  "region": 1,
  "inst_pipe": {
    "ld": 65536,
    "st": 32768,
    "tex": 0,
    "uniform": 0,
    "fp32": 8454144,
    "fp16": 0,
    "fp64": 0,
    "int": 16809984,
    "sfu": 0,
    "tensor": 0,
    "barrier": 0,
    "membar": 0,
    "branch": 8388608,
    "callret": 65536,
    "special": 0,
    "other": 688128
  }
}
```

### Pipeline Categories

| Pipeline | Description |
|----------|-------------|
| `ld` | Load instructions (LDG, LDS, LDL, etc.) |
| `st` | Store instructions (STG, STS, STL, etc.) |
| `tex` | Texture/surface operations |
| `uniform` | Uniform datapath operations |
| `fp32` | FP32 arithmetic (FADD, FMUL, FFMA) |
| `fp16` | FP16/BF16 arithmetic |
| `fp64` | FP64 arithmetic (DADD, DMUL, DFMA) |
| `int` | Integer arithmetic (IADD, IMAD, ISETP) |
| `sfu` | Special function unit (MUFU: sin, cos, rsqrt) |
| `tensor` | Tensor Core operations (HMMA, WGMMA) |
| `barrier` | Barrier synchronization (BAR.SYNC) |
| `membar` | Memory barriers (MEMBAR) |
| `branch` | Branch instructions (BRA, JMP) |
| `callret` | Call/return (CAL, RET) |
| `special` | Special instructions (S2R, CS2R) |
| `other` | All other instructions |

---

## Basic Block Hotspots and Branch Sites

When `IKP_NVBIT_ENABLE_BB_HOT=1` and/or `IKP_NVBIT_ENABLE_BRANCH_SITES=1`, an additional
`hotspots_*.json` file is produced with per-basic-block execution counts and per-branch
taken/fallthrough analysis.

### Run

```bash
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_ENABLE_BB_HOT=1 \
IKP_NVBIT_ENABLE_BRANCH_SITES=1 \
IKP_NVBIT_KERNEL_REGEX=my_kernel \
IKP_NVBIT_TRACE_PATH=./nvbit_out \
LD_PRELOAD=tools/nvbit_region_profiler/region_profiler.so \
./my_kernel_nvbit
```

### Output (`hotspots_*.json`)

```json
{
  "kernel": "my_kernel",
  "bb_entries": [
    {
      "function": "my_kernel",
      "bb_idx": 0,
      "start_pc": 0,
      "num_instrs": 12,
      "exec_count": 32768,
      "region": 0
    },
    {
      "function": "my_kernel",
      "bb_idx": 3,
      "start_pc": 256,
      "num_instrs": 4,
      "exec_count": 8388608,
      "region": 1
    }
  ],
  "branch_sites": [
    {
      "function": "my_kernel",
      "pc": 304,
      "taken": 8355840,
      "fallthrough": 32768,
      "region": 1
    }
  ]
}
```

### Interpreting Hotspot Data

- **bb_entries**: Sorted by `exec_count` descending. The hottest basic blocks are where
  the GPU spends the most dynamic instructions. Use `start_pc` to correlate with SASS
  and CUPTI per-PC data.

- **branch_sites**: Each branch instruction reports its taken vs. fallthrough counts.
  - `taken >> fallthrough` → loop back-edge (expected for hot loops)
  - `taken ≈ fallthrough` → conditional branch (potential divergence source)
  - Combine with `branch_div_hist` in `region_stats` to assess warp-level divergence.

---

## SASS Metadata and Line Info

### SASS metadata (`IKP_NVBIT_DUMP_SASS_META=1`)

Annotates each SASS instruction in the dump with:
- Opcode classification
- Predicate register usage
- Memory space (global/shared/local)
- Load/store width flags

### Source line info (`IKP_NVBIT_DUMP_SASS_LINEINFO=1`)

If the binary was compiled with `-lineinfo`, each instruction is annotated with the
source file and line number.

### nvdisasm SASS (`IKP_NVBIT_DUMP_NVDISASM_SASS=1`)

Uses `nvdisasm` (from the CUDA toolkit) to produce higher-quality SASS listings with
proper instruction encoding. Requires `nvdisasm` in `PATH` and `IKP_NVBIT_KEEP_CUBIN=1`
to preserve the extracted CUBIN file.

---

## Analysis Scripts

Several post-processing scripts are available in the top-level `scripts/` directory.

### `nvbit_locality.py` — Memory Locality Analysis

Computes reuse distance histograms, working set sizes, and inter-warp/inter-CTA
sharing from NVBit memory traces.

```bash
python3 scripts/nvbit_locality.py \
  --trace nvbit_out/mem_trace_*.jsonl \
  --region-stats nvbit_out/region_stats_*.json \
  --out locality_analysis.json \
  --line-bytes 128 \
  --window-records 128,512,2048
```

Output fields:
- `reuse_distance`: Histogram of cache line reuse distances (global/CTA/warp scope)
- `working_set`: Unique cache lines in sliding windows (avg, p50, p95, max)
- `inter_warp_sharing`: How many warps access each cache line
- `inter_cta_sharing`: How many CTAs access each cache line
- `lines_per_k_inst`: Cache lines per 1000 instructions (memory intensity)

### `nvbit_viz.py` / `ikp_viz_mpl.py` — Visualization Dashboards

Matplotlib-based dashboards for region stats, instruction mix, and memory patterns.

```bash
python3 scripts/nvbit_viz.py --input nvbit_out/region_stats_*.json --out nvbit_dashboard.png
```

### `validate_json.py` — JSON Output Validator

Lightweight schema validation for all Intra-Kernel Profiler output files (CUPTI, NVBit, merge).

```bash
python3 scripts/validate_json.py \
  cupti_out/sassmetrics_raw.json \
  nvbit_out/pc2region_*.json \
  --require-nonempty
```

Supported output types: `ikp_cupti_pcsamp`, `ikp_cupti_sassmetrics`, `ikp_cupti_instrexec`,
NVBit `pc2region`, and merge outputs.

### `pin_env.sh` — Environment Pinning

Captures GPU/CPU/driver/clock state into `data/env.json` for reproducibility. Optionally
locks GPU/memory clocks, sets power limits, and configures CPU governors.

```bash
# Read-only snapshot:
bash scripts/pin_env.sh

# Lock clocks for benchmarking (requires sudo):
bash scripts/pin_env.sh --apply --lock-gpu 1590,1590 --lock-mem 2600,2600
```
