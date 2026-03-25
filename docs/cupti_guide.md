# CUPTI Injection Collectors Guide

The Intra-Kernel Profiler CUPTI suite provides four **injection libraries** (`.so`) that collect
per-PC profiling data from any CUDA application — no source changes required.
They are loaded via `CUDA_INJECTION64_PATH` and write structured JSON output.

---

## Overview

```
Your CUDA App                Injection Library (.so)              Output
┌──────────────────┐        ┌──────────────────────────┐        ┌──────────────────────────┐
│                  │        │ CUDA_INJECTION64_PATH    │        │                          │
│  kernel<<<...>>> │───────▶│ hooks driver/runtime API │───────▶│  *_raw.json              │
│                  │        │ collects per-PC data     │        │  (structured JSON)       │
│                  │        │ writes JSON on exit      │        │                          │
└──────────────────┘        └──────────────────────────┘        └──────────────────────────┘
```

| Collector | `.so` | What it collects | Output |
|-----------|-------|-----------------|--------|
| **PC Sampling** | `ikp_cupti_pcsamp.so` | Per-PC sample counts + stall reasons | `pcsampling_raw.json` |
| **SASS Metrics** | `ikp_cupti_sassmetrics.so` | Per-PC hardware counters (instruction exec, threads, etc.) | `sassmetrics_raw.json` |
| **InstructionExecution** | `ikp_cupti_instrexec.so` | Per-PC thread counts + predication | `instrexec_raw.json` |
| **PM Sampling** | `ikp_cupti_pmsamp.so` | Performance monitor sampling (CUDA 12.6+) | `pmsamp_raw.json` |

**Key advantage over `ncu`**: these collectors run as *injection libraries* — no replay,
no serialization, no `ncu` CLI overhead. Your application runs at (near) native speed.

---

## Prerequisites

- CUDA Toolkit with CUPTI (usually at `$CUDA_HOME/extras/CUPTI`)
- Some collectors require **admin/unrestricted profiling** privileges
  - On shared HPC clusters, PC sampling and InstructionExecution may return empty results with `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`
  - SASS metrics typically works even on restricted nodes
- Build the collectors: `make -C tools/cupti_region_profiler -j`

---

## Step 1: Build the Collectors

```bash
make -C tools/cupti_region_profiler -j
```

This produces four shared libraries:

```
tools/cupti_region_profiler/
├── ikp_cupti_pcsamp.so
├── ikp_cupti_sassmetrics.so
├── ikp_cupti_instrexec.so
└── ikp_cupti_pmsamp.so
```

## Step 2: Build a Target Kernel

Use any CUDA binary. A minimal example is provided:

```bash
cd examples/cupti
nvcc -O3 -std=c++17 -arch=sm_90a \
  minimal_cupti_target.cu -o minimal_cupti_target
```

The `cupti_target_kernel` performs FP32 multiply-add iterations — simple enough to profile,
complex enough to produce interesting stall distributions.

---

## Collector 1: PC Sampling (`ikp_cupti_pcsamp.so`)

PC sampling uses hardware counters to periodically snapshot which PC each warp is
executing and **why it stalled**. This is the single most useful profiling signal
for identifying bottlenecks.

### Run

```bash
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_pcsamp.so \
IKP_CUPTI_PCSAMP_OUT=./pcsampling_raw.json \
IKP_CUPTI_PCSAMP_COLLECTION_MODE=serialized \
IKP_CUPTI_PCSAMP_KERNEL_REGEX=cupti_target_kernel \
IKP_CUPTI_PCSAMP_PERIOD=5 \
IKP_CUPTI_PCSAMP_MAX_PCS=10000 \
IKP_CUPTI_PCSAMP_MAX_RECORDS=0 \
IKP_CUPTI_PCSAMP_VERBOSE=1 \
./minimal_cupti_target --iters=20 --inner=4096
```

### Output Format (`pcsampling_raw.json`)

```json
{
  "tool": "ikp_cupti_pcsamp",
  "version": 1,
  "pid": 214283,
  "timestamp_ns": 1770842784959692972,
  "collection_mode": "serialized",
  "sampling_period": 5,
  "kernel_regex": "cupti_target_kernel",

  "stall_reason_table": [
    { "index": 0, "name": "smsp__pcsamp_sample_count" },
    { "index": 2, "name": "smsp__pcsamp_warps_issue_stalled_barrier" },
    { "index": 14, "name": "smsp__pcsamp_warps_issue_stalled_long_scoreboard" },
    ...
  ],

  "invocations": [
    {
      "invocation_uid": "ctx1-seq0",
      "context_uid": 1,
      "correlation_id": 131,
      "kernel_name": "cupti_target_kernel",
      "stream": 0,
      "grid": [4096, 1, 1],
      "block": [256, 1, 1],
      "selected": true
    }
  ],

  "ranges": [
    {
      "context_uid": 1,
      "range_id": 0,
      "total_samples": 52480,
      "dropped_samples": 0,
      "hardware_buffer_full": 0
    }
  ],

  "pc_records": [
    {
      "cubinCrc": 12345678,
      "pcOffset": 256,
      "functionIndex": 0,
      "functionName": "cupti_target_kernel",
      "correlationId": 131,
      "selected": true,
      "stall": [
        { "reasonIndex": 0, "samples": 1024 },
        { "reasonIndex": 14, "samples": 890 },
        { "reasonIndex": 16, "samples": 134 }
      ]
    }
  ],

  "warnings": []
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `stall_reason_table` | Maps `reasonIndex` → human-readable stall reason name |
| `pc_records[].pcOffset` | PC offset within the CUBIN — the join key with NVBit `pc2region` |
| `pc_records[].stall` | Per-stall-reason sample counts for this PC |
| `ranges[].total_samples` | Total samples collected in this range |
| `ranges[].dropped_samples` | Samples lost (hardware buffer overflow) |
| `invocations[].selected` | Whether this kernel matched the regex filter |

### Stall Reason Reference

The stall reason names follow NVIDIA's CUPTI naming convention. Here are the most
important ones for performance analysis:

| Stall Reason | What It Means | Optimization Hint |
|-------------|---------------|-------------------|
| `barrier` | Waiting at `__syncthreads()` or `bar.sync` | Reduce sync frequency; overlap work |
| `long_scoreboard` | Waiting for global/local memory load | Increase occupancy; prefetch; improve locality |
| `short_scoreboard` | Waiting for shared memory or L1 | Check bank conflicts; reduce smem pressure |
| `math_pipe_throttle` | Math pipe is full (good!) | This means compute-bound — the GPU is working |
| `lg_throttle` | Load/store unit is full | Memory-bound; reduce memory pressure |
| `mio_throttle` | Memory I/O pipe throttle | Memory subsystem saturated |
| `tex_throttle` | Texture/surface unit throttle | Texture cache pressure |
| `wait` | Waiting on `nanosleep` / warp-group arrive | Expected for warp-specialized kernels |
| `warpgroup_arrive` | Waiting for warp-group synchronization | Normal for WGMMA/TMA pipelines |
| `not_selected` | Eligible but not chosen by scheduler | Too many eligible warps; consider reducing occupancy |
| `no_instructions` | No valid instructions to issue | Usually after branch; may indicate I-cache miss |
| `sleeping` | Warp is explicitly sleeping | Expected for producer/consumer patterns |
| `drain` | Warp is being retired | Normal at kernel end |
| `imc_miss` | Instruction cache miss | Kernel code too large; reduce code footprint |
| `membar` | Waiting on memory barrier (`membar.gl`) | Reduce fence frequency |
| `branch_resolving` | Branch target being resolved | Reduce divergent branches |

**Tip**: `*_not_issued` variants mean the warp was stalled AND not even considered for
scheduling (vs. stalled but still in the eligible pool).

### Collection Modes

| Mode | Behavior | Use When |
|------|----------|----------|
| `serialized` | Samples collected per-kernel, barriers inserted | You want per-kernel isolation |
| `continuous` | Background sampling across kernel boundaries | Low overhead; long-running apps |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_CUPTI_PCSAMP_ENABLE` | `1` | Enable/disable |
| `IKP_CUPTI_PCSAMP_OUT` | `pcsampling_raw.{pid}.json` | Output path |
| `IKP_CUPTI_PCSAMP_COLLECTION_MODE` | `serialized` | `serialized` or `continuous` |
| `IKP_CUPTI_PCSAMP_PERIOD` | `0` | Sampling period (GPU cycles between samples) |
| `IKP_CUPTI_PCSAMP_MAX_PCS` | `5000` | Max PC entries per collection |
| `IKP_CUPTI_PCSAMP_MAX_RECORDS` | `0` | Max total records (0 = unlimited) |
| `IKP_CUPTI_PCSAMP_KERNEL_REGEX` | all | Regex to filter kernel names |
| `IKP_CUPTI_PCSAMP_VERBOSE` | `0` | Verbosity (0–3) |
| `IKP_CUPTI_PCSAMP_SCRATCH_BUF_BYTES` | `0` | Scratch buffer size |
| `IKP_CUPTI_PCSAMP_HW_BUF_BYTES` | `0` | Hardware buffer size |
| `IKP_CUPTI_PCSAMP_DRAIN_EVERY_N` | `0` | Drain every N launches (continuous mode) |
| `IKP_CUPTI_PCSAMP_DRAIN_INTERVAL_MS` | `50` | Time-based drain interval |
| `IKP_CUPTI_PCSAMP_USE_START_STOP` | `0` | Per-kernel start/stop control |
| `IKP_CUPTI_PCSAMP_ACTIVITY_BUFFER_BYTES` | `1048576` | Activity buffer size |
| `IKP_CUPTI_PCSAMP_WORKER_SLEEP_MS` | `0` | Worker thread sleep interval (ms; 0 = busy poll) |
| `IKP_CUPTI_PCSAMP_DRAIN_RETRY_MS` | `0` | Retry interval for final drain (ms) |
| `IKP_CUPTI_PCSAMP_DRAIN_RETRY_ITERS` | `0` | Max retries for final drain |
| `IKP_CUPTI_PCSAMP_ENABLE_KERNEL_ACTIVITY` | auto | Track kernel activity records (auto-enabled when KERNEL_REGEX is set) |

---

## Collector 2: SASS Metrics (`ikp_cupti_sassmetrics.so`)

SASS metrics provides **per-PC hardware counter values** — instruction execution counts,
thread-level execution, predication. Unlike PC sampling (statistical), SASS metrics
are **exact counts** via code patching.

### Run

```bash
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_sassmetrics.so \
IKP_CUPTI_SASS_OUT=./sassmetrics_raw.json \
IKP_CUPTI_SASS_PROFILE=core \
IKP_CUPTI_SASS_LAZY_PATCHING=1 \
IKP_CUPTI_SASS_ENABLE_SOURCE=0 \
./minimal_cupti_target --iters=20 --inner=4096
```

### Metric Profiles

Metrics are defined in `tools/cupti_region_profiler/config/metrics_profiles.json`:

```json
{
  "profiles": {
    "core": ["smsp__sass_inst_executed"],
    "divergence": [
      "smsp__sass_inst_executed",
      "smsp__sass_thread_inst_executed",
      "smsp__sass_thread_inst_executed_pred_on"
    ],
    "memory": [
      "smsp__sass_inst_executed",
      "smsp__sass_inst_executed_op_global_ld",
      "smsp__sass_inst_executed_op_global_st",
      "smsp__sass_inst_executed_op_shared_ld",
      "smsp__sass_inst_executed_op_shared_st",
      "smsp__sass_sectors_mem_global",
      "smsp__sass_sectors_mem_global_ideal"
    ],
    "instruction_mix": [
      "smsp__sass_inst_executed",
      "smsp__sass_inst_executed_op_branch",
      "smsp__sass_inst_executed_op_global",
      "smsp__sass_inst_executed_op_shared",
      "smsp__sass_inst_executed_op_tma",
      "smsp__sass_inst_executed_op_shared_gmma",
      "smsp__sass_thread_inst_executed"
    ],
    "branch": [
      "smsp__sass_inst_executed_op_branch",
      "smsp__sass_thread_inst_executed_op_branch",
      "smsp__sass_branch_targets_threads_divergent",
      "smsp__sass_branch_targets_threads_uniform"
    ]
  }
}
```

| Profile | Metrics | Use Case |
|---------|---------|----------|
| `core` | Instruction execution count | Basic hotspot analysis |
| `divergence` | Inst executed + thread counts + pred-on | Divergence/predication analysis |
| `memory` | Global/shared loads/stores + sector counts | Memory efficiency, coalescing |
| `instruction_mix` | Branch, global, shared, TMA, WGMMA counts | Instruction class breakdown |
| `branch` | Branch targets: divergent vs uniform | Branch divergence analysis |

To discover all available metrics on your GPU:

```bash
IKP_CUPTI_SASS_LIST=1 \
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_sassmetrics.so \
./minimal_cupti_target
```

On GH200 (sm_90), this lists 60 metrics including TMA and WGMMA-specific counters.

### Output Format (`sassmetrics_raw.json`)

Real output from `cupti_target_kernel` (1M threads, 4096 inner iterations, `core` profile):

```json
{
  "tool": "ikp_cupti_sassmetrics",
  "version": 1,
  "pid": 1575772,
  "timestamp_ns": 1774293411079535150,
  "metrics_profile": "core",
  "metrics_json": ".../tools/cupti_region_profiler/config/metrics_profiles.json",
  "lazy_patching": true,
  "per_launch": false,
  "enable_source": false,

  "invocations": [
    {
      "invocation_uid": "ctx1-seq0",
      "context_uid": 1,
      "correlation_id": 1,
      "kernel_name": "cupti_target_kernel",
      "stream": 0,
      "grid": [4096, 1, 1],
      "block": [256, 1, 1],
      "shared_mem_bytes": 0,
      "selected": true
    }
  ],

  "records": [
    {
      "cubinCrc": 602204138,
      "functionIndex": 0,
      "functionName": "cupti_target_kernel",
      "pcOffset": 0,
      "correlationId": 0,
      "metrics": { "smsp__sass_inst_executed": 32768 }
    },
    {
      "cubinCrc": 602204138,
      "functionIndex": 0,
      "functionName": "cupti_target_kernel",
      "pcOffset": 288,
      "correlationId": 0,
      "metrics": { "smsp__sass_inst_executed": 134217728 }
    }
  ],

  "warnings": []
}
```

**Note:** Each metric is collected in a separate instrumentation pass. When using
multi-metric profiles (like `divergence`), records appear N times per PC (once per
metric). Merge records by `pcOffset` to get the full picture for each instruction.

### Key Fields

| Field | Description |
|-------|-------------|
| `records[].pcOffset` | PC offset — join key with NVBit `pc2region` |
| `records[].metrics` | Map of metric name → exact counter value |
| `records[].source` | Optional source file/line (requires `ENABLE_SOURCE=1`) |
| `metrics_profile` | Which profile from `metrics_profiles.json` was used |

### Interpreting Results

Real analysis of `cupti_target_kernel` with `divergence` profile (24 unique PCs, 72 raw records):

```
=== Merged by pcOffset (24 unique PCs) ===
  PC=0x0000  warp_inst=     32,768  thread_inst=   1,048,576  pred_on=   1,048,576  active=1.0000  pred_on=1.0000
  PC=0x0070  warp_inst=     32,768  thread_inst=   1,048,576  pred_on=           0  active=1.0000  pred_on=0.0000  ← predicated branch
  PC=0x0120  warp_inst=134,217,728  thread_inst=4,294,967,296  pred_on=4,294,967,296  active=1.0000  pred_on=1.0000  ← hot loop body
  PC=0x0150  warp_inst=134,217,728  thread_inst=4,294,967,296  pred_on=4,293,918,720  active=1.0000  pred_on=0.9998  ← conditional store

=== Aggregate (all 20 iters × 1M threads) ===
  Total warp instructions:          537,526,272
  Total thread instructions:     17,200,840,704
  Thread instructions (pred on): 17,197,694,976
  Overall active ratio:          1.0000 (fully converged)
  Overall pred-on ratio:         0.9998 (negligible predication)
```

Insights:
- 4 PCs dominate (the inner FP32 loop body), each executing 134M warp instructions
- `active_ratio=1.0` everywhere — no branch divergence (uniform kernel)
- PC 0x0150 has `pred_on=0.9998` — the `if (idx < n)` guard for the final store

### Available Metrics (sm_90 / GH200)

| Metric | Description |
|--------|-------------|
| `smsp__sass_inst_executed` | Total warp-level instruction executions |
| `smsp__sass_thread_inst_executed` | Total thread-level instruction executions |
| `smsp__sass_thread_inst_executed_pred_on` | Thread executions where predicate was ON |
| `smsp__sass_inst_executed_op_global_ld/st` | Global load/store instruction count |
| `smsp__sass_inst_executed_op_shared_ld/st` | Shared memory load/store count |
| `smsp__sass_inst_executed_op_tma_ld/st` | TMA load/store instruction count |
| `smsp__sass_inst_executed_op_shared_gmma` | WGMMA (Tensor Core) instruction count |
| `smsp__sass_inst_executed_op_branch` | Branch instruction count |
| `smsp__sass_sectors_mem_global` | Actual global memory sectors transferred |
| `smsp__sass_sectors_mem_global_ideal` | Ideal sectors (perfectly coalesced) |
| `smsp__sass_branch_targets_threads_divergent` | Branch targets with divergent threads |
| `smsp__sass_branch_targets_threads_uniform` | Branch targets with uniform threads |

**Memory coalescing ratio** per PC:

```
coalescing = sectors_mem_global_ideal / sectors_mem_global
```

A ratio of 1.0 means perfectly coalesced; less than 1.0 indicates wasted memory transactions.

**Divergence ratio** per PC (merge records by pcOffset first):

```
divergence = 1.0 - (thread_inst_executed_pred_on / (inst_executed × 32))
```

A ratio of 0 means fully converged (all 32 threads active); close to 1 means
maximally diverged.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_CUPTI_SASS_ENABLE` | `1` | Enable/disable |
| `IKP_CUPTI_SASS_OUT` | `sassmetrics_raw.{pid}.json` | Output path |
| `IKP_CUPTI_SASS_PROFILE` | `core` | Profile name from `metrics_profiles.json` |
| `IKP_CUPTI_SASS_METRICS_JSON` | auto (next to `.so`) | Path to metrics profile JSON |
| `IKP_CUPTI_SASS_LAZY_PATCHING` | `1` | Inject instrumentation on-demand |
| `IKP_CUPTI_SASS_PER_LAUNCH` | `0` | Isolate metrics per kernel launch |
| `IKP_CUPTI_SASS_PER_LAUNCH_SYNC` | `0` | Use `cuCtxSynchronize` boundaries (high overhead) |
| `IKP_CUPTI_SASS_ENABLE_SOURCE` | `0` | Include source file/line info |
| `IKP_CUPTI_SASS_MAX_RECORDS` | `0` | Max records (0 = unlimited) |
| `IKP_CUPTI_SASS_MAX_CUBIN_BYTES` | `536870912` | CUBIN cache cap (512 MB) |
| `IKP_CUPTI_SASS_KERNEL_REGEX` | all | Regex filter (requires `PER_LAUNCH=1`) |
| `IKP_CUPTI_SASS_LIST` | `0` | List available metrics and exit |
| `IKP_CUPTI_SASS_LIST_OUT` | stdout | Where to write metric list |
| `IKP_CUPTI_SASS_METRICS` | (none) | Comma-separated metric names to override profile selection |
| `IKP_CUPTI_SASS_ACTIVITY_BUFFER_BYTES` | `1048576` | CUPTI activity buffer size (bytes; min 4096) |

### Walkthrough: All Profiles on `cupti_target_kernel`

Real results from profiling `cupti_target_kernel` (1M threads, 4096 inner iterations of `x = x * 1.00001f + 0.00001f`):

**Memory profile** — coalescing analysis:

```
smsp__sass_inst_executed:         537,526,272
smsp__sass_inst_executed_op_global_st:  32,768     (1M threads × 1 store each ÷ 32 = 32K warps)
smsp__sass_sectors_mem_global:        131,072     (32K warps × 4 sectors per warp)
smsp__sass_sectors_mem_global_ideal:  131,072     ← matches actual → perfectly coalesced
coalescing_ratio: 1.0000
```

**Branch profile** — zero divergence:

```
smsp__sass_inst_executed_op_branch:           134,250,496
smsp__sass_branch_targets_threads_uniform:    134,250,496  ← all branches are uniform
smsp__sass_branch_targets_threads_divergent:            0  ← no divergence
```

**Instruction mix profile** — 25% branches, 75% ALU:

```
smsp__sass_inst_executed:            537,526,272  (total)
smsp__sass_inst_executed_op_branch:  134,250,496  (25.0% — loop back-edge)
smsp__sass_inst_executed_op_global:       32,768  ( 0.0% — single store per thread)
smsp__sass_thread_inst_executed:  17,200,840,704  (32× warp count = all threads active)
```

---

## Collector 3: InstructionExecution (`ikp_cupti_instrexec.so`)

InstructionExecution provides **exact execution counts and thread participation**
for every dynamic instruction. This is the most direct way to measure branch
divergence and predication effects.

### Run

```bash
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_instrexec.so \
IKP_CUPTI_INSTREXEC_OUT=./instrexec_raw.json \
IKP_CUPTI_INSTREXEC_KERNEL_REGEX=cupti_target_kernel \
IKP_CUPTI_INSTREXEC_MAX_RECORDS=0 \
./minimal_cupti_target --iters=20 --inner=4096
```

### Output Format (`instrexec_raw.json`)

```json
{
  "tool": "ikp_cupti_instrexec",
  "version": 1,
  "pid": 214283,
  "timestamp_ns": 1770842784959692972,

  "invocations": [
    {
      "invocation_uid": "ctx1-seq0",
      "kernel_name": "cupti_target_kernel",
      "grid": [4096, 1, 1],
      "block": [256, 1, 1],
      "selected": true
    }
  ],

  "records": [
    {
      "cubinCrc": 12345678,
      "functionId": 0,
      "functionIndex": 0,
      "functionName": "cupti_target_kernel",
      "pcOffset": 256,
      "correlationId": 131,
      "threadsExecuted": 33554432,
      "notPredOffThreadsExecuted": 33554432,
      "executed": 1048576,
      "source": {
        "file": "minimal_cupti_target.cu",
        "line": 15
      }
    }
  ],

  "warnings": []
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `executed` | Warp-level execution count for this PC |
| `threadsExecuted` | Total threads that reached this PC |
| `notPredOffThreadsExecuted` | Threads where the predicate was ON (actually executed) |
| `source` | Source file/line mapping |

### Divergence Analysis

For each instruction:

```
active_thread_ratio = threadsExecuted / (executed × 32)
predication_ratio   = notPredOffThreadsExecuted / threadsExecuted
effective_ratio     = notPredOffThreadsExecuted / (executed × 32)
```

| Ratio | Value | Meaning |
|-------|-------|---------|
| `active_thread_ratio` | 1.0 | All 32 threads reach this instruction |
| `active_thread_ratio` | 0.5 | Only 16/32 threads active (branch divergence) |
| `predication_ratio` | 1.0 | No predicated-off threads |
| `predication_ratio` | 0.5 | Half the threads are predicated off |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_CUPTI_INSTREXEC_ENABLE` | `1` | Enable/disable |
| `IKP_CUPTI_INSTREXEC_OUT` | `instrexec_raw.{pid}.json` | Output path |
| `IKP_CUPTI_INSTREXEC_BUFFER_BYTES` | `1048576` | Activity buffer size |
| `IKP_CUPTI_INSTREXEC_MAX_RECORDS` | `0` | Max records (0 = unlimited) |
| `IKP_CUPTI_INSTREXEC_KERNEL_REGEX` | all | Regex to filter kernels |
| `IKP_CUPTI_INSTREXEC_ALLOW_CORRID0` | `0` | Include records with unknown correlationId |

---

## Collector 4: PM Sampling (`ikp_cupti_pmsamp.so`)

PM sampling provides low-overhead performance monitor counter sampling. On Hopper
(sm_90a) and later architectures, PM sampling replaces the traditional PC sampling
mechanism — it is the mechanism `ncu --set full` uses internally for stall-reason
and warp scheduling analysis.

> **Current status:** The collector is a **stub** — it outputs a `"not_supported"`
> JSON. The CUPTI PM sampling API (`cupti_pmsampling.h`) is available starting from
> CUDA 12.6, but implementing a full collector requires the NVIDIA Profiler Host
> library for metric config image creation and counter data decoding. This is planned
> for a future release.
>
> **Workaround for Hopper stall analysis:** Use `ncu --set full` which collects
> PM sampling data internally, or use SASS metrics (`ikp_cupti_sassmetrics.so`) for
> per-PC instruction counts, which works on all architectures.

### Run

```bash
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_pmsamp.so \
IKP_CUPTI_PMSAMP_OUT=./pmsamp_raw.json \
./minimal_cupti_target
```

### Output (stub)

```json
{
  "tool": "ikp_cupti_pmsamp",
  "version": 1,
  "pid": 214283,
  "timestamp_ns": 1770842784959692972,
  "not_supported": true,
  "reason": "CUDA 12.4 does not provide cupti_pmsampling.h; upgrade to 12.6+"
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IKP_CUPTI_PMSAMP_OUT` | `pmsamp_raw.{pid}.json` | Output path |

---

## Run All Collectors at Once

The demo script runs all collectors end-to-end:

```bash
bash scripts/run_all_examples.sh --out=_demo_out
```

Or manually, against the same target:

```bash
cd examples/cupti
nvcc -O3 -std=c++17 -arch=sm_90a minimal_cupti_target.cu -o minimal_cupti_target

# 1) PC sampling
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_pcsamp.so \
IKP_CUPTI_PCSAMP_OUT=./pcsampling_raw.json \
IKP_CUPTI_PCSAMP_COLLECTION_MODE=serialized \
IKP_CUPTI_PCSAMP_KERNEL_REGEX=cupti_target_kernel \
IKP_CUPTI_PCSAMP_PERIOD=5 \
IKP_CUPTI_PCSAMP_MAX_PCS=10000 \
./minimal_cupti_target --iters=20 --inner=4096

# 2) SASS metrics
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_sassmetrics.so \
IKP_CUPTI_SASS_OUT=./sassmetrics_raw.json \
IKP_CUPTI_SASS_PROFILE=divergence \
IKP_CUPTI_SASS_LAZY_PATCHING=1 \
./minimal_cupti_target --iters=20 --inner=4096

# 3) Instruction execution
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_instrexec.so \
IKP_CUPTI_INSTREXEC_OUT=./instrexec_raw.json \
IKP_CUPTI_INSTREXEC_KERNEL_REGEX=cupti_target_kernel \
./minimal_cupti_target --iters=20 --inner=4096

# 4) PM sampling (CUDA 12.6+ only)
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_pmsamp.so \
IKP_CUPTI_PMSAMP_OUT=./pmsamp_raw.json \
./minimal_cupti_target --iters=20 --inner=4096
```

---

## Joining NVBit Regions with CUPTI Data

The most powerful workflow combines **NVBit pc2region mapping** with **CUPTI per-PC
metrics** to produce region-level hardware statistics.

### Step 1: Collect NVBit pc2region

Build the target with NVBit markers (see [`nvbit_guide.md`](nvbit_guide.md)):

```bash
cd examples/region_demo
nvcc -O3 -std=c++17 -arch=sm_90a -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I ../../include \
  minimal_region_target.cu ../../src/nvbit_marker_device.cu \
  -o minimal_region_target
```

Run with NVBit injection:

```bash
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=region_demo_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_TRACE_PATH=./nvbit_out \
LD_PRELOAD=../../tools/nvbit_region_profiler/region_profiler.so \
./minimal_region_target --iters=20 --inner=4096
```

This produces `nvbit_out/pc2region_*.json`:

```json
{
  "kernel": "region_demo_kernel",
  "pc2region": [
    { "pc_offset": 256, "dominant_region": 1, "executed_count": 1048576 },
    { "pc_offset": 512, "dominant_region": 2, "executed_count": 32768 }
  ]
}
```

### Step 2: Collect CUPTI SASS metrics

Run the **same binary** (without NVBit, without `-DIKP_ENABLE_NVBIT_MARKERS` rebuild
needed — the binary is unchanged) with CUPTI:

```bash
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_sassmetrics.so \
IKP_CUPTI_SASS_OUT=./cupti_out/sassmetrics_raw.json \
IKP_CUPTI_SASS_PROFILE=divergence \
IKP_CUPTI_SASS_LAZY_PATCHING=1 \
./minimal_region_target --iters=20 --inner=4096
```

### Step 3: Join on pcOffset

Both files share the `pcOffset` key. Use the provided analysis script:

```bash
python3 scripts/analyze_cupti_join.py \
  --nvbit-dir ./nvbit_out \
  --cupti-dir ./cupti_out \
  --labels "0:outside_markers,1:compute_loop,2:store"
```

Or do a manual Python join:

```python
import json, glob

# Load NVBit pc2region
pc2region = {}
for f in glob.glob("nvbit_out/pc2region_*.json"):
    with open(f) as fh:
        data = json.load(fh)
    for entry in data["pc2region"]:
        pc2region[entry["pc_offset"]] = entry["dominant_region"]

# Load CUPTI SASS metrics and merge records by pcOffset
with open("cupti_out/sassmetrics_raw.json") as f:
    sass = json.load(f)

merged = {}  # pcOffset -> {metric -> value}
for rec in sass["records"]:
    pc = rec["pcOffset"]
    if pc not in merged:
        merged[pc] = {}
    for metric, value in rec["metrics"].items():
        if value > 0:
            merged[pc][metric] = value

# Join: aggregate by region
region_metrics = {}
for pc, metrics in merged.items():
    region = pc2region.get(pc, -1)
    if region == -1:
        continue  # PC from marker overhead, skip
    if region not in region_metrics:
        region_metrics[region] = {}
    for m, v in metrics.items():
        region_metrics[region][m] = region_metrics[region].get(m, 0) + v

for region, metrics in sorted(region_metrics.items()):
    print(f"Region {region}: {metrics}")
```

Real result from `region_demo_kernel` (divergence profile):

```
=== NVBit + CUPTI Join: Divergence by Region ===

  Region 0 (outside_markers):
    smsp__sass_inst_executed:            819,200
    smsp__sass_thread_inst_executed:  22,151,168
    smsp__sass_thread_inst_executed_pred_on:  21,069,824
    active_thread_ratio:  0.8450
    pred_on_ratio:        0.9512

  Region 1 (compute_loop):
    smsp__sass_inst_executed:      537,133,056
    smsp__sass_thread_inst_executed:  17,187,209,216
    smsp__sass_thread_inst_executed_pred_on:  17,185,112,064
    active_thread_ratio:  0.9999
    pred_on_ratio:        0.9999
```

Cross-validation with NVBit `instmix`:

```
NVBit Region 0: inst_total=  1,212,416  (alu_fp32=65K, alu_int=229K, branch=65K)
NVBit Region 1: inst_total=537,395,200  (alu_fp32=134M, alu_int=268M, branch=134M)
NVBit Region 2: inst_total=    491,520  (st_global=32K, gmem_bytes=4MB)
```

The CUPTI SASS metrics (537M warp inst in Region 1) closely match NVBit counts (537M).
Region 0 has `active_ratio=0.845` due to marker setup/teardown overhead; Region 1
(the compute loop) is at `0.9999` — fully converged as expected.

---

## Choosing the Right Collector

| Question | Best Collector |
|----------|---------------|
| "Where is my kernel spending time?" | **PC Sampling** (sm_70–sm_89) or `ncu --set full` (sm_90+) |
| "Which instructions execute most?" | **SASS Metrics** (`core` profile) — all architectures |
| "Is my kernel divergent?" | **SASS Metrics** (`divergence` profile) or **InstructionExecution** |
| "How many threads are predicated off?" | **InstructionExecution** — exact pred-on/off counts |
| "What's the instruction mix per region?" | **NVBit** `instmix` mode (see [`nvbit_guide.md`](nvbit_guide.md)) |
| "End-to-end region analysis?" | **NVBit pcmap** + **SASS Metrics** join |

---

## Troubleshooting

### `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`

PC sampling and InstructionExecution require elevated profiling permissions. On shared
HPC clusters, you may see:

```
[ikp_cupti_pcsamp] WARN: ... INSUFFICIENT_PRIVILEGES ... disabling.
```

**Workaround**: Run on a node where you have exclusive access, or ask your admin to
enable unrestricted profiling (`/proc/driver/nvidia/params` or
`nvidia-modprobe -u -c 0`).

SASS metrics typically works even on restricted nodes.

### Empty `pc_records` or `records` arrays

If the output JSON has `"pc_records": []`, check:

1. **GPU architecture**: PC sampling is **not supported on sm_90a (Hopper / GH200)**.
   The CUPTI PC sampling API configures without errors and retrieves the stall reason
   table, but the hardware never produces PC-level sample records. This is confirmed by
   `ncu --query-metrics | grep pcsamp` returning zero results on sm_90a.
   Hopper uses **PM sampling** instead (see below). For stall-reason analysis on
   Hopper, use `ncu --set full` which collects via PM sampling internally.
2. **Privileges**: See above
3. **Kernel regex**: Verify your regex matches the (mangled) kernel name. Set
   `IKP_CUPTI_PCSAMP_VERBOSE=3` to see which kernels were detected.
4. **Short kernels**: Very fast kernels may finish before a sample is taken. Increase
   `--iters` or `--inner` to make kernels run longer.

### PC Sampling Architecture Support

| Architecture | Compute Capability | PC Sampling | Notes |
|-------------|-------------------|-------------|-------|
| Volta | sm_70 | Supported | Full per-PC stall reason breakdown |
| Turing | sm_75 | Supported | Full per-PC stall reason breakdown |
| Ampere | sm_80, sm_86 | Supported | Full per-PC stall reason breakdown |
| Ada Lovelace | sm_89 | Supported | Full per-PC stall reason breakdown |
| Hopper | sm_90, sm_90a | **Not supported** | Use PM sampling or `ncu --set full` |

On Hopper (GH200, H100), the stall reason table is retrievable but sample data
collection yields zero PC records regardless of buffer sizes or collection mode.
The `invocations` array will still correctly track kernel launches.
SASS metrics (`ikp_cupti_sassmetrics.so`) works correctly on all architectures.

### `metrics_profiles.json` not found

SASS metrics auto-locates `metrics_profiles.json` relative to the `.so` file,
checking both `<dir>/metrics_profiles.json` and `<dir>/config/metrics_profiles.json`.
If you move the `.so`, also copy the config, or set:

```bash
IKP_CUPTI_SASS_METRICS_JSON=/absolute/path/to/metrics_profiles.json
```

### Activity callback ownership conflict

Only one injection library can own CUPTI activity callbacks. If you load two collectors
simultaneously, one will print:

```
WARN: activity callbacks already owned by ...
```

**Solution**: Run collectors separately (one `CUDA_INJECTION64_PATH` per invocation).

---

## Comparison with Nsight Compute (`ncu`)

| Feature | Intra-Kernel Profiler CUPTI | Nsight Compute |
|---------|---------------|----------------|
| **Mechanism** | Injection (single pass) | Replay-based (multi-pass) |
| **Overhead** | Low (no kernel replay) | High (multiple replays per kernel) |
| **Metrics** | Per-PC via SASS patching | Full metric catalog |
| **Stall reasons** | Per-PC via PC sampling | Per-kernel aggregate |
| **Region attribution** | Via NVBit join | Manual source mapping |
| **Ease of use** | `CUDA_INJECTION64_PATH=...` | `ncu --set full ./app` |
| **Restricted clusters** | Some features limited | Some features limited |
| **JSON output** | Native | Requires `--csv` / `ncu-rep` |

Intra-Kernel Profiler CUPTI collectors are best for:
- **Production workloads** where replay overhead is unacceptable
- **Per-PC analysis** with region attribution via NVBit join
- **CI/CD integration** (JSON output, no GUI needed)
- **Warp-specialized kernels** where instruction-level stall analysis matters

---

## Custom Metric Override (`IKP_CUPTI_SASS_METRICS`)

Instead of using a predefined profile from `metrics_profiles.json`, you can specify
exact metric names via the `IKP_CUPTI_SASS_METRICS` environment variable:

```bash
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_sassmetrics.so \
IKP_CUPTI_SASS_OUT=./custom_metrics.json \
IKP_CUPTI_SASS_METRICS="smsp__sass_inst_executed,smsp__sass_inst_executed_op_global_ld,smsp__sass_inst_executed_op_tma_ld" \
./my_cuda_app
```

This bypasses profile selection entirely. Each metric is collected in a separate
instrumentation pass (N metrics → N passes), so the output will contain N×PCs records.
Merge by `pcOffset` to reconstruct per-PC metrics.

This is useful when:
- You need a specific combination not in any predefined profile
- You want to test a single metric in isolation
- You're exploring which metrics are available on your architecture

---

## Analysis Scripts

Post-processing scripts for merging, validating, and visualizing CUPTI outputs are
in the top-level `scripts/` directory.

### `ikp_cupti_sassmetrics_merge.py` — Region-Level SASS Metrics

Joins CUPTI SASS metrics with NVBit `pc2region` to produce per-region aggregated
hardware counter values.

```bash
python3 scripts/ikp_cupti_sassmetrics_merge.py \
  --sass sassmetrics_raw.json \
  --pc2region nvbit_out/pc2region_*.json \
  --out merged_sassmetrics.json
```

### `ikp_cupti_divergence_merge.py` — Divergence Analysis

Computes per-region active thread ratio, predication ratio, and lane efficiency from
the `divergence` SASS metrics profile.

```bash
python3 scripts/ikp_cupti_divergence_merge.py \
  --sass sassmetrics_raw.json \
  --pc2region nvbit_out/pc2region_*.json \
  --out divergence_report.json
```

### `ikp_cupti_pcsamp_merge.py` — PC Sampling Stall Aggregation

Aggregates PC sampling stall reasons by NVBit region.

```bash
python3 scripts/ikp_cupti_pcsamp_merge.py \
  --pcsamp pcsampling_raw.json \
  --pc2region nvbit_out/pc2region_*.json \
  --out stall_by_region.json
```

### `analyze_cupti_join.py` — Lightweight Join Script

A simpler join script (in `intra_kernel_profiler/scripts/`) that matches NVBit and CUPTI
data by pcOffset directly:

```bash
python3 scripts/analyze_cupti_join.py \
  --nvbit-dir nvbit_out/ \
  --cupti-dir cupti_out/ \
  --labels "0:outside,1:compute,2:store"
```

### `validate_json.py` — JSON Validator

```bash
python3 scripts/validate_json.py \
  sassmetrics_raw.json pcsampling_raw.json instrexec_raw.json \
  --require-nonempty
```

### `plot_trace_summary.py` — Trace Distribution Visualization

Plots per-region duration histograms from trace summary JSON files:

```bash
python3 scripts/plot_trace_summary.py \
  --summary trace_summary.json \
  --out_dir plots/
```

Produces individual per-region PNGs plus an overlay plot comparing all regions.
