# IKP Explorer

Self-contained single-page HTML dashboard that fuses trace timing, NVBit
instruction attribution, CUPTI hardware counters, PC sampling, memory traces,
and locality analysis into one interactive view (Monaco Editor + ECharts +
Split.js).


## Tabs

### 1. Overview

Kernel-wide summary: total instructions, active regions, bottleneck badge
(compute / memory / branch / balanced), total memory, trace event count, and
data source availability. A radar chart compares regions on 5 axes (FP32%,
Memory%, Branch%, Divergence%, Pred-off%). A data quality table shows which
profiler data streams are present. When both NVBit and CUPTI data exist, a
cross-validation table compares instruction mix fractions per region with delta
color-coding. Auto-generated bottleneck hints give a one-liner per region.

### 2. Line Metrics

Click a source line to see per-PC CUPTI SASS metrics: instruction count, active
thread %, predication efficiency, hotness, plus a bar chart of the top 8 metrics
and a full sortable metrics table. If the line has no per-PC data (compiler
merged it), falls back to region-level summary with links to nearby lines that
have data.

### 3. Regions

Select a region to see its full breakdown across up to 10 sections:

| Section | Content |
|---------|---------|
| **Summary** | 8 cards: instructions, pred-off%, BB exec, bottleneck, gmem, smem, spill, entropy |
| **Instruction Class** | Donut chart of 16 instruction categories (alu_fp32, tensor_wgmma, ld_global, ...) |
| **Pipeline** | Horizontal bar of per-pipeline instruction counts (requires `IKP_NVBIT_ENABLE_INST_PIPE=1`) |
| **Branch Divergence** | Two histograms (divergent / all branches) of active lane counts (0-32), plus avg/entropy stats |
| **Global Memory** | Coalescing efficiency bar, alignment histogram, stride classification pie, sectors-per-instruction chart |
| **Shared Memory** | Bank conflict histogram (0-32 way), address span histogram |
| **Local / Spill** | Local memory cards + red spill warning if register pressure detected |
| **CUPTI Efficiency** | 5 efficiency metrics (SIMT util, pred eff, coalescing, smem eff, branch uniformity) compared against whole-kernel, instruction mix breakdown, NVBit x CUPTI cross-validation |
| **Stall Profile** | Per-region PC sampling stall reasons attributed via pc2region |
| **InstrExec** | Per-region thread counts, warp executions, SIMT utilization, predication overhead |

### 4. Execution

Cross-region instruction-level comparison: 100% stacked bar chart of instruction
class mix across all regions, top-20 basic block hotspots with execution counts
(requires `IKP_NVBIT_ENABLE_BB_HOT=1`), and per-branch taken/fallthrough
analysis with divergence detection (requires `IKP_NVBIT_ENABLE_BRANCH_SITES=1`).

### 5. Memory

Memory traffic overview (global / shared / local pie chart), per-region locality
analysis with reuse distance histograms at 3 scopes (warp / CTA / global),
working set curves (avg / p50 / p95 / max), inter-warp and inter-CTA cache line
sharing charts. If raw memory trace is available, shows a 32-lane address
heatmap for visual coalescing inspection and per-PC coalescing summary table.

### 6. Stalls

Whole-kernel PC sampling stall profile (horizontal bar sorted by sample count),
SASS efficiency metrics bar (SIMT utilization, predication, coalescing, shared
mem, branch uniformity at 0-100%), per-region efficiency comparison chart, and
per-PC instruction execution analysis with thread utilization bars.

### 7. Trace

Per-region duration statistics: grouped bar chart (P50 / Mean / P95 / P99), full
percentile table (P5-P99 + min/max, with tail-latency warnings), per-region
duration histograms with zoomable bin sliders, Perfetto link for the Chrome
Trace file, and per-block x per-warp mean-duration heatmaps for load imbalance
detection.


**Cross-source fusion:** (1) NVBit x CUPTI join via pc2region gives per-region
hardware efficiency, stall attribution, and cross-validation. (2) CUPTI x Source
join via `-lineinfo` gives per-line metrics and editor hotness.