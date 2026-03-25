# Trace Output Format Reference

`HostSession::write_trace(path, opt)` generates two files:

- `path` — **Chrome Trace JSON** (viewable in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`)
- `path` with `_summary.json` suffix — **Summary JSON** (aggregate statistics)

---

## Chrome Trace JSON

Top-level:

```json
{
  "displayTimeUnit": "ns",
  "traceEvents": [ ... ]
}
```

### Event types

**Metadata** — process and thread names for the viewer UI:

```json
{"ph":"M", "name":"process_name", "pid":124, "tid":0,
 "args":{"name":"sm 124"}}
{"ph":"M", "name":"thread_name", "pid":124, "tid":2,
 "args":{"name":"block 0 warp 2"}}
```

**Complete duration events** (default, `emit_complete_events=true`):

```json
{"name":"compute", "ph":"X", "ts":12345.0, "dur":758.0,
 "pid":124, "tid":2,
 "cname":"thread_state_running",
 "args":{"sm":124, "block":0, "warp":2}}
```

**Instant events** (for marks recorded with `IKP_TRACE_REC_M`):

```json
{"name":"checkpoint", "ph":"i", "s":"t", "ts":12345.0,
 "pid":124, "tid":2,
 "args":{"sm":124, "block":0, "warp":2}}
```

### pid/tid mapping

With `group_by_smid=true` (default):

| Field | Value | Viewer meaning |
|-------|-------|----------------|
| `pid` | SM ID | Each SM is a "process" (collapsible row group) |
| `tid` | `(block << 6) \| warp` | Each warp is a "thread" (individual timeline row) |

With `group_by_smid=false`:

| Field | Value | Viewer meaning |
|-------|-------|----------------|
| `pid` | block ID | Each CTA is a "process" |
| `tid` | `warp * 32` | Each warp is a "thread" |

---

## Summary JSON

Aggregates only **successfully paired** begin/end events (complete duration slices).

### Top-level fields

```json
{
  "trace": "gemm_trace.json",
  "displayTimeUnit": "ns",
  "scale": 1.0,
  "blocks": 256,
  "warps_per_block": 8,
  "per_warp_cap": 8192,
  "block_filter": [0, 1, 2, 3],
  "unmatched_begin": 0,
  "unmatched_end": 0,
  "regions": [ ... ],
  "by_block_warp_regions": { ... }
}
```

| Field | Description |
|-------|-------------|
| `scale` | Multiplier applied to raw globaltimer ticks |
| `unmatched_begin` | Begin events with no matching end. Should be 0. |
| `unmatched_end` | End events with no matching begin. Should be 0. |
| `block_filter` | Which blocks were included (empty = all) |

### Region entry

```json
{
  "region": 2,
  "name": "compute",
  "count": 2048,
  "mean_dur": 758.0,
  "cv_dur": 0.12,
  "var_dur_pop": 8281.0,
  "var_dur_sample": 8285.0,
  "min_dur": 544.0,
  "max_dur": 1408.0,
  "percentiles": {
    "p5": 640.0, "p10": 672.0, "p25": 704.0,
    "p50": 767.0, "p75": 800.0, "p90": 832.0,
    "p95": 830.0, "p99": 1024.0
  },
  "hist": {
    "bins": 128,
    "min": 544.0,
    "max": 1408.0,
    "prob": [0.05, 0.12, ...]
  }
}
```

| Field | Description |
|-------|-------------|
| `count` | Number of paired begin/end events |
| `mean_dur` | Mean duration (in scaled units) |
| `cv_dur` | Coefficient of variation (std/mean) |
| `percentiles` | p5 through p99 from the histogram |
| `hist.prob` | Normalized histogram (sums to 1.0) |

### Per-block-warp breakdown

When `summary_dump_by_block_warp=true` (default), each region also has per-(block, warp) stats:

```json
"by_block_warp_regions": {
  "region_compute": {
    "region": 2,
    "name": "compute",
    "by_block_warp": [
      {"block": 0, "warp": 0, "count": 64, "mean_dur": 762.0, ...},
      {"block": 0, "warp": 1, "count": 64, "mean_dur": 755.0, ...}
    ]
  }
}
```

This is useful for identifying per-warp load imbalance.

---

## TraceWriteOptions

```cpp
struct TraceWriteOptions {
  double scale = 1.0;                  // raw ticks → display units
  bool emit_complete_events = true;    // pair B/E into ph:"X" slices
  bool group_by_smid = true;           // pid=SM, tid=block|warp
  bool emit_summary_json = true;       // write *_summary.json
  uint32_t summary_hist_bins = 128;    // histogram resolution
  bool summary_dump_by_block_warp = true;

  bool emit_block_region_distributions = false;  // per-(block,region) JSONs
  uint32_t block_region_hist_bins = 128;

  // Top-K filters (0 = no limit)
  uint32_t summary_topk_block_warp_per_region = 0;
  uint32_t block_region_topk_blocks = 0;
  uint32_t block_region_topk_regions_per_block = 0;
};
```

| Field | Default | Description |
|-------|---------|-------------|
| `scale` | `1.0` | Multiplier applied to raw globaltimer ticks |
| `emit_complete_events` | `true` | Pair B/E events into Chrome "X" (complete) events |
| `group_by_smid` | `true` | `pid=SM, tid=block\|warp` (false: `pid=block, tid=warp*32`) |
| `emit_summary_json` | `true` | Write `*_summary.json` alongside the trace |
| `summary_hist_bins` | `128` | Number of histogram bins in summary |
| `summary_dump_by_block_warp` | `true` | Include per-(block,warp) breakdown in summary |
| `emit_block_region_distributions` | `false` | Write per-block region distribution files (see below) |
| `block_region_hist_bins` | `128` | Histogram bins for per-block distributions |
| `summary_topk_block_warp_per_region` | `0` | Limit by_block_warp to top-K entries per region |
| `block_region_topk_blocks` | `0` | Limit block distribution output to top-K blocks |
| `block_region_topk_regions_per_block` | `0` | Limit to top-K regions per block |

---

## Per-Block Region Distributions

When `emit_block_region_distributions = true`, a side directory is created alongside
the trace output containing per-block, per-region duration distributions. This is
useful for diagnosing block-level load imbalance or identifying outlier blocks.

**Requires** `emit_summary_json = true` (the default).

### Enable

```cpp
TraceWriteOptions opt;
opt.emit_block_region_distributions = true;
opt.block_region_topk_blocks = 16;  // optional: limit to hottest 16 blocks
sess.write_trace("trace.json", opt);
```

### Output layout

If the trace path is `foo.json`, the feature creates:

```
foo_block_region_dists/
  index.json            ← lists all emitted blocks
  block_0.json          ← per-region stats for block 0
  block_1.json
  ...
```

### `index.json`

```json
{
  "trace": "foo.json",
  "dir": "foo_block_region_dists",
  "filters": {
    "block_region_hist_bins": 128,
    "block_region_topk_blocks": 0,
    "block_region_topk_regions_per_block": 0
  },
  "blocks": [0, 1, 2, 3]
}
```

### `block_N.json`

Each file contains the same region statistics as the main summary, but scoped to a
single block:

```json
{
  "trace": "foo.json",
  "block": 0,
  "regions": [
    {
      "region": 2,
      "name": "compute",
      "count": 64,
      "mean_dur": 762.0,
      "cv_dur": 0.08,
      "min_dur": 512.0,
      "max_dur": 896.0,
      "hist": { "bins": 128, "min": 512.0, "max": 896.0, "prob": [...] }
    }
  ]
}
```

### Visualization

Use `plot_trace_summary.py` in `--blocks_dir` mode:

```bash
python3 scripts/plot_trace_summary.py \
  --blocks_dir foo_block_region_dists/ \
  --out_dir plots/ \
  --regions 1,2
```

This generates per-block per-region histograms and overlay plots.
