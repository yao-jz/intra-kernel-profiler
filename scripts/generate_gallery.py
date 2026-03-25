#!/usr/bin/env python3
"""Generate a gallery of publication-quality visualizations from Intra-Kernel Profiler demo outputs.

Usage:
    python3 scripts/generate_gallery.py --demo-dir _demo_out --out-dir _demo_out/gallery
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "monospace",
    "font.size": 11,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.facecolor": "#0d1117",
})

PALETTE = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#f0883e",
           "#79c0ff", "#ffa657", "#7ee787", "#bc8cff", "#ff7b72"]

def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# ── 1. Trace timeline (simplified) ──────────────────────────────────
def plot_trace_timeline(demo_dir, out_dir):
    """Render a simplified Chrome-trace-style timeline from GEMM trace."""
    p = Path(demo_dir) / "trace" / "gemm_trace.json"
    if not p.exists():
        return
    with open(p) as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    if not events:
        return

    # Only take complete events (ph=X) for a clean view
    xs = [e for e in events if e.get("ph") == "X"]
    if not xs:
        return

    # Identify region names from event names
    labels = {0: "total", 1: "load_tile", 2: "compute", 3: "store"}

    # Filter out the "total" envelope region to reveal sub-region structure
    inner = [e for e in xs if e.get("name") != "total" and
             e.get("args", {}).get("region_id", e.get("name", 0)) != 0]
    if not inner:
        inner = xs  # fallback if no sub-regions

    # Take first 800 events for readability
    inner = inner[:800]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Group by tid (warp)
    tids = sorted(set(e["tid"] for e in inner))
    tid_y = {t: i for i, t in enumerate(tids)}

    for e in inner:
        rid = e.get("args", {}).get("region_id", 0) if "args" in e else 0
        # Try to get from name if region_id not available
        name = e.get("name", "")
        if name == "load_tile":
            rid = 1
        elif name == "compute":
            rid = 2
        elif name == "store":
            rid = 3
        color = PALETTE[rid % len(PALETTE)]
        y = tid_y[e["tid"]]
        ax.barh(y, e["dur"], left=e["ts"], height=0.7, color=color, alpha=0.85, linewidth=0)

    ax.set_yticks(range(len(tids)))
    ax.set_yticklabels([f"W{t & 0x3F}" for t in tids], fontsize=8)
    ax.set_xlabel("Timestamp (GPU ticks)")
    ax.set_title("Intra-kernel Trace Timeline — Tiled GEMM", fontsize=13, fontweight="bold", color="#58a6ff")
    ax.invert_yaxis()
    ax.grid(axis="x")

    # Legend (exclude "total")
    handles = [plt.Rectangle((0,0),1,1, color=PALETTE[i]) for i in [1, 2, 3]]
    ax.legend(handles, [labels[i] for i in [1, 2, 3]], loc="upper right",
              fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    save(fig, out_dir / "trace_timeline.png")


# ── 2. Region duration distributions ────────────────────────────────
def plot_region_distributions(demo_dir, out_dir):
    """Overlaid histograms of per-region durations from GEMM trace summary.
    Excludes the 'total' envelope region to keep the interesting sub-regions visible."""
    p = Path(demo_dir) / "trace" / "gemm_trace_summary.json"
    if not p.exists():
        return
    with open(p) as f:
        summary = json.load(f)

    regions = summary.get("regions", [])
    if not regions:
        return

    # Exclude the outer "total" region — it spans the entire kernel and squishes sub-regions
    regions = [r for r in regions if r.get("name", "") != "total"]
    if not regions:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    has_data = False
    for i, r in enumerate(regions):
        hist = r.get("hist", {})
        name = r.get("name", f"region_{r.get('id', i)}")

        # Support both formats: edges/counts and bins/min/max/prob
        edges = hist.get("edges", [])
        counts = hist.get("counts", [])

        if not edges or not counts:
            # bins/min/max/prob format
            nbins = hist.get("bins", 0)
            hmin = hist.get("min", 0)
            hmax = hist.get("max", 0)
            prob = hist.get("prob", [])
            if nbins > 0 and prob and hmax > hmin:
                step = (hmax - hmin) / nbins
                centers = [hmin + (j + 0.5) * step for j in range(len(prob))]
                counts = prob  # probability density
            else:
                continue
        else:
            centers = [(edges[j] + edges[j+1]) / 2 for j in range(len(counts))]

        if not centers:
            continue
        has_data = True
        ax.fill_between(centers, counts, alpha=0.4, color=PALETTE[(i+1) % len(PALETTE)], label=name)
        ax.plot(centers, counts, color=PALETTE[(i+1) % len(PALETTE)], linewidth=1.5)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Duration (GPU ticks)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Region Duration Distributions — Tiled GEMM (sub-regions)",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(True)
    save(fig, out_dir / "region_distributions.png")


# ── 3. NVBit instruction class breakdown ────────────────────────────
def plot_inst_class_breakdown(demo_dir, out_dir):
    """Stacked bar chart of instruction classes per region."""
    # Prefer all mode (has inst_class data) over pcmap
    for sub in ["all", "pcmap", "inst_pipe"]:
        p = Path(demo_dir) / "nvbit" / sub / "region_stats_nvbit_marked_kernel_0.json"
        if p.exists():
            break
    else:
        return
    with open(p) as f:
        data = json.load(f)

    regions = data.get("regions", [])
    if not regions:
        return

    # Collect instruction classes across all regions
    all_classes = set()
    for r in regions:
        ic = r.get("inst_class", {})
        all_classes.update(ic.keys())
    all_classes = sorted(all_classes)

    if not all_classes:
        return

    labels_map = {0: "outside", 1: "compute", 2: "store"}
    region_names = [labels_map.get(r.get("region", i), f"Region {r.get('region', i)}") for i, r in enumerate(regions)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(regions))
    bottom = np.zeros(len(regions))

    plotted = False
    for ci, cls in enumerate(all_classes):
        vals = np.array([r.get("inst_class", {}).get(cls, 0) for r in regions], dtype=float)
        if vals.sum() == 0:
            continue
        plotted = True
        ax.bar(x, vals, bottom=bottom, label=cls, color=PALETTE[ci % len(PALETTE)], width=0.6)
        bottom += vals

    if not plotted:
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels(region_names, fontsize=10)
    ax.set_ylabel("Instruction Count")
    ax.set_title("NVBit — Instruction Class Breakdown by Region", fontsize=13, fontweight="bold", color="#58a6ff")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(axis="y")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    save(fig, out_dir / "nvbit_inst_class.png")


# ── 4. NVBit instruction pipeline attribution ───────────────────────
def plot_inst_pipeline(demo_dir, out_dir):
    """Horizontal stacked bar chart of per-pipeline instruction counts.
    Falls back to inst_class from 'all' mode if inst_pipe is empty."""
    # Try inst_pipe mode first, then fall back to all mode
    p = Path(demo_dir) / "nvbit" / "inst_pipe" / "region_stats_nvbit_marked_kernel_0.json"
    if not p.exists():
        p = Path(demo_dir) / "nvbit" / "all" / "region_stats_nvbit_marked_kernel_0.json"
    if not p.exists():
        p = Path(demo_dir) / "nvbit" / "pcmap" / "region_stats_nvbit_marked_kernel_0.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    regions = data.get("regions", [])
    if not regions:
        return

    # Try inst_pipe first, then inst_class from same file
    all_pipes = {}
    for r in regions:
        ip = r.get("inst_pipe")
        if ip:
            for k, v in ip.items():
                all_pipes[k] = all_pipes.get(k, 0) + v

    # Fall back to inst_class from same file
    if not any(all_pipes.values()):
        all_pipes = {}
        for r in regions:
            ic = r.get("inst_class", {})
            for k, v in ic.items():
                all_pipes[k] = all_pipes.get(k, 0) + v

    # If still nothing, try loading from 'all' or 'pcmap' mode
    if not any(all_pipes.values()):
        for sub in ["all", "pcmap"]:
            alt = Path(demo_dir) / "nvbit" / sub / "region_stats_nvbit_marked_kernel_0.json"
            if alt.exists():
                with open(alt) as f2:
                    alt_data = json.load(f2)
                for r in alt_data.get("regions", []):
                    ic = r.get("inst_class", {})
                    for k, v in ic.items():
                        all_pipes[k] = all_pipes.get(k, 0) + v
                if any(all_pipes.values()):
                    break

    if not all_pipes:
        return

    # Sort by count descending, filter zero
    items = [(k, v) for k, v in sorted(all_pipes.items(), key=lambda x: -x[1]) if v > 0]
    if not items:
        return

    labels, values = zip(*items)

    fig, ax = plt.subplots(figsize=(10, max(4, len(items) * 0.35)))
    y = np.arange(len(items))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(items))]

    bars = ax.barh(y, values, color=colors, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Instruction Count")
    ax.set_title("NVBit — Instruction Pipeline Attribution", fontsize=13, fontweight="bold", color="#58a6ff")
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=8, color="#8b949e")

    save(fig, out_dir / "nvbit_inst_pipeline.png")


# ── 5. CUPTI SASS metrics heatmap ───────────────────────────────────
def plot_sass_metrics_heatmap(demo_dir, out_dir):
    """Heatmap of per-PC instruction execution counts from SASS metrics."""
    p = Path(demo_dir) / "cupti" / "sassmetrics_core.json"
    if not p.exists():
        p = Path(demo_dir) / "cupti" / "sassmetrics_raw.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        return

    # Get per-PC metric values
    pcs = []
    vals = []
    funcs = []
    for r in records:
        metrics = r.get("metrics", {})
        v = metrics.get("smsp__sass_inst_executed", 0)
        if v > 0:
            pcs.append(r.get("pcOffset", 0))
            vals.append(v)
            funcs.append(r.get("functionName", "?"))

    if not pcs:
        return

    # Sort by PC offset
    order = np.argsort(pcs)
    pcs = [pcs[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    log_vals = np.log10(np.array(vals, dtype=float) + 1)
    max_log = max(log_vals) if max(log_vals) > 0 else 1
    colors = plt.cm.inferno(log_vals / max_log)

    bars = ax.bar(range(len(pcs)), vals, color=colors, width=0.8, linewidth=0)
    ax.set_xlabel("PC Offset (instruction index)")
    ax.set_ylabel("Instruction Executions")
    ax.set_title("CUPTI SASS Metrics — Per-PC Execution Hotspot", fontsize=13, fontweight="bold", color="#58a6ff")
    ax.grid(axis="y")
    ax.set_yscale("log")

    # Annotate the hottest PC
    max_idx = np.argmax(vals)
    ax.annotate(f"{vals[max_idx]:,.0f}", xy=(max_idx, vals[max_idx]),
                xytext=(max_idx - 3, vals[max_idx] * 0.3), fontsize=9, color="#f78166",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="#f78166", lw=1.5))

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, max(vals)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Execution Count", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")

    save(fig, out_dir / "cupti_sass_hotspot.png")


# ── 6. CUPTI PC sampling stall reasons ──────────────────────────────
def plot_stall_reasons(demo_dir, out_dir):
    """Donut chart of stall reason distribution from PC sampling."""
    p = Path(demo_dir) / "cupti" / "pcsampling_raw.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    pc_records = data.get("pc_records", [])

    # Build stall reason index→name mapping
    stall_names = {}
    stall_table = data.get("stall_reason_table", [])
    if isinstance(stall_table, list) and stall_table:
        if isinstance(stall_table[0], dict):
            for entry in stall_table:
                stall_names[entry["index"]] = entry["name"]
        else:
            for i, name in enumerate(stall_table):
                stall_names[i] = name
    if not stall_names:
        stall_tables = data.get("stall_reason_tables", [])
        if stall_tables:
            entries = stall_tables[0].get("entries", stall_tables[0]) if isinstance(stall_tables[0], dict) else stall_tables[0]
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        stall_names[entry["index"]] = entry["name"]

    if not pc_records or not stall_names:
        return

    # Aggregate stall samples
    totals = {}
    for rec in pc_records:
        stalls = rec.get("stallReasons", rec.get("stall_reasons", []))
        for i, count in enumerate(stalls):
            if count > 0 and i in stall_names:
                name = stall_names[i]
                short = name.replace("smsp__pcsamp_warps_issue_stalled_", "").replace("_not_issued", "")
                if short in ("smsp__pcsamp_sample_count", "smsp__pcsamp_samples_data_dropped"):
                    continue
                totals[short] = totals.get(short, 0) + count

    if not totals:
        return

    # Top reasons
    items = sorted(totals.items(), key=lambda x: -x[1])
    top = items[:8]
    other = sum(v for _, v in items[8:])
    if other > 0:
        top.append(("other", other))

    labels, values = zip(*top)
    total = sum(values)
    pcts = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=PALETTE[:len(values)], startangle=90,
        pctdistance=0.8, wedgeprops=dict(width=0.4, edgecolor="#0d1117", linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("#c9d1d9")

    ax.legend(wedges, [f"{l} ({v:,})" for l, v in zip(labels, values)],
              loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_title("CUPTI PC Sampling — Stall Reason Distribution",
                 fontsize=13, fontweight="bold", color="#58a6ff", pad=20)
    save(fig, out_dir / "cupti_stall_reasons.png")


# ── 7. CUPTI divergence analysis ────────────────────────────────────
def plot_divergence(demo_dir, out_dir):
    """Bar chart comparing warp executions across PCs with convergence overlay.

    Top panel: per-PC warp execution counts (log scale for visibility).
    Bottom panel: convergence ratio colored by divergence level."""
    p = Path(demo_dir) / "cupti" / "sassmetrics_divergence.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        return

    pcs = []
    executed = []
    active = []
    for r in records:
        m = r.get("metrics", {})
        e = m.get("smsp__sass_inst_executed", 0)
        t = m.get("smsp__sass_thread_inst_executed", 0)
        p_on = m.get("smsp__sass_thread_inst_executed_pred_on", 0)
        if e > 0:
            pcs.append(r.get("pcOffset", 0))
            executed.append(e)
            active.append(t if t > 0 else p_on)

    if not pcs:
        return

    order = np.argsort(pcs)
    pcs_s = [pcs[i] for i in order]
    executed_s = np.array([executed[i] for i in order], dtype=float)
    active_s = np.array([active[i] for i in order], dtype=float)

    # Compute divergence ratio (thread_inst / (warp_inst * 32))
    # If active is 0 (no thread-level data), compute from warp-level only
    has_thread_data = active_s.sum() > 0
    div_ratio = np.where(executed_s > 0, active_s / (executed_s * 32), 1.0) if has_thread_data else np.ones_like(executed_s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]})

    x = np.arange(len(pcs_s))

    # Top: per-PC execution count as bar chart
    bars = ax1.bar(x, executed_s, width=0.6, color=PALETTE[0], label="Warp Executions", alpha=0.9)
    # Annotate top-3 hottest PCs
    top_k = min(3, len(executed_s))
    top_idx = np.argsort(executed_s)[-top_k:]
    for idx in top_idx:
        if executed_s[idx] > 0:
            ax1.annotate(f"{executed_s[idx]:,.0f}", xy=(idx, executed_s[idx]),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=7, color="#ffa657", fontweight="bold")
    ax1.set_ylabel("Warp Executions")
    ax1.set_title("CUPTI — Per-PC Execution Profile & Convergence", fontsize=13, fontweight="bold", color="#58a6ff")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax1.grid(axis="y")
    ax1.yaxis.set_major_formatter(ticker.EngFormatter())
    if executed_s.max() > 100 * executed_s[executed_s > 0].min():
        ax1.set_yscale("log")

    # Bottom: convergence ratio
    colors = [PALETTE[2] if d > 0.95 else PALETTE[3] if d > 0.8 else PALETTE[1] if d > 0.5 else PALETTE[4]
              for d in div_ratio]
    ax2.bar(x, div_ratio, color=colors, width=0.6)
    ax2.axhline(y=1.0, color="#3fb950", linestyle="--", alpha=0.5, label="Perfect convergence")
    ax2.set_xlabel("PC Offset (instruction index)")
    ax2.set_ylabel("Convergence Ratio")
    ax2.set_ylim(0, 1.15)
    if has_thread_data:
        avg_conv = np.mean(div_ratio[executed_s > 0]) if (executed_s > 0).any() else 1.0
        ax2.axhline(y=avg_conv, color="#ffa657", linestyle=":", alpha=0.7,
                     label=f"Avg convergence: {avg_conv:.2%}")
    else:
        ax2.text(0.5, 0.5, "Thread-level data not available\n(warp-level only)",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=10, color="#8b949e")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    ax2.grid(axis="y")

    fig.tight_layout()
    save(fig, out_dir / "cupti_divergence.png")


# ── 8. NVBit+CUPTI join — per-region metric comparison ──────────────
def plot_join_analysis(demo_dir, out_dir):
    """Per-region metric breakdown from the NVBit+CUPTI join."""
    p = Path(demo_dir) / "join" / "merged_sassmetrics.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    regions = data.get("regions", [])
    if not regions:
        return

    labels_map = {0: "outside", 1: "compute", 2: "store"}

    fig, ax = plt.subplots(figsize=(8, 5))

    names = []
    values = []
    for r in regions:
        rid = r.get("region_id", 0)
        name = labels_map.get(rid, f"region_{rid}")
        names.append(name)
        m = r.get("metrics_total", r.get("metrics", {}))
        # Try to get the main metric
        v = m.get("smsp__sass_inst_executed", 0)
        values.append(v)

    if not any(values):
        return

    x = np.arange(len(names))
    bars = ax.bar(x, values, color=PALETTE[:len(names)], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("smsp__sass_inst_executed")
    ax.set_title("NVBit + CUPTI Join — Per-Region Hardware Metrics",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    ax.grid(axis="y")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f"{val:,.0f}", ha="center", fontsize=9, color="#c9d1d9")

    save(fig, out_dir / "join_per_region.png")


# ── 9. NVBit branch divergence histogram ────────────────────────────
def plot_branch_div_histogram(demo_dir, out_dir):
    """Branch divergence histogram from NVBit region stats.
    The field is 'branch_div_hist' (33 buckets: 0..32 active lanes) at region top level."""
    # Try bb_hot first (has branch_sites), then pcmap, then all
    for sub in ["bb_hot", "pcmap", "all", "inst_pipe"]:
        p = Path(demo_dir) / "nvbit" / sub / "region_stats_nvbit_marked_kernel_0.json"
        if p.exists():
            break
    else:
        return

    with open(p) as f:
        data = json.load(f)

    regions = data.get("regions", [])

    fig, ax = plt.subplots(figsize=(12, 5))
    has_data = False

    labels_map = {0: "outside", 1: "compute", 2: "store"}
    bar_width = 0.8 / max(len(regions), 1)

    for i, r in enumerate(regions):
        # Support both formats: nested branch_div.histogram and top-level branch_div_hist
        hist = r.get("branch_div_hist", [])
        if not hist:
            bd = r.get("branch_div", {})
            hist = bd.get("histogram", [])
        if not hist or not any(hist):
            continue

        has_data = True
        rid = r.get("region", r.get("id", i))
        name = labels_map.get(rid, f"Region {rid}")
        x = np.arange(len(hist))
        offset = (i - len(regions) / 2 + 0.5) * bar_width
        ax.bar(x + offset, hist, width=bar_width, color=PALETTE[i % len(PALETTE)],
               label=name, alpha=0.85)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Active Lanes per Branch (0 = fully diverged, 32 = fully converged)")
    ax.set_ylabel("Branch Executions")
    ax.set_title("NVBit — Branch Divergence Histogram", fontsize=13, fontweight="bold", color="#58a6ff")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(axis="y")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    # Annotate: most branches at lane=31/32 means mostly converged
    ax.axvline(x=32, color="#3fb950", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(32, ax.get_ylim()[1] * 0.9, "fully\nconverged", fontsize=8, color="#3fb950",
            ha="center", va="top")
    save(fig, out_dir / "nvbit_branch_divergence.png")


# ── 10. CUPTI multi-profile comparison ──────────────────────────────
def plot_multi_profile(demo_dir, out_dir):
    """Show record counts and key metrics across all SASS profiles."""
    profiles = ["core", "divergence", "memory", "instruction_mix", "branch"]
    counts = []
    found_profiles = []

    for prof in profiles:
        p = Path(demo_dir) / "cupti" / f"sassmetrics_{prof}.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        recs = data.get("records", [])
        counts.append(len(recs))
        found_profiles.append(prof)

    if not found_profiles:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(found_profiles))
    bars = ax.bar(x, counts, color=PALETTE[:len(found_profiles)], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(found_profiles, fontsize=10)
    ax.set_ylabel("PC Records")
    ax.set_title("CUPTI SASS Metrics — Profile Coverage Comparison",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    ax.grid(axis="y")

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                str(val), ha="center", fontsize=10, color="#c9d1d9", fontweight="bold")

    save(fig, out_dir / "cupti_profile_comparison.png")


# ── 11. Memory instruction mix (from instruction_mix profile) ───────
def plot_memory_instruction_mix(demo_dir, out_dir):
    """Donut chart of instruction categories from NVBit inst_class data.

    Falls back to CUPTI sassmetrics_instruction_mix.json if NVBit data unavailable.
    Groups fine-grained NVBit categories into user-friendly groups."""
    # Prefer NVBit inst_class data (has real per-opcode breakdown)
    nvbit_path = Path(demo_dir) / "nvbit" / "all" / "region_stats_nvbit_marked_kernel_0.json"
    if not nvbit_path.exists():
        nvbit_path = Path(demo_dir) / "nvbit" / "pcmap" / "region_stats_nvbit_marked_kernel_0.json"

    items = []
    if nvbit_path.exists():
        with open(nvbit_path) as f:
            data = json.load(f)
        # Aggregate inst_class across all regions
        totals = {}
        for r in data.get("regions", []):
            ic = r.get("inst_class", {})
            for k, v in ic.items():
                totals[k] = totals.get(k, 0) + v
        # Group into readable categories
        groups = {
            "FP32 ALU": totals.get("alu_fp32", 0),
            "Int ALU": totals.get("alu_int", 0),
            "Tensor/WGMMA": totals.get("tensor_wgmma", 0),
            "Global Load": totals.get("ld_global", 0),
            "Global Store": totals.get("st_global", 0),
            "Shared Load": totals.get("ld_shared", 0),
            "Shared Store": totals.get("st_shared", 0),
            "Branch": totals.get("branch", 0),
            "Call/Ret": totals.get("call", 0) + totals.get("ret", 0),
            "Barrier": totals.get("barrier", 0) + totals.get("membar", 0),
            "Other": totals.get("other", 0) + totals.get("special", 0),
        }
        items = [(k, v) for k, v in groups.items() if v > 0]
        items.sort(key=lambda x: -x[1])
    else:
        # Fallback to CUPTI
        p = Path(demo_dir) / "cupti" / "sassmetrics_instruction_mix.json"
        if not p.exists():
            return
        with open(p) as f:
            data = json.load(f)
        records = data.get("records", [])
        if not records:
            return
        totals = {}
        for r in records:
            for k, v in r.get("metrics", {}).items():
                totals[k] = totals.get(k, 0) + v
        if not totals:
            return
        items = [(k.replace("smsp__sass_inst_executed_op_", "").replace("smsp__sass_", ""), v)
                 for k, v in sorted(totals.items(), key=lambda x: -x[1]) if v > 0]

    if not items:
        return

    if len(items) > 10:
        top = items[:9]
        other = sum(v for _, v in items[9:])
        top.append(("other", other))
        items = top

    labels, values = zip(*items)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        colors=PALETTE[:len(values)], startangle=140,
        pctdistance=0.78, wedgeprops=dict(width=0.45, edgecolor="#0d1117", linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("#c9d1d9")

    ax.legend(wedges, [f"{l} ({v:,.0f})" for l, v in zip(labels, values)],
              loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_title("Instruction Mix Breakdown", fontsize=13, fontweight="bold",
                 color="#58a6ff", pad=20)
    save(fig, out_dir / "cupti_instruction_mix.png")


# ── 12. SASS annotated code view ────────────────────────────────────
def plot_sass_annotated(demo_dir, out_dir):
    """Render annotated SASS listing from nvdisasm output with source-line annotations."""
    # Prefer nvdisasm output (high-quality SASS with source lines)
    sass_path = Path(demo_dir) / "nvbit" / "nvdisasm" / "nvdisasm_all_nvbit_marked_kernel_0.sass"
    if not sass_path.exists():
        sass_path = Path(demo_dir) / "nvbit" / "nvdisasm" / "sass_all_nvbit_marked_kernel_0.sass"
    if not sass_path.exists():
        return

    with open(sass_path) as f:
        sass_lines = f.readlines()

    # Find the kernel function body — skip headers until first instruction /*0x...*/
    start = 0
    for i, line in enumerate(sass_lines):
        if "/*0" in line and any(op in line for op in ["LDC", "S2R", "MOV", "IMAD"]):
            start = i
            break

    # Collect lines: instructions + source-line annotations
    code_lines = []
    for line in sass_lines[start:]:
        stripped = line.rstrip()
        if not stripped:
            continue
        # Skip assembler directives
        if stripped.startswith(("\t.", ".L_x_")):
            continue
        # Keep instructions (/*0x...*/) and source file comments (//## File)
        if "/*0" in stripped or "//##" in stripped or ".L_x_" in stripped:
            code_lines.append(stripped)
    code_lines = code_lines[:50]  # limit for readability

    if not code_lines:
        return

    fig, ax = plt.subplots(figsize=(16, max(5, len(code_lines) * 0.28)))
    ax.axis("off")

    for i, line in enumerate(code_lines):
        color = "#c9d1d9"
        if "//##" in line:
            # Source line annotation — extract just filename:line
            color = "#8b949e"
            # Shorten paths for readability
            line = line.replace("//## File ", "// ")
            # Strip any absolute build path prefix up to the project root
            line = re.sub(r'.*/intra_kernel_profiler/', '', line)
            if "inlined at" in line:
                parts = line.split("inlined at")
                line = parts[-1].strip().rstrip('"')
                line = "// inlined: " + line
        elif "FFMA" in line or "FMUL" in line or "FADD" in line or "HMMA" in line or "WGMMA" in line:
            color = "#3fb950"  # green for FP compute
        elif "LDG" in line or "STG" in line or "LD.E" in line or "ST.E" in line:
            color = "#f78166"  # orange for global memory
        elif "LDS" in line or "STS" in line:
            color = "#ffa657"  # lighter orange for shared memory
        elif "BRA" in line or "EXIT" in line or "BSSY" in line or "BSYNC" in line:
            color = "#d2a8ff"  # purple for control flow
        elif "S2R" in line or "VOTE" in line:
            color = "#79c0ff"  # light blue for special regs
        ax.text(0.02, 1 - (i + 0.5) / len(code_lines), line,
                transform=ax.transAxes, fontsize=6.5, fontfamily="monospace",
                color=color, verticalalignment="center")

    ax.set_title("Annotated SASS — nvdisasm with Source Lines",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    save(fig, out_dir / "sass_annotated.png")


# ── 13. Memory access patterns ──────────────────────────────────────
def plot_mem_trace(demo_dir, out_dir):
    """Scatter plot of memory addresses from NVBit memtrace."""
    p = Path(demo_dir) / "nvbit" / "all" / "mem_trace_nvbit_marked_kernel_0.jsonl"
    if not p.exists():
        return

    ops = []
    count = 0
    with open(p) as f:
        for line in f:
            if count >= 500:  # limit records for readability
                break
            try:
                rec = json.loads(line)
                lane_addrs = rec.get("addrs", [])
                if not lane_addrs:
                    addr = rec.get("addr", rec.get("address", 0))
                    lane_addrs = [addr] if addr else []
                st = rec.get("is_store", 0)
                region = rec.get("region", 0)
                if lane_addrs:
                    ops.append({"addrs": lane_addrs, "is_store": bool(st), "region": region})
                    count += 1
            except json.JSONDecodeError:
                continue

    if not ops:
        return

    # Plot: each operation shows its 32 lane addresses as a vertical scatter
    fig, ax = plt.subplots(figsize=(14, 6))

    all_addrs = []
    for op in ops:
        all_addrs.extend(op["addrs"])
    addr_min = min(all_addrs)

    for i, op in enumerate(ops):
        addrs_norm = [(a - addr_min) for a in op["addrs"]]
        color = PALETTE[1] if op["is_store"] else PALETTE[0]
        ax.scatter([i] * len(addrs_norm), addrs_norm, c=color, s=1, alpha=0.5, rasterized=True)

    # Legend
    ax.scatter([], [], c=PALETTE[0], s=20, label="Load")
    ax.scatter([], [], c=PALETTE[1], s=20, label="Store")

    ax.set_xlabel("Memory Operation Index")
    ax.set_ylabel("Address Offset (bytes)")
    ax.set_title("NVBit — Per-Lane Memory Access Pattern",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(True)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    save(fig, out_dir / "nvbit_mem_trace.png")


# ── 14. Basic-block hotspot bar chart ─────────────────────────────────
def plot_bb_hotspots(demo_dir, out_dir):
    """Bar chart of basic-block execution counts — highlights hot loops."""
    p = Path(demo_dir) / "nvbit" / "bb_hot" / "hotspots_nvbit_marked_kernel_0.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    entries = data.get("bb_entries", [])
    if not entries:
        return

    entries = sorted(entries, key=lambda e: -e.get("exec_count", 0))
    top = entries[:20]

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"BB{e['bb_index']}@0x{e['entry_pc']:x}\n({e['n_instrs']}i)" for e in top]
    counts = [e["exec_count"] for e in top]
    colors = [PALETTE[0] if c == max(counts) else PALETTE[2] if c > max(counts)*0.1 else PALETTE[5]
              for c in counts]

    bars = ax.bar(range(len(top)), counts, color=colors, width=0.7)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Execution Count")
    ax.set_title("NVBit — Basic Block Hotspots (Top 20)", fontsize=13, fontweight="bold", color="#58a6ff")
    ax.grid(axis="y")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    # Annotate the hottest block
    ax.annotate(f"{counts[0]:,.0f}", xy=(0, counts[0]),
                xytext=(2, counts[0] * 0.9), fontsize=9, color="#f78166", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f78166", lw=1.5))

    save(fig, out_dir / "nvbit_bb_hotspots.png")


# ── 15. Branch site analysis ─────────────────────────────────────────
def plot_branch_sites(demo_dir, out_dir):
    """Branch site taken vs fallthrough analysis."""
    p = Path(demo_dir) / "nvbit" / "bb_hot" / "hotspots_nvbit_marked_kernel_0.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    sites = data.get("branch_sites", [])
    if not sites:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f"{s['opcode']}@0x{s['pc_offset']:x}" for s in sites]
    taken = [s.get("taken_lanes", 0) for s in sites]
    ft = [s.get("fallthrough_lanes", 0) for s in sites]

    x = np.arange(len(sites))
    w = 0.35
    ax.bar(x - w/2, taken, w, label="Taken Lanes", color=PALETTE[0], alpha=0.9)
    ax.bar(x + w/2, ft, w, label="Fallthrough Lanes", color=PALETTE[1], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Lane Executions")
    ax.set_title("NVBit — Branch Site Analysis (Taken vs Fallthrough)",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(axis="y")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    save(fig, out_dir / "nvbit_branch_sites.png")


# ── 16. Memory locality analysis ─────────────────────────────────────
def plot_locality_analysis(demo_dir, out_dir):
    """Visualize memory locality metrics from nvbit_locality.py output.
    Format: regions dict keyed by region id, each has reuse_distance, working_set, inter_warp_sharing."""
    p = Path(demo_dir) / "nvbit" / "all" / "locality_analysis.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    regions = data.get("regions", {})
    hist_bounds = data.get("hist_bounds", [])
    if not regions:
        return

    labels_map = {"0": "outside", "1": "compute", "2": "store"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: reuse distance histogram (global scope)
    ax = axes[0]
    has_reuse = False
    for ri, (rid, rdata) in enumerate(regions.items()):
        rd = rdata.get("reuse_distance", {}).get("global", {})
        hist = rd.get("hist", [])
        cold = rd.get("cold", 0)
        if hist and any(hist):
            has_reuse = True
            name = labels_map.get(rid, f"region {rid}")
            x = np.arange(len(hist))
            ax.bar(x + ri * 0.3, hist, width=0.28, color=PALETTE[ri % len(PALETTE)],
                   label=f"{name} (cold={cold})", alpha=0.85)
    if has_reuse:
        ax.set_xlabel("Reuse Distance Bucket")
        ax.set_ylabel("Count")
        ax.set_title("Reuse Distance", fontsize=11, fontweight="bold", color="#58a6ff")
        ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
        ax.grid(axis="y")
    else:
        ax.text(0.5, 0.5, "All cold misses\n(no reuse)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")
        ax.set_title("Reuse Distance", fontsize=11, fontweight="bold", color="#58a6ff")

    # Middle: working set info
    ax = axes[1]
    ws_data = []
    ws_names = []
    for rid, rdata in regions.items():
        ws = rdata.get("working_set", {})
        for win_key, ws_val in ws.items():
            ws_data.append(ws_val)
            ws_names.append(labels_map.get(rid, f"r{rid}"))
    if ws_data:
        avgs = [w.get("avg", 0) for w in ws_data]
        p95s = [w.get("p95", 0) for w in ws_data]
        maxs = [w.get("max", 0) for w in ws_data]
        x = np.arange(len(ws_names))
        w = 0.25
        ax.bar(x - w, avgs, w, label="avg", color=PALETTE[0])
        ax.bar(x, p95s, w, label="p95", color=PALETTE[2])
        ax.bar(x + w, maxs, w, label="max", color=PALETTE[1])
        ax.set_xticks(x)
        ax.set_xticklabels(ws_names, fontsize=10)
        ax.set_ylabel("Cache Lines")
        ax.set_title("Working Set Size", fontsize=11, fontweight="bold", color="#58a6ff")
        ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
        ax.grid(axis="y")

    # Right: inter-warp sharing
    ax = axes[2]
    for ri, (rid, rdata) in enumerate(regions.items()):
        iws = rdata.get("inter_warp_sharing", {})
        ratio = iws.get("shared_line_ratio", 0)
        avg_warps = iws.get("avg_warps_per_line", 0)
        name = labels_map.get(rid, f"r{rid}")
        ax.bar(ri, avg_warps, color=PALETTE[ri % len(PALETTE)], width=0.5, label=name)
        ax.text(ri, avg_warps + 0.05, f"shared={ratio:.0%}", ha="center", fontsize=8, color="#c9d1d9")
    ax.set_ylabel("Avg Warps per Cache Line")
    ax.set_title("Inter-Warp Sharing", fontsize=11, fontweight="bold", color="#58a6ff")
    ax.grid(axis="y")
    if regions:
        ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    fig.suptitle("NVBit — Memory Locality Analysis", fontsize=14, fontweight="bold", color="#58a6ff", y=1.02)
    fig.tight_layout()
    save(fig, out_dir / "nvbit_locality.png")


# ── 17. Per-region radar chart ────────────────────────────────────────
def plot_region_radar(demo_dir, out_dir):
    """Radar/spider chart comparing regions across multiple metrics.

    Uses log-scaled normalization to prevent extreme metric imbalance from
    collapsing the radar into a degenerate cross shape.  Also derives
    per-instruction-class fractions so every region has meaningful shape."""
    p = Path(demo_dir) / "nvbit" / "all" / "region_stats_nvbit_marked_kernel_0.json"
    if not p.exists():
        p = Path(demo_dir) / "nvbit" / "pcmap" / "region_stats_nvbit_marked_kernel_0.json"
    if not p.exists():
        return
    with open(p) as f:
        data = json.load(f)

    regions = data.get("regions", [])
    if len(regions) < 2:
        return

    labels_map = {0: "outside", 1: "compute", 2: "store"}

    # Derive useful per-region ratios from inst_class breakdown
    for r in regions:
        ic = r.get("inst_class", {})
        total = r.get("inst_total", 1) or 1
        r["_frac_fp"] = (ic.get("alu_fp32", 0) + ic.get("tensor_wgmma", 0)) / total
        r["_frac_int"] = ic.get("alu_int", 0) / total
        r["_frac_mem"] = (ic.get("ld_global", 0) + ic.get("st_global", 0) +
                          ic.get("ld_shared", 0) + ic.get("st_shared", 0)) / total
        r["_frac_branch"] = ic.get("branch", 0) / total
        r["_bb_density"] = r.get("bb_exec", 0) / total  # exec density

    # Use derived fractions + raw totals
    metric_defs = [
        ("FP Fraction",   "_frac_fp"),
        ("Int Fraction",  "_frac_int"),
        ("Mem Fraction",  "_frac_mem"),
        ("Branch Fraction", "_frac_branch"),
        ("Instructions (log)", "inst_total"),
        ("BB Exec (log)", "bb_exec"),
    ]

    # Filter to metrics where at least one region is non-zero
    active = [(label, key) for label, key in metric_defs
              if any(r.get(key, 0) > 0 for r in regions)]

    if len(active) < 3:
        return

    names = [a[0] for a in active]
    n = len(names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161b22")

    for i, r in enumerate(regions):
        rid = r.get("region", r.get("id", i))
        rname = labels_map.get(rid, f"Region {rid}")
        vals = [r.get(a[1], 0) for a in active]
        # Normalize: fractions stay as-is (0-1), large counts use log normalization
        max_vals = [max(rr.get(a[1], 0) for rr in regions) for a in active]
        norm = []
        for v, m, (label, _) in zip(vals, max_vals, active):
            if "(log)" in label and m > 0:
                # Log-scale normalization to compress dynamic range
                norm.append(np.log1p(v) / np.log1p(m) if m > 0 else 0)
            elif m > 0:
                norm.append(v / m)
            else:
                norm.append(0)
        norm += norm[:1]
        ax.fill(angles, norm, alpha=0.25, color=PALETTE[i % len(PALETTE)])
        ax.plot(angles, norm, linewidth=2.5, color=PALETTE[i % len(PALETTE)],
                label=rname, marker="o", markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, fontsize=9, color="#c9d1d9")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.set_title("NVBit — Per-Region Metric Comparison",
                 fontsize=13, fontweight="bold", color="#58a6ff", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1),
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.spines["polar"].set_color("#30363d")
    ax.grid(True, color="#21262d", linestyle="--", alpha=0.6)
    save(fig, out_dir / "nvbit_region_radar.png")


# ── 18. Trace per-region summary stats ────────────────────────────────
def plot_region_summary_stats(demo_dir, out_dir):
    """Per-region timing breakdown: mean, p50, p95 as grouped bars.
    Excludes 'total' to keep sub-region bars visible."""
    p = Path(demo_dir) / "trace" / "gemm_trace_summary.json"
    if not p.exists():
        return
    with open(p) as f:
        summary = json.load(f)

    regions = summary.get("regions", [])
    # Exclude "total" — its scale (80K ticks) hides the sub-regions (~1K ticks)
    regions = [r for r in regions if r.get("name", "") != "total"]
    if not regions:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [r.get("name", f"region_{i}") for i, r in enumerate(regions)]
    means = [r.get("mean_dur", 0) for r in regions]
    p50s = [r.get("percentiles", {}).get("p50", 0) for r in regions]
    p95s = [r.get("percentiles", {}).get("p95", 0) for r in regions]
    p99s = [r.get("percentiles", {}).get("p99", 0) for r in regions]

    x = np.arange(len(names))
    w = 0.2

    ax.bar(x - 1.5*w, means, w, label="Mean", color=PALETTE[0], alpha=0.9)
    ax.bar(x - 0.5*w, p50s, w, label="p50", color=PALETTE[2], alpha=0.9)
    ax.bar(x + 0.5*w, p95s, w, label="p95", color=PALETTE[3], alpha=0.9)
    ax.bar(x + 1.5*w, p99s, w, label="p99", color=PALETTE[1], alpha=0.9)

    # Add value annotations
    for xi, (mean, p99) in enumerate(zip(means, p99s)):
        ax.text(xi, p99 + max(p99s) * 0.03, f"{mean:,.0f}", ha="center",
                fontsize=9, color="#c9d1d9", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Duration (GPU ticks)")
    ax.set_title("Trace — Per-Region Timing Breakdown (sub-regions)",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(axis="y")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    save(fig, out_dir / "trace_region_stats.png")


# ── 19. CUPTI SASS per-PC multi-metric heatmap ───────────────────────
def plot_sass_multi_metric(demo_dir, out_dir):
    """Grid heatmap: rows=PC offsets, cols=CUPTI SASS profiles.

    Uses each profile's best available metric and normalizes per-column
    so relative hotspots are visible even when absolute magnitudes differ."""
    profiles = ["core", "divergence", "memory", "instruction_mix", "branch"]
    # Each profile may have a different primary metric
    metric_pref = {
        "core": "smsp__sass_inst_executed",
        "divergence": "smsp__sass_inst_executed",
        "memory": "smsp__sass_inst_executed",
        "instruction_mix": "smsp__sass_inst_executed",
        "branch": "smsp__sass_inst_executed_op_branch",
    }
    data_map = {}  # pc -> {prof: value}

    for prof in profiles:
        p = Path(demo_dir) / "cupti" / f"sassmetrics_{prof}.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        pref_key = metric_pref.get(prof, "smsp__sass_inst_executed")
        for r in d.get("records", []):
            pc = r.get("pcOffset", 0)
            m = r.get("metrics", {})
            # Use preferred metric, fallback to first available
            v = m.get(pref_key, 0)
            if v == 0 and m:
                v = next(iter(m.values()))
            if pc not in data_map:
                data_map[pc] = {}
            data_map[pc][prof] = v

    if not data_map:
        return

    pcs = sorted(data_map.keys())
    found = sorted(set(p for row in data_map.values() for p in row))
    if len(found) < 2:
        return

    # Build matrix (limit to 40 PCs)
    pcs = pcs[:40]
    mat = np.zeros((len(pcs), len(found)))
    for i, pc in enumerate(pcs):
        for j, prof in enumerate(found):
            mat[i, j] = data_map[pc].get(prof, 0)

    # Normalize per-column so each profile's hotspots are visible
    col_max = mat.max(axis=0)
    col_max[col_max == 0] = 1
    mat_norm = mat / col_max[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(8, max(5, len(pcs) * 0.2)))
    im = ax.imshow(mat_norm, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_xticks(range(len(found)))
    ax.set_xticklabels(found, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("PC Offset Index")
    ax.set_title("CUPTI — Multi-Profile Per-PC Heatmap (normalized)",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Relative intensity (per-profile)", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")
    fig.tight_layout()
    save(fig, out_dir / "cupti_multi_metric_heatmap.png")


# ── 20. Overview dashboard (2x2) ─────────────────────────────────────
def plot_overview_dashboard(demo_dir, out_dir):
    """Compact 2x2 overview: trace histogram, inst breakdown, divergence, join."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: trace sub-region durations (exclude "total")
    ax = axes[0, 0]
    sp = Path(demo_dir) / "trace" / "gemm_trace_summary.json"
    if sp.exists():
        with open(sp) as f:
            s = json.load(f)
        regions = [r for r in s.get("regions", []) if r.get("name", "") != "total"]
        names = [r.get("name", f"r{i}") for i, r in enumerate(regions)]
        means = [r.get("mean_dur", 0) for r in regions]
        if means:
            bars = ax.barh(range(len(names)), means, color=PALETTE[1:len(names)+1], height=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("Mean Duration (ticks)")
            ax.set_title("Sub-region Timing", fontsize=11, fontweight="bold", color="#58a6ff")
            ax.grid(axis="x")
            ax.xaxis.set_major_formatter(ticker.EngFormatter())
            ax.invert_yaxis()
            for bar, v in zip(bars, means):
                ax.text(bar.get_width() + max(means)*0.02, bar.get_y()+bar.get_height()/2,
                        f"{v:,.0f}", va="center", fontsize=9, color="#c9d1d9")

    # Top-right: inst_class pie from all/pcmap
    ax = axes[0, 1]
    for sub in ["all", "pcmap"]:
        ip = Path(demo_dir) / "nvbit" / sub / "region_stats_nvbit_marked_kernel_0.json"
        if ip.exists():
            break
    else:
        ip = None
    if ip and ip.exists():
        with open(ip) as f:
            d = json.load(f)
        total_ic = {}
        for r in d.get("regions", []):
            for k, v in r.get("inst_class", {}).items():
                total_ic[k] = total_ic.get(k, 0) + v
        items = [(k, v) for k, v in sorted(total_ic.items(), key=lambda x: -x[1]) if v > 0]
        if items:
            if len(items) > 8:
                top = items[:7]
                top.append(("other", sum(v for _, v in items[7:])))
                items = top
            labels, vals = zip(*items)
            wedges, _, autotexts = ax.pie(vals, labels=None,
                autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
                colors=PALETTE[:len(vals)], startangle=90,
                wedgeprops=dict(width=0.45, edgecolor="#0d1117", linewidth=1))
            for t in autotexts:
                t.set_fontsize(7)
                t.set_color("#c9d1d9")
            ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5),
                      fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
            ax.set_title("Instruction Mix", fontsize=11, fontweight="bold", color="#58a6ff")

    # Bottom-left: BB hotspot or SASS hotspot
    ax = axes[1, 0]
    bb_p = Path(demo_dir) / "nvbit" / "bb_hot" / "hotspots_nvbit_marked_kernel_0.json"
    cp = Path(demo_dir) / "cupti" / "sassmetrics_core.json"
    plotted_bl = False
    if bb_p.exists():
        with open(bb_p) as f:
            bbd = json.load(f)
        entries = sorted(bbd.get("bb_entries", []), key=lambda e: -e.get("exec_count", 0))[:8]
        if entries:
            labels_bb = [f"BB{e['bb_index']}" for e in entries]
            counts_bb = [e["exec_count"] for e in entries]
            colors_bb = [PALETTE[0] if c == max(counts_bb) else PALETTE[5] for c in counts_bb]
            ax.barh(range(len(entries)), counts_bb, color=colors_bb, height=0.6)
            ax.set_yticks(range(len(entries)))
            ax.set_yticklabels(labels_bb, fontsize=8)
            ax.set_xlabel("Exec Count")
            ax.set_title("BB Hotspots", fontsize=11, fontweight="bold", color="#58a6ff")
            ax.grid(axis="x")
            ax.xaxis.set_major_formatter(ticker.EngFormatter())
            ax.invert_yaxis()
            plotted_bl = True
    if not plotted_bl and cp.exists():
        with open(cp) as f:
            cd = json.load(f)
        recs = cd.get("records", [])
        pcs_vals = [(r.get("pcOffset", 0), r.get("metrics", {}).get("smsp__sass_inst_executed", 0))
                    for r in recs]
        pcs_vals = [(pc, v) for pc, v in pcs_vals if v > 0]
        if pcs_vals:
            pcs_vals.sort()
            _, vals = zip(*pcs_vals)
            colors_arr = plt.cm.hot(np.array(vals) / max(vals))
            ax.bar(range(len(vals)), vals, color=colors_arr, width=1.0, linewidth=0)
            ax.set_xlabel("PC Index")
            ax.set_ylabel("Exec Count")
            ax.set_title("SASS Hotspot", fontsize=11, fontweight="bold", color="#58a6ff")
            ax.grid(axis="y")
            ax.yaxis.set_major_formatter(ticker.EngFormatter())

    # Bottom-right: join per-region
    ax = axes[1, 1]
    jp = Path(demo_dir) / "join" / "merged_sassmetrics.json"
    if jp.exists():
        with open(jp) as f:
            jd = json.load(f)
        jregions = jd.get("regions", [])
        lm = {0: "outside", 1: "compute", 2: "store"}
        jnames = [lm.get(r.get("region_id", 0), f"r{r.get('region_id',0)}") for r in jregions]
        jvals = [r.get("metrics_total", r.get("metrics", {})).get("smsp__sass_inst_executed", 0)
                 for r in jregions]
        if any(jvals):
            bars = ax.bar(range(len(jnames)), jvals, color=PALETTE[:len(jnames)], width=0.5)
            ax.set_xticks(range(len(jnames)))
            ax.set_xticklabels(jnames, fontsize=10)
            ax.set_ylabel("inst_executed")
            ax.set_title("NVBit+CUPTI Join", fontsize=11, fontweight="bold", color="#58a6ff")
            ax.grid(axis="y")
            ax.yaxis.set_major_formatter(ticker.EngFormatter())

    fig.suptitle("Intra-Kernel Profiler — Overview Dashboard",
                 fontsize=15, fontweight="bold", color="#58a6ff", y=1.02)
    fig.tight_layout()
    save(fig, out_dir / "overview_dashboard.png")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate visualization gallery for Intra-Kernel Profiler.")
    ap.add_argument("--demo-dir", required=True, help="Demo output directory (_demo_out)")
    ap.add_argument("--out-dir", required=True, help="Gallery output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualization gallery...")

    plot_trace_timeline(args.demo_dir, out_dir)
    plot_region_distributions(args.demo_dir, out_dir)
    plot_inst_class_breakdown(args.demo_dir, out_dir)
    plot_inst_pipeline(args.demo_dir, out_dir)
    plot_sass_metrics_heatmap(args.demo_dir, out_dir)
    plot_stall_reasons(args.demo_dir, out_dir)
    plot_divergence(args.demo_dir, out_dir)
    plot_join_analysis(args.demo_dir, out_dir)
    plot_branch_div_histogram(args.demo_dir, out_dir)
    plot_multi_profile(args.demo_dir, out_dir)
    plot_memory_instruction_mix(args.demo_dir, out_dir)
    plot_sass_annotated(args.demo_dir, out_dir)
    plot_mem_trace(args.demo_dir, out_dir)
    plot_bb_hotspots(args.demo_dir, out_dir)
    plot_branch_sites(args.demo_dir, out_dir)
    plot_locality_analysis(args.demo_dir, out_dir)
    plot_region_radar(args.demo_dir, out_dir)
    plot_region_summary_stats(args.demo_dir, out_dir)
    plot_sass_multi_metric(args.demo_dir, out_dir)
    plot_overview_dashboard(args.demo_dir, out_dir)

    print(f"Gallery: {out_dir}/")


if __name__ == "__main__":
    main()
