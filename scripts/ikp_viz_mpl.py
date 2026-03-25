#!/usr/bin/env python3
"""
Matplotlib-based visualization for Intra-Kernel Profiler NVBit profiler outputs.

Design goals:
- "Sellable" dashboard: region fingerprints + memory patterns + locality + trace.
- Avoid clutter: use heatmaps, small multiples, CDFs, top-k tables.
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _first_glob(pat):
    if any(ch in pat for ch in ["*", "?", "[", "]"]):
        xs = sorted(glob.glob(pat))
        if not xs:
            raise FileNotFoundError(pat)
        return xs[0]
    return pat


def load_region_stats(path_or_glob):
    path = _first_glob(path_or_glob)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    regions = {r["region"]: r for r in data.get("regions", [])}
    return data.get("kernel", ""), regions, path


def load_locality(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_mem_trace(paths, max_records=None):
    n = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)
                n += 1
                if max_records and n >= max_records:
                    return


def style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


INST_KEYS = [
    "alu_fp32",
    "alu_int",
    "tensor_wgmma",
    "ld_global",
    "st_global",
    "ld_shared",
    "st_shared",
    "ld_local",
    "st_local",
    "barrier",
    "membar",
    "branch",
    "call",
    "ret",
    "special",
    "other",
]

HIST_BOUNDS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def plot_inst_mix(regions, out_png):
    style()
    rids = sorted(regions.keys())
    mat = np.zeros((len(rids), len(INST_KEYS)), dtype=np.float64)
    for i, rid in enumerate(rids):
        cls = regions[rid].get("inst_class", {})
        for j, k in enumerate(INST_KEYS):
            mat[i, j] = cls.get(k, 0)

    totals = mat.sum(axis=1)
    frac = np.divide(mat, totals[:, None] + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = np.zeros(len(rids))
    cmap = plt.get_cmap("tab20")
    for j, k in enumerate(INST_KEYS):
        ax.bar([str(r) for r in rids], frac[:, j], bottom=bottoms, label=k, color=cmap(j))
        bottoms += frac[:, j]
    ax.set_title("Instruction Mix by Region (normalized)")
    ax.set_xlabel("Region")
    ax.set_ylabel("Fraction")
    ax.legend(ncol=4, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def _sorted_region_ids_from_locality(locality):
    try:
        rids = sorted(int(k) for k in locality.get("regions", {}).keys())
        return rids
    except Exception:
        return []


def plot_mem_heatmaps(regions, out_png):
    style()
    rids = sorted(regions.keys())
    gsec = []
    sbank = []
    for rid in rids:
        gsec.append(regions[rid].get("gmem_sectors_per_inst_hist", []))
        sbank.append(regions[rid].get("smem_bank_conflict_max_hist", []))
    gsec = np.array(gsec, dtype=np.float64)
    sbank = np.array(sbank, dtype=np.float64)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    if gsec.size:
        im = axs[0].imshow(gsec + 1e-9, aspect="auto", norm=LogNorm(vmin=1e-6, vmax=max(1.0, gsec.max())))
        axs[0].set_title("Gmem sectors per instruction hist (log)")
        axs[0].set_xlabel("sectors bins (0..32)")
        axs[0].set_ylabel("region")
        axs[0].set_yticks(range(len(rids)))
        axs[0].set_yticklabels([str(r) for r in rids])
        fig.colorbar(im, ax=axs[0], fraction=0.046)
    if sbank.size:
        im = axs[1].imshow(sbank + 1e-9, aspect="auto", norm=LogNorm(vmin=1e-6, vmax=max(1.0, sbank.max())))
        axs[1].set_title("Smem bank conflict max degree hist (log)")
        axs[1].set_xlabel("degree bins (0..32)")
        axs[1].set_ylabel("region")
        axs[1].set_yticks(range(len(rids)))
        axs[1].set_yticklabels([str(r) for r in rids])
        fig.colorbar(im, ax=axs[1], fraction=0.046)
    fig.suptitle("Memory Pattern Heatmaps", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_working_set(locality, out_png):
    style()
    loc_regions = locality.get("regions", {})
    win_list = [int(x) for x in locality.get("window_records", [128, 512, 2048])]
    rids = _sorted_region_ids_from_locality(locality)
    if not rids:
        return

    mat = np.zeros((len(rids), len(win_list)), dtype=np.float64)
    for i, rid in enumerate(rids):
        ws = loc_regions.get(str(rid), {}).get("working_set", {})
        for j, w in enumerate(win_list):
            mat[i, j] = float(ws.get(str(w), {}).get("p95", 0) or 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat + 1e-9, aspect="auto", norm=LogNorm(vmin=1e-6, vmax=max(1.0, float(mat.max()))), cmap="magma")
    ax.set_title("Working set (p95) heatmap")
    ax.set_ylabel("region")
    ax.set_xlabel("window (records)")
    ax.set_yticks(range(len(rids)))
    ax.set_yticklabels([str(r) for r in rids])
    ax.set_xticks(range(len(win_list)))
    ax.set_xticklabels([str(w) for w in win_list])
    fig.colorbar(im, ax=ax, fraction=0.046, label="unique lines (p95)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_reuse_distance(locality, out_png, scope="global"):
    style()
    loc_regions = locality.get("regions", {})
    rids = _sorted_region_ids_from_locality(locality)
    if not rids:
        return

    # matrix of hist bins per region (log)
    bins = len(HIST_BOUNDS) + 1
    mat = np.zeros((len(rids), bins), dtype=np.float64)
    for i, rid in enumerate(rids):
        rd = loc_regions.get(str(rid), {}).get("reuse_distance", {}).get(scope, {})
        hist = rd.get("hist", [])
        for j in range(min(bins, len(hist))):
            mat[i, j] = float(hist[j])

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mat + 1e-9, aspect="auto", norm=LogNorm(vmin=1e-6, vmax=max(1.0, float(mat.max()))), cmap="viridis")
    ax.set_title(f"Reuse distance histogram heatmap (scope={scope}, log)")
    ax.set_ylabel("region")
    ax.set_xlabel("reuse distance bin (<= bound)")
    ax.set_yticks(range(len(rids)))
    ax.set_yticklabels([str(r) for r in rids])
    ax.set_xticks(list(range(0, bins, 2)))
    ax.set_xticklabels([str(HIST_BOUNDS[i]) if i < len(HIST_BOUNDS) else "inf" for i in range(0, bins, 2)], rotation=30, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046, label="count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_inter_warp_sharing(locality, out_png):
    style()
    loc_regions = locality.get("regions", {})
    rids = _sorted_region_ids_from_locality(locality)
    if not rids:
        return

    ratios = []
    avgs = []
    for rid in rids:
        iw = loc_regions.get(str(rid), {}).get("inter_warp_sharing", {})
        ratios.append(float(iw.get("shared_line_ratio", 0) or 0))
        avgs.append(float(iw.get("avg_warps_per_line", 0) or 0))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(rids))
    axs[0].bar(xs, avgs, color="#4e79a7", alpha=0.9)
    axs[0].set_title("Avg warps per cache line")
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels([str(r) for r in rids])
    axs[0].set_xlabel("region")
    axs[0].set_ylabel("avg warps/line")

    axs[1].bar(xs, ratios, color="#e15759", alpha=0.9)
    axs[1].set_title("Shared line ratio")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels([str(r) for r in rids])
    axs[1].set_xlabel("region")
    axs[1].set_ylabel("ratio")
    axs[1].set_ylim(0, 1.05)

    fig.suptitle("Inter-warp line sharing", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_region_fingerprint(regions, locality, out_png):
    style()
    # Build a compact feature table per region (normalize to z-score on log scale).
    rids = sorted(regions.keys())

    def safe(x):
        return float(x) if x is not None else 0.0

    feat_names = [
        "inst_total",
        "tensor_wgmma_frac",
        "gmem_req_bytes",
        "smem_req_bytes",
        "gmem_sectors_32b",
        "gmem_unique_lines_est",
        "branch_div_entropy",
        "working_set_p95_512",
        "shared_line_ratio",
    ]
    X = np.zeros((len(rids), len(feat_names)), dtype=np.float64)

    loc_regions = locality.get("regions", {})
    for i, rid in enumerate(rids):
        r = regions[rid]
        inst_total = safe(r.get("inst_total", 0))
        cls = r.get("inst_class", {})
        tensor = safe(cls.get("tensor_wgmma", 0))
        denom = sum(safe(cls.get(k, 0)) for k in INST_KEYS) + 1e-9
        tensor_frac = tensor / denom
        X[i, 0] = inst_total
        X[i, 1] = tensor_frac
        X[i, 2] = safe(r.get("gmem_req_bytes", r.get("gmem_bytes", 0)))
        X[i, 3] = safe(r.get("smem_req_bytes", r.get("smem_bytes", 0)))
        X[i, 4] = safe(r.get("gmem_sectors_32b", 0))
        X[i, 5] = safe(r.get("gmem_unique_lines_est", 0))
        X[i, 6] = safe(r.get("branch_div_entropy", 0))

        loc = loc_regions.get(str(rid), {})
        ws = loc.get("working_set", {}).get("512", {})
        X[i, 7] = safe(ws.get("p95", 0))
        X[i, 8] = safe(loc.get("inter_warp_sharing", {}).get("shared_line_ratio", 0))

    # log-scale select columns
    X_log = X.copy()
    for j in [0, 2, 3, 4, 5, 7]:
        X_log[:, j] = np.log10(X_log[:, j] + 1.0)

    # z-score
    mu = X_log.mean(axis=0)
    sig = X_log.std(axis=0) + 1e-9
    Z = (X_log - mu) / sig

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(Z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_title("Region Fingerprint (normalized features)")
    ax.set_yticks(range(len(rids)))
    ax.set_yticklabels([str(r) for r in rids])
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=30, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046, label="z-score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def mem_trace_dashboard(mem_trace_paths, out_png, region=None, space=None, max_records=400, line_bytes=128):
    style()
    recs = []
    for rec in iter_mem_trace(mem_trace_paths, max_records=None):
        if region is not None and rec.get("region") != region:
            continue
        if space is not None and rec.get("space") != space:
            continue
        recs.append(rec)
        if len(recs) >= max_records:
            break
    if not recs:
        return False

    lanes = 32
    # matrix of normalized deltas (element index delta) OR fallback scatter
    deltas = np.full((len(recs), lanes), np.nan, dtype=np.float64)
    sectors = np.zeros(len(recs), dtype=np.int32)
    pcs = []
    popcs = np.zeros(len(recs), dtype=np.int32)
    for i, r in enumerate(recs):
        addrs = r.get("addrs", [])
        mask = int(r.get("active_mask", 0))
        popcs[i] = mask.bit_count()
        size = max(1, int(r.get("access_size", 1)))
        active_addrs = [addrs[l] for l in range(min(lanes, len(addrs))) if (mask >> l) & 1 and addrs[l] != 0]
        if active_addrs:
            base = min(active_addrs)
        else:
            base = 0
        sec_set = set()
        for l in range(lanes):
            if l < len(addrs) and ((mask >> l) & 1) and addrs[l] != 0:
                deltas[i, l] = (addrs[l] - base) / size
                sec_set.add(addrs[l] >> 5)
        sectors[i] = len(sec_set)
        pcs.append(int(r.get("pc_offset", 0)))

    avg_active = float(popcs.mean()) if len(popcs) else 0.0

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], width_ratios=[1.5, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Panel A: heatmap (when many lanes) OR address-time scatter (lane-leader patterns)
    if avg_active >= 4.0:
        vmax = np.nanpercentile(deltas, 99)
        vmax = max(1.0, float(vmax))
        im = ax0.imshow(np.ma.masked_invalid(deltas), aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax0.set_title("Lane address delta heatmap (normalized)")
        ax0.set_xlabel("lane")
        ax0.set_ylabel("record index")
        fig.colorbar(im, ax=ax0, fraction=0.03, pad=0.02, label="delta (elements)")
    else:
        xs = []
        ys = []
        cs = []
        for i, r in enumerate(recs):
            addrs = r.get("addrs", [])
            mask = int(r.get("active_mask", 0))
            warp = int(r.get("warp", 0))
            for l in range(lanes):
                if l < len(addrs) and ((mask >> l) & 1) and addrs[l] != 0:
                    xs.append(i)
                    ys.append((int(addrs[l]) // int(line_bytes)))
                    cs.append(warp)
        if ys:
            y0 = min(ys)
            ys = [y - y0 for y in ys]
        sc = ax0.scatter(xs, ys, c=cs, cmap="tab10", s=10, alpha=0.85, linewidths=0)
        ax0.set_title(f"Address timeline scatter (active lanes ~ {avg_active:.1f})")
        ax0.set_xlabel("record index")
        ax0.set_ylabel(f"cache line (relative, {line_bytes}B)")
        fig.colorbar(sc, ax=ax0, fraction=0.03, pad=0.02, label="warp id")

    # sectors time series
    ax1.plot(sectors, lw=1.8, color="#4e79a7")
    ax1.set_title("Sectors touched per record (32B)")
    ax1.set_xlabel("record index")
    ax1.set_ylabel("unique sectors")

    # Panel C: either coalescing footprint hist OR cache-line step hist
    if avg_active >= 4.0:
        ax2.hist(sectors, bins=range(0, int(sectors.max()) + 2), color="#59a14f", alpha=0.85)
        ax2.set_title("Coalescing footprint distribution")
        ax2.set_xlabel("unique sectors")
        ax2.set_ylabel("records")
    else:
        # Use first active lane's cache line as a proxy and show step histogram.
        lines = []
        for r in recs:
            addrs = r.get("addrs", [])
            mask = int(r.get("active_mask", 0))
            line = None
            for l in range(lanes):
                if l < len(addrs) and ((mask >> l) & 1) and addrs[l] != 0:
                    line = int(addrs[l]) // int(line_bytes)
                    break
            if line is not None:
                lines.append(line)
        if len(lines) >= 2:
            steps = np.diff(np.array(lines, dtype=np.int64))
            ax2.hist(steps, bins=50, color="#59a14f", alpha=0.85)
            ax2.set_title("Cache-line step distribution (proxy)")
            ax2.set_xlabel("Δ line (signed)")
            ax2.set_ylabel("records")
        else:
            ax2.text(0.5, 0.5, "not enough records", ha="center", va="center")
            ax2.set_axis_off()

    # top PC hotspots
    pc_ctr = Counter(pcs)
    top = pc_ctr.most_common(12)
    labels = [f"0x{pc:x}" for pc, _ in top]
    vals = [c for _, c in top]
    ax3.barh(labels[::-1], vals[::-1], color="#e15759", alpha=0.85)
    ax3.set_title("Top PC offsets (trace count)")
    ax3.set_xlabel("records")

    title = "mem_trace dashboard"
    if region is not None:
        title += f" | region={region}"
    if space is not None:
        title += f" | space={space}"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return True


def write_html(out_dir, items):
    html = os.path.join(out_dir, "report.html")
    with open(html, "w", encoding="utf-8") as f:
        f.write("<html><head><style>\n")
        f.write("body{font-family:Arial, sans-serif;margin:24px;} img{max-width:100%;margin:12px 0;border:1px solid #ddd;}\n")
        f.write("</style></head><body>\n")
        f.write("<h1>Intra-Kernel Profiler NVBit Profiler Report (matplotlib)</h1>\n")
        for title, fn in items:
            f.write(f"<h2>{title}</h2>\n")
            f.write(f"<img src=\"{fn}\"/>\n")
        f.write("</body></html>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-stats", required=True, help="region_stats json (or glob)")
    ap.add_argument("--locality", required=True, help="locality stats json")
    ap.add_argument("--mem-trace", required=True, help="mem_trace jsonl (or glob)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--max-trace-records", type=int, default=400, help="records per dashboard")
    ap.add_argument("--line-bytes", type=int, default=128, help="cache line size for trace plots")
    ap.add_argument("--dashboards", type=int, default=3, help="how many (region,space) dashboards to render")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kernel, regions, rs_path = load_region_stats(args.region_stats)
    locality = load_locality(args.locality)

    mem_paths = []
    if any(ch in args.mem_trace for ch in ["*", "?", "[", "]"]):
        mem_paths = sorted(glob.glob(args.mem_trace))
    else:
        mem_paths = [args.mem_trace]
    if not mem_paths:
        raise FileNotFoundError(args.mem_trace)

    items = []

    fn = "00_region_fingerprint.png"
    plot_region_fingerprint(regions, locality, os.path.join(args.out_dir, fn))
    items.append(("Region fingerprint", fn))

    fn = "01_inst_mix.png"
    plot_inst_mix(regions, os.path.join(args.out_dir, fn))
    items.append(("Instruction mix (normalized)", fn))

    fn = "02_mem_heatmaps.png"
    plot_mem_heatmaps(regions, os.path.join(args.out_dir, fn))
    items.append(("Memory pattern heatmaps", fn))

    fn = "03_working_set.png"
    plot_working_set(locality, os.path.join(args.out_dir, fn))
    items.append(("Working set (p95) heatmap", fn))

    fn = "04_reuse_distance.png"
    plot_reuse_distance(locality, os.path.join(args.out_dir, fn), scope="global")
    items.append(("Reuse distance heatmap (global scope)", fn))

    fn = "05_inter_warp_sharing.png"
    plot_inter_warp_sharing(locality, os.path.join(args.out_dir, fn))
    items.append(("Inter-warp line sharing", fn))

    # mem trace dashboards: pick top-K (region,space) pairs by record count (auto)
    rs_ctr = Counter()
    for rec in iter_mem_trace(mem_paths, max_records=200000):
        rs_ctr[(rec.get("region"), rec.get("space"))] += 1
    top_pairs = [k for k, _ in rs_ctr.most_common(max(1, int(args.dashboards)))]
    dash_idx = 6
    for rid, sp in top_pairs:
        fn = f"{dash_idx:02d}_memtrace_region{rid}_{sp}.png"
        ok = mem_trace_dashboard(
            mem_paths,
            os.path.join(args.out_dir, fn),
            region=rid,
            space=sp,
            max_records=args.max_trace_records,
            line_bytes=args.line_bytes,
        )
        if ok:
            items.append((f"mem_trace dashboard (region {rid}, {sp})", fn))
            dash_idx += 1

    write_html(args.out_dir, items)
    print("kernel:", kernel)
    print("region_stats:", rs_path)
    print("out_dir:", args.out_dir)


if __name__ == "__main__":
    main()

