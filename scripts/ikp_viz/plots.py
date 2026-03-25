import math


PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac", "#6f4e7c", "#86bc86",
    "#d37295", "#fabfd2", "#7f7f7f", "#bcbd22",
]


def _svg_header(w, h):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _svg_footer():
    return ["</svg>"]


def _svg_text(x, y, text, size=12, anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-family="Arial" text-anchor="{anchor}">{text}</text>'


def _svg_line(x1, y1, x2, y2, stroke="#333", width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def _svg_rect(x, y, w, h, fill="#666", stroke="none"):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}"/>'


def _color_scale(v, vmax):
    if vmax <= 0:
        return "#f0f0f0"
    t = min(1.0, v / vmax)
    # blue-ish gradient
    r = int(240 - 120 * t)
    g = int(245 - 140 * t)
    b = int(255 - 180 * t)
    return f"rgb({r},{g},{b})"


def _write_svg(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_inst_class(region_stats, out_path):
    regions = region_stats["regions"]
    region_ids = sorted(regions.keys())
    keys = [
        "alu_fp32", "alu_int", "tensor_wgmma", "ld_global", "st_global",
        "ld_shared", "st_shared", "ld_local", "st_local", "barrier", "membar",
        "branch", "call", "ret", "special", "other",
    ]
    data = {k: [] for k in keys}
    for rid in region_ids:
        cls = regions[rid].get("inst_class", {})
        for k in keys:
            data[k].append(cls.get(k, 0))

    w, h = 900, 480
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 50
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b
    max_total = 1
    totals = []
    for i in range(len(region_ids)):
        total = sum(data[k][i] for k in keys)
        totals.append(total)
        max_total = max(max_total, total)

    lines = _svg_header(w, h)
    lines.append(_svg_text(w / 2, 24, "Instruction Mix by Region", 16, "middle"))
    lines.append(_svg_line(pad_l, pad_t, pad_l, pad_t + chart_h))
    lines.append(_svg_line(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h))

    bar_w = chart_w / max(1, len(region_ids))
    for i, rid in enumerate(region_ids):
        x = pad_l + i * bar_w + 4
        y = pad_t + chart_h
        for k_idx, k in enumerate(keys):
            val = data[k][i]
            if val == 0:
                continue
            bh = (val / max_total) * chart_h
            y -= bh
            lines.append(_svg_rect(x, y, bar_w - 8, bh, PALETTE[k_idx % len(PALETTE)]))
        lines.append(_svg_text(x + (bar_w - 8) / 2, pad_t + chart_h + 18, str(rid), 10, "middle"))

    # legend
    lx, ly = pad_l + chart_w - 240, pad_t + 10
    for k_idx, k in enumerate(keys):
        lines.append(_svg_rect(lx, ly + k_idx * 14, 10, 10, PALETTE[k_idx % len(PALETTE)]))
        lines.append(_svg_text(lx + 14, ly + k_idx * 14 + 10, k, 9))

    lines.extend(_svg_footer())
    _write_svg(out_path, lines)


def plot_mem_patterns(region_stats, out_dir):
    regions = region_stats["regions"]
    region_ids = sorted(regions.keys())

    def heatmap(data, title, filename, x_label, y_label):
        if not data:
            return
        rows = len(data)
        cols = len(data[0])
        w, h = 900, 360
        pad_l, pad_r, pad_t, pad_b = 80, 20, 40, 40
        cell_w = (w - pad_l - pad_r) / cols
        cell_h = (h - pad_t - pad_b) / rows
        vmax = max(max(row) for row in data) if data else 1

        lines = _svg_header(w, h)
        lines.append(_svg_text(w / 2, 24, title, 16, "middle"))
        for r in range(rows):
            for c in range(cols):
                x = pad_l + c * cell_w
                y = pad_t + r * cell_h
                lines.append(_svg_rect(x, y, cell_w, cell_h, _color_scale(data[r][c], vmax)))
        # axes labels
        for r, rid in enumerate(region_ids):
            lines.append(_svg_text(pad_l - 8, pad_t + (r + 0.7) * cell_h, str(rid), 10, "end"))
        lines.append(_svg_text(w / 2, h - 8, x_label, 11, "middle"))
        lines.append(_svg_text(16, h / 2, y_label, 11, "middle"))
        lines.extend(_svg_footer())
        _write_svg(f"{out_dir}/{filename}", lines)

    gsec = [regions[rid].get("gmem_sectors_per_inst_hist", []) for rid in region_ids]
    if gsec and any(len(row) for row in gsec):
        heatmap(gsec, "Gmem Sectors per Instruction (32B)", "gmem_sectors_hist.svg", "sectors", "region")

    sbank = [regions[rid].get("smem_bank_conflict_max_hist", []) for rid in region_ids]
    if sbank and any(len(row) for row in sbank):
        heatmap(sbank, "Shared Bank Conflict Max Degree", "smem_bank_conflict_hist.svg", "degree", "region")


def plot_branch_divergence(region_stats, out_path):
    regions = region_stats["regions"]
    region_ids = sorted(regions.keys())
    w, h = 900, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 40
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b
    vmax = 1
    series = []
    for rid in region_ids:
        hist = regions[rid].get("branch_div_hist", [])
        if hist:
            vmax = max(vmax, max(hist))
            series.append((rid, hist))
    if not series:
        return
    lines = _svg_header(w, h)
    lines.append(_svg_text(w / 2, 24, "Branch Divergence Histogram", 16, "middle"))
    lines.append(_svg_line(pad_l, pad_t, pad_l, pad_t + chart_h))
    lines.append(_svg_line(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h))

    for idx, (rid, hist) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        pts = []
        for i, v in enumerate(hist):
            x = pad_l + (i / max(1, len(hist) - 1)) * chart_w
            y = pad_t + chart_h - (v / vmax) * chart_h
            pts.append((x, y))
        for i in range(1, len(pts)):
            lines.append(_svg_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], color, 2))
        lines.append(_svg_text(pad_l + chart_w - 10, pad_t + 14 + idx * 14, f"region {rid}", 10, "end"))
    lines.extend(_svg_footer())
    _write_svg(out_path, lines)


def plot_working_set(locality_stats, out_path):
    regions = locality_stats.get("regions", {})
    w, h = 900, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 40
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    series = []
    vmax = 1
    for rid, r in regions.items():
        ws = r.get("working_set", {})
        xs = []
        ys = []
        for wsize, stats in ws.items():
            xs.append(int(wsize))
            ys.append(stats.get("p95", 0))
        if xs:
            xs, ys = zip(*sorted(zip(xs, ys)))
            vmax = max(vmax, max(ys))
            series.append((rid, xs, ys))
    if not series:
        return

    lines = _svg_header(w, h)
    lines.append(_svg_text(w / 2, 24, "Working Set (p95 unique lines)", 16, "middle"))
    lines.append(_svg_line(pad_l, pad_t, pad_l, pad_t + chart_h))
    lines.append(_svg_line(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h))

    for idx, (rid, xs, ys) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        pts = []
        for i, xval in enumerate(xs):
            x = pad_l + (i / max(1, len(xs) - 1)) * chart_w
            y = pad_t + chart_h - (ys[i] / vmax) * chart_h
            pts.append((x, y))
        for i in range(1, len(pts)):
            lines.append(_svg_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], color, 2))
        lines.append(_svg_text(pad_l + chart_w - 10, pad_t + 14 + idx * 14, f"region {rid}", 10, "end"))
    lines.extend(_svg_footer())
    _write_svg(out_path, lines)


def plot_reuse_distance(locality_stats, out_path):
    regions = locality_stats.get("regions", {})
    w, h = 900, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 40
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    series = []
    vmax = 1
    for rid, r in regions.items():
        hist = r.get("reuse_distance", {}).get("global", {}).get("hist", [])
        if hist:
            vmax = max(vmax, max(hist))
            series.append((rid, hist))
    if not series:
        return
    lines = _svg_header(w, h)
    lines.append(_svg_text(w / 2, 24, "Reuse Distance Histogram (Global)", 16, "middle"))
    lines.append(_svg_line(pad_l, pad_t, pad_l, pad_t + chart_h))
    lines.append(_svg_line(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h))
    for idx, (rid, hist) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        pts = []
        for i, v in enumerate(hist):
            x = pad_l + (i / max(1, len(hist) - 1)) * chart_w
            y = pad_t + chart_h - (v / vmax) * chart_h
            pts.append((x, y))
        for i in range(1, len(pts)):
            lines.append(_svg_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], color, 2))
        lines.append(_svg_text(pad_l + chart_w - 10, pad_t + 14 + idx * 14, f"region {rid}", 10, "end"))
    lines.extend(_svg_footer())
    _write_svg(out_path, lines)


def plot_inter_warp_sharing(locality_stats, out_path):
    regions = locality_stats.get("regions", {})
    w, h = 900, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 40
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    series = []
    vmax = 1
    for rid, r in regions.items():
        hist = r.get("inter_warp_sharing", {}).get("lines_by_warps", {})
        if hist:
            xs = sorted(int(k) for k in hist.keys())
            ys = [hist[str(k)] if isinstance(hist, dict) else hist.get(k, 0) for k in xs]
            vmax = max(vmax, max(ys) if ys else 1)
            series.append((rid, xs, ys))
    if not series:
        return
    lines = _svg_header(w, h)
    lines.append(_svg_text(w / 2, 24, "Inter-warp Line Sharing", 16, "middle"))
    lines.append(_svg_line(pad_l, pad_t, pad_l, pad_t + chart_h))
    lines.append(_svg_line(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h))
    for idx, (rid, xs, ys) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        pts = []
        for i in range(len(xs)):
            x = pad_l + (i / max(1, len(xs) - 1)) * chart_w
            y = pad_t + chart_h - (ys[i] / vmax) * chart_h
            pts.append((x, y))
        for i in range(1, len(pts)):
            lines.append(_svg_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], color, 2))
        lines.append(_svg_text(pad_l + chart_w - 10, pad_t + 14 + idx * 14, f"region {rid}", 10, "end"))
    lines.extend(_svg_footer())
    _write_svg(out_path, lines)


def plot_lane_address_heatmap(mem_trace_sample, out_path, line_bytes=128, max_records=200):
    sample = mem_trace_sample[:max_records]
    if not sample:
        return
    rows = len(sample)
    cols = 32
    w, h = 900, 400
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 40
    cell_w = (w - pad_l - pad_r) / cols
    cell_h = (h - pad_t - pad_b) / rows

    mat = []
    vmax = 1
    for rec in sample:
        addrs = rec.get("addrs", [])
        mask = rec.get("active_mask", 0)
        row = []
        for lane in range(32):
            if lane < len(addrs) and (mask & (1 << lane)):
                val = addrs[lane] // line_bytes
            else:
                val = 0
            row.append(val)
            vmax = max(vmax, val)
        mat.append(row)

    lines = _svg_header(w, h)
    lines.append(_svg_text(w / 2, 24, "Lane Address Heatmap (line id)", 16, "middle"))
    for r in range(rows):
        for c in range(cols):
            x = pad_l + c * cell_w
            y = pad_t + r * cell_h
            lines.append(_svg_rect(x, y, cell_w, cell_h, _color_scale(mat[r][c], vmax)))
    lines.extend(_svg_footer())
    _write_svg(out_path, lines)
