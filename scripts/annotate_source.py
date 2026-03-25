#!/usr/bin/env python3
"""Annotate kernel source code with per-line SASS metrics and region attribution.

Usage:
    # CUPTI source metrics only:
    python3 scripts/annotate_source.py \
        --sass sassmetrics_source.json \
        --source examples/cupti/minimal_cupti_target.cu

    # CUPTI + NVBit join (source + region labels):
    python3 scripts/annotate_source.py \
        --sass sassmetrics_core.json sassmetrics_divergence.json \
        --pc2region pc2region_region_demo_kernel_0.json \
        --source examples/region_demo/minimal_region_target.cu \
        --labels "0:outside,1:compute,2:store" \
        --html annotated_source.html

Reads CUPTI SASS metrics with enable_source=1 and aggregates per source line.
Optionally joins with NVBit pc2region to label each line's region.
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def load_sass_metrics(paths):
    """Load and merge SASS metrics from one or more JSON files."""
    all_records = []
    profiles = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        profile = data.get("metrics_profile", os.path.basename(p))
        profiles.append(profile)
        for rec in data.get("records", []):
            rec["_profile"] = profile
            all_records.append(rec)
    return all_records, profiles


def load_pc2region(paths):
    """Load pc2region mapping from NVBit JSON files."""
    mapping = {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for entry in data.get("pc2region", []):
            pc = entry.get("pc_offset", entry.get("pcOffset"))
            region = entry.get("dominant_region", entry.get("region"))
            if pc is not None and region is not None:
                mapping[int(pc)] = int(region)
    return mapping


def aggregate_by_line(records, pc2region):
    """Aggregate metrics by source file:line. Returns {file: {line: info}}."""
    by_file = defaultdict(lambda: defaultdict(lambda: {
        "metrics": defaultdict(float),
        "regions": defaultdict(int),
        "pc_count": 0,
    }))

    for rec in records:
        src = rec.get("source")
        if not src or not src.get("file") or not src.get("line"):
            continue

        filepath = src["file"]
        line = src["line"]
        info = by_file[filepath][line]
        info["pc_count"] += 1

        for metric, value in rec.get("metrics", {}).items():
            info["metrics"][metric] += value

        pc = rec.get("pcOffset")
        if pc is not None and pc2region:
            region = pc2region.get(int(pc))
            if region is not None:
                info["regions"][region] += 1

    return by_file


def dominant_region(regions_dict):
    """Get the dominant region for a source line."""
    if not regions_dict:
        return None
    return max(regions_dict, key=regions_dict.get)


def format_metric_value(value):
    """Format a metric value for display."""
    if value == 0:
        return "-"
    if value >= 1e9:
        return f"{value/1e9:.1f}G"
    if value >= 1e6:
        return f"{value/1e6:.1f}M"
    if value >= 1e3:
        return f"{value/1e3:.1f}K"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(int(value))


def pick_display_metrics(by_line):
    """Choose the most informative metrics to display."""
    # Count how many lines have each metric
    metric_coverage = defaultdict(int)
    for line_num, info in by_line.items():
        for m in info["metrics"]:
            metric_coverage[m] += 1

    # Priority order
    priority = [
        "smsp__sass_inst_executed",
        "smsp__sass_thread_inst_executed",
        "smsp__sass_thread_inst_executed_pred_on",
        "smsp__sass_inst_executed_op_fp32",
        "smsp__sass_inst_executed_op_integer",
        "smsp__sass_inst_executed_op_memory",
        "smsp__sass_inst_executed_op_control",
    ]

    selected = []
    for m in priority:
        if metric_coverage.get(m, 0) > 0:
            selected.append(m)

    # Add remaining metrics sorted by coverage
    for m, count in sorted(metric_coverage.items(), key=lambda x: -x[1]):
        if m not in selected:
            selected.append(m)

    return selected[:5]  # Show at most 5 columns


def short_metric_name(metric):
    """Shorten metric names for display."""
    name = metric
    name = name.replace("smsp__sass_", "")
    name = name.replace("_executed", "")
    name = name.replace("thread_inst", "thr_inst")
    name = name.replace("inst_", "")
    name = name.replace("_pred_on", "_pred")
    name = name.replace("op_", "")
    return name


def render_terminal(source_lines, by_line, labels, display_metrics):
    """Render annotated source to terminal with colors."""
    # ANSI colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    REGION_COLORS = [DIM, GREEN, CYAN, YELLOW, MAGENTA, RED, BLUE]

    # Find max metric value for heatmap scaling
    max_inst = 0
    for info in by_line.values():
        v = info["metrics"].get("smsp__sass_inst_executed", 0)
        max_inst = max(max_inst, v)

    # Header
    metric_headers = [short_metric_name(m) for m in display_metrics]
    col_width = 10
    header_cols = "".join(f"{h:>{col_width}}" for h in metric_headers)
    print(f"\n{BOLD}{'Line':>6}  {'Region':>12}  {header_cols}  Source{RESET}")
    print(f"{'─' * 6}  {'─' * 12}  {'─' * (col_width * len(display_metrics))}  {'─' * 60}")

    for i, line in enumerate(source_lines, start=1):
        info = by_line.get(i)
        if info and info["pc_count"] > 0:
            # Region
            dr = dominant_region(info["regions"])
            if dr is not None:
                region_str = labels.get(dr, f"region_{dr}")
                region_color = REGION_COLORS[dr % len(REGION_COLORS)]
            else:
                region_str = ""
                region_color = DIM

            # Metrics columns
            cols = ""
            for m in display_metrics:
                v = info["metrics"].get(m, 0)
                cols += f"{format_metric_value(v):>{col_width}}"

            # Intensity coloring based on instruction count
            inst = info["metrics"].get("smsp__sass_inst_executed", 0)
            if max_inst > 0:
                frac = inst / max_inst
                if frac > 0.8:
                    line_color = RED + BOLD
                elif frac > 0.3:
                    line_color = YELLOW
                elif frac > 0.0:
                    line_color = ""
                else:
                    line_color = DIM
            else:
                line_color = ""

            print(f"{line_color}{i:>6}  {region_color}{region_str:>12}{RESET}  "
                  f"{line_color}{cols}{RESET}  {line.rstrip()}")
        else:
            print(f"{DIM}{i:>6}  {'':>12}  {'':>{col_width * len(display_metrics)}}  {line.rstrip()}{RESET}")

    print()


def render_html(source_lines, by_line, labels, display_metrics, output_path, source_path):
    """Render annotated source as a standalone HTML file."""
    import html as html_mod

    max_inst = 0
    for info in by_line.values():
        v = info["metrics"].get("smsp__sass_inst_executed", 0)
        max_inst = max(max_inst, v)

    region_colors = {
        0: "#6e7681",   # outside — gray
        1: "#3fb950",   # compute — green
        2: "#58a6ff",   # store — blue
        3: "#d2a8ff",   # purple
        4: "#f78166",   # orange
        5: "#ffa657",   # yellow-orange
    }

    metric_headers = [short_metric_name(m) for m in display_metrics]

    rows_html = []
    for i, line in enumerate(source_lines, start=1):
        info = by_line.get(i)
        escaped_line = html_mod.escape(line.rstrip())

        if info and info["pc_count"] > 0:
            dr = dominant_region(info["regions"])
            region_str = labels.get(dr, f"region_{dr}") if dr is not None else ""
            region_color = region_colors.get(dr, "#8b949e") if dr is not None else "#8b949e"

            inst = info["metrics"].get("smsp__sass_inst_executed", 0)
            if max_inst > 0:
                frac = inst / max_inst
            else:
                frac = 0

            if frac > 0.8:
                bg = "rgba(248, 81, 73, 0.25)"
                text_color = "#ff7b72"
            elif frac > 0.3:
                bg = "rgba(210, 153, 34, 0.15)"
                text_color = "#e3b341"
            elif frac > 0.0:
                bg = "rgba(56, 58, 64, 0.5)"
                text_color = "#c9d1d9"
            else:
                bg = "transparent"
                text_color = "#8b949e"

            # Bar width for visual heat indicator
            bar_width = max(2, int(frac * 100)) if frac > 0 else 0
            bar_color = f"rgba(248, 81, 73, {min(frac * 1.5, 0.8):.2f})"

            metric_cells = ""
            for m in display_metrics:
                v = info["metrics"].get(m, 0)
                metric_cells += f'<td class="metric">{format_metric_value(v)}</td>'

            rows_html.append(
                f'<tr style="background:{bg}; color:{text_color}">'
                f'<td class="lineno">{i}</td>'
                f'<td class="heat"><div class="heatbar" style="width:{bar_width}%;background:{bar_color}"></div></td>'
                f'<td class="region" style="color:{region_color}">{html_mod.escape(region_str)}</td>'
                f'{metric_cells}'
                f'<td class="code"><pre>{escaped_line}</pre></td>'
                f'</tr>'
            )
        else:
            empty_cells = '<td class="metric">-</td>' * len(display_metrics)
            rows_html.append(
                f'<tr style="color:#6e7681">'
                f'<td class="lineno">{i}</td>'
                f'<td class="heat"></td>'
                f'<td class="region"></td>'
                f'{empty_cells}'
                f'<td class="code"><pre>{escaped_line}</pre></td>'
                f'</tr>'
            )

    metric_headers_html = "".join(f'<th class="metric">{html_mod.escape(h)}</th>' for h in metric_headers)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Annotated Source — {html_mod.escape(os.path.basename(source_path))}</title>
<style>
body {{
  background: #0d1117;
  color: #c9d1d9;
  font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;
  font-size: 13px;
  margin: 0;
  padding: 20px;
}}
h1 {{
  color: #58a6ff;
  font-size: 18px;
  margin-bottom: 4px;
}}
.subtitle {{
  color: #8b949e;
  font-size: 12px;
  margin-bottom: 16px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
}}
tr:hover {{
  background: rgba(110, 118, 129, 0.15) !important;
}}
td, th {{
  padding: 1px 8px;
  vertical-align: top;
  white-space: nowrap;
  border: none;
}}
th {{
  color: #8b949e;
  font-weight: 600;
  text-align: right;
  border-bottom: 1px solid #21262d;
  padding-bottom: 4px;
}}
th.code {{
  text-align: left;
}}
.lineno {{
  color: #484f58;
  text-align: right;
  user-select: none;
  width: 40px;
  min-width: 40px;
}}
.heat {{
  width: 100px;
  min-width: 100px;
  padding: 0;
  vertical-align: middle;
}}
.heatbar {{
  height: 14px;
  border-radius: 2px;
  min-width: 0;
}}
.region {{
  text-align: right;
  font-weight: 600;
  font-size: 11px;
  width: 80px;
  min-width: 80px;
}}
.metric {{
  text-align: right;
  font-variant-numeric: tabular-nums;
  width: 75px;
  min-width: 75px;
}}
.code {{
  text-align: left;
  width: 100%;
}}
.code pre {{
  margin: 0;
  font-family: inherit;
  white-space: pre;
}}
.legend {{
  display: flex;
  gap: 16px;
  margin-bottom: 12px;
  font-size: 12px;
  color: #8b949e;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 4px;
}}
.legend-dot {{
  width: 10px;
  height: 10px;
  border-radius: 2px;
  display: inline-block;
}}
</style>
</head>
<body>
<h1>Annotated Source: {html_mod.escape(os.path.basename(source_path))}</h1>
<div class="subtitle">
  SASS metrics aggregated per source line | Intra-Kernel Profiler
</div>
<div class="legend">
  <span class="legend-item"><span class="legend-dot" style="background:rgba(248,81,73,0.6)"></span> Hot (&gt;80%)</span>
  <span class="legend-item"><span class="legend-dot" style="background:rgba(210,153,34,0.4)"></span> Warm (&gt;30%)</span>
  <span class="legend-item"><span class="legend-dot" style="background:rgba(56,58,64,0.8)"></span> Cold</span>
</div>
<table>
<thead>
<tr>
  <th>Line</th>
  <th class="heat">Heat</th>
  <th class="region">Region</th>
  {metric_headers_html}
  <th class="code">Source</th>
</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate kernel source with per-line SASS metrics and region labels.")
    parser.add_argument("--sass", required=True, nargs="+",
                        help="CUPTI sassmetrics JSON (with enable_source=1)")
    parser.add_argument("--pc2region", nargs="*", default=[],
                        help="NVBit pc2region JSON (for region labels)")
    parser.add_argument("--source", required=True,
                        help="Kernel source file (.cu)")
    parser.add_argument("--labels", default="",
                        help="Region labels: 0:outside,1:compute,2:store")
    parser.add_argument("--html", default="",
                        help="Output HTML file (optional)")
    parser.add_argument("--function", default="",
                        help="Filter to a specific function name")
    args = parser.parse_args()

    # Parse labels
    labels = {}
    if args.labels:
        for pair in args.labels.split(","):
            k, v = pair.split(":")
            labels[int(k)] = v

    # Load data
    records, profiles = load_sass_metrics(args.sass)
    print(f"Loaded {len(records)} SASS records from {len(profiles)} profile(s): {', '.join(profiles)}")

    # Check source mapping
    src_count = sum(1 for r in records if r.get("source") and r["source"].get("file"))
    if src_count == 0:
        print("ERROR: No source mapping found in SASS metrics.", file=sys.stderr)
        print("  Make sure you ran with IKP_CUPTI_SASS_ENABLE_SOURCE=1", file=sys.stderr)
        print("  and compiled the target with -lineinfo", file=sys.stderr)
        sys.exit(1)
    print(f"  {src_count}/{len(records)} records have source mapping")

    # Optionally filter by function
    if args.function:
        records = [r for r in records if args.function in r.get("functionName", "")]
        print(f"  Filtered to function '{args.function}': {len(records)} records")

    # Load pc2region
    pc2region = {}
    if args.pc2region:
        pc2region = load_pc2region(args.pc2region)
        print(f"Loaded pc2region: {len(pc2region)} PC entries")

    # Aggregate by source line
    by_file = aggregate_by_line(records, pc2region)

    # Find matching file
    source_path = os.path.abspath(args.source)
    matched_file = None
    for f in by_file:
        if os.path.basename(f) == os.path.basename(source_path):
            matched_file = f
            break
        if source_path.endswith(f) or f.endswith(source_path):
            matched_file = f
            break

    if matched_file is None and by_file:
        # Try partial match
        src_base = os.path.basename(source_path)
        for f in by_file:
            if os.path.basename(f) == src_base:
                matched_file = f
                break

    if matched_file is None:
        print(f"WARNING: Source file '{source_path}' not found in SASS records.", file=sys.stderr)
        print(f"  Files found in records: {list(by_file.keys())}", file=sys.stderr)
        if by_file:
            matched_file = list(by_file.keys())[0]
            print(f"  Using first match: {matched_file}", file=sys.stderr)
        else:
            sys.exit(1)

    by_line = by_file[matched_file]
    print(f"  Source lines with data: {len(by_line)}")

    # Read source
    with open(args.source, "r") as f:
        source_lines = f.readlines()

    # Pick metrics to display
    display_metrics = pick_display_metrics(by_line)
    print(f"  Metrics: {', '.join(short_metric_name(m) for m in display_metrics)}")

    # Render
    render_terminal(source_lines, by_line, labels, display_metrics)

    if args.html:
        render_html(source_lines, by_line, labels, display_metrics, args.html, args.source)


if __name__ == "__main__":
    main()
