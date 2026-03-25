#!/usr/bin/env python3
"""Generate a Compiler-Explorer-style interactive profiler viewer.

Uses Monaco Editor (CDN) for CUDA/PTX/SASS code panels, Split.js for
resizable layout, and ECharts for interactive metric charts.  All profiler
data is consolidated into a single self-contained HTML page.

Usage:
    python3 scripts/generate_explorer.py \\
        --demo-dir _demo_out \\
        --source examples/cupti/minimal_cupti_target.cu \\
        --output explorer.html

    # Start a local server to view (Monaco requires HTTP, not file://):
    python3 scripts/generate_explorer.py \\
        --demo-dir _demo_out \\
        --source examples/cupti/minimal_cupti_target.cu \\
        --output explorer.html --serve
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict


# ── Data loaders ──────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_sass_records(paths):
    all_records, profiles_seen = [], set()
    for p in paths:
        data = load_json(p)
        profile = data.get("metrics_profile", os.path.basename(p))
        profiles_seen.add(profile)
        for rec in data.get("records", []):
            rec["_profile"] = profile
            all_records.append(rec)
    return all_records, sorted(profiles_seen)


def collect_pc2region(paths):
    mapping = {}
    for p in paths:
        data = load_json(p)
        for entry in data.get("pc2region", []):
            pc = entry.get("pc_offset", entry.get("pcOffset"))
            region = entry.get("dominant_region", entry.get("region"))
            if pc is not None and region is not None:
                mapping[int(pc)] = int(region)
    return mapping


def collect_region_stats(paths):
    regions = {}
    for p in paths:
        data = load_json(p)
        for r in data.get("regions", []):
            rid = r["region"]
            # Prefer the entry with more data (higher inst_total)
            if rid not in regions or r.get("inst_total", 0) > regions[rid].get("inst_total", 0):
                regions[rid] = r
    return regions


def collect_trace_summary(demo_dir):
    path = os.path.join(demo_dir, "trace", "gemm_trace_summary.json")
    if os.path.exists(path):
        return load_json(path)
    for f in glob.glob(os.path.join(demo_dir, "trace", "*_summary.json")):
        return load_json(f)
    return None


def collect_hotspots(demo_dir):
    for f in glob.glob(os.path.join(demo_dir, "nvbit", "bb_hot", "hotspots_*.json")):
        return load_json(f)
    return None


def collect_pcsampling(demo_dir):
    path = os.path.join(demo_dir, "cupti", "pcsampling_raw.json")
    if not os.path.exists(path):
        return None
    data = load_json(path)
    stall_table = {s["index"]: s["name"].replace("smsp__pcsamp_warps_issue_stalled_", "")
                   for s in data.get("stall_reason_table", [])}
    return {"stall_table": stall_table, "records": data.get("pc_records", []),
            "mode": data.get("collection_mode", ""), "period": data.get("sampling_period", 0)}


def collect_instrexec(demo_dir):
    path = os.path.join(demo_dir, "cupti", "instrexec_raw.json")
    if not os.path.exists(path):
        return None
    return load_json(path).get("records", [])


def collect_locality(demo_dir):
    for f in glob.glob(os.path.join(demo_dir, "nvbit", "all", "locality_analysis.json")):
        return load_json(f)
    return None


def collect_all_sass_profiles(demo_dir):
    profiles = {}
    for name in ["core", "divergence", "memory", "instruction_mix", "branch"]:
        path = os.path.join(demo_dir, "cupti", f"sassmetrics_{name}.json")
        if os.path.exists(path):
            data = load_json(path)
            totals = defaultdict(float)
            for rec in data.get("records", []):
                for m, v in rec.get("metrics", {}).items():
                    totals[m] += v
            profiles[name] = dict(totals)
    return profiles


def aggregate_sass_per_region(demo_dir, pc2region):
    """Aggregate per-PC SASS metrics to per-region using pc2region mapping.

    Returns: {region_id: {profile_name: {metric: value}}}
    Also returns coverage stats: {profile_name: (matched, total)}
    """
    if not pc2region:
        return {}, {}
    per_region = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    coverage = {}
    for name in ["core", "divergence", "memory", "instruction_mix", "branch"]:
        path = os.path.join(demo_dir, "cupti", f"sassmetrics_{name}.json")
        if not os.path.exists(path):
            continue
        data = load_json(path)
        recs = data.get("records", [])
        matched = 0
        for rec in recs:
            pc = rec.get("pcOffset", 0)
            rid = pc2region.get(pc)
            if rid is not None:
                matched += 1
                for m, v in rec.get("metrics", {}).items():
                    per_region[rid][name][m] += v
        coverage[name] = (matched, len(recs))
    # Convert defaultdicts to plain dicts
    result = {}
    for rid, profiles in per_region.items():
        result[int(rid)] = {pname: dict(metrics) for pname, metrics in profiles.items()}
    return result, coverage


def aggregate_instrexec_per_region(instrexec, source_line_regions):
    """Aggregate instrexec per-PC records to per-region using source line mapping.

    Returns: {region_id: {"threads_executed": N, "executed": N, "inst_count": N}}
    """
    if not instrexec or not source_line_regions:
        return {}
    per_region = defaultdict(lambda: {"threads_executed": 0, "executed": 0,
                                       "not_pred_off": 0, "inst_count": 0})
    for rec in instrexec:
        src_line = rec.get("source", {}).get("line")
        if src_line is None:
            continue
        rid = source_line_regions.get(int(src_line))
        if rid is not None:
            per_region[rid]["threads_executed"] += rec.get("threadsExecuted", 0)
            per_region[rid]["executed"] += rec.get("executed", 0)
            per_region[rid]["not_pred_off"] += rec.get("notPredOffThreadsExecuted", 0)
            per_region[rid]["inst_count"] += 1
    return dict(per_region)


def aggregate_pcsamp_per_region(pcsampling, pc2region):
    """Aggregate PC sampling stall reasons per-region.

    Returns: {region_id: {stall_name: count}}
    """
    if not pcsampling or not pc2region:
        return {}
    stall_table = pcsampling.get("stall_table", {})
    per_region = defaultdict(lambda: defaultdict(int))
    for rec in pcsampling.get("records", []):
        pc = rec.get("pcOffset", rec.get("pc_offset", 0))
        rid = pc2region.get(pc)
        if rid is None:
            continue
        for idx, count in rec.get("stall_reasons", {}).items():
            name = stall_table.get(int(idx), f"stall_{idx}")
            per_region[rid][name] += count
    return {int(rid): dict(stalls) for rid, stalls in per_region.items()}


def collect_mem_trace(demo_dir, max_records_per_region=512):
    """Load NVBit memory trace JSONL files and subsample for visualization.

    Returns: {region_id: {
        "records": [{pc, cta, warp, space, is_load, access_size, lane_lines: [32 ints], unique_lines, active_count}],
        "total_records": N,
        "per_pc": {pc_hex: {count, avg_unique_lines, load_pct, spaces}},
        "base_line": int  # minimum cache line ID for normalization
    }}
    """
    trace_files = sorted(glob.glob(os.path.join(demo_dir, "nvbit", "all", "mem_trace_*.jsonl")))
    if not trace_files:
        return None

    # Read all records grouped by region
    by_region = defaultdict(list)
    for tf in trace_files:
        with open(tf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_region[rec["region"]].append(rec)

    result = {}
    for rid, records in by_region.items():
        # Subsample if too many records (uniform sampling to preserve patterns)
        total = len(records)
        if total > max_records_per_region:
            step = total / max_records_per_region
            records = [records[int(i * step)] for i in range(max_records_per_region)]

        # Find global min cache line for relative addressing
        all_addrs = []
        for rec in records:
            all_addrs.extend(a for a in rec["addrs"] if a > 0)
        if not all_addrs:
            continue
        base_line = min(a // 128 for a in all_addrs)

        # Process records
        js_records = []
        per_pc = defaultdict(lambda: {"count": 0, "total_unique": 0, "loads": 0, "stores": 0, "spaces": set()})
        for rec in records:
            addrs = rec["addrs"]
            active = [a for a in addrs if a > 0]
            if not active:
                continue
            lane_lines = [a // 128 - base_line if a > 0 else -1 for a in addrs]
            unique_lines = len(set(l for l in lane_lines if l >= 0))
            pc_hex = f"0x{rec['pc_offset']:x}"

            js_records.append({
                "pc": pc_hex,
                "cta": rec["cta"],
                "warp": rec["warp"],
                "space": rec["space"],
                "ld": rec["is_load"],
                "sz": rec["access_size"],
                "lanes": lane_lines,
                "ul": unique_lines,
                "ac": len(active),
            })

            pp = per_pc[pc_hex]
            pp["count"] += 1
            pp["total_unique"] += unique_lines
            pp["loads"] += rec["is_load"]
            pp["stores"] += rec["is_store"]
            pp["spaces"].add(rec["space"])

        # Finalize per_pc
        js_per_pc = {}
        for pc, pp in per_pc.items():
            js_per_pc[pc] = {
                "count": pp["count"],
                "avg_ul": round(pp["total_unique"] / pp["count"], 2) if pp["count"] else 0,
                "load_pct": round(pp["loads"] / pp["count"] * 100, 1) if pp["count"] else 0,
                "spaces": sorted(pp["spaces"]),
            }

        result[int(rid)] = {
            "records": js_records,
            "total": total,
            "per_pc": js_per_pc,
            "base_line": 0,  # already normalized
        }

    return result if result else None


def _clean_sass(text):
    """Strip ELF metadata preamble, keep from first //## or instruction line."""
    lines = text.split('\n')
    out = []
    in_code = False
    for line in lines:
        if not in_code:
            if '//##' in line or re.search(r'/\*[0-9a-fA-F]+\*/', line):
                in_code = True
                out.append(line)
        else:
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return '\n'.join(out) if out else text


def _clean_ptx(text):
    """Strip metadata header and collapse runs of empty/comment-only lines."""
    lines = text.split('\n')
    out = []
    started = False
    blank_run = 0
    for line in lines:
        stripped = line.strip()
        if not started:
            if stripped.startswith('.version') or stripped.startswith('.target') or \
               stripped.startswith('.address_size') or stripped.startswith('.visible') or \
               stripped.startswith('.entry') or stripped.startswith('.func') or \
               stripped.startswith('.extern'):
                started = True
            elif stripped and not stripped.startswith('//'):
                started = True
            else:
                continue
        if not stripped or stripped == '//':
            blank_run += 1
            if blank_run <= 1:
                out.append(line)
        else:
            blank_run = 0
            out.append(line)
    return '\n'.join(out)


def collect_ptx(demo_dir):
    for f in glob.glob(os.path.join(demo_dir, "nvbit", "ptx", "ptx_all_*.ptx")):
        with open(f) as fh:
            return _clean_ptx(fh.read())
    return ""


def collect_sass_text(demo_dir, nvdisasm_path=""):
    if nvdisasm_path and os.path.exists(nvdisasm_path):
        with open(nvdisasm_path) as f:
            return _clean_sass(f.read())
    for f in glob.glob(os.path.join(demo_dir, "nvbit", "nvdisasm", "nvdisasm_*.sass")):
        with open(f) as fh:
            return _clean_sass(fh.read())
    for f in glob.glob(os.path.join(demo_dir, "nvbit", "pcmap", "sass_all_*.sass")):
        with open(f) as fh:
            return _clean_sass(fh.read())
    return ""


def _parse_source_line_regions(source_code):
    """Determine which profiling region each source line belongs to by parsing
    IKP_NVBIT_BEGIN/END and IKP_TRACE_REC_B/E markers in the source code.

    Returns dict: line_number -> region_id (int), only for lines inside markers.
    """
    # Step 1: Parse enum/constexpr/define to resolve symbolic names to region IDs
    name_to_id = {}
    for line in source_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        # Match: kFoo = 3, or kFoo = 3
        em = re.match(r'\s*(\w+)\s*=\s*(\d+)\s*[,;]?', stripped)
        if em:
            name_to_id[em.group(1)] = int(em.group(2))

    def resolve_id(token):
        """Convert 'kLoadA' or '2' to integer region ID."""
        token = token.strip()
        if token.isdigit():
            return int(token)
        return name_to_id.get(token)

    # Step 2: Scan for BEGIN/END markers, track region stack per line
    line_region = {}
    region_stack = []
    begin_re = re.compile(
        r'IKP_(?:NVBIT_BEGIN|TRACE_REC_B)\s*\((?:[^,]*,\s*[^,]*,\s*)?(\w+)\s*\)')
    end_re = re.compile(
        r'IKP_(?:NVBIT_END|TRACE_REC_E)\s*\((?:[^,]*,\s*[^,]*,\s*)?(\w+)\s*\)')

    for i, line in enumerate(source_code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//"):
            if region_stack:
                line_region[i] = region_stack[-1]
            continue

        # Process all BEGIN markers on this line (before assigning region)
        for bm in begin_re.finditer(stripped):
            rid = resolve_id(bm.group(1))
            if rid is not None:
                region_stack.append(rid)

        # Assign innermost region to this line
        if region_stack:
            line_region[i] = region_stack[-1]

        # Process all END markers on this line (after assigning region)
        for em in end_re.finditer(stripped):
            rid = resolve_id(em.group(1))
            if rid is not None and region_stack and region_stack[-1] == rid:
                region_stack.pop()

    return line_region


def aggregate_per_line(records, pc2region, source_line_regions=None):
    """Aggregate CUPTI per-PC records into per-source-line metrics.

    Region assignment uses source_line_regions (from marker parsing) when available,
    falling back to PC-offset-based pc2region join only if needed.
    """
    by_line = defaultdict(lambda: {"metrics": defaultdict(float), "regions": defaultdict(int),
                                    "pcs": [], "profiles": set()})
    for rec in records:
        src = rec.get("source")
        if not src or not src.get("line"):
            continue
        line = src["line"]
        info = by_line[line]
        info["profiles"].add(rec.get("_profile", ""))
        for m, v in rec.get("metrics", {}).items():
            info["metrics"][m] += v
        pc = rec.get("pcOffset")
        if pc is not None:
            info["pcs"].append(pc)
            # Use source-line-based region (from marker parsing) if available
            if source_line_regions and line in source_line_regions:
                region = source_line_regions[line]
                info["regions"][region] = info["regions"].get(region, 0) + 1
            elif not source_line_regions:
                # Fallback: PC-offset join (only when no source parsing available)
                region = pc2region.get(int(pc))
                if region is not None:
                    info["regions"][region] += 1
    for info in by_line.values():
        info["profiles"] = sorted(info["profiles"])
        info["regions"] = dict(info["regions"])
        info["metrics"] = dict(info["metrics"])
    return dict(by_line)


def build_sass_line_map(sass_text):
    line_map = {}
    cur = None
    for i, line in enumerate(sass_text.split('\n')):
        m = re.search(r'line (\d+)', line) if '//' in line else None
        if m:
            cur = int(m.group(1))
        if cur is not None:
            line_map.setdefault(cur, []).append(i + 1)
    return line_map


def build_ptx_line_map(ptx_text, source_path=""):
    """Build a PTX line map from .loc directives.

    PTX .loc format: .loc <file_id> <line> <column>
    We only map file_id=1 (the main .cu file) to avoid cross-file confusion.
    If multiple file IDs reference the source, we detect the right one.
    """
    line_map = {}
    cur_src_line = None
    # Determine which file_id corresponds to the main source
    # Typically file_id=1, but verify via .file directives if present
    main_file_id = 1
    for line in ptx_text.split('\n'):
        if source_path:
            bname = os.path.basename(source_path)
            if '.file' in line and bname in line:
                m = re.search(r'\.file\s+(\d+)', line)
                if m:
                    main_file_id = int(m.group(1))
                    break

    for i, line in enumerate(ptx_text.split('\n')):
        m = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)', line)
        if m:
            fid = int(m.group(1))
            sline = int(m.group(2))
            if fid == main_file_id and sline > 0:
                cur_src_line = sline
            else:
                cur_src_line = None
        if cur_src_line is not None:
            line_map.setdefault(cur_src_line, []).append(i + 1)
    return line_map


def build_nvdisasm_pc2src(sass_text, source_path):
    """Parse nvdisasm SASS to build {pc_offset_int: source_line_int} map.

    Scans for //## File "path", line N  followed by /*XXXX*/ instruction lines.
    Only maps to lines from source_path (the main .cu file).
    """
    source_basename = os.path.basename(source_path) if source_path else ""
    pc2src = {}
    current_src_line = None
    for line in sass_text.split('\n'):
        m = re.match(r'\s*//##\s+File\s+"([^"]+)",\s+line\s+(\d+)', line)
        if m:
            fpath = m.group(1)
            sline = int(m.group(2))
            if source_basename and os.path.basename(fpath) == source_basename:
                current_src_line = sline
            elif 'inlined at' in line:
                m2 = re.search(r'inlined at\s+"([^"]+)",\s+line\s+(\d+)', line)
                if m2 and source_basename and os.path.basename(m2.group(1)) == source_basename:
                    current_src_line = int(m2.group(2))
                else:
                    current_src_line = None
            else:
                current_src_line = None
            continue
        m = re.search(r'/\*([0-9a-fA-F]+)\*/', line)
        if m and current_src_line is not None:
            pc = int(m.group(1), 16)
            pc2src[pc] = current_src_line
    return pc2src


def dominant_region(regions):
    return max(regions, key=regions.get) if regions else None


# ── Metric definitions ────────────────────────────────────────────────

METRIC_DEFS = {
    # NVBit region stats - instructions
    "inst_total": {
        "short": "Total Instructions",
        "long": "Total warp-level instructions executed in this region (NVBit count). Each count = one warp executing one SASS instruction.",
        "unit": "instr"
    },
    "inst_pred_off": {
        "short": "Predicated Off",
        "long": "Instructions where all active threads had predication OFF (not executed). High values indicate compiler-generated predicated code with poor utilization.",
        "unit": "instr"
    },
    "bb_exec": {
        "short": "BB Executions",
        "long": "Total basic-block execution count across all warps. Indicates dynamic control flow activity.",
        "unit": "count"
    },
    # Global memory
    "gmem_load": {
        "short": "Global Loads",
        "long": "Number of global memory load operations (warp-level). Each operation may touch multiple cache lines.",
        "unit": "ops"
    },
    "gmem_store": {
        "short": "Global Stores",
        "long": "Number of global memory store operations (warp-level).",
        "unit": "ops"
    },
    "gmem_bytes": {
        "short": "Global Bytes",
        "long": "Total bytes actually transferred for global memory operations, including over-fetch due to misalignment or non-coalesced access.",
        "unit": "bytes"
    },
    "gmem_req_bytes": {
        "short": "Global Requested Bytes",
        "long": "Bytes actually needed by the program (requested). Compare with gmem_bytes to find wasted bandwidth from poor coalescing.",
        "unit": "bytes"
    },
    "gmem_inst_load_count": {
        "short": "Global Load Insts",
        "long": "Count of unique global load instructions (static). Different from gmem_load which counts dynamic executions.",
        "unit": "count"
    },
    "gmem_inst_store_count": {
        "short": "Global Store Insts",
        "long": "Count of unique global store instructions (static).",
        "unit": "count"
    },
    "gmem_sectors_32b": {
        "short": "Global Sectors",
        "long": "32-byte sectors accessed in global memory. Ideal coalescing: 4 sectors per 128B cache line. Higher sector count relative to ops = poor coalescing.",
        "unit": "sectors"
    },
    "gmem_unique_lines_est": {
        "short": "Unique Cache Lines",
        "long": "Estimated unique 128B cache lines touched by global memory operations. Indicates memory footprint.",
        "unit": "lines"
    },
    "gmem_sectors_per_inst_hist": {
        "short": "Sectors/Inst Distribution",
        "long": "Distribution of sectors accessed per global memory instruction (0-32). Bin 4 = ideal 128B coalesced access. Higher bins = worse coalescing.",
        "unit": "histogram"
    },
    "gmem_alignment_hist": {
        "short": "Alignment Distribution",
        "long": "Distribution of access alignment offsets (8 bins). Bin 0 = aligned. Non-zero bins indicate misaligned accesses causing extra sector fetches.",
        "unit": "histogram"
    },
    "gmem_stride_class_hist": {
        "short": "Stride Classification",
        "long": "Access pattern classification: [0]=Sequential (best), [1]=Strided (moderate), [2]=Random (worst for coalescing).",
        "unit": "histogram"
    },
    # Shared memory
    "smem_load": {
        "short": "Shared Loads",
        "long": "Number of shared memory load operations (warp-level).",
        "unit": "ops"
    },
    "smem_store": {
        "short": "Shared Stores",
        "long": "Number of shared memory store operations (warp-level).",
        "unit": "ops"
    },
    "smem_bytes": {
        "short": "Shared Bytes",
        "long": "Total bytes transferred for shared memory operations.",
        "unit": "bytes"
    },
    "smem_req_bytes": {
        "short": "Shared Requested Bytes",
        "long": "Bytes actually needed for shared memory operations.",
        "unit": "bytes"
    },
    "smem_inst_load_count": {
        "short": "Shared Load Insts",
        "long": "Count of unique shared memory load instructions (static).",
        "unit": "count"
    },
    "smem_inst_store_count": {
        "short": "Shared Store Insts",
        "long": "Count of unique shared memory store instructions (static).",
        "unit": "count"
    },
    "smem_bank_conflict_max_hist": {
        "short": "Bank Conflict Distribution",
        "long": "Distribution of max bank conflict ways (0-32) across shared memory ops. Bin 0 = no conflict, bin 1 = no conflict (1-way). Bins >1 indicate N-way bank conflicts causing serialization.",
        "unit": "histogram"
    },
    "smem_broadcast_count": {
        "short": "Broadcast Count",
        "long": "Number of shared memory broadcast operations where all threads read the same address (free, no conflict).",
        "unit": "count"
    },
    "smem_addr_span_hist": {
        "short": "Address Span Distribution",
        "long": "Distribution of address span (range of addresses accessed by a warp in a single shared memory instruction). Narrow spans indicate good locality.",
        "unit": "histogram"
    },
    # Local memory
    "lmem_load": {
        "short": "Local Loads",
        "long": "Number of local memory load operations. Local memory is backed by global memory (slow) and indicates register spills.",
        "unit": "ops"
    },
    "lmem_store": {
        "short": "Local Stores",
        "long": "Number of local memory store operations.",
        "unit": "ops"
    },
    "lmem_bytes": {
        "short": "Local Bytes",
        "long": "Total bytes transferred for local memory. High values indicate excessive register spilling.",
        "unit": "bytes"
    },
    "lmem_req_bytes": {
        "short": "Local Requested Bytes",
        "long": "Bytes actually needed for local memory operations.",
        "unit": "bytes"
    },
    "reg_spill_suspected": {
        "short": "Register Spill",
        "long": "Number of suspected register spill operations. Spills use local memory (backed by global DRAM) and significantly hurt performance.",
        "unit": "count"
    },
    "spill_ld_local_inst": {
        "short": "Spill Loads",
        "long": "Local memory load instructions attributed to register spills.",
        "unit": "instr"
    },
    "spill_st_local_inst": {
        "short": "Spill Stores",
        "long": "Local memory store instructions attributed to register spills.",
        "unit": "instr"
    },
    # Branch divergence
    "branch_div_hist": {
        "short": "Divergence Histogram",
        "long": "Distribution of active lane count at divergent branches (33 bins, 0-32). Shows how many threads are active when the warp diverges. Bin 32 = no divergence.",
        "unit": "histogram"
    },
    "branch_active_hist": {
        "short": "Active Lanes Histogram",
        "long": "Distribution of active lane count across all branch executions (33 bins, 0-32). Bin 32 = all threads active = full SIMT utilization.",
        "unit": "histogram"
    },
    "branch_active_avg_lanes": {
        "short": "Avg Active Lanes",
        "long": "Average number of active lanes across all branch executions. 32 = perfect utilization, lower values indicate thread divergence or early exit.",
        "unit": "lanes"
    },
    "branch_div_avg_active": {
        "short": "Divergent Avg Active",
        "long": "Average active lanes specifically at divergent branches. Lower than branch_active_avg_lanes indicates divergence concentrates in low-activity warps.",
        "unit": "lanes"
    },
    "branch_div_entropy": {
        "short": "Divergence Entropy",
        "long": "Shannon entropy of the divergence histogram. 0 = perfectly uniform execution (all divergent branches have same active count). Higher = more varied divergence patterns.",
        "unit": "bits"
    },
    # Instruction class
    "alu_fp32": {"short": "FP32 ALU", "long": "32-bit floating-point arithmetic instructions (FADD, FMUL, FFMA, etc.).", "unit": "instr"},
    "alu_int": {"short": "Integer ALU", "long": "Integer arithmetic instructions (IADD, IMAD, etc.).", "unit": "instr"},
    "tensor_wgmma": {"short": "Tensor/WGMMA", "long": "Tensor core (HMMA/WGMMA) instructions. Primary compute units for matrix multiplication.", "unit": "instr"},
    "ld_global": {"short": "Load Global", "long": "Global memory load instructions.", "unit": "instr"},
    "st_global": {"short": "Store Global", "long": "Global memory store instructions.", "unit": "instr"},
    "ld_shared": {"short": "Load Shared", "long": "Shared memory load instructions.", "unit": "instr"},
    "st_shared": {"short": "Store Shared", "long": "Shared memory store instructions.", "unit": "instr"},
    "ld_local": {"short": "Load Local", "long": "Local memory load instructions (often register spills).", "unit": "instr"},
    "st_local": {"short": "Store Local", "long": "Local memory store instructions (often register spills).", "unit": "instr"},
    "barrier": {"short": "Barrier", "long": "Barrier synchronization instructions (__syncthreads).", "unit": "instr"},
    "membar": {"short": "Memory Barrier", "long": "Memory fence/barrier instructions.", "unit": "instr"},
    "branch": {"short": "Branch", "long": "Branch instructions (conditional and unconditional).", "unit": "instr"},
    "call": {"short": "Call", "long": "Function call instructions.", "unit": "instr"},
    "ret": {"short": "Return", "long": "Function return instructions.", "unit": "instr"},
    "special": {"short": "Special", "long": "Special-purpose instructions (S2R, MUFU, etc.).", "unit": "instr"},
    "other": {"short": "Other", "long": "Other instructions not classified above.", "unit": "instr"},
    # Instruction pipeline
    "inst_pipe": {
        "short": "Instruction Pipeline",
        "long": "Distribution of instructions across hardware pipelines. Requires IKP_NVBIT_ENABLE_INST_PIPE=1.",
        "unit": "instr"
    },
    # Trace metrics
    "mean_dur": {"short": "Mean Duration", "long": "Average duration across all invocations of this region.", "unit": "ticks"},
    "cv_dur": {"short": "CV Duration", "long": "Coefficient of variation of region timing. CV > 0.2 suggests significant jitter. CV > 1.0 indicates extreme variability.", "unit": "ratio"},
    "min_dur": {"short": "Min Duration", "long": "Minimum observed duration.", "unit": "ticks"},
    "max_dur": {"short": "Max Duration", "long": "Maximum observed duration.", "unit": "ticks"},
    "var_dur_pop": {"short": "Population Variance", "long": "Population variance of duration.", "unit": "ticks^2"},
    "var_dur_sample": {"short": "Sample Variance", "long": "Sample variance of duration (Bessel-corrected).", "unit": "ticks^2"},
    # Derived metrics
    "predication_rate": {"short": "Predication Rate", "long": "Fraction of instructions with predication OFF. High rate = wasted issue slots.", "unit": "%"},
    "gmem_efficiency": {"short": "Global Mem Efficiency", "long": "Ratio of requested bytes to transferred bytes. 100% = perfect coalescing, <50% = severe waste.", "unit": "%"},
    "branch_uniformity": {"short": "Branch Uniformity", "long": "Fraction of branch executions with all 32 lanes active. 100% = no divergence.", "unit": "%"},
    "compute_intensity": {"short": "Compute Intensity", "long": "Ratio of compute instructions (FP32+tensor) to memory instructions. Higher = more compute-bound.", "unit": "ratio"},
    "bottleneck": {"short": "Bottleneck", "long": "Heuristic from NVBit instruction mix. compute_frac = (alu_fp32 + tensor_wgmma) / inst_total, memory_frac = (ld/st_global + ld/st_shared) / inst_total, branch_frac = branch / inst_total. If the largest fraction > 30% it is the bottleneck; otherwise 'balanced'.", "unit": "class"},
    # CUPTI / PC sampling
    "heat": {"short": "Hotness", "long": "Instruction hotness relative to the hottest source line. 100% = most-executed line.", "unit": "%"},
    "smsp__sass_inst_executed": {"short": "Instructions Executed", "long": "Total SASS instructions executed (CUPTI SASS metric). Counts warp-level instruction executions.", "unit": "instr"},
    "smsp__sass_thread_inst_executed": {"short": "Thread-Insts Executed", "long": "Total thread-level instruction executions. Divide by (inst_executed * 32) for thread utilization.", "unit": "thread-instr"},
    "smsp__sass_thread_inst_executed_pred_on": {"short": "Thread-Insts Pred On", "long": "Thread instructions where predicate was ON (actually executed).", "unit": "thread-instr"},
    # Stall reasons
    "no_instruction": {"short": "No Instruction", "long": "Warp stalled because no instruction was available. Often indicates instruction cache miss or branch resolution delay.", "unit": "samples"},
    "not_selected": {"short": "Not Selected", "long": "Warp was eligible but not selected by the scheduler. Indicates scheduler contention.", "unit": "samples"},
    "wait": {"short": "Wait", "long": "Warp stalled on a fixed-latency instruction dependency (e.g., ALU result not ready).", "unit": "samples"},
    "sleeping": {"short": "Sleeping", "long": "Warp is sleeping (nanosleep instruction). Intentional delay.", "unit": "samples"},
    "barrier": {"short": "Barrier Stall", "long": "Warp stalled at a __syncthreads or named barrier.", "unit": "samples"},
    "membar": {"short": "Memory Barrier Stall", "long": "Warp stalled at a memory fence instruction.", "unit": "samples"},
    "short_scoreboard": {"short": "Short Scoreboard", "long": "Warp waiting for short-latency operation (shared memory, L1 cache). Common for shared memory bank conflicts.", "unit": "samples"},
    "long_scoreboard": {"short": "Long Scoreboard", "long": "Warp waiting for long-latency operation (global memory, L2 cache, texture). Dominant stall for memory-bound kernels.", "unit": "samples"},
    "math_pipe_throttle": {"short": "Math Pipe Throttle", "long": "Math execution pipeline is full. Indicates compute saturation.", "unit": "samples"},
    "lg_throttle": {"short": "LG Throttle", "long": "Local/global memory pipeline is full. Indicates memory bandwidth saturation.", "unit": "samples"},
    "tex_throttle": {"short": "Texture Throttle", "long": "Texture pipeline is full.", "unit": "samples"},
    "mio_throttle": {"short": "MIO Throttle", "long": "Miscellaneous IO pipeline throttle.", "unit": "samples"},
    "drain": {"short": "Drain", "long": "Warp is draining (completing pending operations before exit).", "unit": "samples"},
    "branch_resolving": {"short": "Branch Resolving", "long": "Warp stalled while a branch target is being resolved.", "unit": "samples"},
    "dispatch_stall": {"short": "Dispatch Stall", "long": "Instruction dispatch stall.", "unit": "samples"},
    "imc_miss": {"short": "I-Cache Miss", "long": "Instruction cache miss. Kernel is too large to fit in instruction cache.", "unit": "samples"},
    "misc": {"short": "Miscellaneous", "long": "Other stall reasons not categorized.", "unit": "samples"},
    "selected": {"short": "Selected (Issued)", "long": "Warp was selected and issued an instruction this cycle.", "unit": "samples"},
    "warpgroup_arrive": {"short": "Warpgroup Arrive", "long": "Warp stalled at warpgroup arrive barrier (Hopper+ architecture).", "unit": "samples"},
    "mma": {"short": "MMA Stall", "long": "Warp stalled waiting for matrix multiply-accumulate result.", "unit": "samples"},
    # Locality
    "reuse_distance": {"short": "Reuse Distance", "long": "Stack distance between consecutive accesses to the same cache line. Shorter distance = better temporal locality = higher cache hit rate.", "unit": "histogram"},
    "working_set": {"short": "Working Set", "long": "Number of unique cache lines accessed within a sliding window. Indicates memory footprint at different time scales.", "unit": "lines"},
    "shared_line_ratio": {"short": "Shared Line Ratio", "long": "Fraction of cache lines accessed by more than one warp. Higher = more inter-warp data sharing.", "unit": "%"},
    "avg_warps_per_line": {"short": "Avg Warps/Line", "long": "Average number of warps accessing each cache line. Higher = more sharing and potential for L1 cache contention.", "unit": "warps"},
    # CUPTI SASS raw metrics (per-PC, used in efficiency calculations)
    "smsp__sass_sectors_mem_global": {"short": "Global Sectors", "long": "32-byte sectors accessed for global memory. More sectors per instruction = worse coalescing. Compare with _ideal variant.", "unit": "sectors"},
    "smsp__sass_sectors_mem_global_ideal": {"short": "Global Sectors (Ideal)", "long": "Minimum sectors needed if accesses were perfectly coalesced. Ratio ideal/actual = coalescing efficiency.", "unit": "sectors"},
    "smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared": {"short": "Shared Mem Wavefronts", "long": "LSU pipe wavefronts for shared memory. More wavefronts = more passes due to bank conflicts.", "unit": "wavefronts"},
    "smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal": {"short": "Shared Wavefronts (Ideal)", "long": "Minimum wavefronts if no bank conflicts. Ratio ideal/actual = shared memory efficiency.", "unit": "wavefronts"},
    "smsp__sass_branch_targets_threads_uniform": {"short": "Uniform Branches", "long": "Branch target evaluations where all threads in a warp took the same path (no divergence).", "unit": "count"},
    "smsp__sass_branch_targets_threads_divergent": {"short": "Divergent Branches", "long": "Branch target evaluations where threads in a warp took different paths (warp divergence).", "unit": "count"},
    "smsp__sass_inst_executed_op_branch": {"short": "Branch Insts", "long": "Branch instructions executed (warp-level). From CUPTI SASS instruction mix profile.", "unit": "instr"},
    "smsp__sass_inst_executed_op_global": {"short": "Global Mem Insts", "long": "Global memory instructions executed (loads + stores, warp-level).", "unit": "instr"},
    "smsp__sass_inst_executed_op_shared": {"short": "Shared Mem Insts", "long": "Shared memory instructions executed (warp-level).", "unit": "instr"},
    "smsp__sass_inst_executed_op_global_ld": {"short": "Global Load Insts", "long": "Global memory load instructions executed (warp-level).", "unit": "instr"},
    "smsp__sass_inst_executed_op_global_st": {"short": "Global Store Insts", "long": "Global memory store instructions executed (warp-level).", "unit": "instr"},
    "smsp__sass_inst_executed_op_shared_ld": {"short": "Shared Load Insts", "long": "Shared memory load instructions executed (warp-level).", "unit": "instr"},
    "smsp__sass_inst_executed_op_shared_st": {"short": "Shared Store Insts", "long": "Shared memory store instructions executed (warp-level).", "unit": "instr"},
    "smsp__sass_thread_inst_executed_op_branch": {"short": "Thread Branch Insts", "long": "Thread-level branch instruction executions.", "unit": "thread-instr"},
    # Derived CUPTI efficiency metrics
    "simt_utilization": {"short": "SIMT Utilization", "long": "Active threads per warp instruction / 32. Measures lane utilization across the warp. NOT SM occupancy. 100% = all 32 lanes active on every instruction (no divergence). Lower values indicate branch divergence or partial warps.", "unit": "%"},
    "predication_eff": {"short": "Predication Efficiency", "long": "Fraction of thread-instructions where predicate was ON (actually executed). Lower = more predicated-off work from compiler-generated conditional code.", "unit": "%"},
    "global_coalescing": {"short": "Global Mem Coalescing", "long": "Ratio of ideal sectors to actual sectors for global memory. 100% = perfectly coalesced. Lower values mean scattered memory accesses fetching unnecessary data.", "unit": "%"},
    "shared_efficiency": {"short": "Shared Mem Efficiency", "long": "Ratio of ideal wavefronts to actual wavefronts for shared memory. < 100% indicates bank conflicts requiring multiple passes.", "unit": "%"},
    # InstrExec fields
    "threads_executed": {"short": "Threads Executed", "long": "Total thread-level executions from CUPTI instruction execution profiling. Sum of active threads across all warp executions of instructions in this region.", "unit": "threads"},
    "executed": {"short": "Warp Executions", "long": "Total warp-level instruction executions from CUPTI. Each count = one warp executing one instruction.", "unit": "warps"},
    "inst_count": {"short": "Unique Instructions", "long": "Number of unique PC addresses (static instructions) observed in this region by CUPTI.", "unit": "count"},
    "notPredOffThreadsExecuted": {"short": "Not Pred-Off Threads", "long": "Thread executions where predicate was not OFF. Ratio to threadsExecuted = thread-level utilization.", "unit": "threads"},
    # Trace percentiles
    "percentile": {"short": "Percentile", "long": "Duration at which N% of region invocations complete. P50 = median. P95/P99 = tail latency. Large P99/P50 ratio indicates occasional slow invocations.", "unit": "ticks"},
    # Memory trace fields
    "mem_trace_records": {"short": "Memory Trace Records", "long": "Raw memory access records captured by NVBit. Each record = one warp memory instruction with 32 lane addresses.", "unit": "records"},
    "unique_cache_lines": {"short": "Unique Cache Lines", "long": "Distinct 128B cache lines accessed per warp instruction. 1 = perfect coalescing, >4 = poor coalescing for 4B accesses.", "unit": "lines"},
    "cache_line_span": {"short": "Cache Line Span", "long": "Distance between the first and last cache line touched by a single warp instruction. Larger span = more scattered access pattern.", "unit": "lines"},
    "coalescing_ratio": {"short": "Coalescing Ratio", "long": "Minimum possible cache lines / actual cache lines per instruction. 100% = optimal. Lower = wasted bandwidth from non-contiguous access.", "unit": "%"},
    # CUPTI per-region breakdown metrics (instruction mix)
    "global_mem_frac": {"short": "Global Mem Fraction", "long": "Fraction of warp instructions that are global memory ops (CUPTI). Higher = memory-intensive region.", "unit": "%"},
    "shared_mem_frac": {"short": "Shared Mem Fraction", "long": "Fraction of warp instructions that are shared memory ops (CUPTI).", "unit": "%"},
    "branch_frac": {"short": "Branch Fraction", "long": "Fraction of warp instructions that are branches (CUPTI). Higher = more control flow overhead.", "unit": "%"},
    "compute_frac": {"short": "Compute Fraction", "long": "Fraction of instructions that are pure compute (not memory or branch). Higher = compute-dominated region.", "unit": "%"},
    "tma_frac": {"short": "TMA Fraction", "long": "Fraction of instructions using Tensor Memory Accelerator (Hopper+). Present only in TMA-enabled kernels.", "unit": "%"},
    "tensor_frac": {"short": "Tensor/WGMMA Fraction", "long": "Fraction of instructions using Warp Group Matrix-Multiply-Accumulate (WGMMA/GMMA). Core compute for matrix operations on Hopper+.", "unit": "%"},
    "global_load_frac": {"short": "Global Load Ratio", "long": "Fraction of global memory ops that are loads (vs stores). Near 1.0 = read-dominated, near 0 = write-dominated.", "unit": "ratio"},
    "shared_load_frac": {"short": "Shared Load Ratio", "long": "Fraction of shared memory ops that are loads (vs stores).", "unit": "ratio"},
    "sectors_per_global_inst": {"short": "Sectors/Global Inst", "long": "Average 32B sectors per global memory instruction. Ideal coalescing for 4B per thread = 4 sectors/inst. Higher = poor coalescing.", "unit": "sectors"},
    "wavefronts_per_shared_inst": {"short": "Wavefronts/Shared Inst", "long": "Average L1TEX wavefronts per shared memory instruction. >1.0 indicates bank conflicts causing replay.", "unit": "wavefronts"},
    "branch_avg_active_lanes": {"short": "Branch Avg Active Lanes", "long": "Average active thread lanes during branch execution (CUPTI thread_inst/inst). 32 = no divergence.", "unit": "lanes"},
    "branch_lane_utilization": {"short": "Branch Lane Utilization", "long": "Fraction of possible lanes active during branches (avg_lanes/32). <1.0 = warp divergence at branches.", "unit": "%"},
    # Cross-validation
    "nvbit_vs_cupti": {"short": "NVBit vs CUPTI", "long": "Cross-validation of instruction counts between NVBit (binary instrumentation) and CUPTI (hardware counters). Delta < 5% = good agreement. Larger delta indicates pc2region coverage gaps.", "unit": ""},
}


# ── Build data JSON ───────────────────────────────────────────────────

REGION_COLORS = {
    0: "#656d76", 1: "#1a7f37", 2: "#0969da", 3: "#8250df",
    4: "#bc4c00", 5: "#9a6700", 6: "#cf222e", 7: "#6639ba",
}


def _compute_derived(stats):
    """Compute derived metrics from raw region stats."""
    derived = {}
    inst_total = stats.get("inst_total", 0)
    inst_pred_off = stats.get("inst_pred_off", 0)
    gmem_bytes = stats.get("gmem_bytes", 0)
    gmem_req = stats.get("gmem_req_bytes", 0)
    ic = stats.get("inst_class", {})

    derived["predication_rate"] = round(inst_pred_off / inst_total, 4) if inst_total > 0 else 0
    derived["gmem_efficiency"] = round(gmem_req / gmem_bytes, 4) if gmem_bytes > 0 else None

    bah = stats.get("branch_active_hist", [])
    total_branches = sum(bah)
    derived["branch_uniformity"] = round(bah[32] / total_branches, 4) if total_branches > 0 and len(bah) > 32 else None

    compute_insts = ic.get("alu_fp32", 0) + ic.get("tensor_wgmma", 0)
    mem_insts = (ic.get("ld_global", 0) + ic.get("st_global", 0) +
                 ic.get("ld_shared", 0) + ic.get("st_shared", 0))
    derived["compute_intensity"] = round(compute_insts / mem_insts, 4) if mem_insts > 0 else None

    # Bottleneck heuristic — pick the dominant category
    if inst_total > 0:
        compute_frac = compute_insts / inst_total
        mem_frac = mem_insts / inst_total
        branch_frac = ic.get("branch", 0) / inst_total
        fracs = {"compute": compute_frac, "memory": mem_frac, "branch": branch_frac}
        top = max(fracs, key=fracs.get)
        if fracs[top] > 0.3:
            derived["bottleneck"] = top
        else:
            derived["bottleneck"] = "balanced"
    else:
        derived["bottleneck"] = "balanced"

    return derived


def build_data(source_code, source_path, per_line, labels, region_stats,
               sass_text, sass_line_map, ptx_text, ptx_line_map, trace_summary, hotspots,
               pcsampling, instrexec, locality, sass_profiles, profiles,
               source_line_regions=None, pc2region=None, nvdisasm_pc2src=None,
               sass_per_region=None, sass_per_region_coverage=None,
               instrexec_per_region=None, pcsamp_per_region=None,
               mem_trace=None):
    max_inst = max((info["metrics"].get("smsp__sass_inst_executed", 0)
                    for info in per_line.values()), default=0)

    js_per_line = {}
    for ln, info in per_line.items():
        dr = dominant_region(info["regions"])
        ie = info["metrics"].get("smsp__sass_inst_executed", 0)
        js_per_line[int(ln)] = {
            "m": info["metrics"], "region": dr,
            "rlabel": labels.get(dr, f"region_{dr}") if dr is not None else None,
            "pcs": len(info["pcs"]), "profiles": info["profiles"],
            "heat": ie / max_inst if max_inst > 0 else 0,
        }

    # Pass ALL region stats fields + derived metrics
    js_regions = {}
    for rid, stats in region_stats.items():
        entry = {
            "label": labels.get(int(rid), f"region_{rid}"),
            **{k: v for k, v in stats.items() if k != "region"},
            "derived": _compute_derived(stats),
        }
        js_regions[int(rid)] = entry

    # Expanded trace data
    js_trace = None
    js_trace_meta = None
    if trace_summary:
        js_trace = []
        for r in trace_summary.get("regions", []):
            tr = {
                "name": r["name"], "count": r["count"],
                "mean": r["mean_dur"], "cv": r.get("cv_dur", 0),
                "min": r["min_dur"], "max": r["max_dur"],
                "percentiles": r.get("percentiles", {}),
                "var_dur_pop": r.get("var_dur_pop", 0),
                "var_dur_sample": r.get("var_dur_sample", 0),
                "hist": r.get("hist", {}).get("prob", []),
                "hist_min": r.get("hist", {}).get("min", 0),
                "hist_max": r.get("hist", {}).get("max", 0),
                "hist_bins": r.get("hist", {}).get("bins", 0),
            }
            js_trace.append(tr)

        # Trace metadata
        js_trace_meta = {
            "blocks": trace_summary.get("blocks", 0),
            "warps_per_block": trace_summary.get("warps_per_block", 0),
            "trace_file": trace_summary.get("trace", ""),
        }
        # by_block_warp: truncate for size
        bw = trace_summary.get("by_block_warp_regions", {})
        bw_trunc = {}
        for rname, rdata in list(bw.items())[:2]:
            entries = rdata.get("by_block_warp", [])[:64]
            bw_trunc[rname] = {
                "region": rdata.get("region"),
                "name": rdata.get("name", rname),
                "by_block_warp": entries,
            }
        js_trace_meta["by_block_warp"] = bw_trunc

    # Expanded hotspots — enrich with source line + region mapping
    js_hotspots = None
    if hotspots:
        pc2src = nvdisasm_pc2src or {}
        pc2r = pc2region or {}
        slr = source_line_regions or {}
        raw_bbs = sorted(hotspots.get("bb_entries", hotspots.get("basic_blocks", [])),
                         key=lambda b: -b.get("exec_count", 0))
        enriched_bbs = []
        for bb in raw_bbs[:30]:
            epc = bb.get("entry_pc", 0)
            src_line = pc2src.get(epc)
            region_id = pc2r.get(epc)
            # Try nearby PCs if exact match fails
            if src_line is None:
                for delta in [16, -16, 32, -32]:
                    if epc + delta in pc2src:
                        src_line = pc2src[epc + delta]
                        break
            if region_id is None:
                for delta in [16, -16, 32, -32]:
                    if epc + delta in pc2r:
                        region_id = pc2r[epc + delta]
                        break
            # Fallback: use source_line_regions if pc2region has no data
            if region_id is None and src_line is not None:
                region_id = slr.get(src_line)
            enriched = dict(bb)
            enriched["source_line"] = src_line
            enriched["region"] = region_id
            enriched["region_label"] = labels.get(region_id, f"region_{region_id}") if region_id is not None else None
            enriched["total_inst_exec"] = bb.get("exec_count", 0) * bb.get("n_instrs", 0)
            enriched_bbs.append(enriched)

        enriched_branches = []
        for br in hotspots.get("branch_sites", [])[:30]:
            pc = br.get("pc_offset", 0)
            src_line = pc2src.get(pc)
            region_id = pc2r.get(pc)
            if region_id is None and src_line is not None:
                region_id = slr.get(src_line)
            enriched_br = dict(br)
            enriched_br["source_line"] = src_line
            enriched_br["region"] = region_id
            enriched_br["region_label"] = labels.get(region_id) if region_id is not None else None
            enriched_branches.append(enriched_br)

        js_hotspots = {"bbs": enriched_bbs, "branches": enriched_branches}

    js_pcsamp = None
    if pcsampling and pcsampling["records"]:
        js_pcsamp = {"stalls": pcsampling["stall_table"],
                     "records": pcsampling["records"][:500],
                     "total": len(pcsampling["records"]),
                     "period": pcsampling["period"]}

    # Expanded locality
    js_locality = None
    if locality and locality.get("regions"):
        js_locality = {
            "line_bytes": locality.get("line_bytes", 128),
            "hist_bounds": locality.get("hist_bounds", []),
            "regions": {},
        }
        for rid, rd in locality["regions"].items():
            js_locality["regions"][rid] = {
                "records": rd.get("records", 0),
                "unique_lines": rd.get("unique_lines", 0),
                "lines_per_record": rd.get("lines_per_record", 0),
                "lines_per_1k_records": rd.get("lines_per_1k_records", 0),
                "reuse_distance": rd.get("reuse_distance", {}),
                "working_set": rd.get("working_set", {}),
                "sharing": rd.get("inter_warp_sharing", {}),
                "cta_sharing": rd.get("inter_cta_sharing", {}),
            }

    # Detect available NVBit modes
    nvbit_modes = []
    for mode in ["pcmap", "all", "bb_hot", "inst_pipe", "mem_pattern"]:
        mode_dir = os.path.join(os.path.dirname(source_path) if source_path else ".",
                                "_demo_out", "nvbit", mode)
        # Fallback: check via region_stats keys
    if any(s.get("inst_pipe") is not None for s in region_stats.values()):
        nvbit_modes.append("inst_pipe")
    if hotspots:
        nvbit_modes.append("bb_hot")
    if locality:
        nvbit_modes.append("all")
    if region_stats:
        nvbit_modes.append("pcmap")

    cupti_profiles_found = []
    for name in ["core", "divergence", "memory", "instruction_mix", "branch"]:
        if sass_profiles and name in sass_profiles:
            cupti_profiles_found.append(name)

    data_quality = {
        "nvbit_modes": nvbit_modes,
        "cupti_profiles": cupti_profiles_found,
        "has_trace": trace_summary is not None,
        "has_pcsamp": pcsampling is not None and len(pcsampling.get("records", [])) > 0,
        "pcsamp_total": len(pcsampling["records"]) if pcsampling else 0,
        "has_instrexec": instrexec is not None and len(instrexec) > 0,
        "instrexec_total": len(instrexec) if instrexec else 0,
        "has_locality": locality is not None,
        "has_source_mapping": len(per_line) > 0,
    }

    # Compute per-region CUPTI efficiency + breakdown metrics (derived from sass_per_region)
    cupti_per_region = {}
    if sass_per_region:
        for rid, profiles_data in sass_per_region.items():
            eff = {}
            breakdown = {}
            # SIMT Utilization from divergence profile (active threads / warp_size, NOT SM occupancy)
            div = profiles_data.get("divergence", {})
            ie = div.get("smsp__sass_inst_executed", 0)
            tie = div.get("smsp__sass_thread_inst_executed", 0)
            tpon = div.get("smsp__sass_thread_inst_executed_pred_on", 0)
            if ie > 0 and tie > 0:
                eff["simt_utilization"] = tie / (ie * 32)
            if tie > 0 and tpon > 0:
                eff["predication_eff"] = tpon / tie

            # Memory efficiency from memory profile
            mem = profiles_data.get("memory", {})
            gs = mem.get("smsp__sass_sectors_mem_global", 0)
            gsi = mem.get("smsp__sass_sectors_mem_global_ideal", 0)
            if gs > 0 and gsi > 0:
                eff["global_coalescing"] = gsi / gs
            sw = mem.get("smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared", 0)
            swi = mem.get("smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal", 0)
            if sw > 0 and swi > 0:
                eff["shared_efficiency"] = swi / sw

            # Branch uniformity from branch profile
            br = profiles_data.get("branch", {})
            bu = br.get("smsp__sass_branch_targets_threads_uniform", 0)
            bd = br.get("smsp__sass_branch_targets_threads_divergent", 0)
            if bu + bd > 0:
                eff["branch_uniformity"] = bu / (bu + bd)

            # ── Additional derived metrics from instruction_mix ──
            imix = profiles_data.get("instruction_mix", {})
            imix_ie = imix.get("smsp__sass_inst_executed", ie) or ie
            op_global = imix.get("smsp__sass_inst_executed_op_global", 0)
            op_shared = imix.get("smsp__sass_inst_executed_op_shared", 0)
            op_branch = imix.get("smsp__sass_inst_executed_op_branch", 0)
            op_tma = imix.get("smsp__sass_inst_executed_op_tma", 0)
            op_gmma = imix.get("smsp__sass_inst_executed_op_shared_gmma", 0)
            if imix_ie > 0:
                # Instruction mix fractions (CUPTI-measured)
                breakdown["global_mem_frac"] = op_global / imix_ie
                breakdown["shared_mem_frac"] = op_shared / imix_ie
                breakdown["branch_frac"] = op_branch / imix_ie
                breakdown["compute_frac"] = max(0, 1.0 - (op_global + op_shared + op_branch) / imix_ie)
                if op_tma > 0:
                    breakdown["tma_frac"] = op_tma / imix_ie
                if op_gmma > 0:
                    breakdown["tensor_frac"] = op_gmma / imix_ie

            # ── Memory detail from memory profile ──
            gld = mem.get("smsp__sass_inst_executed_op_global_ld", 0)
            gst = mem.get("smsp__sass_inst_executed_op_global_st", 0)
            sld = mem.get("smsp__sass_inst_executed_op_shared_ld", 0)
            sst = mem.get("smsp__sass_inst_executed_op_shared_st", 0)
            if gld + gst > 0:
                breakdown["global_load_frac"] = gld / (gld + gst)
                # Sectors per global instruction (ideal coalescing = 4 for 128B lines)
                if gs > 0:
                    breakdown["sectors_per_global_inst"] = gs / (gld + gst)
                    breakdown["sectors_per_global_inst_ideal"] = gsi / (gld + gst) if gsi > 0 else 0
            if sld + sst > 0:
                breakdown["shared_load_frac"] = sld / (sld + sst)
                if sw > 0:
                    breakdown["wavefronts_per_shared_inst"] = sw / (sld + sst)

            # ── Branch thread activity from branch profile ──
            br_inst = br.get("smsp__sass_inst_executed_op_branch", 0)
            br_thread = br.get("smsp__sass_thread_inst_executed_op_branch", 0)
            if br_inst > 0 and br_thread > 0:
                breakdown["branch_avg_active_lanes"] = br_thread / br_inst
                breakdown["branch_lane_utilization"] = br_thread / (br_inst * 32)

            # ── CUPTI instruction count (for cross-validation with NVBit) ──
            breakdown["cupti_inst_executed"] = ie

            cupti_per_region[int(rid)] = {
                "raw": profiles_data,
                "efficiency": eff,
                "breakdown": breakdown,
            }

    # Cross-validate NVBit vs CUPTI instruction mix fractions per region
    # Raw counts aren't comparable (CUPTI may aggregate across invocations differently).
    # Instead, compare instruction mix fractions which are ratio-based.
    cross_validation = {}
    for rid in set(list(cupti_per_region.keys()) + list(region_stats.keys())):
        rid_int = int(rid)
        nvbit_stats = region_stats.get(rid, region_stats.get(str(rid), {}))
        cupti_data = cupti_per_region.get(rid_int, {})
        bk = cupti_data.get("breakdown", {})
        ic = nvbit_stats.get("inst_class", {})
        it = nvbit_stats.get("inst_total", 0)

        if it > 0 and bk:
            cv = {}
            # NVBit fractions
            nvbit_global = ((ic.get("ld_global", 0) + ic.get("st_global", 0)) / it) if it > 0 else 0
            nvbit_shared = ((ic.get("ld_shared", 0) + ic.get("st_shared", 0)) / it) if it > 0 else 0
            nvbit_branch = (ic.get("branch", 0) / it) if it > 0 else 0
            nvbit_compute = max(0, 1.0 - nvbit_global - nvbit_shared - nvbit_branch)
            # CUPTI fractions
            cupti_global = bk.get("global_mem_frac", 0)
            cupti_shared = bk.get("shared_mem_frac", 0)
            cupti_branch = bk.get("branch_frac", 0)
            cupti_compute = bk.get("compute_frac", 0)

            cv["fractions"] = {
                "compute": {"nvbit": nvbit_compute, "cupti": cupti_compute,
                            "delta": abs(nvbit_compute - cupti_compute)},
                "global_mem": {"nvbit": nvbit_global, "cupti": cupti_global,
                               "delta": abs(nvbit_global - cupti_global)},
                "shared_mem": {"nvbit": nvbit_shared, "cupti": cupti_shared,
                               "delta": abs(nvbit_shared - cupti_shared)},
                "branch": {"nvbit": nvbit_branch, "cupti": cupti_branch,
                           "delta": abs(nvbit_branch - cupti_branch)},
            }
            cv["avg_delta"] = sum(f["delta"] for f in cv["fractions"].values()) / 4
            cv["nvbit_inst"] = it
            cv["cupti_inst"] = bk.get("cupti_inst_executed", 0)
            cross_validation[rid_int] = cv

    return {
        "source": {"code": source_code, "path": os.path.basename(source_path)},
        "sass": {"text": sass_text, "lineMap": sass_line_map},
        "ptx": {"text": ptx_text, "lineMap": ptx_line_map},
        "perLine": js_per_line,
        "regions": js_regions,
        "labels": {str(k): v for k, v in labels.items()},
        "colors": REGION_COLORS,
        "trace": js_trace,
        "traceMeta": js_trace_meta,
        "hotspots": js_hotspots,
        "pcsamp": js_pcsamp,
        "instrexec": instrexec[:500] if instrexec else None,
        "locality": js_locality,
        "sassProfiles": sass_profiles or {},
        "cuptiPerRegion": cupti_per_region,
        "cuptiCoverage": sass_per_region_coverage or {},
        "instrexecPerRegion": instrexec_per_region or {},
        "pcsampPerRegion": pcsamp_per_region or {},
        "crossValidation": cross_validation,
        "profiles": profiles,
        "defs": METRIC_DEFS,
        "dataQuality": data_quality,
        "lineRegions": {str(k): v for k, v in (source_line_regions or {}).items()},
        "memTrace": mem_trace,
        "pc2src": {str(k): v for k, v in (nvdisasm_pc2src or {}).items()},
    }


# ── HTML template ─────────────────────────────────────────────────────

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IKP Explorer</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs/editor/editor.main.css">
<style>
:root {
  --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de;
  --text:#1f2328; --dim:#656d76; --bright:#1f2328;
  --accent:#0969da; --green:#1a7f37; --orange:#bc4c00;
  --purple:#8250df; --yellow:#9a6700; --red:#cf222e;
}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);
  font-family:'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace;font-size:13px;}

/* Header */
.hdr{background:var(--bg2);border-bottom:1px solid var(--border);padding:6px 16px;
  display:flex;align-items:center;gap:12px;flex-shrink:0;height:38px;}
.hdr h1{font-size:14px;color:var(--accent);font-weight:700;white-space:nowrap;}
.hdr .fname{color:var(--bright);font-size:13px;}
.hdr .sep{color:var(--dim);}
.hdr .info{color:var(--dim);font-size:11px;margin-left:auto;display:flex;gap:14px;}
.hdr .info b{color:var(--accent);font-weight:600;}

/* Main area */
.main{display:flex;height:calc(100vh - 38px);}

/* Panels */
.panel{display:flex;flex-direction:column;overflow:hidden;min-width:80px;}
.panel-hdr{background:var(--bg2);border-bottom:1px solid var(--border);padding:4px 10px;
  font-size:11px;color:var(--dim);font-weight:600;text-transform:uppercase;letter-spacing:.5px;
  display:flex;align-items:center;gap:8px;flex-shrink:0;height:30px;}
.panel-hdr .active{color:var(--bright);border-bottom:2px solid var(--accent);}
.asm-tab{cursor:pointer;padding:2px 6px;}
.asm-tab:hover{color:var(--text);}
.asm-tab.active{color:var(--bright);}
.asm-status{margin-left:auto;font-weight:400;text-transform:none;letter-spacing:0;font-size:10px;}
.editor-wrap{flex:1;overflow:hidden;}

/* Gutter split */
.gutter{background:var(--border);cursor:col-resize;position:relative;
  transition:background .15s;flex-shrink:0;}
.gutter-vertical{cursor:row-resize !important;}
.gutter:hover{background:var(--accent);}
.gutter::after{content:'';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:2px;height:30px;background:var(--dim);border-radius:2px;transition:background .15s;}
.gutter-vertical::after{width:30px !important;height:2px !important;}
.gutter:hover::after{background:var(--bright);}
.split{overflow:hidden;}

/* Right panel: metrics */
.metrics{display:flex;flex-direction:column;overflow:hidden;}
.tabs{display:flex;background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0;overflow-x:auto;}
.tab{padding:5px 9px;font-size:11px;color:var(--dim);cursor:pointer;border-bottom:2px solid transparent;
  font-weight:500;white-space:nowrap;}
.tab:hover{color:var(--text);}
.tab.active{color:var(--bright);border-bottom-color:var(--accent);}
.mscroll{flex:1;overflow-y:auto;padding:10px;}
.tc{display:none;}
.tc.active{display:block;}

/* Metric components */
.msec{margin-bottom:14px;}
.msec h4{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;
  margin-bottom:5px;padding-bottom:3px;border-bottom:1px solid var(--border);}
.mrow{display:flex;justify-content:space-between;padding:2px 0;font-size:12px;}
.mrow .mn{color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60%;}
.mrow .mv{color:var(--accent);font-weight:600;font-variant-numeric:tabular-nums;}
.mrow.hl .mn{color:var(--text);} .mrow.hl .mv{color:var(--bright);font-size:13px;}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px;}
.cards-3{grid-template-columns:1fr 1fr 1fr;}
.cards-4{grid-template-columns:1fr 1fr 1fr 1fr;}
.card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px;text-align:center;}
.card .cv{font-size:20px;font-weight:700;color:var(--accent);}
.card .cl{font-size:9px;color:var(--dim);text-transform:uppercase;margin-top:2px;}
.bar-row{display:flex;align-items:center;gap:5px;margin-bottom:3px;font-size:11px;}
.bar-label{width:80px;text-align:right;color:var(--dim);overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;flex-shrink:0;}
.bar-track{flex:1;height:14px;background:var(--bg);border-radius:3px;overflow:hidden;}
.bar-fill{height:100%;border-radius:3px;transition:width .2s;}
.bar-val{width:50px;text-align:right;color:var(--accent);font-weight:600;flex-shrink:0;
  font-variant-numeric:tabular-nums;}
.empty{color:var(--dim);font-size:12px;text-align:center;padding:30px 12px;}

/* Region items */
.ri{display:flex;align-items:center;gap:6px;padding:5px 6px;border-radius:4px;cursor:pointer;margin-bottom:2px;}
.ri:hover{background:var(--bg3);}
.ri.active{background:rgba(88,166,255,.12);}
.ri-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.ri-name{flex:1;font-size:12px;}
.ri-val{font-size:11px;color:var(--dim);}

/* ECharts containers */
.chart{width:100%;margin:6px 0;}

/* Info icons — CSS-rendered "i" circle, no Unicode dependency */
.info-icon{cursor:pointer;display:inline-flex;align-items:center;justify-content:center;
  width:13px;height:13px;border-radius:50%;border:1.5px solid var(--dim);
  font-size:9px;font-weight:700;font-style:italic;font-family:Georgia,serif;
  color:var(--dim);margin-left:3px;opacity:0.7;vertical-align:middle;
  line-height:1;user-select:none;position:relative;flex-shrink:0;}
.info-icon:hover{opacity:1;color:var(--accent);border-color:var(--accent);}
/* Tooltip popup on click */
.info-tip{display:none;position:fixed;background:var(--bright);color:#fff;font-size:11px;
  font-style:normal;font-weight:400;font-family:inherit;padding:8px 10px;border-radius:6px;
  max-width:300px;min-width:200px;z-index:999999;line-height:1.5;
  box-shadow:0 4px 12px rgba(0,0,0,.2);pointer-events:auto;text-align:left;}
.info-icon.show .info-tip{display:block;}

/* Bottleneck badges */
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.3px;}
.badge-compute{background:#dafbe1;color:#1a7f37;}
.badge-memory{background:#ddf4ff;color:#0969da;}
.badge-branch{background:#fff8c5;color:#9a6700;}
.badge-balanced{background:#eaeef2;color:#656d76;}
.badge-warn{background:#fff1e5;color:#bc4c00;}
.badge-ok{background:#dafbe1;color:#1a7f37;}
.badge-miss{background:#ffebe9;color:#cf222e;}
.badge-present{background:#dafbe1;color:#1a7f37;}

/* Section collapse */
.sec-hdr{cursor:pointer;user-select:none;display:flex;align-items:center;gap:4px;}
.sec-hdr:hover h4{color:var(--accent);}
.sec-hdr .arrow{font-size:8px;transition:transform .15s;}
.sec-hdr.collapsed .arrow{transform:rotate(-90deg);}
.sec-body{overflow:hidden;transition:max-height .2s;}
.sec-body.collapsed{max-height:0 !important;padding:0;margin:0;}

/* Percentile table */
.ptable{width:100%;font-size:10px;border-collapse:collapse;margin:6px 0;}
.ptable th,.ptable td{padding:3px 5px;text-align:right;border-bottom:1px solid var(--border);}
.ptable th{color:var(--dim);font-weight:600;text-align:right;}
.ptable td{font-variant-numeric:tabular-nums;}
.ptable .warn{background:#fff8c5;}

/* Efficiency bar */
.eff-bar{height:20px;border-radius:4px;position:relative;overflow:hidden;margin:4px 0;}
.eff-fill{height:100%;border-radius:4px;transition:width .3s;}
.eff-label{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-size:11px;font-weight:600;color:var(--bright);}

/* Monaco line decorations */
.line-hot{background:rgba(207,34,46,.10) !important;}
.line-warm{background:rgba(154,103,0,.08) !important;}
.line-cold{background:rgba(9,105,218,.06) !important;}
.line-selected{background:rgba(9,105,218,.14) !important;}
.line-flash{animation:region-flash .8s ease-out !important;}
@keyframes region-flash{0%{background:rgba(9,105,218,.30)}100%{background:transparent}}
.asm-highlight{background:rgba(88,166,255,.15) !important;}
.region-bar{width:3px !important;margin-left:2px !important;}
.region-bar-0{background:#656d76 !important;}
.region-bar-1{background:#1a7f37 !important;}
.region-bar-2{background:#0969da !important;}
.region-bar-3{background:#8250df !important;}
.region-bar-4{background:#bc4c00 !important;}
.region-bar-5{background:#9a6700 !important;}
.region-bar-6{background:#cf222e !important;}
.region-bar-7{background:#6639ba !important;}
/* Region label in glyph margin */
.region-glyph{font-size:8px !important;line-height:20px !important;text-align:center;
  display:flex !important;align-items:center;justify-content:center;opacity:0.7;
  font-weight:600;letter-spacing:-0.3px;color:var(--dim) !important;}
/* Source panel legend */
.src-legend{display:flex;gap:10px;padding:2px 8px;font-size:9px;color:var(--dim);
  border-top:1px solid var(--border);flex-wrap:wrap;align-items:center;flex-shrink:0;}
.src-legend .leg-item{display:flex;align-items:center;gap:3px;cursor:pointer;}
.src-legend .leg-item:hover{color:var(--accent);}
.src-legend .leg-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.src-legend .leg-heat{display:inline-block;width:18px;height:8px;border-radius:2px;}
/* Load data button */
.load-btn{background:var(--bg3);border:1px solid var(--border);border-radius:4px;
  padding:2px 8px;font-size:10px;cursor:pointer;color:var(--dim);white-space:nowrap;}
.load-btn:hover{color:var(--accent);border-color:var(--accent);}

/* Scrollbar */
::-webkit-scrollbar{width:8px;height:8px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}
</style>
</head>
<body>

<div class="hdr">
  <h1>IKP Explorer</h1>
  <span class="sep">/</span>
  <span class="fname" id="fileName"></span>
  <div class="info" id="headerInfo"></div>
  <button class="load-btn" id="loadBtn" title="Load a different profiler dataset">Load Data...</button>
  <input type="file" id="loadFile" accept=".html,.json" style="display:none">
</div>

<div class="main" id="main">
  <!-- Source panel -->
  <div class="panel" id="srcPanel">
    <div class="panel-hdr"><span>CUDA Source</span></div>
    <div class="editor-wrap" id="srcWrap"></div>
    <div class="src-legend" id="srcLegend"></div>
  </div>

  <!-- Assembly panel: PTX + SASS stacked -->
  <div class="panel" id="asmPanel">
    <div id="ptxPane" style="display:flex;flex-direction:column;overflow:hidden">
      <div class="panel-hdr" style="display:flex;align-items:center;gap:6px">
        <span>PTX</span>
        <span class="asm-status" id="ptxStatus" style="flex:1"></span>
      </div>
      <div class="editor-wrap" id="ptxWrap"></div>
    </div>
    <div id="sassPane" style="display:flex;flex-direction:column;overflow:hidden">
      <div class="panel-hdr" style="display:flex;align-items:center;gap:6px">
        <span>SASS</span>
        <span class="asm-status" id="sassStatus" style="flex:1"></span>
      </div>
      <div class="editor-wrap" id="sassWrap"></div>
    </div>
  </div>

  <!-- Metrics panel -->
  <div class="panel metrics" id="metPanel">
    <div class="tabs" id="tabBar">
      <span class="tab active" data-t="ov">Overview</span>
      <span class="tab" data-t="line">Line</span>
      <span class="tab" data-t="region">Regions</span>
      <span class="tab" data-t="exec">Execution</span>
      <span class="tab" data-t="mem">Memory</span>
      <span class="tab" data-t="stalls">Stalls</span>
      <span class="tab" data-t="trace">Trace</span>
    </div>
    <div class="mscroll">
      <div class="tc active" id="tc-ov"><div id="ovCt"></div></div>
      <div class="tc" id="tc-line"><div id="lineMet"><div class="empty">Click a source line to see metrics</div></div></div>
      <div class="tc" id="tc-region"><div id="regionList"></div><div id="regionDet"></div></div>
      <div class="tc" id="tc-exec"><div id="execCt"></div></div>
      <div class="tc" id="tc-mem"><div id="memCt"></div></div>
      <div class="tc" id="tc-stalls"><div id="stallsCt"></div></div>
      <div class="tc" id="tc-trace"><div id="traceCt"></div></div>
    </div>
  </div>
</div>

<div id="loading" style="position:fixed;inset:0;background:var(--bg);display:flex;align-items:center;
  justify-content:center;z-index:999;color:var(--dim);font-size:14px;">
  Loading Monaco Editor from CDN...
</div>

<!-- CDN -->
<script src="https://cdn.jsdelivr.net/npm/split.js@1.6.5/dist/split.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs/loader.min.js"></script>

<script>
// ── Data ──
let D = __DATA__;

// ── Load Data (switch kernel) ──
document.getElementById('loadBtn').addEventListener('click', () => document.getElementById('loadFile').click());
document.getElementById('loadFile').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {
    try {
      const text = ev.target.result;
      if (file.name.endsWith('.html')) {
        // Open another explorer HTML in a new tab
        const blob = new Blob([text], {type: 'text/html'});
        const url = URL.createObjectURL(blob);
        window.open(url, '_blank');
      } else if (file.name.endsWith('.json')) {
        // Standalone JSON data — wrap into a new explorer HTML
        const newData = JSON.parse(text);
        if (!newData.source) throw new Error('JSON must contain "source" key (use generate_explorer.py output)');
        const newHtml = document.documentElement.outerHTML.replace(
          /let D = .*?;\n/s,
          'let D = ' + JSON.stringify(newData) + ';\n'
        );
        const blob = new Blob([newHtml], {type: 'text/html'});
        window.open(URL.createObjectURL(blob), '_blank');
      } else {
        throw new Error('Unsupported file type. Use .html (explorer) or .json (data).');
      }
    } catch(err) {
      alert('Load failed: ' + err.message +
        '\n\nTo switch kernels, regenerate with:\n' +
        '  python3 scripts/generate_explorer.py \\\n' +
        '    --demo-dir <output_dir> \\\n' +
        '    --source <kernel.cu> \\\n' +
        '    --output explorer.html');
    }
  };
  reader.readAsText(file);
});

// ── Helpers ──
function fmt(v) {
  if (v===0||v==null) return '0';
  if (typeof v==='number'&&!Number.isInteger(v)){
    if(Math.abs(v)>=1e9)return(v/1e9).toFixed(2)+'G';
    if(Math.abs(v)>=1e6)return(v/1e6).toFixed(2)+'M';
    if(Math.abs(v)>=1e3)return(v/1e3).toFixed(2)+'K';
    return v.toFixed(3);
  }
  v=Math.round(v);
  if(Math.abs(v)>=1e9)return(v/1e9).toFixed(2)+'G';
  if(Math.abs(v)>=1e6)return(v/1e6).toFixed(2)+'M';
  if(Math.abs(v)>=1e3)return(v/1e3).toFixed(1)+'K';
  return v.toString();
}
function fmtBytes(b) {
  if (b==null||b===0) return '0 B';
  if (b>=1073741824) return (b/1073741824).toFixed(2)+' GB';
  if (b>=1048576) return (b/1048576).toFixed(2)+' MB';
  if (b>=1024) return (b/1024).toFixed(1)+' KB';
  return b+' B';
}
function fmtPct(v) { return v==null?'N/A':(v*100).toFixed(1)+'%'; }
function infoIcon(key) {
  const d = D.defs && D.defs[key];
  if (!d) return '';
  const tip = (d.long||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const unit = d.unit && d.unit !== 'histogram' && d.unit !== 'class' ? ' <b>['+d.unit+']</b>' : '';
  return '<span class="info-icon" data-def="'+key+'" onclick="event.stopPropagation();showInfoTip(this)">i</span>';
}
const _tipEl = document.createElement('div');
_tipEl.className = 'info-tip';
_tipEl.style.display = 'none';
document.body.appendChild(_tipEl);
let _tipOwner = null;
function showInfoTip(el) {
  if (_tipOwner === el) { _tipEl.style.display = 'none'; _tipOwner = null; return; }
  const key = el.getAttribute('data-def');
  const d = D.defs && D.defs[key];
  if (!d) return;
  const unit = d.unit && d.unit !== 'histogram' && d.unit !== 'class' ? ' <b>['+d.unit+']</b>' : '';
  _tipEl.innerHTML = '<b>'+d.short+'</b><br>'+d.long+unit;
  _tipEl.style.display = 'block';
  const r = el.getBoundingClientRect();
  let top = r.top - _tipEl.offsetHeight - 6;
  let left = r.left + r.width/2 - _tipEl.offsetWidth/2;
  if (top < 4) top = r.bottom + 6;
  if (left < 4) left = 4;
  if (left + _tipEl.offsetWidth > window.innerWidth - 4) left = window.innerWidth - _tipEl.offsetWidth - 4;
  _tipEl.style.top = top + 'px';
  _tipEl.style.left = left + 'px';
  _tipOwner = el;
}
document.addEventListener('click', function(e) {
  if (!e.target.closest('.info-icon')) { _tipEl.style.display = 'none'; _tipOwner = null; }
});
function barRow(label,value,maxVal,color,defKey){
  const pct=maxVal>0?(value/maxVal*100):0;
  const tip=defKey&&D.defs&&D.defs[defKey]?' title="'+D.defs[defKey].long.replace(/"/g,'&quot;')+'"':'';
  const d=D.defs&&D.defs[defKey];
  const unit=d&&d.unit&&d.unit!=='histogram'&&d.unit!=='class'?' <span style="font-size:9px;color:var(--dim)">'+d.unit+'</span>':'';
  return '<div class="bar-row"><span class="bar-label"'+tip+'>'+label+'</span>'+
    '<div class="bar-track"><div class="bar-fill" style="width:'+pct+'%;background:'+(color||'var(--accent)')+'"></div></div>'+
    '<span class="bar-val">'+fmt(value)+unit+'</span></div>';
}
// Card helper: value, label, defKey for info icon + unit
function card(val, label, defKey, opts) {
  opts = opts || {};
  const d = D.defs && D.defs[defKey];
  const unit = d && d.unit && d.unit!=='histogram'&&d.unit!=='class' ? d.unit : '';
  const isByte = defKey && (defKey.includes('bytes') || defKey === 'lmem_bytes' || defKey === 'smem_bytes' || defKey === 'gmem_bytes' || defKey === 'gmem_req_bytes' || defKey === 'smem_req_bytes' || defKey === 'lmem_req_bytes');
  const fmtVal = opts.raw ? val : (isByte ? fmtBytes(val) : fmt(val));
  const unitHtml = (!isByte && unit && !opts.noUnit && !String(fmtVal).includes('%')) ?
    '<div style="font-size:8px;color:var(--dim);margin-top:1px">'+unit+'</div>' : '';
  const style = opts.style ? ' style="'+opts.style+'"' : '';
  return '<div class="card"><div class="cv"'+style+'>'+fmtVal+'</div>'+unitHtml+
    '<div class="cl">'+label+(defKey?infoIcon(defKey):'')+'</div></div>';
}
function badgeFor(type) {
  const m = {compute:'badge-compute',memory:'badge-memory',branch:'badge-branch',balanced:'badge-balanced'};
  const tips = {
    compute: 'Compute-bound: >30% of instructions are FP32 ALU or Tensor (WGMMA). Bottleneck = arithmetic pipeline throughput.',
    memory: 'Memory-bound: >30% of instructions are global/shared loads or stores. Bottleneck = memory bandwidth or latency.',
    branch: 'Branch-bound: >30% of instructions are branches. Bottleneck = control flow overhead.',
    balanced: 'Balanced: no single category exceeds 30% of instructions. Workload is mixed or dominated by other op types.'
  };
  return '<span class="badge '+(m[type]||'badge-balanced')+'" title="'+(tips[type]||tips.balanced)+'">'+type+'-bound</span>';
}
function effBar(pct, label) {
  const clr = pct>90?'var(--green)':pct>50?'var(--yellow)':'var(--red)';
  return '<div class="eff-bar" style="background:var(--bg3)"><div class="eff-fill" style="width:'+pct+'%;background:'+clr+'"></div>'+
    '<div class="eff-label">'+label+': '+pct.toFixed(1)+'%</div></div>';
}
function secToggle(id) {
  const hdr=document.getElementById('sh_'+id);
  const body=document.getElementById('sb_'+id);
  if(hdr&&body){hdr.classList.toggle('collapsed');body.classList.toggle('collapsed');}
}
function sectionStart(id, title) {
  return '<div class="msec"><div class="sec-hdr" id="sh_'+id+'" onclick="secToggle(\''+id+'\')">'+
    '<span class="arrow">\u25BC</span><h4 style="border:none;margin:0;padding:0">'+title+'</h4></div>'+
    '<div class="sec-body" id="sb_'+id+'" style="max-height:9999px">';
}
function sectionEnd() { return '</div></div>'; }

// Track ECharts instances for cleanup
const _charts = {};
function mkChart(id, h) {
  if(_charts[id]){_charts[id].dispose();delete _charts[id];}
  const el=document.getElementById(id);
  if(!el)return null;
  const c=echarts.init(el);
  // Patch setOption to ensure all tooltips render on body (avoids clipping by overflow:hidden parents)
  const _origSet=c.setOption.bind(c);
  c.setOption=function(opt){
    if(opt.tooltip) opt.tooltip=Object.assign({appendToBody:true,confine:true},opt.tooltip);
    return _origSet(opt);
  };
  _charts[id]=c;
  new ResizeObserver(()=>c.resize()).observe(el);
  return c;
}

const INST_CLASS_COLORS = {
  alu_fp32:'#1a7f37',alu_int:'#0969da',tensor_wgmma:'#8250df',
  ld_global:'#bc4c00',st_global:'#e16f24',ld_shared:'#0550ae',st_shared:'#368cf9',
  ld_local:'#9a6700',st_local:'#bf8700',barrier:'#cf222e',membar:'#da3633',
  branch:'#6639ba',call:'#8b949e',ret:'#afb8c1',special:'#57606a',other:'#d0d7de'
};

// ── Header ──
document.getElementById('fileName').textContent = D.source.path;
const infoParts = [];
if (Object.keys(D.perLine).length) infoParts.push('<b>'+Object.keys(D.perLine).length+'</b> annotated lines');
if (Object.keys(D.regions).length) infoParts.push('<b>'+Object.keys(D.regions).length+'</b> regions');
if (D.sass.text) infoParts.push('SASS');
if (D.ptx.text) infoParts.push('PTX');
if (D.trace) infoParts.push('Trace');
if (D.sassProfiles && Object.keys(D.sassProfiles).length) infoParts.push('<b>'+Object.keys(D.sassProfiles).length+'</b> SASS profiles');
document.getElementById('headerInfo').innerHTML = infoParts.join(' <span class="sep">|</span> ');

// ── Monaco Setup ──
const MCDN = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min";
require.config({ paths: { vs: MCDN + "/vs" } });
window.MonacoEnvironment = {
  getWorkerUrl(_,__) {
    return `data:text/javascript;charset=utf-8,${encodeURIComponent(
      `self.MonacoEnvironment={baseUrl:"${MCDN}/"};importScripts("${MCDN}/vs/base/worker/workerMain.js");`
    )}`;
  }
};

require(["vs/editor/editor.main"], function () {
  document.getElementById('loading').style.display = 'none';

  // ── Languages ──
  monaco.languages.register({ id: 'cuda' });
  monaco.languages.setMonarchTokensProvider('cuda', {
    keywords: ['void','int','float','double','char','unsigned','long','short','bool','const',
      'static','extern','inline','constexpr','if','else','for','while','do','return','break',
      'continue','struct','class','template','typename','namespace','using','true','false',
      'nullptr','auto','sizeof','typedef','switch','case','default','enum','volatile'],
    cudaKw: ['__global__','__device__','__host__','__shared__','__constant__','__restrict__',
      '__launch_bounds__','__forceinline__','blockIdx','blockDim','threadIdx','gridDim',
      'warpSize','atomicAdd','atomicCAS','__syncthreads','__syncwarp',
      '__shfl_sync','__shfl_xor_sync','__shfl_up_sync','__shfl_down_sync'],
    types: ['dim3','uint32_t','int32_t','uint64_t','int64_t','size_t','float2','float4',
      'int2','int4','cudaError_t','cudaStream_t'],
    tokenizer: { root: [
      [/\/\/.*$/, 'comment'], [/\/\*/, 'comment', '@comment'],
      [/"[^"]*"/, 'string'], [/'[^']*'/, 'string'],
      [/#\w+/, 'keyword.directive'],
      [/\bIKP_\w+\b/, 'keyword.macro'],
      [/\b\d[\d.]*[fFeEuUlL]*\b/, 'number'],
      [/0x[0-9a-fA-F]+/, 'number.hex'],
      [/[a-zA-Z_]\w*/, { cases: { '@keywords':'keyword', '@cudaKw':'keyword.cuda', '@types':'type', '@default':'identifier' }}],
      [/<<</, 'delimiter.cuda'], [/>>>/, 'delimiter.cuda'],
    ], comment: [[/\*\//, 'comment', '@pop'], [/./, 'comment']] }
  });

  monaco.languages.register({ id: 'sass-asm' });
  monaco.languages.setMonarchTokensProvider('sass-asm', {
    fp:['FFMA','FMUL','FADD','HMMA','WGMMA','HFMA2','DFMA','MUFU'],
    mem:['LDG','STG','LDS','STS','LDC','LDSM','LDGSTS','ATOMS','ATOMG'],
    ctrl:['BRA','EXIT','BSSY','BSYNC','RET','CALL','BAR','YIELD','WARPSYNC','NANOSLEEP'],
    intop:['IMAD','IADD','ISETP','S2R','MOV','SEL','SHF','PRMT','LEA','LOP3','IMNMX','SGXT'],
    tokenizer: { root: [
      [/\/\/.*$/, 'comment'],
      [/\/\*[0-9a-fA-F]+\*\//, 'number.hex'],
      [/\b[A-Z][A-Z0-9.]+\b/, { cases: { '@fp':'keyword.fp', '@mem':'keyword.mem', '@ctrl':'keyword.ctrl', '@intop':'keyword.int', '@default':'keyword' }}],
      [/R\d+/, 'variable'], [/P\d+/, 'variable'], [/UR\d+/, 'variable'], [/UP\d+/, 'variable'],
      [/0x[0-9a-fA-F]+/, 'number.hex'], [/\b\d+\b/, 'number'],
    ] }
  });

  monaco.languages.register({ id: 'ptx' });
  monaco.languages.setMonarchTokensProvider('ptx', {
    tokenizer: { root: [
      [/\/\/.*$/, 'comment'],
      [/\.(version|target|address_size|visible|entry|func|global|local|shared|const|param|reg|pred)\b/, 'keyword.directive'],
      [/\.(s8|s16|s32|s64|u8|u16|u32|u64|f16|f32|f64|b8|b16|b32|b64)\b/, 'type'],
      [/\b(ld|st|mov|add|mul|mad|sub|div|rem|and|or|xor|not|shl|shr|setp|selp|bra|ret|exit|bar|atom|red|cvt|abs|neg|min|max|fma|rcp|sqrt)\b/, 'keyword'],
      [/%\w+/, 'variable'], [/0x[0-9a-fA-F]+/, 'number.hex'], [/\b\d+\b/, 'number'], [/"[^"]*"/, 'string'],
    ] }
  });

  // ── Theme ──
  monaco.editor.defineTheme('ikp-light', {
    base: 'vs', inherit: true,
    rules: [
      { token: 'keyword.cuda', foreground: '0550ae', fontStyle: 'bold' },
      { token: 'keyword.macro', foreground: 'bc4c00', fontStyle: 'bold' },
      { token: 'keyword.directive', foreground: '8250df' },
      { token: 'type', foreground: '0550ae' },
      { token: 'keyword.fp', foreground: '1a7f37', fontStyle: 'bold' },
      { token: 'keyword.mem', foreground: 'bc4c00' },
      { token: 'keyword.ctrl', foreground: '8250df' },
      { token: 'keyword.int', foreground: '0969da' },
      { token: 'number.hex', foreground: '656d76' },
      { token: 'variable', foreground: '0550ae' },
    ],
    colors: {
      'editor.background': '#ffffff',
      'editor.lineHighlightBackground': '#f6f8fa00',
      'editorGutter.background': '#f6f8fa',
      'editorLineNumber.foreground': '#656d76',
      'editor.selectionBackground': '#b6d5f2',
      'minimap.background': '#f6f8fa',
    }
  });

  // ── Editor options ──
  const viewerOpts = {
    readOnly: true, domReadOnly: true, theme: 'ikp-light',
    minimap: { enabled: true, renderCharacters: false, scale: 1 },
    scrollBeyondLastLine: false, lineNumbersMinChars: 3,
    glyphMargin: false, folding: false, contextmenu: false, links: false,
    quickSuggestions: false, suggestOnTriggerCharacters: false,
    parameterHints: { enabled: false }, renderValidationDecorations: 'off',
    matchBrackets: 'never', occurrencesHighlight: 'off', selectionHighlight: false,
    overviewRulerLanes: 3, scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
    fontSize: 13, lineHeight: 20, fontFamily: "'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace",
  };

  // ── Create editors (source + PTX + SASS) ──
  const srcEditor = monaco.editor.create(document.getElementById('srcWrap'), {
    ...viewerOpts, value: D.source.code, language: 'cuda', lineDecorationsWidth: 5,
    glyphMargin: true, glyphMarginWidth: 28,
  });
  const ptxEditor = monaco.editor.create(document.getElementById('ptxWrap'), {
    ...viewerOpts, value: D.ptx.text || '// No PTX data', language: 'ptx',
  });
  const sassEditor = monaco.editor.create(document.getElementById('sassWrap'), {
    ...viewerOpts, value: D.sass.text || '// No SASS data', language: 'sass-asm',
  });

  // ── Build reverse maps (ASM line → source line) ──
  const _sassToSrc = {};
  for (const [srcLine, sassLines] of Object.entries(D.sass.lineMap||{})) {
    for (const sl of sassLines) _sassToSrc[sl] = parseInt(srcLine);
  }
  const _ptxToSrc = {};
  for (const [srcLine, ptxLines] of Object.entries(D.ptx.lineMap||{})) {
    for (const pl of ptxLines) _ptxToSrc[pl] = parseInt(srcLine);
  }

  // ── Inject dynamic CSS for region glyph labels ──
  const _glyphStyle = document.createElement('style');
  let _glyphCss = '';
  for (const [rid, lbl] of Object.entries(D.labels)) {
    // Abbreviate: up to 4 chars
    const abbr = lbl.length <= 4 ? lbl : lbl.replace(/[aeiou_]/gi,'').substring(0,4) || lbl.substring(0,4);
    const c = D.colors[rid] || '#8b949e';
    _glyphCss += `.rg-${rid}::after{content:'${abbr}';color:${c};}`;
  }
  _glyphStyle.textContent = _glyphCss;
  document.head.appendChild(_glyphStyle);

  // ── Build region nesting map (parent region for each line) ──
  const _regionNesting = {};  // rid -> parent_rid
  {
    const LR = D.lineRegions || {};
    const lines = Object.keys(LR).map(Number).sort((a,b)=>a-b);
    // Track which regions contain which others based on line coverage
    const regionLines = {};
    for (const [ln, rid] of Object.entries(LR)) {
      if (!regionLines[rid]) regionLines[rid] = [];
      regionLines[rid].push(parseInt(ln));
    }
    for (const [rid, rlines] of Object.entries(regionLines)) {
      const rmin = Math.min(...rlines), rmax = Math.max(...rlines);
      for (const [pid, plines] of Object.entries(regionLines)) {
        if (pid === rid) continue;
        const pmin = Math.min(...plines), pmax = Math.max(...plines);
        if (pmin <= rmin && pmax >= rmax && plines.length > rlines.length) {
          // pid contains rid — pid is parent
          if (!_regionNesting[rid] || regionLines[_regionNesting[rid]].length > plines.length) {
            _regionNesting[rid] = parseInt(pid);  // choose smallest containing parent
          }
        }
      }
    }
  }

  // ── Source decorations ──
  const srcDecos = [];
  const srcModel = srcEditor.getModel();
  const LR = D.lineRegions || {};  // line -> region_id (from source parsing)
  for (let i = 1; i <= srcModel.getLineCount(); i++) {
    const info = D.perLine[i];
    const lineRegion = LR[i];  // region from source code parsing
    if (!info && lineRegion == null) continue;

    // Determine the effective region for this line
    const rid = (lineRegion != null) ? lineRegion : (info ? info.region : null);
    const inRegion = (rid != null);

    let cls = '';
    let hoverMd = `**Line ${i}**`;
    const opts = { isWholeLine: true };

    if (info) {
      // Only apply heat-based background highlighting for lines INSIDE a region
      if (inRegion) {
        if (info.heat > 0.8) cls = 'line-hot';
        else if (info.heat > 0.3) cls = 'line-warm';
        else if (info.heat > 0.01) cls = 'line-cold';
      }
      const ie = info.m['smsp__sass_inst_executed'] || 0;
      const te = info.m['smsp__sass_thread_inst_executed'] || 0;
      const rlabel = inRegion ? (D.labels[rid] || 'region_'+rid) : null;
      if (rlabel) hoverMd += ` \u2014 *${rlabel}*`;
      // Show parent region if nested
      if (rid != null && _regionNesting[rid] != null) {
        const plab = D.labels[_regionNesting[rid]] || 'region_'+_regionNesting[rid];
        hoverMd += ` (inside *${plab}*)`;
      }
      hoverMd += `\n\nInstructions: **${fmt(ie)}** | Hotness: **${(info.heat*100).toFixed(1)}%**`;
      if (te > 0 && ie > 0) hoverMd += `\n\nActive threads: ${(te/(ie*32)*100).toFixed(1)}%`;
      hoverMd += `\n\n${info.pcs} PCs | ${info.profiles.join(', ')}`;
      if (inRegion) {
        opts.overviewRuler = info.heat > 0 ? {
          color: info.heat > 0.8 ? '#cf222e' : info.heat > 0.3 ? '#9a6700' : '#0969da',
          position: monaco.editor.OverviewRulerLane.Full
        } : undefined;
        opts.minimap = info.heat > 0 ? {
          color: info.heat > 0.8 ? '#cf222e' : info.heat > 0.3 ? '#9a6700' : '#0969da',
          position: monaco.editor.MinimapPosition.Inline
        } : undefined;
      }
    } else if (lineRegion != null) {
      // Line inside a region but no CUPTI data — show region bar + hover
      const rlabel = D.labels[lineRegion] || 'region_'+lineRegion;
      hoverMd += ` \u2014 *${rlabel}*`;
      if (_regionNesting[lineRegion] != null) {
        const plab = D.labels[_regionNesting[lineRegion]] || 'region_'+_regionNesting[lineRegion];
        hoverMd += ` (inside *${plab}*)`;
      }
    }

    opts.className = cls;
    opts.hoverMessage = [{ value: hoverMd }];

    // Add region gutter bar + glyph label for lines inside a profiling region
    if (rid != null) {
      opts.linesDecorationsClassName = 'region-bar region-bar-'+rid;
      opts.glyphMarginClassName = 'region-glyph rg-'+rid;
    }
    srcDecos.push({ range: new monaco.Range(i,1,i,1), options: opts });
  }
  srcEditor.createDecorationsCollection(srcDecos);

  // ── Source legend (region color map + heat legend) ──
  {
    const leg = document.getElementById('srcLegend');
    let lh = '';
    // Region colors
    const rids = Object.keys(D.labels).map(Number).sort();
    for (const rid of rids) {
      const c = D.colors[rid] || '#8b949e';
      lh += '<span class="leg-item" onclick="window._selectRegion('+rid+');switchTab(\'region\')" title="Click to show region '+D.labels[rid]+'">'+
        '<span class="leg-dot" style="background:'+c+'"></span>'+D.labels[rid]+'</span>';
    }
    // Heat legend
    lh += '<span style="margin-left:8px;border-left:1px solid var(--border);padding-left:8px">Heat:</span>';
    lh += '<span class="leg-item"><span class="leg-heat" style="background:rgba(207,34,46,.10)"></span>hot</span>';
    lh += '<span class="leg-item"><span class="leg-heat" style="background:rgba(154,103,0,.08)"></span>warm</span>';
    lh += '<span class="leg-item"><span class="leg-heat" style="background:rgba(9,105,218,.06)"></span>cold</span>';
    leg.innerHTML = lh;
  }

  // ── Bidirectional cross-highlighting (Source ↔ PTX ↔ SASS) ──
  let selectedLine = null;
  let _crossLock = false;  // prevent recursive triggers
  let sassDecoCollection = sassEditor.createDecorationsCollection([]);
  let ptxDecoCollection = ptxEditor.createDecorationsCollection([]);
  let srcSelectCollection = srcEditor.createDecorationsCollection([]);

  function highlightFromSource(srcLine, opts) {
    opts = opts || {};
    const ptxStatus = document.getElementById('ptxStatus');
    const sassStatus = document.getElementById('sassStatus');
    // PTX
    const ptxIndices = (D.ptx.lineMap||{})[srcLine] || [];
    if (ptxIndices.length) {
      ptxDecoCollection.set(ptxIndices.map(ln => ({
        range: new monaco.Range(ln,1,ln,1),
        options: { isWholeLine: true, className: 'asm-highlight' }
      })));
      if (!opts.noPtxScroll) ptxEditor.revealLineInCenter(ptxIndices[0]);
      if (ptxStatus) ptxStatus.textContent = ptxIndices.length + ' line' + (ptxIndices.length>1?'s':'') + ' (L' + srcLine + ')';
    } else {
      ptxDecoCollection.set([]);
      if (ptxStatus) ptxStatus.textContent = 'L' + srcLine + ': no PTX mapping';
    }
    // SASS
    const sassIndices = (D.sass.lineMap||{})[srcLine] || [];
    if (sassIndices.length) {
      sassDecoCollection.set(sassIndices.map(ln => ({
        range: new monaco.Range(ln,1,ln,1),
        options: { isWholeLine: true, className: 'asm-highlight' }
      })));
      if (!opts.noSassScroll) sassEditor.revealLineInCenter(sassIndices[0]);
      if (sassStatus) sassStatus.textContent = sassIndices.length + ' line' + (sassIndices.length>1?'s':'') + ' (L' + srcLine + ')';
    } else {
      sassDecoCollection.set([]);
      if (sassStatus) sassStatus.textContent = 'L' + srcLine + ': no SASS mapping';
    }
  }

  // Click source → highlight PTX + SASS
  srcEditor.onMouseDown(e => {
    if (e.target.position && !_crossLock) selectLine(e.target.position.lineNumber);
  });

  function selectLine(n) {
    selectedLine = n;
    srcSelectCollection.set([{
      range: new monaco.Range(n,1,n,1),
      options: { isWholeLine: true, className: 'line-selected' }
    }]);
    highlightFromSource(n);
    buildLineMetrics(n);
    switchTab('line');
  }

  // Click PTX → highlight source + SASS
  ptxEditor.onMouseDown(e => {
    if (!e.target.position || _crossLock) return;
    const ptxLine = e.target.position.lineNumber;
    const srcLine = _ptxToSrc[ptxLine];
    if (srcLine) {
      _crossLock = true;
      selectedLine = srcLine;
      srcSelectCollection.set([{
        range: new monaco.Range(srcLine,1,srcLine,1),
        options: { isWholeLine: true, className: 'line-selected' }
      }]);
      srcEditor.revealLineInCenter(srcLine);
      highlightFromSource(srcLine, {noPtxScroll: true});
      buildLineMetrics(srcLine);
      switchTab('line');
      _crossLock = false;
    }
  });

  // Click SASS → highlight source + PTX
  sassEditor.onMouseDown(e => {
    if (!e.target.position || _crossLock) return;
    const sassLine = e.target.position.lineNumber;
    const srcLine = _sassToSrc[sassLine];
    if (srcLine) {
      _crossLock = true;
      selectedLine = srcLine;
      srcSelectCollection.set([{
        range: new monaco.Range(srcLine,1,srcLine,1),
        options: { isWholeLine: true, className: 'line-selected' }
      }]);
      srcEditor.revealLineInCenter(srcLine);
      highlightFromSource(srcLine, {noSassScroll: true});
      buildLineMetrics(srcLine);
      switchTab('line');
      _crossLock = false;
    }
  });

  // ── Split.js ──
  const hasSass = !!(D.sass.text);
  const hasPtx = !!(D.ptx.text);
  const hasAsm = hasSass || hasPtx;
  function layoutAll() { srcEditor.layout(); ptxEditor.layout(); sassEditor.layout(); }
  Split(['#srcPanel', '#asmPanel', '#metPanel'], {
    sizes: hasAsm ? [35, 30, 35] : [55, 0, 45],
    minSize: [200, hasAsm ? 150 : 0, 250],
    gutterSize: 8,
    onDrag: layoutAll,
    onDragEnd: layoutAll,
  });
  // Vertical split inside asmPanel: PTX top, SASS bottom
  if (hasAsm) {
    Split(['#ptxPane', '#sassPane'], {
      sizes: hasPtx && hasSass ? [50, 50] : (hasPtx ? [100, 0] : [0, 100]),
      minSize: [0, 0],
      gutterSize: 6,
      direction: 'vertical',
      onDrag: layoutAll,
      onDragEnd: layoutAll,
    });
  }
  if (!hasPtx) document.getElementById('ptxPane').style.display = 'none';
  if (!hasSass) document.getElementById('sassPane').style.display = 'none';
  setTimeout(layoutAll, 50);
  window.addEventListener('resize', layoutAll);

  // ── Metric tabs ──
  const tabBuilders = {
    ov: buildOverview, line: ()=>{}, region: ()=>{}, exec: buildExecution,
    mem: buildMemory, stalls: buildStalls, trace: buildTrace
  };
  let builtTabs = {};
  function switchTab(name) {
    document.querySelectorAll('#tabBar .tab').forEach(t => t.classList.toggle('active', t.dataset.t === name));
    document.querySelectorAll('.tc').forEach(t => t.classList.toggle('active', t.id === 'tc-' + name));
    // Reset scroll position so new tab starts at top
    const ms = document.querySelector('.mscroll');
    if (ms) ms.scrollTop = 0;
    if (!builtTabs[name] && tabBuilders[name]) { tabBuilders[name](); builtTabs[name]=true; }
  }
  document.getElementById('tabBar').addEventListener('click', e => {
    const tab = e.target.closest('.tab');
    if (tab) switchTab(tab.dataset.t);
  });

  // ══════════════════════════════════════════════════════════════════
  // TAB 1: OVERVIEW
  // ══════════════════════════════════════════════════════════════════
  function buildOverview() {
    const ct = document.getElementById('ovCt');
    const rids = Object.keys(D.regions).map(Number).sort();
    let h = '';

    // Summary cards
    let totalInst=0, totalGmem=0, totalSmem=0, totalLmem=0;
    for (const r of Object.values(D.regions)) {
      totalInst += r.inst_total||0;
      totalGmem += r.gmem_bytes||0;
      totalSmem += r.smem_bytes||0;
      totalLmem += r.lmem_bytes||0;
    }
    h += '<div class="cards cards-3">';
    h += card(totalInst, 'Total Instructions', 'inst_total');
    h += card(rids.length, 'Active Regions', null, {raw:true,noUnit:true});
    const bottlenecks = rids.map(rid=>D.regions[rid]?.derived?.bottleneck).filter(Boolean);
    const topBn = bottlenecks.length ? bottlenecks.sort((a,b)=>bottlenecks.filter(x=>x===b).length-bottlenecks.filter(x=>x===a).length)[0] : 'balanced';
    h += '<div class="card"><div class="cv" style="font-size:14px">'+badgeFor(topBn)+'</div><div class="cl">Bottleneck'+infoIcon('bottleneck')+'</div></div>';
    h += card(totalGmem+totalSmem+totalLmem, 'Total Memory', 'gmem_bytes');
    if (D.trace) {
      const total = D.trace.find(r=>r.name==='total');
      h += card(total?total.count:D.trace.reduce((s,r)=>s+r.count,0), 'Trace Events', null, {noUnit:true});
    } else {
      h += '<div class="card"><div class="cv" style="color:var(--dim)">N/A</div><div class="cl">Trace Events</div></div>';
    }
    const nSources = (D.dataQuality.nvbit_modes.length>0?1:0) + (D.dataQuality.cupti_profiles.length>0?1:0) +
      (D.dataQuality.has_trace?1:0) + (D.dataQuality.has_pcsamp?1:0) + (D.dataQuality.has_locality?1:0);
    h += card(nSources, 'Data Sources', null, {raw:true,noUnit:true});
    h += '</div>';

    // Radar chart (region comparison)
    if (rids.length > 0) {
      h += sectionStart('ov_radar','Region Comparison');
      h += '<div style="font-size:10px;color:var(--dim);margin-bottom:4px;line-height:1.5">'+
        'Compares instruction mix across regions. Each axis is a <b>fraction of total instructions</b> in that region:<br>'+
        '<b>FP32%</b> = ALU FP32 ops, <b>Memory%</b> = global+shared load/store ops, '+
        '<b>Branch%</b> = branch instructions, <b>Divergence%</b> = 1 \u2212 branch uniformity, '+
        '<b>Pred-off%</b> = predicated-off instruction fraction.</div>';
      h += '<div id="ovRadar" class="chart" style="height:300px;"></div>';
      h += sectionEnd();
    }

    // Data quality table
    h += sectionStart('ov_dq','Data Quality');
    h += '<table class="ptable" style="text-align:left"><tr><th style="text-align:left">Source</th><th style="text-align:left">Status</th><th>Records</th></tr>';
    const dq = D.dataQuality;
    const dqRow = (name,present,count) => '<tr><td style="text-align:left">'+name+'</td><td style="text-align:left">'+
      (present?'<span class="badge badge-present">Present</span>':'<span class="badge badge-miss">Missing</span>')+
      '</td><td>'+fmt(count)+'</td></tr>';
    h += dqRow('NVBit Regions', Object.keys(D.regions).length>0, Object.keys(D.regions).length);
    h += dqRow('SASS Profiles', dq.cupti_profiles.length>0, dq.cupti_profiles.length);
    h += dqRow('PC Sampling', dq.has_pcsamp, dq.pcsamp_total);
    h += dqRow('InstrExec', dq.has_instrexec, dq.instrexec_total);
    h += dqRow('Trace', dq.has_trace, D.trace?D.trace.reduce((s,r)=>s+r.count,0):0);
    h += dqRow('Locality', dq.has_locality, D.locality?Object.keys(D.locality.regions).length:0);
    h += dqRow('Source Mapping', dq.has_source_mapping, Object.keys(D.perLine).length);
    h += dqRow('CUPTI\u00D7NVBit Cross-ref', D.crossValidation && Object.keys(D.crossValidation).length>0,
      D.crossValidation ? Object.keys(D.crossValidation).length : 0);
    if (dq.cupti_profiles.length) h += '<tr><td style="text-align:left;color:var(--dim)" colspan=3>Profiles: '+dq.cupti_profiles.join(', ')+'</td></tr>';
    if (dq.nvbit_modes.length) h += '<tr><td style="text-align:left;color:var(--dim)" colspan=3>NVBit modes: '+dq.nvbit_modes.join(', ')+'</td></tr>';
    h += '</table>';

    // Cross-validation summary: instruction mix fraction agreement
    if (D.crossValidation && Object.keys(D.crossValidation).length > 0) {
      h += '<div style="margin-top:6px;font-size:10px;font-weight:600;color:var(--bright)">NVBit \u00d7 CUPTI Instruction Mix Agreement'+infoIcon('nvbit_vs_cupti')+'</div>';
      h += '<div style="font-size:9px;color:var(--dim);margin-bottom:4px">Compares compute/memory/branch instruction fractions between tools. Lower delta = better agreement.</div>';
      h += '<table class="ptable" style="text-align:left;font-size:10px;margin-top:4px">';
      h += '<tr><th style="text-align:left">Region</th><th style="text-align:right">Compute \u0394</th><th style="text-align:right">GlobMem \u0394</th><th style="text-align:right">SharedMem \u0394</th><th style="text-align:right">Branch \u0394</th><th style="text-align:right">Avg \u0394</th><th style="text-align:left">Status</th></tr>';
      for (const [cvRid, cv] of Object.entries(D.crossValidation).sort((a,b)=>Number(a[0])-Number(b[0]))) {
        const label = D.labels[cvRid] || 'region_'+cvRid;
        const color = D.colors[cvRid] || '#656d76';
        const avgD = cv.avg_delta||0;
        const avgColor = avgD < 0.02 ? 'var(--green)' : avgD < 0.1 ? 'var(--orange)' : 'var(--red)';
        h += '<tr>';
        h += '<td style="text-align:left"><span style="color:'+color+'">\u25CF</span> '+label+'</td>';
        for (const cat of ['compute','global_mem','shared_mem','branch']) {
          const f = cv.fractions && cv.fractions[cat];
          if (f) {
            const dc = f.delta < 0.02 ? 'var(--green)' : f.delta < 0.1 ? 'var(--orange)' : 'var(--red)';
            h += '<td style="text-align:right;color:'+dc+'">'+(f.delta*100).toFixed(1)+'pp</td>';
          } else {
            h += '<td style="text-align:right;color:var(--dim)">-</td>';
          }
        }
        h += '<td style="text-align:right;color:'+avgColor+';font-weight:600">'+(avgD*100).toFixed(1)+'pp</td>';
        h += '<td style="text-align:left;color:'+avgColor+';font-size:9px">'+(avgD<0.02?'Excellent':avgD<0.1?'Good':'Investigate')+'</td>';
        h += '</tr>';
      }
      h += '</table>';
    }
    h += sectionEnd();

    // Bottleneck hints
    if (rids.length) {
      h += sectionStart('ov_bn','Bottleneck Hints');
      h += '<div style="font-size:9px;color:var(--dim);margin-bottom:6px;line-height:1.5">'+
        'Classification: <b>compute_frac</b> = (alu_fp32 + tensor_wgmma) / inst_total, '+
        '<b>memory_frac</b> = (ld/st_global + ld/st_shared) / inst_total, '+
        '<b>branch_frac</b> = branch / inst_total. '+
        'Largest fraction > 30% \u2192 that type; otherwise balanced.</div>';
      for (const rid of rids) {
        const r = D.regions[rid];
        if (!r) continue;
        const label = r.label || 'region_'+rid;
        const bn = r.derived?.bottleneck || 'balanced';
        const ic = r.inst_class || {};
        const it = r.inst_total || 1;
        const computeFrac = ((ic.alu_fp32||0)+(ic.tensor_wgmma||0))/it*100;
        const memFrac = ((ic.ld_global||0)+(ic.st_global||0)+(ic.ld_shared||0)+(ic.st_shared||0))/it*100;
        const branchFrac = (ic.branch||0)/it*100;
        let hint = label + ': ' + fmt(r.inst_total||0) + ' instr';
        hint += ' (compute=' + computeFrac.toFixed(1) + '%, memory=' + memFrac.toFixed(1) + '%, branch=' + branchFrac.toFixed(1) + '%)';
        if ((r.gmem_bytes||0)>0) hint += ', ' + fmtBytes(r.gmem_bytes) + ' gmem';
        if ((r.reg_spill_suspected||0)>0) hint += ', SPILL DETECTED';
        h += '<div style="font-size:11px;margin-bottom:4px;display:flex;align-items:center;gap:6px">'+
          '<span style="color:'+(D.colors[rid]||'var(--dim)')+'">\u25CF</span>'+
          badgeFor(bn)+' <span style="color:var(--dim)">'+hint+'</span></div>';
      }
      h += sectionEnd();
    }

    // Tips
    h += '<div class="msec"><h4>Tips</h4><div style="font-size:11px;color:var(--dim);line-height:1.8">';
    h += 'Click source line \u2192 see metrics + SASS highlight<br>';
    h += 'Click region \u2192 highlight all lines + full analysis<br>';
    h += 'Drag panel borders \u2192 resize<br>';
    h += 'SASS/PTX tabs \u2192 switch assembly view<br>';
    h += 'Click <b style="font-style:italic;border:1px solid var(--dim);border-radius:50%;width:11px;height:11px;display:inline-flex;align-items:center;justify-content:center;font-size:8px">i</b> icons \u2192 metric descriptions';
    h += '</div></div>';

    ct.innerHTML = h;

    // Radar chart — use dynamic axis max and filter empty regions
    if (rids.length > 0) {
      const el = document.getElementById('ovRadar');
      if (el) {
        const ch = mkChart('ovRadar');
        const axes = ['FP32%','Memory%','Branch%','Divergence%','Pred-off%'];
        const series = [];
        const colors = ['#0969da','#1a7f37','#bc4c00','#8250df','#9a6700','#cf222e','#6639ba','#656d76'];
        // Compute per-axis max for dynamic scaling
        const axisMax = [0,0,0,0,0];
        const rawData = [];
        for (const rid of rids) {
          const r = D.regions[rid];
          if (!r || !r.inst_total) continue;
          // Skip region 0 (_outside/prologue) if label starts with underscore
          if ((r.label||'').startsWith('_')) continue;
          const ic = r.inst_class||{};
          const it = r.inst_total;
          const vals = [
            (ic.alu_fp32||0)/it,
            ((ic.ld_global||0)+(ic.st_global||0)+(ic.ld_shared||0)+(ic.st_shared||0))/it,
            (ic.branch||0)/it,
            r.derived?.branch_uniformity!=null ? 1-r.derived.branch_uniformity : 0,
            r.derived?.predication_rate||0
          ];
          vals.forEach((v,i)=>{if(v>axisMax[i])axisMax[i]=v;});
          rawData.push({rid, r, vals});
        }
        // Dynamic max: round up to nearest nice value, minimum 0.05 so chart isn't invisible
        const niceMax = axisMax.map(m => {
          if (m <= 0.001) return 0.05;
          if (m <= 0.05) return 0.1;
          if (m <= 0.15) return 0.2;
          if (m <= 0.3) return 0.4;
          if (m <= 0.5) return 0.6;
          if (m <= 0.8) return 1.0;
          return 1.0;
        });
        for (const {rid, r, vals} of rawData) {
          series.push({
            name: r.label||'region_'+rid,
            value: vals,
            lineStyle:{color:colors[rid%8],width:2},
            itemStyle:{color:colors[rid%8]},
            areaStyle:{color:colors[rid%8],opacity:0.15},
            symbol:'circle',symbolSize:5,
          });
        }
        if (series.length > 0) {
          ch.setOption({
            backgroundColor:'transparent',
            legend:{top:0,left:'center',textStyle:{color:'#656d76',fontSize:10},padding:[0,0,8,0],
              itemWidth:14,itemHeight:8,itemGap:12},
            tooltip:{trigger:'item',formatter:function(p){
              if(!p.value)return'';
              return '<b>'+p.seriesName+'</b><br>'+axes.map((a,i)=>'  '+a+': '+(p.value[i]*100).toFixed(1)+'%').join('<br>');
            }},
            radar:{center:['50%','55%'],radius:'60%',
              indicator:axes.map((n,i)=>({name:n+' ('+Math.round(niceMax[i]*100)+'%)',max:niceMax[i]})),
              axisName:{color:'#656d76',fontSize:9},
              axisLabel:{show:true,color:'#8b949e',fontSize:8,formatter:function(v){return(v*100).toFixed(0)+'%';}},
              splitNumber:4,
              splitLine:{lineStyle:{color:'#d0d7de'}},
              splitArea:{areaStyle:{color:['transparent','rgba(0,0,0,0.02)']}}},
            series:[{type:'radar',data:series}],
          });
        } else {
          el.innerHTML = '<div style="color:var(--dim);font-size:11px;text-align:center;padding:20px">No regions with instruction data for radar chart</div>';
        }
      }
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // TAB 2: LINE METRICS
  // ══════════════════════════════════════════════════════════════════
  function buildLineMetrics(n) {
    const ct = document.getElementById('lineMet');
    const info = D.perLine[n];
    if (!info || !Object.keys(info.m).length) {
      const lr = (D.lineRegions||{})[n];
      if (lr != null) {
        // Line is inside a region — show region-level aggregate stats as fallback
        const rid = parseInt(lr);
        const rl = D.labels[rid]||'region_'+rid;
        const c = D.colors[rid]||'#8b949e';
        const stats = D.regions[rid];
        let h = '<div style="margin-bottom:8px;display:flex;align-items:center;gap:6px">'+
          '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+c+'"></span>'+
          '<span style="color:'+c+';font-weight:600;font-size:14px">'+rl+'</span>';
        if (_regionNesting[rid] != null) {
          const plab = D.labels[_regionNesting[rid]]||'region_'+_regionNesting[rid];
          h += '<span style="color:var(--dim);font-size:9px">(inside '+plab+')</span>';
        }
        h += '<span style="color:var(--dim);font-size:9px;cursor:pointer;text-decoration:underline" '+
          'onclick="window._selectRegion('+rid+');switchTab(\'region\')">View region \u2192</span></div>';
        // Explain why no per-line data
        h += '<div style="font-size:10px;color:var(--dim);margin-bottom:8px;padding:4px 8px;background:var(--bg2);border-radius:4px">'+
          'Line '+n+': compiler merged this line\'s instructions into a neighboring line. '+
          'Showing <b>'+rl+'</b> region-level aggregates below.</div>';
        // Show region-level summary cards
        if (stats) {
          h += '<div class="cards cards-3">';
          h += card(stats.inst_total||0, 'Region Instructions', 'inst_total');
          const predPct = stats.inst_total>0 ? ((stats.inst_pred_off||0)/stats.inst_total*100).toFixed(1)+'%' : '0%';
          h += card(predPct, 'Pred Off', 'inst_pred_off', {raw:true,noUnit:true});
          h += '<div class="card"><div class="cv" style="font-size:12px">'+badgeFor(stats.derived?.bottleneck||'balanced')+'</div><div class="cl">Bottleneck</div></div>';
          h += card(stats.gmem_bytes||0, 'Global Mem', 'gmem_bytes');
          h += card(stats.smem_bytes||0, 'Shared Mem', 'smem_bytes');
          h += card(stats.bb_exec||0, 'BB Execs', 'bb_exec');
          h += '</div>';
          // Instruction class summary
          const ic = stats.inst_class||{};
          const top3 = Object.entries(ic).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]).slice(0,5);
          if (top3.length) {
            h += '<div class="msec"><h4>Region Top Instruction Classes</h4>';
            for (const [cls,cnt] of top3) {
              const pct = stats.inst_total>0 ? (cnt/stats.inst_total*100).toFixed(1) : '0';
              h += '<div class="mrow"><span class="mn" style="color:'+(INST_CLASS_COLORS[cls]||'var(--dim)')+'">'+cls+'</span><span class="mv">'+fmt(cnt)+' ('+pct+'%)</span></div>';
            }
            h += '</div>';
          }
        }
        // Find nearest lines WITH data in same region
        const nearLines = [];
        for (let delta = 1; delta <= 15; delta++) {
          for (const off of [-delta, delta]) {
            const nl = n + off;
            if (nl > 0 && D.perLine[String(nl)] && (D.lineRegions||{})[String(nl)] == rid) {
              nearLines.push(nl);
            }
          }
        }
        if (nearLines.length) {
          h += '<div style="font-size:10px;color:var(--dim);margin-top:6px">Nearest lines with per-line data: '+
            nearLines.slice(0,6).map(l => {
              const lInfo = D.perLine[String(l)];
              const lInst = lInfo ? fmt(lInfo.m['smsp__sass_inst_executed']||0) : '?';
              const lCode = D.source.code.split('\\n')[l-1]?.trim().substring(0,40)||'';
              return '<span style="cursor:pointer;color:var(--accent);text-decoration:underline" '+
                'onclick="selectLine('+l+')">L'+l+'</span> <span style="color:var(--dim);font-size:9px">('+lInst+' inst)</span>';
            }).join(', ')+'</div>';
        }
        ct.innerHTML = h;
      } else {
        ct.innerHTML = '<div class="empty">No metrics for line ' + n + '</div>';
      }
      return;
    }
    let h = '';
    // Determine effective region: prefer lineRegions, fall back to perLine.region
    const effectiveRid = (D.lineRegions||{})[n] != null ? parseInt((D.lineRegions||{})[n]) : info.region;
    const effectiveLabel = effectiveRid != null ? (D.labels[effectiveRid]||'region_'+effectiveRid) : null;
    if (effectiveLabel) {
      const c = D.colors[effectiveRid] || '#8b949e';
      h += '<div style="margin-bottom:8px;display:flex;align-items:center;gap:6px">'+
        '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+c+'"></span>'+
        '<span style="color:'+c+';font-weight:600;font-size:14px">'+effectiveLabel+'</span>';
      // Show nesting
      if (_regionNesting[effectiveRid] != null) {
        const plab = D.labels[_regionNesting[effectiveRid]]||'region_'+_regionNesting[effectiveRid];
        h += '<span style="color:var(--dim);font-size:9px">(inside '+plab+')</span>';
      }
      h += '<span style="color:var(--dim);font-size:10px">'+info.pcs+' PCs</span>'+
        '<span style="color:var(--dim);font-size:9px;cursor:pointer;text-decoration:underline" '+
        'onclick="window._selectRegion('+effectiveRid+');switchTab(\'region\')">View region \u2192</span></div>';
    } else {
      // Line has CUPTI data but is outside any profiling region
      h += '<div style="margin-bottom:8px;color:var(--dim);font-size:10px">'+
        'Line '+n+': outside profiling regions (compiler debug-info artifact)</div>';
    }
    const ie=info.m['smsp__sass_inst_executed']||0, te=info.m['smsp__sass_thread_inst_executed']||0,
          tp=info.m['smsp__sass_thread_inst_executed_pred_on']||0;
    h += '<div class="cards">';
    h += card(ie, 'Instructions', 'smsp__sass_inst_executed');
    if(te>0&&ie>0) h += card((te/(ie*32)*100).toFixed(1)+'%', 'Active Threads', 'smsp__sass_thread_inst_executed', {raw:true,noUnit:true});
    if(tp>0&&te>0) h += card((tp/te*100).toFixed(1)+'%', 'Pred On', 'smsp__sass_thread_inst_executed_pred_on', {raw:true,noUnit:true});
    h += card((info.heat*100).toFixed(0)+'%', 'Hotness', 'heat', {raw:true,noUnit:true});
    h += '</div>';
    h += '<div id="lineChart" class="chart" style="height:200px;"></div>';
    h += '<div class="msec"><h4>All Metrics (Line '+n+')</h4>';
    const sorted = Object.entries(info.m).sort((a,b)=>b[1]-a[1]);
    for (const [m,v] of sorted) {
      const short = m.replace('smsp__sass_','').replace('_executed','');
      const defKey = m;
      const unit = D.defs[defKey]?.unit ? ' <span style="color:var(--dim);font-size:9px">'+D.defs[defKey].unit+'</span>' : '';
      h += '<div class="mrow'+(m==='smsp__sass_inst_executed'?' hl':'')+'"><span class="mn">'+short+infoIcon(defKey)+'</span><span class="mv">'+fmt(v)+unit+'</span></div>';
    }
    h += '</div>';
    ct.innerHTML = h;
    // Render chart
    const chartEl = document.getElementById('lineChart');
    if (chartEl && sorted.length > 1) {
      const chart = mkChart('lineChart');
      chart.setOption({
        backgroundColor: 'transparent',
        grid: { left: 10, right: 10, top: 10, bottom: 20, containLabel: true },
        xAxis: { type: 'category', data: sorted.slice(0,8).map(([m])=>m.replace('smsp__sass_','').replace('_executed','')),
          axisLabel: { fontSize: 9, color: '#656d76', rotate: 30 } },
        yAxis: { type: 'value', name: 'Count', nameTextStyle:{color:'#656d76',fontSize:9},
          axisLabel: { fontSize: 9, color: '#656d76', formatter: v=>fmt(v) }, splitLine: { lineStyle: { color: '#d0d7de' } } },
        series: [{ type: 'bar', data: sorted.slice(0,8).map(([,v])=>v), itemStyle: { color: '#0969da', borderRadius: [3,3,0,0] }, barWidth: '60%' }],
        tooltip: { trigger: 'axis', formatter: function(p) {
          const key = 'smsp__sass_'+p[0].name.replace(/^(thread_)?inst/,'inst')+'_executed';
          const desc = D.defs[key]?.long||'';
          return '<b>'+p[0].name+'</b>: '+fmt(p[0].value)+(desc?'<br><span style="font-size:10px;color:#999">'+desc+'</span>':'');
        }},
      });
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // TAB 3: REGIONS
  // ══════════════════════════════════════════════════════════════════
  function buildRegions() {
    const ct = document.getElementById('regionList');
    let h = '<div class="msec"><h4>Regions</h4>';
    const rids = Object.keys(D.regions).map(Number).sort();
    for (const [,info] of Object.entries(D.perLine)) {
      if (info.region != null && !rids.includes(info.region)) rids.push(info.region);
    }
    rids.sort((a,b)=>a-b);
    for (const rid of rids) {
      const label = (D.labels[rid]||D.regions[rid]?.label||'region_'+rid);
      const color = D.colors[rid]||'#8b949e';
      const inst = D.regions[rid]?.inst_total||0;
      const bn = D.regions[rid]?.derived?.bottleneck;
      h += '<div class="ri" onclick="window._selectRegion('+rid+')"><span class="ri-dot" style="background:'+color+'"></span>'+
           '<span class="ri-name">'+label+'</span>';
      if (bn) h += '<span style="font-size:8px;margin-right:2px">'+badgeFor(bn)+'</span>';
      h += '<span class="ri-val">'+fmt(inst)+'</span></div>';
    }
    h += '</div>';
    ct.innerHTML = h;
  }
  buildRegions();

  let regionHighlightCollection = srcEditor.createDecorationsCollection([]);
  window._selectRegion = function(rid) {
    document.querySelectorAll('.ri').forEach(el => {
      const match = el.querySelector('.ri-name');
      const label = D.labels[rid]||D.regions[rid]?.label||'region_'+rid;
      el.classList.toggle('active', match && match.textContent===label);
    });
    // Highlight ALL source lines in this region (from lineRegions, which is authoritative)
    const decos = [];
    const LR = D.lineRegions || {};
    let firstLine = Infinity;
    // Collect lines from lineRegions (innermost match)
    for (const [ln, lrid] of Object.entries(LR)) {
      if (parseInt(lrid) === rid) {
        const lineNum = parseInt(ln);
        if (lineNum < firstLine) firstLine = lineNum;
        decos.push({ range: new monaco.Range(lineNum,1,lineNum,1),
          options: { isWholeLine: true, className: 'line-flash' } });
      }
    }
    // Also include lines that are inside child regions (nested)
    for (const [ln, lrid] of Object.entries(LR)) {
      const childRid = parseInt(lrid);
      if (childRid !== rid && _regionNesting[childRid] === rid) {
        const lineNum = parseInt(ln);
        if (lineNum < firstLine) firstLine = lineNum;
        if (!decos.find(d => d.range.startLineNumber === lineNum)) {
          decos.push({ range: new monaco.Range(lineNum,1,lineNum,1),
            options: { isWholeLine: true, className: 'line-flash' } });
        }
      }
    }
    regionHighlightCollection.set(decos);
    // Scroll source editor to the first line of this region
    if (firstLine < Infinity) {
      srcEditor.revealLineInCenter(firstLine);
    }
    // Clear flash after animation completes
    setTimeout(() => {
      // Re-apply as static highlight
      const staticDecos = decos.map(d => ({
        range: d.range,
        options: { isWholeLine: true, className: 'line-selected' }
      }));
      regionHighlightCollection.set(staticDecos);
    }, 900);

    const det = document.getElementById('regionDet');
    const stats = D.regions[rid];
    const label = D.labels[rid]||stats?.label||'region_'+rid;
    const color = D.colors[rid]||'#8b949e';
    let h = '';

    if (!stats) { det.innerHTML = '<div class="empty">No data for region '+rid+'</div>'; return; }

    // Section A: Summary Cards
    h += sectionStart('r_summary','Summary');
    h += '<div class="cards cards-4">';
    h += card(stats.inst_total||0, 'Instructions', 'inst_total');
    const predPct = stats.inst_total>0 ? ((stats.inst_pred_off||0)/stats.inst_total*100).toFixed(1)+'%' : '0%';
    h += card(predPct, 'Pred Off', 'inst_pred_off', {raw:true,noUnit:true});
    h += card(stats.bb_exec||0, 'BB Executions', 'bb_exec');
    h += '<div class="card"><div class="cv" style="font-size:12px">'+badgeFor(stats.derived?.bottleneck||'balanced')+'</div><div class="cl">Bottleneck'+infoIcon('bottleneck')+'</div></div>';
    h += card(stats.gmem_bytes||0, 'Global Memory', 'gmem_bytes');
    h += card(stats.smem_bytes||0, 'Shared Memory', 'smem_bytes');
    const spillWarn = (stats.reg_spill_suspected||0)>0;
    h += '<div class="card"><div class="cv" style="color:'+(spillWarn?'var(--red)':'var(--accent)')+'">'+fmt(stats.reg_spill_suspected||0)+'</div>'+
      '<div style="font-size:8px;color:var(--dim);margin-top:1px">'+(spillWarn?'spills':'count')+'</div>'+
      '<div class="cl">'+(spillWarn?'\u26A0 ':'')+'Register Spill'+infoIcon('reg_spill_suspected')+'</div></div>';
    h += card((stats.branch_div_entropy!=null?stats.branch_div_entropy.toFixed(2):'0')+' bits', 'Divergence Entropy', 'branch_div_entropy', {raw:true,noUnit:true});
    h += '</div>';
    h += sectionEnd();

    // Section B: Instruction Class donut
    const ic = stats.inst_class||{};
    const icEntries = Object.entries(ic).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]);
    if (icEntries.length) {
      h += sectionStart('r_instclass','Instruction Class');
      h += '<div id="rMixChart" class="chart" style="height:220px;"></div>';
      h += sectionEnd();
    }

    // Section C: Instruction Pipeline
    const pipe = stats.inst_pipe;
    if (pipe) {
      const pipeEntries = Object.entries(pipe).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]);
      if (pipeEntries.length) {
        h += sectionStart('r_pipe','Instruction Pipeline');
        h += '<div id="rPipeChart" class="chart" style="height:'+Math.max(120,pipeEntries.length*22)+'px;"></div>';
        h += sectionEnd();
      }
    } else {
      h += sectionStart('r_pipe','Instruction Pipeline');
      h += '<div style="color:var(--dim);font-size:11px;padding:6px">Pipeline data requires <code>IKP_NVBIT_ENABLE_INST_PIPE=1</code></div>';
      h += sectionEnd();
    }

    // Section D: Branch Divergence
    const bdh = stats.branch_div_hist||[];
    const bah = stats.branch_active_hist||[];
    const hasBranch = bdh.some(v=>v>0) || bah.some(v=>v>0);
    h += sectionStart('r_branch','Branch Divergence');
    if (hasBranch) {
      h += '<div style="display:flex;gap:8px">';
      h += '<div style="flex:1"><div id="rDivHist" class="chart" style="height:160px;"></div></div>';
      h += '<div style="flex:1"><div id="rActHist" class="chart" style="height:160px;"></div></div>';
      h += '</div>';
      h += '<div style="display:flex;gap:12px;font-size:11px;color:var(--dim);margin-top:4px">';
      h += '<span>Avg active: <b style="color:var(--accent)">'+(stats.branch_active_avg_lanes||0)+'</b> lanes'+infoIcon('branch_active_avg_lanes')+'</span>';
      h += '<span>Div avg: <b style="color:var(--accent)">'+(stats.branch_div_avg_active||0).toFixed(1)+'</b>'+infoIcon('branch_div_avg_active')+'</span>';
      h += '<span>Entropy: <b style="color:var(--accent)">'+(stats.branch_div_entropy||0).toFixed(2)+'</b>'+infoIcon('branch_div_entropy')+'</span>';
      h += '</div>';
    } else {
      h += '<div style="color:var(--dim);font-size:11px;padding:6px">No branch divergence detected \u2014 all warps fully converged</div>';
    }
    h += sectionEnd();

    // Section E: Global Memory Analysis
    h += sectionStart('r_gmem','Global Memory Analysis');
    if ((stats.gmem_bytes||0) > 0) {
      h += '<div class="cards cards-3">';
      h += card(stats.gmem_load||0, 'Load Ops', 'gmem_load');
      h += card(stats.gmem_store||0, 'Store Ops', 'gmem_store');
      h += card(stats.gmem_bytes, 'Transferred', 'gmem_bytes');
      h += card(stats.gmem_req_bytes||0, 'Requested', 'gmem_req_bytes');
      h += card(stats.gmem_sectors_32b||0, '32B Sectors', 'gmem_sectors_32b');
      h += card(stats.gmem_unique_lines_est||0, 'Unique Cache Lines', 'gmem_unique_lines_est');
      h += '</div>';
      // Efficiency bar
      const eff = stats.gmem_bytes>0 ? (stats.gmem_req_bytes||0)/stats.gmem_bytes*100 : 0;
      h += effBar(Math.min(eff,100), 'Coalescing Efficiency');
      // Charts
      h += '<div style="display:flex;gap:8px">';
      const gah = stats.gmem_alignment_hist||[];
      if (gah.some(v=>v>0)) h += '<div style="flex:1"><div id="rGmemAlign" class="chart" style="height:140px;"></div></div>';
      const gsh = stats.gmem_stride_class_hist||[];
      if (gsh.some(v=>v>0)) h += '<div style="flex:1"><div id="rGmemStride" class="chart" style="height:140px;"></div></div>';
      h += '</div>';
      const gsph = stats.gmem_sectors_per_inst_hist||[];
      if (gsph.some(v=>v>0)) h += '<div id="rGmemSectors" class="chart" style="height:140px;"></div>';
    } else {
      h += '<div style="color:var(--dim);font-size:11px;padding:6px">No global memory accesses in this region</div>';
    }
    h += sectionEnd();

    // Section F: Shared Memory Analysis
    h += sectionStart('r_smem','Shared Memory Analysis');
    if ((stats.smem_bytes||0) > 0) {
      h += '<div class="cards cards-4">';
      h += card(stats.smem_load||0, 'Load Ops', 'smem_load');
      h += card(stats.smem_store||0, 'Store Ops', 'smem_store');
      h += card(stats.smem_bytes, 'Transferred', 'smem_bytes');
      h += card(stats.smem_broadcast_count||0, 'Broadcast Ops', 'smem_broadcast_count');
      h += '</div>';
      const bch = stats.smem_bank_conflict_max_hist||[];
      if (bch.some(v=>v>0)) h += '<div id="rSmemBank" class="chart" style="height:160px;"></div>';
      const sah = stats.smem_addr_span_hist||[];
      if (sah.some(v=>v>0)) h += '<div id="rSmemSpan" class="chart" style="height:130px;"></div>';
    } else {
      h += '<div style="color:var(--dim);font-size:11px;padding:6px">No shared memory accesses in this region</div>';
    }
    h += sectionEnd();

    // Section G: Local Memory / Register Spill
    if ((stats.lmem_bytes||0) > 0 || (stats.reg_spill_suspected||0) > 0) {
      h += sectionStart('r_lmem','Local Memory / Register Spill');
      h += '<div class="cards cards-3">';
      h += card(stats.lmem_load||0, 'Load Ops', 'lmem_load');
      h += card(stats.lmem_store||0, 'Store Ops', 'lmem_store');
      h += card(stats.lmem_bytes||0, 'Transferred', 'lmem_bytes');
      h += '</div>';
      if ((stats.reg_spill_suspected||0)>0) {
        h += '<div style="background:#ffebe9;border:1px solid #cf222e;border-radius:6px;padding:8px;margin:6px 0;font-size:11px;color:#cf222e">'+
          '\u26A0 <b>Register spill detected!</b> '+fmt(stats.spill_ld_local_inst||0)+' loads + '+fmt(stats.spill_st_local_inst||0)+
          ' stores to local memory. Consider reducing register pressure.</div>';
      }
      h += sectionEnd();
    }

    // Section H: CUPTI SASS Efficiency (per-region, cross-referenced via pc2region)
    const cuptiR = D.cuptiPerRegion && D.cuptiPerRegion[rid];
    if (cuptiR && cuptiR.efficiency && Object.keys(cuptiR.efficiency).length > 0) {
      h += sectionStart('r_cupti','CUPTI SASS Efficiency (Per-Region)');
      h += '<div style="font-size:10px;color:var(--dim);margin-bottom:6px;padding:4px 8px;background:var(--bg2);border-radius:4px;line-height:1.6">'+
        'CUPTI per-PC SASS metrics cross-referenced with NVBit pc2region mapping. '+
        'Coverage: ~78% of PCs matched. These are <b>per-region</b> values, not whole-kernel.</div>';
      const eff = cuptiR.efficiency;
      const effMetrics = [];
      if (eff.simt_utilization!=null) effMetrics.push({name:'SIMT Utilization',value:eff.simt_utilization,
        desc:'Active threads per warp instruction / 32. Measures lane utilization, NOT SM occupancy. < 100% = divergence or partial warps.'});
      if (eff.predication_eff!=null) effMetrics.push({name:'Predication Efficiency',value:eff.predication_eff,
        desc:'Threads with predicate ON / total thread-instructions.'});
      if (eff.global_coalescing!=null) effMetrics.push({name:'Global Mem Coalescing',value:eff.global_coalescing,
        desc:'Ideal sectors / actual sectors. 100% = perfect coalescing.'});
      if (eff.shared_efficiency!=null) effMetrics.push({name:'Shared Mem Efficiency',value:eff.shared_efficiency,
        desc:'Ideal wavefronts / actual. < 100% = bank conflicts.'});
      if (eff.branch_uniformity!=null) effMetrics.push({name:'Branch Uniformity',value:eff.branch_uniformity,
        desc:'Uniform branches / total branches. Lower = more warp divergence.'});
      if (effMetrics.length) {
        h += '<div id="rCuptiEffChart" class="chart" style="height:'+Math.max(120, effMetrics.length*28)+'px;"></div>';
        h += '<table class="ptable" style="text-align:left;font-size:10px;margin-top:4px">';
        h += '<tr><th style="text-align:left">Metric</th><th>This Region</th>';
        // Add whole-kernel comparison column
        const wk = D.sassProfiles && Object.keys(D.sassProfiles).length > 0;
        if (wk) h += '<th>Whole Kernel</th><th>Delta</th>';
        h += '<th style="text-align:left">Description</th></tr>';
        // Pre-compute whole-kernel efficiencies for comparison
        const wkEff = {};
        if (wk) {
          const sp = D.sassProfiles;
          const wkIe = sp.core?.['smsp__sass_inst_executed'] || sp.divergence?.['smsp__sass_inst_executed'] || 1;
          const wkTie = sp.divergence?.['smsp__sass_thread_inst_executed'] || 0;
          const wkTpon = sp.divergence?.['smsp__sass_thread_inst_executed_pred_on'] || 0;
          if (wkTie > 0) wkEff['SIMT Utilization'] = wkTie / (wkIe * 32);
          if (wkTie > 0 && wkTpon > 0) wkEff['Predication Efficiency'] = wkTpon / wkTie;
          const gS = sp.memory?.['smsp__sass_sectors_mem_global'] || 0;
          const gSi = sp.memory?.['smsp__sass_sectors_mem_global_ideal'] || 0;
          if (gS > 0 && gSi > 0) wkEff['Global Mem Coalescing'] = gSi / gS;
          const sW = sp.memory?.['smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared'] || 0;
          const sWi = sp.memory?.['smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal'] || 0;
          if (sW > 0 && sWi > 0) wkEff['Shared Mem Efficiency'] = sWi / sW;
          const bU = sp.branch?.['smsp__sass_branch_targets_threads_uniform'] || 0;
          const bD = sp.branch?.['smsp__sass_branch_targets_threads_divergent'] || 0;
          if (bU + bD > 0) wkEff['Branch Uniformity'] = bU / (bU + bD);
        }
        for (const m of effMetrics) {
          const vStr = (m.value*100).toFixed(1)+'%';
          const color = m.value >= 0.9 ? 'var(--green)' : m.value >= 0.5 ? 'var(--orange)' : 'var(--red)';
          h += '<tr><td style="text-align:left;font-weight:600">'+m.name+'</td>';
          h += '<td style="color:'+color+';font-weight:600">'+vStr+'</td>';
          if (wk) {
            const wkVal = wkEff[m.name];
            if (wkVal != null) {
              const wkStr = (wkVal*100).toFixed(1)+'%';
              const delta = m.value - wkVal;
              const dColor = delta > 0.01 ? 'var(--green)' : delta < -0.01 ? 'var(--red)' : 'var(--dim)';
              const dStr = (delta>=0?'+':'')+(delta*100).toFixed(1)+'%';
              h += '<td style="color:var(--dim)">'+wkStr+'</td>';
              h += '<td style="color:'+dColor+';font-weight:600">'+dStr+'</td>';
            } else {
              h += '<td style="color:var(--dim)">N/A</td><td>-</td>';
            }
          }
          h += '<td style="text-align:left;color:var(--dim);font-size:9px">'+m.desc+'</td></tr>';
        }
        h += '</table>';
      }
      // ── CUPTI Instruction Mix Breakdown (from breakdown field) ──
      const bk = cuptiR.breakdown || {};
      if (bk.compute_frac != null) {
        h += '<div style="margin-top:10px;font-size:10px;font-weight:600;color:var(--bright)">Instruction Mix (CUPTI, per-region)'+infoIcon('compute_frac')+'</div>';
        h += '<div id="rCuptiMixChart" class="chart" style="height:180px;"></div>';
        // Breakdown cards
        h += '<div class="cards cards-4" style="margin-top:4px">';
        const fmtP = v => v!=null ? (v*100).toFixed(1)+'%' : '-';
        h += card(fmtP(bk.compute_frac), 'Compute'+infoIcon('compute_frac'), null, {raw:true,noUnit:true});
        h += card(fmtP(bk.global_mem_frac), 'Global Mem'+infoIcon('global_mem_frac'), null, {raw:true,noUnit:true});
        h += card(fmtP(bk.shared_mem_frac), 'Shared Mem'+infoIcon('shared_mem_frac'), null, {raw:true,noUnit:true});
        h += card(fmtP(bk.branch_frac), 'Branch'+infoIcon('branch_frac'), null, {raw:true,noUnit:true});
        h += '</div>';
        if (bk.tma_frac != null || bk.tensor_frac != null) {
          h += '<div class="cards cards-2" style="margin-top:2px">';
          if (bk.tma_frac != null) h += card(fmtP(bk.tma_frac), 'TMA'+infoIcon('tma_frac'), null, {raw:true,noUnit:true});
          if (bk.tensor_frac != null) h += card(fmtP(bk.tensor_frac), 'Tensor/WGMMA'+infoIcon('tensor_frac'), null, {raw:true,noUnit:true});
          h += '</div>';
        }
      }

      // ── Memory Detail (per-region, CUPTI) ──
      if (bk.global_load_frac != null || bk.shared_load_frac != null) {
        h += '<div style="margin-top:10px;font-size:10px;font-weight:600;color:var(--bright)">Memory Detail (CUPTI, per-region)</div>';
        h += '<table class="ptable" style="text-align:left;font-size:10px;margin-top:4px">';
        h += '<tr><th style="text-align:left">Metric</th><th>Value</th><th style="text-align:left">Explanation</th></tr>';
        if (bk.global_load_frac != null) {
          h += '<tr><td style="text-align:left">Global Load Ratio'+infoIcon('global_load_frac')+'</td>';
          h += '<td style="font-weight:600">'+(bk.global_load_frac*100).toFixed(1)+'% loads / '+(100-bk.global_load_frac*100).toFixed(1)+'% stores</td>';
          h += '<td style="text-align:left;color:var(--dim);font-size:9px">'+
            (bk.global_load_frac > 0.9 ? 'Read-dominated (typical for loading tiles)' :
             bk.global_load_frac < 0.1 ? 'Write-dominated (store-back phase)' : 'Mixed read/write')+'</td></tr>';
        }
        if (bk.sectors_per_global_inst != null) {
          const spi = bk.sectors_per_global_inst;
          const spiIdeal = bk.sectors_per_global_inst_ideal || 0;
          const spiColor = spi <= 4.5 ? 'var(--green)' : spi <= 8 ? 'var(--orange)' : 'var(--red)';
          h += '<tr><td style="text-align:left">Sectors/Global Inst'+infoIcon('sectors_per_global_inst')+'</td>';
          h += '<td><span style="color:'+spiColor+';font-weight:600">'+spi.toFixed(2)+'</span>';
          if (spiIdeal > 0) h += ' <span style="color:var(--dim);font-size:9px">(ideal: '+spiIdeal.toFixed(2)+')</span>';
          h += '</td>';
          h += '<td style="text-align:left;color:var(--dim);font-size:9px">Ideal = 4 for 32 threads \u00d7 4B = 128B per cache line. Higher = poor coalescing.</td></tr>';
        }
        if (bk.shared_load_frac != null) {
          h += '<tr><td style="text-align:left">Shared Load Ratio'+infoIcon('shared_load_frac')+'</td>';
          h += '<td style="font-weight:600">'+(bk.shared_load_frac*100).toFixed(1)+'% loads / '+(100-bk.shared_load_frac*100).toFixed(1)+'% stores</td>';
          h += '<td style="text-align:left;color:var(--dim);font-size:9px">Shared memory load vs store balance</td></tr>';
        }
        if (bk.wavefronts_per_shared_inst != null) {
          const wpi = bk.wavefronts_per_shared_inst;
          const wpiColor = wpi <= 1.05 ? 'var(--green)' : wpi <= 2 ? 'var(--orange)' : 'var(--red)';
          h += '<tr><td style="text-align:left">Wavefronts/Shared Inst'+infoIcon('wavefronts_per_shared_inst')+'</td>';
          h += '<td><span style="color:'+wpiColor+';font-weight:600">'+wpi.toFixed(2)+'</span></td>';
          h += '<td style="text-align:left;color:var(--dim);font-size:9px">1.0 = no bank conflicts. >1.0 = replayed due to bank conflicts.</td></tr>';
        }
        if (bk.branch_avg_active_lanes != null) {
          const bal = bk.branch_avg_active_lanes;
          const balColor = bal >= 30 ? 'var(--green)' : bal >= 16 ? 'var(--orange)' : 'var(--red)';
          h += '<tr><td style="text-align:left">Branch Avg Active Lanes'+infoIcon('branch_avg_active_lanes')+'</td>';
          h += '<td><span style="color:'+balColor+';font-weight:600">'+bal.toFixed(1)+'</span> / 32 ('+
            (bk.branch_lane_utilization != null ? (bk.branch_lane_utilization*100).toFixed(1)+'%' : '-')+')</td>';
          h += '<td style="text-align:left;color:var(--dim);font-size:9px">Thread-level branch activity from CUPTI. 32 = all threads take same path.</td></tr>';
        }
        h += '</table>';
      }

      // ── NVBit vs CUPTI Cross-Validation (instruction mix fractions) ──
      const cv = D.crossValidation && D.crossValidation[rid];
      if (cv && cv.fractions) {
        h += '<div style="margin-top:10px;font-size:10px;font-weight:600;color:var(--bright)">NVBit \u00d7 CUPTI Instruction Mix Cross-Validation'+infoIcon('nvbit_vs_cupti')+'</div>';
        h += '<div style="font-size:9px;color:var(--dim);margin:2px 0 4px">Compares instruction mix fractions (%) between NVBit binary instrumentation and CUPTI hardware counters. Fractions are invocation-independent.</div>';
        h += '<table class="ptable" style="text-align:left;font-size:10px;margin-top:4px">';
        h += '<tr><th style="text-align:left">Category</th><th style="text-align:right">NVBit</th><th style="text-align:right">CUPTI</th><th style="text-align:right">Delta</th><th style="min-width:100px">Visual</th></tr>';
        const catLabels = {compute:'Compute',global_mem:'Global Mem',shared_mem:'Shared Mem',branch:'Branch'};
        const catColors = {compute:'#1a7f37',global_mem:'#bc4c00',shared_mem:'#0969da',branch:'#8250df'};
        for (const [cat, label] of Object.entries(catLabels)) {
          const f = cv.fractions[cat];
          if (!f) continue;
          const dColor = f.delta < 0.02 ? 'var(--green)' : f.delta < 0.1 ? 'var(--orange)' : 'var(--red)';
          h += '<tr>';
          h += '<td style="text-align:left;font-weight:600"><span style="color:'+catColors[cat]+'">\u25CF</span> '+label+'</td>';
          h += '<td style="text-align:right">'+(f.nvbit*100).toFixed(1)+'%</td>';
          h += '<td style="text-align:right">'+(f.cupti*100).toFixed(1)+'%</td>';
          h += '<td style="text-align:right;color:'+dColor+';font-weight:600">'+(f.delta*100).toFixed(1)+'pp</td>';
          // Visual: two overlapping bars
          h += '<td><div style="position:relative;height:12px;background:var(--bg3);border-radius:3px;overflow:hidden">';
          h += '<div style="position:absolute;height:6px;top:0;left:0;width:'+(f.nvbit*100).toFixed(1)+'%;background:'+catColors[cat]+';opacity:0.7;border-radius:3px 3px 0 0" title="NVBit"></div>';
          h += '<div style="position:absolute;height:6px;bottom:0;left:0;width:'+(f.cupti*100).toFixed(1)+'%;background:'+catColors[cat]+';opacity:0.4;border-radius:0 0 3px 3px" title="CUPTI"></div>';
          h += '</div></td>';
          h += '</tr>';
        }
        const avgD = cv.avg_delta||0;
        const avgColor = avgD < 0.02 ? 'var(--green)' : avgD < 0.1 ? 'var(--orange)' : 'var(--red)';
        h += '<tr style="border-top:1px solid var(--border)"><td style="text-align:left;font-weight:600" colspan=3>Average Delta</td>';
        h += '<td style="text-align:right;color:'+avgColor+';font-weight:600">'+(avgD*100).toFixed(1)+'pp</td>';
        h += '<td style="color:'+avgColor+';font-size:9px">'+(avgD<0.02?'Excellent agreement':avgD<0.1?'Good agreement':'Investigate gaps')+'</td></tr>';
        h += '</table>';
      }

      // Show raw CUPTI metric counts for this region
      const rawProfiles = cuptiR.raw;
      if (rawProfiles) {
        h += '<details style="margin-top:6px"><summary style="font-size:10px;color:var(--dim);cursor:pointer">Raw CUPTI metric counts for this region</summary>';
        h += '<table class="ptable" style="text-align:left;font-size:9px;margin-top:4px">';
        h += '<tr><th style="text-align:left">Profile</th><th style="text-align:left">Metric</th><th>Value</th></tr>';
        for (const [prof, metrics] of Object.entries(rawProfiles)) {
          for (const [mname, mval] of Object.entries(metrics)) {
            h += '<tr><td style="text-align:left;color:var(--dim)">'+prof+'</td>';
            h += '<td style="text-align:left;font-family:monospace;font-size:9px">'+mname+'</td>';
            h += '<td>'+fmt(mval)+'</td></tr>';
          }
        }
        h += '</table></details>';
      }
      h += sectionEnd();
    }

    // Section I: Per-Region Stall Profile (from PC sampling)
    const pcsampR = D.pcsampPerRegion && D.pcsampPerRegion[rid];
    if (pcsampR && Object.keys(pcsampR).length > 0) {
      h += sectionStart('r_stalls','Per-Region Stall Profile (PC Sampling)');
      h += '<div style="font-size:10px;color:var(--dim);margin-bottom:6px;padding:4px 8px;background:var(--bg2);border-radius:4px;line-height:1.6">'+
        'PC sampling stall reasons attributed to this region via pc2region mapping.</div>';
      h += '<div id="rStallChart" class="chart" style="height:'+Math.max(120, Object.keys(pcsampR).length*22)+'px;"></div>';
      h += sectionEnd();
    }

    // Section J: Per-Region InstrExec
    const instrR = D.instrexecPerRegion && D.instrexecPerRegion[rid];
    if (instrR) {
      h += sectionStart('r_instrexec','Instruction Execution (CUPTI)');
      h += '<div style="font-size:10px;color:var(--dim);margin-bottom:6px;padding:4px 8px;background:var(--bg2);border-radius:4px;line-height:1.6">'+
        'CUPTI instrexec data attributed to this region via source line mapping.</div>';
      h += '<div class="cards cards-4">';
      h += card(instrR.threads_executed||0, 'Threads Executed', 'threads_executed');
      h += card(instrR.executed||0, 'Warp Executions', 'executed');
      h += card(instrR.not_pred_off||0, 'Pred-Off Threads', 'notPredOffThreadsExecuted');
      h += card(instrR.inst_count||0, 'Unique Instructions', null, {noUnit:true});
      h += '</div>';
      if (instrR.executed > 0 && instrR.threads_executed > 0) {
        const tUtil = instrR.threads_executed / (instrR.executed * 32);
        const tColor = tUtil >= 0.9 ? 'var(--green)' : tUtil >= 0.5 ? 'var(--orange)' : 'var(--red)';
        h += '<div style="font-size:11px;margin-top:4px">SIMT Utilization: <b style="color:'+tColor+'">'+(tUtil*100).toFixed(1)+'%</b>'+
          infoIcon('simt_utilization')+
          ' <span style="color:var(--dim);font-size:9px">(threads_executed / (warp_executed \u00d7 32))</span></div>';
        // Predication efficiency from instrexec
        if (instrR.not_pred_off > 0) {
          const pEff = instrR.not_pred_off / instrR.threads_executed;
          const pColor = pEff >= 0.9 ? 'var(--green)' : pEff >= 0.5 ? 'var(--orange)' : 'var(--red)';
          h += '<div style="font-size:11px;margin-top:2px">Predication Overhead: <b style="color:'+pColor+'">'+(pEff*100).toFixed(1)+'%</b> pred-off'+
            ' <span style="color:var(--dim);font-size:9px">(not_pred_off / threads_executed)</span></div>';
        }
      }
      h += sectionEnd();
    }

    det.innerHTML = h;

    // ── Render ECharts for region detail ──

    // Instruction class donut
    if (icEntries.length) {
      const ch = mkChart('rMixChart');
      ch.setOption({
        backgroundColor:'transparent',
        tooltip:{trigger:'item',formatter:function(p){
          const desc=D.defs[p.name]?.long||'';
          return '<b>'+p.name+'</b>: '+fmt(p.value)+' ('+p.percent+'%)'+(desc?'<br><span style="font-size:10px;color:#999">'+desc+'</span>':'');
        }},
        series:[{type:'pie',radius:['35%','65%'],
          data:icEntries.map(([n,v])=>({name:n,value:v,itemStyle:{color:INST_CLASS_COLORS[n]||'#8b949e'}})),
          label:{color:'#656d76',fontSize:10,formatter:'{b}\n{d}%'},
          itemStyle:{borderColor:'#ffffff',borderWidth:2},
          emphasis:{label:{fontSize:12,fontWeight:'bold'}},
        }],
      });
    }

    // Pipeline horizontal bar
    if (pipe) {
      const pipeEntries = Object.entries(pipe).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]);
      if (pipeEntries.length) {
        const ch = mkChart('rPipeChart');
        ch.setOption({
          backgroundColor:'transparent',
          grid:{left:10,right:30,top:5,bottom:5,containLabel:true},
          yAxis:{type:'category',data:pipeEntries.map(([n])=>n).reverse(),axisLabel:{color:'#656d76',fontSize:9}},
          xAxis:{type:'value',name:'Instructions',nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:9,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:pipeEntries.map(([,v])=>v).reverse(),
            itemStyle:{color:'#0969da',borderRadius:[0,3,3,0]},barWidth:'60%'}],
          tooltip:{trigger:'axis',formatter:function(p){
            const desc=D.defs['inst_pipe']?.long||'';
            return p[0].name+': '+fmt(p[0].value)+(desc?'<br><span style="font-size:10px;color:#999">'+desc+'</span>':'');
          }},
        });
      }
    }

    // Branch divergence histograms
    if (hasBranch) {
      const lanes = Array.from({length:33},(_,i)=>i);
      if (bdh.some(v=>v>0)) {
        const ch = mkChart('rDivHist');
        ch.setOption({
          backgroundColor:'transparent',
          title:{text:'Divergent Branches',textStyle:{color:'#656d76',fontSize:11},left:'center',top:0},
          grid:{left:10,right:10,top:30,bottom:20,containLabel:true},
          xAxis:{type:'category',data:lanes,name:'Active lanes',nameLocation:'center',nameGap:18,
            nameTextStyle:{color:'#656d76',fontSize:9},axisLabel:{color:'#656d76',fontSize:8}},
          yAxis:{type:'value',name:'Warps',nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:8,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:bdh,itemStyle:{color:function(p){return p.dataIndex===32?'#1a7f37':'#bc4c00';}},barWidth:'80%'}],
          tooltip:{trigger:'axis',formatter:p=>p[0].dataIndex+' active lanes: '+fmt(p[0].value)+' warps'},
        });
      }
      if (bah.some(v=>v>0)) {
        const ch = mkChart('rActHist');
        ch.setOption({
          backgroundColor:'transparent',
          title:{text:'All Branches',textStyle:{color:'#656d76',fontSize:11},left:'center',top:0},
          grid:{left:10,right:10,top:30,bottom:20,containLabel:true},
          xAxis:{type:'category',data:lanes,name:'Active lanes',nameLocation:'center',nameGap:18,
            nameTextStyle:{color:'#656d76',fontSize:9},axisLabel:{color:'#656d76',fontSize:8}},
          yAxis:{type:'value',name:'Executions',nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:8,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:bah,itemStyle:{color:function(p){return p.dataIndex===32?'#1a7f37':'#0969da';}},barWidth:'80%'}],
          tooltip:{trigger:'axis',formatter:p=>p[0].dataIndex+' active lanes: '+fmt(p[0].value)+' executions'},
        });
      }
    }

    // Global memory charts
    if ((stats.gmem_bytes||0)>0) {
      const gah = stats.gmem_alignment_hist||[];
      if (gah.some(v=>v>0)) {
        const ch = mkChart('rGmemAlign');
        const alLabels = ['Aligned','Off+4B','Off+8B','Off+12B','Off+16B','Off+20B','Off+24B','Off+28B'];
        ch.setOption({
          backgroundColor:'transparent',
          title:{text:'Alignment',textStyle:{color:'#656d76',fontSize:10},left:'center',top:0},
          grid:{left:5,right:5,top:25,bottom:15,containLabel:true},
          xAxis:{type:'category',data:alLabels.slice(0,gah.length),axisLabel:{color:'#656d76',fontSize:8,rotate:30}},
          yAxis:{type:'value',axisLabel:{color:'#656d76',fontSize:8,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:gah,itemStyle:{color:function(p){return p.dataIndex===0?'#1a7f37':'#bc4c00';}},barWidth:'60%'}],
          tooltip:{trigger:'axis',formatter:p=>alLabels[p[0].dataIndex]+': '+fmt(p[0].value)},
        });
      }
      const gsh = stats.gmem_stride_class_hist||[];
      if (gsh.some(v=>v>0)) {
        const ch = mkChart('rGmemStride');
        const strideLabels = ['Sequential','Strided','Random'];
        const strideColors = ['#1a7f37','#9a6700','#cf222e'];
        ch.setOption({
          backgroundColor:'transparent',
          tooltip:{trigger:'item',formatter:'{b}: {c} ({d}%)'},
          series:[{type:'pie',radius:['30%','60%'],
            data:gsh.map((v,i)=>({name:strideLabels[i]||'bin_'+i,value:v,itemStyle:{color:strideColors[i]||'#8b949e'}})),
            label:{color:'#656d76',fontSize:10,formatter:'{b}\n{d}%'},
            itemStyle:{borderColor:'#fff',borderWidth:2}}],
        });
      }
      const gsph = stats.gmem_sectors_per_inst_hist||[];
      if (gsph.some(v=>v>0)) {
        const ch = mkChart('rGmemSectors');
        ch.setOption({
          backgroundColor:'transparent',
          title:{text:'Sectors per Instruction',textStyle:{color:'#656d76',fontSize:10},left:'center',top:0},
          grid:{left:10,right:10,top:25,bottom:20,containLabel:true},
          xAxis:{type:'category',data:Array.from({length:gsph.length},(_,i)=>i),name:'Sectors/inst',
            nameLocation:'center',nameGap:18,nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:8}},
          yAxis:{type:'value',name:'Count',nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:8,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:gsph,itemStyle:{color:function(p){return p.dataIndex===4?'#1a7f37':'#0969da';}},barWidth:'80%'}],
          tooltip:{trigger:'axis',formatter:function(p){
            const ideal = p[0].dataIndex===4?' (ideal for 128B)':'';
            return p[0].dataIndex+' sectors: '+fmt(p[0].value)+ideal;
          }},
        });
      }
    }

    // Shared memory charts
    if ((stats.smem_bytes||0)>0) {
      const bch = stats.smem_bank_conflict_max_hist||[];
      if (bch.some(v=>v>0)) {
        const ch = mkChart('rSmemBank');
        ch.setOption({
          backgroundColor:'transparent',
          title:{text:'Bank Conflict Distribution',textStyle:{color:'#656d76',fontSize:10},left:'center',top:0},
          grid:{left:10,right:10,top:25,bottom:20,containLabel:true},
          xAxis:{type:'category',data:Array.from({length:bch.length},(_,i)=>i),name:'Max conflict ways',
            nameLocation:'center',nameGap:18,nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:8}},
          yAxis:{type:'value',name:'Count',nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:8,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:bch,itemStyle:{color:function(p){
            const i=p.dataIndex; return i<=1?'#1a7f37':i<=4?'#9a6700':'#cf222e';
          }},barWidth:'80%'}],
          tooltip:{trigger:'axis',formatter:function(p){
            const ways=p[0].dataIndex;
            return ways+'-way conflict: '+fmt(p[0].value)+' operations'+(ways<=1?' (no conflict)':'');
          }},
        });
      }
      const sah = stats.smem_addr_span_hist||[];
      if (sah.some(v=>v>0)) {
        const ch = mkChart('rSmemSpan');
        ch.setOption({
          backgroundColor:'transparent',
          title:{text:'Address Span',textStyle:{color:'#656d76',fontSize:10},left:'center',top:0},
          grid:{left:10,right:10,top:25,bottom:15,containLabel:true},
          xAxis:{type:'category',data:Array.from({length:sah.length},(_,i)=>'Bin '+i),axisLabel:{color:'#656d76',fontSize:8}},
          yAxis:{type:'value',axisLabel:{color:'#656d76',fontSize:8,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:sah,itemStyle:{color:'#0550ae'},barWidth:'60%'}],
          tooltip:{trigger:'axis'},
        });
      }
    }

    // CUPTI efficiency bar chart (Section H)
    const cuptiR2 = D.cuptiPerRegion && D.cuptiPerRegion[rid];
    if (cuptiR2 && cuptiR2.efficiency) {
      const eff = cuptiR2.efficiency;
      const items = [];
      if (eff.simt_utilization!=null) items.push({n:'SIMT Utilization',v:eff.simt_utilization});
      if (eff.predication_eff!=null) items.push({n:'Predication Eff.',v:eff.predication_eff});
      if (eff.global_coalescing!=null) items.push({n:'Global Coalescing',v:eff.global_coalescing});
      if (eff.shared_efficiency!=null) items.push({n:'Shared Mem Eff.',v:eff.shared_efficiency});
      if (eff.branch_uniformity!=null) items.push({n:'Branch Uniformity',v:eff.branch_uniformity});
      if (items.length) {
        const ch = mkChart('rCuptiEffChart');
        if (ch) {
          ch.setOption({
            backgroundColor:'transparent',
            grid:{left:10,right:40,top:5,bottom:5,containLabel:true},
            yAxis:{type:'category',data:items.map(i=>i.n).reverse(),axisLabel:{color:'#656d76',fontSize:9}},
            xAxis:{type:'value',min:0,max:1,axisLabel:{color:'#656d76',fontSize:9,formatter:v=>(v*100)+'%'},
              splitLine:{lineStyle:{color:'#d0d7de'}}},
            series:[{type:'bar',data:items.map(i=>i.v).reverse(),
              label:{show:true,position:'right',fontSize:9,color:'#656d76',formatter:p=>(p.value*100).toFixed(1)+'%'},
              itemStyle:{color:function(p){const v=p.value;return v>=0.9?'#1a7f37':v>=0.5?'#bc4c00':'#cf222e';},borderRadius:[0,3,3,0]},barWidth:'55%'}],
            tooltip:{trigger:'axis',formatter:function(p){return p[0].name+': '+(p[0].value*100).toFixed(1)+'%';}},
          });
        }
      }
    }

    // CUPTI Instruction Mix donut (Section H breakdown)
    const bk2 = cuptiR2 && cuptiR2.breakdown;
    if (bk2 && bk2.compute_frac != null) {
      const ch = mkChart('rCuptiMixChart');
      if (ch) {
        const mixItems = [];
        if (bk2.compute_frac > 0) mixItems.push({name:'Compute',value:+(bk2.compute_frac*100).toFixed(1),itemStyle:{color:'#1a7f37'}});
        if (bk2.global_mem_frac > 0) mixItems.push({name:'Global Mem',value:+(bk2.global_mem_frac*100).toFixed(1),itemStyle:{color:'#bc4c00'}});
        if (bk2.shared_mem_frac > 0) mixItems.push({name:'Shared Mem',value:+(bk2.shared_mem_frac*100).toFixed(1),itemStyle:{color:'#0969da'}});
        if (bk2.branch_frac > 0) mixItems.push({name:'Branch',value:+(bk2.branch_frac*100).toFixed(1),itemStyle:{color:'#8250df'}});
        if (bk2.tma_frac > 0) mixItems.push({name:'TMA',value:+(bk2.tma_frac*100).toFixed(1),itemStyle:{color:'#cf222e'}});
        if (bk2.tensor_frac > 0) mixItems.push({name:'Tensor/WGMMA',value:+(bk2.tensor_frac*100).toFixed(1),itemStyle:{color:'#6639ba'}});
        ch.setOption({
          backgroundColor:'transparent',
          tooltip:{trigger:'item',formatter:function(p){
            const defKey = {'Compute':'compute_frac','Global Mem':'global_mem_frac','Shared Mem':'shared_mem_frac',
              'Branch':'branch_frac','TMA':'tma_frac','Tensor/WGMMA':'tensor_frac'}[p.name]||'';
            const desc = D.defs[defKey]?.long||'';
            return '<b>'+p.name+'</b>: '+p.value.toFixed(1)+'%'+(desc?'<br><span style="font-size:10px;color:#999">'+desc+'</span>':'');
          }},
          legend:{bottom:0,textStyle:{color:'#656d76',fontSize:9},itemWidth:10,itemHeight:10},
          series:[{type:'pie',radius:['30%','60%'],center:['50%','45%'],
            data:mixItems,
            label:{color:'#656d76',fontSize:10,formatter:'{b}\n{d}%'},
            emphasis:{itemStyle:{shadowBlur:8,shadowColor:'rgba(0,0,0,0.2)'}},
            itemStyle:{borderColor:'#ffffff',borderWidth:2}}],
        });
      }
    }

    // Per-region stall chart (Section I)
    const pcsampR2 = D.pcsampPerRegion && D.pcsampPerRegion[rid];
    if (pcsampR2 && Object.keys(pcsampR2).length > 0) {
      const ch = mkChart('rStallChart');
      if (ch) {
        const entries = Object.entries(pcsampR2).sort((a,b)=>b[1]-a[1]);
        ch.setOption({
          backgroundColor:'transparent',
          grid:{left:10,right:30,top:5,bottom:5,containLabel:true},
          yAxis:{type:'category',data:entries.map(([n])=>n.replace(/_/g,' ')).reverse(),axisLabel:{color:'#656d76',fontSize:9}},
          xAxis:{type:'value',name:'Samples',nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:9,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:entries.map(([,v])=>v).reverse(),
            itemStyle:{color:'#bc4c00',borderRadius:[0,3,3,0]},barWidth:'60%'}],
          tooltip:{trigger:'axis',formatter:function(p){
            const name=p[0].name;
            const desc=D.defs[name.replace(/ /g,'_')]?.long||'';
            return '<b>'+name+'</b>: '+fmt(p[0].value)+' samples'+(desc?'<br><span style="font-size:10px;color:#999">'+desc+'</span>':'');
          }},
        });
      }
    }
  };

  // ══════════════════════════════════════════════════════════════════
  // TAB 4: EXECUTION
  // ══════════════════════════════════════════════════════════════════
  function buildExecution() {
    const ct = document.getElementById('execCt');
    let h = '';
    const rids = Object.keys(D.regions).map(Number).sort();

    // Section A: Instruction Mix Comparison (stacked bar)
    const hasIc = rids.some(rid=>D.regions[rid]?.inst_class && Object.values(D.regions[rid].inst_class).some(v=>v>0));
    if (hasIc) {
      h += sectionStart('ex_mix','Instruction Mix Comparison');
      h += '<div id="exMixChart" class="chart" style="height:300px;"></div>';
      h += sectionEnd();
    }

    // Section B: Basic Block Hotspots
    h += sectionStart('ex_bb','Basic Block Hotspots');
    if (D.hotspots?.bbs?.length) {
      h += '<div id="exBbChart" class="chart" style="height:220px;"></div>';
      // Detailed table with source mapping
      h += '<table class="ptable" style="text-align:left;font-size:10px;margin-top:6px">';
      h += '<tr><th style="text-align:left">BB</th><th style="text-align:left">Source Line</th>'+
        '<th style="text-align:left">Region</th><th>Instrs</th><th>Exec Count</th>'+
        '<th>Total Inst Exec</th><th style="text-align:left">PC</th></tr>';
      for (const bb of D.hotspots.bbs.slice(0,20)) {
        const srcLine = bb.source_line;
        const rlabel = bb.region_label;
        const rid = bb.region;
        const rcolor = rid!=null ? (D.colors[rid]||'#8b949e') : '#8b949e';
        h += '<tr>';
        h += '<td style="text-align:left;font-weight:600">BB'+bb.bb_id+'</td>';
        // Source line — clickable to navigate
        if (srcLine) {
          h += '<td style="text-align:left"><a href="#" onclick="srcEditor.revealLineInCenter('+srcLine+');srcEditor.setPosition({lineNumber:'+srcLine+',column:1});return false;" style="color:var(--accent);text-decoration:none;font-family:monospace" title="Click to navigate to source">L'+srcLine+'</a></td>';
        } else {
          h += '<td style="text-align:left;color:var(--dim)">-</td>';
        }
        // Region badge
        if (rlabel) {
          h += '<td style="text-align:left"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:'+rcolor+';margin-right:3px;vertical-align:middle"></span>'+rlabel+'</td>';
        } else {
          h += '<td style="text-align:left;color:var(--dim)">outside</td>';
        }
        h += '<td>'+fmt(bb.n_instrs||0)+'</td>';
        h += '<td>'+fmt(bb.exec_count||0)+'</td>';
        h += '<td>'+fmt(bb.total_inst_exec||0)+'</td>';
        h += '<td style="text-align:left;font-family:monospace;color:var(--dim)">0x'+(bb.entry_pc||0).toString(16)+'</td>';
        h += '</tr>';
      }
      h += '</table>';
    } else {
      h += '<div style="color:var(--dim);font-size:11px;padding:6px">BB hotspot data requires NVBit mode=bb_hot with <code>IKP_NVBIT_ENABLE_BB_HOT=1</code></div>';
    }
    h += sectionEnd();

    // Section C: Branch Site Analysis
    h += sectionStart('ex_branch','Branch Site Analysis');
    if (D.hotspots?.branches?.length) {
      h += '<table class="ptable" style="text-align:left;font-size:10px">';
      h += '<tr><th style="text-align:left">PC</th><th style="text-align:left">Opcode</th>'+
        '<th style="text-align:left">Source</th><th style="text-align:left">Region</th><th>Exec</th>'+
        '<th>Taken(W)</th><th>Fall(W)</th><th>Taken(T)</th><th>Fall(T)</th><th style="text-align:left">Div?</th></tr>';
      for (const br of D.hotspots.branches) {
        const pc = '0x'+(br.pc_offset||br.pc||0).toString(16);
        const tw = br.taken_warp||0, fw = br.fallthrough_warp||0;
        const tl = br.taken_lanes||0, fl = br.fallthrough_lanes||0;
        const totalW = tw+fw;
        const isDivergent = totalW>0 && tw>0 && fw>0;
        const takenPct = totalW>0?(tw/totalW*100).toFixed(0):'0';
        const brSrcLine = br.source_line;
        const brRid = br.region;
        const brRlabel = br.region_label;
        const brRcolor = brRid!=null ? (D.colors[brRid]||'#8b949e') : '#8b949e';
        h += '<tr><td style="text-align:left;font-family:monospace">'+pc+'</td>';
        h += '<td style="text-align:left">'+(br.opcode||'?')+'</td>';
        // Source line
        if (brSrcLine) {
          h += '<td style="text-align:left"><a href="#" onclick="srcEditor.revealLineInCenter('+brSrcLine+');srcEditor.setPosition({lineNumber:'+brSrcLine+',column:1});return false;" style="color:var(--accent);text-decoration:none;font-family:monospace">L'+brSrcLine+'</a></td>';
        } else {
          h += '<td style="text-align:left;color:var(--dim)">-</td>';
        }
        // Region
        if (brRlabel) {
          h += '<td style="text-align:left"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:'+brRcolor+';margin-right:3px;vertical-align:middle"></span>'+brRlabel+'</td>';
        } else {
          h += '<td style="text-align:left;color:var(--dim)">outside</td>';
        }
        h += '<td>'+fmt(br.exec_count||0)+'</td>';
        h += '<td>'+fmt(tw)+'</td><td>'+fmt(fw)+'</td>';
        h += '<td>'+fmt(tl)+'</td><td>'+fmt(fl)+'</td>';
        h += '<td style="text-align:left">'+(isDivergent?'<span style="color:var(--red)">\u26A0 Yes ('+takenPct+'% taken)</span>':'<span style="color:var(--green)">\u2713 No</span>')+'</td>';
        h += '</tr>';
        // Mini stacked bar
        if (totalW>0) {
          h += '<tr><td colspan=10 style="padding:0 0 4px 0"><div style="display:flex;height:8px;border-radius:3px;overflow:hidden">'+
            '<div style="width:'+takenPct+'%;background:var(--green)"></div>'+
            '<div style="width:'+(100-parseFloat(takenPct))+'%;background:var(--orange)"></div></div></td></tr>';
        }
      }
      h += '</table>';
    } else {
      h += '<div style="color:var(--dim);font-size:11px;padding:6px">No branch sites recorded</div>';
    }
    h += sectionEnd();

    ct.innerHTML = h;

    // Charts
    if (hasIc) {
      const ch = mkChart('exMixChart');
      // Collect all classes
      const allClasses = new Set();
      for (const rid of rids) {
        const ic = D.regions[rid]?.inst_class||{};
        Object.keys(ic).forEach(k=>{if(ic[k]>0)allClasses.add(k);});
      }
      const classes = [...allClasses];
      const regionNames = rids.map(rid=>D.regions[rid]?.label||'R'+rid);
      ch.setOption({
        backgroundColor:'transparent',
        legend:{top:0,textStyle:{color:'#656d76',fontSize:9},type:'scroll',padding:[0,0,5,0]},
        grid:{left:10,right:10,top:55,bottom:20,containLabel:true},
        xAxis:{type:'category',data:regionNames,axisLabel:{color:'#656d76',fontSize:10}},
        yAxis:{type:'value',name:'% of Instructions',nameTextStyle:{color:'#656d76',fontSize:9},
          axisLabel:{color:'#656d76',fontSize:9,formatter:v=>v+'%'},splitLine:{lineStyle:{color:'#d0d7de'}},max:100},
        series:classes.map(cls=>({
          name:cls,type:'bar',stack:'total',
          data:rids.map(rid=>{
            const ic=D.regions[rid]?.inst_class||{};
            const total=D.regions[rid]?.inst_total||1;
            return parseFloat(((ic[cls]||0)/total*100).toFixed(2));
          }),
          itemStyle:{color:INST_CLASS_COLORS[cls]||'#8b949e'},
          emphasis:{focus:'series'},
        })),
        tooltip:{trigger:'axis',formatter:function(params){
          let s = params[0].axisValue+'<br>';
          for(const p of params) if(p.value>0) s += '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:'+p.color+';margin-right:4px"></span>'+p.seriesName+': '+p.value.toFixed(1)+'%<br>';
          return s;
        }},
      });
    }

    if (D.hotspots?.bbs?.length) {
      const bbs = D.hotspots.bbs.slice(0,20);
      const ch = mkChart('exBbChart');
      // Color each bar by region
      const barColors = bbs.map(b=>{
        const rid = b.region;
        return rid!=null ? (D.colors[rid]||'#8b949e') : '#8b949e';
      });
      ch.setOption({
        backgroundColor:'transparent',
        grid:{left:10,right:10,top:10,bottom:20,containLabel:true},
        xAxis:{type:'category',data:bbs.map(b=>{
          const sl = b.source_line ? ' (L'+b.source_line+')' : '';
          return 'BB'+b.bb_id+sl;
        }),axisLabel:{color:'#656d76',fontSize:9,rotate:bbs.length>10?30:0}},
        yAxis:{type:'value',name:'Warp Executions',nameTextStyle:{color:'#656d76',fontSize:9},
          axisLabel:{color:'#656d76',fontSize:9,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
        series:[{type:'bar',data:bbs.map((b,i)=>({value:b.exec_count,itemStyle:{color:barColors[i],borderRadius:[3,3,0,0]}})),}],
        tooltip:{trigger:'axis',formatter:function(p){
          const bb = bbs[p[0].dataIndex];
          let s = '<b>BB '+bb.bb_id+'</b><br>';
          if (bb.source_line) s += 'Source: line '+bb.source_line+'<br>';
          if (bb.region_label) s += 'Region: '+bb.region_label+'<br>';
          s += 'Executions: '+fmt(bb.exec_count)+'<br>';
          s += 'Instructions: '+fmt(bb.n_instrs||0)+'<br>';
          s += 'Total inst executed: '+fmt(bb.total_inst_exec||0);
          return s;
        }},
      });
      // Click on bar → navigate to source line
      ch.on('click',function(p){
        const bb = bbs[p.dataIndex];
        if (bb.source_line) {
          srcEditor.revealLineInCenter(bb.source_line);
          srcEditor.setPosition({lineNumber:bb.source_line,column:1});
        }
      });
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // TAB 5: MEMORY
  // ══════════════════════════════════════════════════════════════════
  function buildMemory() {
    const ct = document.getElementById('memCt');
    let h = '';
    const rids = Object.keys(D.regions).map(Number).sort();

    // Memory Traffic Overview
    let totalGmem=0,totalSmem=0,totalLmem=0;
    for (const r of Object.values(D.regions)) {
      totalGmem+=r.gmem_bytes||0; totalSmem+=r.smem_bytes||0; totalLmem+=r.lmem_bytes||0;
    }
    const totalMem = totalGmem+totalSmem+totalLmem;

    if (totalMem > 0) {
      h += sectionStart('m_overview','Memory Traffic Overview');
      h += '<div class="cards cards-4">';
      h += card(totalMem, 'Total Traffic', 'gmem_bytes');
      h += card(totalGmem, 'Global', 'gmem_bytes');
      h += card(totalSmem, 'Shared', 'smem_bytes');
      h += card(totalLmem, 'Local', 'lmem_bytes');
      h += '</div>';
      h += '<div id="mSplitChart" class="chart" style="height:160px;"></div>';
      h += sectionEnd();
    }

    // Locality Analysis
    if (D.locality && D.locality.regions && Object.keys(D.locality.regions).length) {
      h += '<div style="font-size:11px;color:var(--dim);margin-bottom:6px">Cache line: '+D.locality.line_bytes+'B</div>';

      for (const [rid, rd] of Object.entries(D.locality.regions)) {
        const label = D.labels[rid]||'region_'+rid;
        const color = D.colors[parseInt(rid)]||'#8b949e';

        h += sectionStart('m_loc_'+rid, label+' \u2014 Locality');
        h += '<div class="cards cards-4">';
        h += card(rd.records, 'Memory Ops', null, {noUnit:true});
        h += card(rd.unique_lines, 'Unique Cache Lines', 'gmem_unique_lines_est');
        h += card(rd.lines_per_record.toFixed(2)+' lines', 'Lines/Op', null, {raw:true,noUnit:true});
        h += card(((rd.sharing?.shared_line_ratio||0)*100).toFixed(1)+'%', 'Shared Ratio', 'shared_line_ratio', {raw:true,noUnit:true});
        h += '</div>';

        // Reuse distance histogram
        const reuseData = rd.reuse_distance||{};
        const scopes = Object.keys(reuseData);
        if (scopes.length) {
          h += '<div id="mReuse_'+rid+'" class="chart" style="height:200px;"></div>';
        }

        // Working set
        const ws = rd.working_set||{};
        if (Object.keys(ws).length) {
          h += '<div id="mWs_'+rid+'" class="chart" style="height:160px;"></div>';
        }

        // Inter-warp sharing
        const sharing = rd.sharing||{};
        if (sharing.lines_by_warps && Object.keys(sharing.lines_by_warps).length) {
          h += '<div id="mShare_'+rid+'" class="chart" style="height:140px;"></div>';
          h += '<div style="font-size:10px;color:var(--dim);margin:2px 0">Shared line ratio: '+
            ((sharing.shared_line_ratio||0)*100).toFixed(1)+'% | Avg warps/line: '+(sharing.avg_warps_per_line||0).toFixed(2)+'</div>';
        }

        // Inter-CTA sharing
        const ctaSharing = rd.cta_sharing||{};
        if (ctaSharing.lines_by_ctas && Object.keys(ctaSharing.lines_by_ctas).length) {
          h += '<div id="mCtaShare_'+rid+'" class="chart" style="height:140px;"></div>';
        }

        h += sectionEnd();
      }
    } else if (totalMem === 0) {
      h += '<div class="empty">No memory data available.<br>Run NVBit mode=all to collect memory traces.</div>';
    }

    // Per-region global/shared analysis summary
    const memRegions = rids.filter(rid=>((D.regions[rid]?.gmem_bytes||0)>0||(D.regions[rid]?.smem_bytes||0)>0));
    if (memRegions.length && !D.locality) {
      h += sectionStart('m_regsum','Per-Region Memory Summary');
      for (const rid of memRegions) {
        const r = D.regions[rid];
        const label = r.label||'region_'+rid;
        h += '<div style="font-size:11px;margin-bottom:6px"><b style="color:'+(D.colors[rid]||'var(--dim)')+'">'+label+'</b>: ';
        if ((r.gmem_bytes||0)>0) {
          const eff = r.gmem_bytes>0?(r.gmem_req_bytes||0)/r.gmem_bytes*100:0;
          h += 'Global '+fmtBytes(r.gmem_bytes)+' ('+eff.toFixed(0)+'% efficient) ';
        }
        if ((r.smem_bytes||0)>0) h += 'Shared '+fmtBytes(r.smem_bytes)+' ';
        h += '</div>';
      }
      h += sectionEnd();
    }

    // ── Memory Trace Visualization ──
    if (D.memTrace) {
      const mtRids = Object.keys(D.memTrace).map(Number).sort();
      for (const rid of mtRids) {
        const mt = D.memTrace[rid];
        if (!mt || !mt.records || !mt.records.length) continue;
        const label = D.labels[rid]||'region_'+rid;
        const color = D.colors[rid]||'var(--dim)';
        h += sectionStart('m_trace_'+rid,'Memory Trace: '+label+' ('+fmt(mt.total)+' total, '+mt.records.length+' shown)');
        h += '<div style="font-size:10px;color:var(--dim);margin-bottom:6px;padding:4px 8px;background:var(--bg2);border-radius:4px;line-height:1.6">'+
          'NVBit raw memory trace — each column is one warp memory instruction, each row is one lane (0-31). '+
          'Color = relative cache line ID. Uniform columns = good coalescing. Scattered colors = poor coalescing.'+infoIcon('mem_trace_records')+'</div>';

        // Lane address heatmap
        h += '<div id="mtHeat_'+rid+'" class="chart" style="height:360px;"></div>';

        // Per-PC coalescing summary
        const pcEntries = Object.entries(mt.per_pc).sort((a,b)=>b[1].count-a[1].count);
        if (pcEntries.length) {
          h += '<div style="margin-top:8px"><b style="font-size:11px">Per-Instruction Coalescing Summary</b>'+infoIcon('coalescing_ratio')+'</div>';
          h += '<div id="mtCoal_'+rid+'" class="chart" style="height:'+Math.max(120,pcEntries.length*24)+'px;"></div>';
          h += '<table class="ptable" style="text-align:left;font-size:10px;margin-top:4px">';
          h += '<tr><th style="text-align:left">PC</th><th>Count</th><th>Avg Cache Lines'+infoIcon('unique_cache_lines')+'</th><th>Load%</th><th style="text-align:left">Spaces</th><th>Coalescing'+infoIcon('coalescing_ratio')+'</th></tr>';
          for (const [pc, info] of pcEntries) {
            const idealLines = 1; // best case for 4B access with 32 lanes = 1 line
            const coalPct = idealLines / Math.max(info.avg_ul, 0.01) * 100;
            const coalColor = coalPct >= 90 ? 'var(--green)' : coalPct >= 50 ? 'var(--orange)' : 'var(--red)';
            h += '<tr><td style="text-align:left;font-family:monospace">'+pc+'</td>';
            h += '<td>'+fmt(info.count)+'</td>';
            h += '<td>'+info.avg_ul.toFixed(1)+'</td>';
            h += '<td>'+info.load_pct.toFixed(0)+'%</td>';
            h += '<td style="text-align:left">'+info.spaces.join(', ')+'</td>';
            h += '<td style="color:'+coalColor+';font-weight:600">'+coalPct.toFixed(0)+'%</td></tr>';
          }
          h += '</table>';
        }
        h += sectionEnd();
      }
    }

    if (!h) h = '<div class="empty">No memory data available.<br>Run NVBit mode=all to collect memory traces.</div>';
    ct.innerHTML = h;

    // ── Render memory charts ──

    if (totalMem > 0) {
      const split = [{name:'Global',value:totalGmem,itemStyle:{color:'#bc4c00'}},
        {name:'Shared',value:totalSmem,itemStyle:{color:'#0550ae'}},
        {name:'Local',value:totalLmem,itemStyle:{color:'#9a6700'}}].filter(d=>d.value>0);
      const ch = mkChart('mSplitChart');
      ch.setOption({
        backgroundColor:'transparent',
        tooltip:{trigger:'item',formatter:'{b}: '+'{c} bytes ({d}%)'},
        series:[{type:'pie',radius:['30%','60%'],data:split,
          label:{color:'#656d76',fontSize:10,formatter:'{b}\n{d}%'},
          itemStyle:{borderColor:'#ffffff',borderWidth:2},}],
      });
    }

    // Locality charts
    if (D.locality && D.locality.regions) {
      const histBounds = D.locality.hist_bounds||[];
      const boundLabels = histBounds.map((b,i)=>{
        if (b>=16384) return '16K+';
        if (b>=1024) return (b/1024)+'K';
        return b.toString();
      });
      for (const [rid, rd] of Object.entries(D.locality.regions)) {
        const reuseData = rd.reuse_distance||{};
        const scopes = Object.keys(reuseData);
        if (scopes.length) {
          const ch = mkChart('mReuse_'+rid);
          if(ch){const scopeColors={warp:'#0969da',cta:'#1a7f37',global:'#bc4c00'};
          const categories=['Cold',...boundLabels];
          const series=scopes.map(scope=>{const sd=reuseData[scope]||{};return{name:scope,type:'bar',data:[sd.cold||0,...(sd.hist||[])],itemStyle:{color:scopeColors[scope]||'#8b949e'}};});
          ch.setOption({backgroundColor:'transparent',title:{text:'Reuse Distance',textStyle:{color:'#656d76',fontSize:10},left:'center',top:0},
            legend:{top:20,textStyle:{color:'#656d76',fontSize:9}},grid:{left:10,right:10,top:45,bottom:20,containLabel:true},
            xAxis:{type:'category',data:categories,axisLabel:{color:'#656d76',fontSize:8,rotate:30}},
            yAxis:{type:'value',name:'Count',nameTextStyle:{color:'#656d76',fontSize:9},axisLabel:{color:'#656d76',fontSize:8,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
            series:series,tooltip:{trigger:'axis',formatter:function(params){let s=params[0].axisValue+'<br>';for(const p of params)if(p.value>0)s+=p.seriesName+': '+fmt(p.value)+'<br>';return s;}}});}
        }
        const ws=rd.working_set||{};const wsKeys=Object.keys(ws).sort((a,b)=>parseInt(a)-parseInt(b));
        if(wsKeys.length){const ch=mkChart('mWs_'+rid);if(ch){ch.setOption({backgroundColor:'transparent',
          title:{text:'Working Set',textStyle:{color:'#656d76',fontSize:10},left:'center',top:0},
          legend:{top:20,textStyle:{color:'#656d76',fontSize:9}},grid:{left:10,right:10,top:45,bottom:20,containLabel:true},
          xAxis:{type:'category',data:wsKeys.map(k=>'win='+k),axisLabel:{color:'#656d76',fontSize:9}},
          yAxis:{type:'value',name:'Unique Lines',nameTextStyle:{color:'#656d76',fontSize:9},axisLabel:{color:'#656d76',fontSize:9,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{name:'avg',type:'bar',data:wsKeys.map(k=>ws[k].avg||0),itemStyle:{color:'#0969da'}},
            {name:'p50',type:'line',data:wsKeys.map(k=>ws[k].p50||0),lineStyle:{color:'#1a7f37'},itemStyle:{color:'#1a7f37'}},
            {name:'p95',type:'line',data:wsKeys.map(k=>ws[k].p95||0),lineStyle:{color:'#bc4c00'},itemStyle:{color:'#bc4c00'}},
            {name:'max',type:'line',data:wsKeys.map(k=>ws[k].max||0),lineStyle:{color:'#cf222e',type:'dashed'},itemStyle:{color:'#cf222e'}}],
          tooltip:{trigger:'axis',formatter:function(params){let s='Window '+params[0].axisValue.replace('win=','')+'<br>';for(const p of params)s+=p.seriesName+': '+fmt(p.value)+'<br>';return s;}}});}}
        const sharing=rd.sharing||{};
        if(sharing.lines_by_warps&&Object.keys(sharing.lines_by_warps).length){const ch=mkChart('mShare_'+rid);if(ch){
          const entries=Object.entries(sharing.lines_by_warps).sort((a,b)=>parseInt(a[0])-parseInt(b[0]));
          ch.setOption({backgroundColor:'transparent',title:{text:'Inter-Warp Sharing',textStyle:{color:'#656d76',fontSize:10},left:'center',top:0},
            grid:{left:10,right:10,top:25,bottom:20,containLabel:true},
            xAxis:{type:'category',data:entries.map(([k])=>k+' warps'),name:'# Warps sharing a line',nameLocation:'center',nameGap:18,nameTextStyle:{color:'#656d76',fontSize:9},axisLabel:{color:'#656d76',fontSize:9}},
            yAxis:{type:'value',name:'Cache Lines',nameTextStyle:{color:'#656d76',fontSize:9},axisLabel:{color:'#656d76',fontSize:9,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
            series:[{type:'bar',data:entries.map(([,v])=>v),itemStyle:{color:'#8250df',borderRadius:[3,3,0,0]}}],
            tooltip:{trigger:'axis',formatter:p=>p[0].name+': '+fmt(p[0].value)+' lines'}});}}
        const ctaSharing=rd.cta_sharing||{};
        if(ctaSharing.lines_by_ctas&&Object.keys(ctaSharing.lines_by_ctas).length){const ch=mkChart('mCtaShare_'+rid);if(ch){
          const entries=Object.entries(ctaSharing.lines_by_ctas).sort((a,b)=>parseInt(a[0])-parseInt(b[0]));
          ch.setOption({backgroundColor:'transparent',title:{text:'Inter-CTA Sharing',textStyle:{color:'#656d76',fontSize:10},left:'center',top:0},
            grid:{left:10,right:10,top:25,bottom:20,containLabel:true},
            xAxis:{type:'category',data:entries.map(([k])=>k+' CTAs'),axisLabel:{color:'#656d76',fontSize:9}},
            yAxis:{type:'value',name:'Cache Lines',nameTextStyle:{color:'#656d76',fontSize:9},axisLabel:{color:'#656d76',fontSize:9,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
            series:[{type:'bar',data:entries.map(([,v])=>v),itemStyle:{color:'#0550ae',borderRadius:[3,3,0,0]}}],
            tooltip:{trigger:'axis',formatter:p=>p[0].name+': '+fmt(p[0].value)+' lines'}});}}
      }
    }

    // Memory trace heatmaps
    if (D.memTrace) {
      for (const rid of Object.keys(D.memTrace).map(Number).sort()) {
        const mt = D.memTrace[rid];
        if (!mt || !mt.records || !mt.records.length) continue;

        // Build heatmap data: [recordIdx, laneIdx, cacheLineId]
        const recs = mt.records;
        const heatData = [];
        let maxLine = 0;
        for (let i = 0; i < recs.length; i++) {
          const lanes = recs[i].lanes;
          for (let lane = 0; lane < 32; lane++) {
            const v = lanes[lane];
            if (v >= 0) {
              heatData.push([i, lane, v]);
              if (v > maxLine) maxLine = v;
            }
          }
        }

        const ch = mkChart('mtHeat_'+rid);
        if (ch) {
          ch.setOption({
            backgroundColor: 'transparent',
            title: {text: 'Lane Address Heatmap (32 lanes \u00d7 '+recs.length+' accesses)',
              textStyle:{color:'#656d76',fontSize:11},left:'center',top:0},
            grid: {left:50, right:30, top:35, bottom:50},
            xAxis: {type:'category',
              data: Array.from({length:recs.length},(_,i)=>i),
              name:'Memory Access #', nameLocation:'center', nameGap:30,
              nameTextStyle:{color:'#656d76',fontSize:10},
              axisLabel:{color:'#656d76',fontSize:8,interval:Math.floor(recs.length/10),
                formatter:function(v){
                  const r=recs[parseInt(v)];
                  return r?r.pc:'';
                }},
              splitLine:{show:false}},
            yAxis: {type:'category',
              data: Array.from({length:32},(_,i)=>'L'+i),
              name:'Lane', nameLocation:'center', nameGap:35,
              nameTextStyle:{color:'#656d76',fontSize:10},
              axisLabel:{color:'#656d76',fontSize:8},
              splitLine:{show:false}},
            visualMap: {min:0, max:Math.max(maxLine,1), calculable:true, orient:'horizontal',
              left:'center', bottom:5,
              inRange:{color:['#dafbe1','#1a7f37','#0969da','#8250df','#cf222e','#ffbe98']},
              textStyle:{color:'#656d76',fontSize:9},
              text:['Distant line','Near line']},
            series:[{type:'heatmap', data:heatData,
              itemStyle:{borderWidth:0},
              emphasis:{itemStyle:{borderColor:'#fff',borderWidth:1}},
            }],
            tooltip:{trigger:'item',formatter:function(p){
              const rec=recs[p.value[0]];
              return 'Access #'+p.value[0]+', Lane '+p.value[1]+
                '<br>Cache line: '+p.value[2]+
                '<br>PC: '+rec.pc+' ('+rec.space+(rec.ld?' load':' store')+')'+
                '<br>CTA: '+rec.cta+', Warp: '+rec.warp+
                '<br>Unique lines: '+rec.ul+'/'+rec.ac+' lanes';
            }},
            dataZoom:[{type:'slider',xAxisIndex:0,bottom:25,height:12,
              textStyle:{color:'#656d76',fontSize:8}}],
          });
        }

        // Per-PC coalescing bar chart
        const pcEntries = Object.entries(mt.per_pc).sort((a,b)=>b[1].count-a[1].count);
        if (pcEntries.length) {
          const ch2 = mkChart('mtCoal_'+rid);
          if (ch2) {
            ch2.setOption({
              backgroundColor:'transparent',
              grid:{left:10,right:30,top:5,bottom:5,containLabel:true},
              yAxis:{type:'category',data:pcEntries.map(([pc])=>pc).reverse(),
                axisLabel:{color:'#656d76',fontSize:9,fontFamily:'monospace'}},
              xAxis:{type:'value',name:'Avg Cache Lines per Access',
                nameTextStyle:{color:'#656d76',fontSize:9},
                axisLabel:{color:'#656d76',fontSize:9},splitLine:{lineStyle:{color:'#d0d7de'}}},
              series:[{type:'bar',data:pcEntries.map(([,d])=>d.avg_ul).reverse(),
                label:{show:true,position:'right',fontSize:9,color:'#656d76',
                  formatter:function(p){return p.value.toFixed(1)+' lines';}},
                itemStyle:{color:function(p){
                  return p.value<=1?'#1a7f37':p.value<=2?'#0969da':p.value<=4?'#bc4c00':'#cf222e';
                },borderRadius:[0,3,3,0]},barWidth:'55%'}],
              tooltip:{trigger:'axis',formatter:function(p){
                const pc=pcEntries[pcEntries.length-1-p[0].dataIndex];
                if(!pc)return'';
                return 'PC '+pc[0]+': '+pc[1].avg_ul.toFixed(2)+' cache lines/access'+
                  '<br>'+fmt(pc[1].count)+' accesses, '+pc[1].load_pct+'% loads';
              }},
            });
          }
        }
      }
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // TAB 6: STALLS (renamed from CUPTI)
  // ══════════════════════════════════════════════════════════════════
  function buildStalls() {
    const ct = document.getElementById('stallsCt');
    let h = '';

    // Section A: Aggregated Stall Profile
    if (D.pcsamp && D.pcsamp.records && D.pcsamp.records.length) {
      h += sectionStart('st_agg','Aggregated Stall Profile — Whole Kernel ('+D.pcsamp.total+' samples, period='+D.pcsamp.period+')');
      h += '<div style="font-size:10px;color:var(--dim);margin-bottom:6px;padding:4px 8px;background:var(--bg2);border-radius:4px;line-height:1.6">'+
        '<b style="color:var(--orange)">\u26A0 Whole-kernel scope</b> — PC sampling stall reasons are collected across the <b>entire kernel</b>. '+
        'The Per-PC table below shows which instruction addresses contribute most stalls.</div>';
      h += '<div id="stStallChart" class="chart" style="height:'+Math.max(180,12*22)+'px;"></div>';
      h += sectionEnd();

      // Section C: Per-PC Top Stalls (enhanced with source line + region + stall breakdown)
      h += sectionStart('st_perpc','Per-PC Top Stalls');
      // Aggregate stalls per PC
      const pcStalls = {};
      for (const rec of D.pcsamp.records) {
        const pc = rec.pcOffset||rec.pc_offset||0;
        if (!pcStalls[pc]) pcStalls[pc] = {total:0,stalls:{}};
        for (const [idx,c] of Object.entries(rec.stall_reasons||{})) {
          const n = D.pcsamp.stalls[idx]||'stall_'+idx;
          pcStalls[pc].stalls[n] = (pcStalls[pc].stalls[n]||0)+c;
          pcStalls[pc].total += c;
        }
      }
      const topPcs = Object.entries(pcStalls).sort((a,b)=>b[1].total-a[1].total).slice(0,20);
      if (topPcs.length) {
        const globalMax = topPcs[0][1].total;
        h += '<table class="ptable" style="text-align:left;font-size:10px;width:100%">';
        h += '<tr><th style="text-align:left">PC</th><th style="text-align:left">Source</th><th style="text-align:left">Region</th>'+
          '<th style="text-align:left">Top Stall</th><th>Samples</th><th style="text-align:left;min-width:120px">Stall Distribution</th></tr>';
        for (const [pc, data] of topPcs) {
          const topStall = Object.entries(data.stalls).sort((a,b)=>b[1]-a[1])[0];
          // Source line lookup via pc2src
          const srcLine = D.pc2src ? D.pc2src[String(parseInt(pc))] : null;
          // Region lookup via source line → lineRegions
          let regionId = null, regionLabel = null, regionColor = null;
          if (srcLine && D.lineRegions && D.lineRegions[String(srcLine)]) {
            regionId = D.lineRegions[String(srcLine)];
            regionLabel = D.labels[String(regionId)] || ('region_'+regionId);
            regionColor = D.colors[String(regionId)] || '#656d76';
          }
          h += '<tr>';
          h += '<td style="text-align:left;font-family:monospace;white-space:nowrap">0x'+parseInt(pc).toString(16)+'</td>';
          // Source line (clickable)
          if (srcLine) {
            h += '<td style="text-align:left"><a href="#" onclick="srcEditor.revealLineInCenter('+srcLine+');srcEditor.setPosition({lineNumber:'+srcLine+',column:1});return false;" '+
              'style="color:var(--accent);text-decoration:none" title="Go to source line '+srcLine+'">L'+srcLine+'</a></td>';
          } else {
            h += '<td style="text-align:left;color:var(--dim)">-</td>';
          }
          // Region badge
          if (regionLabel) {
            h += '<td style="text-align:left"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:'+regionColor+';margin-right:3px;vertical-align:middle"></span>'+
              '<span style="font-size:9px">'+regionLabel+'</span></td>';
          } else {
            h += '<td style="text-align:left;color:var(--dim)">-</td>';
          }
          h += '<td style="text-align:left">'+((topStall?topStall[0]:'none').replace(/_/g,' '))+infoIcon(topStall?topStall[0]:'')+'</td>';
          h += '<td style="text-align:right">'+fmt(data.total)+'</td>';
          // Mini stall distribution bar
          const sortedStalls = Object.entries(data.stalls).sort((a,b)=>b[1]-a[1]).slice(0,5);
          const stallColors = ['#cf222e','#bc4c00','#9a6700','#0969da','#8250df'];
          let barHtml = '<div style="display:flex;height:14px;border-radius:3px;overflow:hidden;width:100%;background:var(--bg3)" title="';
          barHtml += sortedStalls.map(([n,v])=>n.replace(/_/g,' ')+': '+v).join(', ');
          barHtml += '">';
          for (let si=0; si<sortedStalls.length; si++) {
            const pct = (sortedStalls[si][1] / data.total * 100);
            if (pct > 1) barHtml += '<div style="width:'+pct.toFixed(1)+'%;background:'+stallColors[si]+';min-width:2px" title="'+sortedStalls[si][0].replace(/_/g,' ')+': '+sortedStalls[si][1]+'"></div>';
          }
          barHtml += '</div>';
          h += '<td>'+barHtml+'</td>';
          h += '</tr>';
        }
        h += '</table>';
        // Legend for stall bar colors
        h += '<div style="margin-top:4px;font-size:9px;color:var(--dim);display:flex;gap:10px;flex-wrap:wrap">';
        h += '<span>Bar = top-5 stall reasons per PC: ';
        const legendColors = ['#cf222e','#bc4c00','#9a6700','#0969da','#8250df'];
        ['1st','2nd','3rd','4th','5th'].forEach((lbl,i)=> {
          h += '<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:'+legendColors[i]+';vertical-align:middle;margin:0 2px"></span>'+lbl+' ';
        });
        h += '</span></div>';
      } else {
        h += '<div style="color:var(--dim);font-size:11px">No per-PC data available</div>';
      }
      h += sectionEnd();
    }

    // Section B: SASS Profile — Derived Efficiency Metrics
    if (D.sassProfiles && Object.keys(D.sassProfiles).length) {
      h += sectionStart('st_sass','SASS Profile — Whole-Kernel Efficiency Metrics');
      h += '<div style="font-size:10px;color:var(--dim);margin-bottom:6px;padding:4px 8px;background:var(--bg2);border-radius:4px;line-height:1.6">'+
        '<b style="color:var(--orange)">\u26A0 Whole-kernel scope</b> — These metrics are aggregated across the <b>entire kernel</b>, not per-region. '+
        'CUPTI SASS profiles cannot attribute to individual source regions. '+
        'For per-region instruction-level analysis, see the <b>Regions</b> tab (NVBit data).</div>';
      // Compute derived ratios from raw SASS profile aggregates
      const sp = D.sassProfiles;
      const ie = sp.core?.['smsp__sass_inst_executed'] || sp.divergence?.['smsp__sass_inst_executed'] || 1;
      const tie = sp.divergence?.['smsp__sass_thread_inst_executed'] || 0;
      const tpon = sp.divergence?.['smsp__sass_thread_inst_executed_pred_on'] || 0;
      const gSectors = sp.memory?.['smsp__sass_sectors_mem_global'] || 0;
      const gSectorsIdeal = sp.memory?.['smsp__sass_sectors_mem_global_ideal'] || 0;
      const sWave = sp.memory?.['smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared'] || 0;
      const sWaveIdeal = sp.memory?.['smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal'] || 0;
      const brUniform = sp.branch?.['smsp__sass_branch_targets_threads_uniform'] || 0;
      const brDivergent = sp.branch?.['smsp__sass_branch_targets_threads_divergent'] || 0;
      const brInst = sp.instruction_mix?.['smsp__sass_inst_executed_op_branch'] || 0;
      const gInst = sp.instruction_mix?.['smsp__sass_inst_executed_op_global'] || 0;
      const sInst = sp.instruction_mix?.['smsp__sass_inst_executed_op_shared'] || 0;

      const metrics = [];
      if (tie > 0) {
        const occ = tie / (ie * 32);
        metrics.push({name:'SIMT Utilization', value: occ, pct: true,
          desc:'Active threads per warp instruction / 32. Measures lane utilization, NOT SM occupancy. < 100% = divergence or partial warps.'});
      }
      if (tpon > 0 && tie > 0) {
        const pred = tpon / tie;
        metrics.push({name:'Predication Efficiency', value: pred, pct: true,
          desc:'Threads with predicate ON / total thread-instr. Lower = more predicated-off work.'});
      }
      if (gSectors > 0 && gSectorsIdeal > 0) {
        const coal = gSectorsIdeal / gSectors;
        metrics.push({name:'Global Mem Coalescing', value: coal, pct: true,
          desc:'Ideal sectors / actual sectors. 100% = perfect coalescing. Lower = scattered accesses.'});
      }
      if (sWave > 0 && sWaveIdeal > 0) {
        const smeff = sWaveIdeal / sWave;
        metrics.push({name:'Shared Mem Efficiency', value: smeff, pct: true,
          desc:'Ideal wavefronts / actual wavefronts. < 100% = bank conflicts.'});
      }
      if (brUniform + brDivergent > 0) {
        const bunif = brUniform / (brUniform + brDivergent);
        metrics.push({name:'Branch Uniformity', value: bunif, pct: true,
          desc:'Uniform branches / total branches. Lower = more warp divergence at branches.'});
      }
      if (ie > 0) {
        metrics.push({name:'Branch Fraction', value: brInst/ie, pct: true,
          desc:'Branch instructions / total instructions. High = control-flow heavy kernel.'});
        metrics.push({name:'Global Mem Fraction', value: gInst/ie, pct: true,
          desc:'Global memory ops / total instructions.'});
        metrics.push({name:'Shared Mem Fraction', value: sInst/ie, pct: true,
          desc:'Shared memory ops / total instructions.'});
      }

      if (metrics.length) {
        h += '<div id="stSassBarChart" class="chart" style="height:'+Math.max(160, metrics.length*32)+'px;"></div>';
        h += '<table class="ptable" style="text-align:left;font-size:10px;margin-top:6px">';
        h += '<tr><th style="text-align:left">Metric</th><th>Value</th><th style="text-align:left">Description</th></tr>';
        for (const m of metrics) {
          const vStr = m.pct ? (m.value*100).toFixed(1)+'%' : m.value.toFixed(3);
          const color = m.value >= 0.9 ? 'var(--green)' : m.value >= 0.5 ? 'var(--orange)' : 'var(--red)';
          h += '<tr><td style="text-align:left;font-weight:600">'+m.name+'</td>';
          h += '<td style="color:'+color+';font-weight:600">'+vStr+'</td>';
          h += '<td style="text-align:left;color:var(--dim);font-size:9px">'+m.desc+'</td></tr>';
        }
        h += '</table>';
      } else {
        h += '<div style="color:var(--dim);font-size:11px">No SASS efficiency metrics derivable from available profiles.</div>';
      }
      h += sectionEnd();
    }

    // Section C: Per-Region CUPTI Efficiency Comparison
    if (D.cuptiPerRegion && Object.keys(D.cuptiPerRegion).length > 0) {
      const rids = Object.keys(D.cuptiPerRegion).map(Number).filter(rid=>!(D.labels[rid]||'').startsWith('_')).sort();
      if (rids.length > 0) {
        h += sectionStart('st_perregion','Per-Region SASS Efficiency Comparison');
        h += '<div style="font-size:10px;color:var(--dim);margin-bottom:6px;padding:4px 8px;background:var(--bg2);border-radius:4px;line-height:1.6">'+
          'CUPTI SASS metrics attributed to individual regions via NVBit pc2region mapping (~78% PC coverage). '+
          'Compare efficiency across regions to identify bottleneck locations.</div>';
        h += '<div id="stPerRegionChart" class="chart" style="height:300px;"></div>';
        // Table
        const effNames = ['SIMT Utilization','Predication Eff.','Global Coalescing','Shared Mem Eff.','Branch Uniformity'];
        const effKeys = ['simt_utilization','predication_eff','global_coalescing','shared_efficiency','branch_uniformity'];
        h += '<table class="ptable" style="text-align:left;font-size:10px;margin-top:6px">';
        h += '<tr><th style="text-align:left">Region</th>';
        for (const n of effNames) h += '<th>'+n+'</th>';
        h += '</tr>';
        for (const rid of rids) {
          const eff = D.cuptiPerRegion[rid]?.efficiency || {};
          const label = D.labels[rid]||'region_'+rid;
          const c = D.colors[rid]||'var(--dim)';
          h += '<tr><td style="text-align:left"><span style="color:'+c+';font-weight:600">\u25CF '+label+'</span></td>';
          for (const k of effKeys) {
            const v = eff[k];
            if (v != null) {
              const color = v >= 0.9 ? 'var(--green)' : v >= 0.5 ? 'var(--orange)' : 'var(--red)';
              h += '<td style="color:'+color+';font-weight:600">'+(v*100).toFixed(1)+'%</td>';
            } else {
              h += '<td style="color:var(--dim)">-</td>';
            }
          }
          h += '</tr>';
        }
        h += '</table>';
        h += sectionEnd();
      }
    }

    // Section D: InstrExec Analysis (enhanced with source + region + detailed table)
    if (D.instrexec && D.instrexec.length) {
      h += sectionStart('st_instrexec','Instruction Execution ('+D.instrexec.length+' records)');
      h += '<div id="stInstrChart" class="chart" style="height:200px;"></div>';
      // Detailed table: top 30 PCs with source line + region + utilization
      const instrSorted = [...D.instrexec].sort((a,b)=>(b.threadsExecuted||b.threadCount||0)-(a.threadsExecuted||a.threadCount||0)).slice(0,30);
      h += '<div style="margin-top:8px;font-size:10px;font-weight:600;color:var(--bright)">Per-PC Detail</div>';
      h += '<div style="overflow-x:auto;margin-top:4px"><table class="ptable" style="text-align:left;font-size:10px;width:100%">';
      h += '<tr><th style="text-align:left">PC</th><th style="text-align:left">Source</th><th style="text-align:left">Region</th>'+
        '<th style="text-align:right">Threads'+infoIcon('threads_executed')+'</th>'+
        '<th style="text-align:right">Warps'+infoIcon('executed')+'</th>'+
        '<th style="text-align:right">Pred Off'+infoIcon('notPredOffThreadsExecuted')+'</th>'+
        '<th style="text-align:right">Utilization'+infoIcon('simt_utilization')+'</th></tr>';
      for (const rec of instrSorted) {
        const pc = rec.pcOffset||0;
        const te = rec.threadsExecuted||rec.threadCount||0;
        const ex = rec.executed||0;
        const npo = rec.notPredOffThreadsExecuted||0;
        const srcLine = (rec.source && rec.source.line) ? rec.source.line : null;
        // Region lookup
        let regionId = null, regionLabel = null, regionColor = null;
        if (srcLine && D.lineRegions && D.lineRegions[String(srcLine)]) {
          regionId = D.lineRegions[String(srcLine)];
          regionLabel = D.labels[String(regionId)] || ('region_'+regionId);
          regionColor = D.colors[String(regionId)] || '#656d76';
        }
        const util = (ex > 0 && te > 0) ? (te / (ex * 32) * 100) : null;
        h += '<tr>';
        h += '<td style="text-align:left;font-family:monospace;white-space:nowrap">0x'+pc.toString(16)+'</td>';
        // Source line (clickable)
        if (srcLine) {
          h += '<td style="text-align:left"><a href="#" onclick="srcEditor.revealLineInCenter('+srcLine+');srcEditor.setPosition({lineNumber:'+srcLine+',column:1});return false;" '+
            'style="color:var(--accent);text-decoration:none" title="Go to source line '+srcLine+'">L'+srcLine+'</a></td>';
        } else {
          h += '<td style="text-align:left;color:var(--dim)">-</td>';
        }
        // Region badge
        if (regionLabel) {
          h += '<td style="text-align:left"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:'+regionColor+';margin-right:3px;vertical-align:middle"></span>'+
            '<span style="font-size:9px">'+regionLabel+'</span></td>';
        } else {
          h += '<td style="text-align:left;color:var(--dim)">-</td>';
        }
        h += '<td style="text-align:right">'+fmt(te)+'</td>';
        h += '<td style="text-align:right">'+fmt(ex)+'</td>';
        h += '<td style="text-align:right">'+fmt(npo)+'</td>';
        if (util !== null) {
          const uColor = util >= 90 ? 'var(--green)' : util >= 50 ? 'var(--orange)' : 'var(--red)';
          h += '<td style="text-align:right"><span style="color:'+uColor+';font-weight:600">'+util.toFixed(1)+'%</span>'+
            '<div style="height:3px;background:var(--bg3);border-radius:2px;margin-top:1px"><div style="height:100%;width:'+Math.min(100,util).toFixed(1)+'%;background:'+uColor+';border-radius:2px"></div></div></td>';
        } else {
          h += '<td style="text-align:right;color:var(--dim)">-</td>';
        }
        h += '</tr>';
      }
      h += '</table></div>';
      h += sectionEnd();
    }

    if (!h) h = '<div class="empty">No CUPTI/stall data available.<br>Collect PC sampling or SASS metrics to populate this tab.</div>';
    ct.innerHTML = h;

    // Charts
    if (D.pcsamp && D.pcsamp.records && D.pcsamp.records.length) {
      // Aggregate stalls
      const stalls = {};
      for (const rec of D.pcsamp.records) {
        for (const [idx,c] of Object.entries(rec.stall_reasons||{})) {
          const n = D.pcsamp.stalls[idx]||'stall_'+idx;
          stalls[n] = (stalls[n]||0)+c;
        }
      }
      const se = Object.entries(stalls).sort((a,b)=>b[1]-a[1]);
      if (se.length) {
        const ch = mkChart('stStallChart');
        const names = se.map(([n])=>n.replace(/_/g,' '));
        const values = se.map(([,v])=>v);
        ch.setOption({
          backgroundColor:'transparent',
          grid:{left:10,right:30,top:5,bottom:5,containLabel:true},
          yAxis:{type:'category',data:names.reverse(),axisLabel:{color:'#656d76',fontSize:9}},
          xAxis:{type:'value',name:'Samples',nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:9,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
          series:[{type:'bar',data:values.reverse(),
            itemStyle:{color:function(p){
              const maxV=Math.max(...values);
              const ratio=p.value/maxV;
              return ratio>0.5?'#cf222e':ratio>0.2?'#bc4c00':'#0969da';
            },borderRadius:[0,3,3,0]},barWidth:'60%'}],
          tooltip:{trigger:'axis',formatter:function(p){
            const stallName = p[0].name.replace(/ /g,'_');
            const desc = D.defs[stallName]?.long||'';
            return '<b>'+p[0].name+'</b>: '+fmt(p[0].value)+' samples'+(desc?'<br><span style="font-size:10px;color:#999">'+desc+'</span>':'');
          }},
        });
      }
    }

    // SASS Profile efficiency bar chart
    if (D.sassProfiles && Object.keys(D.sassProfiles).length) {
      const barEl = document.getElementById('stSassBarChart');
      if (barEl) {
        // Re-compute same metrics as above (they're in the DOM logic, recompute for chart)
        const sp = D.sassProfiles;
        const ie = sp.core?.['smsp__sass_inst_executed'] || sp.divergence?.['smsp__sass_inst_executed'] || 1;
        const tie = sp.divergence?.['smsp__sass_thread_inst_executed'] || 0;
        const tpon = sp.divergence?.['smsp__sass_thread_inst_executed_pred_on'] || 0;
        const gS = sp.memory?.['smsp__sass_sectors_mem_global'] || 0;
        const gSI = sp.memory?.['smsp__sass_sectors_mem_global_ideal'] || 0;
        const sW = sp.memory?.['smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared'] || 0;
        const sWI = sp.memory?.['smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal'] || 0;
        const brU = sp.branch?.['smsp__sass_branch_targets_threads_uniform'] || 0;
        const brD = sp.branch?.['smsp__sass_branch_targets_threads_divergent'] || 0;
        const items = [];
        if (tie > 0) items.push({name:'SIMT Utilization', value: tie/(ie*32)*100});
        if (tpon > 0 && tie > 0) items.push({name:'Pred Efficiency', value: tpon/tie*100});
        if (gS > 0 && gSI > 0) items.push({name:'Global Coalescing', value: gSI/gS*100});
        if (sW > 0 && sWI > 0) items.push({name:'Shared Mem Eff', value: sWI/sW*100});
        if (brU+brD > 0) items.push({name:'Branch Uniformity', value: brU/(brU+brD)*100});
        if (items.length) {
          const ch = mkChart('stSassBarChart');
          ch.setOption({
            backgroundColor:'transparent',
            grid:{left:10,right:30,top:5,bottom:5,containLabel:true},
            yAxis:{type:'category',data:items.map(i=>i.name).reverse(),axisLabel:{color:'#656d76',fontSize:10}},
            xAxis:{type:'value',max:100,axisLabel:{color:'#656d76',fontSize:9,formatter:v=>v+'%'},
              splitLine:{lineStyle:{color:'#d0d7de'}}},
            series:[{type:'bar',data:items.map(i=>i.value).reverse(),
              itemStyle:{color:function(p){
                return p.value>=90?'#1a7f37':p.value>=50?'#bc4c00':'#cf222e';
              },borderRadius:[0,3,3,0]},barWidth:'60%',
              label:{show:true,position:'right',formatter:p=>p.value.toFixed(1)+'%',fontSize:9,color:'#656d76'}}],
            tooltip:{trigger:'axis',formatter:function(p){
              return '<b>'+p[0].name+'</b>: '+p[0].value.toFixed(1)+'%';
            }},
          });
        }
      }
    }

    // Per-region CUPTI efficiency grouped bar chart
    if (D.cuptiPerRegion && Object.keys(D.cuptiPerRegion).length > 0) {
      const el = document.getElementById('stPerRegionChart');
      if (el) {
        const rids = Object.keys(D.cuptiPerRegion).map(Number).filter(rid=>!(D.labels[rid]||'').startsWith('_')).sort();
        const effKeys = ['simt_utilization','predication_eff','global_coalescing','shared_efficiency','branch_uniformity'];
        const effLabels = ['SIMT Util.','Pred. Eff.','Global Coal.','Shared Eff.','Branch Unif.'];
        const effColors = ['#0969da','#8250df','#bc4c00','#0550ae','#6639ba'];
        const regionNames = rids.map(rid=>D.labels[rid]||'region_'+rid);
        const series = [];
        for (let i = 0; i < effKeys.length; i++) {
          const data = rids.map(rid => {
            const v = D.cuptiPerRegion[rid]?.efficiency?.[effKeys[i]];
            return v != null ? +(v * 100).toFixed(1) : null;
          });
          // Only include if at least one region has this metric
          if (data.some(v => v != null)) {
            series.push({
              name: effLabels[i], type: 'bar',
              data: data.map(v => v ?? 0),
              itemStyle: {color: effColors[i], borderRadius: [3, 3, 0, 0]},
              barGap: '10%',
              label: {show: rids.length <= 6, position: 'top', fontSize: 8, color: '#656d76',
                formatter: function(p) { return p.value > 0 ? p.value.toFixed(0) + '%' : ''; }},
            });
          }
        }
        if (series.length) {
          const ch = mkChart('stPerRegionChart');
          ch.setOption({
            backgroundColor: 'transparent',
            legend: {top: 0, textStyle: {color: '#656d76', fontSize: 9}, itemWidth: 12, itemHeight: 8},
            grid: {left: 10, right: 10, top: 40, bottom: 10, containLabel: true},
            xAxis: {type: 'category', data: regionNames,
              axisLabel: {color: '#656d76', fontSize: 9}},
            yAxis: {type: 'value', max: 100, name: 'Efficiency %', nameTextStyle: {color: '#656d76', fontSize: 9},
              axisLabel: {color: '#656d76', fontSize: 9, formatter: v => v + '%'},
              splitLine: {lineStyle: {color: '#d0d7de'}}},
            series: series,
            tooltip: {trigger: 'axis', formatter: function(params) {
              let s = '<b>' + params[0].name + '</b>';
              for (const p of params) {
                if (p.value > 0) {
                  const color = p.value >= 90 ? '#1a7f37' : p.value >= 50 ? '#bc4c00' : '#cf222e';
                  s += '<br>' + p.marker + ' ' + p.seriesName + ': <b style="color:' + color + '">' + p.value.toFixed(1) + '%</b>';
                }
              }
              return s;
            }},
          });
        }
      }
    }

    // InstrExec chart (enhanced: source line labels + region coloring + richer tooltip)
    if (D.instrexec && D.instrexec.length) {
      const sorted = [...D.instrexec].sort((a,b)=>(b.threadsExecuted||b.threadCount||0)-(a.threadsExecuted||a.threadCount||0)).slice(0,20);
      const ch = mkChart('stInstrChart');
      // Build x-axis labels with source line info
      const xLabels = sorted.map(r => {
        const pc = '0x'+(r.pcOffset||0).toString(16);
        const sl = (r.source && r.source.line) ? ' L'+r.source.line : '';
        return pc + sl;
      });
      // Color bars by region
      const barColors = sorted.map(r => {
        const sl = (r.source && r.source.line) ? r.source.line : null;
        if (sl && D.lineRegions && D.lineRegions[String(sl)]) {
          const rid = D.lineRegions[String(sl)];
          return D.colors[String(rid)] || '#1a7f37';
        }
        return '#1a7f37';
      });
      ch.setOption({
        backgroundColor:'transparent',
        grid:{left:10,right:10,top:10,bottom:20,containLabel:true},
        xAxis:{type:'category',data:xLabels,
          axisLabel:{color:'#656d76',fontSize:8,rotate:30}},
        yAxis:{type:'value',name:'Threads Executed',nameTextStyle:{color:'#656d76',fontSize:9},
          axisLabel:{color:'#656d76',fontSize:9,formatter:v=>fmt(v)},splitLine:{lineStyle:{color:'#d0d7de'}}},
        series:[{type:'bar',data:sorted.map((r,i)=>({
          value:r.threadsExecuted||r.threadCount||0,
          itemStyle:{color:barColors[i],borderRadius:[3,3,0,0]}
        }))}],
        tooltip:{trigger:'axis',formatter:function(p){
          const r=sorted[p[0].dataIndex];
          const te=r.threadsExecuted||r.threadCount||0;
          const ex=r.executed||0;
          const npo=r.notPredOffThreadsExecuted||0;
          const sl=(r.source&&r.source.line)?r.source.line:null;
          let s = '<b>PC 0x'+(r.pcOffset||0).toString(16)+'</b>';
          if (sl) s += ' \u2192 <b>Line '+sl+'</b>';
          s += '<br>Threads: '+fmt(te);
          if (ex>0) s += '<br>Warps: '+fmt(ex);
          if (ex>0&&te>0) {
            const util=(te/(ex*32)*100);
            const uc=util>=90?'#1a7f37':util>=50?'#bc4c00':'#cf222e';
            s += '<br>SIMT Utilization: <span style="color:'+uc+';font-weight:bold">'+util.toFixed(1)+'%</span>';
          }
          if (npo>0&&te>0) s += '<br>Pred Off: '+fmt(npo)+' ('+(npo/te*100).toFixed(1)+'%)';
          // Region info
          if (sl && D.lineRegions && D.lineRegions[String(sl)]) {
            const rid = D.lineRegions[String(sl)];
            const rl = D.labels[String(rid)] || 'region_'+rid;
            const rc = D.colors[String(rid)] || '#656d76';
            s += '<br>Region: <span style="color:'+rc+'">\u25CF</span> '+rl;
          }
          return s;
        }},
      });
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // TAB 7: TRACE
  // ══════════════════════════════════════════════════════════════════
  function buildTrace() {
    const ct = document.getElementById('traceCt');
    if (!D.trace || !D.trace.length) { ct.innerHTML = '<div class="empty">No trace data</div>'; return; }
    let h = '';
    const regions = D.trace.filter(r => r.name !== 'total');
    const total = D.trace.find(r => r.name === 'total');

    // Section A: Summary Cards
    h += '<div class="cards cards-3">';
    if (total) {
      h += card(total.count, 'Total Events', null, {noUnit:true});
      h += card(total.mean, 'Mean Duration', 'mean_dur');
      h += card((total.cv||0).toFixed(2), 'CV (variation)', 'cv_dur', {raw:true,noUnit:true});
    }
    h += '</div>';

    // Section B: Region Timing Comparison
    if (regions.length) {
      h += sectionStart('tr_compare','Region Timing Comparison');
      h += '<div id="trCompChart" class="chart" style="height:220px;"></div>';
      h += sectionEnd();
    }

    // Section C: Full Percentile Table
    h += sectionStart('tr_pct','Full Percentiles');
    const pctKeys = ['p5','p10','p15','p20','p25','p30','p35','p40','p45','p50','p55','p60','p65','p70','p75','p80','p85','p90','p95','p99'];
    h += '<div style="overflow-x:auto"><table class="ptable">';
    h += '<tr><th style="text-align:left">Region</th><th>Min</th>';
    for (const k of pctKeys) h += '<th>'+k+'</th>';
    h += '<th>Max</th><th>CV</th></tr>';
    for (const r of D.trace) {
      const pcts = r.percentiles||{};
      const p50 = pcts.p50||0, p99 = pcts.p99||0;
      const tailWarn = p50>0 && p99/p50>2;
      h += '<tr><td style="text-align:left;color:var(--bright)">'+r.name+'</td>';
      h += '<td>'+fmt(r.min)+'</td>';
      for (const k of pctKeys) {
        const v = pcts[k]||0;
        const isWarn = tailWarn && (k==='p95'||k==='p99');
        h += '<td'+(isWarn?' class="warn"':'')+'>'+fmt(v)+'</td>';
      }
      h += '<td>'+fmt(r.max)+'</td>';
      h += '<td'+((r.cv||0)>0.2?' class="warn"':'')+'>'+(r.cv||0).toFixed(2)+'</td>';
      h += '</tr>';
    }
    h += '</table></div>';
    h += sectionEnd();

    // Section D: Per-Region Histograms
    for (const r of regions) {
      if (r.hist && r.hist.length) {
        h += sectionStart('tr_hist_'+r.name, r.name+' Distribution');
        h += '<div id="trHist_'+r.name+'" class="chart" style="height:180px;"></div>';
        h += sectionEnd();
      }
    }

    // Section E: Perfetto Link
    if (D.traceMeta && D.traceMeta.trace_file) {
      h += sectionStart('tr_perfetto','Open in Perfetto');
      h += '<div style="font-size:11px;color:var(--dim);padding:4px">'+
        'Open trace in <a href="https://ui.perfetto.dev/" target="_blank" style="color:var(--accent)">Perfetto UI</a> \u2014 drag and drop the file:<br>'+
        '<code style="background:var(--bg3);padding:2px 6px;border-radius:3px;display:inline-block;margin-top:4px;user-select:all">'+
        D.traceMeta.trace_file+'</code></div>';
      h += sectionEnd();
    }

    // Section F: Per-Block-Warp Heatmap
    if (D.traceMeta && D.traceMeta.by_block_warp) {
      const bwEntries = Object.entries(D.traceMeta.by_block_warp);
      for (const [rname, rdata] of bwEntries) {
        if (rname.includes('total')) continue;
        const bwData = rdata.by_block_warp||[];
        if (bwData.length) {
          h += sectionStart('tr_hm_'+rname, rdata.name+' \u2014 Block\u00D7Warp Heatmap');
          h += '<div id="trHm_'+rname+'" class="chart" style="height:300px;"></div>';
          h += sectionEnd();
        }
      }
    }

    ct.innerHTML = h;

    // ── Charts ──

    // Region timing comparison
    if (regions.length) {
      const ch = mkChart('trCompChart');
      ch.setOption({
        backgroundColor: 'transparent',
        legend: { top: 0, textStyle: { color: '#656d76', fontSize: 10 } },
        grid: { left: 10, right: 10, top: 35, bottom: 30, containLabel: true },
        xAxis: { type: 'category', data: regions.map(r=>r.name), axisLabel: { color: '#656d76', fontSize: 10 } },
        yAxis: { type: 'value', name: 'Duration (clock ticks)', nameTextStyle: { color: '#656d76', fontSize: 9 },
          axisLabel: { color: '#656d76', fontSize: 9, formatter: v=>fmt(v) }, splitLine: { lineStyle: { color: '#d0d7de' } } },
        dataZoom: [{ type: 'inside' }],
        series: [
          { name: 'P50', type: 'bar', data: regions.map(r=>(r.percentiles||{}).p50||0), itemStyle: { color: '#0969da', borderRadius: [3,3,0,0] } },
          { name: 'Mean', type: 'bar', data: regions.map(r=>r.mean), itemStyle: { color: '#1a7f37', borderRadius: [3,3,0,0] } },
          { name: 'P95', type: 'bar', data: regions.map(r=>(r.percentiles||{}).p95||0), itemStyle: { color: '#bc4c00', borderRadius: [3,3,0,0] } },
          { name: 'P99', type: 'bar', data: regions.map(r=>(r.percentiles||{}).p99||0), itemStyle: { color: '#cf222e', borderRadius: [3,3,0,0] } },
        ],
        tooltip: { trigger: 'axis', formatter: function(params) {
          let s = params[0].axisValue+'<br>';
          for (const p of params) s += p.seriesName+': '+fmt(p.value)+' ticks<br>';
          return s;
        }},
      });
    }

    // Per-region histograms with proper axes
    for (const r of regions) {
      const el = document.getElementById('trHist_'+r.name);
      if (el && r.hist && r.hist.length) {
        const ch = mkChart('trHist_'+r.name);
        const nBins = r.hist.length;
        const hMin = r.hist_min||r.min||0;
        const hMax = r.hist_max||r.max||1;
        const binWidth = nBins>0?(hMax-hMin)/nBins:1;
        const binLabels = r.hist.map((_,i)=>{
          const lo = hMin + i*binWidth;
          return fmt(lo);
        });
        ch.setOption({
          backgroundColor: 'transparent',
          grid: { left: 10, right: 10, top: 10, bottom: 30, containLabel: true },
          xAxis: { type: 'category', data: binLabels, name: 'Duration (ticks)',
            nameLocation: 'center', nameGap: 20, nameTextStyle: { color: '#656d76', fontSize: 9 },
            axisLabel: { color: '#656d76', fontSize: 8, rotate: 30, interval: Math.max(0,Math.floor(nBins/10)-1) } },
          yAxis: { type: 'value', name: 'Probability', nameTextStyle: { color: '#656d76', fontSize: 9 },
            axisLabel: { color: '#656d76', fontSize: 8, formatter: v=>(v*100).toFixed(1)+'%' },
            splitLine: { lineStyle: { color: '#d0d7de' } } },
          dataZoom: [{ type: 'slider', bottom: 0, height: 15 }],
          series: [{ type: 'bar', data: r.hist, itemStyle: { color: '#0969da' }, barWidth: '90%' }],
          tooltip: { trigger: 'axis', formatter: function(p) {
            const i = p[0].dataIndex;
            const lo = hMin + i*binWidth;
            const hi = lo + binWidth;
            return 'Bin ['+fmt(lo)+', '+fmt(hi)+') ticks: '+(p[0].value*100).toFixed(2)+'% of events';
          }},
        });
      }
    }

    // Per-block-warp heatmap
    if (D.traceMeta && D.traceMeta.by_block_warp) {
      for (const [rname, rdata] of Object.entries(D.traceMeta.by_block_warp)) {
        if (rname.includes('total')) continue;
        const bwData = rdata.by_block_warp||[];
        if (!bwData.length) continue;
        const el = document.getElementById('trHm_'+rname);
        if (!el) continue;

        const blocks = [...new Set(bwData.map(d=>d.block))].sort((a,b)=>a-b);
        const warps = [...new Set(bwData.map(d=>d.warp))].sort((a,b)=>a-b);
        const heatData = [];
        let maxDur = 0;
        for (const d of bwData) {
          const bi = blocks.indexOf(d.block);
          const wi = warps.indexOf(d.warp);
          if (bi>=0 && wi>=0) {
            heatData.push([bi, wi, d.mean_dur||0]);
            maxDur = Math.max(maxDur, d.mean_dur||0);
          }
        }

        const ch = mkChart('trHm_'+rname);
        ch.setOption({
          backgroundColor:'transparent',
          grid:{left:50,right:50,top:10,bottom:30},
          xAxis:{type:'category',data:blocks.map(b=>'B'+b),name:'Block ID',
            nameLocation:'center',nameGap:20,nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:8},splitArea:{show:true}},
          yAxis:{type:'category',data:warps.map(w=>'W'+w),name:'Warp ID',
            nameTextStyle:{color:'#656d76',fontSize:9},
            axisLabel:{color:'#656d76',fontSize:8},splitArea:{show:true}},
          visualMap:{min:0,max:maxDur,calculable:true,orient:'vertical',right:0,top:'center',
            inRange:{color:['#dafbe1','#9a6700','#cf222e']},
            textStyle:{color:'#656d76',fontSize:9}},
          series:[{type:'heatmap',data:heatData,
            emphasis:{itemStyle:{shadowBlur:5,shadowColor:'rgba(0,0,0,0.3)'}}}],
          tooltip:{formatter:function(p){
            const d=bwData.find(x=>blocks.indexOf(x.block)===p.value[0]&&warps.indexOf(x.warp)===p.value[1]);
            if(!d)return '';
            return 'Block '+d.block+', Warp '+d.warp+'<br>Mean: '+fmt(d.mean_dur)+' ticks<br>CV: '+(d.cv_dur||0).toFixed(2);
          }},
        });
      }
    }
  }

  // ── Init ──
  buildOverview();
  builtTabs['ov'] = true;
});
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────

def extract_region_labels(source_code, trace_summary=None):
    labels = {}
    # 1. IKP_REGION_BEGIN(id, "name") macro
    for m in re.finditer(r'IKP_REGION_BEGIN\s*\(\s*(\d+)\s*,\s*"([^"]+)"', source_code):
        labels[int(m.group(1))] = m.group(2)
    # 2. set_region_names({"name0", "name1", ...}) — vector-style
    m = re.search(r'set_region_names\s*\(\s*\{([^}]+)\}', source_code)
    if m:
        names = re.findall(r'"([^"]+)"', m.group(1))
        for i, name in enumerate(names):
            if name.startswith("_"):
                continue  # skip placeholders like "_outside"
            if i not in labels:
                labels[i] = name
    # 3. enum Region with assigned values (e.g., kTotal = 1, // comment)
    for line in source_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue  # skip full-line comments
        em = re.match(r'\s*(\w+)\s*=\s*(\d+)\s*,?\s*//\s*(.+)', line)
        if em:
            rid = int(em.group(2))
            name = em.group(1).lstrip('k')  # kTotal -> Total, kLoadA -> LoadA
            if rid not in labels:
                labels[rid] = name
    # 4. Trace summary region names (highest priority — runtime-defined)
    if trace_summary:
        for r in trace_summary.get("regions", []):
            rid = r.get("region")
            name = r.get("name")
            if rid is not None and name:
                labels[int(rid)] = name
    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate IKP Explorer HTML")
    parser.add_argument("--demo-dir", required=True, help="Directory with profiler output")
    parser.add_argument("--source", required=True, help="CUDA source file")
    parser.add_argument("--output", default="explorer.html", help="Output HTML path")
    parser.add_argument("--nvdisasm", default="", help="Path to nvdisasm SASS file")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server after generating")
    args = parser.parse_args()

    # Load source
    with open(args.source) as f:
        source_code = f.read()
    # Collect all data
    sass_paths = sorted(glob.glob(os.path.join(args.demo_dir, "cupti", "sassmetrics_*.json")))
    pc2region_paths = sorted(glob.glob(os.path.join(args.demo_dir, "nvbit", "*", "pc2region_*.json")))
    region_stats_paths = sorted(glob.glob(os.path.join(args.demo_dir, "nvbit", "*", "region_stats_*.json")))

    sass_records, profiles = collect_sass_records(sass_paths)
    pc2region = collect_pc2region(pc2region_paths)
    region_stats = collect_region_stats(region_stats_paths)
    trace_summary = collect_trace_summary(args.demo_dir)

    labels = extract_region_labels(source_code, trace_summary)
    hotspots = collect_hotspots(args.demo_dir)
    pcsampling = collect_pcsampling(args.demo_dir)
    instrexec = collect_instrexec(args.demo_dir)
    locality = collect_locality(args.demo_dir)
    sass_profiles = collect_all_sass_profiles(args.demo_dir)

    source_line_regions = _parse_source_line_regions(source_code)
    per_line = aggregate_per_line(sass_records, pc2region, source_line_regions)
    sass_text = collect_sass_text(args.demo_dir, args.nvdisasm)
    sass_line_map = build_sass_line_map(sass_text)
    ptx_text = collect_ptx(args.demo_dir)
    ptx_line_map = build_ptx_line_map(ptx_text, args.source)
    nvdisasm_pc2src = build_nvdisasm_pc2src(sass_text, args.source)

    # Cross-reference: aggregate CUPTI per-PC data to NVBit regions
    sass_per_region, sass_per_region_coverage = aggregate_sass_per_region(args.demo_dir, pc2region)
    instrexec_per_region = aggregate_instrexec_per_region(instrexec, source_line_regions)
    pcsamp_per_region = aggregate_pcsamp_per_region(pcsampling, pc2region)
    mem_trace = collect_mem_trace(args.demo_dir)

    data = build_data(source_code, args.source, per_line, labels, region_stats,
                      sass_text, sass_line_map, ptx_text, ptx_line_map, trace_summary, hotspots,
                      pcsampling, instrexec, locality, sass_profiles, profiles,
                      source_line_regions, pc2region, nvdisasm_pc2src,
                      sass_per_region, sass_per_region_coverage,
                      instrexec_per_region, pcsamp_per_region, mem_trace)

    data_json = json.dumps(data, separators=(',', ':'))
    html = TEMPLATE.replace("__DATA__", data_json)

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.demo_dir, out_path)
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Generated: {out_path} ({len(html)//1024} KB)")

    if args.serve:
        import http.server
        import socketserver
        port = 8080
        os.chdir(os.path.dirname(out_path))
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Serving at http://localhost:{port}/{os.path.basename(out_path)}")
            httpd.serve_forever()


if __name__ == "__main__":
    main()
