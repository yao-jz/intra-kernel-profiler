#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict, deque


HIST_BOUNDS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def bin_index(dist):
    # dist is non-negative integer
    for i, b in enumerate(HIST_BOUNDS):
        if dist <= b:
            return i
    return len(HIST_BOUNDS)


class LRUStack:
    def __init__(self, max_size=None):
        self.stack = []
        self.max_size = max_size

    def access(self, line):
        if line in self.stack:
            idx = self.stack.index(line)
            dist = len(self.stack) - 1 - idx
            self.stack.pop(idx)
        else:
            dist = None
        self.stack.append(line)
        if self.max_size and len(self.stack) > self.max_size:
            self.stack.pop(0)
        return dist


class WorkingSetWindow:
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.count = Counter()
        self.samples = []

    def add(self, lines):
        self.queue.append(lines)
        for line in lines:
            self.count[line] += 1
        if len(self.queue) > self.size:
            old = self.queue.popleft()
            for line in old:
                self.count[line] -= 1
                if self.count[line] <= 0:
                    del self.count[line]
        if len(self.queue) == self.size:
            self.samples.append(len(self.count))

    def summary(self):
        if not self.samples:
            return {"avg": 0, "p50": 0, "p95": 0, "max": 0}
        vals = sorted(self.samples)
        n = len(vals)
        p50 = vals[int(0.5 * (n - 1))]
        p95 = vals[int(0.95 * (n - 1))]
        return {
            "avg": sum(vals) / n,
            "p50": p50,
            "p95": p95,
            "max": vals[-1],
        }


def normalize_space(rec):
    space = rec.get("space")
    if space:
        return space
    flags = rec.get("flags", 0)
    if flags & 4:
        return "global"
    if flags & 16:
        return "local"
    if flags & 8:
        return "shared"
    return "other"


def iter_trace(paths, max_records=None):
    count = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield path, rec
                count += 1
                if max_records and count >= max_records:
                    return


def extract_lines(rec, line_bytes, spaces):
    space = normalize_space(rec)
    if space not in spaces:
        return []
    addrs = rec.get("addrs", [])
    mask = rec.get("active_mask", 0)
    lines = set()
    for lane in range(min(32, len(addrs))):
        if mask & (1 << lane):
            addr = addrs[lane]
            if addr == 0:
                continue
            lines.add(addr // line_bytes)
    return list(lines)


def parse_region_stats(path):
    if not path:
        return {}
    paths = [path]
    if glob.has_magic(path):
        paths = sorted(glob.glob(path))
    if not paths:
        return {}
    with open(paths[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data.get("regions", []):
        out[str(r.get("region"))] = r
    return out


def main():
    ap = argparse.ArgumentParser(description="Offline locality stats from NVBit mem_trace JSONL.")
    ap.add_argument("--trace", action="append", default=[], help="mem_trace jsonl path (repeatable)")
    ap.add_argument("--trace-glob", default="", help="glob pattern for mem_trace jsonl")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--line-bytes", type=int, default=128, help="cache line size (bytes)")
    ap.add_argument("--window-records", default="128,512,2048", help="comma list of window sizes")
    ap.add_argument("--spaces", default="global,shared,other", help="comma list of spaces to include")
    ap.add_argument("--scope", default="global,cta,warp", help="comma list: global,cta,warp")
    ap.add_argument("--max-records", type=int, default=0, help="limit records for debug")
    ap.add_argument("--max-stack", type=int, default=0, help="max LRU stack size (0=unbounded)")
    ap.add_argument("--region-stats", default="", help="optional region_stats json to merge")
    args = ap.parse_args()

    paths = list(args.trace)
    if args.trace_glob:
        paths.extend(sorted(glob.glob(args.trace_glob)))
    if not paths:
        raise SystemExit("No trace files provided")

    spaces = set(s.strip() for s in args.spaces.split(",") if s.strip())
    scopes = set(s.strip() for s in args.scope.split(",") if s.strip())
    windows = [int(x) for x in args.window_records.split(",") if x.strip()]
    max_records = args.max_records if args.max_records > 0 else None
    max_stack = args.max_stack if args.max_stack > 0 else None

    region_stats = parse_region_stats(args.region_stats)

    # per region aggregates
    agg = {}

    def ensure_region(region):
        if region not in agg:
            agg[region] = {
                "records": 0,
                "lines_per_record": 0,
                "unique_lines_set": set(),
                "reuse": {
                    "global": {"cold": 0, "hist": [0] * (len(HIST_BOUNDS) + 1), "stacks": {}},
                    "cta": {"cold": 0, "hist": [0] * (len(HIST_BOUNDS) + 1), "stacks": {}},
                    "warp": {"cold": 0, "hist": [0] * (len(HIST_BOUNDS) + 1), "stacks": {}},
                },
                "ws_windows": {w: WorkingSetWindow(w) for w in windows},
                "line_to_warps": defaultdict(set),
                "line_to_ctas": defaultdict(set),
            }
        return agg[region]

    for _, rec in iter_trace(paths, max_records=max_records):
        region = str(rec.get("region", 0))
        cta = rec.get("cta", 0)
        warp = rec.get("warp", 0)
        lines = extract_lines(rec, args.line_bytes, spaces)
        if not lines:
            continue
        st = ensure_region(region)
        st["records"] += 1
        st["lines_per_record"] += len(lines)
        for line in lines:
            st["unique_lines_set"].add(line)
            st["line_to_warps"][line].add((cta, warp))
            st["line_to_ctas"][line].add(cta)

        # working set windows (global)
        for w in windows:
            st["ws_windows"][w].add(lines)

        # reuse distance
        if "global" in scopes:
            stack = st["reuse"]["global"]["stacks"].setdefault(
                "all", LRUStack(max_size=max_stack)
            )
            for line in lines:
                dist = stack.access(line)
                if dist is None:
                    st["reuse"]["global"]["cold"] += 1
                else:
                    st["reuse"]["global"]["hist"][bin_index(dist)] += 1
        if "cta" in scopes:
            stack = st["reuse"]["cta"]["stacks"].setdefault(
                cta, LRUStack(max_size=max_stack)
            )
            for line in lines:
                dist = stack.access(line)
                if dist is None:
                    st["reuse"]["cta"]["cold"] += 1
                else:
                    st["reuse"]["cta"]["hist"][bin_index(dist)] += 1
        if "warp" in scopes:
            key = (cta, warp)
            stack = st["reuse"]["warp"]["stacks"].setdefault(
                key, LRUStack(max_size=max_stack)
            )
            for line in lines:
                dist = stack.access(line)
                if dist is None:
                    st["reuse"]["warp"]["cold"] += 1
                else:
                    st["reuse"]["warp"]["hist"][bin_index(dist)] += 1

    # finalize output
    out = {
        "trace_files": paths,
        "line_bytes": args.line_bytes,
        "spaces": sorted(spaces),
        "scopes": sorted(scopes),
        "window_records": windows,
        "regions": {},
        "hist_bounds": HIST_BOUNDS,
    }

    for region, st in agg.items():
        unique_lines = len(st["unique_lines_set"])
        inter_warp_hist = Counter(len(v) for v in st["line_to_warps"].values())
        inter_cta_hist = Counter(len(v) for v in st["line_to_ctas"].values())
        total_lines = sum(inter_warp_hist.values())
        shared_lines = sum(v for k, v in inter_warp_hist.items() if k >= 2)
        avg_warps = 0.0
        if total_lines > 0:
            avg_warps = sum(k * v for k, v in inter_warp_hist.items()) / total_lines

        region_out = {
            "records": st["records"],
            "lines_per_record": st["lines_per_record"] / st["records"] if st["records"] else 0,
            "unique_lines": unique_lines,
            "lines_per_1k_records": (st["lines_per_record"] / st["records"] * 1000)
            if st["records"]
            else 0,
            "reuse_distance": {},
            "working_set": {},
            "inter_warp_sharing": {
                "lines_by_warps": dict(sorted(inter_warp_hist.items())),
                "shared_line_ratio": (shared_lines / total_lines) if total_lines else 0,
                "avg_warps_per_line": avg_warps,
            },
            "inter_cta_sharing": {
                "lines_by_ctas": dict(sorted(inter_cta_hist.items())),
            },
        }

        for scope in ["global", "cta", "warp"]:
            if scope not in scopes:
                continue
            region_out["reuse_distance"][scope] = {
                "cold": st["reuse"][scope]["cold"],
                "hist": st["reuse"][scope]["hist"],
            }

        for w in windows:
            region_out["working_set"][str(w)] = st["ws_windows"][w].summary()

        # merge inst_total if provided
        if region in region_stats:
            inst_total = region_stats[region].get("inst_total", 0)
            if inst_total:
                region_out["lines_per_k_inst"] = unique_lines / (inst_total / 1000.0)
            region_out["inst_total"] = inst_total

        out["regions"][region] = region_out

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

