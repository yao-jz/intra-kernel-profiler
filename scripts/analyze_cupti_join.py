#!/usr/bin/env python3
"""Analyze NVBit + CUPTI join results.

Usage:
    python3 scripts/analyze_cupti_join.py --nvbit-dir DIR --cupti-dir DIR [--labels R0:name,R1:name,...]
"""
import argparse
import json
import glob
import os
import sys


def load_pc2region(nvbit_dir):
    """Load pc2region mapping from NVBit output."""
    pc2region = {}
    for f in sorted(glob.glob(os.path.join(nvbit_dir, "pc2region_*.json"))):
        with open(f) as fh:
            data = json.load(fh)
        for entry in data.get("pc2region", []):
            pc2region[entry["pc_offset"]] = entry["dominant_region"]
        print(f"  Loaded {len(data.get('pc2region', []))} PC entries from {os.path.basename(f)}")
    return pc2region


def load_region_stats(nvbit_dir):
    """Load NVBit region stats."""
    regions = {}
    for f in sorted(glob.glob(os.path.join(nvbit_dir, "region_stats_*.json"))):
        with open(f) as fh:
            data = json.load(fh)
        for r in data.get("regions", []):
            regions[r["region"]] = r
    return regions


def merge_sass_records(sass_path):
    """Load SASS metrics JSON and merge records by pcOffset."""
    with open(sass_path) as f:
        data = json.load(f)
    merged = {}
    for r in data.get("records", []):
        pc = r["pcOffset"]
        if pc not in merged:
            merged[pc] = {}
        for m, v in r.get("metrics", {}).items():
            if v > 0:
                merged[pc][m] = v
    return merged, data


def join_by_region(merged_records, pc2region):
    """Join CUPTI per-PC metrics to NVBit regions."""
    region_metrics = {}
    matched = 0
    unmatched = 0
    for pc, metrics in merged_records.items():
        region = pc2region.get(pc, -1)
        if region == -1:
            unmatched += 1
            continue
        matched += 1
        if region not in region_metrics:
            region_metrics[region] = {}
        for m, v in metrics.items():
            region_metrics[region][m] = region_metrics[region].get(m, 0) + v
    return region_metrics, matched, unmatched


def main():
    parser = argparse.ArgumentParser(description="NVBit + CUPTI join analysis")
    parser.add_argument("--nvbit-dir", required=True, help="NVBit output directory (contains pc2region_*.json)")
    parser.add_argument("--cupti-dir", required=True, help="CUPTI output directory (contains sassmetrics_*.json)")
    parser.add_argument("--labels", default="", help="Region labels: 0:outside,1:compute,2:store")
    args = parser.parse_args()

    labels = {}
    if args.labels:
        for pair in args.labels.split(","):
            k, v = pair.split(":")
            labels[int(k)] = v

    # Load NVBit data
    print("=== NVBit Data ===")
    pc2region = load_pc2region(args.nvbit_dir)
    print(f"  Total PCs mapped: {len(pc2region)}")

    nvbit_stats = load_region_stats(args.nvbit_dir)
    if nvbit_stats:
        print("\n  NVBit instruction counts:")
        for rid in sorted(nvbit_stats.keys()):
            r = nvbit_stats[rid]
            label = labels.get(rid, "")
            label_str = f" ({label})" if label else ""
            print(f"    Region {rid}{label_str}: inst_total={r.get('inst_total', 0):>14,}")

    # Process each CUPTI SASS metrics file
    sass_files = sorted(glob.glob(os.path.join(args.cupti_dir, "sassmetrics_*.json")))
    if not sass_files:
        print(f"\nNo sassmetrics_*.json files found in {args.cupti_dir}")
        return

    for sass_path in sass_files:
        profile_name = os.path.basename(sass_path).replace("sassmetrics_", "").replace(".json", "")
        merged, raw_data = merge_sass_records(sass_path)
        region_metrics, matched, unmatched = join_by_region(merged, pc2region)

        print(f"\n{'='*60}")
        print(f"Profile: {profile_name} ({len(raw_data.get('records', []))} raw records, {len(merged)} unique PCs)")
        print(f"Join: {matched} matched, {unmatched} unmatched PCs")

        for rid in sorted(region_metrics.keys()):
            m = region_metrics[rid]
            label = labels.get(rid, "")
            label_str = f" ({label})" if label else ""
            print(f"\n  Region {rid}{label_str}:")
            for metric, val in sorted(m.items()):
                print(f"    {metric}: {val:>14,}")

            # Derived metrics
            ie = m.get("smsp__sass_inst_executed", 0)
            te = m.get("smsp__sass_thread_inst_executed", 0)
            tp = m.get("smsp__sass_thread_inst_executed_pred_on", 0)
            if ie > 0 and te > 0:
                active = te / (ie * 32)
                pred = tp / te
                print(f"    --- derived ---")
                print(f"    active_thread_ratio:  {active:.4f}  (1.0 = all 32 threads active)")
                print(f"    pred_on_ratio:        {pred:.4f}  (1.0 = no predicated-off threads)")

            actual_sectors = m.get("smsp__sass_sectors_mem_global", 0)
            ideal_sectors = m.get("smsp__sass_sectors_mem_global_ideal", 0)
            if actual_sectors > 0 and ideal_sectors > 0:
                print(f"    --- memory efficiency ---")
                print(f"    coalescing_ratio:     {ideal_sectors/actual_sectors:.4f}  (1.0 = perfect coalescing)")


if __name__ == "__main__":
    main()
