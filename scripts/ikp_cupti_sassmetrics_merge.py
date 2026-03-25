#!/usr/bin/env python3
import argparse
import glob
import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_paths(patterns):
    out = []
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            out.extend(hits)
        else:
            out.append(pat)
    return out


def normalize_pc2region_entry(entry):
    pc = entry.get("pc_offset", entry.get("pcOffset"))
    if pc is None:
        return None
    func_index = entry.get("function_index", entry.get("functionIndex"))
    func_name = entry.get("function_name", entry.get("functionName"))
    func_name_mangled = entry.get("function_name_mangled", entry.get("functionNameMangled"))
    dominant = entry.get("dominant_region")
    if dominant is None and isinstance(entry.get("region"), int):
        dominant = entry["region"]
    regions = entry.get("regions")
    counts = entry.get("region_exec_counts")
    dominant_frac = entry.get("dominant_frac")
    ambiguity = entry.get("ambiguity_entropy_norm")
    if ambiguity is None:
        ambiguity = entry.get("ambiguity_score")
    if dominant is None and regions:
        if counts and len(counts) == len(regions):
            total = sum(counts)
            if total > 0:
                idx = max(range(len(counts)), key=lambda i: counts[i])
                dominant = regions[idx]
                dominant_frac = counts[idx] / total
        else:
            dominant = regions[0]
    return {
        "pc": int(pc),
        "function_index": func_index,
        "function_name": func_name,
        "function_name_mangled": func_name_mangled,
        "dominant_region": dominant,
        "dominant_frac": dominant_frac,
        "ambiguity": ambiguity,
    }


def load_pc2region(paths):
    mapping_by_cubin = {}
    global_map = {}
    warnings = []
    collisions = 0
    saw_function_dim = False
    for path in paths:
        data = load_json(path)
        pc2region = data.get("pc2region")
        if not pc2region:
            warnings.append(f"pc2region missing or empty in {path}")
            continue
        cubin = data.get("cubinCrc")
        # Many pc2region generators emit cubinCrc=0 when unknown; treat that as global.
        target = global_map if cubin is None or int(cubin) == 0 else mapping_by_cubin.setdefault(int(cubin), {})
        for entry in pc2region:
            norm = normalize_pc2region_entry(entry)
            if not norm:
                continue
            pc = int(norm["pc"])
            key = pc
            keys = []
            if norm.get("function_index") is not None:
                keys = [(int(norm["function_index"]), pc)]
                saw_function_dim = True
            else:
                name_keys = []
                if norm.get("function_name_mangled"):
                    name_keys.append(str(norm["function_name_mangled"]))
                if norm.get("function_name"):
                    name_keys.append(str(norm["function_name"]))
                name_keys = list(dict.fromkeys(name_keys))  # unique, keep order
                if name_keys:
                    keys = [(nk, pc) for nk in name_keys]
                    saw_function_dim = True
                else:
                    keys = [pc]

            for k in keys:
                prev = target.get(k)
                if prev is not None and prev.get("dominant_region") != norm.get("dominant_region"):
                    collisions += 1
                    continue
                target[k] = norm

    if collisions:
        warnings.append(f"pc2region key collisions detected: {collisions} (mapping kept first-seen)")
    if not saw_function_dim:
        warnings.append(
            "pc2region entries have no function_index/function_name; join key falls back to pcOffset only (may be ambiguous if pcOffset is function-relative)"
        )
    return mapping_by_cubin, global_map, warnings


def load_aggregation(metrics_json_path):
    aggregation = {}
    if not metrics_json_path:
        return aggregation
    try:
        data = load_json(metrics_json_path)
    except FileNotFoundError:
        return aggregation
    for name, spec in data.get("aggregation", {}).items():
        kind = spec.get("kind", "SUM")
        denom = spec.get("denominator")
        aggregation[name] = {"kind": kind, "denominator": denom}
    return aggregation


def get_agg_spec(aggregation, metric):
    return aggregation.get(metric, {"kind": "SUM", "denominator": None})


def merge_records(raw, pc2region_by_cubin, global_map, aggregation, ambiguity_threshold):
    region_stats = {}
    line_stats = {}
    unknown_pc_records = 0
    ambiguous_pc_records = 0

    for record in raw.get("records", []):
        cubin = record.get("cubinCrc")
        pc = record.get("pcOffset")
        fidx = record.get("functionIndex")
        fname = record.get("functionName")
        mapping = None
        if cubin is not None:
            per = pc2region_by_cubin.get(int(cubin), {})
            if fidx is not None:
                mapping = per.get((int(fidx), int(pc)))
            if mapping is None and fname:
                mapping = per.get((str(fname), int(pc)))
            if mapping is None and pc is not None:
                mapping = per.get(int(pc))
        if mapping is None and pc is not None:
            if fidx is not None:
                mapping = global_map.get((int(fidx), int(pc)))
            if mapping is None and fname:
                mapping = global_map.get((str(fname), int(pc)))
            if mapping is None:
                mapping = global_map.get(int(pc))
        if not mapping or mapping.get("dominant_region") is None:
            unknown_pc_records += 1
            continue
        if mapping.get("ambiguity") is not None and mapping["ambiguity"] > ambiguity_threshold:
            ambiguous_pc_records += 1
            unknown_pc_records += 1
            continue

        region_id = int(mapping["dominant_region"])
        region = region_stats.setdefault(region_id, {
            "pc_records": 0,
            "metrics_sum": {},
            "metrics_weighted": {},
        })
        region["pc_records"] += 1

        for metric, value in record.get("metrics", {}).items():
            spec = get_agg_spec(aggregation, metric)
            if spec["kind"] == "WEIGHTED_AVG":
                denom = spec.get("denominator")
                denom_val = record.get("metrics", {}).get(denom)
                if denom_val is None:
                    continue
                accum = region["metrics_weighted"].setdefault(metric, {"num": 0.0, "den": 0.0})
                accum["num"] += value * denom_val
                accum["den"] += denom_val
            else:
                region["metrics_sum"][metric] = region["metrics_sum"].get(metric, 0) + value

        src = record.get("source")
        if src and src.get("file") and src.get("line"):
            key = f"{src['file']}:{src['line']}"
            line = line_stats.setdefault(key, {
                "file": src["file"],
                "line": src["line"],
                "metrics_sum": {},
                "metrics_weighted": {},
            })
            for metric, value in record.get("metrics", {}).items():
                spec = get_agg_spec(aggregation, metric)
                if spec["kind"] == "WEIGHTED_AVG":
                    denom = spec.get("denominator")
                    denom_val = record.get("metrics", {}).get(denom)
                    if denom_val is None:
                        continue
                    accum = line["metrics_weighted"].setdefault(metric, {"num": 0.0, "den": 0.0})
                    accum["num"] += value * denom_val
                    accum["den"] += denom_val
                else:
                    line["metrics_sum"][metric] = line["metrics_sum"].get(metric, 0) + value

    return region_stats, line_stats, unknown_pc_records, ambiguous_pc_records


def finalize_stats(region_stats, line_stats):
    regions_out = []
    for region_id, stats in region_stats.items():
        metrics = dict(stats["metrics_sum"])
        for metric, accum in stats["metrics_weighted"].items():
            if accum["den"] > 0:
                metrics[metric] = accum["num"] / accum["den"]
        regions_out.append({
            "region_id": region_id,
            "pc_records": stats["pc_records"],
            "metrics": metrics,
        })

    lines_out = []
    for _, stats in line_stats.items():
        metrics = dict(stats["metrics_sum"])
        for metric, accum in stats["metrics_weighted"].items():
            if accum["den"] > 0:
                metrics[metric] = accum["num"] / accum["den"]
        lines_out.append({
            "file": stats["file"],
            "line": stats["line"],
            "metrics": metrics,
        })
    return regions_out, lines_out


def main():
    parser = argparse.ArgumentParser(description="Merge CUPTI SASS metrics with NVBit pc2region.")
    parser.add_argument("--sassmetrics", required=True, help="sassmetrics_raw.json")
    parser.add_argument("--pc2region", required=True, nargs="+", help="pc2region JSON path or glob")
    parser.add_argument("--metrics-json", default="./include/cupti/region_profiler/metrics_profiles.json",
                        help="metrics_profiles.json for aggregation rules")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--ambiguity-threshold", type=float, default=0.2,
                        help="ambiguity score threshold (PCs above this are treated as unknown)")
    args = parser.parse_args()

    raw = load_json(args.sassmetrics)
    pc2region_paths = collect_paths(args.pc2region)
    pc2region_by_cubin, global_map, warnings = load_pc2region(pc2region_paths)
    aggregation = load_aggregation(args.metrics_json)

    region_stats, line_stats, unknown_pc_records, ambiguous_pc_records = merge_records(
        raw, pc2region_by_cubin, global_map, aggregation, args.ambiguity_threshold
    )
    regions_out, lines_out = finalize_stats(region_stats, line_stats)

    output = {
        "tool": "ikp_cupti_sassmetrics_merge",
        "version": 1,
        "inputs": {
            "sassmetrics": args.sassmetrics,
            "pc2region": pc2region_paths,
            "metrics_json": args.metrics_json,
        },
        "summary": {
            "total_records": len(raw.get("records", [])),
            "unknown_pc_records": unknown_pc_records,
            "unknown_pc_fraction": (unknown_pc_records / max(len(raw.get("records", [])), 1)),
            "ambiguous_pc_records": ambiguous_pc_records,
        },
        "regions": regions_out,
        "lines": lines_out,
        "warnings": warnings,
    }

    if output["summary"]["unknown_pc_fraction"] > 0.5:
        output["warnings"].append(
            "unknown_pc_fraction is high; join may be failing (pcOffset semantics mismatch or missing function disambiguation)"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
