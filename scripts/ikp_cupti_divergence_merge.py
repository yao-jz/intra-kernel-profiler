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
                name_keys = list(dict.fromkeys(name_keys))
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


def merge(raw, pc2region_by_cubin, global_map):
    region_stats = {}
    line_stats = {}
    unknown = 0

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
            unknown += 1
            continue

        region_id = int(mapping["dominant_region"])
        threads = record.get("threadsExecuted", 0)
        executed = record.get("executed", 0)
        not_pred_off = record.get("notPredOffThreadsExecuted", 0)
        pred_off = max(0, threads - not_pred_off)

        region = region_stats.setdefault(region_id, {
            "sum_threads": 0,
            "sum_executed": 0,
            "sum_pred_off": 0,
            "pc_records": 0,
        })
        region["sum_threads"] += threads
        region["sum_executed"] += executed
        region["sum_pred_off"] += pred_off
        region["pc_records"] += 1

        src = record.get("source")
        if src and src.get("file") and src.get("line"):
            key = f"{src['file']}:{src['line']}"
            line = line_stats.setdefault(key, {
                "file": src["file"],
                "line": src["line"],
                "sum_threads": 0,
                "sum_executed": 0,
                "sum_pred_off": 0,
            })
            line["sum_threads"] += threads
            line["sum_executed"] += executed
            line["sum_pred_off"] += pred_off

    return region_stats, line_stats, unknown


def finalize(region_stats, line_stats):
    regions_out = []
    for region_id, stats in region_stats.items():
        avg_active = stats["sum_threads"] / stats["sum_executed"] if stats["sum_executed"] else 0
        warp_lane_eff = avg_active / 32.0 if avg_active else 0
        pred_off_rate = stats["sum_pred_off"] / stats["sum_threads"] if stats["sum_threads"] else 0
        regions_out.append({
            "region_id": region_id,
            "pc_records": stats["pc_records"],
            "avg_active_lanes": avg_active,
            "warp_lane_eff": warp_lane_eff,
            "pred_off_rate": pred_off_rate,
        })

    lines_out = []
    for _, stats in line_stats.items():
        avg_active = stats["sum_threads"] / stats["sum_executed"] if stats["sum_executed"] else 0
        warp_lane_eff = avg_active / 32.0 if avg_active else 0
        pred_off_rate = stats["sum_pred_off"] / stats["sum_threads"] if stats["sum_threads"] else 0
        lines_out.append({
            "file": stats["file"],
            "line": stats["line"],
            "avg_active_lanes": avg_active,
            "warp_lane_eff": warp_lane_eff,
            "pred_off_rate": pred_off_rate,
        })

    lines_sorted = sorted(lines_out, key=lambda x: x["warp_lane_eff"])
    top_divergent = lines_sorted[:10]
    return regions_out, lines_out, top_divergent


def main():
    parser = argparse.ArgumentParser(description="Merge CUPTI InstructionExecution with NVBit pc2region.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--instrexec", help="instrexec_raw.json (InstructionExecution)")
    src.add_argument("--sassmetrics", help="sassmetrics_raw.json (derive divergence from SASS metrics)")
    parser.add_argument("--pc2region", required=True, nargs="+", help="pc2region JSON path or glob")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()

    raw = load_json(args.instrexec or args.sassmetrics)
    pc2region_paths = collect_paths(args.pc2region)
    pc2region_by_cubin, global_map, warnings = load_pc2region(pc2region_paths)

    if args.sassmetrics:
        # Derive lane efficiency / predication loss from SASS metrics:
        # - smsp__sass_inst_executed
        # - smsp__sass_thread_inst_executed
        # - smsp__sass_thread_inst_executed_pred_on
        needed = [
            "smsp__sass_inst_executed",
            "smsp__sass_thread_inst_executed",
            "smsp__sass_thread_inst_executed_pred_on",
        ]

        # Convert sassmetrics records into instrexec-like records for reuse.
        pseudo = {"records": []}
        for rec in raw.get("records", []):
            m = rec.get("metrics", {})
            if not all(k in m for k in needed):
                continue
            inst = m["smsp__sass_inst_executed"]
            thread_inst = m["smsp__sass_thread_inst_executed"]
            pred_on = m["smsp__sass_thread_inst_executed_pred_on"]
            # executed ~= inst (warp-issued). threadsExecuted ~= thread_inst
            pseudo_rec = {
                "cubinCrc": rec.get("cubinCrc"),
                "pcOffset": rec.get("pcOffset"),
                "executed": inst,
                "threadsExecuted": thread_inst,
                "notPredOffThreadsExecuted": pred_on,
                "source": rec.get("source"),
            }
            pseudo["records"].append(pseudo_rec)

        region_stats, line_stats, unknown = merge(pseudo, pc2region_by_cubin, global_map)
        regions_out, lines_out, top_divergent = finalize(region_stats, line_stats)
        input_kind = "sassmetrics"
    else:
        region_stats, line_stats, unknown = merge(raw, pc2region_by_cubin, global_map)
        regions_out, lines_out, top_divergent = finalize(region_stats, line_stats)
        input_kind = "instrexec"

    output = {
        "tool": "ikp_cupti_divergence_merge",
        "version": 1,
        "inputs": {
            "input_kind": input_kind,
            "instrexec": args.instrexec,
            "sassmetrics": args.sassmetrics,
            "pc2region": pc2region_paths,
        },
        "summary": {
            "total_records": len(raw.get("records", [])),
            "unknown_pc_records": unknown,
            "unknown_pc_fraction": (unknown / max(len(raw.get("records", [])), 1)),
        },
        "regions": regions_out,
        "lines": lines_out,
        "top_divergent_lines": top_divergent,
        "warnings": warnings,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
