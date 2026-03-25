#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_pc2region_entry(entry):
    pc = entry.get("pc_offset", entry.get("pcOffset"))
    if pc is None:
        return None
    func_index = entry.get("function_index", entry.get("functionIndex"))
    func_name = entry.get("function_name", entry.get("functionName"))
    func_name_mangled = entry.get("function_name_mangled", entry.get("functionNameMangled"))
    dominant = entry.get("dominant_region")
    if dominant is None:
        if "region" in entry and isinstance(entry["region"], int):
            dominant = entry["region"]
    regions = entry.get("regions")
    counts = entry.get("region_exec_counts")
    dominant_frac = None
    if dominant is None and regions:
        if counts and len(counts) == len(regions):
            total = sum(counts)
            if total > 0:
                max_idx = max(range(len(counts)), key=lambda i: counts[i])
                dominant = regions[max_idx]
                dominant_frac = counts[max_idx] / total
        else:
            dominant = regions[0]
    if dominant_frac is None and entry.get("dominant_frac") is not None:
        dominant_frac = entry.get("dominant_frac")
    ambiguity = entry.get("ambiguity_entropy_norm")
    if ambiguity is None:
        ambiguity = entry.get("ambiguity_score")
    return {
        "pc": pc,
        "function_index": func_index,
        "function_name": func_name,
        "function_name_mangled": func_name_mangled,
        "dominant_region": dominant,
        "regions": regions,
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
        if cubin is None or int(cubin) == 0:
            target = global_map
        else:
            target = mapping_by_cubin.setdefault(int(cubin), {})
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
                    # Prefer the more confident mapping when possible.
                    def score(m):
                        # Higher dominant_frac is better; lower ambiguity is better.
                        df = m.get("dominant_frac")
                        if df is None:
                            df = -1.0
                        amb = m.get("ambiguity")
                        if amb is None:
                            amb = 1e9
                        return (float(df), -float(amb))

                    if score(norm) > score(prev):
                        target[k] = norm
                    continue
                target[k] = norm

    if collisions:
        warnings.append(f"pc2region key collisions detected: {collisions} (mapping kept first-seen)")
    if not saw_function_dim:
        warnings.append(
            "pc2region entries have no function_index/function_name; join key falls back to pcOffset only (may be ambiguous if pcOffset is function-relative)"
        )
    return mapping_by_cubin, global_map, warnings


def build_invocation_index(invocations):
    by_corr = {}
    for inv in invocations:
        corr = inv.get("correlation_id")
        if corr is None:
            corr = inv.get("correlationId")
        if corr:
            by_corr[int(corr)] = inv
    return by_corr


def _stall_reason_table_from_entries(entries):
    table = {}
    for entry in entries or []:
        idx = entry.get("index")
        name = entry.get("name")
        if idx is None:
            continue
        table[int(idx)] = name if name else f"reason_{idx}"
    return table


def stall_reason_tables(raw):
    """
    Returns:
      legacy_table: dict[int->str] (first-seen, backward compatible)
      by_context_uid: dict[int context_uid -> dict[int->str]]
    """
    legacy = _stall_reason_table_from_entries(raw.get("stall_reason_table", []))
    by_ctx = {}
    for t in raw.get("stall_reason_tables", []) or []:
        ctx = t.get("context_uid")
        if ctx is None:
            continue
        by_ctx[int(ctx)] = _stall_reason_table_from_entries(t.get("entries", []))
    return legacy, by_ctx


def collect_paths(patterns):
    out = []
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            out.extend(hits)
        else:
            out.append(pat)
    return out


def merge_pcsampling(raw, pc_maps, global_map, ambiguity_threshold):
    invocations = raw.get("invocations", [])
    inv_by_corr = build_invocation_index(invocations)
    legacy_stall, stall_by_ctx = stall_reason_tables(raw)
    range_ctx = {}
    for r in raw.get("ranges", []) or []:
        rid = r.get("range_id", r.get("rangeId"))
        ctx = r.get("context_uid")
        if rid is None or ctx is None:
            continue
        range_ctx[int(rid)] = int(ctx)

    agg = {}
    for record in raw.get("pc_records", []):
        corr = int(record.get("correlationId", 0))
        inv = inv_by_corr.get(corr)
        ctx_uid = None
        if inv is not None:
            inv_uid = inv.get("invocation_uid")
            if not inv_uid:
                inv_uid = f"ctx{inv.get('context_uid', 0)}-corr{corr}"
            ctx_uid = inv.get("context_uid")
        else:
            inv_uid = "unknown"
            rid = record.get("rangeId", record.get("range_id"))
            if rid is not None:
                ctx_uid = range_ctx.get(int(rid))

        bucket = agg.setdefault(inv_uid, {
            "invocation": inv,
            "region_stats": {},
            "total_samples": 0,
            "unknown_pc_samples": 0,
            "unknown_pc_records": 0,
            "ambiguous_samples": 0,
        })

        samples = record.get("stall", [])
        total = sum(int(s.get("samples", 0)) for s in samples)
        bucket["total_samples"] += total

        cubin = record.get("cubinCrc")
        pc = record.get("pcOffset")
        fidx = record.get("functionIndex")
        fname = record.get("functionName")
        mapping = None
        if cubin is not None:
            per = pc_maps.get(int(cubin), {})
            if fidx is not None:
                mapping = per.get((int(fidx), int(pc)))
            if mapping is None and fname:
                mapping = per.get((str(fname), int(pc)))
            if mapping is None:
                mapping = per.get(int(pc))
        if mapping is None and pc is not None:
            if fidx is not None:
                mapping = global_map.get((int(fidx), int(pc)))
            if mapping is None and fname:
                mapping = global_map.get((str(fname), int(pc)))
            if mapping is None:
                mapping = global_map.get(int(pc))

        if not mapping or mapping.get("dominant_region") is None:
            bucket["unknown_pc_samples"] += total
            bucket["unknown_pc_records"] += 1
            continue

        if mapping.get("ambiguity") is not None and mapping["ambiguity"] > ambiguity_threshold:
            # Avoid hard-assigning highly ambiguous PCs.
            bucket["ambiguous_samples"] += total
            bucket["unknown_pc_samples"] += total
            bucket["unknown_pc_records"] += 1
            continue

        region_id = int(mapping["dominant_region"])

        region = bucket["region_stats"].setdefault(region_id, {
            "stall_total": 0,
            "stall_reason_samples": {},
            "pc_records": 0,
        })
        region["stall_total"] += total
        region["pc_records"] += 1
        stall_names = legacy_stall
        if ctx_uid is not None:
            stall_names = stall_by_ctx.get(int(ctx_uid), legacy_stall)
        for s in samples:
            reason_idx = int(s.get("reasonIndex", 0))
            name = stall_names.get(reason_idx, f"reason_{reason_idx}")
            region["stall_reason_samples"][name] = region["stall_reason_samples"].get(name, 0) + int(
                s.get("samples", 0)
            )

    return agg


def finalize_output(raw, agg, pc2region_paths, warnings):
    output = {
        "tool": "ikp_cupti_pcsamp_merge",
        "version": 1,
        "inputs": {
            "pcsampling": raw.get("source_path", ""),
            "pc2region": pc2region_paths,
        },
        "collection_mode": raw.get("collection_mode"),
        "invocations": [],
        "unattributed": {},
        "summary": {},
        "warnings": warnings,
    }

    total_samples = 0
    unknown_samples = 0
    for inv_uid, entry in agg.items():
        total_samples += entry["total_samples"]
        unknown_samples += entry["unknown_pc_samples"]
        if inv_uid == "unknown":
            output["unattributed"] = {
                "total_samples": entry["total_samples"],
                "unknown_pc_samples": entry["unknown_pc_samples"],
                "unknown_pc_records": entry["unknown_pc_records"],
                "ambiguous_samples": entry["ambiguous_samples"],
            }
            continue

        inv = entry["invocation"] or {}
        inv_out = {
            "invocation_uid": inv_uid,
            "kernel_name": inv.get("kernel_name", ""),
            "context_uid": inv.get("context_uid"),
            "correlation_id": inv.get("correlation_id"),
            "grid": inv.get("grid"),
            "block": inv.get("block"),
            "stream": inv.get("stream"),
            "selected": inv.get("selected"),
            "total_samples": entry["total_samples"],
            "unknown_pc_samples": entry["unknown_pc_samples"],
            "unknown_pc_records": entry["unknown_pc_records"],
            "ambiguous_samples": entry["ambiguous_samples"],
            "regions": [],
        }

        for region_id, stats in entry["region_stats"].items():
            stall_total = stats["stall_total"]
            stall_pct = {}
            if stall_total > 0:
                for k, v in stats["stall_reason_samples"].items():
                    stall_pct[k] = v / stall_total
            top_reason = None
            if stats["stall_reason_samples"]:
                top_reason = max(stats["stall_reason_samples"].items(), key=lambda kv: kv[1])[0]
            inv_out["regions"].append({
                "region_id": region_id,
                "stall_total": stall_total,
                "stall_reason_samples": stats["stall_reason_samples"],
                "stall_pct": stall_pct,
                "top_reason": top_reason,
                "pc_records": stats["pc_records"],
            })
        output["invocations"].append(inv_out)

    output["summary"] = {
        "total_samples": total_samples,
        "unknown_pc_samples": unknown_samples,
        "unknown_pc_fraction": (unknown_samples / total_samples) if total_samples else 0.0,
    }
    if output["summary"]["unknown_pc_fraction"] > 0.5:
        output["warnings"].append(
            "unknown_pc_fraction is high; join may be failing (pcOffset semantics mismatch or missing function disambiguation)"
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Merge CUPTI PC sampling with NVBit pc2region.")
    parser.add_argument("--pcsampling", required=True, help="pcsampling_raw.json")
    parser.add_argument("--pc2region", required=True, nargs="+",
                        help="pc2region JSON path or glob (repeatable)")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--ambiguity-threshold", type=float, default=0.2,
                        help="ambiguity score threshold to flag ambiguous PCs")
    args = parser.parse_args()

    pcsamp = load_json(args.pcsampling)
    pcsamp["source_path"] = args.pcsampling

    pc2region_paths = collect_paths(args.pc2region)
    mapping_by_cubin, global_map, warnings = load_pc2region(pc2region_paths)
    if not mapping_by_cubin and not global_map:
        warnings.append("no pc2region mappings loaded")

    agg = merge_pcsampling(pcsamp, mapping_by_cubin, global_map, args.ambiguity_threshold)
    output = finalize_output(pcsamp, agg, pc2region_paths, warnings)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=False)
        f.write("\n")


if __name__ == "__main__":
    main()
