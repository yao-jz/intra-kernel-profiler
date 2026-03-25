#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict, List


def die(msg: str) -> None:
    print(f"[validate_json][ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"[validate_json][WARN] {msg}", file=sys.stderr)


def load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        die(f"file not found: {path}")
    except json.JSONDecodeError as e:
        die(f"invalid JSON: {path}: {e}")


def require_keys(obj: Dict[str, Any], keys: List[str], ctx: str) -> None:
    for k in keys:
        if k not in obj:
            die(f"missing key {k!r} in {ctx}")


def validate_cupti_common(d: Dict[str, Any], path: str) -> None:
    require_keys(d, ["tool", "version", "pid", "timestamp_ns"], path)
    if not isinstance(d["tool"], str):
        die(f"{path}: tool must be string")


def validate_pcsamp(d: Dict[str, Any], path: str, require_nonempty: bool) -> None:
    validate_cupti_common(d, path)
    require_keys(d, ["invocations", "ranges", "pc_records", "warnings"], path)
    if require_nonempty and d.get("pc_records") == []:
        warn(f"{path}: pc_records is empty (could be permissions/disabled)")


def validate_sassmetrics(d: Dict[str, Any], path: str, require_nonempty: bool) -> None:
    validate_cupti_common(d, path)
    require_keys(d, ["invocations", "records", "warnings"], path)
    if require_nonempty and d.get("records") == []:
        warn(f"{path}: records is empty (could be permissions/disabled)")


def validate_instrexec(d: Dict[str, Any], path: str, require_nonempty: bool) -> None:
    validate_cupti_common(d, path)
    require_keys(d, ["invocations", "records", "warnings"], path)
    if require_nonempty and d.get("records") == []:
        warn(f"{path}: records is empty (often restricted by cluster policy)")


def validate_merge(d: Dict[str, Any], path: str) -> None:
    require_keys(d, ["tool", "version", "inputs", "warnings"], path)
    if "summary" not in d:
        warn(f"{path}: missing 'summary' (not fatal)")


def validate_nvbit_pc2region(d: Dict[str, Any], path: str) -> None:
    require_keys(d, ["pc2region", "pc2region_format_version"], path)
    pc2 = d.get("pc2region", [])
    if not isinstance(pc2, list):
        die(f"{path}: pc2region must be a list")
    # Spot-check a few entries (don't validate entire list to keep it cheap).
    for e in pc2[:20]:
        if not isinstance(e, dict):
            die(f"{path}: pc2region entry must be object")
        if "pc_offset" not in e and "pcOffset" not in e:
            die(f"{path}: pc2region entry missing pc_offset/pcOffset")
        if "dominant_region" not in e and "region" not in e:
            die(f"{path}: pc2region entry missing dominant_region/region")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight JSON validator for Intra-Kernel Profiler outputs.")
    ap.add_argument("paths", nargs="+", help="JSON files to validate")
    ap.add_argument(
        "--require-nonempty",
        action="store_true",
        help="Warn when primary records arrays are empty",
    )
    args = ap.parse_args()

    for p in args.paths:
        d = load(p)
        tool = d.get("tool")
        # NVBit pc2region doesn't have a 'tool' field; detect by keys.
        if tool is None and "pc2region" in d and "pc2region_format_version" in d:
            validate_nvbit_pc2region(d, p)
            continue

        if tool == "ikp_cupti_pcsamp":
            validate_pcsamp(d, p, args.require_nonempty)
        elif tool == "ikp_cupti_sassmetrics":
            validate_sassmetrics(d, p, args.require_nonempty)
        elif tool == "ikp_cupti_instrexec":
            validate_instrexec(d, p, args.require_nonempty)
        elif isinstance(tool, str) and tool.endswith("_merge"):
            validate_merge(d, p)
        else:
            # Still enforce it's valid JSON object; beyond that we just warn.
            warn(f"{p}: unknown tool={tool!r}, skipping strict validation")

    print("[validate_json] OK")


if __name__ == "__main__":
    main()

