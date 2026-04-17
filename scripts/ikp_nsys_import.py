#!/usr/bin/env python3
"""Import NVIDIA Nsight Systems (.nsys-rep) profiling data into IKP JSON format.

Converts an nsys report to SQLite via ``nsys export``, then queries the
database for kernel launches, memory operations, NVTX ranges, CUDA API
calls, and synchronization events.  Produces two JSON files consumed by
downstream IKP scripts (timeline merge, NCCL extraction, Explorer).

Usage:
    python3 scripts/ikp_nsys_import.py \\
        --nsys-rep report.nsys-rep \\
        --out-dir _out/nsys/ \\
        [--kernel-regex "kernel_name"] \\
        [--skip-export]

Requirements:
    - ``nsys`` CLI on PATH (unless --skip-export with pre-existing .sqlite)
    - Python 3.8+ (stdlib only: sqlite3, json, argparse, subprocess, re)
"""
import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys


# ── NSys SQLite export ───────────────────────────────────────────────

def export_nsys_to_sqlite(nsys_rep_path, sqlite_path):
    """Run ``nsys export --type=sqlite`` to produce a SQLite database."""
    nsys_bin = shutil.which("nsys")
    if nsys_bin is None:
        print("ERROR: 'nsys' not found in PATH.", file=sys.stderr)
        print("  Install NVIDIA Nsight Systems or add it to PATH.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        nsys_bin, "export",
        "--type=sqlite",
        "--output", sqlite_path,
        nsys_rep_path,
    ]
    print(f"  nsys export: {nsys_rep_path} -> {sqlite_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: nsys export failed (rc={result.returncode})", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(sqlite_path):
        print(f"ERROR: nsys export did not produce {sqlite_path}", file=sys.stderr)
        sys.exit(1)


# ── Schema discovery ─────────────────────────────────────────────────

def table_exists(conn, table_name):
    cur = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone()[0] > 0


def get_columns(conn, table_name):
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cur.fetchall()}


def col_or_null(columns, preferred, fallback=None):
    """Return the column name to use, or None if neither exists."""
    if preferred in columns:
        return preferred
    if fallback and fallback in columns:
        return fallback
    return None


# ── Kernel queries ───────────────────────────────────────────────────

def query_kernels(conn, kernel_regex=None):
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if not table_exists(conn, table):
        # Try alternative table names across nsys versions.
        for alt in ["CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"]:
            if table_exists(conn, alt):
                table = alt
                break
        else:
            print(f"  WARNING: no kernel activity table found", file=sys.stderr)
            return []

    cols = get_columns(conn, table)

    # Column name varies across nsys versions.
    name_col = col_or_null(cols, "shortName", "demangledName")
    demangled_col = col_or_null(cols, "demangledName", "shortName")
    mangled_col = col_or_null(cols, "mangledName")
    device_col = col_or_null(cols, "deviceId")
    stream_col = col_or_null(cols, "streamId")
    corr_col = col_or_null(cols, "correlationId")
    grid_x = col_or_null(cols, "gridX", "gridDimX")
    grid_y = col_or_null(cols, "gridY", "gridDimY")
    grid_z = col_or_null(cols, "gridZ", "gridDimZ")
    block_x = col_or_null(cols, "blockX", "blockDimX")
    block_y = col_or_null(cols, "blockY", "blockDimY")
    block_z = col_or_null(cols, "blockZ", "blockDimZ")
    reg_col = col_or_null(cols, "registersPerThread")
    smem_static = col_or_null(cols, "staticSharedMemory")
    smem_dynamic = col_or_null(cols, "dynamicSharedMemory")

    select_parts = ["start", "end"]
    labels = ["start", "end"]

    def add(col, label):
        if col:
            select_parts.append(col)
            labels.append(label)
        else:
            select_parts.append("NULL")
            labels.append(label)

    add(name_col, "name")
    add(demangled_col, "demangled_name")
    add(mangled_col, "mangled_name")
    add(device_col, "device_id")
    add(stream_col, "stream_id")
    add(corr_col, "correlation_id")
    add(grid_x, "grid_x")
    add(grid_y, "grid_y")
    add(grid_z, "grid_z")
    add(block_x, "block_x")
    add(block_y, "block_y")
    add(block_z, "block_z")
    add(reg_col, "registers_per_thread")
    add(smem_static, "static_shared_memory")
    add(smem_dynamic, "dynamic_shared_memory")

    query = f"SELECT {', '.join(select_parts)} FROM {table} ORDER BY start"
    rows = conn.execute(query).fetchall()

    # In some nsys versions, name columns are string IDs (integers) rather
    # than actual strings.  Try to resolve them from the StringIds table.
    string_ids = {}
    if table_exists(conn, "StringIds"):
        for srow in conn.execute("SELECT id, value FROM StringIds"):
            string_ids[srow[0]] = srow[1]

    def resolve_str(val):
        """Resolve a value that might be a string, an integer string-ID, or None."""
        if val is None:
            return ""
        if isinstance(val, int):
            return string_ids.get(val, str(val))
        return str(val)

    kernels = []
    for row in rows:
        rec = dict(zip(labels, row))
        name = resolve_str(rec.get("name")) or resolve_str(rec.get("demangled_name")) or ""
        if kernel_regex and not re.search(kernel_regex, name):
            continue
        start_ns = int(rec["start"])
        end_ns = int(rec["end"])
        kernels.append({
            "name": name,
            "demangled_name": resolve_str(rec.get("demangled_name")),
            "mangled_name": resolve_str(rec.get("mangled_name")),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "device_id": int(rec["device_id"]) if rec.get("device_id") is not None else 0,
            "stream_id": int(rec["stream_id"]) if rec.get("stream_id") is not None else 0,
            "correlation_id": int(rec["correlation_id"]) if rec.get("correlation_id") is not None else 0,
            "grid": [
                int(rec.get("grid_x") or 1),
                int(rec.get("grid_y") or 1),
                int(rec.get("grid_z") or 1),
            ],
            "block": [
                int(rec.get("block_x") or 1),
                int(rec.get("block_y") or 1),
                int(rec.get("block_z") or 1),
            ],
            "registers_per_thread": int(rec["registers_per_thread"]) if rec.get("registers_per_thread") is not None else None,
            "static_shared_memory": int(rec["static_shared_memory"]) if rec.get("static_shared_memory") is not None else None,
            "dynamic_shared_memory": int(rec["dynamic_shared_memory"]) if rec.get("dynamic_shared_memory") is not None else None,
        })
    return kernels


# ── Memory operations ────────────────────────────────────────────────

MEMCPY_KIND_MAP = {
    0: "unknown",
    1: "HtoD",
    2: "DtoH",
    3: "HtoH",
    4: "DtoD",
    8: "PtoP",
}


def query_memcpy(conn):
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not table_exists(conn, table):
        return []

    cols = get_columns(conn, table)
    bytes_col = col_or_null(cols, "bytes", "size")
    kind_col = col_or_null(cols, "copyKind", "kind")
    stream_col = col_or_null(cols, "streamId")
    corr_col = col_or_null(cols, "correlationId")

    select_parts = ["start", "end"]
    labels = ["start", "end"]

    def add(col, label):
        select_parts.append(col if col else "NULL")
        labels.append(label)

    add(bytes_col, "bytes")
    add(kind_col, "kind")
    add(stream_col, "stream_id")
    add(corr_col, "correlation_id")

    query = f"SELECT {', '.join(select_parts)} FROM {table} ORDER BY start"
    rows = conn.execute(query).fetchall()

    results = []
    for row in rows:
        rec = dict(zip(labels, row))
        start_ns = int(rec["start"])
        end_ns = int(rec["end"])
        kind_int = int(rec["kind"]) if rec.get("kind") is not None else 0
        results.append({
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "bytes": int(rec["bytes"]) if rec.get("bytes") is not None else 0,
            "kind": MEMCPY_KIND_MAP.get(kind_int, f"unknown_{kind_int}"),
            "kind_id": kind_int,
            "stream_id": int(rec["stream_id"]) if rec.get("stream_id") is not None else 0,
            "correlation_id": int(rec["correlation_id"]) if rec.get("correlation_id") is not None else 0,
        })
    return results


def query_memset(conn):
    table = "CUPTI_ACTIVITY_KIND_MEMSET"
    if not table_exists(conn, table):
        return []

    cols = get_columns(conn, table)
    bytes_col = col_or_null(cols, "bytes", "size")
    stream_col = col_or_null(cols, "streamId")

    select_parts = ["start", "end"]
    labels = ["start", "end"]

    def add(col, label):
        select_parts.append(col if col else "NULL")
        labels.append(label)

    add(bytes_col, "bytes")
    add(stream_col, "stream_id")

    query = f"SELECT {', '.join(select_parts)} FROM {table} ORDER BY start"
    rows = conn.execute(query).fetchall()

    results = []
    for row in rows:
        rec = dict(zip(labels, row))
        start_ns = int(rec["start"])
        end_ns = int(rec["end"])
        results.append({
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "bytes": int(rec["bytes"]) if rec.get("bytes") is not None else 0,
            "stream_id": int(rec["stream_id"]) if rec.get("stream_id") is not None else 0,
        })
    return results


# ── NVTX ranges ──────────────────────────────────────────────────────

def query_nvtx(conn):
    table = "NVTX_EVENTS"
    if not table_exists(conn, table):
        return []

    cols = get_columns(conn, table)
    text_col = col_or_null(cols, "text")
    domain_col = col_or_null(cols, "domainId")
    range_col = col_or_null(cols, "rangeId")
    # Some nsys versions use 'eventType' to distinguish push/pop/range.
    type_col = col_or_null(cols, "eventType")

    select_parts = ["start", "end"]
    labels = ["start", "end"]

    def add(col, label):
        select_parts.append(col if col else "NULL")
        labels.append(label)

    add(text_col, "text")
    add(domain_col, "domain_id")
    add(range_col, "range_id")
    add(type_col, "event_type")

    query = f"SELECT {', '.join(select_parts)} FROM {table} ORDER BY start"
    rows = conn.execute(query).fetchall()

    # Resolve string IDs — in some nsys versions, text fields are integer
    # references into the StringIds table.
    string_ids = {}
    if table_exists(conn, "StringIds"):
        for srow in conn.execute("SELECT id, value FROM StringIds"):
            string_ids[srow[0]] = srow[1]

    def resolve_str(val):
        if val is None:
            return ""
        if isinstance(val, int):
            return string_ids.get(val, str(val))
        return str(val)

    # Also try to load domain names from NVTX_DOMAIN tables.
    domain_names = {}
    for dtable in ["NVTX_DOMAINS"]:
        if table_exists(conn, dtable):
            dcols = get_columns(conn, dtable)
            if "domainId" in dcols and "text" in dcols:
                for drow in conn.execute(f"SELECT domainId, text FROM {dtable}"):
                    domain_names[int(drow[0])] = resolve_str(drow[1])

    results = []
    for row in rows:
        rec = dict(zip(labels, row))
        start_ns = int(rec["start"]) if rec["start"] is not None else 0
        end_ns = int(rec["end"]) if rec["end"] is not None else start_ns
        text = resolve_str(rec.get("text"))
        domain_id = int(rec["domain_id"]) if rec.get("domain_id") is not None else 0
        domain_name = domain_names.get(domain_id, "")
        results.append({
            "text": text,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": max(0, end_ns - start_ns),
            "domain_id": domain_id,
            "domain": domain_name,
            "range_id": int(rec["range_id"]) if rec.get("range_id") is not None else 0,
        })
    return results


# ── CUDA Runtime API ─────────────────────────────────────────────────

# Common CUPTI callback IDs for runtime API functions.
RUNTIME_CBID_NAMES = {
    1: "cudaMalloc",
    2: "cudaFree",
    3: "cudaMallocHost",
    4: "cudaFreeHost",
    11: "cudaMemcpy",
    35: "cudaConfigureCall",
    47: "cudaDeviceSynchronize",
    164: "cudaLaunchKernel",
    211: "cudaMemcpyAsync",
    319: "cudaStreamSynchronize",
}


def query_runtime_api(conn):
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if not table_exists(conn, table):
        return []

    cols = get_columns(conn, table)

    # Resolve string IDs for name lookup.
    string_ids = {}
    if table_exists(conn, "StringIds"):
        for srow in conn.execute("SELECT id, value FROM StringIds"):
            string_ids[srow[0]] = srow[1]

    # nsys versions vary: older have "cbid", newer use "nameId" -> StringIds.
    cbid_col = col_or_null(cols, "cbid")
    name_id_col = col_or_null(cols, "nameId")
    corr_col = col_or_null(cols, "correlationId")
    tid_col = col_or_null(cols, "globalTid", "threadId")

    select_parts = ["start", "end"]
    labels = ["start", "end"]

    def add(col, label):
        select_parts.append(col if col else "NULL")
        labels.append(label)

    add(cbid_col, "cbid")
    add(name_id_col, "name_id")
    add(corr_col, "correlation_id")
    add(tid_col, "thread_id")

    query = f"SELECT {', '.join(select_parts)} FROM {table} ORDER BY start"
    rows = conn.execute(query).fetchall()

    results = []
    for row in rows:
        rec = dict(zip(labels, row))
        start_ns = int(rec["start"])
        end_ns = int(rec["end"])

        # Resolve the API function name.
        name = None
        if rec.get("name_id") is not None:
            name = string_ids.get(int(rec["name_id"]))
        if not name and rec.get("cbid") is not None:
            cbid = int(rec["cbid"])
            name = RUNTIME_CBID_NAMES.get(cbid, f"cbid_{cbid}")
        if not name:
            name = "unknown"

        # Clean up versioned suffixes (e.g., "cudaMalloc_v3020" -> "cudaMalloc")
        clean_name = re.sub(r"_v\d+$", "", name)

        results.append({
            "name": clean_name,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "correlation_id": int(rec["correlation_id"]) if rec.get("correlation_id") is not None else 0,
            "thread_id": int(rec["thread_id"]) if rec.get("thread_id") is not None else 0,
        })
    return results


# ── Synchronization ──────────────────────────────────────────────────

SYNC_TYPE_NAMES = {
    0: "unknown",
    1: "event_synchronize",
    2: "stream_wait_event",
    3: "stream_synchronize",
    4: "context_synchronize",
}


def query_sync(conn):
    table = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
    if not table_exists(conn, table):
        return []

    cols = get_columns(conn, table)
    sync_type_col = col_or_null(cols, "syncType", "type")
    stream_col = col_or_null(cols, "streamId")

    select_parts = ["start", "end"]
    labels = ["start", "end"]

    def add(col, label):
        select_parts.append(col if col else "NULL")
        labels.append(label)

    add(sync_type_col, "sync_type")
    add(stream_col, "stream_id")

    query = f"SELECT {', '.join(select_parts)} FROM {table} ORDER BY start"
    rows = conn.execute(query).fetchall()

    results = []
    for row in rows:
        rec = dict(zip(labels, row))
        start_ns = int(rec["start"])
        end_ns = int(rec["end"])
        stype = int(rec["sync_type"]) if rec.get("sync_type") is not None else 0
        results.append({
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "sync_type": SYNC_TYPE_NAMES.get(stype, f"type_{stype}"),
            "sync_type_id": stype,
            "stream_id": int(rec["stream_id"]) if rec.get("stream_id") is not None else 0,
        })
    return results


# ── NCCL classification ──────────────────────────────────────────────

NCCL_KERNEL_PATTERNS = [
    re.compile(r"ncclDevKernel", re.IGNORECASE),
    re.compile(r"ncclKernel", re.IGNORECASE),
    re.compile(r"nccl.*AllGather", re.IGNORECASE),
    re.compile(r"nccl.*AllReduce", re.IGNORECASE),
    re.compile(r"nccl.*Broadcast", re.IGNORECASE),
    re.compile(r"nccl.*ReduceScatter", re.IGNORECASE),
    re.compile(r"nccl.*SendRecv", re.IGNORECASE),
]

NCCL_NVTX_PATTERNS = [
    re.compile(r"nccl", re.IGNORECASE),
]


def classify_nccl_kernels(kernels):
    """Tag kernels that are NCCL operations."""
    nccl_kernels = []
    for k in kernels:
        name = k.get("name", "") + " " + k.get("demangled_name", "")
        is_nccl = any(p.search(name) for p in NCCL_KERNEL_PATTERNS)
        if is_nccl:
            # Try to extract the collective type from the kernel name.
            coll_type = "unknown"
            for ctype in ["AllGather", "AllReduce", "Broadcast", "ReduceScatter", "SendRecv", "Reduce"]:
                if ctype.lower() in name.lower():
                    coll_type = ctype
                    break
            nccl_kernels.append({**k, "nccl_collective": coll_type})
    return nccl_kernels


def classify_nccl_nvtx(nvtx_ranges):
    """Filter NVTX ranges that are NCCL-related."""
    nccl_ranges = []
    for r in nvtx_ranges:
        text = r.get("text", "") + " " + r.get("domain", "")
        if any(p.search(text) for p in NCCL_NVTX_PATTERNS):
            nccl_ranges.append(r)
    return nccl_ranges


# ── Output helpers ───────────────────────────────────────────────────

def write_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import NVIDIA Nsight Systems (.nsys-rep) profiling data into IKP JSON format.",
    )
    parser.add_argument(
        "--nsys-rep", required=True,
        help="Path to the .nsys-rep file",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--kernel-regex", default=None,
        help="Regex to filter kernel names (default: include all)",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip nsys export step (use existing .sqlite file)",
    )
    args = parser.parse_args()

    nsys_rep = args.nsys_rep
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Determine SQLite path.
    sqlite_path = os.path.splitext(nsys_rep)[0] + ".sqlite"

    # Step 1: Export to SQLite.
    if not args.skip_export:
        if not os.path.exists(nsys_rep):
            print(f"ERROR: {nsys_rep} not found", file=sys.stderr)
            sys.exit(1)
        export_nsys_to_sqlite(nsys_rep, sqlite_path)
    else:
        if not os.path.exists(sqlite_path):
            print(f"ERROR: {sqlite_path} not found (--skip-export requires existing .sqlite)", file=sys.stderr)
            sys.exit(1)
        print(f"  using existing {sqlite_path}")

    # Step 2: Query SQLite.
    conn = sqlite3.connect(sqlite_path)
    try:
        print("  querying kernel launches...")
        all_kernels = query_kernels(conn, kernel_regex=None)  # All kernels for nsys_events.
        filtered_kernels = query_kernels(conn, kernel_regex=args.kernel_regex) if args.kernel_regex else all_kernels

        print("  querying memory operations...")
        memcpy = query_memcpy(conn)
        memset = query_memset(conn)

        print("  querying NVTX ranges...")
        nvtx = query_nvtx(conn)

        print("  querying CUDA runtime API...")
        runtime = query_runtime_api(conn)

        print("  querying synchronization events...")
        sync = query_sync(conn)

        # NCCL classification.
        nccl_kernels = classify_nccl_kernels(all_kernels)
        nccl_nvtx = classify_nccl_nvtx(nvtx)
    finally:
        conn.close()

    # Step 3: Write output JSON files.
    print(f"\n  summary:")
    print(f"    kernels:       {len(all_kernels)} total, {len(filtered_kernels)} matched")
    print(f"    memcpy:        {len(memcpy)}")
    print(f"    memset:        {len(memset)}")
    print(f"    nvtx ranges:   {len(nvtx)}")
    print(f"    runtime API:   {len(runtime)}")
    print(f"    sync events:   {len(sync)}")
    print(f"    NCCL kernels:  {len(nccl_kernels)}")
    print(f"    NCCL NVTX:     {len(nccl_nvtx)}")

    # nsys_events.json — complete system-level events.
    nsys_events = {
        "tool": "ikp_nsys_import",
        "version": 1,
        "source": os.path.basename(nsys_rep),
        "time_base": "ns",
        "gpu_events": {
            "kernels": all_kernels,
            "memcpy": memcpy,
            "memset": memset,
            "sync": sync,
        },
        "api_events": {
            "runtime": runtime,
        },
        "nvtx_ranges": nvtx,
        "nccl": {
            "kernels": nccl_kernels,
            "nvtx_ranges": nccl_nvtx,
        },
    }
    write_json(nsys_events, os.path.join(out_dir, "nsys_events.json"))

    # nsys_kernels.json — kernel-focused (for timeline alignment and downstream use).
    nsys_kernels = {
        "tool": "ikp_nsys_import",
        "version": 1,
        "source": os.path.basename(nsys_rep),
        "kernel_regex": args.kernel_regex,
        "kernels": filtered_kernels,
    }
    write_json(nsys_kernels, os.path.join(out_dir, "nsys_kernels.json"))

    print("\nikp_nsys_import: done.")


if __name__ == "__main__":
    main()
