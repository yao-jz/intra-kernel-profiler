#!/usr/bin/env python3
"""Merge NSys system-level events with IKP intra-kernel trace into a unified Chrome Trace JSON.

Produces a single Perfetto-viewable trace file where IKP's per-SM/warp
region timings and NSys's kernel launches, memory copies, NCCL ops, and
NVTX ranges coexist on the same timeline.

Usage:
    python3 scripts/ikp_nsys_merge.py \\
        --nsys-events _out/nsys/nsys_events.json \\
        --ikp-trace   _out/trace/gemm_trace.json \\
        [--nsys-kernels _out/nsys/nsys_kernels.json] \\
        [--kernel-regex "tiled_gemm_kernel"] \\
        --out merged_trace.json

Time alignment:
    Both NSys GPU activity timestamps and IKP's globaltimer come from the
    same hardware clock.  IKP rebases events to min_ts=0 in
    host_session.hpp.  NSys kernel start provides the absolute anchor.
    The merge rebases all NSys events relative to the matched kernel start,
    so that IKP's t=0 aligns with the kernel launch boundary.
"""
import argparse
import json
import os
import re
import sys


# ── PID allocation (avoids collision with IKP SM pids 0-255) ─────────

PID_NSYS_KERNELS = 10000
PID_NSYS_MEMORY  = 10001
PID_NSYS_API     = 10002
PID_NSYS_NCCL    = 10003
PID_NSYS_NVTX    = 10004


# ── Loaders ──────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ikp_trace(path):
    """Load an IKP Chrome Trace JSON and return (events_list, metadata)."""
    data = load_json(path)
    events = data.get("traceEvents", [])

    # Try to load the companion summary JSON for scale/metadata.
    summary_path = path.replace(".json", "_summary.json")
    summary = None
    if os.path.exists(summary_path):
        summary = load_json(summary_path)

    scale = 1.0
    if summary:
        scale = summary.get("scale", 1.0)

    return events, {"scale": scale, "summary": summary}


# ── Time alignment ───────────────────────────────────────────────────
#
# Strategy: proportional rescaling per kernel launch.
#
# IKP trace events are already rebased (min_ts subtracted) and scaled.
# NSys kernel launches have absolute (start_ns, end_ns).  We:
#   1. Detect per-launch groups in the IKP trace by finding time gaps.
#   2. Find matching NSys kernel launches (by regex + temporal order).
#   3. For each group, linearly map [ikp_min, ikp_max] → [nsys_start, nsys_end].
#
# This ensures IKP events always sit precisely inside the NSys kernel bar
# in Perfetto, regardless of clock offsets or scale factors.

def find_matching_kernels(nsys_kernels_data, nsys_events_data, kernel_regex=None):
    """Find all NSys kernel launches matching the regex, sorted by start time."""
    kernels = []
    if nsys_kernels_data:
        kernels = nsys_kernels_data.get("kernels", [])
    if not kernels and nsys_events_data:
        kernels = nsys_events_data.get("gpu_events", {}).get("kernels", [])
    if not kernels:
        return []

    if kernel_regex:
        pat = re.compile(kernel_regex)
        kernels = [k for k in kernels if pat.search(k.get("name", ""))]

    return sorted(kernels, key=lambda k: k["start_ns"])


def _detect_ikp_launch_groups(ikp_events):
    """Split IKP trace events into per-kernel-launch groups by detecting time gaps.

    IKP traces from a single HostSession may contain events from multiple
    kernel launches (e.g., split GEMM).  Between launches there is typically
    a large time gap (>> intra-kernel event spacing).  We detect these gaps
    to segment the events.

    Returns a list of groups, each group is a list of event indices.
    """
    # Collect (timestamp, index) for all duration events.
    timed = []
    for i, e in enumerate(ikp_events):
        if e.get("ph") == "X" and "ts" in e:
            timed.append((e["ts"], i))
        elif e.get("ph") in ("B", "E", "i") and "ts" in e:
            timed.append((e["ts"], i))

    if not timed:
        return []

    timed.sort()

    # Find gaps.  Use a heuristic: a gap > 10x the median inter-event
    # interval is a launch boundary.
    if len(timed) < 2:
        return [[idx for _, idx in timed]]

    intervals = [timed[i + 1][0] - timed[i][0] for i in range(len(timed) - 1)]
    intervals_nonzero = sorted(v for v in intervals if v > 0)
    if not intervals_nonzero:
        return [[idx for _, idx in timed]]

    median_interval = intervals_nonzero[len(intervals_nonzero) // 2]
    # Use 100x median as the gap threshold, with a minimum of 1000 ns.
    gap_threshold = max(median_interval * 100, 1000.0)

    groups = []
    current_group = [timed[0][1]]
    for i in range(1, len(timed)):
        gap = timed[i][0] - timed[i - 1][0]
        if gap > gap_threshold:
            groups.append(current_group)
            current_group = []
        current_group.append(timed[i][1])
    if current_group:
        groups.append(current_group)

    return groups


def _get_group_time_range(ikp_events, indices):
    """Get the [min_ts, max_ts] of a group of IKP events (including duration)."""
    min_ts = float("inf")
    max_ts = float("-inf")
    for i in indices:
        e = ikp_events[i]
        ts = e.get("ts", 0)
        dur = e.get("dur", 0)
        if ts < min_ts:
            min_ts = ts
        end = ts + dur
        if end > max_ts:
            max_ts = end
    return min_ts, max_ts


def remap_ikp_events(ikp_events, nsys_kernels, nsys_offset_ns):
    """Remap IKP trace events so they align precisely inside NSys kernel windows.

    For each detected IKP launch group, finds the matching NSys kernel and
    linearly maps [ikp_min, ikp_max] → [nsys_start, nsys_end].

    Events not belonging to any matched group (metadata events, etc.) are
    placed relative to the first NSys kernel's start time.

    Returns (remapped_events, alignment_info).
    """
    if not nsys_kernels:
        # No NSys kernels to align to — return events unchanged with a
        # simple offset so NSys system events are on the same timeline.
        return list(ikp_events), {"method": "passthrough", "groups": 0}

    groups = _detect_ikp_launch_groups(ikp_events)
    n_groups = len(groups)
    n_kernels = len(nsys_kernels)

    # Build a set of all indices that belong to some group.
    grouped_indices = set()
    for g in groups:
        grouped_indices.update(g)

    # Match groups to NSys kernels.  When there are more NSys kernels than
    # IKP groups (common: warmup + profiled + benchmark launches in NSys,
    # but IKP only captured the profiled launch), we pick the NSys kernel
    # with the longest duration — IKP instrumentation adds overhead, so the
    # profiled launch is always the slowest instance of that kernel.
    mappings = []  # (group_indices, nsys_kernel, ikp_min, ikp_max)
    used_kernel_indices = set()
    # Sort kernel indices by duration descending (longest = most likely profiled).
    kernel_by_dur = sorted(range(n_kernels),
                           key=lambda i: nsys_kernels[i]["end_ns"] - nsys_kernels[i]["start_ns"],
                           reverse=True)
    for gi, group in enumerate(groups):
        ikp_min, ikp_max = _get_group_time_range(ikp_events, group)
        # Pick the longest unused kernel.
        best_ki = None
        for ki in kernel_by_dur:
            if ki not in used_kernel_indices:
                best_ki = ki
                break
        if best_ki is None:
            best_ki = 0
        used_kernel_indices.add(best_ki)
        nk = nsys_kernels[best_ki]
        mappings.append((group, nk, ikp_min, ikp_max))

    # Build remapped events.
    # For each grouped event: linear map into the NSys kernel window.
    # For ungrouped events (metadata): use the first kernel's start as base.
    remapped = list(ikp_events)  # shallow copy
    first_kernel_start = (nsys_kernels[0]["start_ns"] - nsys_offset_ns)

    remap_info = []
    for group, nk, ikp_min, ikp_max in mappings:
        nsys_start = nk["start_ns"] - nsys_offset_ns
        nsys_end = nk["end_ns"] - nsys_offset_ns
        nsys_dur = nsys_end - nsys_start
        ikp_dur = ikp_max - ikp_min

        # Scale factor: how to stretch/compress IKP time into NSys window.
        if ikp_dur > 0:
            scale = nsys_dur / ikp_dur
        else:
            scale = 1.0

        for i in group:
            e = dict(remapped[i])  # copy to avoid mutating original
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            # Map: new_ts = nsys_start + (ts - ikp_min) * scale
            e["ts"] = nsys_start + (ts - ikp_min) * scale
            if dur > 0:
                e["dur"] = dur * scale
            remapped[i] = e

        remap_info.append({
            "nsys_kernel": nk.get("name", ""),
            "nsys_start_ns": nk["start_ns"],
            "nsys_end_ns": nk["end_ns"],
            "ikp_range": [ikp_min, ikp_max],
            "scale": round(scale, 6),
            "events": len(group),
        })

    alignment = {
        "method": "proportional_rescale",
        "groups_detected": n_groups,
        "nsys_kernels_matched": min(n_groups, n_kernels),
        "per_group": remap_info,
    }

    return remapped, alignment


# ── NSys -> Chrome Trace conversion ─────────────────────────────────

def _meta_event(pid, tid, name_type, name):
    return {
        "ph": "M",
        "name": name_type,
        "pid": pid,
        "tid": tid,
        "args": {"name": name},
    }


def convert_nsys_events(nsys_events_data, offset_ns, scale):
    """Convert NSys events to Chrome Trace event dicts."""
    chrome_events = []
    used_pids = set()
    flow_id = [0]  # mutable counter for flow event IDs

    def ts(ns):
        """Convert absolute NSys nanoseconds to merged trace time."""
        return (ns - offset_ns) * scale

    def dur_label(ns):
        """Human-readable duration for event name annotations."""
        if ns >= 1e9:
            return f"{ns / 1e9:.2f}s"
        if ns >= 1e6:
            return f"{ns / 1e6:.1f}ms"
        if ns >= 1e3:
            return f"{ns / 1e3:.0f}us"
        return f"{ns}ns"

    # Color scheme — only colors confirmed safe in both chrome://tracing and Perfetto.
    # Safe set: thread_state_running, thread_state_iowait, thread_state_sleeping,
    #   generic_work, good, bad, terrible, black, grey, white, yellow, olive,
    #   rail_response, rail_animation, rail_idle, rail_load, startup,
    #   vsync_highlight_color, cq_build_passed, cq_build_failed
    CNAME_KERNEL  = "thread_state_running"     # green
    CNAME_HTOD    = "good"                     # green
    CNAME_DTOH    = "rail_response"            # light blue
    CNAME_DTOD    = "olive"                    # olive
    CNAME_PTOP    = "yellow"                   # yellow
    CNAME_MEMSET  = "grey"                     # grey
    CNAME_API_LAUNCH = "rail_load"             # purple
    CNAME_API_SYNC   = "bad"                   # red
    CNAME_API_MEM    = "olive"                 # olive
    CNAME_API_OTHER  = "generic_work"          # grey
    CNAME_NCCL    = "rail_animation"           # orange
    CNAME_NVTX    = "yellow"                   # yellow

    MEMCPY_CNAME = {
        "HtoD": CNAME_HTOD,
        "DtoH": CNAME_DTOH,
        "DtoD": CNAME_DTOD,
        "PtoP": CNAME_PTOP,
        "HtoH": "white",
    }

    def api_cname(name):
        n = name.lower()
        if "launch" in n:
            return CNAME_API_LAUNCH
        if "synchronize" in n or "sync" in n:
            return CNAME_API_SYNC
        if "memcpy" in n or "memset" in n or "malloc" in n or "free" in n:
            return CNAME_API_MEM
        return CNAME_API_OTHER

    # Build correlation map: correlation_id -> kernel start event (for flow arrows).
    corr_to_kernel = {}
    for k in nsys_events_data.get("gpu_events", {}).get("kernels", []):
        cid = k.get("correlation_id")
        if cid:
            corr_to_kernel[cid] = k

    # Map large thread IDs (nsys globalTid) to small integers for chrome://tracing.
    _tid_map = {}
    _tid_next = [0]
    def compact_tid(raw_tid):
        if raw_tid not in _tid_map:
            _tid_map[raw_tid] = _tid_next[0]
            _tid_next[0] += 1
        return _tid_map[raw_tid]

    # Build set of NCCL kernel names so we can avoid duplicating them
    # in both the GPU Kernels row and the NCCL row.
    nccl_kernel_keys = set()
    for k in nsys_events_data.get("nccl", {}).get("kernels", []):
        nccl_kernel_keys.add((k["start_ns"], k["end_ns"]))

    # ── Kernel launches (excluding NCCL kernels — those go to NCCL row) ─
    kernels = nsys_events_data.get("gpu_events", {}).get("kernels", [])
    non_nccl_kernels = [k for k in kernels
                        if (k["start_ns"], k["end_ns"]) not in nccl_kernel_keys]
    if non_nccl_kernels:
        used_pids.add(PID_NSYS_KERNELS)
        stream_tids = set()
        for k in non_nccl_kernels:
            tid = k.get("stream_id", 0)
            stream_tids.add(tid)
            grid = k.get("grid", [1, 1, 1])
            block = k.get("block", [1, 1, 1])
            dur_ns = k["end_ns"] - k["start_ns"]
            chrome_events.append({
                "name": f"{k.get('name', 'kernel')}  [{dur_label(dur_ns)}]",
                "ph": "X",
                "ts": ts(k["start_ns"]),
                "dur": dur_ns * scale,
                "pid": PID_NSYS_KERNELS,
                "tid": tid,
                "cname": CNAME_KERNEL,
                "args": {
                    "stream": tid,
                    "grid": f"{grid[0]}x{grid[1]}x{grid[2]}",
                    "block": f"{block[0]}x{block[1]}x{block[2]}",
                    "duration_us": round(dur_ns / 1e3, 1),
                    "device": k.get("device_id", 0),
                },
            })
        for tid in sorted(stream_tids):
            chrome_events.append(_meta_event(PID_NSYS_KERNELS, tid, "thread_name", f"stream {tid}"))

    # ── Memory operations (filter out tiny NCCL-internal transfers) ─
    _MIN_MEM_BYTES = 4096  # Filter out tiny NCCL-internal transfers
    memcpy = [m for m in nsys_events_data.get("gpu_events", {}).get("memcpy", [])
              if m.get("bytes", 0) >= _MIN_MEM_BYTES]
    memset = [m for m in nsys_events_data.get("gpu_events", {}).get("memset", [])
              if m.get("bytes", 0) >= _MIN_MEM_BYTES]
    if memcpy or memset:
        used_pids.add(PID_NSYS_MEMORY)
        mem_tids = set()
        for m in memcpy:
            tid = m.get("stream_id", 0)
            mem_tids.add(tid)
            size_str = _human_bytes(m.get("bytes", 0))
            kind = m.get("kind", "memcpy")
            dur_ns = m["end_ns"] - m["start_ns"]
            # Compute bandwidth.
            bw_str = ""
            if dur_ns > 0 and m.get("bytes", 0) > 0:
                bw_gbps = m["bytes"] / dur_ns  # bytes/ns = GB/s
                bw_str = f"  {bw_gbps:.1f} GB/s"
            chrome_events.append({
                "name": f"{kind} {size_str}{bw_str}",
                "ph": "X",
                "ts": ts(m["start_ns"]),
                "dur": dur_ns * scale,
                "pid": PID_NSYS_MEMORY,
                "tid": tid,
                "cname": MEMCPY_CNAME.get(kind, "rail_load"),
                "args": {
                    "bytes": m.get("bytes", 0),
                    "size": size_str,
                    "kind": kind,
                    "stream": tid,
                    "duration_us": round(dur_ns / 1e3, 1),
                    "bandwidth_GBps": round(m.get("bytes", 0) / max(dur_ns, 1), 2),
                },
            })
        for m in memset:
            tid = m.get("stream_id", 0)
            mem_tids.add(tid)
            size_str = _human_bytes(m.get("bytes", 0))
            chrome_events.append({
                "name": f"memset {size_str}",
                "ph": "X",
                "ts": ts(m["start_ns"]),
                "dur": (m["end_ns"] - m["start_ns"]) * scale,
                "pid": PID_NSYS_MEMORY,
                "tid": tid,
                "cname": CNAME_MEMSET,
                "args": {"bytes": m.get("bytes", 0), "stream": tid},
            })
        for tid in sorted(mem_tids):
            chrome_events.append(_meta_event(PID_NSYS_MEMORY, tid, "thread_name", f"stream {tid}"))

    # ── CUDA Runtime API (filter out driver-internal noise) ─────
    # NCCL and other libraries generate thousands of cuMem*, cuGet*,
    # cudaThread* calls that clutter the trace.  Keep only user-visible
    # runtime/driver calls.
    _USER_API_PREFIXES = (
        "cudaMalloc", "cudaFree",
        "cudaMemcpy", "cudaMemset",  # sync versions only
        "cudaLaunchKernel",
        "cudaDeviceSynchronize", "cudaEventSynchronize",
        "cudaSetDevice",
    )
    _EXCLUDE_API = (
        "cudaMemcpyAsync", "cudaMemsetAsync",  # NCCL internal
    )
    runtime = nsys_events_data.get("api_events", {}).get("runtime", [])
    runtime = [r for r in runtime
               if any(r.get("name", "").startswith(p) for p in _USER_API_PREFIXES)
               and not any(r.get("name", "").startswith(x) for x in _EXCLUDE_API)]
    if runtime:
        used_pids.add(PID_NSYS_API)
        api_tids = set()
        for r in runtime:
            tid = compact_tid(r.get("thread_id", 0))
            api_tids.add(tid)
            name = r.get("name", "api")
            dur_ns = r["end_ns"] - r["start_ns"]
            chrome_events.append({
                "name": f"{name}  [{dur_label(dur_ns)}]",
                "ph": "X",
                "ts": ts(r["start_ns"]),
                "dur": dur_ns * scale,
                "pid": PID_NSYS_API,
                "tid": tid,
                "cname": api_cname(name),
                "args": {
                    "correlation_id": r.get("correlation_id", 0),
                    "duration_us": round(dur_ns / 1e3, 1),
                },
            })

            # ── Flow arrow: CUDA API launch → GPU kernel ─────────
            cid = r.get("correlation_id", 0)
            if cid and cid in corr_to_kernel:
                k = corr_to_kernel[cid]
                fid = flow_id[0]
                flow_id[0] += 1
                # Flow start: from the API call
                chrome_events.append({
                    "name": "launch",
                    "ph": "s",
                    "ts": ts(r["end_ns"]),
                    "pid": PID_NSYS_API,
                    "tid": tid,
                    "id": fid,
                    "cat": "launch",
                })
                # Flow end: at the kernel start
                k_tid = k.get("stream_id", 0)
                chrome_events.append({
                    "name": "launch",
                    "ph": "f",
                    "ts": ts(k["start_ns"]),
                    "pid": PID_NSYS_KERNELS,
                    "tid": k_tid,
                    "id": fid,
                    "bp": "e",
                    "cat": "launch",
                })

        for tid in sorted(api_tids):
            chrome_events.append(_meta_event(PID_NSYS_API, tid, "thread_name", f"CPU thread"))

    # Compute total kernel execution time for filtering giant NVTX ranges.
    # Use sum of kernel durations (not span), so ranges much longer than
    # all actual GPU work are filtered out.
    _total_kernel_time = sum(k["end_ns"] - k["start_ns"] for k in kernels) if kernels else 1

    # ── NCCL (filter zero-duration events) ─────────────────────────
    nccl_data = nsys_events_data.get("nccl", {})
    nccl_kernels = [k for k in nccl_data.get("kernels", [])
                    if k["end_ns"] - k["start_ns"] > 0]
    nccl_nvtx = nccl_data.get("nvtx_ranges", [])
    if nccl_kernels or nccl_nvtx:
        used_pids.add(PID_NSYS_NCCL)
        nccl_tids = set()
        for k in nccl_kernels:
            tid = k.get("device_id", 0)
            nccl_tids.add(tid)
            dur_ns = k["end_ns"] - k["start_ns"]
            coll = k.get("nccl_collective", "")
            chrome_events.append({
                "name": f"{coll or k.get('name', 'nccl')}  [{dur_label(dur_ns)}]",
                "ph": "X",
                "ts": ts(k["start_ns"]),
                "dur": dur_ns * scale,
                "pid": PID_NSYS_NCCL,
                "tid": tid,
                "cname": CNAME_NCCL,
                "args": {
                    "kernel": k.get("name", ""),
                    "collective": coll,
                    "device": tid,
                    "stream": k.get("stream_id", 0),
                    "duration_us": round(dur_ns / 1e3, 1),
                },
            })
        # Filter out giant NCCL NVTX lifecycle ranges.
        for r in nccl_nvtx:
            nvtx_dur = r.get("duration_ns", r["end_ns"] - r["start_ns"])
            if _total_kernel_time > 0 and nvtx_dur > _total_kernel_time * 5:
                continue
            tid = 100 + r.get("domain_id", 0)
            nccl_tids.add(tid)
            chrome_events.append({
                "name": r.get("text", "nccl_nvtx"),
                "ph": "X",
                "ts": ts(r["start_ns"]),
                "dur": max(0, nvtx_dur) * scale,
                "pid": PID_NSYS_NCCL,
                "tid": tid,
                "cname": "rail_animation",
                "args": {"domain": r.get("domain", ""), "text": r.get("text", "")},
            })
        for tid in sorted(nccl_tids):
            label = f"device {tid}" if tid < 100 else f"nvtx domain {tid - 100}"
            chrome_events.append(_meta_event(PID_NSYS_NCCL, tid, "thread_name", label))

    # ── General NVTX ranges (non-NCCL, excluding giant lifecycle ranges) ─
    all_nvtx = nsys_events_data.get("nvtx_ranges", [])
    nccl_nvtx_set = {(r["start_ns"], r["end_ns"], r.get("text", "")) for r in nccl_nvtx}
    non_nccl_nvtx = [
        r for r in all_nvtx
        if (r["start_ns"], r["end_ns"], r.get("text", "")) not in nccl_nvtx_set
        and r.get("duration_ns", r["end_ns"] - r["start_ns"]) < _total_kernel_time * 5
    ]
    if non_nccl_nvtx:
        used_pids.add(PID_NSYS_NVTX)
        nvtx_tids = set()
        for r in non_nccl_nvtx:
            tid = r.get("domain_id", 0)
            nvtx_tids.add(tid)
            chrome_events.append({
                "name": r.get("text", "nvtx"),
                "ph": "X",
                "ts": ts(r["start_ns"]),
                "dur": max(0, (r["end_ns"] - r["start_ns"])) * scale,
                "pid": PID_NSYS_NVTX,
                "tid": tid,
                "cname": CNAME_NVTX,
                "args": {"domain": r.get("domain", ""), "text": r.get("text", "")},
            })
        for tid in sorted(nvtx_tids):
            chrome_events.append(_meta_event(PID_NSYS_NVTX, tid, "thread_name", f"domain {tid}"))

    # ── Process metadata with sort order ─────────────────────────
    # Perfetto sorts processes by the "process_sort_index" metadata.
    # We place NSys rows above IKP SM rows for a natural top-down reading.
    pid_info = {
        PID_NSYS_API:     ("[NSys] CUDA API",     -50),
        PID_NSYS_KERNELS: ("[NSys] GPU Kernels",  -40),
        PID_NSYS_MEMORY:  ("[NSys] Memory Ops",   -30),
        PID_NSYS_NCCL:    ("[NSys] NCCL",         -20),
        PID_NSYS_NVTX:    ("[NSys] NVTX",         -10),
    }
    for pid in sorted(used_pids):
        name, sort_idx = pid_info.get(pid, (f"nsys_{pid}", 0))
        chrome_events.append(_meta_event(pid, 0, "process_name", name))
        chrome_events.append({
            "ph": "M", "name": "process_sort_index",
            "pid": pid, "tid": 0,
            "args": {"sort_index": sort_idx},
        })

    return chrome_events


def _human_bytes(n):
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f}GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f}KB"
    return f"{n}B"


# ── Merge ────────────────────────────────────────────────────────────

def merge_traces(ikp_events, nsys_chrome_events, alignment_info):
    """Merge IKP and NSys Chrome Trace events into a single trace dict."""
    merged_events = list(ikp_events) + list(nsys_chrome_events)

    return {
        "displayTimeUnit": "ns",
        "traceEvents": merged_events,
        "metadata": {
            "ikp_nsys_merge": {
                "version": 2,
                "alignment": alignment_info,
                "nsys_event_count": len(nsys_chrome_events),
                "ikp_event_count": len(ikp_events),
            }
        },
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge NSys system-level events with IKP intra-kernel trace.",
    )
    parser.add_argument(
        "--nsys-events", required=True,
        help="Path to nsys_events.json (from ikp_nsys_import.py)",
    )
    parser.add_argument(
        "--ikp-trace", required=True,
        help="Path to IKP Chrome Trace JSON (e.g., gemm_trace.json)",
    )
    parser.add_argument(
        "--nsys-kernels", default=None,
        help="Path to nsys_kernels.json (for anchor kernel selection)",
    )
    parser.add_argument(
        "--kernel-regex", default=None,
        help="Regex to select the anchor kernel for time alignment",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for merged Chrome Trace JSON",
    )
    args = parser.parse_args()

    # Load inputs.
    nsys_events_data = load_json(args.nsys_events)
    ikp_events, ikp_meta = load_ikp_trace(args.ikp_trace)

    nsys_kernels_data = None
    if args.nsys_kernels and os.path.exists(args.nsys_kernels):
        nsys_kernels_data = load_json(args.nsys_kernels)

    # Find all matching NSys kernel launches (sorted by time).
    matched_kernels = find_matching_kernels(
        nsys_kernels_data, nsys_events_data, args.kernel_regex)

    if matched_kernels:
        print(f"  matched {len(matched_kernels)} NSys kernel launch(es): {matched_kernels[0]['name']}")
    else:
        print("  WARNING: no matching kernel launches found", file=sys.stderr)

    # Time reference: use the earliest event across ALL NSys categories so
    # that the trace starts at t=0 with the first cudaMalloc/cudaMemcpy and
    # the kernel appears at its natural position in the execution timeline.
    all_start_ns = []
    for k in nsys_events_data.get("gpu_events", {}).get("kernels", []):
        all_start_ns.append(k["start_ns"])
    for m in nsys_events_data.get("gpu_events", {}).get("memcpy", []):
        all_start_ns.append(m["start_ns"])
    for m in nsys_events_data.get("gpu_events", {}).get("memset", []):
        all_start_ns.append(m["start_ns"])
    for r in nsys_events_data.get("api_events", {}).get("runtime", []):
        all_start_ns.append(r["start_ns"])
    for s in nsys_events_data.get("gpu_events", {}).get("sync", []):
        all_start_ns.append(s["start_ns"])
    offset_ns = min(all_start_ns) if all_start_ns else 0

    # Remap IKP events: proportionally rescale each launch group into the
    # corresponding NSys kernel's [start, end] window.
    remapped_ikp, alignment_info = remap_ikp_events(
        ikp_events, matched_kernels, offset_ns)
    print(f"  alignment: {alignment_info['method']}, "
          f"{alignment_info.get('groups_detected', 0)} group(s)")

    # Convert NSys events to Chrome Trace format (using the same offset).
    nsys_chrome = convert_nsys_events(nsys_events_data, offset_ns, 1.0)

    # Trim: crop the timeline to a window around the IKP-profiled kernel.
    # This anchors on the kernel that has IKP trace data, ensuring the
    # interesting part dominates the view.  Warmup/benchmark kernels and
    # CUDA init/cleanup are trimmed away.
    all_nsys_x = [e for e in nsys_chrome if e.get("ph") == "X"]
    # Use IKP events as the anchor (they've been remapped to the profiled kernel).
    ikp_x = [e for e in remapped_ikp if e.get("ph") == "X"]
    if ikp_x:
        k_min = min(e["ts"] for e in ikp_x)
        k_max = max(e["ts"] + e.get("dur", 0) for e in ikp_x)
    else:
        user_kernels = [e for e in all_nsys_x if e.get("pid") == PID_NSYS_KERNELS]
        if user_kernels:
            k_min = min(e["ts"] for e in user_kernels)
            k_max = max(e["ts"] + e.get("dur", 0) for e in user_kernels)
        else:
            k_min, k_max = 0, 0
    if k_max > k_min:
        k_span = k_max - k_min
        # Margin: 10x the kernel duration — enough to show the surrounding
        # warmup kernel, NCCL AllGather, and API calls.
        margin = max(k_span * 10, 5000)
        trim_lo = k_min - margin
        trim_hi = k_max + margin

        def in_window(e):
            if e.get("ph") == "M":
                return True  # keep metadata
            if e.get("ph") in ("s", "f"):
                return True  # keep flow arrows
            t = e.get("ts", 0)
            return t >= trim_lo and t <= trim_hi

        nsys_chrome = [e for e in nsys_chrome if in_window(e)]
        n_trimmed = len(all_nsys_x) - len([e for e in nsys_chrome if e.get("ph") == "X"])
        if n_trimmed > 0:
            print(f"  trimmed {n_trimmed} events outside kernel window")

    # Merge.
    merged = merge_traces(remapped_ikp, nsys_chrome, alignment_info)

    # Write output.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    total = len(remapped_ikp) + len(nsys_chrome)
    print(f"  merged {len(remapped_ikp)} IKP + {len(nsys_chrome)} NSys = {total} events")
    print(f"  wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
