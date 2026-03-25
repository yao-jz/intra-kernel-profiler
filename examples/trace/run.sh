#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

OUT="${HERE}/_out"
mkdir -p "${OUT}"

echo "=== Building trace examples (CMake) ==="
cmake -S "${ROOT}" -B "${ROOT}/_ci_build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${ROOT}/_ci_build" -j >/dev/null

echo "=== [1/3] ikp_trace_record ==="
"${ROOT}/_ci_build/ikp_trace_record" --iters=1000 --out="${OUT}/record_trace.json"

echo "=== [2/3] ikp_trace_block_filter ==="
"${ROOT}/_ci_build/ikp_trace_block_filter" --iters=1000 --out="${OUT}/block_filter_trace.json"

echo "=== [3/3] ikp_trace_sampled_loop ==="
"${ROOT}/_ci_build/ikp_trace_sampled_loop" --iters=10000 --sample_shift=8 --out="${OUT}/sampled_loop_trace.json"

echo ""
echo "DONE. Outputs:"
ls -lh "${OUT}/"*.json
echo ""
echo "Open *_trace.json in https://ui.perfetto.dev to visualize."
