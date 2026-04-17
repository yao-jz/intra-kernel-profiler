#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# examples/nsys_nccl/run.sh — NSys + NCCL collective profiling demo
# ═══════════════════════════════════════════════════════════════════════
#
# Profiles AllReduce, AllGather, ReduceScatter, Broadcast, and Reduce
# on 2 GPUs with NSys, then imports the data so you can see every NCCL
# collective type in a Perfetto trace.
#
# Usage:
#   bash examples/nsys_nccl/run.sh
#   bash examples/nsys_nccl/run.sh --ngpus=4 --gpu-base=4
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

NGPUS="${NGPUS:-2}"
GPU_BASE="${GPU_BASE:-0}"
SM_ARCH="${SM_ARCH:-sm_90a}"
OUT="${HERE}/_out"
NCCL_HOME="${NCCL_HOME:-}"

for a in "$@"; do
  case "$a" in
    --ngpus=*)    NGPUS="${a#--ngpus=}" ;;
    --gpu-base=*) GPU_BASE="${a#--gpu-base=}" ;;
    --sm=*)       SM_ARCH="${a#--sm=}" ;;
    --nccl=*)     NCCL_HOME="${a#--nccl=}" ;;
    --out=*)      OUT="${a#--out=}" ;;
    -h|--help)    echo "Usage: $0 [--ngpus=2] [--gpu-base=0] [--nccl=PATH]"; exit 0 ;;
  esac
done

command -v nvcc  >/dev/null 2>&1 || { echo "ERROR: nvcc not found" >&2; exit 1; }
command -v nsys  >/dev/null 2>&1 || { echo "ERROR: nsys not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 1; }
[[ -f "${NCCL_HOME}/include/nccl.h" ]] || { echo "ERROR: NCCL not found at ${NCCL_HOME}" >&2; exit 1; }

GPU_LIST=""
for (( i=0; i<NGPUS; i++ )); do
  [[ -n "$GPU_LIST" ]] && GPU_LIST+=","
  GPU_LIST+="$((GPU_BASE + i))"
done
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

rm -rf "${OUT}"
mkdir -p "${OUT}/nsys"

echo "═══════════════════════════════════════════════════════════════"
echo "  NSys + NCCL Collective Profiling"
echo "  GPUs: ${NGPUS} (devices ${GPU_LIST})"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Build
echo "[1/3] Build"
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" \
  -I "${NCCL_HOME}/include" -L "${NCCL_HOME}/lib" -lnccl \
  -Xlinker -rpath="${NCCL_HOME}/lib" -lpthread \
  "${HERE}/nccl_demo.cu" -o "${HERE}/nccl_demo"

# NSys profile
echo "[2/3] NSys profile"
nsys profile \
  --output="${OUT}/nsys/report" \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  "${HERE}/nccl_demo" "${NGPUS}"

# Import
echo "[3/3] Import"
python3 "${ROOT}/scripts/ikp_nsys_import.py" \
  --nsys-rep "${OUT}/nsys/report.nsys-rep" \
  --out-dir "${OUT}/nsys/"

echo ""
echo "═══════════════════════════════════════════════════════════════"

# Show what was captured
python3 -c "
import json
d = json.load(open('${OUT}/nsys/nsys_events.json'))
nccl = d.get('nccl', {}).get('kernels', [])
if not nccl:
    print('  No NCCL kernels detected')
else:
    from collections import Counter
    c = Counter(k.get('nccl_collective', '?') for k in nccl)
    print('  NCCL collectives captured:')
    for name, count in sorted(c.items()):
        print(f'    {name:>20}  x{count}')
print()
print('  nsys_events.json -> ${OUT}/nsys/nsys_events.json')
"

# Cleanup binary
rm -f "${HERE}/nccl_demo"
