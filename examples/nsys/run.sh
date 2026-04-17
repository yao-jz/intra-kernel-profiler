#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# examples/nsys/run.sh — NSys + IKP trace integration demo
# ═══════════════════════════════════════════════════════════════════════
#
# Single kernel launch, profiled simultaneously by NSys and IKP.
# The merged trace shows system-level events (kernel launch, memcpy,
# CUDA API) perfectly aligned with intra-kernel region timing.
#
# Usage:
#   bash examples/nsys/run.sh
#   bash examples/nsys/run.sh --gpu=0 --sm=sm_90a
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

GPU="${GPU:-0}"
SM_ARCH="${SM_ARCH:-sm_90a}"
OUT="${HERE}/_out"

for a in "$@"; do
  case "$a" in
    --gpu=*)  GPU="${a#--gpu=}" ;;
    --sm=*)   SM_ARCH="${a#--sm=}" ;;
    --out=*)  OUT="${a#--out=}" ;;
    -h|--help) echo "Usage: $0 [--gpu=N] [--sm=sm_90a]"; exit 0 ;;
    *) echo "Unknown: $a" >&2; exit 2 ;;
  esac
done

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found." >&2; exit 1; }; }
need_cmd nvcc; need_cmd nsys; need_cmd python3

export CUDA_VISIBLE_DEVICES="${GPU}"
rm -rf "${OUT}"
mkdir -p "${OUT}"/{trace,nsys}

echo "═══════════════════════════════════════════════════════════════"
echo "  IKP + NSys Integration Demo  (GPU ${GPU})"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Build
echo "[0/3] Build"
if [[ -z "${CUDA_HOME:-}" ]]; then
  CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -lineinfo \
  -I "${ROOT}/include" \
  "${HERE}/nsys_demo.cu" -o "${HERE}/nsys_demo"

# Single execution: nsys + IKP simultaneously
echo "[1/3] NSys + IKP profile (single kernel launch)"
nsys profile \
  --output="${OUT}/nsys/report" \
  --force-overwrite=true \
  --trace=cuda \
  "${HERE}/nsys_demo" --m=2048 --n=2048 --k=2048 \
    --out="${OUT}/trace/gemm_trace.json"

# Import + merge
echo "[2/3] Import & merge"
python3 "${ROOT}/scripts/ikp_nsys_import.py" \
  --nsys-rep "${OUT}/nsys/report.nsys-rep" \
  --out-dir "${OUT}/nsys/" \
  --kernel-regex "gemm_kernel"

python3 "${ROOT}/scripts/ikp_nsys_merge.py" \
  --nsys-events "${OUT}/nsys/nsys_events.json" \
  --ikp-trace "${OUT}/trace/gemm_trace.json" \
  --nsys-kernels "${OUT}/nsys/nsys_kernels.json" \
  --kernel-regex "gemm_kernel" \
  --out "${OUT}/trace/merged_trace.json"

# Explorer
echo "[3/3] Explorer"
python3 "${ROOT}/scripts/generate_explorer.py" \
  --demo-dir "${OUT}" \
  --source "${HERE}/nsys_demo.cu" \
  --output "${OUT}/explorer.html"

echo ""
echo "Done. Open in Perfetto: ${OUT}/trace/merged_trace.json"

# Cleanup binary
rm -f "${HERE}/nsys_demo"
