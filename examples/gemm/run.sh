#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# examples/gemm/run.sh — Full profiling pipeline for the tiled GEMM
# ═══════════════════════════════════════════════════════════════════════
#
# This is THE reference script for profiling your first kernel with
# Intra-Kernel Profiler.  It runs every profiling stage on the tiled GEMM
# example (examples/gemm/tiled_gemm.cu) and finishes by generating a
# Compiler-Explorer-style HTML viewer.
#
# Stages:
#   0  Build everything (CMake trace binary, CUPTI collectors, NVBit
#      tool, and a separate NVBit-enabled GEMM binary)
#   1  Intra-kernel trace  (ring-buffer timing per region)
#   2  NVBit profiling     (5 modes: pcmap, all, inst_pipe, bb_hot, nvdisasm)
#   3  CUPTI profiling     (5 SASS profiles + PC sampling + instrexec + source)
#   4  Locality analysis   (stride & reuse-distance from NVBit memtrace)
#   5  Generate Explorer   (self-contained HTML dashboard)
#
# Usage:
#   bash examples/gemm/run.sh --nvbit-path=/path/to/nvbit
#   bash examples/gemm/run.sh --nvbit-path=/path/to/nvbit --arch=90a --sm=sm_90a
#
# Requirements:
#   cmake, nvcc, make, python3, and a valid NVBIT_PATH.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────
NVBIT_PATH="${NVBIT_PATH:-}"
ARCH="${ARCH:-90a}"
SM_ARCH="${SM_ARCH:-sm_90a}"
OUT="${HERE}/_out"

# ── Parse arguments ───────────────────────────────────────────────────
usage() {
  cat <<'EOF'
Usage:
  bash examples/gemm/run.sh [OPTIONS]

Options:
  --nvbit-path=PATH   Path to NVBit installation (must contain core/libnvbit.a)
  --arch=ARCH         GPU architecture for NVBit (default: 90a)
  --sm=SM             nvcc -arch flag (default: sm_90a)
  --out=DIR           Output directory (default: examples/gemm/_out)
  -h, --help          Show this help

Environment variables NVBIT_PATH, ARCH, SM_ARCH are also accepted.
EOF
}

for a in "$@"; do
  case "$a" in
    --nvbit-path=*) NVBIT_PATH="${a#--nvbit-path=}" ;;
    --arch=*)       ARCH="${a#--arch=}" ;;
    --sm=*)         SM_ARCH="${a#--sm=}" ;;
    --out=*)        OUT="${a#--out=}" ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown argument: $a" >&2; usage; exit 2 ;;
  esac
done

# ── Prerequisite checks ──────────────────────────────────────────────
need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found in PATH." >&2; exit 1; }
}
need_cmd cmake
need_cmd nvcc
need_cmd make
need_cmd python3

if [[ -z "${NVBIT_PATH}" ]] || [[ ! -f "${NVBIT_PATH}/core/libnvbit.a" ]]; then
  echo "ERROR: NVBIT_PATH not set or invalid." >&2
  echo "  Provide --nvbit-path=/path/to/nvbit  (must contain core/libnvbit.a)" >&2
  exit 1
fi

# Resolve CUDA_HOME for CUPTI libraries
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else
    echo "ERROR: CUDA_HOME not set and nvcc not in PATH." >&2; exit 1
  fi
fi
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"

# Kernel name used in tiled_gemm.cu (__global__ void tiled_gemm_kernel)
KERNEL_REGEX="tiled_gemm_kernel"

# ── Prepare output directory ──────────────────────────────────────────
rm -rf "${OUT}"
mkdir -p "${OUT}"/{trace,nvbit/{pcmap,all,inst_pipe,bb_hot,nvdisasm,ptx},cupti}

echo "═══════════════════════════════════════════════════════════════"
echo "  Intra-Kernel Profiler — GEMM full pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo "  ROOT:       ${ROOT}"
echo "  OUT:        ${OUT}"
echo "  NVBIT_PATH: ${NVBIT_PATH}"
echo "  ARCH:       ${ARCH}  SM: ${SM_ARCH}"
echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 0: Build everything
# ═════════════════════════════════════════════════════════════════════
echo "[0/5] Building..."

# 0a. CMake build (trace-only binary: ikp_gemm_demo)
echo "  CMake: trace binary (ikp_gemm_demo)"
cmake -S "${ROOT}" -B "${ROOT}/_ci_build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${ROOT}/_ci_build" -j >/dev/null

# 0b. CUPTI injection collectors
echo "  Make:  CUPTI collectors"
make -C "${ROOT}/tools/cupti_region_profiler" -j >/dev/null

# 0c. NVBit region profiler tool
echo "  Make:  NVBit region profiler"
make -C "${ROOT}/tools/nvbit_region_profiler" NVBIT_PATH="${NVBIT_PATH}" ARCH="${ARCH}" -j >/dev/null

# 0d. NVBit-enabled GEMM binary (separate from the CMake trace binary)
#     Requires -DIKP_ENABLE_NVBIT_MARKERS, -rdc=true, and linking
#     src/nvbit_marker_device.cu.
echo "  nvcc:  NVBit-enabled GEMM binary (tiled_gemm_nvbit)"
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -lineinfo -rdc=true \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I "${ROOT}/include" \
  "${HERE}/tiled_gemm.cu" "${ROOT}/src/nvbit_marker_device.cu" \
  -o "${HERE}/tiled_gemm_nvbit"

# 0e. CUPTI target — plain build WITHOUT NVBit markers.
#     CUPTI injection should run against a clean binary (no extern
#     device calls from NVBit markers that could perturb metrics).
#     The CMake-built ikp_gemm_demo already satisfies this, but we
#     also build a standalone binary with -lineinfo for CUPTI source
#     mapping to reference the local .cu file path directly.
echo "  nvcc:  CUPTI target (tiled_gemm)"
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -lineinfo \
  -I "${ROOT}/include" \
  "${HERE}/tiled_gemm.cu" \
  -o "${HERE}/tiled_gemm"

echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 1: Intra-kernel trace
# ═════════════════════════════════════════════════════════════════════
echo "[1/5] Intra-kernel trace"
"${ROOT}/_ci_build/ikp_gemm_demo" \
  --m=2048 --n=2048 --k=2048 \
  --out="${OUT}/trace/gemm_trace.json"
echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 2: NVBit (5 modes)
# ═════════════════════════════════════════════════════════════════════
echo "[2/5] NVBit profiling (5 modes)"
NVBIT_BIN="${HERE}/tiled_gemm_nvbit"
NVBIT_SO="${ROOT}/tools/nvbit_region_profiler/region_profiler.so"

nvbit_run() {
  local label="$1"; shift
  local outdir="$1"; shift
  echo "  mode: ${label}"
  env \
    IKP_NVBIT_ENABLE=1 \
    IKP_NVBIT_KERNEL_REGEX="${KERNEL_REGEX}" \
    IKP_NVBIT_TRACE_PATH="${outdir}" \
    "$@" \
    LD_PRELOAD="${NVBIT_SO}" \
    "${NVBIT_BIN}" --m=2048 --n=2048 --k=2048 --iters=5 >/dev/null
}

# 2a. pcmap — PC-to-region attribution
nvbit_run pcmap "${OUT}/nvbit/pcmap" \
  IKP_NVBIT_MODE=pcmap

# 2b. all — pcmap + instmix + memtrace
nvbit_run all "${OUT}/nvbit/all" \
  IKP_NVBIT_MODE=all IKP_NVBIT_TRACE_CAP=4096

# 2c. inst_pipe — per-pipeline instruction counts (16 categories)
nvbit_run inst_pipe "${OUT}/nvbit/inst_pipe" \
  IKP_NVBIT_MODE=pcmap IKP_NVBIT_ENABLE_INST_PIPE=1

# 2d. bb_hot — basic-block hotspots + branch sites
nvbit_run bb_hot "${OUT}/nvbit/bb_hot" \
  IKP_NVBIT_MODE=pcmap IKP_NVBIT_ENABLE_BB_HOT=1 IKP_NVBIT_ENABLE_BRANCH_SITES=1

# 2e. nvdisasm — high-quality SASS with metadata + lineinfo
nvbit_run nvdisasm "${OUT}/nvbit/nvdisasm" \
  IKP_NVBIT_MODE=pcmap \
  IKP_NVBIT_DUMP_NVDISASM_SASS=1 IKP_NVBIT_DUMP_SASS_META=1 \
  IKP_NVBIT_DUMP_SASS_LINEINFO=1 IKP_NVBIT_KEEP_CUBIN=1

# 2f. ptx — PTX dump with per-region slices
nvbit_run ptx "${OUT}/nvbit/ptx" \
  IKP_NVBIT_MODE=pcmap IKP_NVBIT_DUMP_PTX=1 IKP_NVBIT_DUMP_PTX_BY_REGION=1

echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 3: CUPTI profiling
# ═════════════════════════════════════════════════════════════════════
# NOTE: CUPTI runs use the plain binary (tiled_gemm), NOT the NVBit
# binary.  NVBit markers inject extern device calls that would change
# the instruction mix and perturb hardware counter measurements.
echo "[3/5] CUPTI profiling"
CUPTI_BIN="${HERE}/tiled_gemm"
CUPTI_DIR="${ROOT}/tools/cupti_region_profiler"

# 3a. SASS metrics — 5 hardware counter profiles
echo "  SASS metrics (5 profiles)"
for profile in core divergence memory instruction_mix branch; do
  echo "    profile: ${profile}"
  CUDA_INJECTION64_PATH="${CUPTI_DIR}/ikp_cupti_sassmetrics.so" \
  IKP_CUPTI_SASS_OUT="${OUT}/cupti/sassmetrics_${profile}.json" \
  IKP_CUPTI_SASS_PROFILE="${profile}" \
  IKP_CUPTI_SASS_LAZY_PATCHING=1 \
  IKP_CUPTI_SASS_ENABLE_SOURCE=0 \
  "${CUPTI_BIN}" --m=2048 --n=2048 --k=2048 --iters=20 >/dev/null
done

# 3b. SASS metrics with source mapping (requires -lineinfo)
echo "  SASS metrics with source mapping"
CUDA_INJECTION64_PATH="${CUPTI_DIR}/ikp_cupti_sassmetrics.so" \
IKP_CUPTI_SASS_OUT="${OUT}/cupti/sassmetrics_source.json" \
IKP_CUPTI_SASS_PROFILE=core \
IKP_CUPTI_SASS_LAZY_PATCHING=1 \
IKP_CUPTI_SASS_ENABLE_SOURCE=1 \
"${CUPTI_BIN}" --m=2048 --n=2048 --k=2048 --iters=20 >/dev/null

# 3c. PC sampling
echo "  PC sampling"
CUDA_INJECTION64_PATH="${CUPTI_DIR}/ikp_cupti_pcsamp.so" \
IKP_CUPTI_PCSAMP_OUT="${OUT}/cupti/pcsampling_raw.json" \
IKP_CUPTI_PCSAMP_COLLECTION_MODE=serialized \
IKP_CUPTI_PCSAMP_KERNEL_REGEX="${KERNEL_REGEX}" \
IKP_CUPTI_PCSAMP_PERIOD=5 \
IKP_CUPTI_PCSAMP_MAX_PCS=10000 \
IKP_CUPTI_PCSAMP_VERBOSE=1 \
"${CUPTI_BIN}" --m=2048 --n=2048 --k=2048 --iters=20 > "${OUT}/cupti/pcsamp_run.log"

# 3d. Instruction execution
echo "  Instruction execution"
CUDA_INJECTION64_PATH="${CUPTI_DIR}/ikp_cupti_instrexec.so" \
IKP_CUPTI_INSTREXEC_OUT="${OUT}/cupti/instrexec_raw.json" \
IKP_CUPTI_INSTREXEC_KERNEL_REGEX="${KERNEL_REGEX}" \
IKP_CUPTI_INSTREXEC_MAX_RECORDS=0 \
"${CUPTI_BIN}" --m=2048 --n=2048 --k=2048 --iters=20 >/dev/null

echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 4: Locality analysis
# ═════════════════════════════════════════════════════════════════════
echo "[4/5] Locality analysis"

# Find the memtrace JSONL produced by NVBit 'all' mode.
MEMTRACE=$(find "${OUT}/nvbit/all" -name 'mem_trace_*.jsonl' 2>/dev/null | head -1 || true)
if [[ -n "${MEMTRACE}" ]]; then
  python3 "${ROOT}/scripts/nvbit_locality.py" \
    --trace "${MEMTRACE}" \
    --out "${OUT}/nvbit/all/locality_analysis.json" \
    --line-bytes 128 --window-records 64
  echo "  wrote ${OUT}/nvbit/all/locality_analysis.json"
else
  echo "  SKIP: no mem_trace_*.jsonl found (NVBit 'all' mode may not have produced memtrace)"
fi
echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 5: Generate Explorer
# ═════════════════════════════════════════════════════════════════════
echo "[5/5] Generate Explorer"
python3 "${ROOT}/scripts/generate_explorer.py" \
  --demo-dir "${OUT}" \
  --source "${HERE}/tiled_gemm.cu" \
  --output "${OUT}/explorer.html"
echo ""

# ═════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════"
echo "  DONE. Output directory: ${OUT}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Key outputs:"
echo ""

print_size() {
  if [[ -f "$1" ]]; then
    printf "  %-50s %s\n" "$1" "$(du -h "$1" | cut -f1)"
  fi
}

print_size "${OUT}/trace/gemm_trace.json"
print_size "${OUT}/trace/gemm_trace_summary.json"
echo ""
echo "  NVBit (5 modes):"
for mode in pcmap all inst_pipe bb_hot nvdisasm ptx; do
  count=$(find "${OUT}/nvbit/${mode}" -type f 2>/dev/null | wc -l)
  printf "    %-20s %d files\n" "${mode}/" "${count}"
done
echo ""
echo "  CUPTI:"
for f in "${OUT}"/cupti/*.json; do
  [[ -f "$f" ]] && printf "    %-50s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done
echo ""
print_size "${OUT}/explorer.html"
echo ""
echo "To view the Explorer:"
echo "  1. Copy explorer.html to a machine with a browser, or"
echo "  2. python3 -m http.server -d ${OUT} 8080"
echo "     then open http://localhost:8080/explorer.html"
echo ""
echo "To view the trace in Perfetto:"
echo "  Open ${OUT}/trace/gemm_trace.json at https://ui.perfetto.dev"
