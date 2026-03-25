#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT="${ROOT}/_demo_out"
NVBIT_PATH_DEFAULT=""
NVBIT_PATH="${NVBIT_PATH:-}"
ARCH="${ARCH:-90a}"
SM_ARCH="${SM_ARCH:-sm_90a}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all_examples.sh [--out=DIR] [--nvbit-path=PATH] [--arch=90a] [--sm=sm_90a]

This script builds and runs the full Intra-Kernel Profiler demo pipeline:

  Phase 1 — Intra-kernel trace examples (CMake)
  Phase 2 — CUPTI Activity API sanity check
  Phase 3 — CUPTI injection collectors (pcsamp, sassmetrics ×5 profiles, instrexec, source mapping)
  Phase 4 — NVBit tool (pcmap, all, inst_pipe, bb_hot, nvdisasm, ptx)
  Phase 5 — Region demo (joinable NVBit pcmap + CUPTI sassmetrics)
  Phase 6 — Post-processing: join analysis, merge scripts, trace plots, JSON validation, HTML report

Outputs go to --out (default: _demo_out/).

Notes:
  - NVBit requires NVBIT_PATH to point to NVBit root (contains core/libnvbit.a).
  - Some CUPTI features may be restricted on clusters; outputs may contain warnings.
EOF
}

for a in "$@"; do
  case "$a" in
    --out=*) OUT="${a#--out=}" ;;
    --nvbit-path=*) NVBIT_PATH="${a#--nvbit-path=}" ;;
    --arch=*) ARCH="${a#--arch=}" ;;
    --sm=*) SM_ARCH="${a#--sm=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $a" >&2; usage; exit 2 ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 1; }
}

need_cmd cmake
need_cmd make
need_cmd nvcc
need_cmd python3

if [[ -z "${NVBIT_PATH}" ]]; then
  if [[ -f "${NVBIT_PATH_DEFAULT}/core/libnvbit.a" ]]; then
    NVBIT_PATH="${NVBIT_PATH_DEFAULT}"
  fi
fi

if [[ ! -f "${NVBIT_PATH}/core/libnvbit.a" ]]; then
  echo "NVBit not found. Set --nvbit-path=... or NVBIT_PATH env var." >&2
  echo "Expected: <NVBIT_PATH>/core/libnvbit.a" >&2
  exit 1
fi

rm -rf "${OUT}"
mkdir -p "${OUT}"/{trace,cupti,nvbit,join}

# ─────────────────────────────────────────────────────────────────────
# Phase 1: Build + run trace examples
# ─────────────────────────────────────────────────────────────────────
echo "[1/6] Build + run trace examples"
cmake -S "${ROOT}" -B "${ROOT}/_ci_build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${ROOT}/_ci_build" -j >/dev/null

"${ROOT}/_ci_build/ikp_trace_record" --iters=1000 --out="${OUT}/trace/record_trace.json" >/dev/null
"${ROOT}/_ci_build/ikp_trace_block_filter" --iters=1000 --out="${OUT}/trace/block_filter_trace.json" >/dev/null
"${ROOT}/_ci_build/ikp_trace_sampled_loop" --iters=10000 --sample_shift=8 --out="${OUT}/trace/sampled_loop_trace.json" >/dev/null
"${ROOT}/_ci_build/ikp_gemm_demo" --m=1024 --n=1024 --k=1024 --out="${OUT}/trace/gemm_trace.json"

# ─────────────────────────────────────────────────────────────────────
# Phase 2: CUPTI minimal activity
# ─────────────────────────────────────────────────────────────────────
echo "[2/6] CUPTI minimal activity"
make -C "${ROOT}/tools/cupti_minimal_activity" -j >/dev/null
bash "${ROOT}/tools/cupti_minimal_activity/run.sh" > "${OUT}/cupti/minimal_activity.log"

# ─────────────────────────────────────────────────────────────────────
# Phase 3: CUPTI injection collectors
# ─────────────────────────────────────────────────────────────────────
echo "[3/6] CUPTI injection collectors"
make -C "${ROOT}/tools/cupti_region_profiler" -j >/dev/null
pushd "${ROOT}/examples/cupti" >/dev/null
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -lineinfo minimal_cupti_target.cu -o minimal_cupti_target

# 3a. PC sampling — serialized mode
CUDA_INJECTION64_PATH="${ROOT}/tools/cupti_region_profiler/ikp_cupti_pcsamp.so" \
IKP_CUPTI_PCSAMP_OUT="${OUT}/cupti/pcsampling_raw.json" \
IKP_CUPTI_PCSAMP_COLLECTION_MODE=serialized \
IKP_CUPTI_PCSAMP_KERNEL_REGEX=cupti_target_kernel \
IKP_CUPTI_PCSAMP_PERIOD=5 \
IKP_CUPTI_PCSAMP_MAX_PCS=10000 \
IKP_CUPTI_PCSAMP_MAX_RECORDS=0 \
IKP_CUPTI_PCSAMP_VERBOSE=1 \
./minimal_cupti_target --iters=20 --inner=4096 > "${OUT}/cupti/pcsamp_run.log"

# 3b. SASS metrics — all five profiles
for profile in core divergence memory instruction_mix branch; do
  CUDA_INJECTION64_PATH="${ROOT}/tools/cupti_region_profiler/ikp_cupti_sassmetrics.so" \
  IKP_CUPTI_SASS_OUT="${OUT}/cupti/sassmetrics_${profile}.json" \
  IKP_CUPTI_SASS_PROFILE="${profile}" \
  IKP_CUPTI_SASS_LAZY_PATCHING=1 \
  IKP_CUPTI_SASS_ENABLE_SOURCE=0 \
  ./minimal_cupti_target --iters=20 --inner=4096 > "${OUT}/cupti/sassmetrics_${profile}_run.log"
done

# 3c. SASS metrics with source mapping (requires -lineinfo)
CUDA_INJECTION64_PATH="${ROOT}/tools/cupti_region_profiler/ikp_cupti_sassmetrics.so" \
IKP_CUPTI_SASS_OUT="${OUT}/cupti/sassmetrics_source.json" \
IKP_CUPTI_SASS_PROFILE=core \
IKP_CUPTI_SASS_LAZY_PATCHING=1 \
IKP_CUPTI_SASS_ENABLE_SOURCE=1 \
./minimal_cupti_target --iters=20 --inner=4096 > "${OUT}/cupti/sassmetrics_source_run.log"

# 3d. SASS metrics — list available metrics
CUDA_INJECTION64_PATH="${ROOT}/tools/cupti_region_profiler/ikp_cupti_sassmetrics.so" \
IKP_CUPTI_SASS_LIST=1 \
IKP_CUPTI_SASS_LIST_OUT="${OUT}/cupti/sass_metrics_list.txt" \
./minimal_cupti_target --iters=1 --inner=1 > /dev/null

# 3e. Instruction execution
CUDA_INJECTION64_PATH="${ROOT}/tools/cupti_region_profiler/ikp_cupti_instrexec.so" \
IKP_CUPTI_INSTREXEC_OUT="${OUT}/cupti/instrexec_raw.json" \
IKP_CUPTI_INSTREXEC_KERNEL_REGEX=cupti_target_kernel \
IKP_CUPTI_INSTREXEC_MAX_RECORDS=0 \
./minimal_cupti_target --iters=20 --inner=4096 > "${OUT}/cupti/instrexec_run.log"

popd >/dev/null

# ─────────────────────────────────────────────────────────────────────
# Phase 4: NVBit tool — all modes + advanced features
# ─────────────────────────────────────────────────────────────────────
echo "[4/6] NVBit tool (pcmap, all, inst_pipe, bb_hot, nvdisasm, ptx)"
make -C "${ROOT}/tools/nvbit_region_profiler" NVBIT_PATH="${NVBIT_PATH}" ARCH="${ARCH}" -j >/dev/null

pushd "${ROOT}/examples/nvbit" >/dev/null
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I "${ROOT}/include" \
  minimal_nvbit_target.cu "${ROOT}/src/nvbit_marker_device.cu" \
  -o minimal_nvbit_target

NVBIT_SO="${ROOT}/tools/nvbit_region_profiler/region_profiler.so"

# Common env for all NVBit runs
nvbit_run() {
  local mode="$1"; shift
  local outdir="${OUT}/nvbit/${mode}"
  rm -rf "${outdir}"; mkdir -p "${outdir}"
  env \
    IKP_NVBIT_ENABLE=1 \
    IKP_NVBIT_KERNEL_REGEX=nvbit_marked_kernel \
    IKP_NVBIT_TRACE_PATH="${outdir}" \
    "$@" \
    LD_PRELOAD="${NVBIT_SO}" \
    ./minimal_nvbit_target > "${outdir}/run.log"
}

# 4a. pcmap — basic PC-to-region attribution
nvbit_run pcmap IKP_NVBIT_MODE=pcmap

# 4b. all — pcmap + instmix + memtrace
nvbit_run all IKP_NVBIT_MODE=all IKP_NVBIT_TRACE_CAP=4096

# 4c. inst_pipe — instruction pipeline attribution (16 categories)
nvbit_run inst_pipe IKP_NVBIT_MODE=pcmap IKP_NVBIT_ENABLE_INST_PIPE=1

# 4d. bb_hot + branch_sites — basic block hotspots
nvbit_run bb_hot IKP_NVBIT_MODE=pcmap IKP_NVBIT_ENABLE_BB_HOT=1 IKP_NVBIT_ENABLE_BRANCH_SITES=1

# 4e. nvdisasm — high-quality SASS with metadata + lineinfo
nvbit_run nvdisasm IKP_NVBIT_MODE=pcmap \
  IKP_NVBIT_DUMP_NVDISASM_SASS=1 IKP_NVBIT_DUMP_SASS_META=1 \
  IKP_NVBIT_DUMP_SASS_LINEINFO=1 IKP_NVBIT_KEEP_CUBIN=1

# 4f. ptx — PTX dump with per-region slices
nvbit_run ptx IKP_NVBIT_MODE=pcmap IKP_NVBIT_DUMP_PTX=1 IKP_NVBIT_DUMP_PTX_BY_REGION=1

popd >/dev/null

# ─────────────────────────────────────────────────────────────────────
# Phase 5: Region demo — joinable NVBit + CUPTI dataset
# ─────────────────────────────────────────────────────────────────────
echo "[5/6] Region demo (NVBit pcmap + CUPTI sassmetrics)"
pushd "${ROOT}/examples/region_demo" >/dev/null
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I "${ROOT}/include" \
  minimal_region_target.cu "${ROOT}/src/nvbit_marker_device.cu" \
  -o minimal_region_target

rm -rf "${OUT}/join/nvbit" "${OUT}/join/cupti"
mkdir -p "${OUT}/join/nvbit" "${OUT}/join/cupti"

IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=region_demo_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_TRACE_PATH="${OUT}/join/nvbit" \
LD_PRELOAD="${ROOT}/tools/nvbit_region_profiler/region_profiler.so" \
./minimal_region_target --iters=20 --inner=4096 > "${OUT}/join/nvbit/run.log"

# Run all five CUPTI SASS profiles on the same target
for profile in core divergence memory instruction_mix; do
  CUDA_INJECTION64_PATH="${ROOT}/tools/cupti_region_profiler/ikp_cupti_sassmetrics.so" \
  IKP_CUPTI_SASS_OUT="${OUT}/join/cupti/sassmetrics_${profile}.json" \
  IKP_CUPTI_SASS_PROFILE="${profile}" \
  IKP_CUPTI_SASS_LAZY_PATCHING=1 \
  IKP_CUPTI_SASS_ENABLE_SOURCE=1 \
  ./minimal_region_target --iters=20 --inner=4096 > "${OUT}/join/cupti/sassmetrics_${profile}_run.log"
done

popd >/dev/null

# ─────────────────────────────────────────────────────────────────────
# Phase 6: Post-processing
# ─────────────────────────────────────────────────────────────────────
echo "[6/6] Post-processing: join, merge, plots, report"

# 6a. NVBit + CUPTI join analysis
python3 "${ROOT}/scripts/analyze_cupti_join.py" \
  --nvbit-dir "${OUT}/join/nvbit" \
  --cupti-dir "${OUT}/join/cupti" \
  --labels "0:outside,1:compute,2:store" \
  > "${OUT}/join/join_analysis.txt"

# 6b. Merge scripts
python3 "${ROOT}/scripts/ikp_cupti_sassmetrics_merge.py" \
  --sassmetrics "${OUT}/join/cupti/sassmetrics_core.json" \
  --pc2region "${OUT}/join/nvbit/pc2region_region_demo_kernel_0.json" \
  --out "${OUT}/join/merged_sassmetrics.json"

# 6c. Trace summary plots
python3 "${ROOT}/scripts/plot_trace_summary.py" \
  --summary "${OUT}/trace/gemm_trace_summary.json" \
  --out_dir "${OUT}/trace/plots"

# 6d. Locality analysis (from memtrace)
python3 "${ROOT}/scripts/nvbit_locality.py" \
  --trace "${OUT}/nvbit/all/mem_trace_nvbit_marked_kernel_0.jsonl" \
  --out "${OUT}/nvbit/all/locality_analysis.json" \
  --line-bytes 128 --window-records 64

# 6e. Visualization gallery
python3 "${ROOT}/scripts/generate_gallery.py" \
  --demo-dir "${OUT}" \
  --out-dir "${OUT}/gallery"

# 6f. Validate all JSON
python3 "${ROOT}/scripts/validate_json.py" \
  "${OUT}"/trace/*.json \
  "${OUT}"/cupti/pcsampling_raw.json \
  "${OUT}"/cupti/sassmetrics_*.json \
  "${OUT}"/cupti/instrexec_raw.json \
  "${OUT}"/nvbit/pcmap/*.json \
  "${OUT}"/nvbit/all/*.json \
  "${OUT}"/join/nvbit/*.json \
  "${OUT}"/join/cupti/*.json

# 6g. Source annotation
python3 "${ROOT}/scripts/annotate_source.py" \
  --sass "${OUT}/cupti/sassmetrics_source.json" \
  --source "${ROOT}/examples/cupti/minimal_cupti_target.cu" \
  --html "${OUT}/annotated_source.html"

# 6h. IKP Explorer
python3 "${ROOT}/scripts/generate_explorer.py" \
  --demo-dir "${OUT}" \
  --source "${ROOT}/examples/cupti/minimal_cupti_target.cu" \
  --output "${OUT}/explorer.html"

echo ""
echo "DONE. Outputs:"
echo "  ${OUT}/"
echo "  ${OUT}/explorer.html    (IKP Explorer)"
echo "  ${OUT}/annotated_source.html"
echo "  ${OUT}/gallery/  (visualization images)"
