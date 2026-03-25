#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
SM_ARCH="${SM_ARCH:-sm_90a}"

OUT="${HERE}/_out"
mkdir -p "${OUT}"

# Resolve CUDA_HOME for CUPTI libraries
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else
    echo "ERROR: CUDA_HOME not set and nvcc not in PATH." >&2; exit 1
  fi
fi
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"

echo "=== Building CUPTI injection collectors ==="
make -C "${ROOT}/tools/cupti_region_profiler" -j >/dev/null

echo "=== Compiling minimal_cupti_target ==="
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -lineinfo \
  "${HERE}/minimal_cupti_target.cu" -o "${HERE}/minimal_cupti_target"

BIN="${HERE}/minimal_cupti_target"
CUPTI="${ROOT}/tools/cupti_region_profiler"

echo "=== [1/5] PC Sampling ==="
CUDA_INJECTION64_PATH="${CUPTI}/ikp_cupti_pcsamp.so" \
IKP_CUPTI_PCSAMP_OUT="${OUT}/pcsampling_raw.json" \
IKP_CUPTI_PCSAMP_COLLECTION_MODE=serialized \
IKP_CUPTI_PCSAMP_KERNEL_REGEX=cupti_target_kernel \
IKP_CUPTI_PCSAMP_PERIOD=5 \
IKP_CUPTI_PCSAMP_MAX_PCS=10000 \
IKP_CUPTI_PCSAMP_VERBOSE=1 \
"${BIN}" --iters=20 --inner=4096

echo ""
echo "=== [2/5] SASS Metrics (5 profiles) ==="
for profile in core divergence memory instruction_mix branch; do
  echo "  profile: ${profile}"
  CUDA_INJECTION64_PATH="${CUPTI}/ikp_cupti_sassmetrics.so" \
  IKP_CUPTI_SASS_OUT="${OUT}/sassmetrics_${profile}.json" \
  IKP_CUPTI_SASS_PROFILE="${profile}" \
  IKP_CUPTI_SASS_LAZY_PATCHING=1 \
  IKP_CUPTI_SASS_ENABLE_SOURCE=0 \
  "${BIN}" --iters=20 --inner=4096 >/dev/null
done

echo ""
echo "=== [3/5] SASS Metrics with source mapping ==="
CUDA_INJECTION64_PATH="${CUPTI}/ikp_cupti_sassmetrics.so" \
IKP_CUPTI_SASS_OUT="${OUT}/sassmetrics_source.json" \
IKP_CUPTI_SASS_PROFILE=core \
IKP_CUPTI_SASS_LAZY_PATCHING=1 \
IKP_CUPTI_SASS_ENABLE_SOURCE=1 \
"${BIN}" --iters=20 --inner=4096 >/dev/null

echo ""
echo "=== [4/5] List available SASS metrics ==="
CUDA_INJECTION64_PATH="${CUPTI}/ikp_cupti_sassmetrics.so" \
IKP_CUPTI_SASS_LIST=1 \
IKP_CUPTI_SASS_LIST_OUT="${OUT}/sass_metrics_list.txt" \
"${BIN}" --iters=1 --inner=1 >/dev/null

echo ""
echo "=== [5/5] Instruction Execution ==="
CUDA_INJECTION64_PATH="${CUPTI}/ikp_cupti_instrexec.so" \
IKP_CUPTI_INSTREXEC_OUT="${OUT}/instrexec_raw.json" \
IKP_CUPTI_INSTREXEC_KERNEL_REGEX=cupti_target_kernel \
IKP_CUPTI_INSTREXEC_MAX_RECORDS=0 \
"${BIN}" --iters=20 --inner=4096 >/dev/null

echo ""
echo "DONE. Outputs:"
ls -lh "${OUT}/"
