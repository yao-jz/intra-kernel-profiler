#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
SM_ARCH="${SM_ARCH:-sm_90a}"
ARCH="${ARCH:-90a}"
NVBIT_PATH="${NVBIT_PATH:-}"

if [[ -z "${NVBIT_PATH}" ]] || [[ ! -f "${NVBIT_PATH}/core/libnvbit.a" ]]; then
  echo "ERROR: NVBIT_PATH not set or invalid." >&2
  echo "  export NVBIT_PATH=/path/to/nvbit   # must contain core/libnvbit.a" >&2
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

OUT="${HERE}/_out"
rm -rf "${OUT}"
mkdir -p "${OUT}/nvbit" "${OUT}/cupti"

echo "=== Building NVBit region profiler ==="
make -C "${ROOT}/tools/nvbit_region_profiler" NVBIT_PATH="${NVBIT_PATH}" ARCH="${ARCH}" -j >/dev/null

echo "=== Building CUPTI injection collectors ==="
make -C "${ROOT}/tools/cupti_region_profiler" -j >/dev/null

echo "=== Compiling minimal_region_target (markers enabled) ==="
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I "${ROOT}/include" \
  "${HERE}/minimal_region_target.cu" "${ROOT}/src/nvbit_marker_device.cu" \
  -o "${HERE}/minimal_region_target"

BIN="${HERE}/minimal_region_target"

echo "=== [1/3] NVBit pcmap run ==="
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=region_demo_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_TRACE_PATH="${OUT}/nvbit" \
LD_PRELOAD="${ROOT}/tools/nvbit_region_profiler/region_profiler.so" \
"${BIN}" --iters=20 --inner=4096 >/dev/null

echo "=== [2/3] CUPTI SASS metrics (4 profiles) ==="
for profile in core divergence memory instruction_mix; do
  echo "  profile: ${profile}"
  CUDA_INJECTION64_PATH="${ROOT}/tools/cupti_region_profiler/ikp_cupti_sassmetrics.so" \
  IKP_CUPTI_SASS_OUT="${OUT}/cupti/sassmetrics_${profile}.json" \
  IKP_CUPTI_SASS_PROFILE="${profile}" \
  IKP_CUPTI_SASS_LAZY_PATCHING=1 \
  IKP_CUPTI_SASS_ENABLE_SOURCE=1 \
  "${BIN}" --iters=20 --inner=4096 >/dev/null
done

echo "=== [3/3] Join analysis ==="
python3 "${ROOT}/scripts/analyze_cupti_join.py" \
  --nvbit-dir "${OUT}/nvbit" \
  --cupti-dir "${OUT}/cupti" \
  --labels "0:outside,1:compute,2:store" \
  | tee "${OUT}/join_analysis.txt"

echo ""
echo "DONE. Outputs:"
find "${OUT}" -type f | sort
