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

OUT="${HERE}/_out"
rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "=== Building NVBit region profiler tool ==="
make -C "${ROOT}/tools/nvbit_region_profiler" NVBIT_PATH="${NVBIT_PATH}" ARCH="${ARCH}" -j >/dev/null

echo "=== Compiling minimal_nvbit_target (markers enabled) ==="
nvcc -O3 -std=c++17 -arch="${SM_ARCH}" -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I "${ROOT}/include" \
  "${HERE}/minimal_nvbit_target.cu" "${ROOT}/src/nvbit_marker_device.cu" \
  -o "${HERE}/minimal_nvbit_target"

BIN="${HERE}/minimal_nvbit_target"
NVBIT_SO="${ROOT}/tools/nvbit_region_profiler/region_profiler.so"

nvbit_run() {
  local mode="$1"; shift
  local outdir="${OUT}/${mode}"
  mkdir -p "${outdir}"
  echo "  mode: ${mode}"
  IKP_NVBIT_ENABLE=1 \
  IKP_NVBIT_KERNEL_REGEX=nvbit_marked_kernel \
  IKP_NVBIT_TRACE_PATH="${outdir}" \
  "$@" \
  LD_PRELOAD="${NVBIT_SO}" \
  "${BIN}" >/dev/null
}

echo "=== Running 6 NVBit modes ==="
nvbit_run pcmap      IKP_NVBIT_MODE=pcmap
nvbit_run all        IKP_NVBIT_MODE=all IKP_NVBIT_TRACE_CAP=4096
nvbit_run inst_pipe  IKP_NVBIT_MODE=pcmap IKP_NVBIT_ENABLE_INST_PIPE=1
nvbit_run bb_hot     IKP_NVBIT_MODE=pcmap IKP_NVBIT_ENABLE_BB_HOT=1 IKP_NVBIT_ENABLE_BRANCH_SITES=1
nvbit_run nvdisasm   IKP_NVBIT_MODE=pcmap IKP_NVBIT_DUMP_NVDISASM_SASS=1 IKP_NVBIT_DUMP_SASS_META=1 IKP_NVBIT_DUMP_SASS_LINEINFO=1 IKP_NVBIT_KEEP_CUBIN=1
nvbit_run ptx        IKP_NVBIT_MODE=pcmap IKP_NVBIT_DUMP_PTX=1 IKP_NVBIT_DUMP_PTX_BY_REGION=1

echo ""
echo "DONE. Outputs:"
find "${OUT}" -type f | sort
