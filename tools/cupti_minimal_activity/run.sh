#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  elif [[ -d /usr/local/cuda ]]; then
    CUDA_HOME="/usr/local/cuda"
  else
    echo "CUDA_HOME is not set and nvcc not found." >&2
    exit 1
  fi
fi

export CUDA_HOME
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"

exec "${HERE}/cupti_minimal_activity" "$@"

