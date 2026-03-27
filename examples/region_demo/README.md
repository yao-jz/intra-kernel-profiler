## Region Demo (NVBit + CUPTI Join)

This example demonstrates the most powerful workflow: running **NVBit** and **CUPTI** on the same kernel and joining the results to get per-region hardware metrics. NVBit tells you *which PC belongs to which region*; CUPTI tells you *what each PC costs*. The join gives you a complete per-region performance breakdown.

For the full walkthrough, see [`tutorial.md`](../../docs/tutorial.md).

### Workflow

```bash
# 1. Build target (markers enabled for NVBit, -lineinfo for CUPTI source mapping)
nvcc -O3 -std=c++17 -arch=sm_90a -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS -I ../../include \
  minimal_region_target.cu ../../src/nvbit_marker_device.cu \
  -o minimal_region_target

# 2. NVBit pcmap — produces pc2region mapping
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=region_demo_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_TRACE_PATH=./nvbit_out \
LD_PRELOAD=../../tools/nvbit_region_profiler/region_profiler.so \
./minimal_region_target --iters=20 --inner=4096

# 3. CUPTI SASS metrics — produces per-PC hardware counters
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_sassmetrics.so \
IKP_CUPTI_SASS_OUT=./sassmetrics_core.json \
IKP_CUPTI_SASS_PROFILE=core \
IKP_CUPTI_SASS_LAZY_PATCHING=1 \
IKP_CUPTI_SASS_ENABLE_SOURCE=1 \
./minimal_region_target --iters=20 --inner=4096

# 4. Join NVBit regions with CUPTI metrics
python3 ../../scripts/analyze_cupti_join.py \
  --nvbit-dir ./nvbit_out \
  --cupti-dir . \
  --labels "0:outside,1:compute,2:store"
```

See `run.sh` in this directory for the full automated pipeline (NVBit pcmap + 4 CUPTI SASS profiles + join analysis).

For the complete GEMM pipeline, see `examples/gemm/run.sh`.
