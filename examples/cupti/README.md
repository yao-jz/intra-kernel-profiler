## CUPTI Injection Collectors

CUPTI collectors are **CUDA injection libraries** (`.so`) that attach to any CUDA binary via `CUDA_INJECTION64_PATH` and collect per-PC hardware metrics, stall-reason samples, and instruction execution counts -- no recompilation required.

For the full walkthrough (build, run, join with NVBit, generate Explorer), see [`tutorial.md`](../../tutorial.md).

### Quickstart

```bash
# 1. Build collectors
make -C ../../tools/cupti_region_profiler -j

# 2. Build target (must use -lineinfo for source mapping)
nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo minimal_cupti_target.cu -o minimal_cupti_target

# 3. Run SASS metrics (core profile)
CUDA_INJECTION64_PATH=../../tools/cupti_region_profiler/ikp_cupti_sassmetrics.so \
IKP_CUPTI_SASS_OUT=./sassmetrics_core.json \
IKP_CUPTI_SASS_PROFILE=core \
IKP_CUPTI_SASS_LAZY_PATCHING=1 \
./minimal_cupti_target --iters=20 --inner=4096
```

Output: `sassmetrics_core.json` with per-PC hardware counters.

### Notes

- **PC sampling** is not supported on Hopper (sm_90a / H100 / GH200). The JSON will contain an empty `pc_records` array. Use SASS metrics instead, or `ncu --set full` for stall-reason analysis.
- **SASS metrics** works on all architectures (Volta through Hopper).
- See `run.sh` in this directory for the full set of collectors (PC sampling, 5 SASS profiles, source mapping, instruction execution).

For interactive visualization, see [`tutorial.md`](../../tutorial.md) Step 4: Generate the Explorer.
