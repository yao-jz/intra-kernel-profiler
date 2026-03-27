## NVBit Region Profiler

The NVBit region profiler uses **device-call markers** compiled into the target binary to maintain a per-warp region stack at runtime, then attributes every executed PC to its enclosing region -- producing instruction counts, pipeline breakdowns, SASS listings, and memory traces per named code region.

For the full walkthrough (build, run all modes, join with CUPTI, generate Explorer), see [`tutorial.md`](../../docs/tutorial.md).

### Quickstart

```bash
# 1. Build target with NVBit markers enabled
nvcc -O3 -std=c++17 -arch=sm_90a -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS -I ../../include \
  minimal_nvbit_target.cu ../../src/nvbit_marker_device.cu \
  -o minimal_nvbit_target

# 2. Build the NVBit tool (set NVBIT_PATH to your NVBit installation)
make -C ../../tools/nvbit_region_profiler NVBIT_PATH=$NVBIT_PATH ARCH=90a -j

# 3. Run pcmap mode (PC-to-region attribution)
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=nvbit_marked_kernel \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_TRACE_PATH=./nvbit_pcmap \
LD_PRELOAD=../../tools/nvbit_region_profiler/region_profiler.so \
./minimal_nvbit_target
```

Output: `pc2region_*.json`, `region_stats_*.json`, `summary_*.txt`, and SASS listings in `nvbit_pcmap/`.

See `run.sh` in this directory for all 6 modes (pcmap, all, inst_pipe, bb_hot, nvdisasm, ptx).

For interactive visualization, see [`tutorial.md`](../../docs/tutorial.md) Step 4: Generate the Explorer.
