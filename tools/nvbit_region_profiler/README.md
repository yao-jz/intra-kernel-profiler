## NVBit Region Profiler (standalone)

This NVBit-based region profiler maintains a region stack via marker device calls and writes:

- `tutorial.md` provides a complete “run everything” walkthrough for this standalone repo.

- `region_stats_*.json`: per-region instruction/memory statistics (optional mem pattern)
- `pc2region_*.json`: PC→region mapping (to join with CUPTI per-PC results)
- `mem_trace_*.jsonl`: memory trace (high overhead; can be sampled)

### Build

```bash
make NVBIT_PATH=/path/to/nvbit ARCH=90a
```

Notes:

- `NVBIT_PATH` must point to the NVBit root (contains `core/` and `core/libnvbit.a`)
- Optional: `IKP_NVBIT_USE_CUPTI_CRC=0` disables CUPTI dependency (enabled by default for cubin CRC compatibility with CUPTI tools)

### Run

1) Build the target binary and enable markers (only for NVBit runs; keep markers disabled for performance measurement):

```bash
nvcc -O3 -std=c++17 -arch=sm_90a -rdc=true -lineinfo \
  -DIKP_ENABLE_NVBIT_MARKERS \
  -I /path/to/intra_kernel_profiler/include \
  /path/to/your_app.cu /path/to/intra_kernel_profiler/src/nvbit_marker_device.cu \
  -o your_app_nvbit
```

2) Inject-run:

```bash
IKP_NVBIT_ENABLE=1 \
IKP_NVBIT_KERNEL_REGEX=your_kernel_regex \
IKP_NVBIT_MODE=pcmap \
IKP_NVBIT_TRACE_PATH=./nvbit_trace \
LD_PRELOAD=/path/to/intra_kernel_profiler/tools/nvbit_region_profiler/region_profiler.so \
./your_app_nvbit
```

Environment variables and output fields are defined by the tool source (`region_profiler.cu`). The commands above are a minimal working setup.

### Common modes

- `IKP_NVBIT_MODE=pcmap`: emit `pc2region_*.json` (recommended baseline)
- `IKP_NVBIT_MODE=instmix`: instruction + memory instruction aggregation
- `IKP_NVBIT_MODE=memtrace`: enable per-address memory trace (high overhead)
- `IKP_NVBIT_MODE=all`: enable everything

### Useful outputs (default settings)

With defaults, you will typically see:

- `pc2region_*.json`
- `region_stats_*.json`
- `summary_*.txt`
- `sass_all_*.sass`
- `sass_regions_*/region_*.sass` (requires pcmap; enabled by default)

