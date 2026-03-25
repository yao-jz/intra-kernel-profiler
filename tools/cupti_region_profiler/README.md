## CUPTI Region Profiler (injection collectors, standalone)

This directory contains **CUDA injection libraries** (loaded via `CUDA_INJECTION64_PATH`) to collect per-PC / per-SASS profiling data.

`tutorial.md` provides a complete “run everything” walkthrough for this standalone repo.

### Build

```bash
make -j
```

Artifacts (`.so`):

- `ikp_cupti_pcsamp.so`: PC sampling (PC + stall reasons)
- `ikp_cupti_sassmetrics.so`: SASS metrics (per-PC metrics, optional source mapping)
- `ikp_cupti_instrexec.so`: InstructionExecution (divergence/predication)
- `ikp_cupti_pmsamp.so`: PM sampling (may be a NotSupported stub for some CUDA versions)

### Standalone conventions

- `metrics_profiles.json` lives in `config/`. `ikp_cupti_sassmetrics.so` uses `dladdr()` to locate it, checking both `<dir>/metrics_profiles.json` and `<dir>/config/metrics_profiles.json`.
- Override with `IKP_CUPTI_SASS_METRICS_JSON=/path/to/metrics_profiles.json` if needed.
- For comprehensive documentation, see [`docs/cupti_guide.md`](../../docs/cupti_guide.md).

### Run example (PC sampling)

```bash
export CUDA_INJECTION64_PATH=/path/to/ikp_cupti_pcsamp.so
export IKP_CUPTI_PCSAMP_OUT=./pcsampling_raw.json
export IKP_CUPTI_PCSAMP_COLLECTION_MODE=serialized
export IKP_CUPTI_PCSAMP_KERNEL_REGEX=myKernelNameRegex
export IKP_CUPTI_PCSAMP_PERIOD=5
export IKP_CUPTI_PCSAMP_MAX_PCS=10000
export IKP_CUPTI_PCSAMP_MAX_RECORDS=0
export IKP_CUPTI_PCSAMP_VERBOSE=3

./your_cuda_app
```

### Run example (SASS metrics)

```bash
export CUDA_INJECTION64_PATH=/path/to/ikp_cupti_sassmetrics.so
export IKP_CUPTI_SASS_OUT=./sassmetrics_raw.json
export IKP_CUPTI_SASS_PROFILE=core
export IKP_CUPTI_SASS_LAZY_PATCHING=1
export IKP_CUPTI_SASS_ENABLE_SOURCE=1

./your_cuda_app
```

> Note: clusters may enable restricted profiling, which can lead to `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`. In that case, collectors try to degrade gracefully (avoid killing the process) and report the reason in `warnings[]` in the output JSON.

