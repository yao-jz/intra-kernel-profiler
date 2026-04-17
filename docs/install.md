# Environment Setup

This guide covers installing the **external dependencies** needed to build and
run Intra-Kernel Profiler.  The core trace library is header-only and only needs
CUDA; the NVBit/CUPTI tools have additional requirements.

## Dependency matrix

| Component | Required for | Version |
|-----------|-------------|---------|
| CUDA Toolkit | Everything | >= 11.0 (tested on 12.x) |
| C++17 compiler | Everything | GCC >= 9, Clang >= 10, or MSVC 19.14+ |
| CMake | CMake build | >= 3.20 |
| NVBit | `tools/nvbit_region_profiler` | 1.7+ (aarch64 or x86_64) |
| NSys | NSys integration (`scripts/ikp_nsys_*.py`) | 2023.4+ (bundled with CUDA) |
| Python 3 | Analysis scripts | >= 3.8 |
| NumPy + Matplotlib | Visualization scripts | any recent version |

---

## 1. CUDA Toolkit

CUPTI and `nvcc` are both included in the CUDA Toolkit. Install the version
that matches your driver.

### Option A: runfile installer (recommended for non-root)

```bash
# Example: CUDA 12.4.1
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
chmod +x cuda_12.4.1_550.54.15_linux.run
sh cuda_12.4.1_550.54.15_linux.run --silent --toolkit \
  --toolkitpath=$HOME/cuda-12.4 \
  --no-man-page
```

### Option B: package manager

```bash
# Ubuntu / Debian (example for 12.4)
sudo apt-get install cuda-toolkit-12-4
```

### Environment variables

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export CUDA_HOME=$HOME/cuda-12.4          # adjust to your install path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Verify

```bash
nvcc --version          # should print 12.x
ls $CUDA_HOME/extras/CUPTI/include/cupti.h   # CUPTI header
ls $CUDA_HOME/extras/CUPTI/lib64/libcupti.so # CUPTI library
```

> **Note:** Some minimal CUDA installs omit the `extras/CUPTI` directory.
> If it is missing, re-run the installer with the full toolkit option or
> download the
> [CUPTI samples package](https://developer.nvidia.com/cupti-ctk12_4)
> separately.

---

## 2. C++ Compiler

`nvcc` delegates host compilation to a C++ compiler.  Any compiler with C++17
support works; GCC 13 is recommended for CUDA 12.x compatibility.

### Build GCC from source (if system version is too old)

```bash
wget https://gcc.gnu.org/pub/gcc/releases/gcc-13.4.0/gcc-13.4.0.tar.xz
tar -xf gcc-13.4.0.tar.xz && cd gcc-13.4.0
./contrib/download_prerequisites
mkdir build && cd build
../configure --prefix=$HOME/gcc-13.4 --enable-languages=c,c++ --disable-multilib
make -j$(nproc) && make install

# Add to your shell profile
export PATH=$HOME/gcc-13.4/bin:$PATH
export LD_LIBRARY_PATH=$HOME/gcc-13.4/lib64:$LD_LIBRARY_PATH
```

---

## 3. CMake

CMake >= 3.20 is required.

```bash
# pip (easiest)
pip install cmake

# or download binary
wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-$(uname -m).tar.gz
tar -xf cmake-3.29.3-linux-*.tar.gz
export PATH=$PWD/cmake-3.29.3-linux-$(uname -m)/bin:$PATH
```

---

## 4. NVBit

NVBit is only needed if you want to build `tools/nvbit_region_profiler` (the
SASS-level PC-to-region attribution tool).  The core trace library and CUPTI
tools do **not** require NVBit.

### Download

NVBit releases are architecture-specific.  Pick the one for your system:

```bash
# x86_64
wget https://github.com/NVlabs/NVBit/releases/download/1.7/nvbit-Linux-x86_64-1.7.tar.bz2
tar -xjf nvbit-Linux-x86_64-1.7.tar.bz2

# aarch64 (e.g. GH200 / Grace Hopper)
wget https://github.com/NVlabs/NVBit/releases/download/1.7/nvbit-Linux-aarch64-1.7.tar.bz2
tar -xjf nvbit-Linux-aarch64-1.7.tar.bz2
```

### Verify

The extracted directory should contain `core/libnvbit.a`:

```bash
ls nvbit_release/core/libnvbit.a    # must exist
```

### Set `NVBIT_PATH`

```bash
export NVBIT_PATH=$PWD/nvbit_release
```

You will pass this to the build system:

```bash
# CMake
cmake -S . -B build -DIKP_BUILD_TOOLS=ON -DNVBIT_PATH=$NVBIT_PATH

# Make
make -C tools/nvbit_region_profiler NVBIT_PATH=$NVBIT_PATH ARCH=90a -j
```

> **Architecture note:** NVBit currently supports GPUs up to Hopper (sm_90).
> Set `ARCH=90a` for GH200.

---

## 5. CUPTI

CUPTI ships inside the CUDA Toolkit (`$CUDA_HOME/extras/CUPTI/`).  No separate
install is needed if you installed the full toolkit.

The CUPTI tool build locates the library automatically via `nvcc`:

```bash
make -C tools/cupti_region_profiler -j
```

If auto-detection fails, override explicitly:

```bash
make -C tools/cupti_region_profiler \
  CUDA_HOME=$HOME/cuda-12.4 \
  CUPTI_HOME=$HOME/cuda-12.4/extras/CUPTI \
  -j
```

### Architecture support

| Collector | sm_70-89 (Volta-Ada) | sm_90 (Hopper) |
|-----------|:---:|:---:|
| PC Sampling (`ikp_cupti_pcsamp.so`) | Yes | No |
| SASS Metrics (`ikp_cupti_sassmetrics.so`) | Yes | Yes |
| InstructionExecution (`ikp_cupti_instrexec.so`) | Yes | Yes |
| PM Sampling (`ikp_cupti_pmsamp.so`) | CUDA 12.6+ only | CUDA 12.6+ only |

> **HPC cluster note:** PC sampling requires unrestricted profiling permissions.
> On managed clusters it may return empty results.  SASS metrics generally
> works even on restricted nodes.

---

## 6. NSys (for system-level timeline merge)

NSys (NVIDIA Nsight Systems) is only needed if you want to merge system-level
profiling data (kernel launches, memory copies, NCCL communication) with IKP's
intra-kernel traces.  The core trace library, NVBit, and CUPTI tools do
**not** require NSys.

NSys ships with the CUDA Toolkit (11.1+).  Verify it is on your PATH:

```bash
nsys --version    # should print 2023.x or newer
```

If `nsys` is not found, add it:

```bash
export PATH=$CUDA_HOME/bin:$PATH
```

On some systems the `nsys` binary lives in a separate Nsight Systems install
directory (e.g., `/opt/nvidia/nsight-systems/*/target-linux-x64/`).  Locate
it and add to PATH.

### Verify

```bash
# Quick check: profile a trivial command
nsys profile --stats=true --output=/tmp/nsys_test sleep 0.1
```

If this succeeds, NSys is ready.  See [`docs/nsys_guide.md`](nsys_guide.md) for the
full integration tutorial.

---

## 7. Python (for analysis scripts)

Only needed for post-processing / visualization, not for building or running
the profiler itself.

```bash
pip install numpy matplotlib
```

Scripts that do **not** generate plots (`validate_json.py`, the `*_merge.py`
scripts, `analyze_cupti_join.py`) use only the Python standard library.

---

## Verify the full setup

```bash
# 1. Build core examples (header-only, no NVBit/CUPTI tools)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ikp_gemm_demo --m=512 --n=512 --k=512    # should produce trace JSON

# 2. Build CUPTI tools
make -C tools/cupti_region_profiler -j

# 3. Build NVBit tool (optional)
make -C tools/nvbit_region_profiler NVBIT_PATH=$NVBIT_PATH ARCH=90a -j

# 4. NSys integration (optional)
bash examples/nsys/run.sh   # builds, traces, profiles with nsys, merges, generates Explorer

# 5. Run everything end-to-end
bash scripts/run_all_examples.sh --out=_demo_out
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvcc: command not found` | CUDA not in PATH | `export PATH=$CUDA_HOME/bin:$PATH` |
| `Could not find libcupti.so*` | CUPTI not installed or wrong path | Set `CUPTI_HOME=$CUDA_HOME/extras/CUPTI` |
| `NVBIT_PATH is not set` | NVBit path not provided | `export NVBIT_PATH=/path/to/nvbit_release` |
| `core/libnvbit.a: No such file` | Wrong NVBit directory level | `NVBIT_PATH` should point to the dir that **contains** `core/` |
| `CUDA_ERROR_OPERATING_SYSTEM (304)` | No GPU access on this node | Run on a GPU node or check device permissions |
| `cmake_minimum_required ... 3.20` | CMake too old | `pip install cmake` or download a newer binary |
| PC sampling returns empty JSON | Restricted profiling permissions | Use SASS metrics instead, or request admin profiling access |
| `ptxas` version warnings during NVBit build | ptxas < 12.3 | Update CUDA Toolkit or ignore (build still succeeds with `-maxrregcount=24` fallback) |
| `nsys: command not found` | NSys not in PATH | `export PATH=$CUDA_HOME/bin:$PATH` or locate the Nsight Systems install directory |
| `nsys export` fails or hangs | Permissions or old nsys version | Verify with `nsys --version` (need 2023.4+); on HPC, check profiling permissions |
