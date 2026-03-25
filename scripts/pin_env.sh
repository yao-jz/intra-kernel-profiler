#!/usr/bin/env bash
# scripts/pin_env.sh
# Capture and (optionally) lock down GPU/CPU environment; emit data/env.json.
# Usage:
#   scripts/pin_env.sh [--apply] [--gpu all|0,1] [--lock-gpu MIN,MAX] [--lock-mem MIN,MAX] \
#       [--power-limit W] [--persistence on|off] [--ecc on|off] [--mps on|off] \
#       [--disable-auto-boost] [--governor performance|powersave] [--numa NODE] \
#       [--warmup N] [--iters N] [--report-pcts 50,90,99]
#
# Examples (safe logging only):
#   scripts/pin_env.sh
# Lock clocks & persistence on all GPUs (requires sudo-capable session):
#   scripts/pin_env.sh --apply --lock-gpu 1590,1590 --lock-mem 2600,2600 --persistence on
# Pin one GPU, set power limit, governor=performance, record bench policy:
#   scripts/pin_env.sh --apply --gpu 0 --power-limit 400 --governor performance --warmup 10 --iters 200

set -Eeuo pipefail

umask 027
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPTS_DIR%/scripts}"
OUT_DIR="${ROOT_DIR}/data"
OUT_JSON="${OUT_DIR}/env.json"
mkdir -p "${OUT_DIR}"

log() { printf "[pin_env] %s\n" "$*"; }
die() { printf "[pin_env][ERROR] %s\n" "$*" >&2; exit 1; }

# ----------------------- Defaults & CLI -----------------------
APPLY=0
GPU_SEL="all"
LOCK_GC=""          # MIN,MAX
LOCK_MC=""          # MIN,MAX
PLIMIT=""
PERSIST=""
ECC=""
MPS=""
DISABLE_AB=0
GOV=""
NUMA=""
WARMUP="${WARMUP:-10}"        # discard first N
ITERS="${ITERS:-200}"         # >=200
REPORT_PCTS="${REPORT_PCTS:-50,90,99}"

usage() {
  sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift;;
    --gpu) GPU_SEL="$2"; shift 2;;
    --lock-gpu) LOCK_GC="$2"; APPLY=1; shift 2;;
    --lock-mem) LOCK_MC="$2"; APPLY=1; shift 2;;
    --power-limit) PLIMIT="$2"; APPLY=1; shift 2;;
    --persistence) PERSIST="$2"; APPLY=1; shift 2;;
    --ecc) ECC="$2"; APPLY=1; shift 2;;
    --mps) MPS="$2"; APPLY=1; shift 2;;
    --disable-auto-boost) DISABLE_AB=1; APPLY=1; shift;;
    --governor) GOV="$2"; APPLY=1; shift 2;;
    --numa) NUMA="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --report-pcts) REPORT_PCTS="$2"; shift 2;;
    -h|--help) usage;;
    *) die "Unknown arg: $1 (use --help)";;
esac
done

# ----------------------- Helpers -----------------------
have() { command -v "$1" >/dev/null 2>&1; }
need() { have "$1" || die "Missing dependency: $1"; }

need python3
if ! have nvidia-smi; then
  log "nvidia-smi not found; will still record CPU/OS/build info."
fi

sudo_try() {
  # Try with sudo if available; otherwise just run (may fail).
  if have sudo; then sudo bash -c "$*"; else bash -c "$*"; fi
}

parse_gpu_list() {
  if ! have nvidia-smi; then
    echo ""
    return
  fi
  if [[ "$GPU_SEL" == "all" ]]; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true
  else
    echo "$GPU_SEL" | tr ',' '\n'
  fi
}

# ----------------------- Optional Mutations -----------------------
apply_gpu_settings() {
  local g
  for g in $(parse_gpu_list); do
    [[ -z "$g" ]] && continue

    if [[ -n "$PERSIST" ]]; then
      case "$PERSIST" in
        on|On|ON|1)  log "GPU${g}: enabling persistence mode"; sudo_try "nvidia-smi -i ${g} -pm 1" || true;;
        off|Off|OFF|0) log "GPU${g}: disabling persistence mode"; sudo_try "nvidia-smi -i ${g} -pm 0" || true;;
        *) die "--persistence expects on|off";;
      esac
    fi

    if [[ -n "$PLIMIT" ]]; then
      log "GPU${g}: setting power limit to ${PLIMIT} W"
      sudo_try "nvidia-smi -i ${g} -pl ${PLIMIT}" || true
    fi

    if [[ -n "$LOCK_GC" ]]; then
      log "GPU${g}: locking GPU clocks to ${LOCK_GC}"
      # Prefer --lock-gpu-clocks; fallback to -lgc
      sudo_try "nvidia-smi -i ${g} --lock-gpu-clocks=${LOCK_GC}" || sudo_try "nvidia-smi -i ${g} -lgc ${LOCK_GC}" || true
    fi

    if [[ -n "$LOCK_MC" ]]; then
      log "GPU${g}: locking memory clocks to ${LOCK_MC}"
      # Prefer --lock-memory-clocks; some drivers support -lmc
      sudo_try "nvidia-smi -i ${g} --lock-memory-clocks=${LOCK_MC}" || sudo_try "nvidia-smi -i ${g} -lmc ${LOCK_MC}" || true
    fi

    if [[ -n "$ECC" ]]; then
      case "$ECC" in
        on|On|ON|1)  log "GPU${g}: enabling ECC (may require GPU reset)"; sudo_try "nvidia-smi -i ${g} -e 1" || true;;
        off|Off|OFF|0) log "GPU${g}: disabling ECC (may require GPU reset)"; sudo_try "nvidia-smi -i ${g} -e 0" || true;;
        *) die "--ecc expects on|off";;
      esac
    fi

    if [[ "$DISABLE_AB" -eq 1 ]]; then
      # Auto-boost is legacy and often N/A on recent datacenter GPUs; try anyway.
      log "GPU${g}: attempting to disable auto-boost (may be N/A)"
      sudo_try "nvidia-smi -i ${g} --auto-boost-default=DISABLED" || true
    fi
  done

  if [[ -n "$MPS" ]]; then
    case "$MPS" in
      on|On|ON|1)
        log "Starting NVIDIA MPS control daemon"
        export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
        export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps}"
        mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
        if pgrep -x nvidia-cuda-mps-control >/dev/null; then
          log "MPS seems already running."
        else
          # -d to daemonize; may require same user across jobs
          nvidia-cuda-mps-control -d || true
        fi
        ;;
      off|Off|OFF|0)
        if pgrep -x nvidia-cuda-mps-control >/dev/null; then
          log "Stopping NVIDIA MPS"
          echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
        else
          log "MPS not running."
        fi
        ;;
      *) die "--mps expects on|off";;
    esac
  fi

  if [[ -n "$GOV" ]]; then
    case "$GOV" in
      performance|powersave)
        log "Setting CPU governor=${GOV} for all CPUs (requires privileges)"
        for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
          [[ -e "$f" ]] && echo "$GOV" | sudo_try "tee $f >/dev/null" || true
        done
        ;;
      *) die "--governor expects performance|powersave";;
    esac
  fi
}

# ----------------------- Apply if requested -----------------------
if [[ "$APPLY" -eq 1 ]]; then
  apply_gpu_settings
fi

# ----------------------- NUMA advisory -----------------------
NUMA_BIND_CMD=""
if [[ -n "$NUMA" ]] && have numactl; then
  NUMA_BIND_CMD="numactl --cpunodebind=${NUMA} --membind=${NUMA} --localalloc"
  log "Advisory: bind your job with: ${NUMA_BIND_CMD}  (this script does not launch the job)"
fi

# ----------------------- Collect & Emit JSON via Python -----------------------
python3 - "$OUT_JSON" <<'PY' "$WARMUP" "$ITERS" "$REPORT_PCTS" "$NUMA_BIND_CMD"
import json, os, sys, subprocess, shlex, socket, datetime, re, pathlib

OUT = sys.argv[1]
WARMUP = int(sys.argv[2])
ITERS = int(sys.argv[3])
REPORT_PCTS = [int(x) for x in sys.argv[4].split(',') if x]
NUMA_BIND_CMD = sys.argv[5]

def run(cmd, timeout=10):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as e:
        return f"ERROR: {e}"

def split_csv(s):
    return [x.strip() for x in s.split(',')]

def query_nv(query):
    return run(f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits")

def query_nv_per_gpu(query):
    out = query_nv(query)
    if out.startswith("ERROR") or not out:
        return []
    return [split_csv(line) for line in out.splitlines()]

def gpu_indices():
    out = query_nv("index")
    if out.startswith("ERROR") or not out:
        return []
    return [int(x) for x in out.splitlines() if x.strip().isdigit()]

def try_nv_bool(i, block, key):
    q = run(f"nvidia-smi -i {i} -q -d {block}")
    if q.startswith("ERROR"):
        return None
    m = re.search(rf"{re.escape(key)}\s*:\s*(Enabled|Disabled)", q)
    return (m.group(1) == "Enabled") if m else None

def mig_mode(i):
    q = run(f"nvidia-smi -i {i} -q -d MIG")
    m = re.search(r"Current MIG Mode\s*:\s*(Enabled|Disabled)", q)
    return m.group(1) if m else None

def auto_boost(i):
    q = run(f"nvidia-smi -i {i} -q -d PERFORMANCE")
    m = re.search(r"Auto Boost\s*:\s*(\S+)", q)
    return m.group(1) if m else "Unknown"

def clocks_detail(i):
    q = run(f"nvidia-smi -i {i} -q -d CLOCK")
    def find(k):
        m = re.search(rf"{re.escape(k)}\s*:\s*([0-9]+)", q)
        return int(m.group(1)) if m else None
    return {
        "current_graphics_MHz": find("Graphics"),
        "current_sm_MHz": find("SM"),   # appears on some GPUs
        "current_mem_MHz": find("Memory"),
        "max_graphics_MHz": find("Max Graphics"),
        "max_mem_MHz": find("Max Memory"),
        "app_graphics_MHz": find("Applications Graphics"),
        "app_mem_MHz": find("Applications Memory"),
    }

def power_detail(i):
    q = run(f"nvidia-smi -i {i} -q -d POWER")
    def f(k):
        m = re.search(rf"{re.escape(k)}\s*:\s*([0-9]+(\.[0-9]+)?)", q)
        return float(m.group(1)) if m else None
    return {
        "draw_W": f("Power Draw"),
        "limit_W": f("Power Limit"),
        "default_limit_W": f("Default Power Limit"),
        "enforced_limit_W": f("Enforced Power Limit"),
    }

def temp_detail(i):
    q = run(f"nvidia-smi -i {i} -q -d TEMPERATURE")
    m = re.search(r"GPU Current Temp\s*:\s*([0-9]+)", q)
    return {"gpu_C": int(m.group(1)) if m else None}

def pcie_detail(i):
    q = run(f"nvidia-smi -i {i} -q -d PCI")
    def get(k):
        m = re.search(rf"{re.escape(k)}\s*:\s*([0-9x]+)", q)
        return m.group(1) if m else None
    return {
        "link_gen_current": get("Current Link Gen"),
        "link_gen_max": get("Max Link Gen"),
        "link_width_current": get("Current Link Width"),
        "link_width_max": get("Max Link Width"),
        "bus_id": (re.search(r"Bus Id\s*:\s*([0-9a-fA-F:.-]+)", q).group(1)
                   if re.search(r"Bus Id\s*:\s*([0-9a-fA-F:.-]+)", q) else None)
    }

def firmware_detail(i):
    q = run(f"nvidia-smi -i {i} -q")
    def get(k):
        m = re.search(rf"{re.escape(k)}\s*:\s*([^\n]+)", q)
        return m.group(1).strip() if m else None
    return {
        "vbios_version": get("VBIOS Version"),
        "inforom_oem": get("Inforom OEM"),
        "inforom_ecc": get("Inforom ECC"),
        "inforom_pwr": get("Inforom Power"),
    }

def mps_status():
    running = run("pgrep -x nvidia-cuda-mps-control && echo running || echo stopped")
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")
    logdir = os.environ.get("CUDA_MPS_LOG_DIRECTORY", "")
    return {"status": running.strip(), "pipe_dir": pipe, "log_dir": logdir}

def gpu_processes():
    q = run("nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv,noheader,nounits")
    procs = []
    if q and not q.startswith("ERROR"):
        for line in q.splitlines():
            pid, name, mem, uuid = [x.strip() for x in line.split(",")]
            extra = run(f"ps -o user=,etime=,pcpu=,pmem= -p {pid}")
            procs.append({"pid": int(pid), "process": name, "used_gpu_mem_MB": int(mem), "gpu_uuid": uuid, "ps": extra})
    return procs

def compute_sm_arch(cc):
    # cc like "9.0" -> "sm_90"
    if not cc: return None
    m = re.match(r"(\d+)\.(\d+)", cc)
    if not m: return None
    return f"sm_{int(m.group(1))*10 + int(m.group(2))}"

# Host / OS
now = datetime.datetime.utcnow().isoformat() + "Z"
uname = run("uname -srmo")
hostname = socket.gethostname()
osrel = run("source /etc/os-release >/dev/null 2>&1; echo ${NAME:-Unknown} ${VERSION:-}")
pythonv = run("python3 --version")
gccv = run("gcc --version | head -n1")
clangv = run("clang --version | head -n1")
cmakev = run("cmake --version | head -n1")
git_describe = run("git -C . describe --always --dirty --tags 2>/dev/null || true")

# CPU / NUMA
lscpu = run("lscpu")
cpu_governors = []
try:
    import glob
    for path in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        try:
            with open(path) as f:
                cpu_governors.append(f.read().strip())
        except:
            pass
except:
    pass
numa_hw = run("numactl -H 2>/dev/null || true")

# CUDA/Toolkit/Driver
nvcc_v = run("nvcc --version | tail -n1")
ptxas_v = run("ptxas --version 2>&1 | head -n1")
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or run("dirname $(dirname $(which nvcc)) 2>/dev/null || true")
driver_v = run("nvidia-smi --query-driver=version --format=csv,noheader 2>/dev/null || true")
cuda_drv = run("nvidia-smi --query --display=SYSTEM 2>/dev/null | grep 'Cuda Version' -m1 | awk -F: '{print $2}' | xargs || true")
ld_lib = os.environ.get("LD_LIBRARY_PATH", "")
path_env = os.environ.get("PATH", "")

# Build flags from env (if your build system exports these)
build = {
    "nvcc": nvcc_v,
    "ptxas": ptxas_v,
    "cuda_home": cuda_home,
    "driver_version": driver_v,
    "cuda_driver_version": cuda_drv,
    "NVCC_FLAGS": os.environ.get("NVCC_FLAGS", ""),
    "PTXAS_FLAGS": os.environ.get("PTXAS_FLAGS", ""),
    "CXXFLAGS": os.environ.get("CXXFLAGS", ""),
    "LDFLAGS": os.environ.get("LDFLAGS", ""),
    "OPT_LEVEL": os.environ.get("OPT_LEVEL", ""),
    "INLINE_ASM": os.environ.get("INLINE_ASM", ""),
    "LD_LIBRARY_PATH": ld_lib,
    "PATH": path_env,
    "git_describe": git_describe,
    "gcc": gccv,
    "clang": clangv,
    "cmake": cmakev,
}

# GPUs
gpus = []
idxs = gpu_indices()
if idxs:
    # bulk queries to minimize calls
    names = query_nv_per_gpu("index,gpu_name,uuid,compute_cap,driver_version,temperature.gpu")
    for row in names:
        # row: [index,name,uuid,compute_cap,driver,temperature]
        try:
            i = int(row[0])
        except:
            continue
        cc = row[3] if len(row) > 3 else ""
        info = {
            "index": i,
            "name": row[1] if len(row) > 1 else "",
            "uuid": row[2] if len(row) > 2 else "",
            "compute_cap": cc,
            "sm_arch": compute_sm_arch(cc),
            "driver_version": row[4] if len(row) > 4 else "",
            "temperature_C": int(row[5]) if len(row) > 5 and row[5].isdigit() else None,
            "modes": {
                "mig_mode": mig_mode(i),
                "ecc_mode_enabled": try_nv_bool(i, "ECC", "Current") if try_nv_bool(i,"ECC","Current") is not None else try_nv_bool(i,"ECC","ECC Mode"),  # driver-dependent key
                "persistence_mode": (run(f"nvidia-smi --query-gpu=persistence_mode --format=csv,noheader -i {i}") == "Enabled"),
                "mps": mps_status(),
                "auto_boost": auto_boost(i),
            },
            "clocks": clocks_detail(i),
            "power": power_detail(i),
            "pcie": pcie_detail(i),
            "firmware": firmware_detail(i),
        }
        gpus.append(info)

# GPU noise / background
procs = gpu_processes() if idxs else []

env = {
  "timestamp_utc": now,
  "host": {
    "hostname": hostname, "uname": uname, "os_release": osrel,
    "python": pythonv, "numa_bind_command": NUMA_BIND_CMD
  },
  "cpu": {
    "lscpu": lscpu, "governors": cpu_governors, "numa_hardware": numa_hw
  },
  "cuda_build": build,
  "gpus": gpus,
  "noise_control": {
    "gpu_processes": procs,
    "loadavg": run("cat /proc/loadavg || true"),
  },
  "benchmark_policy": {
    "warmup_discard": WARMUP, "min_iterations": ITERS,
    "report_percentiles": REPORT_PCTS, "timing_modes": ["cold","steady"], "report_variance": True
  },
  "environment": {
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES",""),
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS",""),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS",""),
    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS",""),
  }
}

# Write pretty JSON
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(env, f, indent=2, sort_keys=False)
print(json.dumps({
    "json_written": OUT,
    "gpus": [g["name"] for g in gpus],
    "policy": env["benchmark_policy"]
}, indent=2))
PY

# ----------------------- Human-readable summary -----------------------
echo
log "Summary (safe to copy into logs):"
echo "------------------------------------------------------------------"
echo "Host: $(hostname)  |  $(uname -srmo)"
if have nvidia-smi; then
  echo "Driver: $(nvidia-smi --query-driver=version --format=csv,noheader 2>/dev/null || true)  |  CUDA toolkit: $(nvcc --version 2>/dev/null | tail -n1 || echo 'N/A')"
  echo "GPUs:"
  nvidia-smi --query-gpu=index,name,uuid,compute_cap,temperature.gpu,persistence_mode --format=csv 2>/dev/null || true
  echo "Clocks (current/app/max):"
  for g in $(parse_gpu_list); do
    [[ -z "$g" ]] && continue
    C="$(nvidia-smi -i "$g" -q -d CLOCK 2>/dev/null || true)"
    CG=$(echo "$C" | awk -F: '/Graphics/ && $0 !~ /Max/ && $0 !~ /Applications/{gsub(/ /,"",$2);print $2; exit}')
    CM=$(echo "$C" | awk -F: '/Memory/ && $0 !~ /Max/ && $0 !~ /Applications/{gsub(/ /,"",$2);print $2; exit}')
    AG=$(echo "$C" | awk -F: '/Applications Graphics/{gsub(/ /,"",$2);print $2; exit}')
    AM=$(echo "$C" | awk -F: '/Applications Memory/{gsub(/ /,"",$2);print $2; exit}')
    MG=$(echo "$C" | awk -F: '/Max Graphics/{gsub(/ /,"",$2);print $2; exit}')
    MM=$(echo "$C" | awk -F: '/Max Memory/{gsub(/ /,"",$2);print $2; exit}')
    echo "  GPU${g}: cur ${CG:-?}/${CM:-?} MHz | app ${AG:-?}/${AM:-?} MHz | max ${MG:-?}/${MM:-?} MHz"
  done
  echo "Power (draw/limit):"
  nvidia-smi --query-gpu=index,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null | sed 's/^/  /' || true
  echo "Background GPU processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | sed 's/^/  /' || echo "  (none)"
else
  echo "nvidia-smi: N/A"
fi
echo "NUMA (advisory bind): ${NUMA_BIND_CMD:-(none)}"
echo "Benchmark policy: warmup=${WARMUP}, iters>=${ITERS}, percentiles=${REPORT_PCTS}"
echo "JSON: ${OUT_JSON}"
echo "------------------------------------------------------------------"
