/*
 * Intra-Kernel Profiler NVBit region profiler (minimal).
 *
 * NVBit 1.7+ rules reminder:
 * - Do NOT call CUDA APIs inside nvbit_at_ctx_init() and nvbit_at_ctx_term()
 * - Do NOT allocate/free device/managed memory inside nvbit_at_ctx_term()
 *
 * We allocate in nvbit_tool_init() and intentionally leak at process exit.
 */

#include <stdint.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <ios>
#include <iomanip>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pthread.h>
#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

#include <cuda_runtime.h>

#if !defined(IKP_NVBIT_USE_CUPTI_CRC)
#define IKP_NVBIT_USE_CUPTI_CRC 1
#endif

#if IKP_NVBIT_USE_CUPTI_CRC
// cuptiGetCubinCrc is declared under the PC-sampling API headers (CUDA 12.4).
#include <cupti_pcsampling.h>
#endif

#include "common.h"

namespace {

enum ModeMask : uint32_t {
  kModeInstMix = 1u << 0,
  kModePcMap = 1u << 1,
};

struct SassLine {
  uint64_t pc = 0;        // function-relative SASS offset
  std::string sass;       // raw SASS text (NVBit)
  std::string op_short;   // optional, may be empty
  std::string opcode;     // full opcode, optional
  uint32_t idx = 0;       // instruction index within function (NVBit)
  bool has_pred = false;
  bool pred_neg = false;
  bool pred_uniform = false;
  int pred_num = -1;
  bool is_load = false;
  bool is_store = false;
  int access_size = 0;  // bytes (instr->getSize())
  InstrType::MemorySpace mem_space = InstrType::MemorySpace::NONE;
};

struct PcAgg {
  uint32_t function_id = 0;
  uint64_t pc = 0;
  std::vector<std::pair<uint32_t, uint32_t>> region_counts;  // sorted by region
};

struct AmbigPc {
  uint32_t function_id = 0;
  uint64_t pc = 0;
  std::vector<std::pair<uint32_t, uint32_t>> region_counts;
  uint32_t total = 0;
  uint32_t dominant_region = 0;
  uint32_t dominant_count = 0;
  double top1_frac = 0.0;
  double entropy = 0.0;
  double entropy_norm = 0.0;
};

struct KernelBuffers {
  DeviceParams* params = nullptr;  // managed
  uint32_t* d_sp = nullptr;
  uint32_t* d_stack = nullptr;
  uint32_t* d_curr = nullptr;
  uint64_t* d_marker_mismatch_count = nullptr;
  uint32_t* d_target_iter = nullptr;
  uint32_t* d_target_depth = nullptr;
  uint32_t* d_target_active = nullptr;
  uint64_t* d_counters = nullptr;
  uint64_t* d_inst_class = nullptr;
  uint64_t* d_inst_pipe = nullptr;
  uint64_t* d_bb_exec = nullptr;
  uint64_t* d_bb_hot = nullptr;
  uint64_t* d_branch_div_hist = nullptr;
  uint64_t* d_branch_active_hist = nullptr;
  uint64_t* d_branch_site_exec = nullptr;
  uint64_t* d_branch_site_taken_warp = nullptr;
  uint64_t* d_branch_site_fall_warp = nullptr;
  uint64_t* d_branch_site_taken_lanes = nullptr;
  uint64_t* d_branch_site_fall_lanes = nullptr;
  uint64_t* d_gmem_sector_hist = nullptr;
  uint64_t* d_gmem_sectors = nullptr;
  uint64_t* d_gmem_align_hist = nullptr;
  uint64_t* d_gmem_stride_hist = nullptr;
  uint64_t* d_gmem_set_hist = nullptr;
  uint64_t* d_smem_bank_hist = nullptr;
  uint64_t* d_smem_span_hist = nullptr;
  uint64_t* d_smem_broadcast = nullptr;
  uint64_t* d_gmem_line_bits = nullptr;
  PcMapEntry* d_pcmap = nullptr;
  uint32_t* d_pcmap_count = nullptr;
  uint32_t* d_mem_exec_sample_counter = nullptr;
  uint32_t* d_mem_pattern_sample_counter = nullptr;
  uint32_t* d_trace_sample_counter = nullptr;
  MemTraceEntry* d_trace = nullptr;
  uint32_t* d_trace_count = nullptr;
  uint32_t total_warps = 0;
};

struct CtxState {
  int id = 0;
  bool active = false;
  std::string kernel_name;
  CUfunction kernel_func = nullptr;
  uint64_t kernel_addr = 0;
  int kernel_nregs = 0;
  int kernel_local_size_bytes = 0;  // per-thread local memory (spill/local arrays)
  bool cubin_crc_computed = false;
  uint64_t cubin_crc = 0;
  KernelBuffers buffers{};
  // Disambiguation for pcmap joins (pcOffset is often function-relative).
  uint32_t next_function_id = 0;
  std::unordered_map<CUfunction, uint32_t> function_ids;
  std::unordered_map<uint32_t, CUfunction> function_by_id;
  std::unordered_map<uint32_t, std::string> function_names;  // demangled (NVBit)
  std::unordered_map<uint32_t, std::string> function_names_mangled;  // CUDA driver name
  // Best-effort raw SASS listing for each function_id. This is *static* SASS as seen by NVBit,
  // with function-relative offsets matching pc2region.pc_offset.
  std::unordered_map<uint32_t, std::vector<SassLine>> sass_by_function;
  // Optional source mapping derived from nvdisasm output:
  // function_name_mangled -> (pc_offset -> "path:line" string)
  std::unordered_map<std::string, std::unordered_map<uint64_t, std::string>> src_by_mangled_pc;
  // Function ids that are considered "in-scope" for the most recent kernel launch (kernel +
  // related functions). Used to keep sass_all_* focused on the current kernel.
  std::vector<uint32_t> last_launch_function_ids;
  struct BbMeta {
    uint32_t bb_id = 0;
    uint32_t function_id = 0;
    uint32_t bb_index = 0;      // within function CFG order
    uint32_t entry_pc = 0;      // first instruction offset
    uint32_t n_instrs = 0;
  };
  uint32_t next_bb_id = 0;
  std::vector<BbMeta> bb_meta;

  struct BranchSiteMeta {
    uint32_t site_id = 0;
    uint32_t function_id = 0;
    uint32_t pc_offset = 0;
    std::string opcode_short;
  };
  uint32_t next_branch_site_id = 0;
  std::vector<BranchSiteMeta> branch_sites;
};

pthread_mutex_t g_mutex;
std::unordered_map<CUcontext, CtxState*> g_ctx_state;
std::unordered_map<CUcontext, std::unordered_set<CUfunction>> g_instrumented;
static thread_local bool tls_skip_callback = false;

bool g_enabled = false;
uint32_t g_mode = kModeInstMix;
bool g_enable_inst_exec = true;
bool g_enable_mem_exec = true;
bool g_enable_trace = false;
bool g_enable_inst_class = true;
bool g_enable_inst_pipe = false;
bool g_enable_bb_count = true;
bool g_enable_bb_hot = false;
bool g_enable_branch_div = true;
bool g_enable_branch_sites = false;
bool g_enable_mem_pattern = false;
bool g_instrument_related = false;
uint32_t g_verbose = 0;
bool g_keep_cubin = false;
bool g_reweight_mem_exec = true;
bool g_reweight_mem_pattern = true;
bool g_dump_sass = true;
bool g_dump_sass_by_region = true;
bool g_dump_sass_meta = false;
bool g_dump_sass_lineinfo = false;
// Dump an additional nvdisasm listing (with file:line) into IKP_NVBIT_TRACE_PATH.
// This is a robust fallback when nvbit_get_line_info() does not return data.
bool g_dump_nvdisasm_sass = false;
bool g_dump_ptx = false;
bool g_dump_ptx_by_region = false;
bool g_dump_ptx_lineinfo = true;

uint32_t g_bb_cap = 1u << 16;
uint32_t g_branch_site_cap = 1u << 16;
uint32_t g_gmem_set_bins = 0;

uint32_t g_max_regions = 128;
uint32_t g_max_depth = 16;
uint32_t g_pcmap_cap = 1u << 20;
uint32_t g_trace_cap = 1u << 16;
uint32_t g_max_warps = 1u << 18;

uint32_t g_sample_cta = kIkpNvbitSampleAll;
uint32_t g_sample_warp = kIkpNvbitSampleAll;
uint32_t g_sample_mem_every_n = 1;
uint32_t g_target_region = kIkpNvbitAnyRegion;
uint32_t g_iter_begin = 0;
uint32_t g_iter_end = 0xFFFFFFFFu;

std::string g_kernel_regex;
std::regex g_kernel_re;
bool g_use_regex = false;

std::string g_trace_path = "./nvbit_trace";
uint32_t g_kernel_id = 0;

static uint32_t get_env_u32(const char* name, uint32_t def) {
  const char* v = std::getenv(name);
  if (!v) return def;
  return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
}

static std::string get_env_str(const char* name, const std::string& def) {
  const char* v = std::getenv(name);
  if (!v) return def;
  return std::string(v);
}

static void cuda_check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "NVBIT: %s: %s\n", msg, cudaGetErrorString(e));
    std::fflush(stderr);
    std::exit(1);
  }
}

static void ensure_dir(const std::string& path) {
  if (path.empty()) return;
  std::string cur;
  for (size_t i = 0; i < path.size(); ++i) {
    char c = path[i];
    if (c == '/') {
      if (!cur.empty()) mkdir(cur.c_str(), 0755);
    }
    cur.push_back(c);
  }
  if (!cur.empty()) mkdir(cur.c_str(), 0755);
}

static std::string sanitize_name(const std::string& name);
static std::string json_escape(const std::string& s);

static const char* mem_space_str(InstrType::MemorySpace s) {
  using MS = InstrType::MemorySpace;
  switch (s) {
    case MS::NONE: return "none";
    case MS::GENERIC: return "generic";
    case MS::GLOBAL: return "global";
    case MS::LOCAL: return "local";
    case MS::SHARED: return "shared";
    case MS::CONSTANT: return "const";
    case MS::TEXTURE: return "tex";
    case MS::SURFACE: return "surf";
    case MS::GLOBAL_TO_SHARED: return "global_to_shared";
    case MS::DISTRIBUTED_SHARED: return "distributed_shared";
    default: return "other";
  }
}

static void maybe_emit_lineinfo(std::ostream& out, CUcontext ctx, CUfunction f, const CtxState* st, uint32_t function_id,
                                uint32_t offset) {
  if (!g_dump_sass_lineinfo) return;
  if (ctx && f) {
    char* file = nullptr;
    char* dir = nullptr;
    uint32_t line = 0;
    // Best-effort; requires binary compiled with -lineinfo.
    if (nvbit_get_line_info(ctx, f, offset, &file, &dir, &line) && (file || dir)) {
      out << "// src=";
      if (dir) out << dir << "/";
      if (file) out << file;
      if (line) out << ":" << line;
      out << "\n";
      return;
    }
  }
  // Fallback: consult nvdisasm-derived mapping (if available).
  if (!st) return;
  auto mit = st->function_names_mangled.find(function_id);
  if (mit == st->function_names_mangled.end()) return;
  const std::string& mname = mit->second;
  auto fit = st->src_by_mangled_pc.find(mname);
  if (fit == st->src_by_mangled_pc.end()) return;
  auto pit = fit->second.find(static_cast<uint64_t>(offset));
  if (pit == fit->second.end()) return;
  out << "// src=" << pit->second << "\n";
}

static std::string get_env_str_raw(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

static void rm_rf(const std::string& path) {
  struct stat st{};
  if (lstat(path.c_str(), &st) != 0) return;
  if (S_ISDIR(st.st_mode)) {
    DIR* d = ::opendir(path.c_str());
    if (d) {
      while (dirent* ent = ::readdir(d)) {
        const char* name = ent->d_name;
        if (!name) continue;
        if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0) continue;
        rm_rf(path + "/" + std::string(name));
      }
      ::closedir(d);
    }
    ::rmdir(path.c_str());
    return;
  }
  std::remove(path.c_str());
}

static void parse_nvdisasm_lineinfo(CtxState* st, const std::string& nvdisasm_path) {
  if (!st) return;
  std::ifstream in(nvdisasm_path);
  if (!in) return;

  st->src_by_mangled_pc.clear();

  auto is_sep = [](char c) { return c == ' ' || c == '\t' || c == ',' || c == '"' || c == '-' || c == ':'; };

  std::string cur_func;
  std::string last_src;
  std::string line;
  while (std::getline(in, line)) {
    // Track current function by ".text.<name>" section markers.
    const size_t tpos = line.find(".text.");
    if (tpos != std::string::npos) {
      const size_t start = tpos + 6;  // after ".text."
      size_t end = start;
      while (end < line.size() && !is_sep(line[end])) end++;
      if (end > start) {
        cur_func = line.substr(start, end - start);
        last_src.clear();
      }
    }

    // Track most recent source location.
    // Example: //## File "/path/to/foo.cu", line 53
    const size_t fpos = line.find("File \"");
    if (fpos != std::string::npos) {
      const size_t p0 = fpos + 6;
      const size_t p1 = line.find('"', p0);
      const size_t lpos = line.find("line ", (p1 == std::string::npos) ? p0 : p1);
      if (p1 != std::string::npos && lpos != std::string::npos) {
        const std::string path = line.substr(p0, p1 - p0);
        const size_t n0 = lpos + 5;
        size_t n1 = n0;
        while (n1 < line.size() && std::isdigit(static_cast<unsigned char>(line[n1]))) n1++;
        if (n1 > n0) {
          const std::string ln = line.substr(n0, n1 - n0);
          last_src = path + ":" + ln;
        }
      }
    }

    // Record mapping for each instruction line.
    // Example: /*03d0*/  ...
    if (cur_func.empty() || last_src.empty()) continue;
    const size_t i0 = line.find("/*");
    const size_t i1 = (i0 == std::string::npos) ? std::string::npos : line.find("*/", i0 + 2);
    if (i0 == std::string::npos || i1 == std::string::npos || i1 <= i0 + 2) continue;
    const std::string hex = line.substr(i0 + 2, i1 - (i0 + 2));
    char* endp = nullptr;
    const uint64_t pc = std::strtoull(hex.c_str(), &endp, 16);
    if (endp == hex.c_str()) continue;
    st->src_by_mangled_pc[cur_func][pc] = last_src;
  }
}

static bool lookup_src_cached(const CtxState* st, uint32_t function_id, uint64_t pc, std::string* out_src) {
  if (!st || !out_src) return false;
  auto mit = st->function_names_mangled.find(function_id);
  if (mit == st->function_names_mangled.end()) return false;
  const std::string& mname = mit->second;
  auto fit = st->src_by_mangled_pc.find(mname);
  if (fit == st->src_by_mangled_pc.end()) return false;
  auto pit = fit->second.find(pc);
  if (pit == fit->second.end()) return false;
  *out_src = pit->second;
  return true;
}

struct PtxModule {
  std::string name;  // extracted filename

  // file_id -> path
  std::unordered_map<int, std::string> files;
  struct Instr {
    std::string func;
    std::string src;   // "path:line" if known
    std::string text;  // the PTX line
    size_t line_idx = 0;
  };
  std::vector<Instr> instrs;
  std::vector<std::string> raw_lines;
};

static bool is_system_header_path(const std::string& path) {
  // Heuristic: SASS lineinfo tends to attribute inlined libdevice/CUDA header code
  // back to the callsite in user code, while cuobjdump's PTX `.loc` often points
  // at the deepest inlined header. Prefer "user-ish" file paths when available.
  //
  // This is intentionally conservative: only mark clearly system-ish paths as system.
  return (path.find("/cuda-") != std::string::npos) ||
         (path.find("/targets/") != std::string::npos) ||
         (path.find("/usr/") != std::string::npos);
}

static void parse_ptx_module(PtxModule* m) {
  if (!m) return;
  m->files.clear();
  m->instrs.clear();

  // Pass 1: gather .file table.
  for (size_t li = 0; li < m->raw_lines.size(); ++li) {
    const auto& line = m->raw_lines[li];
    // Accept both: ".file\t1 \"/path\"" and ".file 1 \"path\""
    const size_t p = line.find(".file");
    if (p == std::string::npos) continue;
    // Require it to start with ".file" ignoring leading spaces/tabs.
    size_t s = 0;
    while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) s++;
    if (s != p) continue;

    // Parse: .file <id> "<path>"
    size_t i = p + 5;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
    char* endp = nullptr;
    const int id = static_cast<int>(std::strtol(line.c_str() + i, &endp, 10));
    if (!endp || endp == (line.c_str() + i)) continue;
    const char* q0 = std::strchr(endp, '"');
    if (!q0) continue;
    const char* q1 = std::strchr(q0 + 1, '"');
    if (!q1) continue;
    m->files[id] = std::string(q0 + 1, q1 - (q0 + 1));
  }

  // Pass 2: track current func + current .loc, collect instruction lines.
  std::string cur_func;
  int cur_file_id = -1;
  int cur_line = 0;
  int primary_file_id = -1;
  int primary_line = 0;
  int brace_depth = 0;  // function-scope brace depth (cuobjdump PTX may contain nested "{ }")
  for (size_t li = 0; li < m->raw_lines.size(); ++li) {
    const auto& line = m->raw_lines[li];
    // Track function start: ".entry <name>(" or ".func <name>(".
    // NOTE: cuobjdump often prefixes qualifiers like ".visible", e.g. ".visible .entry foo(".
    {
      size_t s = 0;
      while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) s++;
      // Scan tokens for ".entry"/".func" instead of requiring it to be the first token.
      size_t pos = s;
      bool is_entry = false;
      bool is_func = false;
      while (pos < line.size()) {
        if (line.compare(pos, 6, ".entry") == 0) {
          is_entry = true;
          break;
        }
        if (line.compare(pos, 5, ".func") == 0) {
          is_func = true;
          break;
        }
        size_t ws = line.find_first_of(" \t", pos);
        if (ws == std::string::npos) break;
        pos = line.find_first_not_of(" \t", ws);
        if (pos == std::string::npos) break;
      }
      if (is_entry || is_func) {
        size_t i = pos + (is_entry ? 6 : 5);
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
        size_t j = i;
        while (j < line.size() && line[j] != '(' && line[j] != ' ' && line[j] != '\t') j++;
        if (j > i) {
          cur_func = line.substr(i, j - i);
          brace_depth = 0;
        }
      }
    }

    // Track brace depth within current function. cuobjdump PTX uses extra "{ }" blocks
    // around call sequences, so a simple boolean would incorrectly end the function early.
    if (!cur_func.empty()) {
      for (char ch : line) {
        if (ch == '{') {
          brace_depth++;
        } else if (ch == '}') {
          brace_depth--;
          if (brace_depth < 0) brace_depth = 0;
        }
      }
    }
    const bool in_func_body = (!cur_func.empty() && brace_depth > 0);

    // Track .loc: ".loc <file_id> <line> <col>"
    {
      size_t s = 0;
      while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) s++;
      if (line.compare(s, 4, ".loc") == 0) {
        size_t i = s + 4;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
        char* endp = nullptr;
        int fid = static_cast<int>(std::strtol(line.c_str() + i, &endp, 10));
        if (endp && endp != (line.c_str() + i)) {
          i = static_cast<size_t>(endp - line.c_str());
          while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
          endp = nullptr;
          int ln = static_cast<int>(std::strtol(line.c_str() + i, &endp, 10));
          if (endp && endp != (line.c_str() + i)) {
            cur_file_id = fid;
            cur_line = ln;
            // Update primary (callsite-ish) location when the file looks like user code.
            auto fit = m->files.find(fid);
            if (fit != m->files.end() && ln > 0 && !is_system_header_path(fit->second)) {
              primary_file_id = fid;
              primary_line = ln;
            }
          }
        }
      }
    }

    // Instruction heuristic: in function body, non-empty, not directive/comment/label/brace.
    if (!in_func_body) continue;
    size_t s = 0;
    while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) s++;
    if (s >= line.size()) continue;
    if (line.compare(s, 2, "//") == 0) continue;
    if (line[s] == '.' || line[s] == '{' || line[s] == '}') continue;
    if (line[s] == '(' || line[s] == ')') continue;
    if (line.compare(s, 2, ");") == 0) continue;
    // Label like "$L__BB0_2:" or "L0:".
    // IMPORTANT: PTX opcodes can contain "::" (e.g. cp.async.bulk.shared::cluster...),
    // so we must not treat any ':' before whitespace as a label.
    {
      size_t ws = line.find_first_of(" \t", s);
      const size_t token_end = (ws == std::string::npos) ? line.size() : ws;
      if (token_end > s) {
        const std::string token = line.substr(s, token_end - s);
        if (!token.empty() && token.back() == ':') {
          // Labels are a single token ending with ':' and should not contain '.' or "::".
          if (token.find("::") == std::string::npos && token.find('.') == std::string::npos) {
            continue;
          }
        }
      }
    }

    PtxModule::Instr rec;
    rec.func = cur_func;
    rec.text = line;
    rec.line_idx = li;
    rec.src.clear();
    // Prefer primary (callsite-ish) location when available; it matches SASS lineinfo better
    // for heavily inlined CUDA headers.
    auto pick_src = [&](int fid, int ln) -> std::string {
      auto fit = m->files.find(fid);
      if (fit == m->files.end() || ln <= 0) return {};
      return fit->second + ":" + std::to_string(ln);
    };
    if (primary_file_id >= 0 && primary_line > 0) {
      rec.src = pick_src(primary_file_id, primary_line);
    }
    if (rec.src.empty()) {
      rec.src = pick_src(cur_file_id, cur_line);
    }
    m->instrs.emplace_back(std::move(rec));
  }
}

static bool extract_ptx_from_exe(const std::string& exe_path, const std::string& err_path_abs,
                                std::vector<PtxModule>* modules_out) {
  if (!modules_out) return false;
  modules_out->clear();

  std::string cuobjdump = get_env_str_raw("CUOBJDUMP");
  if (cuobjdump.empty()) cuobjdump = "cuobjdump";

  char tmp_template[] = "/tmp/ikp_nvbit_ptx_XXXXXX";
  char* tmp_dir_c = ::mkdtemp(tmp_template);
  if (!tmp_dir_c) return false;
  const std::string tmp_dir(tmp_dir_c);

  // NOTE: use -all because PTX is often stored in non-executable fatbin sections.
  const std::string cmd_extract =
      "cd \"" + tmp_dir + "\" && " + cuobjdump + " -all -xptx all \"" + exe_path + "\""
      " > /dev/null 2>> \"" + err_path_abs + "\"";
  (void)std::system(cmd_extract.c_str());

  // Enumerate extracted .ptx files.
  if (DIR* d = ::opendir(tmp_dir.c_str())) {
    while (dirent* ent = ::readdir(d)) {
      const char* name = ent->d_name;
      if (!name) continue;
      const std::string s(name);
      if (s.size() >= 4 && s.rfind(".ptx") == (s.size() - 4)) {
        PtxModule m;
        m.name = s;
        const std::string path = tmp_dir + "/" + s;
        std::ifstream in(path);
        if (!in) continue;
        std::string line;
        while (std::getline(in, line)) {
          m.raw_lines.emplace_back(std::move(line));
        }
        parse_ptx_module(&m);
        modules_out->emplace_back(std::move(m));
      }
    }
    ::closedir(d);
  }
  std::sort(modules_out->begin(), modules_out->end(), [](const PtxModule& a, const PtxModule& b) { return a.name < b.name; });

  rm_rf(tmp_dir);
  return !modules_out->empty();
}

static std::pair<std::string, std::string> write_ptx_outputs(const CtxState* st, const std::vector<PcAgg>& pc_aggs,
                                                             const std::vector<PtxModule>& modules) {
  // returns {ptx_all_path, ptx_regions_dir} (may be empty)
  if (!st) return {};
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string ptx_all_path = g_trace_path + "/ptx_all_" + safe + "_" + std::to_string(g_kernel_id) + ".ptx";
  const std::string ptx_regions_dir = g_trace_path + "/ptx_regions_" + safe + "_" + std::to_string(g_kernel_id);

  if (g_dump_ptx) {
    std::ofstream out(ptx_all_path);
    out << "// kernel: " << st->kernel_name << "\n";
    out << "// kernel_id: " << g_kernel_id << "\n";
    out << "// NOTE: PTX extracted from host executable (cuobjdump -all -xptx all)\n\n";
    for (const auto& m : modules) {
      out << "\n// ===== ptx: " << m.name << " =====\n";
      // Emit raw PTX as-is (it already contains .loc/.file), and optionally
      // inject an extra comment before each instruction line we recognized.
      std::unordered_map<size_t, std::string> line_to_src;
      if (g_dump_ptx_lineinfo) {
        line_to_src.reserve(m.instrs.size());
        for (const auto& ins : m.instrs) {
          if (!ins.src.empty()) line_to_src.emplace(ins.line_idx, ins.src);
        }
      }
      for (size_t li = 0; li < m.raw_lines.size(); ++li) {
        const auto& l = m.raw_lines[li];
        if (g_dump_ptx_lineinfo) {
          auto it = line_to_src.find(li);
          if (it != line_to_src.end()) {
            out << "// src=" << it->second << "\n";
          }
        }
        out << l << "\n";
      }
    }
  }

  if (g_dump_ptx_by_region) {
    if (pc_aggs.empty()) return {g_dump_ptx ? ptx_all_path : std::string(), std::string()};
    ensure_dir(ptx_regions_dir);

    // Build region -> src(line) set from cached SASS src mapping.
    std::unordered_map<uint32_t, std::unordered_set<std::string>> region_srcs;
    region_srcs.reserve(64);
    for (const auto& agg : pc_aggs) {
      std::string src;
      if (!lookup_src_cached(st, agg.function_id, agg.pc, &src)) continue;
      for (const auto& rc : agg.region_counts) {
        region_srcs[rc.first].insert(src);
      }
    }

    for (const auto& kv : region_srcs) {
      const uint32_t rid = kv.first;
      const auto& srcset = kv.second;
      const std::string path = ptx_regions_dir + "/region_" + std::to_string(rid) + ".ptx";
      std::ofstream out(path);
      out << "// kernel: " << st->kernel_name << "\n";
      out << "// kernel_id: " << g_kernel_id << "\n";
      out << "// region: " << rid << "\n";
      out << "// NOTE: PTX slice filtered by src line mapping from SASS.\n";
      out << "// If empty, ensure IKP_NVBIT_DUMP_NVDISASM_SASS=1 and IKP_NVBIT_DUMP_SASS_LINEINFO=1.\n\n";

      for (const auto& m : modules) {
        out << "\n// ===== ptx: " << m.name << " =====\n";
        std::string last_func;
        for (const auto& ins : m.instrs) {
          if (ins.src.empty()) continue;
          if (srcset.find(ins.src) == srcset.end()) continue;
          if (ins.func != last_func) {
            out << "\n// ---- function " << ins.func << " ----\n";
            last_func = ins.func;
          }
          out << "// src=" << ins.src << "\n";
          out << ins.text << "\n";
        }
      }
    }
  }

  return {g_dump_ptx ? ptx_all_path : std::string(), g_dump_ptx_by_region ? ptx_regions_dir : std::string()};
}

static std::string dump_nvdisasm_sass(CUcontext ctx, CtxState* st) {
  if (!g_dump_nvdisasm_sass) return {};
  if (!st || !ctx || !st->kernel_func) return {};

  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string out_path_rel = g_trace_path + "/nvdisasm_all_" + safe + "_" + std::to_string(g_kernel_id) + ".sass";
  const std::string err_path_rel = g_trace_path + "/nvdisasm_all_" + safe + "_" + std::to_string(g_kernel_id) + ".err";

  // Build absolute paths for shell redirections (commands may `cd` into temp dirs).
  char cwd_buf[PATH_MAX];
  std::memset(cwd_buf, 0, sizeof(cwd_buf));
  const char* cwd = ::getcwd(cwd_buf, sizeof(cwd_buf) - 1);
  const std::string trace_abs =
      (!g_trace_path.empty() && g_trace_path[0] == '/') ? g_trace_path
                                                        : (cwd ? (std::string(cwd) + "/" + g_trace_path) : g_trace_path);
  const std::string out_path_abs = trace_abs + "/nvdisasm_all_" + safe + "_" + std::to_string(g_kernel_id) + ".sass";
  const std::string err_path_abs = trace_abs + "/nvdisasm_all_" + safe + "_" + std::to_string(g_kernel_id) + ".err";

  // nvbit_dump_cubin() is not always available/reliable across NVBit/CUDA builds.
  // Instead, extract embedded cubin(s) from the host executable via cuobjdump.
  char exe_buf[PATH_MAX];
  std::memset(exe_buf, 0, sizeof(exe_buf));
  const ssize_t n = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (n <= 0) return {};
  const std::string exe_path(exe_buf);

  // Prefer NVBit's own env var; otherwise rely on PATH.
  std::string nvdisasm = get_env_str_raw("NVDISASM");
  if (nvdisasm.empty()) nvdisasm = "nvdisasm";
  std::string cuobjdump = get_env_str_raw("CUOBJDUMP");
  if (cuobjdump.empty()) cuobjdump = "cuobjdump";

  // 1) Extract ELF cubin(s) into a temp directory (do not clutter IKP_NVBIT_TRACE_PATH).
  char tmp_template[] = "/tmp/ikp_nvbit_nvdisasm_XXXXXX";
  char* tmp_dir_c = ::mkdtemp(tmp_template);
  if (!tmp_dir_c) return {};
  const std::string tmp_dir(tmp_dir_c);

  const std::string cmd_extract =
      "cd \"" + tmp_dir + "\" && " + cuobjdump + " -xelf all \"" + exe_path + "\""
      " > /dev/null 2>> \"" + err_path_abs + "\"";
  (void)std::system(cmd_extract.c_str());

  // 2) Enumerate extracted cubins.
  std::vector<std::string> cubins;
  if (DIR* d = ::opendir(tmp_dir.c_str())) {
    while (dirent* ent = ::readdir(d)) {
      const char* name = ent->d_name;
      if (!name) continue;
      const std::string s(name);
      if (s.size() >= 6 && s.rfind(".cubin") == (s.size() - 6)) {
        cubins.emplace_back(tmp_dir + "/" + s);
      }
    }
    ::closedir(d);
  }
  if (cubins.empty()) return {};
  std::sort(cubins.begin(), cubins.end());

  // 3) nvdisasm each cubin and concatenate into one file.
  // nvdisasm's `--print-line-info-inline` shows inlining info, which is useful for templates.
  // We also keep stderr in a sidecar file for debugging.
  {
    std::ofstream out(out_path_abs, std::ios::out | std::ios::trunc);
    out << "// exe: " << exe_path << "\n";
    out << "// extracted_cubins: " << cubins.size() << "\n\n";
  }
  for (const auto& c : cubins) {
    const std::string cmd =
        "echo \"\\n// ===== cubin: " + c + " =====\\n\" >> \"" + out_path_abs + "\" && " +
        nvdisasm + " --print-code --print-line-info-inline \"" + c + "\" >> \"" + out_path_abs + "\""
        " 2>> \"" + err_path_abs + "\"";
    (void)std::system(cmd.c_str());
  }

  // Build (func,pc)->src mapping for later emission into sass_all/region_*.sass.
  parse_nvdisasm_lineinfo(st, out_path_abs);

  // Cleanup temp files. Keep err file only if non-empty.
  rm_rf(tmp_dir);
  struct stat est{};
  if (stat(err_path_abs.c_str(), &est) == 0 && est.st_size == 0) {
    std::remove(err_path_abs.c_str());
  }
  return out_path_rel;
}

static void maybe_cache_sass_listing(CtxState* st, uint32_t func_id, const std::vector<Instr*>& instrs) {
  if (!st) return;
  auto it = st->sass_by_function.find(func_id);
  if (it != st->sass_by_function.end() && !it->second.empty()) return;

  std::vector<SassLine> lines;
  lines.reserve(instrs.size());
  for (auto* instr : instrs) {
    if (!instr) continue;
    SassLine l;
    l.pc = instr->getOffset();
    const char* s = instr->getSass();
    if (s) l.sass = s;
    const char* op = instr->getOpcodeShort();
    if (op) l.op_short = op;
    const char* opcode = instr->getOpcode();
    if (opcode) l.opcode = opcode;
    l.idx = instr->getIdx();
    l.has_pred = instr->hasPred();
    if (l.has_pred) {
      l.pred_num = instr->getPredNum();
      l.pred_neg = instr->isPredNeg();
      l.pred_uniform = instr->isPredUniform();
    }
    l.is_load = instr->isLoad();
    l.is_store = instr->isStore();
    l.access_size = instr->getSize();
    l.mem_space = instr->getMemorySpace();
    lines.emplace_back(std::move(l));
  }
  std::sort(lines.begin(), lines.end(), [](const SassLine& a, const SassLine& b) { return a.pc < b.pc; });
  st->sass_by_function[func_id] = std::move(lines);
}

static uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
  static uint32_t table[256];
  static bool table_inited = false;
  if (!table_inited) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      table[i] = c;
    }
    table_inited = true;
  }

  uint32_t c = crc;
  for (size_t i = 0; i < len; ++i) {
    c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
  }
  return c;
}

static uint32_t crc32_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return 0;

  uint32_t crc = 0xFFFFFFFFu;
  std::vector<uint8_t> buf(1u << 20);
  while (in) {
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    std::streamsize n = in.gcount();
    if (n > 0) {
      crc = crc32_update(crc, buf.data(), static_cast<size_t>(n));
    }
  }
  return crc ^ 0xFFFFFFFFu;
}

static void maybe_compute_cubin_crc(CUcontext ctx, CtxState* st) {
  if (!st) return;
  if (st->cubin_crc_computed) return;
  st->cubin_crc_computed = true;
  st->cubin_crc = 0;

  if (!(g_mode & kModePcMap)) return;
  if (!st->kernel_func) return;

  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string cubin_path = g_trace_path + "/cubin_" + safe + "_" + std::to_string(g_kernel_id) + ".cubin";

  if (!nvbit_dump_cubin(ctx, st->kernel_func, cubin_path.c_str())) {
    return;
  }

#if IKP_NVBIT_USE_CUPTI_CRC
  // Prefer CUPTI's cubin CRC to match CUPTI collectors exactly.
  {
    std::ifstream in(cubin_path, std::ios::binary | std::ios::ate);
    if (in) {
      const std::streamsize n = in.tellg();
      if (n > 0) {
        in.seekg(0, std::ios::beg);
        std::vector<uint8_t> buf(static_cast<size_t>(n));
        if (in.read(reinterpret_cast<char*>(buf.data()), n)) {
          CUpti_GetCubinCrcParams p{};
          p.size = sizeof(CUpti_GetCubinCrcParams);
          p.cubin = buf.data();
          p.cubinSize = static_cast<size_t>(n);
          if (cuptiGetCubinCrc(&p) == CUPTI_SUCCESS) {
            st->cubin_crc = p.cubinCrc;
          }
        }
      }
    }
  }
#endif
  // Fallback: CRC32(IEEE) of the dumped cubin file.
  if (st->cubin_crc == 0) {
    st->cubin_crc = static_cast<uint64_t>(crc32_file(cubin_path));
  }
  if (!g_keep_cubin) {
    std::remove(cubin_path.c_str());
  }
}

static std::string sanitize_name(const std::string& name) {
  std::string out = name;
  for (char& c : out) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) c = '_';
  }
  return out;
}

static std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '\"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          static constexpr char kHex[] = "0123456789abcdef";
          out += "\\u00";
          out.push_back(kHex[(c >> 4) & 0xF]);
          out.push_back(kHex[c & 0xF]);
        } else {
          out.push_back(static_cast<char>(c));
        }
        break;
    }
  }
  return out;
}

static bool match_kernel(const char* name) {
  if (!g_use_regex) return true;
  return std::regex_search(name, g_kernel_re);
}

static bool parse_marker(const char* name, uint32_t* region_id, bool* is_push) {
  const char* push_prefix = "ikp_nvbit_region_push_";
  const char* pop_prefix = "ikp_nvbit_region_pop_";

  const char* pos = std::strstr(name, push_prefix);
  if (pos) {
    pos += std::strlen(push_prefix);
    *is_push = true;
    *region_id = static_cast<uint32_t>(std::strtoul(pos, nullptr, 10));
    return true;
  }
  pos = std::strstr(name, pop_prefix);
  if (pos) {
    pos += std::strlen(pop_prefix);
    *is_push = false;
    *region_id = static_cast<uint32_t>(std::strtoul(pos, nullptr, 10));
    return true;
  }
  return false;
}

static bool parse_marker_in_sass(const char* sass, uint32_t* region_id, bool* is_push) {
  if (!sass) return false;
  const char* push_key = "ikp_nvbit_region_push_";
  const char* pop_key = "ikp_nvbit_region_pop_";
  const char* pos = std::strstr(sass, push_key);
  if (pos) {
    pos += std::strlen(push_key);
    *is_push = true;
    *region_id = static_cast<uint32_t>(std::strtoul(pos, nullptr, 10));
    return true;
  }
  pos = std::strstr(sass, pop_key);
  if (pos) {
    pos += std::strlen(pop_key);
    *is_push = false;
    *region_id = static_cast<uint32_t>(std::strtoul(pos, nullptr, 10));
    return true;
  }
  return false;
}

static bool op_has(const char* s, const char* token) { return s && token && std::strstr(s, token); }

static bool op_eq(const char* s, const char* token) { return s && token && std::strcmp(s, token) == 0; }

static bool is_branch_opcode(const char* op_short) {
  return op_eq(op_short, "BRA") || op_eq(op_short, "BRX") || op_eq(op_short, "JMP") || op_eq(op_short, "JMX") ||
         op_eq(op_short, "SSY") || op_eq(op_short, "PBK") || op_eq(op_short, "PCNT");
}

static uint32_t classify_instr(Instr* instr) {
  const char* op = instr->getOpcode();
  const char* op_short = instr->getOpcodeShort();
  const char* sass = instr->getSass();
  const auto mem_type = instr->getMemorySpace();

  if (op_has(op, "WGMMA") || op_has(op, "HMMA") || op_has(op, "MMA") || op_has(sass, "WGMMA") || op_has(sass, "HMMA")) {
    return kInstClassTensorWgmma;
  }
  if (op_has(op, "CP.ASYNC") || op_has(op, "CPASYNC") || op_has(op, "LDGSTS") || op_has(op, "TMA") ||
      op_has(sass, "CP.ASYNC") || op_has(sass, "TMA")) {
    return kInstClassSpecial;
  }
  if (op_eq(op_short, "CALL")) return kInstClassCall;
  if (op_eq(op_short, "RET") || op_eq(op_short, "EXIT")) return kInstClassRet;
  if (is_branch_opcode(op_short)) return kInstClassBranch;
  if (op_has(op_short, "BAR") || op_has(op, "BAR")) return kInstClassBarrier;
  if (op_has(op, "MEMBAR")) return kInstClassMembar;

  if (instr->isLoad() || instr->isStore()) {
    if (mem_type == InstrType::MemorySpace::SHARED || mem_type == InstrType::MemorySpace::DISTRIBUTED_SHARED) {
      return instr->isLoad() ? kInstClassLdShared : kInstClassStShared;
    }
    if (mem_type == InstrType::MemorySpace::LOCAL) {
      return instr->isLoad() ? kInstClassLdLocal : kInstClassStLocal;
    }
    return instr->isLoad() ? kInstClassLdGlobal : kInstClassStGlobal;
  }

  if (op_has(op, ".F32") || op_has(op, "FADD") || op_has(op, "FFMA") || op_has(op, "FMUL") || op_has(op, "FMA") ||
      op_has(op, "FSET") || op_has(op, "FSETP")) {
    return kInstClassAluFp32;
  }
  if (op_has(op, "IADD") || op_has(op, "IMAD") || op_has(op, "IMUL") || op_has(op, "ISCADD") || op_has(op, "ISETP") ||
      op_has(op, "LOP") || op_has(op, "SHF") || op_has(op, "SEL")) {
    return kInstClassAluInt;
  }
  return kInstClassOther;
}

static uint32_t classify_pipe(Instr* instr) {
  const char* op = instr->getOpcode();
  const char* op_short = instr->getOpcodeShort();
  const char* sass = instr->getSass();
  const auto mem_type = instr->getMemorySpace();
  (void)mem_type;

  if (instr->isLoad()) return kInstPipeLd;
  if (instr->isStore()) return kInstPipeSt;

  if (op_has(op, "TEX") || op_has(op_short, "TEX") || op_has(op, "TLD") || op_has(op, "TLD4")) {
    return kInstPipeTex;
  }
  if (op_eq(op_short, "CALL") || op_eq(op_short, "RET") || op_eq(op_short, "EXIT")) {
    return kInstPipeCallRet;
  }
  if (is_branch_opcode(op_short)) return kInstPipeBranch;
  if (op_has(op_short, "BAR") || op_has(op, "BAR") || op_has(op, "MBAR")) return kInstPipeBarrier;
  if (op_has(op, "MEMBAR")) return kInstPipeMembar;
  if (op_has(op, "WGMMA") || op_has(op, "HMMA") || op_has(op, "MMA") || op_has(sass, "WGMMA") || op_has(sass, "HMMA")) {
    return kInstPipeTensor;
  }
  if (op_has(op, "CP.ASYNC") || op_has(op, "CPASYNC") || op_has(op, "LDGSTS") || op_has(op, "TMA") ||
      op_has(sass, "CP.ASYNC") || op_has(sass, "TMA")) {
    return kInstPipeSpecial;
  }
  if (op_has(op, "ULDC") || op_has(op, "LDC")) return kInstPipeUniform;
  if (op_has(op, "MUFU") || op_has(op, "RCP") || op_has(op, "RSQ") || op_has(op, "EX2") || op_has(op, "LG2") ||
      op_has(op, "SIN") || op_has(op, "COS")) {
    return kInstPipeSfu;
  }
  if (op_has(op, ".F64") || op_has(op, "DADD") || op_has(op, "DMUL") || op_has(op, "DFMA")) {
    return kInstPipeFp64;
  }
  if (op_has(op, ".F16") || op_has(op, "HFMA") || op_has(op, "HADD") || op_has(op, "HMUL")) {
    return kInstPipeFp16;
  }
  if (op_has(op, ".F32") || op_has(op, "FADD") || op_has(op, "FFMA") || op_has(op, "FMUL") || op_has(op, "FMA") ||
      op_has(op, "FSET") || op_has(op, "FSETP")) {
    return kInstPipeFp32;
  }
  if (op_has(op, "IADD") || op_has(op, "IMAD") || op_has(op, "IMUL") || op_has(op, "ISCADD") || op_has(op, "ISETP") ||
      op_has(op, "LOP") || op_has(op, "SHF") || op_has(op, "SEL")) {
    return kInstPipeInt;
  }
  return kInstPipeOther;
}

static void init_context_buffers(KernelBuffers* kb) {
  if (kb->params) return;

  const uint32_t total_warps = g_max_warps;
  kb->total_warps = total_warps;

  cuda_check(cudaMalloc(&kb->d_sp, sizeof(uint32_t) * total_warps), "alloc d_sp");
  cuda_check(cudaMalloc(&kb->d_stack, sizeof(uint32_t) * total_warps * g_max_depth), "alloc d_stack");
  cuda_check(cudaMalloc(&kb->d_curr, sizeof(uint32_t) * total_warps), "alloc d_curr");
  cuda_check(cudaMalloc(&kb->d_marker_mismatch_count, sizeof(uint64_t)), "alloc d_marker_mismatch_count");
  cuda_check(cudaMalloc(&kb->d_target_iter, sizeof(uint32_t) * total_warps), "alloc d_target_iter");
  cuda_check(cudaMalloc(&kb->d_target_depth, sizeof(uint32_t) * total_warps), "alloc d_target_depth");
  cuda_check(cudaMalloc(&kb->d_target_active, sizeof(uint32_t) * total_warps), "alloc d_target_active");
  cuda_check(cudaMalloc(&kb->d_counters, sizeof(uint64_t) * g_max_regions * kCounterCount), "alloc d_counters");
  cuda_check(cudaMalloc(&kb->d_inst_class, sizeof(uint64_t) * g_max_regions * kInstClassCount), "alloc d_inst_class");
  cuda_check(cudaMalloc(&kb->d_inst_pipe, sizeof(uint64_t) * g_max_regions * kInstPipeCount), "alloc d_inst_pipe");
  cuda_check(cudaMalloc(&kb->d_bb_exec, sizeof(uint64_t) * g_max_regions), "alloc d_bb_exec");
  if (g_enable_bb_hot && g_bb_cap > 0) {
    cuda_check(cudaMalloc(&kb->d_bb_hot, sizeof(uint64_t) * g_bb_cap), "alloc d_bb_hot");
  }
  cuda_check(cudaMalloc(&kb->d_branch_div_hist, sizeof(uint64_t) * g_max_regions * kBranchDivHistBins), "alloc d_branch_div_hist");
  cuda_check(cudaMalloc(&kb->d_branch_active_hist, sizeof(uint64_t) * g_max_regions * kBranchActiveHistBins), "alloc d_branch_active_hist");
  if (g_enable_branch_sites && g_branch_site_cap > 0) {
    cuda_check(cudaMalloc(&kb->d_branch_site_exec, sizeof(uint64_t) * g_branch_site_cap), "alloc d_branch_site_exec");
    cuda_check(cudaMalloc(&kb->d_branch_site_taken_warp, sizeof(uint64_t) * g_branch_site_cap), "alloc d_branch_site_taken_warp");
    cuda_check(cudaMalloc(&kb->d_branch_site_fall_warp, sizeof(uint64_t) * g_branch_site_cap), "alloc d_branch_site_fall_warp");
    cuda_check(cudaMalloc(&kb->d_branch_site_taken_lanes, sizeof(uint64_t) * g_branch_site_cap), "alloc d_branch_site_taken_lanes");
    cuda_check(cudaMalloc(&kb->d_branch_site_fall_lanes, sizeof(uint64_t) * g_branch_site_cap), "alloc d_branch_site_fall_lanes");
  }
  cuda_check(cudaMalloc(&kb->d_gmem_sector_hist, sizeof(uint64_t) * g_max_regions * kGmemSectorHistBins), "alloc d_gmem_sector_hist");
  cuda_check(cudaMalloc(&kb->d_gmem_sectors, sizeof(uint64_t) * g_max_regions), "alloc d_gmem_sectors");
  cuda_check(cudaMalloc(&kb->d_gmem_align_hist, sizeof(uint64_t) * g_max_regions * kGmemAlignHistBins), "alloc d_gmem_align_hist");
  cuda_check(cudaMalloc(&kb->d_gmem_stride_hist, sizeof(uint64_t) * g_max_regions * kStrideClassBins), "alloc d_gmem_stride_hist");
  if (g_enable_mem_pattern && g_gmem_set_bins > 0) {
    cuda_check(cudaMalloc(&kb->d_gmem_set_hist, sizeof(uint64_t) * g_max_regions * g_gmem_set_bins), "alloc d_gmem_set_hist");
  }
  cuda_check(cudaMalloc(&kb->d_smem_bank_hist, sizeof(uint64_t) * g_max_regions * kSmemBankHistBins), "alloc d_smem_bank_hist");
  cuda_check(cudaMalloc(&kb->d_smem_span_hist, sizeof(uint64_t) * g_max_regions * kSmemSpanHistBins), "alloc d_smem_span_hist");
  cuda_check(cudaMalloc(&kb->d_smem_broadcast, sizeof(uint64_t) * g_max_regions), "alloc d_smem_broadcast");
  cuda_check(cudaMalloc(&kb->d_gmem_line_bits, sizeof(uint64_t) * g_max_regions * kGmemLineBitsWords), "alloc d_gmem_line_bits");

  if (g_mode & kModePcMap) {
    cuda_check(cudaMalloc(&kb->d_pcmap, sizeof(PcMapEntry) * g_pcmap_cap), "alloc d_pcmap");
    cuda_check(cudaMalloc(&kb->d_pcmap_count, sizeof(uint32_t)), "alloc d_pcmap_count");
  }

  cuda_check(cudaMalloc(&kb->d_mem_exec_sample_counter, sizeof(uint32_t)), "alloc d_mem_exec_sample_counter");
  cuda_check(cudaMalloc(&kb->d_mem_pattern_sample_counter, sizeof(uint32_t)), "alloc d_mem_pattern_sample_counter");
  cuda_check(cudaMalloc(&kb->d_trace_sample_counter, sizeof(uint32_t)), "alloc d_trace_sample_counter");

  if (g_enable_trace && g_trace_cap > 0) {
    cuda_check(cudaMalloc(&kb->d_trace, sizeof(MemTraceEntry) * g_trace_cap), "alloc d_trace");
    cuda_check(cudaMalloc(&kb->d_trace_count, sizeof(uint32_t)), "alloc d_trace_count");
  }

  cuda_check(cudaMallocManaged(&kb->params, sizeof(DeviceParams)), "alloc params");
  kb->params->max_regions = g_max_regions;
  kb->params->max_depth = g_max_depth;
  kb->params->pcmap_cap = g_pcmap_cap;
  kb->params->enable_inst_class = g_enable_inst_class ? 1u : 0u;
  kb->params->enable_inst_pipe = g_enable_inst_pipe ? 1u : 0u;
  kb->params->enable_bb_count = g_enable_bb_count ? 1u : 0u;
  kb->params->enable_bb_hot = g_enable_bb_hot ? 1u : 0u;
  kb->params->enable_branch_div = g_enable_branch_div ? 1u : 0u;
  kb->params->enable_branch_sites = g_enable_branch_sites ? 1u : 0u;
  kb->params->enable_mem_pattern = g_enable_mem_pattern ? 1u : 0u;
  kb->params->reweight_mem_exec = g_reweight_mem_exec ? 1u : 0u;
  kb->params->reweight_mem_pattern = g_reweight_mem_pattern ? 1u : 0u;
  kb->params->sample_cta = g_sample_cta;
  kb->params->sample_warp = g_sample_warp;
  kb->params->sample_mem_every_n = g_sample_mem_every_n;
  kb->params->target_region = g_target_region;
  kb->params->iter_begin = g_iter_begin;
  kb->params->iter_end = g_iter_end;
  kb->params->trace_cap = g_enable_trace ? g_trace_cap : 0u;
  kb->params->max_warps = g_max_warps;
  kb->params->active_warps = 0;
  kb->params->d_sp = kb->d_sp;
  kb->params->d_stack = kb->d_stack;
  kb->params->d_curr = kb->d_curr;
  kb->params->d_marker_mismatch_count = kb->d_marker_mismatch_count;
  kb->params->d_target_iter = kb->d_target_iter;
  kb->params->d_target_depth = kb->d_target_depth;
  kb->params->d_target_active = kb->d_target_active;
  kb->params->d_counters = kb->d_counters;
  kb->params->d_inst_class = kb->d_inst_class;
  kb->params->d_inst_pipe = kb->d_inst_pipe;
  kb->params->d_bb_exec = kb->d_bb_exec;
  kb->params->bb_cap = (g_enable_bb_hot && g_bb_cap > 0) ? g_bb_cap : 0u;
  kb->params->d_bb_hot = kb->d_bb_hot;
  kb->params->d_branch_div_hist = kb->d_branch_div_hist;
  kb->params->d_branch_active_hist = kb->d_branch_active_hist;
  kb->params->branch_site_cap = (g_enable_branch_sites && g_branch_site_cap > 0) ? g_branch_site_cap : 0u;
  kb->params->d_branch_site_exec = kb->d_branch_site_exec;
  kb->params->d_branch_site_taken_warp = kb->d_branch_site_taken_warp;
  kb->params->d_branch_site_fall_warp = kb->d_branch_site_fall_warp;
  kb->params->d_branch_site_taken_lanes = kb->d_branch_site_taken_lanes;
  kb->params->d_branch_site_fall_lanes = kb->d_branch_site_fall_lanes;
  kb->params->d_gmem_sector_hist = kb->d_gmem_sector_hist;
  kb->params->d_gmem_sectors = kb->d_gmem_sectors;
  kb->params->d_gmem_align_hist = kb->d_gmem_align_hist;
  kb->params->d_gmem_stride_hist = kb->d_gmem_stride_hist;
  kb->params->gmem_set_bins = (g_enable_mem_pattern && g_gmem_set_bins > 0) ? g_gmem_set_bins : 0u;
  kb->params->d_gmem_set_hist = kb->d_gmem_set_hist;
  kb->params->d_smem_bank_hist = kb->d_smem_bank_hist;
  kb->params->d_smem_span_hist = kb->d_smem_span_hist;
  kb->params->d_smem_broadcast = kb->d_smem_broadcast;
  kb->params->d_gmem_line_bits = kb->d_gmem_line_bits;
  kb->params->d_pcmap = kb->d_pcmap;
  kb->params->d_pcmap_count = kb->d_pcmap_count;
  kb->params->d_mem_exec_sample_counter = kb->d_mem_exec_sample_counter;
  kb->params->d_mem_pattern_sample_counter = kb->d_mem_pattern_sample_counter;
  kb->params->d_trace_sample_counter = kb->d_trace_sample_counter;
  kb->params->d_trace = kb->d_trace;
  kb->params->d_trace_count = kb->d_trace_count;

  cuda_check(cudaDeviceSynchronize(), "sync after params init");
}

static void reset_kernel_buffers(KernelBuffers* kb, uint32_t active_warps) {
  if (active_warps > g_max_warps) active_warps = g_max_warps;
  kb->params->active_warps = active_warps;

  cuda_check(cudaMemset(kb->d_sp, 0, sizeof(uint32_t) * active_warps), "reset d_sp");
  cuda_check(cudaMemset(kb->d_stack, 0, sizeof(uint32_t) * active_warps * g_max_depth), "reset d_stack");
  cuda_check(cudaMemset(kb->d_curr, 0, sizeof(uint32_t) * active_warps), "reset d_curr");
  if (kb->d_marker_mismatch_count) {
    cuda_check(cudaMemset(kb->d_marker_mismatch_count, 0, sizeof(uint64_t)), "reset d_marker_mismatch_count");
  }
  cuda_check(cudaMemset(kb->d_target_iter, 0, sizeof(uint32_t) * active_warps), "reset d_target_iter");
  cuda_check(cudaMemset(kb->d_target_depth, 0, sizeof(uint32_t) * active_warps), "reset d_target_depth");
  cuda_check(cudaMemset(kb->d_target_active, 0, sizeof(uint32_t) * active_warps), "reset d_target_active");
  cuda_check(cudaMemset(kb->d_counters, 0, sizeof(uint64_t) * g_max_regions * kCounterCount), "reset d_counters");
  cuda_check(cudaMemset(kb->d_inst_class, 0, sizeof(uint64_t) * g_max_regions * kInstClassCount), "reset d_inst_class");
  cuda_check(cudaMemset(kb->d_inst_pipe, 0, sizeof(uint64_t) * g_max_regions * kInstPipeCount), "reset d_inst_pipe");
  cuda_check(cudaMemset(kb->d_bb_exec, 0, sizeof(uint64_t) * g_max_regions), "reset d_bb_exec");
  if (kb->d_bb_hot && kb->params->bb_cap) {
    cuda_check(cudaMemset(kb->d_bb_hot, 0, sizeof(uint64_t) * kb->params->bb_cap), "reset d_bb_hot");
  }
  cuda_check(cudaMemset(kb->d_branch_div_hist, 0, sizeof(uint64_t) * g_max_regions * kBranchDivHistBins), "reset d_branch_div_hist");
  cuda_check(cudaMemset(kb->d_branch_active_hist, 0, sizeof(uint64_t) * g_max_regions * kBranchActiveHistBins), "reset d_branch_active_hist");
  if (kb->d_branch_site_exec && kb->params->branch_site_cap) {
    const size_t n = static_cast<size_t>(kb->params->branch_site_cap);
    cuda_check(cudaMemset(kb->d_branch_site_exec, 0, sizeof(uint64_t) * n), "reset d_branch_site_exec");
    cuda_check(cudaMemset(kb->d_branch_site_taken_warp, 0, sizeof(uint64_t) * n), "reset d_branch_site_taken_warp");
    cuda_check(cudaMemset(kb->d_branch_site_fall_warp, 0, sizeof(uint64_t) * n), "reset d_branch_site_fall_warp");
    cuda_check(cudaMemset(kb->d_branch_site_taken_lanes, 0, sizeof(uint64_t) * n), "reset d_branch_site_taken_lanes");
    cuda_check(cudaMemset(kb->d_branch_site_fall_lanes, 0, sizeof(uint64_t) * n), "reset d_branch_site_fall_lanes");
  }
  cuda_check(cudaMemset(kb->d_gmem_sector_hist, 0, sizeof(uint64_t) * g_max_regions * kGmemSectorHistBins), "reset d_gmem_sector_hist");
  cuda_check(cudaMemset(kb->d_gmem_sectors, 0, sizeof(uint64_t) * g_max_regions), "reset d_gmem_sectors");
  cuda_check(cudaMemset(kb->d_gmem_align_hist, 0, sizeof(uint64_t) * g_max_regions * kGmemAlignHistBins), "reset d_gmem_align_hist");
  cuda_check(cudaMemset(kb->d_gmem_stride_hist, 0, sizeof(uint64_t) * g_max_regions * kStrideClassBins), "reset d_gmem_stride_hist");
  if (kb->d_gmem_set_hist && kb->params->gmem_set_bins) {
    cuda_check(cudaMemset(kb->d_gmem_set_hist, 0, sizeof(uint64_t) * g_max_regions * kb->params->gmem_set_bins), "reset d_gmem_set_hist");
  }
  cuda_check(cudaMemset(kb->d_smem_bank_hist, 0, sizeof(uint64_t) * g_max_regions * kSmemBankHistBins), "reset d_smem_bank_hist");
  cuda_check(cudaMemset(kb->d_smem_span_hist, 0, sizeof(uint64_t) * g_max_regions * kSmemSpanHistBins), "reset d_smem_span_hist");
  cuda_check(cudaMemset(kb->d_smem_broadcast, 0, sizeof(uint64_t) * g_max_regions), "reset d_smem_broadcast");
  cuda_check(cudaMemset(kb->d_gmem_line_bits, 0, sizeof(uint64_t) * g_max_regions * kGmemLineBitsWords), "reset d_gmem_line_bits");

  if (kb->d_pcmap_count) cuda_check(cudaMemset(kb->d_pcmap_count, 0, sizeof(uint32_t)), "reset d_pcmap_count");
  cuda_check(cudaMemset(kb->d_mem_exec_sample_counter, 0, sizeof(uint32_t)), "reset d_mem_exec_sample_counter");
  cuda_check(cudaMemset(kb->d_mem_pattern_sample_counter, 0, sizeof(uint32_t)), "reset d_mem_pattern_sample_counter");
  cuda_check(cudaMemset(kb->d_trace_sample_counter, 0, sizeof(uint32_t)), "reset d_trace_sample_counter");
  if (kb->d_trace_count) cuda_check(cudaMemset(kb->d_trace_count, 0, sizeof(uint32_t)), "reset d_trace_count");
}

static void write_region_stats(const CtxState* st) {
  uint64_t marker_mismatch = 0;
  if (st->buffers.d_marker_mismatch_count) {
    cuda_check(cudaMemcpy(&marker_mismatch, st->buffers.d_marker_mismatch_count, sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "copy marker_mismatch_count");
  }

  // inst_pipe is optional and disabled by default (IKP_NVBIT_ENABLE_INST_PIPE=0).
  // Keep this flag on the host so we can emit JSON null when disabled.
  const bool enable_inst_pipe = (st->buffers.params != nullptr) && (st->buffers.params->enable_inst_pipe != 0u);

  std::vector<uint64_t> counters(g_max_regions * kCounterCount, 0);
  cuda_check(cudaMemcpy(counters.data(), st->buffers.d_counters, counters.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy counters");
  std::vector<uint64_t> inst_class(g_max_regions * kInstClassCount, 0);
  std::vector<uint64_t> inst_pipe(g_max_regions * kInstPipeCount, 0);
  std::vector<uint64_t> bb_exec(g_max_regions, 0);
  std::vector<uint64_t> branch_div(g_max_regions * kBranchDivHistBins, 0);
  std::vector<uint64_t> branch_active(g_max_regions * kBranchActiveHistBins, 0);
  std::vector<uint64_t> gmem_sector_hist(g_max_regions * kGmemSectorHistBins, 0);
  std::vector<uint64_t> gmem_sectors(g_max_regions, 0);
  std::vector<uint64_t> gmem_align_hist(g_max_regions * kGmemAlignHistBins, 0);
  std::vector<uint64_t> gmem_stride_hist(g_max_regions * kStrideClassBins, 0);
  std::vector<uint64_t> gmem_set_hist;
  std::vector<uint64_t> smem_bank_hist(g_max_regions * kSmemBankHistBins, 0);
  std::vector<uint64_t> smem_span_hist(g_max_regions * kSmemSpanHistBins, 0);
  std::vector<uint64_t> smem_broadcast(g_max_regions, 0);
  std::vector<uint64_t> gmem_line_bits(g_max_regions * kGmemLineBitsWords, 0);

  cuda_check(cudaMemcpy(inst_class.data(), st->buffers.d_inst_class, inst_class.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy inst_class");
  cuda_check(cudaMemcpy(inst_pipe.data(), st->buffers.d_inst_pipe, inst_pipe.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy inst_pipe");
  cuda_check(cudaMemcpy(bb_exec.data(), st->buffers.d_bb_exec, bb_exec.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy bb_exec");
  cuda_check(cudaMemcpy(branch_div.data(), st->buffers.d_branch_div_hist, branch_div.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy branch_div_hist");
  cuda_check(cudaMemcpy(branch_active.data(), st->buffers.d_branch_active_hist, branch_active.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy branch_active_hist");
  cuda_check(cudaMemcpy(gmem_sector_hist.data(), st->buffers.d_gmem_sector_hist, gmem_sector_hist.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost),
             "copy gmem_sector_hist");
  cuda_check(cudaMemcpy(gmem_sectors.data(), st->buffers.d_gmem_sectors, gmem_sectors.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy gmem_sectors");
  cuda_check(cudaMemcpy(gmem_align_hist.data(), st->buffers.d_gmem_align_hist, gmem_align_hist.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy gmem_align_hist");
  cuda_check(cudaMemcpy(gmem_stride_hist.data(), st->buffers.d_gmem_stride_hist, gmem_stride_hist.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost),
             "copy gmem_stride_hist");
  const uint32_t gmem_set_bins = st->buffers.params ? st->buffers.params->gmem_set_bins : 0u;
  if (gmem_set_bins && st->buffers.d_gmem_set_hist) {
    gmem_set_hist.assign(static_cast<size_t>(g_max_regions) * gmem_set_bins, 0);
    cuda_check(cudaMemcpy(gmem_set_hist.data(), st->buffers.d_gmem_set_hist, gmem_set_hist.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "copy gmem_set_hist");
  }
  cuda_check(cudaMemcpy(smem_bank_hist.data(), st->buffers.d_smem_bank_hist, smem_bank_hist.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy smem_bank_hist");
  cuda_check(cudaMemcpy(smem_span_hist.data(), st->buffers.d_smem_span_hist, smem_span_hist.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy smem_span_hist");
  cuda_check(cudaMemcpy(smem_broadcast.data(), st->buffers.d_smem_broadcast, smem_broadcast.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy smem_broadcast");
  cuda_check(cudaMemcpy(gmem_line_bits.data(), st->buffers.d_gmem_line_bits, gmem_line_bits.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy gmem_line_bits");

  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string path = g_trace_path + "/region_stats_" + safe + "_" + std::to_string(g_kernel_id) + ".json";
  std::ofstream out(path);
  out << "{\n";
  out << "  \"kernel\": \"" << json_escape(st->kernel_name) << "\",\n";
  out << "  \"kernel_id\": " << g_kernel_id << ",\n";
  out << "  \"kernel_addr\": " << st->kernel_addr << ",\n";
  out << "  \"marker_mismatch_count\": " << marker_mismatch << ",\n";
  out << "  \"regions\": [\n";
  bool first = true;
  for (uint32_t r = 0; r < g_max_regions; ++r) {
    const uint64_t* base = &counters[r * kCounterCount];
    bool any = false;
    for (uint32_t i = 0; i < kCounterCount; ++i) {
      if (base[i] != 0) { any = true; break; }
    }
    if (!any) {
      const uint64_t* cls = &inst_class[r * kInstClassCount];
      for (uint32_t i = 0; i < kInstClassCount; ++i) {
        if (cls[i] != 0) { any = true; break; }
      }
    }
    if (!any && bb_exec[r] != 0) any = true;
    if (!any) {
      const uint64_t* div = &branch_div[r * kBranchDivHistBins];
      for (uint32_t i = 0; i < kBranchDivHistBins; ++i) {
        if (div[i] != 0) { any = true; break; }
      }
    }
    if (!any) {
      const uint64_t* act = &branch_active[r * kBranchActiveHistBins];
      for (uint32_t i = 0; i < kBranchActiveHistBins; ++i) {
        if (act[i] != 0) { any = true; break; }
      }
    }
    if (!any) {
      const uint64_t* hs = &gmem_sector_hist[r * kGmemSectorHistBins];
      for (uint32_t i = 0; i < kGmemSectorHistBins; ++i) {
        if (hs[i] != 0) { any = true; break; }
      }
    }
    if (!any) {
      const uint64_t* hs = &smem_bank_hist[r * kSmemBankHistBins];
      for (uint32_t i = 0; i < kSmemBankHistBins; ++i) {
        if (hs[i] != 0) { any = true; break; }
      }
    }
    if (!any) continue;
    if (!first) out << ",\n";
    first = false;
    const uint64_t* cls = &inst_class[r * kInstClassCount];
    const uint64_t* pipe = &inst_pipe[r * kInstPipeCount];
    const uint64_t* div = &branch_div[r * kBranchDivHistBins];
    const uint64_t* act = &branch_active[r * kBranchActiveHistBins];
    const uint64_t* gsec = &gmem_sector_hist[r * kGmemSectorHistBins];
    const uint64_t* galign = &gmem_align_hist[r * kGmemAlignHistBins];
    const uint64_t* gstride = &gmem_stride_hist[r * kStrideClassBins];
    const uint64_t* gset = (gmem_set_bins && !gmem_set_hist.empty()) ? &gmem_set_hist[r * gmem_set_bins] : nullptr;
    const uint64_t* sbank = &smem_bank_hist[r * kSmemBankHistBins];
    const uint64_t* sspan = &smem_span_hist[r * kSmemSpanHistBins];
    const uint64_t* gline = &gmem_line_bits[r * kGmemLineBitsWords];
    uint64_t gmem_unique_lines_est = 0;
    for (uint32_t w = 0; w < kGmemLineBitsWords; ++w) {
      gmem_unique_lines_est += static_cast<uint64_t>(__builtin_popcountll(gline[w]));
    }
    // Optional derived stats from divergence histogram.
    uint64_t div_total = 0;
    double div_entropy = 0.0;
    double div_avg = 0.0;
    for (uint32_t i = 0; i < kBranchDivHistBins; ++i) {
      div_total += div[i];
      div_avg += static_cast<double>(i) * static_cast<double>(div[i]);
    }
    if (div_total > 0) {
      div_avg /= static_cast<double>(div_total);
      for (uint32_t i = 0; i < kBranchDivHistBins; ++i) {
        if (div[i] == 0) continue;
        double p = static_cast<double>(div[i]) / static_cast<double>(div_total);
        div_entropy -= p * std::log2(p);
      }
    }
    uint64_t act_total = 0;
    double act_avg = 0.0;
    for (uint32_t i = 0; i < kBranchActiveHistBins; ++i) {
      act_total += act[i];
      act_avg += static_cast<double>(i) * static_cast<double>(act[i]);
    }
    if (act_total > 0) act_avg /= static_cast<double>(act_total);

    out << "    {\"region\": " << r
        << ", \"inst_total\": " << base[kInstTotal]
        << ", \"inst_pred_off\": " << base[kInstPredOff]
        << ", \"gmem_load\": " << base[kGmemLoad]
        << ", \"gmem_store\": " << base[kGmemStore]
        << ", \"gmem_bytes\": " << base[kGmemBytes]
        << ", \"gmem_inst_load_count\": " << base[kGmemLoad]
        << ", \"gmem_inst_store_count\": " << base[kGmemStore]
        << ", \"gmem_req_bytes\": " << base[kGmemBytes]
        << ", \"smem_load\": " << base[kSmemLoad]
        << ", \"smem_store\": " << base[kSmemStore]
        << ", \"smem_bytes\": " << base[kSmemBytes]
        << ", \"smem_inst_load_count\": " << base[kSmemLoad]
        << ", \"smem_inst_store_count\": " << base[kSmemStore]
        << ", \"smem_req_bytes\": " << base[kSmemBytes]
        << ", \"lmem_load\": " << base[kLmemLoad]
        << ", \"lmem_store\": " << base[kLmemStore]
        << ", \"lmem_bytes\": " << base[kLmemBytes]
        << ", \"lmem_inst_load_count\": " << base[kLmemLoad]
        << ", \"lmem_inst_store_count\": " << base[kLmemStore]
        << ", \"lmem_req_bytes\": " << base[kLmemBytes]
        << ", \"inst_class\": {"
        << "\"alu_fp32\": " << cls[kInstClassAluFp32]
        << ", \"alu_int\": " << cls[kInstClassAluInt]
        << ", \"tensor_wgmma\": " << cls[kInstClassTensorWgmma]
        << ", \"ld_global\": " << cls[kInstClassLdGlobal]
        << ", \"st_global\": " << cls[kInstClassStGlobal]
        << ", \"ld_shared\": " << cls[kInstClassLdShared]
        << ", \"st_shared\": " << cls[kInstClassStShared]
        << ", \"ld_local\": " << cls[kInstClassLdLocal]
        << ", \"st_local\": " << cls[kInstClassStLocal]
        << ", \"barrier\": " << cls[kInstClassBarrier]
        << ", \"membar\": " << cls[kInstClassMembar]
        << ", \"branch\": " << cls[kInstClassBranch]
        << ", \"call\": " << cls[kInstClassCall]
        << ", \"ret\": " << cls[kInstClassRet]
        << ", \"special\": " << cls[kInstClassSpecial]
        << ", \"other\": " << cls[kInstClassOther]
        << "}";

    out << ", \"inst_pipe\": ";
    if (enable_inst_pipe) {
      out << "{"
          << "\"ld\": " << pipe[kInstPipeLd]
          << ", \"st\": " << pipe[kInstPipeSt]
          << ", \"tex\": " << pipe[kInstPipeTex]
          << ", \"uniform\": " << pipe[kInstPipeUniform]
          << ", \"fp32\": " << pipe[kInstPipeFp32]
          << ", \"fp16\": " << pipe[kInstPipeFp16]
          << ", \"fp64\": " << pipe[kInstPipeFp64]
          << ", \"int\": " << pipe[kInstPipeInt]
          << ", \"sfu\": " << pipe[kInstPipeSfu]
          << ", \"tensor\": " << pipe[kInstPipeTensor]
          << ", \"barrier\": " << pipe[kInstPipeBarrier]
          << ", \"membar\": " << pipe[kInstPipeMembar]
          << ", \"branch\": " << pipe[kInstPipeBranch]
          << ", \"callret\": " << pipe[kInstPipeCallRet]
          << ", \"special\": " << pipe[kInstPipeSpecial]
          << ", \"other\": " << pipe[kInstPipeOther]
          << "}";
    } else {
      out << "null";
    }

    out << ", \"reg_spill_suspected\": "
        << (((cls[kInstClassLdLocal] + cls[kInstClassStLocal]) > 0 || (base[kLmemLoad] + base[kLmemStore]) > 0) ? 1 : 0)
        << ", \"spill_ld_local_inst\": " << cls[kInstClassLdLocal]
        << ", \"spill_st_local_inst\": " << cls[kInstClassStLocal]
        << ", \"bb_exec\": " << bb_exec[r]
        << ", \"branch_div_hist\": [";
    for (uint32_t i = 0; i < kBranchDivHistBins; ++i) {
      out << div[i];
      if (i + 1 != kBranchDivHistBins) out << ", ";
    }
    out << "], \"branch_active_hist\": [";
    for (uint32_t i = 0; i < kBranchActiveHistBins; ++i) {
      out << act[i];
      if (i + 1 != kBranchActiveHistBins) out << ", ";
    }
    out << "], \"branch_active_avg_lanes\": " << act_avg
        << ", \"branch_div_avg_active\": " << div_avg
        << ", \"branch_div_entropy\": " << div_entropy
        << ", \"gmem_sectors_32b\": " << gmem_sectors[r]
        << ", \"gmem_sectors_per_inst_hist\": [";
    for (uint32_t i = 0; i < kGmemSectorHistBins; ++i) {
      out << gsec[i];
      if (i + 1 != kGmemSectorHistBins) out << ", ";
    }
    out << "], \"gmem_alignment_hist\": [";
    for (uint32_t i = 0; i < kGmemAlignHistBins; ++i) {
      out << galign[i];
      if (i + 1 != kGmemAlignHistBins) out << ", ";
    }
    out << "], \"gmem_stride_class_hist\": [";
    for (uint32_t i = 0; i < kStrideClassBins; ++i) {
      out << gstride[i];
      if (i + 1 != kStrideClassBins) out << ", ";
    }
    out << "], \"gmem_unique_lines_est\": " << gmem_unique_lines_est;
    if (gset && gmem_set_bins) {
      out << ", \"gmem_set_bins\": " << gmem_set_bins << ", \"gmem_set_hist\": [";
      for (uint32_t i = 0; i < gmem_set_bins; ++i) {
        out << gset[i];
        if (i + 1 != gmem_set_bins) out << ", ";
      }
      out << "]";
    }
    out << ", \"smem_bank_conflict_max_hist\": [";
    for (uint32_t i = 0; i < kSmemBankHistBins; ++i) {
      out << sbank[i];
      if (i + 1 != kSmemBankHistBins) out << ", ";
    }
    out << "], \"smem_broadcast_count\": " << smem_broadcast[r] << ", \"smem_addr_span_hist\": [";
    for (uint32_t i = 0; i < kSmemSpanHistBins; ++i) {
      out << sspan[i];
      if (i + 1 != kSmemSpanHistBins) out << ", ";
    }
    out << "]}";
  }
  out << "\n  ]\n}\n";
}

static bool collect_pc_aggs(const CtxState* st, std::vector<PcAgg>* pc_aggs, std::vector<AmbigPc>* ambiguous) {
  if (!st || !pc_aggs || !ambiguous) return false;
  if (!(g_mode & kModePcMap)) return false;
  if (!st->buffers.d_pcmap || !st->buffers.d_pcmap_count) return false;

  uint32_t count = 0;
  cuda_check(cudaMemcpy(&count, st->buffers.d_pcmap_count, sizeof(uint32_t), cudaMemcpyDeviceToHost), "copy pcmap count");
  if (count == 0) return false;
  if (count > g_pcmap_cap) count = g_pcmap_cap;

  std::vector<PcMapEntry> entries(count);
  cuda_check(cudaMemcpy(entries.data(), st->buffers.d_pcmap, sizeof(PcMapEntry) * count, cudaMemcpyDeviceToHost), "copy pcmap entries");

  // Aggregate dynamic weights: (function_id, pc_offset) -> (region -> executed_count).
  std::sort(entries.begin(), entries.end(), [](const PcMapEntry& a, const PcMapEntry& b) {
    if (a.function_id != b.function_id) return a.function_id < b.function_id;
    if (a.pc != b.pc) return a.pc < b.pc;
    return a.region < b.region;
  });

  pc_aggs->clear();
  ambiguous->clear();
  pc_aggs->reserve(entries.size());  // upper bound
  ambiguous->reserve(1024);

  constexpr double kAmbiguityTop1FracThresh = 0.90;
  constexpr double kAmbiguityEntropyNormThresh = 0.25;

  for (size_t i = 0; i < entries.size();) {
    const uint32_t fid = entries[i].function_id;
    const uint64_t pc = entries[i].pc;
    PcAgg agg;
    agg.function_id = fid;
    agg.pc = pc;

    uint32_t total = 0;
    uint32_t dominant_region = 0;
    uint32_t dominant_count = 0;

    while (i < entries.size() && entries[i].function_id == fid && entries[i].pc == pc) {
      const uint32_t region = entries[i].region;
      uint32_t c = 0;
      while (i < entries.size() && entries[i].function_id == fid && entries[i].pc == pc &&
             entries[i].region == region) {
        ++c;
        ++i;
      }
      agg.region_counts.emplace_back(region, c);
      total += c;
      if (c > dominant_count) {
        dominant_count = c;
        dominant_region = region;
      }
    }

    // Shannon entropy of the region distribution (in bits).
    double entropy = 0.0;
    if (total > 0 && agg.region_counts.size() > 1) {
      for (const auto& rc : agg.region_counts) {
        const double p = static_cast<double>(rc.second) / static_cast<double>(total);
        if (p > 0.0) entropy -= p * std::log2(p);
      }
    }
    double entropy_norm = 0.0;
    if (agg.region_counts.size() > 1) {
      const double denom = std::log2(static_cast<double>(agg.region_counts.size()));
      if (denom > 0.0) entropy_norm = entropy / denom;
    }
    const double top1_frac =
        (total > 0) ? (static_cast<double>(dominant_count) / static_cast<double>(total)) : 0.0;

    const bool is_ambiguous = (agg.region_counts.size() > 1) &&
                              (top1_frac < kAmbiguityTop1FracThresh ||
                               entropy_norm > kAmbiguityEntropyNormThresh);
    if (is_ambiguous) {
      AmbigPc a;
      a.function_id = fid;
      a.pc = pc;
      a.region_counts = agg.region_counts;
      a.total = total;
      a.dominant_region = dominant_region;
      a.dominant_count = dominant_count;
      a.top1_frac = top1_frac;
      a.entropy = entropy;
      a.entropy_norm = entropy_norm;
      ambiguous->emplace_back(std::move(a));
    }

    pc_aggs->emplace_back(std::move(agg));
  }

  // Keep only the top-K most ambiguous PCs (to avoid huge JSONs on malformed markers).
  constexpr size_t kMaxAmbiguousPcs = 1024;
  if (ambiguous->size() > kMaxAmbiguousPcs) {
    std::partial_sort(ambiguous->begin(), ambiguous->begin() + kMaxAmbiguousPcs, ambiguous->end(),
                      [](const AmbigPc& a, const AmbigPc& b) {
                        if (a.entropy_norm != b.entropy_norm) return a.entropy_norm > b.entropy_norm;
                        return a.top1_frac < b.top1_frac;
                      });
    ambiguous->resize(kMaxAmbiguousPcs);
  } else {
    std::sort(ambiguous->begin(), ambiguous->end(), [](const AmbigPc& a, const AmbigPc& b) {
      if (a.entropy_norm != b.entropy_norm) return a.entropy_norm > b.entropy_norm;
      return a.top1_frac < b.top1_frac;
    });
  }
  return true;
}

static std::string write_pcmap(const CtxState* st, std::vector<PcAgg>* pc_aggs_out) {
  if (!(g_mode & kModePcMap)) return {};
  if (!st) return {};

  std::vector<PcAgg> pc_aggs_local;
  std::vector<AmbigPc> ambiguous;
  if (!collect_pc_aggs(st, &pc_aggs_local, &ambiguous)) return {};
  if (pc_aggs_out) *pc_aggs_out = pc_aggs_local;

  constexpr double kAmbiguityTop1FracThresh = 0.90;
  constexpr double kAmbiguityEntropyNormThresh = 0.25;

  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string path = g_trace_path + "/pc2region_" +
                           safe + "_" + std::to_string(g_kernel_id) + ".json";
  std::ofstream out(path);
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"kernel\": \"" << json_escape(st->kernel_name) << "\",\n";
  out << "  \"kernel_id\": " << g_kernel_id << ",\n";
  out << "  \"kernel_addr\": " << st->kernel_addr << ",\n";
  out << "  \"cubinCrc\": " << st->cubin_crc << ",\n";
  out << "  \"pc2region_format_version\": 3,\n";
  out << "  \"dominant_region_policy\": \"max_executed_count\",\n";
  out << "  \"ambiguity_top1_frac_threshold\": " << kAmbiguityTop1FracThresh << ",\n";
  out << "  \"ambiguity_entropy_norm_threshold\": " << kAmbiguityEntropyNormThresh << ",\n";
  out << "  \"functions\": [\n";
  for (uint32_t fid = 0; fid < st->next_function_id; ++fid) {
    auto it = st->function_names.find(fid);
    const std::string fname = (it == st->function_names.end()) ? std::string() : it->second;
    auto mit = st->function_names_mangled.find(fid);
    const std::string mname =
        (mit == st->function_names_mangled.end()) ? std::string() : mit->second;
    out << "    {\"function_id\": " << fid << ", \"function_name\": \""
        << json_escape(fname) << "\", \"function_name_mangled\": \""
        << json_escape(mname) << "\"}";
    if (fid + 1 != st->next_function_id) out << ",\n";
  }
  out << "\n  ],\n";
  out << "  \"pc2region\": [\n";
  for (size_t i = 0; i < pc_aggs_local.size(); ++i) {
    const auto& agg = pc_aggs_local[i];
    auto fit = st->function_names.find(agg.function_id);
    const std::string fname = (fit == st->function_names.end()) ? std::string() : fit->second;
    auto mit = st->function_names_mangled.find(agg.function_id);
    const std::string mname =
        (mit == st->function_names_mangled.end()) ? std::string() : mit->second;
    uint32_t total = 0;
    uint32_t dominant_region = 0;
    uint32_t dominant_count = 0;
    for (const auto& rc : agg.region_counts) {
      total += rc.second;
      if (rc.second > dominant_count) {
        dominant_count = rc.second;
        dominant_region = rc.first;
      }
    }
    const double top1_frac =
        (total > 0) ? (static_cast<double>(dominant_count) / static_cast<double>(total)) : 0.0;
    double entropy = 0.0;
    if (total > 0 && agg.region_counts.size() > 1) {
      for (const auto& rc : agg.region_counts) {
        const double p = static_cast<double>(rc.second) / static_cast<double>(total);
        if (p > 0.0) entropy -= p * std::log2(p);
      }
    }
    double entropy_norm = 0.0;
    if (agg.region_counts.size() > 1) {
      const double denom = std::log2(static_cast<double>(agg.region_counts.size()));
      if (denom > 0.0) entropy_norm = entropy / denom;
    }

    // `region` is kept for backward compatible joins (dominant region).
    out << "    {\"function_id\": " << agg.function_id << ", \"function_name\": \""
        << json_escape(fname) << "\""
        << ", \"function_name_mangled\": \"" << json_escape(mname) << "\""
        << ", \"pc_offset\": " << agg.pc << ", \"region\": " << dominant_region
        << ", \"regions\": [";
    for (size_t j = 0; j < agg.region_counts.size(); ++j) {
      out << agg.region_counts[j].first;
      if (j + 1 != agg.region_counts.size()) out << ", ";
    }
    out << "], \"region_exec_counts\": [";
    for (size_t j = 0; j < agg.region_counts.size(); ++j) {
      out << agg.region_counts[j].second;
      if (j + 1 != agg.region_counts.size()) out << ", ";
    }
    out << "], \"total_exec_count\": " << total
        << ", \"dominant_region\": " << dominant_region
        << ", \"dominant_exec_count\": " << dominant_count
        << ", \"dominant_frac\": " << top1_frac
        << ", \"ambiguity_n_regions\": " << agg.region_counts.size()
        << ", \"ambiguity_entropy\": " << entropy
        << ", \"ambiguity_entropy_norm\": " << entropy_norm
        << "}";
    if (i + 1 != pc_aggs_local.size()) out << ",\n";
  }
  out << "\n  ],\n";
  out << "  \"ambiguous_pcs\": [\n";
  for (size_t i = 0; i < ambiguous.size(); ++i) {
    const auto& a = ambiguous[i];
    auto fit = st->function_names.find(a.function_id);
    const std::string fname = (fit == st->function_names.end()) ? std::string() : fit->second;
    auto mit = st->function_names_mangled.find(a.function_id);
    const std::string mname =
        (mit == st->function_names_mangled.end()) ? std::string() : mit->second;
    out << "    {\"function_id\": " << a.function_id << ", \"function_name\": \""
        << json_escape(fname) << "\""
        << ", \"function_name_mangled\": \"" << json_escape(mname) << "\""
        << ", \"pc_offset\": " << a.pc
        << ", \"dominant_region\": " << a.dominant_region
        << ", \"dominant_exec_count\": " << a.dominant_count
        << ", \"total_exec_count\": " << a.total
        << ", \"dominant_frac\": " << a.top1_frac
        << ", \"ambiguity_entropy\": " << a.entropy
        << ", \"ambiguity_entropy_norm\": " << a.entropy_norm
        << ", \"regions\": [";
    for (size_t j = 0; j < a.region_counts.size(); ++j) {
      out << a.region_counts[j].first;
      if (j + 1 != a.region_counts.size()) out << ", ";
    }
    out << "], \"region_exec_counts\": [";
    for (size_t j = 0; j < a.region_counts.size(); ++j) {
      out << a.region_counts[j].second;
      if (j + 1 != a.region_counts.size()) out << ", ";
    }
    out << "]}";
    if (i + 1 != ambiguous.size()) out << ",\n";
  }
  out << "\n  ]\n}\n";
  return path;
}

static std::string write_sass_all(CUcontext ctx, const CtxState* st) {
  if (!g_dump_sass || !st) return {};
  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string path = g_trace_path + "/sass_all_" + safe + "_" + std::to_string(g_kernel_id) + ".sass";
  std::ofstream out(path);
  out << "// kernel: " << st->kernel_name << "\n";
  out << "// kernel_id: " << g_kernel_id << "\n";
  out << "// NOTE: function-relative offsets match pc2region.pc_offset\n\n";
  std::vector<uint32_t> fids;
  if (!st->last_launch_function_ids.empty()) {
    fids = st->last_launch_function_ids;
    std::sort(fids.begin(), fids.end());
    fids.erase(std::unique(fids.begin(), fids.end()), fids.end());
  } else {
    fids.reserve(st->next_function_id);
    for (uint32_t fid = 0; fid < st->next_function_id; ++fid) fids.push_back(fid);
  }

  for (uint32_t fid : fids) {
    CUfunction f = nullptr;
    {
      auto itf = st->function_by_id.find(fid);
      if (itf != st->function_by_id.end()) f = itf->second;
    }
    auto fit = st->function_names.find(fid);
    const std::string fname = (fit == st->function_names.end()) ? std::string() : fit->second;
    auto mit = st->function_names_mangled.find(fid);
    const std::string mname = (mit == st->function_names_mangled.end()) ? std::string() : mit->second;
    out << "// ===== function_id=" << fid << " =====\n";
    out << "// function_name=" << fname << "\n";
    out << "// function_name_mangled=" << mname << "\n";
    auto sit = st->sass_by_function.find(fid);
    if (sit == st->sass_by_function.end() || sit->second.empty()) {
      out << "// (no SASS cached for this function)\n\n";
      continue;
    }
    for (const auto& l : sit->second) {
      if (g_dump_sass_meta) {
        out << "// meta: idx=" << l.idx
            << " op_short=" << l.op_short
            << " opcode=" << l.opcode
            << " pred=" << (l.has_pred ? 1 : 0);
        if (l.has_pred) {
          out << " pred_num=" << l.pred_num
              << " pred_neg=" << (l.pred_neg ? 1 : 0)
              << " pred_uniform=" << (l.pred_uniform ? 1 : 0);
        }
        out << " mem_space=" << mem_space_str(l.mem_space)
            << " is_load=" << (l.is_load ? 1 : 0)
            << " is_store=" << (l.is_store ? 1 : 0)
            << " size=" << l.access_size
            << "\n";
      }
      maybe_emit_lineinfo(out, ctx, f, st, fid, static_cast<uint32_t>(l.pc));
      out << "/*0x" << std::hex << l.pc << std::dec << "*/ " << l.sass << "\n";
    }
    out << "\n";
  }
  return path;
}

static std::string write_sass_by_region(CUcontext ctx, const CtxState* st, const std::vector<PcAgg>& pc_aggs) {
  if (!g_dump_sass || !g_dump_sass_by_region || !st) return {};
  if (pc_aggs.empty()) return {};

  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string dir =
      g_trace_path + "/sass_regions_" + safe + "_" + std::to_string(g_kernel_id);
  ensure_dir(dir);

  // Build a fast lookup: (function_id, pc) -> sass line.
  std::unordered_map<uint32_t, std::unordered_map<uint64_t, const SassLine*>> lookup;
  lookup.reserve(st->sass_by_function.size());
  for (const auto& kv : st->sass_by_function) {
    const uint32_t fid = kv.first;
    const auto& lines = kv.second;
    auto& m = lookup[fid];
    m.reserve(lines.size());
    for (const auto& l : lines) {
      m.emplace(l.pc, &l);
    }
  }

  // Collect which region ids appear at least once.
  std::unordered_set<uint32_t> regions;
  regions.reserve(64);
  for (const auto& agg : pc_aggs) {
    for (const auto& rc : agg.region_counts) {
      regions.insert(rc.first);
    }
  }
  std::vector<uint32_t> region_list(regions.begin(), regions.end());
  std::sort(region_list.begin(), region_list.end());

  // For each region, emit a slice file containing every (function,pc) observed in that region.
  for (uint32_t rid : region_list) {
    const std::string path = dir + "/region_" + std::to_string(rid) + ".sass";
    std::ofstream out(path);
    out << "// kernel: " << st->kernel_name << "\n";
    out << "// kernel_id: " << g_kernel_id << "\n";
    out << "// region: " << rid << "\n";
    out << "// Each line is a SASS instruction observed under this region (may include ambiguous PCs).\n\n";

    // Sort output by (function_id, pc) for readability.
    std::vector<std::pair<uint32_t, uint64_t>> pcs;
    pcs.reserve(pc_aggs.size());
    std::unordered_map<uint64_t, uint32_t> key_to_count;  // packed key -> count
    std::unordered_map<uint64_t, uint32_t> key_to_total;  // packed key -> total
    auto pack = [](uint32_t fid, uint64_t pc) -> uint64_t {
      // fid is small; keep pc low bits intact. This is only for local hash keys.
      return (static_cast<uint64_t>(fid) << 48) ^ (pc & 0x0000FFFFFFFFFFFFull);
    };
    for (const auto& agg : pc_aggs) {
      uint32_t c = 0;
      uint32_t total = 0;
      for (const auto& rc : agg.region_counts) {
        total += rc.second;
        if (rc.first == rid) c = rc.second;
      }
      if (c == 0) continue;
      pcs.emplace_back(agg.function_id, agg.pc);
      const uint64_t k = pack(agg.function_id, agg.pc);
      key_to_count[k] = c;
      key_to_total[k] = total;
    }
    std::sort(pcs.begin(), pcs.end(), [](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first < b.first;
      return a.second < b.second;
    });

    uint32_t last_fid = 0xFFFFFFFFu;
    for (const auto& fp : pcs) {
      const uint32_t fid = fp.first;
      const uint64_t pc = fp.second;
      CUfunction f = nullptr;
      {
        auto itf = st->function_by_id.find(fid);
        if (itf != st->function_by_id.end()) f = itf->second;
      }
      if (fid != last_fid) {
        auto fit = st->function_names.find(fid);
        const std::string fname = (fit == st->function_names.end()) ? std::string() : fit->second;
        auto mit = st->function_names_mangled.find(fid);
        const std::string mname = (mit == st->function_names_mangled.end()) ? std::string() : mit->second;
        out << "\n// ---- function_id=" << fid << " ----\n";
        out << "// function_name=" << fname << "\n";
        out << "// function_name_mangled=" << mname << "\n";
        last_fid = fid;
      }

      const uint64_t k = pack(fid, pc);
      const uint32_t c = key_to_count[k];
      const uint32_t total = key_to_total[k];
      const double frac = (total > 0) ? (static_cast<double>(c) / static_cast<double>(total)) : 0.0;

      const SassLine* line = nullptr;
      auto itf = lookup.find(fid);
      if (itf != lookup.end()) {
        auto itpc = itf->second.find(pc);
        if (itpc != itf->second.end()) line = itpc->second;
      }
      if (line) {
        out << "// exec_count=" << c << " total=" << total << " frac=" << std::fixed << std::setprecision(6) << frac
            << std::defaultfloat << "\n";
        if (g_dump_sass_meta) {
          out << "// meta: idx=" << line->idx
              << " op_short=" << line->op_short
              << " opcode=" << line->opcode
              << " pred=" << (line->has_pred ? 1 : 0);
          if (line->has_pred) {
            out << " pred_num=" << line->pred_num
                << " pred_neg=" << (line->pred_neg ? 1 : 0)
                << " pred_uniform=" << (line->pred_uniform ? 1 : 0);
          }
          out << " mem_space=" << mem_space_str(line->mem_space)
              << " is_load=" << (line->is_load ? 1 : 0)
              << " is_store=" << (line->is_store ? 1 : 0)
              << " size=" << line->access_size
              << "\n";
        }
        maybe_emit_lineinfo(out, ctx, f, st, fid, static_cast<uint32_t>(pc));
        out << "/*0x" << std::hex << pc << std::dec << "*/ " << line->sass << "\n";
      } else {
        out << "// exec_count=" << c << " total=" << total << " frac=" << std::fixed << std::setprecision(6) << frac
            << std::defaultfloat << " (SASS not found)\n";
        out << "/*0x" << std::hex << pc << std::dec << "*/ <missing>\n";
      }
    }
    out << "\n";
  }

  return dir;
}

static std::string write_hotspots(const CtxState* st) {
  if (!st) return {};
  if (!st->buffers.params) return {};
  const bool want_bb = (st->buffers.params->enable_bb_hot != 0u) &&
                       st->buffers.params->d_bb_hot && st->buffers.params->bb_cap;
  const bool want_branch = (st->buffers.params->enable_branch_sites != 0u) &&
                           st->buffers.params->d_branch_site_exec &&
                           st->buffers.params->branch_site_cap;
  if (!want_bb && !want_branch) return {};

  const uint32_t bb_cap = want_bb ? st->buffers.params->bb_cap : 0u;
  std::vector<uint64_t> bb_hot;
  if (want_bb) {
    bb_hot.assign(bb_cap, 0);
    cuda_check(cudaMemcpy(bb_hot.data(), st->buffers.d_bb_hot,
                          sizeof(uint64_t) * bb_cap, cudaMemcpyDeviceToHost),
               "copy bb_hot");
  }

  const uint32_t branch_cap = want_branch ? st->buffers.params->branch_site_cap : 0u;
  std::vector<uint64_t> b_exec, b_taken_warp, b_fall_warp, b_taken_lanes, b_fall_lanes;
  if (want_branch) {
    b_exec.assign(branch_cap, 0);
    b_taken_warp.assign(branch_cap, 0);
    b_fall_warp.assign(branch_cap, 0);
    b_taken_lanes.assign(branch_cap, 0);
    b_fall_lanes.assign(branch_cap, 0);
    cuda_check(cudaMemcpy(b_exec.data(), st->buffers.d_branch_site_exec,
                          sizeof(uint64_t) * branch_cap, cudaMemcpyDeviceToHost),
               "copy branch_site_exec");
    cuda_check(cudaMemcpy(b_taken_warp.data(), st->buffers.d_branch_site_taken_warp,
                          sizeof(uint64_t) * branch_cap, cudaMemcpyDeviceToHost),
               "copy branch_site_taken_warp");
    cuda_check(cudaMemcpy(b_fall_warp.data(), st->buffers.d_branch_site_fall_warp,
                          sizeof(uint64_t) * branch_cap, cudaMemcpyDeviceToHost),
               "copy branch_site_fall_warp");
    cuda_check(cudaMemcpy(b_taken_lanes.data(), st->buffers.d_branch_site_taken_lanes,
                          sizeof(uint64_t) * branch_cap, cudaMemcpyDeviceToHost),
               "copy branch_site_taken_lanes");
    cuda_check(cudaMemcpy(b_fall_lanes.data(), st->buffers.d_branch_site_fall_lanes,
                          sizeof(uint64_t) * branch_cap, cudaMemcpyDeviceToHost),
               "copy branch_site_fall_lanes");
  }

  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string path = g_trace_path + "/hotspots_" +
                           safe + "_" + std::to_string(g_kernel_id) + ".json";
  std::ofstream out(path);
  out << "{\n";
  out << "  \"kernel\": \"" << json_escape(st->kernel_name) << "\",\n";
  out << "  \"kernel_id\": " << g_kernel_id << ",\n";
  out << "  \"kernel_addr\": " << st->kernel_addr << ",\n";
  out << "  \"cubinCrc\": " << st->cubin_crc << ",\n";
  out << "  \"bb_cap\": " << bb_cap << ",\n";
  out << "  \"branch_site_cap\": " << branch_cap << ",\n";

  out << "  \"bb_entries\": [\n";
  bool first = true;
  if (want_bb) {
    for (const auto& m : st->bb_meta) {
      if (m.bb_id >= bb_cap) continue;
      const uint64_t c = bb_hot[m.bb_id];
      if (c == 0) continue;
      auto it = st->function_names.find(m.function_id);
      const std::string fname = (it == st->function_names.end()) ? std::string() : it->second;
      auto mit = st->function_names_mangled.find(m.function_id);
      const std::string mname =
          (mit == st->function_names_mangled.end()) ? std::string() : mit->second;
      if (!first) out << ",\n";
      first = false;
      out << "    {\"bb_id\": " << m.bb_id
          << ", \"function_id\": " << m.function_id
          << ", \"function_name\": \"" << json_escape(fname) << "\""
          << ", \"function_name_mangled\": \"" << json_escape(mname) << "\""
          << ", \"bb_index\": " << m.bb_index
          << ", \"entry_pc\": " << m.entry_pc
          << ", \"n_instrs\": " << m.n_instrs
          << ", \"exec_count\": " << c
          << "}";
    }
  }
  out << "\n  ],\n";

  out << "  \"branch_sites\": [\n";
  first = true;
  if (want_branch) {
    for (const auto& s : st->branch_sites) {
      if (s.site_id >= branch_cap) continue;
      const uint64_t c = b_exec[s.site_id];
      if (c == 0) continue;
      auto it = st->function_names.find(s.function_id);
      const std::string fname = (it == st->function_names.end()) ? std::string() : it->second;
      auto mit = st->function_names_mangled.find(s.function_id);
      const std::string mname =
          (mit == st->function_names_mangled.end()) ? std::string() : mit->second;
      if (!first) out << ",\n";
      first = false;
      out << "    {\"site_id\": " << s.site_id
          << ", \"function_id\": " << s.function_id
          << ", \"function_name\": \"" << json_escape(fname) << "\""
          << ", \"function_name_mangled\": \"" << json_escape(mname) << "\""
          << ", \"pc_offset\": " << s.pc_offset
          << ", \"opcode\": \"" << json_escape(s.opcode_short) << "\""
          << ", \"exec_count\": " << c
          << ", \"taken_warp\": " << b_taken_warp[s.site_id]
          << ", \"fallthrough_warp\": " << b_fall_warp[s.site_id]
          << ", \"taken_lanes\": " << b_taken_lanes[s.site_id]
          << ", \"fallthrough_lanes\": " << b_fall_lanes[s.site_id]
          << "}";
    }
  }
  out << "\n  ]\n}\n";
  return path;
}

static std::string write_mem_trace(const CtxState* st) {
  if (!g_enable_trace || !st->buffers.d_trace || !st->buffers.d_trace_count) return {};
  uint32_t count = 0;
  cuda_check(cudaMemcpy(&count, st->buffers.d_trace_count, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost),
             "copy trace count");
  if (count == 0) return {};
  if (count > g_trace_cap) count = g_trace_cap;

  std::vector<MemTraceEntry> entries(count);
  cuda_check(cudaMemcpy(entries.data(), st->buffers.d_trace,
                        sizeof(MemTraceEntry) * count, cudaMemcpyDeviceToHost),
             "copy trace entries");

  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string path = g_trace_path + "/mem_trace_" +
                           safe + "_" + std::to_string(g_kernel_id) + ".jsonl";
  std::ofstream out(path);
  for (const auto& e : entries) {
    const bool is_load = (e.flags & 1u) != 0;
    const bool is_store = (e.flags & 2u) != 0;
    const bool is_global = (e.flags & 4u) != 0;
    const bool is_shared = (e.flags & 8u) != 0;
    const bool is_local = (e.flags & 16u) != 0;
    const char* space =
        is_local ? "local" : (is_shared ? "shared" : (is_global ? "global" : "other"));
    uint32_t space_id = is_local ? 3u : (is_shared ? 2u : (is_global ? 1u : 0u));
    out << "{\"kernel_id\":" << g_kernel_id
        << ",\"pc_offset\":" << e.pc
        << ",\"region\":" << e.region
        << ",\"cta\":" << e.cta_linear
        << ",\"warp\":" << e.warp_id
        << ",\"active_mask\":" << e.active_mask
        << ",\"access_size\":" << e.access_size
        << ",\"is_load\":" << (is_load ? 1 : 0)
        << ",\"is_store\":" << (is_store ? 1 : 0)
        << ",\"space\":\"" << space << "\""
        << ",\"space_id\":" << space_id
        << ",\"flags\":" << e.flags
        << ",\"addrs\":[";
    for (int i = 0; i < 32; ++i) {
      out << e.addrs[i];
      if (i + 1 != 32) out << ",";
    }
    out << "]}\n";
  }
  return path;
}

static void write_summary(const CtxState* st, const std::string& stats_path,
                          const std::string& pcmap_path,
                          const std::string& trace_path,
                          const std::string& hotspots_path,
                          const std::string& sass_all_path,
                          const std::string& sass_regions_dir,
                          const std::string& nvdisasm_sass_path,
                          const std::string& ptx_all_path,
                          const std::string& ptx_regions_dir) {
  ensure_dir(g_trace_path);
  const std::string safe = sanitize_name(st->kernel_name);
  const std::string path = g_trace_path + "/summary_" +
                           safe + "_" + std::to_string(g_kernel_id) + ".txt";
  std::ofstream out(path);
  out << "kernel=" << st->kernel_name << "\n";
  out << "kernel_id=" << g_kernel_id << "\n";
  out << "kernel_addr=" << st->kernel_addr << "\n";
  out << "cubin_crc=" << st->cubin_crc << "\n";
  out << "kernel_nregs=" << st->kernel_nregs << "\n";
  out << "kernel_local_size_bytes=" << st->kernel_local_size_bytes << "\n";
  out << "mode=" << g_mode << "\n";
  out << "sample_cta=" << g_sample_cta << "\n";
  out << "sample_warp=" << g_sample_warp << "\n";
  out << "sample_mem_every_n=" << g_sample_mem_every_n << "\n";
  out << "reweight_mem_exec=" << (g_reweight_mem_exec ? 1 : 0) << "\n";
  out << "reweight_mem_pattern=" << (g_reweight_mem_pattern ? 1 : 0) << "\n";
  out << "enable_inst_pipe=" << (g_enable_inst_pipe ? 1 : 0) << "\n";
  out << "enable_bb_hot=" << (g_enable_bb_hot ? 1 : 0) << "\n";
  out << "bb_cap=" << g_bb_cap << "\n";
  out << "target_region=" << g_target_region << "\n";
  out << "iter_begin=" << g_iter_begin << "\n";
  out << "iter_end=" << g_iter_end << "\n";
  out << "region_stats=" << stats_path << "\n";
  if (!pcmap_path.empty()) out << "pc2region=" << pcmap_path << "\n";
  if (!trace_path.empty()) out << "mem_trace=" << trace_path << "\n";
  if (!hotspots_path.empty()) out << "hotspots=" << hotspots_path << "\n";
  if (!sass_all_path.empty()) out << "sass_all=" << sass_all_path << "\n";
  if (!sass_regions_dir.empty()) out << "sass_regions_dir=" << sass_regions_dir << "\n";
  if (!nvdisasm_sass_path.empty()) out << "nvdisasm_sass=" << nvdisasm_sass_path << "\n";
  if (!ptx_all_path.empty()) out << "ptx_all=" << ptx_all_path << "\n";
  if (!ptx_regions_dir.empty()) out << "ptx_regions_dir=" << ptx_regions_dir << "\n";
}

static void instrument_function_if_needed(CUcontext ctx, CtxState* st, CUfunction func) {
  if (!st) return;
  std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
  related.push_back(func);

  // Marker update can be inserted in two ways:
  //  (1) instrument the marker CUfunction body (when it exists as a separate CUfunction)
  //  (2) instrument the CALL site by scanning SASS (needed when nvcc emits marker as a
  //      local subroutine inside the kernel, not visible as a CUfunction).
  //
  // If we do both, we will double-push/pop and corrupt the region stack.
  bool have_marker_cufunc = false;
  for (auto f : related) {
    const char* fname = nvbit_get_func_name(ctx, f);
    if (!fname) continue;
    uint32_t region_id = 0;
    bool is_push = false;
    if (parse_marker(fname, &region_id, &is_push)) {
      have_marker_cufunc = true;
      break;
    }
  }

  auto& instrumented = g_instrumented[ctx];
  for (auto f : related) {
    if (!instrumented.insert(f).second) continue;

    const char* fname = nvbit_get_func_name(ctx, f);
    if (!fname) continue;

    uint32_t func_id = 0;
    {
      auto it = st->function_ids.find(f);
      if (it == st->function_ids.end()) {
        func_id = st->next_function_id++;
        st->function_ids.emplace(f, func_id);
        st->function_by_id.emplace(func_id, f);
        st->function_names.emplace(func_id, std::string(fname));
        const char* mangled = nullptr;
        if (cuFuncGetName(&mangled, f) == CUDA_SUCCESS && mangled) {
          st->function_names_mangled.emplace(func_id, std::string(mangled));
        } else {
          st->function_names_mangled.emplace(func_id, std::string());
        }
      } else {
        func_id = it->second;
        // Best-effort: keep reverse map populated (for SASS meta/lineinfo emission).
        if (st->function_by_id.find(func_id) == st->function_by_id.end()) {
          st->function_by_id.emplace(func_id, f);
        }
      }
    }

    uint32_t region_id = 0;
    bool is_push = false;
    if (parse_marker(fname, &region_id, &is_push)) {
      if (g_verbose >= 2) {
        std::fprintf(stderr, "NVBIT: marker-func %s rid=%u op=%s\n",
                     fname, region_id, is_push ? "push" : "pop");
        std::fflush(stderr);
      }
      const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
      maybe_cache_sass_listing(st, func_id, instrs);
      if (!instrs.empty()) {
        Instr* first = instrs[0];
        nvbit_insert_call(first, "ikp_nvbit_region_update", IPOINT_BEFORE);
        nvbit_add_call_arg_const_val32(first, is_push ? 1u : 0u);
        nvbit_add_call_arg_const_val32(first, region_id);
        nvbit_add_call_arg_launch_val64(first, 0);
      }
      continue;
    }

    if (std::strncmp(fname, "ikp_nvbit_", 9) == 0 ||
        std::strcmp(fname, "gen_mref_addr") == 0) {
      continue;
    }

    if (!g_instrument_related && f != func) {
      continue;
    }

    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
    maybe_cache_sass_listing(st, func_id, instrs);
    size_t mem_instrs = 0;
    size_t mem_mrefs = 0;
    size_t marker_sites = 0;

    auto add_pred_args = [&](Instr* i) {
      if (!i->hasPred()) {
        nvbit_add_call_arg_const_val32(i, 1);
        nvbit_add_call_arg_const_val32(i, 0);
        return;
      }
      // predicate value (per-thread) + negation flag
      if (i->isPredUniform()) {
        nvbit_add_call_arg_upred_val_at(i, i->getPredNum());
      } else {
        nvbit_add_call_arg_pred_val_at(i, i->getPredNum());
      }
      nvbit_add_call_arg_const_val32(i, i->isPredNeg() ? 1u : 0u);
    };

    if (g_enable_bb_count) {
      const CFG_t& cfg = nvbit_get_CFG(ctx, f);
      if (!cfg.is_degenerate) {
        for (const auto* bb : cfg.bbs) {
          if (!bb || bb->instrs.empty()) continue;
          Instr* first = bb->instrs[0];
          nvbit_insert_call(first, "ikp_nvbit_bb_exec", IPOINT_BEFORE);
          add_pred_args(first);
          nvbit_add_call_arg_launch_val64(first, 0);
        }
      } else if (g_verbose >= 2) {
        std::fprintf(stderr, "NVBIT: CFG degenerate for %s, skip bb_exec\n", fname);
        std::fflush(stderr);
      }
    }

    if (g_enable_bb_hot) {
      const CFG_t& cfg = nvbit_get_CFG(ctx, f);
      if (!cfg.is_degenerate) {
        for (size_t bb_index = 0; bb_index < cfg.bbs.size(); ++bb_index) {
          const auto* bb = cfg.bbs[bb_index];
          if (!bb || bb->instrs.empty()) continue;
          Instr* first = bb->instrs[0];
          const uint32_t bb_id = st->next_bb_id++;
          // Best-effort: cap bb_id to bb_cap to avoid OOB.
          if (bb_id < g_bb_cap) {
            CtxState::BbMeta meta;
            meta.bb_id = bb_id;
            meta.function_id = func_id;
            meta.bb_index = static_cast<uint32_t>(bb_index);
            meta.entry_pc = first->getOffset();
            meta.n_instrs = static_cast<uint32_t>(bb->instrs.size());
            st->bb_meta.emplace_back(meta);

            nvbit_insert_call(first, "ikp_nvbit_bb_hot", IPOINT_BEFORE);
            nvbit_add_call_arg_const_val32(first, bb_id);
            nvbit_add_call_arg_launch_val64(first, 0);
          }
        }
      } else if (g_verbose >= 2) {
        std::fprintf(stderr, "NVBIT: CFG degenerate for %s, skip bb_hot\n", fname);
        std::fflush(stderr);
      }
    }

    for (auto instr : instrs) {
      // Marker handling: in modern nvcc, marker "functions" may be emitted as local
      // subroutines inside the kernel (e.g., _Z...$ikp_nvbit_region_push_3), thus not
      // visible as a separate CUfunction. Detect the CALL instruction by its SASS
      // and update region stack at the call site.
      uint32_t marker_rid = 0;
      bool marker_is_push = false;
      if (!have_marker_cufunc &&
          parse_marker_in_sass(instr->getSass(), &marker_rid, &marker_is_push)) {
        marker_sites++;
        nvbit_insert_call(instr, "ikp_nvbit_region_update", IPOINT_AFTER);
        nvbit_add_call_arg_const_val32(instr, marker_is_push ? 1u : 0u);
        nvbit_add_call_arg_const_val32(instr, marker_rid);
        nvbit_add_call_arg_launch_val64(instr, 0);
      }

      if (g_enable_inst_exec) {
        uint32_t class_id = classify_instr(instr);
        uint32_t pipe_id = classify_pipe(instr);
        nvbit_insert_call(instr, "ikp_nvbit_inst_exec", IPOINT_BEFORE);
        add_pred_args(instr);
        nvbit_add_call_arg_const_val32(instr, class_id);
        nvbit_add_call_arg_const_val32(instr, pipe_id);
        nvbit_add_call_arg_launch_val64(instr, 0);
      }

      if (g_enable_branch_div) {
        const char* op_short = instr->getOpcodeShort();
        if (is_branch_opcode(op_short)) {
          nvbit_insert_call(instr, "ikp_nvbit_branch_div", IPOINT_BEFORE);
          add_pred_args(instr);
          nvbit_add_call_arg_launch_val64(instr, 0);
        }
      }

    if (g_enable_branch_sites) {
      const char* op_short = instr->getOpcodeShort();
      if (is_branch_opcode(op_short)) {
        const uint32_t site_id = st->next_branch_site_id++;
        if (site_id < g_branch_site_cap) {
          CtxState::BranchSiteMeta meta;
          meta.site_id = site_id;
          meta.function_id = func_id;
          meta.pc_offset = instr->getOffset();
          meta.opcode_short = op_short ? std::string(op_short) : std::string();
          st->branch_sites.emplace_back(std::move(meta));

          nvbit_insert_call(instr, "ikp_nvbit_branch_site", IPOINT_BEFORE);
          add_pred_args(instr);
          nvbit_add_call_arg_const_val32(instr, site_id);
          nvbit_add_call_arg_launch_val64(instr, 0);
        }
      }
    }

      if (g_mode & kModePcMap) {
        nvbit_insert_call(instr, "ikp_nvbit_pcmap_record", IPOINT_BEFORE);
        nvbit_add_call_arg_const_val32(instr, func_id);
        nvbit_add_call_arg_const_val64(instr, instr->getOffset());
        nvbit_add_call_arg_launch_val64(instr, 0);
      }

      if (g_enable_mem_exec) {
        if (instr->isLoad() || instr->isStore()) {
          int mem_space = 0;
          const auto mem_type = instr->getMemorySpace();
          if (mem_type == InstrType::MemorySpace::GLOBAL ||
              mem_type == InstrType::MemorySpace::GENERIC ||
              mem_type == InstrType::MemorySpace::GLOBAL_TO_SHARED) {
            mem_space = 1;
          } else if (mem_type == InstrType::MemorySpace::SHARED ||
                     mem_type == InstrType::MemorySpace::DISTRIBUTED_SHARED) {
            mem_space = 2;
          } else if (mem_type == InstrType::MemorySpace::LOCAL) {
            mem_space = 3;
          }

          if (mem_space != 0) {
            mem_instrs++;
            nvbit_insert_call(instr, "ikp_nvbit_mem_exec", IPOINT_BEFORE);
            add_pred_args(instr);
            nvbit_add_call_arg_const_val32(instr, mem_space);
            nvbit_add_call_arg_const_val32(instr, instr->isLoad() ? 1u : 0u);
            nvbit_add_call_arg_const_val32(instr, instr->isStore() ? 1u : 0u);
            nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(instr->getSize()));
            nvbit_add_call_arg_const_val64(instr, instr->getOffset());
            nvbit_add_call_arg_launch_val64(instr, 0);

            if (g_enable_trace || g_enable_mem_pattern) {
              int mref_idx = 0;
              for (int i = 0; i < instr->getNumOperands(); ++i) {
                const InstrType::operand_t* op = instr->getOperand(i);
                if (op->type == InstrType::OperandType::MREF) {
                  mem_mrefs++;
                  if (g_enable_mem_pattern && mem_space != 3) {
                    nvbit_insert_call(instr, "ikp_nvbit_mem_pattern", IPOINT_BEFORE);
                    add_pred_args(instr);
                    nvbit_add_call_arg_const_val32(instr, mem_space);
                    nvbit_add_call_arg_const_val32(instr, instr->isLoad() ? 1u : 0u);
                    nvbit_add_call_arg_const_val32(instr, instr->isStore() ? 1u : 0u);
                    nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(instr->getSize()));
                    nvbit_add_call_arg_const_val64(instr, instr->getOffset());
                    nvbit_add_call_arg_mref_addr64(instr, mref_idx);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                  }
                  if (g_enable_trace) {
                    nvbit_insert_call(instr, "ikp_nvbit_mem_trace", IPOINT_BEFORE);
                    add_pred_args(instr);
                    nvbit_add_call_arg_const_val32(instr, mem_space);
                    nvbit_add_call_arg_const_val32(instr, instr->isLoad() ? 1u : 0u);
                    nvbit_add_call_arg_const_val32(instr, instr->isStore() ? 1u : 0u);
                    nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(instr->getSize()));
                    nvbit_add_call_arg_const_val64(instr, instr->getOffset());
                    nvbit_add_call_arg_mref_addr64(instr, mref_idx);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                  }
                  mref_idx++;
                }
              }
            }
          }
        }
      }
    }

    if (g_verbose >= 2) {
      std::fprintf(stderr,
                   "NVBIT: instrumented %s instrs=%zu mem_instrs=%zu mrefs=%zu marker_sites=%zu\n",
                   fname, instrs.size(), mem_instrs, mem_mrefs, marker_sites);
      std::fflush(stderr);
    }
  }
}

static void record_last_launch_functions(CUcontext ctx, CtxState* st, CUfunction func) {
  if (!st) return;
  st->last_launch_function_ids.clear();
  std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
  related.push_back(func);
  st->last_launch_function_ids.reserve(related.size());
  for (auto f : related) {
    auto it = st->function_ids.find(f);
    if (it != st->function_ids.end()) st->last_launch_function_ids.push_back(it->second);
  }
}

}  // namespace

extern "C" void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

  g_enabled = (get_env_u32("IKP_NVBIT_ENABLE", 0) != 0);
  g_kernel_regex = get_env_str("IKP_NVBIT_KERNEL_REGEX", "");
  std::string mode_str = get_env_str("IKP_NVBIT_MODE", "instmix");
  g_trace_path = get_env_str("IKP_NVBIT_TRACE_PATH", "./nvbit_trace");
  g_keep_cubin = (get_env_u32("IKP_NVBIT_KEEP_CUBIN", 0) != 0);
  g_dump_sass = (get_env_u32("IKP_NVBIT_DUMP_SASS", 1) != 0);
  g_dump_sass_by_region = (get_env_u32("IKP_NVBIT_DUMP_SASS_BY_REGION", 1) != 0);
  g_dump_sass_meta = (get_env_u32("IKP_NVBIT_DUMP_SASS_META", 0) != 0);
  g_dump_sass_lineinfo = (get_env_u32("IKP_NVBIT_DUMP_SASS_LINEINFO", 0) != 0);
  g_dump_nvdisasm_sass = (get_env_u32("IKP_NVBIT_DUMP_NVDISASM_SASS", 0) != 0);
  g_dump_ptx = (get_env_u32("IKP_NVBIT_DUMP_PTX", 0) != 0);
  g_dump_ptx_by_region = (get_env_u32("IKP_NVBIT_DUMP_PTX_BY_REGION", 0) != 0);
  g_dump_ptx_lineinfo = (get_env_u32("IKP_NVBIT_DUMP_PTX_LINEINFO", 1) != 0);

  g_max_regions = get_env_u32("IKP_NVBIT_MAX_REGIONS", 128);
  g_max_depth = get_env_u32("IKP_NVBIT_MAX_DEPTH", 16);
  g_pcmap_cap = get_env_u32("IKP_NVBIT_PCMAP_CAP", 1u << 20);
  g_trace_cap = get_env_u32("IKP_NVBIT_TRACE_CAP", 1u << 16);
  g_max_warps = get_env_u32("IKP_NVBIT_MAX_WARPS", 1u << 18);

  g_sample_cta = get_env_u32("IKP_NVBIT_SAMPLE_CTA", kIkpNvbitSampleAll);
  g_sample_warp = get_env_u32("IKP_NVBIT_SAMPLE_WARP", kIkpNvbitSampleAll);
  g_sample_mem_every_n = get_env_u32("IKP_NVBIT_SAMPLE_MEM_EVERY_N", 1);
  g_target_region = get_env_u32("IKP_NVBIT_TARGET_REGION", kIkpNvbitAnyRegion);
  g_iter_begin = get_env_u32("IKP_NVBIT_ITER_BEGIN", 0);
  g_iter_end = get_env_u32("IKP_NVBIT_ITER_END", 0xFFFFFFFFu);

  g_verbose = get_env_u32("IKP_NVBIT_VERBOSE", 0);
  g_instrument_related = (get_env_u32("IKP_NVBIT_INSTRUMENT_RELATED", 0) != 0);
  g_reweight_mem_exec = (get_env_u32("IKP_NVBIT_REWEIGHT_MEM_EXEC", 1) != 0);
  g_reweight_mem_pattern = (get_env_u32("IKP_NVBIT_REWEIGHT_MEM_PATTERN", 1) != 0);
  g_enable_bb_hot = (get_env_u32("IKP_NVBIT_ENABLE_BB_HOT", 0) != 0);
  g_bb_cap = get_env_u32("IKP_NVBIT_BB_CAP", 1u << 16);
  g_enable_branch_sites = (get_env_u32("IKP_NVBIT_ENABLE_BRANCH_SITES", 0) != 0);
  g_branch_site_cap = get_env_u32("IKP_NVBIT_BRANCH_SITE_CAP", 1u << 16);

  g_mode = 0;
  g_enable_inst_exec = false;
  g_enable_mem_exec = false;
  g_enable_trace = false;
  g_enable_inst_class = (get_env_u32("IKP_NVBIT_ENABLE_INST_CLASS", 1) != 0);
  g_enable_inst_pipe = (get_env_u32("IKP_NVBIT_ENABLE_INST_PIPE", 0) != 0);
  g_enable_bb_count = (get_env_u32("IKP_NVBIT_ENABLE_BB", 1) != 0);
  g_enable_branch_div = (get_env_u32("IKP_NVBIT_ENABLE_BRANCH_DIV", 1) != 0);
  g_enable_mem_pattern =
      (get_env_u32("IKP_NVBIT_ENABLE_MEM_PATTERN",
                   (mode_str.find("memtrace") != std::string::npos ||
                    mode_str == "all") ? 1u : 0u) != 0);
  g_gmem_set_bins = get_env_u32("IKP_NVBIT_GMEM_SET_BINS", 0);

  if (mode_str == "all") {
    g_mode = kModePcMap;
    g_enable_inst_exec = true;
    g_enable_mem_exec = true;
    g_enable_trace = true;
  } else {
    if (mode_str.find("instmix") != std::string::npos) {
      g_enable_inst_exec = true;
      g_enable_mem_exec = true;
    }
    if (mode_str.find("memtrace") != std::string::npos) {
      g_enable_mem_exec = true;
      g_enable_trace = true;
    }
    if (mode_str.find("pcmap") != std::string::npos) {
      g_mode |= kModePcMap;
    }
  }
  // If we want per-region SASS slices, we must collect pc->region mapping.
  if (g_dump_sass && g_dump_sass_by_region) {
    g_mode |= kModePcMap;
  }
  // Per-region PTX slicing also needs pc->region mapping.
  if (g_dump_ptx_by_region) {
    g_mode |= kModePcMap;
  }
  if (!g_enable_inst_exec && !g_enable_mem_exec && g_mode == 0) {
    g_enable_inst_exec = true;
    g_enable_mem_exec = true;
  }

  // Optional override
  if (std::getenv("IKP_NVBIT_TRACE_ADDR")) {
    g_enable_trace = (get_env_u32("IKP_NVBIT_TRACE_ADDR", 0) != 0);
  }

  if (!g_kernel_regex.empty()) {
    g_use_regex = true;
    g_kernel_re = std::regex(g_kernel_regex);
  }

  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&g_mutex, &attr);

  std::printf("IKP_NVBIT_ENABLE=%u\n", g_enabled ? 1u : 0u);
  std::printf("IKP_NVBIT_MODE=%s\n", mode_str.c_str());
  if (g_use_regex) std::printf("IKP_NVBIT_KERNEL_REGEX=%s\n", g_kernel_regex.c_str());
  std::printf("IKP_NVBIT_MAX_REGIONS=%u\n", g_max_regions);
  std::printf("IKP_NVBIT_MAX_DEPTH=%u\n", g_max_depth);
  std::printf("IKP_NVBIT_PCMAP_CAP=%u\n", g_pcmap_cap);
  std::printf("IKP_NVBIT_TRACE_CAP=%u\n", g_trace_cap);
  std::printf("IKP_NVBIT_MAX_WARPS=%u\n", g_max_warps);
  std::printf("IKP_NVBIT_SAMPLE_CTA=%u\n", g_sample_cta);
  std::printf("IKP_NVBIT_SAMPLE_WARP=%u\n", g_sample_warp);
  std::printf("IKP_NVBIT_SAMPLE_MEM_EVERY_N=%u\n", g_sample_mem_every_n);
  std::printf("IKP_NVBIT_TARGET_REGION=%u\n", g_target_region);
  std::printf("IKP_NVBIT_ITER_BEGIN=%u\n", g_iter_begin);
  std::printf("IKP_NVBIT_ITER_END=%u\n", g_iter_end);
  std::printf("IKP_NVBIT_TRACE_ADDR=%u\n", g_enable_trace ? 1u : 0u);
  std::printf("IKP_NVBIT_ENABLE_INST_CLASS=%u\n", g_enable_inst_class ? 1u : 0u);
  std::printf("IKP_NVBIT_ENABLE_INST_PIPE=%u\n", g_enable_inst_pipe ? 1u : 0u);
  std::printf("IKP_NVBIT_ENABLE_BB=%u\n", g_enable_bb_count ? 1u : 0u);
  std::printf("IKP_NVBIT_ENABLE_BRANCH_DIV=%u\n", g_enable_branch_div ? 1u : 0u);
  std::printf("IKP_NVBIT_ENABLE_MEM_PATTERN=%u\n", g_enable_mem_pattern ? 1u : 0u);
  std::printf("IKP_NVBIT_GMEM_SET_BINS=%u\n", g_gmem_set_bins);
  std::printf("IKP_NVBIT_INSTRUMENT_RELATED=%u\n", g_instrument_related ? 1u : 0u);
  std::printf("IKP_NVBIT_REWEIGHT_MEM_EXEC=%u\n", g_reweight_mem_exec ? 1u : 0u);
  std::printf("IKP_NVBIT_REWEIGHT_MEM_PATTERN=%u\n", g_reweight_mem_pattern ? 1u : 0u);
  std::printf("IKP_NVBIT_ENABLE_BB_HOT=%u\n", g_enable_bb_hot ? 1u : 0u);
  std::printf("IKP_NVBIT_BB_CAP=%u\n", g_bb_cap);
  std::printf("IKP_NVBIT_ENABLE_BRANCH_SITES=%u\n", g_enable_branch_sites ? 1u : 0u);
  std::printf("IKP_NVBIT_BRANCH_SITE_CAP=%u\n", g_branch_site_cap);
  std::printf("IKP_NVBIT_VERBOSE=%u\n", g_verbose);
  std::printf("IKP_NVBIT_KEEP_CUBIN=%u\n", g_keep_cubin ? 1u : 0u);
  std::printf("IKP_NVBIT_DUMP_SASS=%u\n", g_dump_sass ? 1u : 0u);
  std::printf("IKP_NVBIT_DUMP_SASS_BY_REGION=%u\n", g_dump_sass_by_region ? 1u : 0u);
  std::printf("IKP_NVBIT_DUMP_SASS_META=%u\n", g_dump_sass_meta ? 1u : 0u);
  std::printf("IKP_NVBIT_DUMP_SASS_LINEINFO=%u\n", g_dump_sass_lineinfo ? 1u : 0u);
  std::printf("IKP_NVBIT_DUMP_NVDISASM_SASS=%u\n", g_dump_nvdisasm_sass ? 1u : 0u);
  std::printf("IKP_NVBIT_DUMP_PTX=%u\n", g_dump_ptx ? 1u : 0u);
  std::printf("IKP_NVBIT_DUMP_PTX_BY_REGION=%u\n", g_dump_ptx_by_region ? 1u : 0u);
  std::printf("IKP_NVBIT_DUMP_PTX_LINEINFO=%u\n", g_dump_ptx_lineinfo ? 1u : 0u);
  std::printf("IKP_NVBIT_TRACE_PATH=%s\n", g_trace_path.c_str());
}

extern "C" void nvbit_at_term() {}

extern "C" void nvbit_at_ctx_init(CUcontext ctx) {
  pthread_mutex_lock(&g_mutex);
  CtxState* st = new CtxState;
  st->id = static_cast<int>(g_ctx_state.size());
  g_ctx_state[ctx] = st;
  pthread_mutex_unlock(&g_mutex);
}

extern "C" void nvbit_tool_init(CUcontext ctx) {
  pthread_mutex_lock(&g_mutex);
  auto it = g_ctx_state.find(ctx);
  if (it != g_ctx_state.end()) {
    init_context_buffers(&it->second->buffers);
  }
  pthread_mutex_unlock(&g_mutex);
}

extern "C" void nvbit_at_ctx_term(CUcontext ctx) {
  pthread_mutex_lock(&g_mutex);
  auto it = g_ctx_state.find(ctx);
  if (it != g_ctx_state.end()) {
    // NVBit 1.7+ limitation: no cudaFree here.
    delete it->second;
    g_ctx_state.erase(it);
  }
  g_instrumented.erase(ctx);
  pthread_mutex_unlock(&g_mutex);
}

static void extract_launch_dims(nvbit_api_cuda_t cbid, void* params,
                                CUfunction* func_out,
                                uint32_t* grid_x, uint32_t* grid_y, uint32_t* grid_z,
                                uint32_t* block_x, uint32_t* block_y, uint32_t* block_z) {
  if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
    auto* p = reinterpret_cast<cuLaunchKernelEx_params*>(params);
    *func_out = p->f;
    *grid_x = p->config->gridDimX;
    *grid_y = p->config->gridDimY;
    *grid_z = p->config->gridDimZ;
    *block_x = p->config->blockDimX;
    *block_y = p->config->blockDimY;
    *block_z = p->config->blockDimZ;
  } else {
    auto* p = reinterpret_cast<cuLaunchKernel_params*>(params);
    *func_out = p->f;
    *grid_x = p->gridDimX;
    *grid_y = p->gridDimY;
    *grid_z = p->gridDimZ;
    *block_x = p->blockDimX;
    *block_y = p->blockDimY;
    *block_z = p->blockDimZ;
  }
}

extern "C" void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                                   const char* /*name*/, void* params, CUresult* /*pStatus*/) {
  if (!g_enabled) return;
  if (tls_skip_callback) return;
  tls_skip_callback = true;

  if (cbid != API_CUDA_cuLaunchKernel && cbid != API_CUDA_cuLaunchKernel_ptsz &&
      cbid != API_CUDA_cuLaunchKernelEx && cbid != API_CUDA_cuLaunchKernelEx_ptsz) {
    tls_skip_callback = false;
    return;
  }

  pthread_mutex_lock(&g_mutex);
  auto it = g_ctx_state.find(ctx);
  if (it == g_ctx_state.end()) {
    pthread_mutex_unlock(&g_mutex);
    tls_skip_callback = false;
    return;
  }
  CtxState* st = it->second;

  if (!is_exit) {
    CUfunction func = nullptr;
    uint32_t gx = 0, gy = 0, gz = 0, bx = 0, by = 0, bz = 0;
    extract_launch_dims(cbid, params, &func, &gx, &gy, &gz, &bx, &by, &bz);
    const char* fname = nvbit_get_func_name(ctx, func);
    if (!fname || !match_kernel(fname)) {
      nvbit_enable_instrumented(ctx, func, false);
      pthread_mutex_unlock(&g_mutex);
      tls_skip_callback = false;
      return;
    }

    instrument_function_if_needed(ctx, st, func);
    record_last_launch_functions(ctx, st, func);
    init_context_buffers(&st->buffers);

    const uint32_t threads_per_block = bx * by * bz;
    const uint32_t warps_per_block = (threads_per_block + 31u) / 32u;
    const uint64_t num_ctas64 = static_cast<uint64_t>(gx) *
                               static_cast<uint64_t>(gy) *
                               static_cast<uint64_t>(gz);
    const uint64_t total_warps64 = num_ctas64 * static_cast<uint64_t>(warps_per_block);
    const uint32_t active_warps =
        (total_warps64 > static_cast<uint64_t>(g_max_warps))
            ? g_max_warps
            : static_cast<uint32_t>(total_warps64);
    reset_kernel_buffers(&st->buffers, active_warps);
    // Ensure resets are ordered w.r.t. non-default streams (PTDS-safe).
    cuda_check(cudaDeviceSynchronize(), "sync after reset");

    st->kernel_name = fname;
    st->kernel_func = func;
    st->kernel_addr = nvbit_get_func_addr(ctx, func);
    st->kernel_nregs = 0;
    st->kernel_local_size_bytes = 0;
    st->cubin_crc = 0;
    st->cubin_crc_computed = false;
    // Static kernel attributes (useful for spill diagnosis).
    (void)cuFuncGetAttribute(&st->kernel_nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
    (void)cuFuncGetAttribute(&st->kernel_local_size_bytes,
                             CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func);
    st->active = true;

    uint64_t launch_data = reinterpret_cast<uint64_t>(st->buffers.params);
    nvbit_set_at_launch(ctx, func, launch_data);
    nvbit_enable_instrumented(ctx, func, true);

    if (g_verbose >= 1) {
      std::fprintf(stderr,
                   "NVBIT: launch %s grid=(%u,%u,%u) block=(%u,%u,%u) warps=%llu (active=%u)\n",
                   fname, gx, gy, gz, bx, by, bz,
                   static_cast<unsigned long long>(total_warps64), active_warps);
      std::fflush(stderr);
    }
  } else {
    if (!st->active) {
      pthread_mutex_unlock(&g_mutex);
      tls_skip_callback = false;
      return;
    }

    if (g_verbose >= 1) {
      std::fprintf(stderr, "NVBIT: sync begin %s\n", st->kernel_name.c_str());
      std::fflush(stderr);
    }
    cuda_check(cudaDeviceSynchronize(), "sync after kernel");
    if (g_verbose >= 1) {
      std::fprintf(stderr, "NVBIT: sync done %s\n", st->kernel_name.c_str());
      std::fflush(stderr);
    }

    const std::string safe = sanitize_name(st->kernel_name);
    const std::string stats_path = g_trace_path + "/region_stats_" +
                                   safe + "_" + std::to_string(g_kernel_id) + ".json";
    const std::string pcmap_path = g_trace_path + "/pc2region_" +
                                   safe + "_" + std::to_string(g_kernel_id) + ".json";
    maybe_compute_cubin_crc(ctx, st);
    write_region_stats(st);
    std::vector<PcAgg> pc_aggs;
    const std::string pcmap_written = write_pcmap(st, &pc_aggs);
    const std::string hotspots_path = write_hotspots(st);
    const std::string trace_path = write_mem_trace(st);

    // If enabled, generate nvdisasm listing first so we can reuse its file:line mapping
    // when emitting `sass_all_*.sass` and `sass_regions_*/region_*.sass`.
    const std::string nvdisasm_sass_path = dump_nvdisasm_sass(ctx, st);
    const std::string sass_all_path = write_sass_all(ctx, st);
    const std::string sass_regions_dir = write_sass_by_region(ctx, st, pc_aggs);

    std::string ptx_all_path;
    std::string ptx_regions_dir;
    if (g_dump_ptx || g_dump_ptx_by_region) {
      // Compute an absolute error file path for shell redirects.
      char cwd_buf[PATH_MAX];
      std::memset(cwd_buf, 0, sizeof(cwd_buf));
      const char* cwd = ::getcwd(cwd_buf, sizeof(cwd_buf) - 1);
      const std::string trace_abs =
          (!g_trace_path.empty() && g_trace_path[0] == '/') ? g_trace_path
                                                            : (cwd ? (std::string(cwd) + "/" + g_trace_path)
                                                                   : g_trace_path);
      const std::string ptx_err_abs =
          trace_abs + "/ptx_dump_" + safe + "_" + std::to_string(g_kernel_id) + ".err";

      char exe_buf[PATH_MAX];
      std::memset(exe_buf, 0, sizeof(exe_buf));
      const ssize_t n = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
      if (n > 0) {
        std::vector<PtxModule> modules;
        if (extract_ptx_from_exe(std::string(exe_buf), ptx_err_abs, &modules)) {
          auto pr = write_ptx_outputs(st, pc_aggs, modules);
          ptx_all_path = pr.first;
          ptx_regions_dir = pr.second;
        }
      }
      struct stat est{};
      if (stat(ptx_err_abs.c_str(), &est) == 0 && est.st_size == 0) {
        std::remove(ptx_err_abs.c_str());
      }
    }

    write_summary(st, stats_path,
                  (g_mode & kModePcMap) ? (pcmap_written.empty() ? pcmap_path : pcmap_written)
                                        : std::string(),
                  trace_path, hotspots_path, sass_all_path, sass_regions_dir,
                  nvdisasm_sass_path, ptx_all_path, ptx_regions_dir);

    st->active = false;
    g_kernel_id++;
  }

  pthread_mutex_unlock(&g_mutex);
  tls_skip_callback = false;
}


