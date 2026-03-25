#include "ikp_cupti_common.h"

#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cuda.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace {

// Some CUPTI versions expose CUpti_ResourceData::contextUid, others may not.
// Use SFINAE to keep builds compatible.
template <typename T>
static auto resource_context_uid(const T* res, int) -> decltype(res->contextUid, uint32_t{0}) {
  return static_cast<uint32_t>(res->contextUid);
}

template <typename T>
static uint32_t resource_context_uid(const T* /*res*/, ...) {
  return 0;
}

struct StallSample {
  uint32_t reason_index = 0;
  uint64_t samples = 0;
};

struct PcRecord {
  uint64_t cubin_crc = 0;
  uint64_t pc_offset = 0;
  uint32_t function_index = 0;
  uint32_t correlation_id = 0;
  uint64_t range_id = 0;
  std::string function_name;
  std::vector<StallSample> stalls;
};

struct RangeRecord {
  uint32_t context_uid = 0;
  uint64_t range_id = 0;
  uint64_t total_samples = 0;
  uint64_t dropped_samples = 0;
  uint64_t non_user_samples = 0;
  size_t total_num_pcs = 0;
  size_t remaining_num_pcs = 0;
  uint8_t hardware_buffer_full = 0;
};

struct Invocation {
  uint64_t launch_seq = 0;
  uint32_t context_uid = 0;
  uint32_t correlation_id = 0;
  uint64_t stream = 0;
  uint32_t grid_x = 0;
  uint32_t grid_y = 0;
  uint32_t grid_z = 0;
  uint32_t block_x = 0;
  uint32_t block_y = 0;
  uint32_t block_z = 0;
  uint32_t shared_mem_bytes = 0;
  std::string kernel_name;
  bool selected = true;
};

struct StallReasonEntry {
  uint32_t index = 0;
  std::string name;
};

struct StallReasonTable {
  uint32_t context_uid = 0;
  std::vector<StallReasonEntry> entries;
};

struct ContextState {
  uint32_t context_uid = 0;
  size_t num_stall_reasons = 0;
  uint32_t* stall_reason_index = nullptr;
  char** stall_reason_names = nullptr;
  CUpti_PCSamplingData pc_data{};
  size_t pc_data_capacity = 0;
  std::mutex drain_mu;
  std::atomic<bool> configured{false};
  std::atomic<bool> destroying{false};
  std::atomic<bool> range_active{false};
  std::atomic<uint64_t> launches_since_drain{0};
  std::atomic<uint64_t> last_drain_steady_ns{0};

  ~ContextState() {
    if (pc_data.pPcData) {
      for (size_t i = 0; i < pc_data_capacity; ++i) {
        free(pc_data.pPcData[i].stallReason);
      }
      free(pc_data.pPcData);
      pc_data.pPcData = nullptr;
    }
    if (stall_reason_names) {
      for (size_t i = 0; i < num_stall_reasons; ++i) {
        free(stall_reason_names[i]);
      }
      free(stall_reason_names);
      stall_reason_names = nullptr;
    }
    free(stall_reason_index);
    stall_reason_index = nullptr;
  }
};

std::mutex g_mutex;
std::unordered_map<CUcontext, std::shared_ptr<ContextState>> g_contexts;
std::vector<PcRecord> g_pc_records;
std::vector<RangeRecord> g_ranges;
std::vector<Invocation> g_invocations;
std::vector<StallReasonEntry> g_stall_reason_table;
std::unordered_map<CUcontext, StallReasonTable> g_stall_reason_tables;
std::unordered_set<uint32_t> g_selected_correlation_ids;
std::unordered_set<uint32_t> g_seen_correlation_ids;
std::vector<std::string> g_warnings;

std::atomic<uint64_t> g_launch_seq{0};
std::atomic<bool> g_initialized{false};

std::atomic<bool> g_enabled{true};
CUpti_PCSamplingCollectionMode g_collection_mode = CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;
uint32_t g_verbose = 0;
uint32_t g_sampling_period = 0;
uint32_t g_worker_sleep_span_ms = 0;
uint32_t g_drain_retry_ms = 0;
uint32_t g_drain_retry_iters = 0;
bool g_use_start_stop = false;
size_t g_scratch_buf_size = 0;
size_t g_hw_buf_size = 0;
size_t g_pc_buffer_records = 5000;
std::string g_output_path = "pcsampling_raw.json";
std::string g_kernel_regex_str;
bool g_has_kernel_regex = false;
std::regex g_kernel_regex;
size_t g_max_records = 0;
static std::atomic<bool> g_warned_overflow{false};
uint64_t g_drain_every_n = 0;
uint64_t g_drain_interval_ms = 50;
bool g_use_kernel_activity = false;
bool g_kernel_activity_enabled = false;
size_t g_activity_buffer_size = 1 << 20;  // 1MB
std::atomic<bool> g_profiler_api_inited{false};

static void die(const char* msg) {
  // Do not abort the target application for profiler failures.
  std::lock_guard<std::mutex> lock(g_mutex);
  g_warnings.emplace_back(std::string("DISABLED: ") + msg);
  g_enabled.store(false, std::memory_order_relaxed);
}

static bool cupti_ok(CUptiResult res, const char* call) {
  if (res == CUPTI_SUCCESS) return true;
  const char* errstr = nullptr;
  (void)cuptiGetResultString(res, &errstr);
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back(std::string("CUPTI ") + call + " failed: " +
                            (errstr ? errstr : "<unknown>") + " (" + std::to_string((int)res) +
                            ")");
    // Hard-disable on common cluster restriction cases. Keep the app running.
    if (res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES || res == CUPTI_ERROR_NOT_SUPPORTED) {
      g_warnings.emplace_back("DISABLED: profiling counters not permitted on this node");
      g_enabled.store(false, std::memory_order_relaxed);
    }
  }
  return false;
}

#define CUPTI_TRY(call) cupti_ok((call), #call)

static void cupti_warn_only(CUptiResult res, const char* call) {
  if (res == CUPTI_SUCCESS) return;
  const char* errstr = nullptr;
  (void)cuptiGetResultString(res, &errstr);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_warnings.emplace_back(std::string("CUPTI ") + call + " failed: " +
                          (errstr ? errstr : "<unknown>") + " (" + std::to_string((int)res) + ")");
}

static void logf(uint32_t level, const char* fmt, ...) {
  if (g_verbose < level) return;
  std::fprintf(stderr, "[ikp_cupti_pcsamp] ");
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
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
          break;
        }
        out.push_back(char(c));
        break;
    }
  }
  return out;
}

static uint64_t now_ns() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

static uint64_t steady_now_ns() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

static void parse_uint64_env_or_warn(const char* name, uint64_t& dst, uint64_t fallback, uint64_t minv = 0,
                                     uint64_t maxv = ULLONG_MAX) {
  uint64_t v = 0;
  const auto st = ikp_cupti::parse_uint64_env_strict(name, v, minv, maxv);
  if (st == ikp_cupti::EnvParseStatus::kOk) {
    dst = v;
  } else if (st == ikp_cupti::EnvParseStatus::kInvalid) {
    dst = fallback;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back(std::string("invalid ") + name + "; using default");
  } else {
    dst = fallback;
  }
}

static bool try_claim_activity_callbacks_owner(const char* tool_name) {
  static constexpr const char* kOwnerEnv = "IKP_CUPTI_ACTIVITY_CALLBACKS_OWNER";
  const char* cur = std::getenv(kOwnerEnv);
  if (!cur || !*cur) {
    std::string val = std::string(tool_name) + ":" + std::to_string((uint64_t)getpid());
    (void)setenv(kOwnerEnv, val.c_str(), /*overwrite=*/0);
    return true;
  }
  const std::string want_prefix = std::string(tool_name) + ":";
  if (std::string(cur).rfind(want_prefix, 0) == 0) return true;
  return false;
}

// Forward decl (kernel activity callbacks are defined before match_kernel()).
static bool match_kernel(const char* name);

template <typename T>
static auto activity_correlation_id(const T* k, int) -> decltype(k->correlationId, uint32_t{0}) {
  return static_cast<uint32_t>(k->correlationId);
}

template <typename T>
static uint32_t activity_correlation_id(const T* /*k*/, ...) {
  return 0;
}

template <typename T>
static auto activity_kernel_name(const T* k, int) -> decltype(k->name, (const char*)nullptr) {
  return k->name;
}

template <typename T>
static const char* activity_kernel_name(const T* /*k*/, ...) {
  return nullptr;
}

static void CUPTIAPI activity_buffer_requested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  if (!g_enabled.load(std::memory_order_relaxed) || !g_kernel_activity_enabled) {
    *buffer = nullptr;
    *size = 0;
    *maxNumRecords = 0;
    return;
  }
  *maxNumRecords = 0;
  *size = g_activity_buffer_size;
  void* ptr = nullptr;
  constexpr size_t kAlign = 8;
  if (posix_memalign(&ptr, kAlign, *size) != 0 || ptr == nullptr) {
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("kernel_activity: posix_memalign failed for activity buffer");
    }
    *buffer = nullptr;
    *size = 0;
    *maxNumRecords = 0;
    return;
  }
  *buffer = reinterpret_cast<uint8_t*>(ptr);
}

static void CUPTIAPI activity_buffer_completed(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
                                               size_t validSize) {
  (void)ctx;
  (void)streamId;
  (void)size;
  if (!buffer || validSize == 0) {
    std::free(buffer);
    return;
  }
  CUpti_Activity* record = nullptr;
  while (true) {
    CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          const auto* k = reinterpret_cast<const CUpti_ActivityKernel9*>(record);
          const uint32_t corr = activity_correlation_id(k, 0);
          const char* name = activity_kernel_name(k, 0);
          if (corr != 0) {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_seen_correlation_ids.insert(corr).second) {
              Invocation inv;
              inv.launch_seq = g_launch_seq.fetch_add(1);
              inv.context_uid = 0;
              inv.correlation_id = corr;
              inv.kernel_name = name ? name : "";
              inv.selected = match_kernel(name);
              g_invocations.push_back(inv);
              if (inv.selected) g_selected_correlation_ids.insert(corr);
            } else {
              if (match_kernel(name)) g_selected_correlation_ids.insert(corr);
            }
          }
          break;
        }
        default:
          break;
      }
      continue;
    }
    if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
    cupti_warn_only(status, "cuptiActivityGetNextRecord");
    break;
  }

  std::free(buffer);
}

static bool match_kernel(const char* name) {
  if (!g_has_kernel_regex) return true;
  if (!name) return false;
  return std::regex_search(name, g_kernel_regex);
}

static const char* collection_mode_string() {
  if (g_collection_mode == CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS) {
    return "continuous";
  }
  if (g_collection_mode == CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED) {
    return "serialized";
  }
  return "invalid";
}

static void init_config() {
  const char* enable = std::getenv("IKP_CUPTI_PCSAMP_ENABLE");
  if (enable && std::atoi(enable) == 0) {
    g_enabled.store(false, std::memory_order_relaxed);
  }

  if (!g_enabled.load(std::memory_order_relaxed)) return;

  g_verbose = (uint32_t)ikp_cupti::parse_uint64_env("IKP_CUPTI_PCSAMP_VERBOSE", 0);
  g_worker_sleep_span_ms = (uint32_t)ikp_cupti::parse_uint64_env("IKP_CUPTI_PCSAMP_WORKER_SLEEP_MS", 0);
  g_drain_retry_ms = (uint32_t)ikp_cupti::parse_uint64_env("IKP_CUPTI_PCSAMP_DRAIN_RETRY_MS", 0);
  g_drain_retry_iters = (uint32_t)ikp_cupti::parse_uint64_env("IKP_CUPTI_PCSAMP_DRAIN_RETRY_ITERS", 0);
  g_use_start_stop = ikp_cupti::parse_bool_env("IKP_CUPTI_PCSAMP_USE_START_STOP", false);

  const char* out = std::getenv("IKP_CUPTI_PCSAMP_OUT");
  if (out && *out) {
    g_output_path = out;
  } else {
    g_output_path = "pcsampling_raw." + std::to_string((uint64_t)getpid()) + ".json";
  }

  if (const char* mode = std::getenv("IKP_CUPTI_PCSAMP_COLLECTION_MODE")) {
    if (std::strcmp(mode, "continuous") == 0 || std::strcmp(mode, "1") == 0) {
      g_collection_mode = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
    } else if (std::strcmp(mode, "serialized") == 0 || std::strcmp(mode, "2") == 0) {
      g_collection_mode = CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;
    } else {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("unknown IKP_CUPTI_PCSAMP_COLLECTION_MODE, using serialized");
    }
  }

  {
    uint64_t v = 0;
    parse_uint64_env_or_warn("IKP_CUPTI_PCSAMP_PERIOD", v, /*fallback=*/0);
    g_sampling_period = (uint32_t)v;
  }
  {
    uint64_t v = 0;
    parse_uint64_env_or_warn("IKP_CUPTI_PCSAMP_SCRATCH_BUF_BYTES", v, /*fallback=*/0);
    g_scratch_buf_size = (size_t)v;
  }
  {
    uint64_t v = 0;
    parse_uint64_env_or_warn("IKP_CUPTI_PCSAMP_HW_BUF_BYTES", v, /*fallback=*/0);
    g_hw_buf_size = (size_t)v;
  }
  {
    uint64_t v = 0;
    parse_uint64_env_or_warn("IKP_CUPTI_PCSAMP_MAX_PCS", v, /*fallback=*/(uint64_t)g_pc_buffer_records, /*minv=*/1,
                             /*maxv=*/(1ull << 30));
    g_pc_buffer_records = (size_t)v;
  }
  {
    uint64_t v = 0;
    parse_uint64_env_or_warn("IKP_CUPTI_PCSAMP_MAX_RECORDS", v, /*fallback=*/0);
    g_max_records = (size_t)v;
  }
  {
    uint64_t v = 0;
    parse_uint64_env_or_warn("IKP_CUPTI_PCSAMP_DRAIN_EVERY_N", v, /*fallback=*/0);
    g_drain_every_n = v;
  }
  {
    uint64_t v = 0;
    parse_uint64_env_or_warn("IKP_CUPTI_PCSAMP_DRAIN_INTERVAL_MS", v, /*fallback=*/g_drain_interval_ms, /*minv=*/0,
                             /*maxv=*/600000);
    g_drain_interval_ms = v;
  }

  if (const char* regex = std::getenv("IKP_CUPTI_PCSAMP_KERNEL_REGEX")) {
    if (regex && *regex) {
      g_kernel_regex_str = regex;
      try {
        g_kernel_regex = std::regex(g_kernel_regex_str, std::regex_constants::optimize);
        g_has_kernel_regex = true;
      } catch (...) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_warnings.emplace_back("invalid IKP_CUPTI_PCSAMP_KERNEL_REGEX, regex disabled");
        g_has_kernel_regex = false;
      }
    }
  }

  // Kernel activity can help build invocations for paths that bypass driver API callbacks (e.g. graphs).
  // Default: enable when a regex is set (so selection/join stays correct). Can be forced on/off.
  const char* ka = std::getenv("IKP_CUPTI_PCSAMP_ENABLE_KERNEL_ACTIVITY");
  if (ka && *ka) {
    g_use_kernel_activity = (std::atoi(ka) != 0);
  } else {
    g_use_kernel_activity = g_has_kernel_regex;
  }
  {
    uint64_t v = 0;
    const auto st = ikp_cupti::parse_uint64_env_strict("IKP_CUPTI_PCSAMP_ACTIVITY_BUFFER_BYTES", v, /*minv=*/4096,
                                                      /*maxv=*/(1ull << 30));
    if (st == ikp_cupti::EnvParseStatus::kOk) {
      g_activity_buffer_size = (size_t)v;
    } else if (st == ikp_cupti::EnvParseStatus::kInvalid) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("invalid IKP_CUPTI_PCSAMP_ACTIVITY_BUFFER_BYTES; using default");
    }
  }

  if (g_collection_mode == CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("continuous mode: correlationId may be 0, join requires filtering");
  }

  logf(1, "init: enabled=1 mode=%s out=%s period=%u pc_buf_records=%zu scratch=%zu hw=%zu", collection_mode_string(),
       g_output_path.c_str(), g_sampling_period, g_pc_buffer_records, g_scratch_buf_size, g_hw_buf_size);
  if (g_has_kernel_regex) logf(1, "init: kernel_regex=%s", g_kernel_regex_str.c_str());
  if (g_worker_sleep_span_ms) logf(1, "init: worker_sleep_ms=%u", g_worker_sleep_span_ms);
  if (g_drain_retry_iters || g_drain_retry_ms) {
    logf(1, "init: drain_retry iters=%u sleep_ms=%u", g_drain_retry_iters, g_drain_retry_ms);
  }
  if (g_use_start_stop) logf(1, "init: start_stop_control=1");
}

static bool should_drain_continuous(ContextState* state) {
  if (!state) return false;
  if (g_collection_mode != CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS) return false;
  if (!g_enabled.load(std::memory_order_relaxed)) return false;
  if (!state->configured.load(std::memory_order_relaxed)) return false;

  bool do_drain = false;
  if (g_drain_every_n) {
    const uint64_t n = state->launches_since_drain.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n % g_drain_every_n) == 0) do_drain = true;
  }
  if (!do_drain && g_drain_interval_ms) {
    const uint64_t now = steady_now_ns();
    const uint64_t last = state->last_drain_steady_ns.load(std::memory_order_relaxed);
    const uint64_t interval_ns = g_drain_interval_ms * 1000000ull;
    if (now > last && (now - last) >= interval_ns) {
      uint64_t expected = last;
      if (state->last_drain_steady_ns.compare_exchange_strong(expected, now, std::memory_order_relaxed)) {
        do_drain = true;
      }
    }
  }
  return do_drain;
}

static void maybe_enable_kernel_activity() {
  if (!g_enabled.load(std::memory_order_relaxed)) return;
  if (!g_use_kernel_activity) return;

  if (!try_claim_activity_callbacks_owner("pcsampling")) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("kernel_activity disabled: CUPTI activity callbacks already owned by another tool");
    return;
  }

  CUptiResult reg = cuptiActivityRegisterCallbacks(activity_buffer_requested, activity_buffer_completed);
  if (reg != CUPTI_SUCCESS) {
    cupti_warn_only(reg, "cuptiActivityRegisterCallbacks(activity_buffer_requested, ...)");
    return;
  }

  CUptiResult ena = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (ena != CUPTI_SUCCESS) {
    cupti_warn_only(ena, "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)");
    ena = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  }
  if (ena != CUPTI_SUCCESS) {
    cupti_warn_only(ena, "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)");
    return;
  }

  g_kernel_activity_enabled = true;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_warnings.emplace_back("kernel_activity enabled: will use kernel activity to improve invocations/join");
}

static void store_stall_reason_table(CUcontext ctx, uint32_t context_uid, size_t count, uint32_t* indices,
                                     char** names) {
  StallReasonTable table;
  table.context_uid = context_uid;
  table.entries.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    StallReasonEntry entry;
    entry.index = indices[i];
    entry.name = names[i] ? names[i] : "";
    table.entries.push_back(std::move(entry));
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  // Backward-compat: keep the legacy flat table as the first-seen table.
  if (g_stall_reason_table.empty()) {
    g_stall_reason_table = table.entries;
  }
  g_stall_reason_tables[ctx] = std::move(table);
}

static void process_pc_sampling_data(ContextState* state) {
  if (!state) return;
  if (state->pc_data.totalNumPcs == 0) return;

  RangeRecord range{};
  range.context_uid = state->context_uid;
  range.range_id = state->pc_data.rangeId;
  range.total_samples = state->pc_data.totalSamples;
  range.dropped_samples = state->pc_data.droppedSamples;
  range.non_user_samples = state->pc_data.nonUsrKernelsTotalSamples;
  range.total_num_pcs = state->pc_data.totalNumPcs;
  range.remaining_num_pcs = state->pc_data.remainingNumPcs;
  range.hardware_buffer_full = state->pc_data.hardwareBufferFull;

  std::vector<PcRecord> local_records;
  local_records.reserve(state->pc_data.totalNumPcs);
  for (size_t i = 0; i < state->pc_data.totalNumPcs; ++i) {
    CUpti_PCSamplingPCData& pc = state->pc_data.pPcData[i];
    PcRecord rec;
    rec.cubin_crc = pc.cubinCrc;
    rec.pc_offset = pc.pcOffset;
    rec.function_index = pc.functionIndex;
    rec.correlation_id = pc.correlationId;
    rec.range_id = state->pc_data.rangeId;
    if (pc.functionName) {
      rec.function_name = pc.functionName;
      free(pc.functionName);
      pc.functionName = nullptr;
    } else {
      rec.function_name.clear();
    }
    for (size_t j = 0; j < pc.stallReasonCount; ++j) {
      if (pc.stallReason[j].samples == 0) continue;
      StallSample sample;
      sample.reason_index = pc.stallReason[j].pcSamplingStallReasonIndex;
      sample.samples = pc.stallReason[j].samples;
      rec.stalls.push_back(sample);
    }
    local_records.push_back(std::move(rec));
  }

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_ranges.push_back(range);
    for (auto& rec : local_records) {
      if (g_max_records && g_pc_records.size() >= g_max_records) {
        if (!g_warned_overflow.exchange(true)) {
          g_warnings.emplace_back(
              "pc_records overflow: dropping further records (set IKP_CUPTI_PCSAMP_MAX_RECORDS=0 to disable cap)");
        }
        break;
      }
      g_pc_records.push_back(std::move(rec));
    }
  }
}

static void drain_pc_sampling(CUcontext ctx, ContextState* state, bool allow_destroying = false) {
  if (!state) return;
  std::lock_guard<std::mutex> lk(state->drain_mu);
  if (!allow_destroying && state->destroying.load(std::memory_order_relaxed)) return;
  if (!state->configured.load(std::memory_order_relaxed)) return;
  if (!g_enabled.load(std::memory_order_relaxed)) return;
  if (g_use_start_stop && state->range_active.load(std::memory_order_relaxed)) return;
  CUpti_PCSamplingGetDataParams get_params = {};
  get_params.size = CUpti_PCSamplingGetDataParamsSize;
  get_params.ctx = ctx;
  get_params.pcSamplingData = (void*)&state->pc_data;

  constexpr int kMaxIters = 64;
  const uint32_t extra_retries = g_drain_retry_iters;
  const uint32_t retry_sleep_ms = g_drain_retry_ms;
  for (int it = 0; it < kMaxIters; ++it) {
    state->pc_data.collectNumPcs = state->pc_data_capacity;

    CUptiResult res = cuptiPCSamplingGetData(&get_params);
    logf(2,
         "drain: ctx_uid=%u it=%d res=%d totalSamples=%u dropped=%u totalNumPcs=%u remaining=%u hwFull=%u",
         state->context_uid, it, (int)res, state->pc_data.totalSamples, state->pc_data.droppedSamples,
         state->pc_data.totalNumPcs, state->pc_data.remainingNumPcs, (uint32_t)state->pc_data.hardwareBufferFull);
    if (res != CUPTI_SUCCESS) {
      if (state->pc_data.hardwareBufferFull) {
        {
          std::lock_guard<std::mutex> lock(g_mutex);
          g_warnings.emplace_back("hardware buffer full, increase HW buffer or sampling period");
        }
        break;
      }
      if (!cupti_ok(res, "cuptiPCSamplingGetData")) {
        break;
      }
    }

    if (state->pc_data.totalNumPcs == 0) {
      if (retry_sleep_ms && it < (int)extra_retries) {
        usleep((useconds_t)retry_sleep_ms * 1000u);
        continue;
      }
      break;
    }

    process_pc_sampling_data(state);

    if (state->pc_data.remainingNumPcs == 0) break;
  }
}

static bool configure_context(CUcontext ctx, ContextState* state) {
  if (!g_enabled.load(std::memory_order_relaxed)) return false;
  logf(1, "configure_context: begin ctx_uid=%u", state ? state->context_uid : 0u);
  CUpti_PCSamplingGetNumStallReasonsParams num_params = {};
  num_params.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
  num_params.ctx = ctx;
  size_t num_stall_reasons = 0;
  num_params.numStallReasons = &num_stall_reasons;
  if (!CUPTI_TRY(cuptiPCSamplingGetNumStallReasons(&num_params))) return false;

  state->num_stall_reasons = num_stall_reasons;
  logf(1, "configure_context: num_stall_reasons=%zu", num_stall_reasons);
  state->stall_reason_index = (uint32_t*)calloc(num_stall_reasons, sizeof(uint32_t));
  state->stall_reason_names = (char**)calloc(num_stall_reasons, sizeof(char*));
  if (!state->stall_reason_index || !state->stall_reason_names) {
    die("stall reason alloc failed");
    return false;
  }
  for (size_t i = 0; i < num_stall_reasons; ++i) {
    state->stall_reason_names[i] = (char*)calloc(CUPTI_STALL_REASON_STRING_SIZE, sizeof(char));
    if (!state->stall_reason_names[i]) {
      die("stall reason name alloc failed");
      return false;
    }
  }

  CUpti_PCSamplingGetStallReasonsParams stall_params = {};
  stall_params.size = CUpti_PCSamplingGetStallReasonsParamsSize;
  stall_params.ctx = ctx;
  stall_params.numStallReasons = num_stall_reasons;
  stall_params.stallReasonIndex = state->stall_reason_index;
  stall_params.stallReasons = state->stall_reason_names;
  if (!CUPTI_TRY(cuptiPCSamplingGetStallReasons(&stall_params))) return false;

  store_stall_reason_table(ctx, state->context_uid, num_stall_reasons, state->stall_reason_index, state->stall_reason_names);

  state->pc_data.size = sizeof(CUpti_PCSamplingData);
  state->pc_data_capacity = g_pc_buffer_records;
  state->pc_data.collectNumPcs = state->pc_data_capacity;
  state->pc_data.pPcData = (CUpti_PCSamplingPCData*)calloc(state->pc_data_capacity, sizeof(CUpti_PCSamplingPCData));
  if (!state->pc_data.pPcData) {
    die("pc sampling buffer alloc failed");
    return false;
  }
  for (size_t i = 0; i < state->pc_data_capacity; ++i) {
    state->pc_data.pPcData[i].stallReason =
        (CUpti_PCSamplingStallReason*)calloc(num_stall_reasons, sizeof(CUpti_PCSamplingStallReason));
    if (!state->pc_data.pPcData[i].stallReason) {
      die("stall reason buffer alloc failed");
      return false;
    }
  }

  std::vector<CUpti_PCSamplingConfigurationInfo> cfg;

  if (g_sampling_period) {
    CUpti_PCSamplingConfigurationInfo period = {};
    period.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    period.attributeData.samplingPeriodData.samplingPeriod = g_sampling_period;
    cfg.push_back(period);
  }

  if (g_worker_sleep_span_ms) {
    CUpti_PCSamplingConfigurationInfo span = {};
    span.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_WORKER_THREAD_PERIODIC_SLEEP_SPAN;
    span.attributeData.workerThreadPeriodicSleepSpanData.workerThreadPeriodicSleepSpan = g_worker_sleep_span_ms;
    cfg.push_back(span);
  }

  if (g_scratch_buf_size) {
    CUpti_PCSamplingConfigurationInfo scratch = {};
    scratch.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    scratch.attributeData.scratchBufferSizeData.scratchBufferSize = g_scratch_buf_size;
    cfg.push_back(scratch);
  }

  if (g_hw_buf_size) {
    CUpti_PCSamplingConfigurationInfo hw = {};
    hw.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    hw.attributeData.hardwareBufferSizeData.hardwareBufferSize = g_hw_buf_size;
    cfg.push_back(hw);
  }

  CUpti_PCSamplingConfigurationInfo collection = {};
  collection.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
  collection.attributeData.collectionModeData.collectionMode = g_collection_mode;
  cfg.push_back(collection);

  if (g_use_start_stop) {
    CUpti_PCSamplingConfigurationInfo ss = {};
    ss.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    ss.attributeData.enableStartStopControlData.enableStartStopControl = 1;
    cfg.push_back(ss);
  }

  CUpti_PCSamplingConfigurationInfo stall = {};
  stall.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
  stall.attributeData.stallReasonData.stallReasonCount = num_stall_reasons;
  stall.attributeData.stallReasonData.pStallReasonIndex = state->stall_reason_index;
  cfg.push_back(stall);

  CUpti_PCSamplingConfigurationInfo data_buf = {};
  data_buf.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  data_buf.attributeData.samplingDataBufferData.samplingDataBuffer = (void*)&state->pc_data;
  cfg.push_back(data_buf);

  CUpti_PCSamplingConfigurationInfo output_format = {};
  output_format.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
  output_format.attributeData.outputDataFormatData.outputDataFormat = CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;
  cfg.push_back(output_format);

  CUpti_PCSamplingConfigurationInfoParams cfg_params = {};
  cfg_params.size = sizeof(CUpti_PCSamplingConfigurationInfoParams);
  cfg_params.ctx = ctx;
  cfg_params.numAttributes = cfg.size();
  cfg_params.pPCSamplingConfigurationInfo = cfg.data();
  if (!CUPTI_TRY(cuptiPCSamplingSetConfigurationAttribute(&cfg_params))) return false;
  logf(1, "configure_context: set_config ok (attrs=%zu)", cfg.size());
  return true;
}

static std::shared_ptr<ContextState> get_or_configure_context(CUcontext ctx, uint32_t context_uid) {
  if (!ctx) return {};
  if (!g_enabled.load(std::memory_order_relaxed)) return {};

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(ctx);
    if (it != g_contexts.end()) return it->second;
  }

  auto state = std::make_shared<ContextState>();
  if (!state) {
    die("context state alloc failed");
    return {};
  }
  state->context_uid = context_uid;
  logf(1, "lazy_context: create ctx_uid=%u", context_uid);

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto [it, inserted] = g_contexts.insert({ctx, state});
    if (!inserted) return it->second;
  }

  CUpti_PCSamplingEnableParams enable_params = {};
  enable_params.size = sizeof(CUpti_PCSamplingEnableParams);
  enable_params.ctx = ctx;
  if (!CUPTI_TRY(cuptiPCSamplingEnable(&enable_params))) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_contexts.erase(ctx);
    return {};
  }
  logf(1, "lazy_context: enable ok ctx_uid=%u", context_uid);
  bool ok = configure_context(ctx, state.get());
  state->configured.store(ok, std::memory_order_relaxed);
  logf(1, "lazy_context: configured=%u ctx_uid=%u", ok ? 1u : 0u, context_uid);
  if (!ok) {
    CUpti_PCSamplingDisableParams disable_params = {};
    disable_params.size = sizeof(CUpti_PCSamplingDisableParams);
    disable_params.ctx = ctx;
    (void)CUPTI_TRY(cuptiPCSamplingDisable(&disable_params));
    std::lock_guard<std::mutex> lock(g_mutex);
    g_contexts.erase(ctx);
    return {};
  }
  return state;
}

static void maybe_start_range(CUcontext ctx, ContextState* state) {
  if (!g_use_start_stop) return;
  if (!state) return;
  if (!state->configured.load(std::memory_order_relaxed)) return;
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  bool expected = false;
  if (!state->range_active.compare_exchange_strong(expected, true, std::memory_order_relaxed)) return;

  CUpti_PCSamplingStartParams sp = {};
  sp.size = CUpti_PCSamplingStartParamsSize;
  sp.ctx = ctx;
  CUptiResult r = cuptiPCSamplingStart(&sp);
  if (!cupti_ok(r, "cuptiPCSamplingStart")) {
    state->range_active.store(false, std::memory_order_relaxed);
    return;
  }
  logf(2, "range: start ctx_uid=%u", state->context_uid);
}

static void maybe_stop_range(CUcontext ctx, ContextState* state) {
  if (!g_use_start_stop) return;
  if (!state) return;
  if (!state->configured.load(std::memory_order_relaxed)) return;
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  bool expected = true;
  if (!state->range_active.compare_exchange_strong(expected, false, std::memory_order_relaxed)) return;

  CUpti_PCSamplingStopParams sp = {};
  sp.size = CUpti_PCSamplingStopParamsSize;
  sp.ctx = ctx;
  CUptiResult r = cuptiPCSamplingStop(&sp);
  (void)cupti_ok(r, "cuptiPCSamplingStop");
  logf(2, "range: stop ctx_uid=%u", state->context_uid);
}

static void handle_kernel_launch(const CUpti_CallbackData* info, CUpti_CallbackId cbid) {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(info->context);
    if (it != g_contexts.end() && it->second && it->second->context_uid == 0) {
      it->second->context_uid = info->contextUid;
    }
    auto tt = g_stall_reason_tables.find(info->context);
    if (tt != g_stall_reason_tables.end() && tt->second.context_uid == 0) {
      tt->second.context_uid = info->contextUid;
    }
  }

  Invocation inv;
  inv.launch_seq = g_launch_seq.fetch_add(1);
  inv.context_uid = info->contextUid;
  inv.correlation_id = info->correlationId;
  inv.kernel_name = info->symbolName ? info->symbolName : "";
  inv.selected = match_kernel(info->symbolName);
  logf(2, "launch_cb: cbid=%u ctx_uid=%u corr=%u selected=%u name=%s", (uint32_t)cbid, info->contextUid,
       info->correlationId, inv.selected ? 1u : 0u, info->symbolName ? info->symbolName : "<null>");

  if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
    const auto* p = (const cuLaunchKernel_params*)info->functionParams;
    inv.grid_x = p->gridDimX;
    inv.grid_y = p->gridDimY;
    inv.grid_z = p->gridDimZ;
    inv.block_x = p->blockDimX;
    inv.block_y = p->blockDimY;
    inv.block_z = p->blockDimZ;
    inv.shared_mem_bytes = p->sharedMemBytes;
    inv.stream = (uint64_t)p->hStream;
  } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
    const auto* p = (const cuLaunchKernel_ptsz_params*)info->functionParams;
    inv.grid_x = p->gridDimX;
    inv.grid_y = p->gridDimY;
    inv.grid_z = p->gridDimZ;
    inv.block_x = p->blockDimX;
    inv.block_y = p->blockDimY;
    inv.block_z = p->blockDimZ;
    inv.shared_mem_bytes = p->sharedMemBytes;
    inv.stream = (uint64_t)p->hStream;
  }

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (inv.correlation_id != 0) {
      if (g_seen_correlation_ids.insert(inv.correlation_id).second) {
        g_invocations.push_back(inv);
      }
      if (inv.selected) g_selected_correlation_ids.insert(inv.correlation_id);
    } else {
      g_invocations.push_back(inv);
    }
  }

  std::shared_ptr<ContextState> state = get_or_configure_context(info->context, info->contextUid);
  if (state) {
    if (g_collection_mode == CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS && should_drain_continuous(state.get())) {
      drain_pc_sampling(info->context, state.get());
    }
  }
}

static bool is_sync_api_name(const char* fn) {
  if (!fn || !*fn) return false;
  if (std::strcmp(fn, "cuCtxSynchronize") == 0) return true;
  if (std::strcmp(fn, "cuStreamSynchronize") == 0) return true;
  if (std::strcmp(fn, "cuStreamSynchronize_ptsz") == 0) return true;
  if (std::strcmp(fn, "cuEventSynchronize") == 0) return true;
  if (std::strcmp(fn, "cudaDeviceSynchronize") == 0) return true;
  if (std::strcmp(fn, "cudaStreamSynchronize") == 0) return true;
  if (std::strcmp(fn, "cudaEventSynchronize") == 0) return true;
  if (std::strcmp(fn, "cudaThreadSynchronize") == 0) return true;  // legacy
  return false;
}

static void handle_sync_point(const CUpti_CallbackData* info) {
  if (!g_enabled.load(std::memory_order_relaxed)) return;
  if (!info) return;
  logf(2, "sync_cb: ctx_uid=%u fn=%s", info->contextUid, info->functionName ? info->functionName : "<null>");
  std::shared_ptr<ContextState> state = get_or_configure_context(info->context, info->contextUid);
  if (!state) return;
  if (g_use_start_stop) maybe_stop_range(info->context, state.get());
  drain_pc_sampling(info->context, state.get());
}

static void write_json() {
  const std::string tmp_path = g_output_path + ".tmp." + std::to_string((uint64_t)getpid());
  std::ofstream out(tmp_path);
  if (!out.is_open()) {
    std::fprintf(stderr, "ikp_cupti_pcsamp: cannot open %s\n", g_output_path.c_str());
    return;
  }

  const uint64_t ts = now_ns();
  out << "{";
  out << "\"tool\":\"ikp_cupti_pcsamp\"";
  out << ",\"version\":1";
  out << ",\"pid\":" << (uint64_t)getpid();
  out << ",\"timestamp_ns\":" << ts;
  out << ",\"collection_mode\":\"" << collection_mode_string() << "\"";
  out << ",\"sampling_period\":" << g_sampling_period;
  out << ",\"scratch_buffer_size\":" << (uint64_t)g_scratch_buf_size;
  out << ",\"hardware_buffer_size\":" << (uint64_t)g_hw_buf_size;
  out << ",\"pc_buffer_records\":" << (uint64_t)g_pc_buffer_records;
  out << ",\"kernel_regex\":\"" << json_escape(g_kernel_regex_str) << "\"";

  std::vector<PcRecord> pc_records;
  std::vector<RangeRecord> ranges;
  std::vector<Invocation> invocations;
  std::vector<StallReasonEntry> stall_reason_table;
  std::unordered_map<CUcontext, StallReasonTable> stall_reason_tables;
  std::unordered_set<uint32_t> selected_corrids;
  std::vector<std::string> warnings;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    pc_records = g_pc_records;
    ranges = g_ranges;
    invocations = g_invocations;
    stall_reason_table = g_stall_reason_table;
    stall_reason_tables = g_stall_reason_tables;
    selected_corrids = g_selected_correlation_ids;
    warnings = g_warnings;
  }

  out << ",\"stall_reason_table\":[";
  for (size_t i = 0; i < stall_reason_table.size(); ++i) {
    if (i) out << ",";
    out << "{\"index\":" << stall_reason_table[i].index << ",\"name\":\"" << json_escape(stall_reason_table[i].name)
        << "\"}";
  }
  out << "]";

  out << ",\"stall_reason_tables\":[";
  bool first_table = true;
  for (const auto& kv : stall_reason_tables) {
    const auto& table = kv.second;
    if (!first_table) out << ",";
    first_table = false;
    out << "{";
    out << "\"context_uid\":" << table.context_uid;
    out << ",\"entries\":[";
    for (size_t i = 0; i < table.entries.size(); ++i) {
      if (i) out << ",";
      out << "{\"index\":" << table.entries[i].index << ",\"name\":\"" << json_escape(table.entries[i].name) << "\"}";
    }
    out << "]";
    out << "}";
  }
  out << "]";

  out << ",\"invocations\":[";
  for (size_t i = 0; i < invocations.size(); ++i) {
    if (i) out << ",";
    const auto& inv = invocations[i];
    out << "{";
    out << "\"invocation_uid\":\"ctx" << inv.context_uid << "-seq" << inv.launch_seq << "\"";
    out << ",\"context_uid\":" << inv.context_uid;
    out << ",\"correlation_id\":" << inv.correlation_id;
    out << ",\"kernel_name\":\"" << json_escape(inv.kernel_name) << "\"";
    out << ",\"stream\":" << inv.stream;
    out << ",\"grid\":[" << inv.grid_x << "," << inv.grid_y << "," << inv.grid_z << "]";
    out << ",\"block\":[" << inv.block_x << "," << inv.block_y << "," << inv.block_z << "]";
    out << ",\"shared_mem_bytes\":" << inv.shared_mem_bytes;
    out << ",\"selected\":" << (inv.selected ? "true" : "false");
    out << "}";
  }
  out << "]";

  out << ",\"ranges\":[";
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (i) out << ",";
    const auto& range = ranges[i];
    out << "{";
    out << "\"context_uid\":" << range.context_uid;
    out << ",\"range_id\":" << range.range_id;
    out << ",\"total_samples\":" << range.total_samples;
    out << ",\"dropped_samples\":" << range.dropped_samples;
    out << ",\"non_user_samples\":" << range.non_user_samples;
    out << ",\"total_num_pcs\":" << (uint64_t)range.total_num_pcs;
    out << ",\"remaining_num_pcs\":" << (uint64_t)range.remaining_num_pcs;
    out << ",\"hardware_buffer_full\":" << (uint32_t)range.hardware_buffer_full;
    out << "}";
  }
  out << "]";

  out << ",\"pc_records\":[";
  for (size_t i = 0; i < pc_records.size(); ++i) {
    if (i) out << ",";
    const auto& rec = pc_records[i];
    out << "{";
    out << "\"cubinCrc\":" << rec.cubin_crc;
    out << ",\"pcOffset\":" << rec.pc_offset;
    out << ",\"functionIndex\":" << rec.function_index;
    out << ",\"functionName\":\"" << json_escape(rec.function_name) << "\"";
    out << ",\"correlationId\":" << rec.correlation_id;
    out << ",\"rangeId\":" << rec.range_id;
    out << ",\"selected\":" << (selected_corrids.count(rec.correlation_id) ? "true" : "false");
    out << ",\"stall\":[";
    for (size_t j = 0; j < rec.stalls.size(); ++j) {
      if (j) out << ",";
      out << "{\"reasonIndex\":" << rec.stalls[j].reason_index << ",\"samples\":" << rec.stalls[j].samples << "}";
    }
    out << "]";
    out << "}";
  }
  out << "]";

  out << ",\"warnings\":[";
  for (size_t i = 0; i < warnings.size(); ++i) {
    if (i) out << ",";
    out << "\"" << json_escape(warnings[i]) << "\"";
  }
  out << "]";

  out << "}\n";
  out.flush();
  out.close();

  if (std::rename(tmp_path.c_str(), g_output_path.c_str()) != 0) {
    std::fprintf(stderr, "ikp_cupti_pcsamp: rename failed: %s -> %s\n", tmp_path.c_str(), g_output_path.c_str());
  }
}

static void finalize() {
  std::vector<std::shared_ptr<ContextState>> to_free;
  std::vector<CUcontext> ctxs;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& kv : g_contexts) {
      ctxs.push_back(kv.first);
      to_free.push_back(kv.second);
    }
    g_contexts.clear();
  }

  for (size_t i = 0; i < ctxs.size(); ++i) {
    CUcontext ctx = ctxs[i];
    std::shared_ptr<ContextState> state = to_free[i];
    if (!state) continue;
    if (g_enabled.load(std::memory_order_relaxed)) {
      maybe_stop_range(ctx, state.get());
      drain_pc_sampling(ctx, state.get(), /*allow_destroying=*/true);
      CUpti_PCSamplingDisableParams disable_params = {};
      disable_params.size = sizeof(CUpti_PCSamplingDisableParams);
      disable_params.ctx = ctx;
      (void)CUPTI_TRY(cuptiPCSamplingDisable(&disable_params));
      {
        std::lock_guard<std::mutex> lk(state->drain_mu);
        process_pc_sampling_data(state.get());
      }
    }
  }

  write_json();
}

static void CUPTIAPI callback_handler(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                                      void* cbdata) {
  (void)userdata;
  switch (domain) {
    case CUPTI_CB_DOMAIN_DRIVER_API: {
      if (!g_enabled.load(std::memory_order_relaxed)) return;
      const CUpti_CallbackData* info = (const CUpti_CallbackData*)cbdata;
      const bool is_launch =
          (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)
           || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx
#endif
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)
           || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz
#endif
          );
      if (is_launch) {
        if (info->callbackSite == CUPTI_API_ENTER) {
          if (match_kernel(info->symbolName)) {
            std::shared_ptr<ContextState> state = get_or_configure_context(info->context, info->contextUid);
            if (state) maybe_start_range(info->context, state.get());
          }
        } else if (info->callbackSite == CUPTI_API_EXIT) {
          handle_kernel_launch(info, cbid);
        }
      } else if (info->callbackSite == CUPTI_API_EXIT && is_sync_api_name(info->functionName)) {
        handle_sync_point(info);
      }
      break;
    }
    case CUPTI_CB_DOMAIN_RUNTIME_API: {
      if (!g_enabled.load(std::memory_order_relaxed)) return;
      const CUpti_CallbackData* info = (const CUpti_CallbackData*)cbdata;
      if (info->callbackSite == CUPTI_API_EXIT && is_sync_api_name(info->functionName)) {
        handle_sync_point(info);
      }
      break;
    }
    case CUPTI_CB_DOMAIN_RESOURCE: {
      const CUpti_ResourceData* res = (const CUpti_ResourceData*)cbdata;
      if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
        std::shared_ptr<ContextState> state;
        {
          std::lock_guard<std::mutex> lock(g_mutex);
          auto it = g_contexts.find(res->context);
          if (it != g_contexts.end()) {
            state = it->second;
            g_contexts.erase(it);
          }
        }
        if (state) {
          if (g_enabled.load(std::memory_order_relaxed)) {
            {
              std::lock_guard<std::mutex> lk(state->drain_mu);
              state->destroying.store(true, std::memory_order_relaxed);
            }
            drain_pc_sampling(res->context, state.get(), /*allow_destroying=*/true);
            CUpti_PCSamplingDisableParams disable_params = {};
            disable_params.size = sizeof(CUpti_PCSamplingDisableParams);
            disable_params.ctx = res->context;
            (void)CUPTI_TRY(cuptiPCSamplingDisable(&disable_params));
            {
              std::lock_guard<std::mutex> lk(state->drain_mu);
              process_pc_sampling_data(state.get());
            }
          }
        }
        {
          std::lock_guard<std::mutex> lock(g_mutex);
          g_stall_reason_tables.erase(res->context);
        }
      } else if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
        if (!g_enabled.load(std::memory_order_relaxed)) return;
        (void)get_or_configure_context(res->context, resource_context_uid(res, 0));
      } else if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
        if (!g_enabled.load(std::memory_order_relaxed)) return;
        std::shared_ptr<ContextState> state = get_or_configure_context(res->context, resource_context_uid(res, 0));
        if (state) {
          if (!g_use_start_stop) drain_pc_sampling(res->context, state.get());
        }
      }
      break;
    }
    default:
      break;
  }
}

static void initialize_once() {
  bool expected = false;
  if (!g_initialized.compare_exchange_strong(expected, true)) return;
  init_config();
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  {
    bool init_expected = false;
    if (g_profiler_api_inited.compare_exchange_strong(init_expected, true)) {
      CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
      CUptiResult r = cuptiProfilerInitialize(&profilerInitializeParams);
      if (r != CUPTI_SUCCESS) {
        cupti_warn_only(r, "cuptiProfilerInitialize(&params)");
      } else {
        logf(1, "profiler_api: cuptiProfilerInitialize ok");
      }
    }
  }

  std::atexit(finalize);

  maybe_enable_kernel_activity();

  CUpti_SubscriberHandle subscriber;
  if (!CUPTI_TRY(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callback_handler, nullptr))) return;

  (void)CUPTI_TRY(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
  (void)CUPTI_TRY(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  (void)CUPTI_TRY(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE));
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx));
#endif
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz));
#endif

  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                                      CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING));
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED));
}

extern "C" int InitializeInjection(void) {
  initialize_once();
  return 1;
}

}  // namespace

