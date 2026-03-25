#include "ikp_cupti_common.h"

#include <cupti_activity.h>
#include <cupti_pcsampling.h>

#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

namespace {

struct SourceLoc {
  std::string file;
  uint32_t line = 0;
};

struct FunctionInfo {
  uint32_t module_id = 0;
  uint32_t function_index = 0;
  std::string name;
};

struct InstExecRaw {
  uint32_t source_id = 0;
  uint32_t correlation_id = 0;
  uint32_t function_id = 0;
  uint32_t pc_offset = 0;
  uint64_t threads_executed = 0;
  uint64_t not_pred_off_threads_executed = 0;
  uint64_t executed = 0;
};

struct ModuleInfo {
  uint64_t cubin_crc = 0;
};

static std::mutex g_mutex;
static std::vector<InstExecRaw> g_records;
static std::unordered_map<uint32_t, SourceLoc> g_source_map;
static std::unordered_map<uint32_t, FunctionInfo> g_function_map;
static std::unordered_map<uint32_t, ModuleInfo> g_module_map;
static std::vector<std::string> g_warnings;

static ikp_cupti::InvocationTracker g_invocations;

static std::atomic<bool> g_enabled{true};
static std::atomic<bool> g_initialized{false};
static size_t g_activity_buffer_size = 1 << 20;  // 1MB
static std::string g_output_path = "instrexec_raw.json";
static bool g_allow_corrid0 = false;
static size_t g_max_records = 0;
static std::atomic<bool> g_warned_overflow{false};

static void cupti_warn_only(CUptiResult res, const char* call) {
  if (res == CUPTI_SUCCESS) return;
  const char* errstr = nullptr;
  (void)cuptiGetResultString(res, &errstr);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_warnings.emplace_back(std::string("CUPTI ") + call + " failed: " +
                          (errstr ? errstr : "<unknown>") + " (" + std::to_string((int)res) + ")");
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
    if (res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES || res == CUPTI_ERROR_NOT_SUPPORTED) {
      g_warnings.emplace_back("DISABLED: profiling counters not permitted on this node");
      g_enabled.store(false, std::memory_order_relaxed);
    }
  }
  return false;
}

#define CUPTI_TRY(call) cupti_ok((call), #call)

static uint64_t now_ns() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

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

static bool compute_cubin_crc(const void* cubin, size_t cubin_size, uint64_t& out_crc) {
  CUpti_GetCubinCrcParams params{};
  params.size = sizeof(CUpti_GetCubinCrcParams);
  params.cubin = cubin;
  params.cubinSize = cubin_size;
  CUptiResult res = cuptiGetCubinCrc(&params);
  if (res != CUPTI_SUCCESS) return false;
  out_crc = params.cubinCrc;
  return true;
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

static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
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
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("DISABLED: posix_memalign failed for activity buffer");
    g_enabled.store(false, std::memory_order_relaxed);
    *buffer = nullptr;
    *size = 0;
    *maxNumRecords = 0;
    return;
  }
  *buffer = reinterpret_cast<uint8_t*>(ptr);
}

static void handle_activity_record(const CUpti_Activity* record) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR: {
      const auto* loc = reinterpret_cast<const CUpti_ActivitySourceLocator*>(record);
      SourceLoc entry;
      entry.file = loc->fileName ? loc->fileName : "";
      entry.line = loc->lineNumber;
      std::lock_guard<std::mutex> lock(g_mutex);
      g_source_map[loc->id] = std::move(entry);
      break;
    }
    case CUPTI_ACTIVITY_KIND_FUNCTION: {
      const auto* fn = reinterpret_cast<const CUpti_ActivityFunction*>(record);
      FunctionInfo info;
      info.module_id = fn->moduleId;
      info.function_index = fn->functionIndex;
      info.name = fn->name ? fn->name : "";
      std::lock_guard<std::mutex> lock(g_mutex);
      g_function_map[fn->id] = std::move(info);
      break;
    }
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION: {
      const auto* inst = reinterpret_cast<const CUpti_ActivityInstructionExecution*>(record);
      InstExecRaw raw;
      raw.source_id = inst->sourceLocatorId;
      raw.correlation_id = inst->correlationId;
      raw.function_id = inst->functionId;
      raw.pc_offset = inst->pcOffset;
      raw.threads_executed = inst->threadsExecuted;
      raw.not_pred_off_threads_executed = inst->notPredOffThreadsExecuted;
      raw.executed = static_cast<uint64_t>(inst->executed);
      std::lock_guard<std::mutex> lock(g_mutex);
      if (g_max_records && g_records.size() >= g_max_records) {
        if (!g_warned_overflow.exchange(true)) {
          g_warnings.emplace_back(
              "records overflow: dropping further records (set IKP_CUPTI_INSTREXEC_MAX_RECORDS=0 to disable cap)");
        }
        break;
      }
      g_records.push_back(raw);
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      const auto* k = reinterpret_cast<const CUpti_ActivityKernel9*>(record);
      const uint32_t corr = activity_correlation_id(k, 0);
      const char* name = activity_kernel_name(k, 0);
      g_invocations.handle_kernel_activity(corr, name);
      break;
    }
    default:
      break;
  }
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
                                     size_t validSize) {
  (void)size;
  if (!buffer || validSize == 0) {
    std::free(buffer);
    return;
  }
  CUpti_Activity* record = nullptr;
  while (true) {
    CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      handle_activity_record(record);
      continue;
    }
    if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
    (void)cupti_ok(status, "cuptiActivityGetNextRecord");
    break;
  }

  size_t dropped = 0;
  (void)CUPTI_TRY(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
  if (dropped != 0) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("dropped activity records: " + std::to_string(dropped));
  }

  std::free(buffer);
}

static void write_json() {
  const std::string tmp_path = g_output_path + ".tmp." + std::to_string((uint64_t)getpid());
  std::ofstream out(tmp_path);
  if (!out.is_open()) return;

  auto invocations = g_invocations.invocations_snapshot();

  std::vector<InstExecRaw> records;
  std::unordered_map<uint32_t, SourceLoc> source_map;
  std::unordered_map<uint32_t, FunctionInfo> function_map;
  std::unordered_map<uint32_t, ModuleInfo> module_map;
  std::vector<std::string> warnings;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    records = g_records;
    source_map = g_source_map;
    function_map = g_function_map;
    module_map = g_module_map;
    warnings = g_warnings;
  }

  out << "{";
  out << "\"tool\":\"ikp_cupti_instrexec\"";
  out << ",\"version\":1";
  out << ",\"pid\":" << (uint64_t)getpid();
  out << ",\"timestamp_ns\":" << now_ns();

  out << ",\"invocations\":[";
  for (size_t i = 0; i < invocations.size(); ++i) {
    if (i) out << ",";
    const auto& inv = invocations[i];
    out << "{";
    out << "\"invocation_uid\":\"" << ikp_cupti::json_escape(inv.uid()) << "\"";
    out << ",\"context_uid\":" << inv.context_uid;
    out << ",\"correlation_id\":" << inv.correlation_id;
    out << ",\"kernel_name\":\"" << ikp_cupti::json_escape(inv.kernel_name) << "\"";
    out << ",\"stream\":" << inv.stream;
    out << ",\"grid\":[" << inv.grid_x << "," << inv.grid_y << "," << inv.grid_z << "]";
    out << ",\"block\":[" << inv.block_x << "," << inv.block_y << "," << inv.block_z << "]";
    out << ",\"shared_mem_bytes\":" << inv.shared_mem_bytes;
    out << ",\"selected\":" << (inv.selected ? "true" : "false");
    out << "}";
  }
  out << "]";

  out << ",\"records\":[";
  bool first = true;
  bool warned_corrid0 = false;
  for (const auto& raw : records) {
    FunctionInfo fn{};
    SourceLoc src{};
    uint64_t cubin_crc = 0;
    auto fit = function_map.find(raw.function_id);
    if (fit != function_map.end()) fn = fit->second;
    auto sit = source_map.find(raw.source_id);
    if (sit != source_map.end()) src = sit->second;
    auto mit = module_map.find(fn.module_id);
    if (mit != module_map.end()) cubin_crc = mit->second.cubin_crc;

    if (g_invocations.filter().enabled) {
      if (raw.correlation_id == 0) {
        warned_corrid0 = true;
        if (!g_allow_corrid0) continue;
      } else if (!g_invocations.is_selected_correlation(raw.correlation_id)) {
        continue;
      }
    }

    if (!first) out << ",";
    first = false;
    out << "{";
    out << "\"cubinCrc\":" << cubin_crc;
    out << ",\"functionId\":" << raw.function_id;
    out << ",\"functionIndex\":" << fn.function_index;
    out << ",\"functionName\":\"" << ikp_cupti::json_escape(fn.name) << "\"";
    out << ",\"pcOffset\":" << raw.pc_offset;
    out << ",\"correlationId\":" << raw.correlation_id;
    out << ",\"threadsExecuted\":" << raw.threads_executed;
    out << ",\"notPredOffThreadsExecuted\":" << raw.not_pred_off_threads_executed;
    out << ",\"executed\":" << raw.executed;
    if (!src.file.empty()) {
      out << ",\"source\":{\"file\":\"" << ikp_cupti::json_escape(src.file) << "\",\"line\":" << src.line
          << "}";
    }
    out << "}";
  }
  out << "]";

  out << ",\"warnings\":[";
  if (warned_corrid0 && g_invocations.filter().enabled) {
    if (g_allow_corrid0) {
      warnings.emplace_back(
          "correlationId==0 seen while kernel_regex enabled; included records due to IKP_CUPTI_INSTREXEC_ALLOW_CORRID0=1");
    } else {
      warnings.emplace_back(
          "correlationId==0 seen while kernel_regex enabled; records were filtered out (set IKP_CUPTI_INSTREXEC_ALLOW_CORRID0=1 to include)");
    }
  }

  for (size_t i = 0; i < warnings.size(); ++i) {
    if (i) out << ",";
    out << "\"" << ikp_cupti::json_escape(warnings[i]) << "\"";
  }
  out << "]";
  out << "}\n";
  out.flush();
  out.close();

  if (std::rename(tmp_path.c_str(), g_output_path.c_str()) != 0) {
    std::fprintf(stderr, "ikp_cupti_instrexec: rename failed: %s -> %s\n", tmp_path.c_str(), g_output_path.c_str());
  }
}

static void finalize() {
  if (g_enabled.load(std::memory_order_relaxed)) {
    (void)CUPTI_TRY(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  }
  write_json();
}

static void CUPTIAPI callback_handler(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                                      void* cbdata) {
  (void)userdata;
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    const CUpti_CallbackData* info = (const CUpti_CallbackData*)cbdata;
    if (info->callbackSite != CUPTI_API_EXIT) return;
    switch (cbid) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
        g_invocations.handle_kernel_launch(info, cbid);
        break;
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
        g_invocations.handle_kernel_launch(info, cbid);
        break;
#endif
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
        g_invocations.handle_kernel_launch(info, cbid);
        break;
#endif
      default:
        break;
    }
  } else if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      const CUpti_ResourceData* res = (const CUpti_ResourceData*)cbdata;
      const CUpti_ModuleResourceData* module = (const CUpti_ModuleResourceData*)res->resourceDescriptor;
      if (!module || !module->pCubin || module->cubinSize == 0) return;
      uint64_t crc = 0;
      if (!compute_cubin_crc(module->pCubin, module->cubinSize, crc)) return;
      std::lock_guard<std::mutex> lock(g_mutex);
      g_module_map[module->moduleId] = ModuleInfo{crc};
    }
  }
}

static void init_config() {
  g_enabled.store(ikp_cupti::parse_bool_env("IKP_CUPTI_INSTREXEC_ENABLE", true), std::memory_order_relaxed);
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  const char* out = std::getenv("IKP_CUPTI_INSTREXEC_OUT");
  if (out && *out) {
    g_output_path = out;
  } else {
    g_output_path = "instrexec_raw." + std::to_string((uint64_t)getpid()) + ".json";
  }

  {
    uint64_t v = 0;
    const auto st = ikp_cupti::parse_uint64_env_strict("IKP_CUPTI_INSTREXEC_BUFFER_BYTES", v, /*minv=*/4096,
                                                      /*maxv=*/(1ull << 30));
    if (st == ikp_cupti::EnvParseStatus::kOk) {
      g_activity_buffer_size = (size_t)v;
    } else if (st == ikp_cupti::EnvParseStatus::kInvalid) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("invalid IKP_CUPTI_INSTREXEC_BUFFER_BYTES; using default");
    }
  }

  g_invocations.set_filter_from_env("IKP_CUPTI_INSTREXEC_KERNEL_REGEX");
  if (g_invocations.filter().invalid) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("invalid IKP_CUPTI_INSTREXEC_KERNEL_REGEX; regex disabled");
  }
  g_allow_corrid0 = ikp_cupti::parse_bool_env("IKP_CUPTI_INSTREXEC_ALLOW_CORRID0", false);
  {
    uint64_t v = 0;
    const auto st = ikp_cupti::parse_uint64_env_strict("IKP_CUPTI_INSTREXEC_MAX_RECORDS", v);
    if (st == ikp_cupti::EnvParseStatus::kOk) {
      g_max_records = (size_t)v;
    } else if (st == ikp_cupti::EnvParseStatus::kInvalid) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("invalid IKP_CUPTI_INSTREXEC_MAX_RECORDS; using default");
    }
  }
}

static void initialize() {
  bool expected = false;
  if (!g_initialized.compare_exchange_strong(expected, true)) return;

  init_config();
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  if (!try_claim_activity_callbacks_owner("instrexec")) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back(
        "DISABLED: CUPTI activity callbacks already owned by another tool (set only one of the activity-based injectors)");
    g_enabled.store(false, std::memory_order_relaxed);
    write_json();
    return;
  }

  std::atexit(finalize);

  if (!CUPTI_TRY(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted))) return;
  if (!CUPTI_TRY(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION))) return;
  if (g_invocations.filter().enabled) {
    CUptiResult ena = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (ena != CUPTI_SUCCESS) {
      cupti_warn_only(ena, "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)");
      ena = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    }
    if (ena != CUPTI_SUCCESS) {
      cupti_warn_only(ena, "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)");
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back(
          "kernel_regex enabled but kernel activity could not be enabled; launches outside driver callbacks may be filtered incorrectly");
    }
  }
  cupti_warn_only(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_FUNCTION),
                  "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_FUNCTION)");
  cupti_warn_only(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR),
                  "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR)");

  CUpti_SubscriberHandle subscriber;
  if (!CUPTI_TRY(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callback_handler, nullptr))) return;
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz));
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx));
#endif
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz));
#endif
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                                      CUPTI_CBID_RESOURCE_MODULE_LOADED));
}

extern "C" int InitializeInjection(void) {
  initialize();
  return 1;
}

}  // namespace

