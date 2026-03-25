#include "ikp_cupti_common.h"

#include <cupti_activity.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_sass_metrics.h>
#include <cupti_target.h>

#include <cuda.h>

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace {

struct MetricRecord {
  uint64_t cubin_crc = 0;
  uint32_t function_index = 0;
  uint32_t pc_offset = 0;
  uint32_t correlation_id = 0;
  std::string function_name;
  std::unordered_map<std::string, uint64_t> metrics;
  std::string source_file;
  uint32_t source_line = 0;
};

struct ModuleCubin {
  std::shared_ptr<const std::vector<uint8_t>> cubin;
  uint64_t cubin_crc = 0;
};

static std::mutex g_mutex;
static std::vector<MetricRecord> g_records;
static std::vector<std::string> g_warnings;
static std::unordered_map<uint64_t, ModuleCubin> g_cubin_by_crc;

static ikp_cupti::InvocationTracker g_invocations;

static std::atomic<bool> g_enabled{true};
static std::atomic<bool> g_initialized{false};
static bool g_lazy_patching = true;
static bool g_per_launch = false;
static bool g_per_launch_sync = false;  // force sync boundaries for accurate per-kernel isolation
static bool g_enable_source = false;
static bool g_list_only = false;
static std::string g_output_path = "sassmetrics_raw.json";
static std::string g_metrics_json_path = ikp_cupti::resolve_sibling_path("metrics_profiles.json");
static std::string g_profile_name = "core";

static std::once_flag g_profiler_init_once;
static bool g_profiler_initialized = false;

struct DeviceConfig {
  std::once_flag config_once;
  bool configured = false;
  // Immutable after configuration.
  std::shared_ptr<const std::unordered_map<uint64_t, std::string>> metric_id_to_name;
};

static std::mutex g_device_mutex;
static std::unordered_map<int, std::unique_ptr<DeviceConfig>> g_device_cfg;

static std::mutex g_state_mutex;
static std::unordered_set<CUcontext> g_contexts;
static std::unordered_set<CUcontext> g_metrics_enabled_contexts;

static size_t g_activity_buffer_size = 1 << 20;  // 1MB
static bool g_per_launch_use_activity = true;
static CUcontext g_active_ctx = nullptr;
static uint32_t g_active_correlation_id = 0;
static bool g_active_flush_on_exit = false;  // fallback path when activity is unavailable
static std::mutex g_per_launch_sync_mu;
static thread_local bool tls_sync_active = false;
static thread_local CUcontext tls_sync_ctx = nullptr;
static thread_local uint32_t tls_sync_corr = 0;
static thread_local bool tls_sync_enabled = false;

static size_t g_max_records = 0;
static std::atomic<bool> g_warned_overflow{false};
static std::atomic<bool> g_warned_corrid0{false};
static std::atomic<bool> g_warned_cubin_cap{false};
static size_t g_cubin_bytes_total = 0;
static size_t g_max_cubin_bytes = (size_t)(512ull << 20);  // 512MB (0 disables cap)

static bool cupti_ok(CUptiResult res, const char* call);

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

static void ensure_profiler_initialized() {
  std::call_once(g_profiler_init_once, []() {
    CUpti_Profiler_Initialize_Params params{CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    if (!cupti_ok(cuptiProfilerInitialize(&params), "cuptiProfilerInitialize(&params)")) return;
    g_profiler_initialized = true;
  });
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

static void cupti_warn_only(CUptiResult res, const char* call) {
  if (res == CUPTI_SUCCESS) return;
  const char* errstr = nullptr;
  (void)cuptiGetResultString(res, &errstr);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_warnings.emplace_back(std::string("CUPTI ") + call + " failed: " +
                          (errstr ? errstr : "<unknown>") + " (" + std::to_string((int)res) +
                          ")");
}

static uint64_t now_ns() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

static void flush_metrics(CUcontext ctx, uint32_t correlation_id, bool best_effort = false);
static void disable_metrics_ctx(CUcontext ctx, bool best_effort = false);

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

static void on_kernel_complete(uint32_t correlation_id) {
  if (!g_enabled.load(std::memory_order_relaxed)) return;
  if (!g_per_launch) return;
  if (!g_per_launch_use_activity) return;
  if (correlation_id == 0) return;

  CUcontext ctx = nullptr;
  uint32_t corr = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (!g_active_ctx) return;
    if (g_active_correlation_id != correlation_id) return;
    ctx = g_active_ctx;
    corr = g_active_correlation_id;
    g_active_ctx = nullptr;
    g_active_correlation_id = 0;
    g_active_flush_on_exit = false;
  }

  if (ctx) {
    flush_metrics(ctx, corr);
    disable_metrics_ctx(ctx);
  }
}

static void CUPTIAPI activity_buffer_requested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  if (!g_enabled.load(std::memory_order_relaxed) || !g_per_launch_use_activity) {
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
      g_warnings.emplace_back("DISABLED: posix_memalign failed for activity buffer");
    }
    g_enabled.store(false, std::memory_order_relaxed);
    *buffer = nullptr;
    *size = 0;
    *maxNumRecords = 0;
    return;
  }
  *buffer = reinterpret_cast<uint8_t*>(ptr);
}

static void CUPTIAPI activity_buffer_completed(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
                                               size_t validSize) {
  (void)size;
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
          g_invocations.handle_kernel_activity(corr, name);
          on_kernel_complete(corr);
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

  size_t dropped = 0;
  (void)cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
  if (dropped != 0) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("dropped activity records: " + std::to_string(dropped));
  }

  std::free(buffer);
}

static std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) return {};
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return content;
}

// Minimal JSON parser to load profiles: { "profiles": { "core": ["m1","m2"] } }
static std::vector<std::string> parse_profiles_json(const std::string& content, const std::string& profile) {
  std::vector<std::string> metrics;
  std::string key = "\"" + profile + "\"";
  auto pos = content.find("\"profiles\"");
  if (pos == std::string::npos) return metrics;
  pos = content.find(key, pos);
  if (pos == std::string::npos) return metrics;
  pos = content.find('[', pos);
  if (pos == std::string::npos) return metrics;
  ++pos;
  while (pos < content.size()) {
    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) ++pos;
    if (pos >= content.size() || content[pos] == ']') break;
    if (content[pos] != '"') {
      ++pos;
      continue;
    }
    ++pos;
    std::string value;
    while (pos < content.size() && content[pos] != '"') {
      if (content[pos] == '\\' && pos + 1 < content.size()) {
        ++pos;
      }
      value.push_back(content[pos++]);
    }
    if (!value.empty()) metrics.push_back(value);
    pos = content.find_first_of(",]", pos);
    if (pos == std::string::npos || content[pos] == ']') break;
    ++pos;
  }
  return metrics;
}

static std::vector<std::string> load_metrics_profile() {
  const char* override_list = std::getenv("IKP_CUPTI_SASS_METRICS");
  if (override_list && *override_list) {
    std::vector<std::string> out;
    std::string current;
    for (const char* p = override_list; *p; ++p) {
      if (*p == ',') {
        if (!current.empty()) out.push_back(current);
        current.clear();
      } else {
        current.push_back(*p);
      }
    }
    if (!current.empty()) out.push_back(current);
    return out;
  }

  std::string content = read_file(g_metrics_json_path);
  if (content.empty()) {
    // Backward/alternate locations (for users running from different working dirs).
    static const char* candidates[] = {
        "./metrics_profiles.json",
        "./cupti_region_profiler/metrics_profiles.json",
    };
    for (const char* p : candidates) {
      if (std::string(p) == g_metrics_json_path) continue;
      content = read_file(p);
      if (!content.empty()) {
        g_metrics_json_path = p;
        break;
      }
    }
  }
  if (content.empty()) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("metrics profile JSON not found or empty");
    return {};
  }
  return parse_profiles_json(content, g_profile_name);
}

static void list_supported_metrics(int device_index) {
  ensure_profiler_initialized();
  if (!g_enabled.load(std::memory_order_relaxed)) return;
  CUpti_Device_GetChipName_Params chip_params{CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  chip_params.deviceIndex = device_index;
  if (!CUPTI_TRY(cuptiDeviceGetChipName(&chip_params))) return;

  CUpti_SassMetrics_GetNumOfMetrics_Params get_num{CUpti_SassMetrics_GetNumOfMetrics_Params_STRUCT_SIZE};
  get_num.pChipName = chip_params.pChipName;
  if (!CUPTI_TRY(cuptiSassMetricsGetNumOfMetrics(&get_num))) return;
  if (get_num.numOfMetrics == 0) return;

  std::vector<CUpti_SassMetrics_MetricDetails> details(get_num.numOfMetrics);
  CUpti_SassMetrics_GetMetrics_Params get_metrics{};
  get_metrics.structSize = sizeof(CUpti_SassMetrics_GetMetrics_Params);
  get_metrics.pPriv = nullptr;
  get_metrics.pChipName = chip_params.pChipName;
  get_metrics.numOfMetrics = get_num.numOfMetrics;
  get_metrics.pMetricsList = details.data();
  if (!CUPTI_TRY(cuptiSassMetricsGetMetrics(&get_metrics))) return;

  const char* out_path = std::getenv("IKP_CUPTI_SASS_LIST_OUT");
  std::ofstream out;
  if (out_path && *out_path) {
    out.open(out_path);
  }

  for (size_t i = 0; i < details.size(); ++i) {
    const char* name = details[i].pMetricName ? details[i].pMetricName : "";
    if (out.is_open()) {
      out << name << "\n";
    } else {
      std::printf("SASS_METRIC: %s\n", name);
    }
  }
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

static void sync_context_best_effort(CUcontext ctx) {
  if (!ctx) return;
  CUcontext popped = nullptr;
  CUresult push = cuCtxPushCurrent(ctx);
  if (push != CUDA_SUCCESS) {
    (void)cuCtxSynchronize();
    return;
  }
  (void)cuCtxSynchronize();
  (void)cuCtxPopCurrent(&popped);
}

static int device_index_for_context_best_effort(CUcontext ctx) {
  CUdevice device = 0;
  CUcontext popped = nullptr;
  CUresult push = cuCtxPushCurrent(ctx);
  if (push == CUDA_SUCCESS) {
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS) device = 0;
    (void)cuCtxPopCurrent(&popped);
  } else {
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS) device = 0;
  }
  return static_cast<int>(device);
}

static DeviceConfig* get_or_create_device_cfg(int device_index) {
  std::lock_guard<std::mutex> lock(g_device_mutex);
  auto& ptr = g_device_cfg[device_index];
  if (!ptr) ptr = std::make_unique<DeviceConfig>();
  return ptr.get();
}

static std::shared_ptr<const std::unordered_map<uint64_t, std::string>> metric_name_map_for_device(int device_index) {
  std::lock_guard<std::mutex> lock(g_device_mutex);
  auto it = g_device_cfg.find(device_index);
  if (it == g_device_cfg.end() || !it->second) return {};
  return it->second->metric_id_to_name;
}

static void store_cubin(uint64_t crc, const char* cubin, size_t cubin_size) {
  if (!g_enable_source) return;
  if (!cubin || cubin_size == 0) return;

  auto buf = std::make_shared<std::vector<uint8_t>>(reinterpret_cast<const uint8_t*>(cubin),
                                                    reinterpret_cast<const uint8_t*>(cubin) + cubin_size);

  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_cubin_by_crc.count(crc)) return;
  if (g_max_cubin_bytes != 0 && (g_cubin_bytes_total + buf->size()) > g_max_cubin_bytes) {
    if (!g_warned_cubin_cap.exchange(true)) {
      g_warnings.emplace_back(
          "cubin cache cap reached; source correlation may be incomplete (set IKP_CUPTI_SASS_MAX_CUBIN_BYTES=0 to disable cap)");
    }
    return;
  }
  ModuleCubin mod;
  mod.cubin_crc = crc;
  mod.cubin = std::move(buf);
  g_cubin_by_crc.emplace(crc, std::move(mod));
  g_cubin_bytes_total += cubin_size;
}

static bool resolve_source(uint64_t cubin_crc, const char* function_name, uint32_t pc_offset, std::string& file,
                           uint32_t& line) {
  if (!g_enable_source) return false;
  std::shared_ptr<const std::vector<uint8_t>> cubin;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_cubin_by_crc.find(cubin_crc);
    if (it == g_cubin_by_crc.end()) return false;
    cubin = it->second.cubin;
  }
  if (!cubin || cubin->empty()) return false;

  CUpti_GetSassToSourceCorrelationParams params{};
  params.size = sizeof(CUpti_GetSassToSourceCorrelationParams);
  params.cubin = cubin->data();
  params.cubinSize = cubin->size();
  params.functionName = function_name;
  params.pcOffset = pc_offset;
  CUptiResult res = cuptiGetSassToSourceCorrelation(&params);
  if (res != CUPTI_SUCCESS) return false;
  if (params.dirName && params.fileName) file = std::string(params.dirName) + "/" + params.fileName;
  else if (params.fileName) file = params.fileName;
  line = params.lineNumber;
  if (params.fileName) free(params.fileName);
  if (params.dirName) free(params.dirName);
  return true;
}

static void configure_metrics(CUcontext ctx) {
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  const int device_index = device_index_for_context_best_effort(ctx);
  DeviceConfig* cfg = get_or_create_device_cfg(device_index);

  std::call_once(cfg->config_once, [cfg, device_index]() {
    ensure_profiler_initialized();
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    CUpti_Profiler_DeviceSupported_Params supported{CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
    supported.cuDevice = device_index;
    supported.api = CUPTI_PROFILER_SASS_METRICS;
    (void)CUPTI_TRY(cuptiProfilerDeviceSupported(&supported));
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (supported.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("DISABLED: cuptiProfilerDeviceSupported reports unsupported configuration");
      g_enabled.store(false, std::memory_order_relaxed);
      return;
    }

    if (g_list_only) {
      list_supported_metrics(device_index);
      g_enabled.store(false, std::memory_order_relaxed);
      return;
    }

    auto metrics = load_metrics_profile();
    if (metrics.empty()) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("no SASS metrics configured, skipping collection");
      g_enabled.store(false, std::memory_order_relaxed);
      return;
    }

    CUpti_Device_GetChipName_Params chip_params{CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chip_params.deviceIndex = device_index;
    if (!CUPTI_TRY(cuptiDeviceGetChipName(&chip_params))) return;

    std::vector<CUpti_SassMetrics_Config> metric_configs;
    metric_configs.resize(metrics.size());
    auto id_to_name_local = std::make_shared<std::unordered_map<uint64_t, std::string>>();
    for (size_t i = 0; i < metrics.size(); ++i) {
      CUpti_SassMetrics_GetProperties_Params props{CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
      props.pChipName = chip_params.pChipName;
      props.pMetricName = metrics[i].c_str();
      if (!CUPTI_TRY(cuptiSassMetricsGetProperties(&props))) return;
      metric_configs[i].metricId = props.metric.metricId;
      metric_configs[i].outputGranularity = CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
      (*id_to_name_local)[props.metric.metricId] = metrics[i];
    }

    CUpti_SassMetricsSetConfig_Params set_params{CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE};
    set_params.pConfigs = metric_configs.data();
    set_params.numOfMetricConfig = metric_configs.size();
    set_params.deviceIndex = device_index;
    if (!CUPTI_TRY(cuptiSassMetricsSetConfig(&set_params))) return;

    cfg->metric_id_to_name = id_to_name_local;
    cfg->configured = true;
  });

  if (!cfg->configured) return;

  if (!g_per_launch) {
    {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      if (g_metrics_enabled_contexts.count(ctx)) return;
    }
    CUpti_SassMetricsEnable_Params enable_params{CUpti_SassMetricsEnable_Params_STRUCT_SIZE};
    enable_params.enableLazyPatching = g_lazy_patching;
    enable_params.ctx = ctx;
    if (!CUPTI_TRY(cuptiSassMetricsEnable(&enable_params))) return;
    {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      g_metrics_enabled_contexts.insert(ctx);
    }
  }
}

static void flush_metrics(CUcontext ctx, uint32_t correlation_id, bool best_effort) {
  if (!best_effort && !g_enabled.load(std::memory_order_relaxed)) return;
  auto metric_map = metric_name_map_for_device(device_index_for_context_best_effort(ctx));
  CUpti_SassMetricsGetDataProperties_Params props{CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE};
  props.ctx = ctx;
  if (!CUPTI_TRY(cuptiSassMetricsGetDataProperties(&props))) return;
  if (props.numOfInstances == 0 || props.numOfPatchedInstructionRecords == 0) return;

  CUpti_SassMetricsFlushData_Params flush{CUpti_SassMetricsFlushData_Params_STRUCT_SIZE};
  flush.ctx = ctx;
  flush.numOfInstances = props.numOfInstances;
  flush.numOfPatchedInstructionRecords = props.numOfPatchedInstructionRecords;
  flush.pMetricsData =
      (CUpti_SassMetrics_Data*)malloc(props.numOfPatchedInstructionRecords * sizeof(CUpti_SassMetrics_Data));
  if (!flush.pMetricsData) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("flush_metrics: OOM allocating pMetricsData");
    return;
  }
  std::memset(flush.pMetricsData, 0, props.numOfPatchedInstructionRecords * sizeof(CUpti_SassMetrics_Data));
  for (size_t i = 0; i < props.numOfPatchedInstructionRecords; ++i) {
    flush.pMetricsData[i].pInstanceValues =
        (CUpti_SassMetrics_InstanceValue*)malloc(props.numOfInstances * sizeof(CUpti_SassMetrics_InstanceValue));
    if (!flush.pMetricsData[i].pInstanceValues) {
      for (size_t j = 0; j < i; ++j) free(flush.pMetricsData[j].pInstanceValues);
      free(flush.pMetricsData);
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("flush_metrics: OOM allocating pInstanceValues");
      return;
    }
  }

  if (!CUPTI_TRY(cuptiSassMetricsFlushData(&flush))) {
    for (size_t i = 0; i < props.numOfPatchedInstructionRecords; ++i) free(flush.pMetricsData[i].pInstanceValues);
    free(flush.pMetricsData);
    return;
  }

  for (size_t i = 0; i < props.numOfPatchedInstructionRecords; ++i) {
    const auto& data = flush.pMetricsData[i];
    MetricRecord rec;
    rec.cubin_crc = data.cubinCrc;
    rec.function_index = data.functionIndex;
    rec.function_name = data.functionName ? data.functionName : "";
    rec.pc_offset = data.pcOffset;
    rec.correlation_id = correlation_id;

    for (size_t j = 0; j < props.numOfInstances; ++j) {
      const auto& inst = data.pInstanceValues[j];
      std::string name;
      if (metric_map) {
        auto it = metric_map->find(inst.metricId);
        if (it != metric_map->end()) name = it->second;
      }
      if (name.empty()) name = "metric_" + std::to_string(inst.metricId);
      rec.metrics[name] += inst.value;
    }

    resolve_source(rec.cubin_crc, rec.function_name.c_str(), rec.pc_offset, rec.source_file, rec.source_line);

    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_max_records && g_records.size() >= g_max_records) {
      if (!g_warned_overflow.exchange(true)) {
        g_warnings.emplace_back(
            "records overflow: dropping further records (set IKP_CUPTI_SASS_MAX_RECORDS=0 to disable cap)");
      }
      continue;
    }
    g_records.push_back(std::move(rec));
  }

  for (size_t i = 0; i < props.numOfPatchedInstructionRecords; ++i) {
    free((void*)flush.pMetricsData[i].functionName);
    free(flush.pMetricsData[i].pInstanceValues);
  }
  free(flush.pMetricsData);
}

static void disable_metrics_ctx(CUcontext ctx, bool best_effort) {
  if (!best_effort && !g_enabled.load(std::memory_order_relaxed)) return;
  CUpti_SassMetricsDisable_Params disable{CUpti_SassMetricsDisable_Params_STRUCT_SIZE};
  disable.ctx = ctx;
  (void)CUPTI_TRY(cuptiSassMetricsDisable(&disable));
}

static void unset_config_all() {
  if (!g_enabled.load(std::memory_order_relaxed)) return;
  std::vector<int> devices;
  {
    std::lock_guard<std::mutex> lock(g_device_mutex);
    for (const auto& kv : g_device_cfg) {
      if (kv.second && kv.second->configured) devices.push_back(kv.first);
    }
  }
  for (int device_index : devices) {
    CUpti_SassMetricsUnsetConfig_Params unset{CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};
    unset.deviceIndex = device_index;
    (void)CUPTI_TRY(cuptiSassMetricsUnsetConfig(&unset));
  }
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_metrics_enabled_contexts.clear();
  }
}

static void write_json() {
  const std::string tmp_path = g_output_path + ".tmp." + std::to_string((uint64_t)getpid());
  std::ofstream out(tmp_path);
  if (!out.is_open()) return;

  out << "{";
  out << "\"tool\":\"ikp_cupti_sassmetrics\"";
  out << ",\"version\":1";
  out << ",\"pid\":" << (uint64_t)getpid();
  out << ",\"timestamp_ns\":" << now_ns();
  out << ",\"metrics_profile\":\"" << ikp_cupti::json_escape(g_profile_name) << "\"";
  out << ",\"metrics_json\":\"" << ikp_cupti::json_escape(g_metrics_json_path) << "\"";
  out << ",\"lazy_patching\":" << (g_lazy_patching ? "true" : "false");
  out << ",\"per_launch\":" << (g_per_launch ? "true" : "false");
  out << ",\"enable_source\":" << (g_enable_source ? "true" : "false");

  auto invocations = g_invocations.invocations_snapshot();
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

  std::vector<MetricRecord> records;
  std::vector<std::string> warnings;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    records = g_records;
    warnings = g_warnings;
  }

  out << ",\"records\":[";
  for (size_t i = 0; i < records.size(); ++i) {
    if (i) out << ",";
    const auto& rec = records[i];
    out << "{";
    out << "\"cubinCrc\":" << rec.cubin_crc;
    out << ",\"functionIndex\":" << rec.function_index;
    out << ",\"functionName\":\"" << ikp_cupti::json_escape(rec.function_name) << "\"";
    out << ",\"pcOffset\":" << rec.pc_offset;
    out << ",\"correlationId\":" << rec.correlation_id;
    if (!rec.source_file.empty()) {
      out << ",\"source\":{\"file\":\"" << ikp_cupti::json_escape(rec.source_file) << "\",\"line\":" << rec.source_line
          << "}";
    }
    out << ",\"metrics\":{";
    bool first = true;
    for (const auto& kv : rec.metrics) {
      if (!first) out << ",";
      first = false;
      out << "\"" << ikp_cupti::json_escape(kv.first) << "\":" << kv.second;
    }
    out << "}";
    out << "}";
  }
  out << "]";

  out << ",\"warnings\":[";
  for (size_t i = 0; i < warnings.size(); ++i) {
    if (i) out << ",";
    out << "\"" << ikp_cupti::json_escape(warnings[i]) << "\"";
  }
  out << "]";
  out << "}\n";
  out.flush();
  out.close();

  if (std::rename(tmp_path.c_str(), g_output_path.c_str()) != 0) {
    std::fprintf(stderr, "ikp_cupti_sassmetrics: rename failed: %s -> %s\n", tmp_path.c_str(), g_output_path.c_str());
  }
}

static void finalize() {
  if (g_enabled.load(std::memory_order_relaxed) && g_per_launch && g_per_launch_use_activity) {
    (void)CUPTI_TRY(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  }

  if (g_enabled.load(std::memory_order_relaxed) && !g_per_launch) {
    std::vector<CUcontext> ctxs;
    {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      ctxs.assign(g_contexts.begin(), g_contexts.end());
    }
    for (CUcontext ctx : ctxs) {
      flush_metrics(ctx, 0);
      disable_metrics_ctx(ctx);
    }
    unset_config_all();
  } else if (g_enabled.load(std::memory_order_relaxed) && g_per_launch) {
    CUcontext ctx = nullptr;
    uint32_t corr = 0;
    {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      ctx = g_active_ctx;
      corr = g_active_correlation_id;
      g_active_ctx = nullptr;
      g_active_correlation_id = 0;
      g_active_flush_on_exit = false;
    }
    if (ctx) {
      flush_metrics(ctx, corr);
      disable_metrics_ctx(ctx);
    }
    unset_config_all();
  }
  write_json();
}

static void CUPTIAPI callback_handler(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void* cbdata) {
  (void)userdata;
  if (domain == CUPTI_CB_DOMAIN_RESOURCE && cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
    const CUpti_ResourceData* res = (const CUpti_ResourceData*)cbdata;
    bool had_metrics_enabled = false;
    bool had_active_window = false;
    uint32_t active_corr = 0;
    {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      g_contexts.erase(res->context);
      had_metrics_enabled = (g_metrics_enabled_contexts.count(res->context) != 0);
      g_metrics_enabled_contexts.erase(res->context);
      if (g_active_ctx == res->context) {
        had_active_window = true;
        active_corr = g_active_correlation_id;
        g_active_ctx = nullptr;
        g_active_correlation_id = 0;
        g_active_flush_on_exit = false;
      }
    }
    if (had_active_window) {
      flush_metrics(res->context, active_corr, /*best_effort=*/true);
    } else if (had_metrics_enabled) {
      flush_metrics(res->context, 0, /*best_effort=*/true);
    }
    disable_metrics_ctx(res->context, /*best_effort=*/true);
    return;
  }

  if (!g_enabled.load(std::memory_order_relaxed)) return;

  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    const CUpti_ResourceData* res = (const CUpti_ResourceData*)cbdata;
    if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
      {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_contexts.insert(res->context);
      }
      configure_metrics(res->context);
    } else if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      if (!g_enable_source) return;
      const CUpti_ModuleResourceData* module = (const CUpti_ModuleResourceData*)res->resourceDescriptor;
      if (!module || !module->pCubin || module->cubinSize == 0) return;
      uint64_t crc = 0;
      if (compute_cubin_crc(module->pCubin, module->cubinSize, crc)) {
        store_cubin(crc, module->pCubin, module->cubinSize);
      }
    }
    return;
  }

  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    const CUpti_CallbackData* info = (const CUpti_CallbackData*)cbdata;
    const bool is_launch = (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) ||
                           (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz)
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)
                           || (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx)
#endif
#if defined(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)
                           || (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz)
#endif
        ;
    if (!is_launch) return;

    if (info->callbackSite == CUPTI_API_ENTER) {
      if (!g_per_launch) return;
      const bool selected = g_invocations.filter().match(info->symbolName);
      if (g_per_launch_sync) {
        if (!selected) {
          g_per_launch_sync_mu.lock();
          g_per_launch_sync_mu.unlock();
          return;
        }
      } else {
        if (!selected) return;
      }

      configure_metrics(info->context);
      if (!g_enabled.load(std::memory_order_relaxed)) return;

      if (g_per_launch_sync) {
        g_per_launch_sync_mu.lock();
        sync_context_best_effort(info->context);
      } else {
        CUcontext prev_ctx = nullptr;
        uint32_t prev_corr = 0;
        {
          std::lock_guard<std::mutex> lock(g_state_mutex);
          if (g_active_ctx != nullptr) {
            prev_ctx = g_active_ctx;
            prev_corr = g_active_correlation_id;
            g_active_ctx = nullptr;
            g_active_correlation_id = 0;
            g_active_flush_on_exit = false;
          }
        }
        if (prev_ctx != nullptr) {
          {
            std::lock_guard<std::mutex> wlock(g_mutex);
            g_warnings.emplace_back("per_launch overlap: forcing sync+flush for previous window to avoid pollution");
          }
          sync_context_best_effort(prev_ctx);
          flush_metrics(prev_ctx, prev_corr, /*best_effort=*/true);
          disable_metrics_ctx(prev_ctx, /*best_effort=*/true);
        }
      }

      CUpti_SassMetricsEnable_Params enable_params{CUpti_SassMetricsEnable_Params_STRUCT_SIZE};
      enable_params.enableLazyPatching = g_lazy_patching;
      enable_params.ctx = info->context;
      if (!CUPTI_TRY(cuptiSassMetricsEnable(&enable_params))) {
        if (g_per_launch_sync) g_per_launch_sync_mu.unlock();
        return;
      }
      if (g_per_launch_sync) {
        tls_sync_active = true;
        tls_sync_ctx = info->context;
        tls_sync_corr = info->correlationId;
        tls_sync_enabled = true;
      }

      const bool corrid0 = (info->correlationId == 0);
      {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_active_ctx = info->context;
        g_active_correlation_id = info->correlationId;
        g_active_flush_on_exit = g_per_launch_sync || !g_per_launch_use_activity || corrid0;
      }
      if (corrid0 && !g_warned_corrid0.exchange(true)) {
        std::lock_guard<std::mutex> wlock(g_mutex);
        g_warnings.emplace_back("per_launch: correlationId==0; using cuCtxSynchronize() on API_EXIT for this kernel");
      }
      return;
    }

    if (info->callbackSite == CUPTI_API_EXIT) {
      g_invocations.handle_kernel_launch(info, cbid);
      if (!g_per_launch) return;
      if (!g_invocations.filter().match(info->symbolName)) return;

      if (g_per_launch_sync && tls_sync_active && tls_sync_enabled) {
        CUcontext ctx = tls_sync_ctx;
        uint32_t corr = tls_sync_corr;
        {
          std::lock_guard<std::mutex> lock(g_state_mutex);
          if (g_active_ctx == ctx) {
            g_active_ctx = nullptr;
            g_active_correlation_id = 0;
            g_active_flush_on_exit = false;
          }
        }

        sync_context_best_effort(ctx);
        flush_metrics(ctx, corr, /*best_effort=*/true);
        disable_metrics_ctx(ctx, /*best_effort=*/true);

        tls_sync_active = false;
        tls_sync_ctx = nullptr;
        tls_sync_corr = 0;
        tls_sync_enabled = false;
        g_per_launch_sync_mu.unlock();
        return;
      }

      CUcontext ctx = nullptr;
      uint32_t corr = 0;
      bool do_flush = false;
      {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        do_flush = g_active_flush_on_exit && g_active_ctx == info->context && g_active_correlation_id == info->correlationId;
        if (do_flush) {
          ctx = g_active_ctx;
          corr = g_active_correlation_id;
          g_active_ctx = nullptr;
          g_active_correlation_id = 0;
          g_active_flush_on_exit = false;
        }
      }
      if (do_flush && ctx) {
        cuCtxSynchronize();
        flush_metrics(ctx, corr, /*best_effort=*/true);
        disable_metrics_ctx(ctx, /*best_effort=*/true);
      }
    }
  }
}

static void init_config() {
  g_enabled.store(ikp_cupti::parse_bool_env("IKP_CUPTI_SASS_ENABLE", true), std::memory_order_relaxed);
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  const char* out = std::getenv("IKP_CUPTI_SASS_OUT");
  if (out && *out) {
    g_output_path = out;
  } else {
    g_output_path = "sassmetrics_raw." + std::to_string((uint64_t)getpid()) + ".json";
  }
  if (const char* json_path = std::getenv("IKP_CUPTI_SASS_METRICS_JSON")) {
    if (json_path && *json_path) g_metrics_json_path = json_path;
  }
  if (const char* profile = std::getenv("IKP_CUPTI_SASS_PROFILE")) {
    if (profile && *profile) g_profile_name = profile;
  }

  g_lazy_patching = ikp_cupti::parse_bool_env("IKP_CUPTI_SASS_LAZY_PATCHING", true);
  g_per_launch = ikp_cupti::parse_bool_env("IKP_CUPTI_SASS_PER_LAUNCH", false);
  g_per_launch_sync = ikp_cupti::parse_bool_env("IKP_CUPTI_SASS_PER_LAUNCH_SYNC", false);
  g_enable_source = ikp_cupti::parse_bool_env("IKP_CUPTI_SASS_ENABLE_SOURCE", false);
  g_list_only = ikp_cupti::parse_bool_env("IKP_CUPTI_SASS_LIST", false);
  {
    uint64_t v = 0;
    const auto st = ikp_cupti::parse_uint64_env_strict("IKP_CUPTI_SASS_ACTIVITY_BUFFER_BYTES", v, /*minv=*/4096,
                                                      /*maxv=*/(1ull << 30));
    if (st == ikp_cupti::EnvParseStatus::kOk) {
      g_activity_buffer_size = (size_t)v;
    } else if (st == ikp_cupti::EnvParseStatus::kInvalid) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("invalid IKP_CUPTI_SASS_ACTIVITY_BUFFER_BYTES; using default");
    }
  }
  {
    uint64_t v = 0;
    const auto st = ikp_cupti::parse_uint64_env_strict("IKP_CUPTI_SASS_MAX_RECORDS", v);
    if (st == ikp_cupti::EnvParseStatus::kOk) {
      g_max_records = (size_t)v;
    } else if (st == ikp_cupti::EnvParseStatus::kInvalid) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("invalid IKP_CUPTI_SASS_MAX_RECORDS; using default");
    }
  }
  {
    uint64_t v = 0;
    const auto st = ikp_cupti::parse_uint64_env_strict("IKP_CUPTI_SASS_MAX_CUBIN_BYTES", v, /*minv=*/0);
    if (st == ikp_cupti::EnvParseStatus::kOk) {
      g_max_cubin_bytes = (size_t)v;
    } else if (st == ikp_cupti::EnvParseStatus::kInvalid) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back("invalid IKP_CUPTI_SASS_MAX_CUBIN_BYTES; using default");
    }
  }

  g_invocations.set_filter_from_env("IKP_CUPTI_SASS_KERNEL_REGEX");
  if (g_invocations.filter().invalid) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back("invalid IKP_CUPTI_SASS_KERNEL_REGEX; regex disabled");
  }
  if (g_invocations.filter().enabled && !g_per_launch) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_warnings.emplace_back(
        "IKP_CUPTI_SASS_KERNEL_REGEX is only used to select per-launch windows; per_launch=0 cannot attribute per-kernel records, so filter will not be applied to output");
  }
}

static void initialize() {
  bool expected = false;
  if (!g_initialized.compare_exchange_strong(expected, true)) return;

  init_config();
  if (!g_enabled.load(std::memory_order_relaxed)) return;

  if (g_per_launch) {
    if (g_per_launch_sync) {
      g_per_launch_use_activity = false;
      std::lock_guard<std::mutex> lock(g_mutex);
      g_warnings.emplace_back(
          "per_launch_sync enabled: using cuCtxSynchronize() boundaries for isolation (high overhead)");
    } else {
      if (!try_claim_activity_callbacks_owner("sassmetrics")) {
        g_per_launch_use_activity = false;
        std::lock_guard<std::mutex> lock(g_mutex);
        g_warnings.emplace_back(
            "per_launch: CUPTI activity callbacks already owned by another tool; falling back to cuCtxSynchronize() on API_EXIT");
      } else {
        CUptiResult reg = cuptiActivityRegisterCallbacks(activity_buffer_requested, activity_buffer_completed);
        if (reg != CUPTI_SUCCESS) {
          cupti_warn_only(reg,
                          "cuptiActivityRegisterCallbacks(activity_buffer_requested, activity_buffer_completed)");
          g_per_launch_use_activity = false;
        } else {
          CUptiResult ena = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
          if (ena != CUPTI_SUCCESS) {
            cupti_warn_only(ena, "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)");
            ena = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
          }
          if (ena != CUPTI_SUCCESS) {
            cupti_warn_only(ena, "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)");
            g_per_launch_use_activity = false;
          }
        }
      }
      if (!g_per_launch_use_activity) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_warnings.emplace_back(
            "per_launch: kernel activity unavailable; falling back to cuCtxSynchronize() on API_EXIT");
      } else {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_warnings.emplace_back(
            "per_launch(activity) is approximate: overlapping kernels may pollute each other's metrics; set IKP_CUPTI_SASS_PER_LAUNCH_SYNC=1 for isolation");
      }
    }
  }

  CUpti_SubscriberHandle subscriber;
  if (!CUPTI_TRY(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callback_handler, nullptr))) return;
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                                      CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING));
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED));
  (void)CUPTI_TRY(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
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

  std::atexit(finalize);
}

extern "C" int InitializeInjection(void) {
  initialize();
  return 1;
}

}  // namespace

