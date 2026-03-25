#pragma once

#include <cupti.h>
#include <cupti_callbacks.h>

#include <atomic>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>

namespace ikp_cupti {

inline std::string json_escape(std::string_view s) {
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
        out.push_back(static_cast<char>(c));
        break;
    }
  }
  return out;
}

inline std::string dirname_of_path(const std::string& p) {
  const size_t pos = p.find_last_of('/');
  if (pos == std::string::npos) return std::string(".");
  if (pos == 0) return std::string("/");
  return p.substr(0, pos);
}

// Resolve a path next to the current injection library (.so).
// Checks both <dir>/<filename> and <dir>/config/<filename>.
inline std::string resolve_sibling_path(const char* filename) {
  Dl_info info{};
  if (dladdr((void*)&resolve_sibling_path, &info) && info.dli_fname && info.dli_fname[0]) {
    const std::string so_path = info.dli_fname;
    const std::string dir = dirname_of_path(so_path);
    // Check <dir>/<filename> first, then <dir>/config/<filename>.
    std::string path = dir + "/" + filename;
    if (access(path.c_str(), R_OK) == 0) return path;
    std::string config_path = dir + "/config/" + filename;
    if (access(config_path.c_str(), R_OK) == 0) return config_path;
    return path;  // return primary path even if missing (for error reporting)
  }
  return std::string("./") + filename;
}

enum class EnvParseStatus { kMissing = 0, kOk = 1, kInvalid = 2 };

// Strict uint64 parsing:
// - rejects negatives
// - rejects trailing garbage (except whitespace)
// - checks ERANGE
// - optional clamp via [minv, maxv]
inline EnvParseStatus parse_uint64_env_strict(const char* name, uint64_t& out, uint64_t minv = 0,
                                              uint64_t maxv = ULLONG_MAX) {
  const char* val = std::getenv(name);
  if (!val || !*val) return EnvParseStatus::kMissing;
  while (*val && std::isspace(static_cast<unsigned char>(*val))) ++val;
  if (!*val) return EnvParseStatus::kMissing;
  if (*val == '-') return EnvParseStatus::kInvalid;

  errno = 0;
  char* end = nullptr;
  unsigned long long x = std::strtoull(val, &end, 10);
  if (errno == ERANGE) return EnvParseStatus::kInvalid;
  if (!end || end == val) return EnvParseStatus::kInvalid;
  while (*end && std::isspace(static_cast<unsigned char>(*end))) ++end;
  if (*end != '\0') return EnvParseStatus::kInvalid;

  uint64_t v = static_cast<uint64_t>(x);
  if (v < minv || v > maxv) return EnvParseStatus::kInvalid;
  out = v;
  return EnvParseStatus::kOk;
}

inline uint64_t parse_uint64_env(const char* name, uint64_t fallback) {
  uint64_t v = 0;
  const auto st = parse_uint64_env_strict(name, v);
  return (st == EnvParseStatus::kOk) ? v : fallback;
}

inline bool parse_bool_env(const char* name, bool fallback) {
  const char* val = std::getenv(name);
  if (!val || !*val) return fallback;
  return std::atoi(val) != 0;
}

struct RegexFilter {
  bool enabled = false;
  bool invalid = false;
  std::string pattern;
  std::regex regex;
  std::string error;

  void set_from_env(const char* env_name) {
    const char* val = std::getenv(env_name);
    if (!val || !*val) return;
    pattern = val;
    try {
      regex = std::regex(pattern, std::regex_constants::optimize);
      enabled = true;
      invalid = false;
      error.clear();
    } catch (const std::regex_error& e) {
      enabled = false;
      invalid = true;
      error = e.what();
    } catch (...) {
      enabled = false;
      invalid = true;
      error = "unknown std::regex error";
    }
  }

  bool match(const char* name) const {
    if (!enabled) return true;
    if (!name) return false;
    return std::regex_search(name, regex);
  }
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

  std::string uid() const { return "ctx" + std::to_string(context_uid) + "-seq" + std::to_string(launch_seq); }
};

class InvocationTracker {
 public:
  void set_filter_from_env(const char* env_name) { filter_.set_from_env(env_name); }

  void handle_kernel_launch(const CUpti_CallbackData* info, CUpti_CallbackId cbid) {
    Invocation inv;
    inv.launch_seq = launch_seq_.fetch_add(1);
    inv.context_uid = info->contextUid;
    inv.correlation_id = info->correlationId;
    inv.kernel_name = info->symbolName ? info->symbolName : "";
    inv.selected = filter_.match(info->symbolName);

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

    std::lock_guard<std::mutex> lock(mutex_);
    invocations_.push_back(inv);
    if (inv.correlation_id != 0) {
      seen_correlation_ids_.insert(inv.correlation_id);
      if (inv.selected) selected_correlation_ids_.insert(inv.correlation_id);
    }
  }

  // Best-effort backfill from kernel activity when driver callbacks miss some paths (e.g. graphs).
  void handle_kernel_activity(uint32_t correlation_id, const char* kernel_name) {
    if (correlation_id == 0) return;
    Invocation inv;
    inv.launch_seq = launch_seq_.fetch_add(1);
    inv.context_uid = 0;
    inv.correlation_id = correlation_id;
    inv.kernel_name = kernel_name ? kernel_name : "";
    inv.selected = filter_.match(kernel_name);
    std::lock_guard<std::mutex> lock(mutex_);
    if (seen_correlation_ids_.count(correlation_id)) return;
    invocations_.push_back(inv);
    seen_correlation_ids_.insert(correlation_id);
    if (inv.selected) selected_correlation_ids_.insert(correlation_id);
  }

  std::vector<Invocation> invocations_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return invocations_;
  }

  bool is_selected_correlation(uint32_t correlation_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return selected_correlation_ids_.count(correlation_id) > 0;
  }

  const RegexFilter& filter() const { return filter_; }

 private:
  mutable std::mutex mutex_;
  std::vector<Invocation> invocations_;
  std::unordered_set<uint32_t> seen_correlation_ids_;
  std::unordered_set<uint32_t> selected_correlation_ids_;
  std::atomic<uint64_t> launch_seq_{0};
  RegexFilter filter_;
};

}  // namespace ikp_cupti

