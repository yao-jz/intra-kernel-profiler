#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>

#include <unistd.h>

namespace {

static std::string g_output_path = "pmsamp_raw.json";

static uint64_t now_ns() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

static void write_stub() {
  std::ofstream out(g_output_path);
  if (!out.is_open()) return;
  out << "{";
  out << "\"tool\":\"ikp_cupti_pmsamp\"";
  out << ",\"version\":1";
  out << ",\"pid\":" << (uint64_t)getpid();
  out << ",\"timestamp_ns\":" << now_ns();
  out << ",\"not_supported\":true";
  out << ",\"reason\":\"CUDA 12.4 does not provide cupti_pmsampling.h; upgrade to 12.6+\"";
  out << "}\n";
}

static void initialize() {
  const char* out = std::getenv("IKP_CUPTI_PMSAMP_OUT");
  if (out && *out) {
    g_output_path = out;
  } else {
    g_output_path = "pmsamp_raw." + std::to_string((uint64_t)getpid()) + ".json";
  }
  write_stub();
}

extern "C" int InitializeInjection(void) {
  initialize();
  return 1;
}

}  // namespace

