#pragma once

#include <stdint.h>

constexpr uint32_t kIkpNvbitSampleAll = 0xFFFFFFFFu;
constexpr uint32_t kIkpNvbitAnyRegion = 0xFFFFFFFFu;

enum RegionCounterIdx : uint32_t {
  kInstTotal = 0,
  kInstPredOff = 1,
  kGmemLoad = 2,
  kGmemStore = 3,
  kGmemBytes = 4,
  kSmemLoad = 5,
  kSmemStore = 6,
  kSmemBytes = 7,
  // "local" memory (often register spill / local arrays).
  kLmemLoad = 8,
  kLmemStore = 9,
  kLmemBytes = 10,
  kCounterCount = 11,
};

// Instruction class counters (per region).
enum InstClassIdx : uint32_t {
  kInstClassAluFp32 = 0,
  kInstClassAluInt = 1,
  kInstClassTensorWgmma = 2,
  kInstClassLdGlobal = 3,
  kInstClassStGlobal = 4,
  kInstClassLdShared = 5,
  kInstClassStShared = 6,
  kInstClassLdLocal = 7,
  kInstClassStLocal = 8,
  kInstClassBarrier = 9,
  kInstClassMembar = 10,
  kInstClassBranch = 11,
  kInstClassCall = 12,
  kInstClassRet = 13,
  kInstClassSpecial = 14,  // cp.async / tma / ldgsts
  kInstClassOther = 15,
  kInstClassCount = 16,
};

// Instruction pipe/category counters (per region), for throughput modeling.
enum InstPipeIdx : uint32_t {
  kInstPipeLd = 0,
  kInstPipeSt = 1,
  kInstPipeTex = 2,
  kInstPipeUniform = 3,
  kInstPipeFp32 = 4,
  kInstPipeFp16 = 5,
  kInstPipeFp64 = 6,
  kInstPipeInt = 7,
  kInstPipeSfu = 8,
  kInstPipeTensor = 9,
  kInstPipeBarrier = 10,
  kInstPipeMembar = 11,
  kInstPipeBranch = 12,
  kInstPipeCallRet = 13,
  kInstPipeSpecial = 14,
  kInstPipeOther = 15,
  kInstPipeCount = 16,
};

constexpr uint32_t kBranchDivHistBins = 33;      // 0..32 active lanes (pred true)
constexpr uint32_t kBranchActiveHistBins = 33;   // 0..32 active lanes (__activemask)
constexpr uint32_t kGmemSectorHistBins = 33;     // 1..32 sectors per instruction
constexpr uint32_t kGmemAlignHistBins = 8;       // addr % 128 in 16B buckets
constexpr uint32_t kStrideClassBins = 3;         // 0=contiguous,1=const_stride,2=gather
constexpr uint32_t kSmemBankHistBins = 33;       // 1..32 max conflict degree
constexpr uint32_t kSmemSpanHistBins = 8;        // span buckets
constexpr uint32_t kGmemLineBitsWords = 16;      // 16*64=1024 bits per region

struct PcMapEntry {
  uint64_t pc;
  uint32_t function_id;
  uint32_t region;
};

struct MemTraceEntry {
  uint64_t pc;
  uint64_t addrs[32];
  uint32_t region;
  uint32_t cta_linear;
  uint32_t warp_id;
  uint32_t active_mask;
  uint32_t access_size;
  uint32_t flags;  // bit0=load, bit1=store, bit2=global, bit3=shared, bit4=local
};

struct DeviceParams {
  uint32_t max_regions;
  uint32_t max_depth;
  uint32_t pcmap_cap;
  uint32_t enable_inst_class;
  uint32_t enable_inst_pipe;
  uint32_t enable_bb_count;
  uint32_t enable_bb_hot;
  uint32_t enable_branch_div;
  uint32_t enable_branch_sites;
  uint32_t enable_mem_pattern;
  // Whether to reweight sampled memory events by sample_mem_every_n.
  // - mem_exec: load/store/bytes
  // - mem_pattern: histograms and absolute intensity counters
  // NOTE: mem_trace is intentionally NOT reweighted.
  uint32_t reweight_mem_exec;
  uint32_t reweight_mem_pattern;
  uint32_t sample_cta;
  uint32_t sample_warp;
  uint32_t sample_mem_every_n;
  uint32_t target_region;
  uint32_t iter_begin;
  uint32_t iter_end;
  uint32_t trace_cap;
  uint32_t max_warps;
  uint32_t active_warps;
  uint32_t* d_sp;
  uint32_t* d_stack;
  uint32_t* d_curr;
  uint64_t* d_marker_mismatch_count;
  uint32_t* d_target_iter;
  uint32_t* d_target_depth;
  uint32_t* d_target_active;
  uint64_t* d_counters;
  uint64_t* d_inst_class;
  uint64_t* d_inst_pipe;
  uint64_t* d_bb_exec;
  uint32_t bb_cap;
  uint64_t* d_bb_hot;  // [bb_cap] dynamic BB entry counts (warp-issued)
  uint64_t* d_branch_div_hist;
  uint64_t* d_branch_active_hist;
  uint32_t branch_site_cap;
  uint64_t* d_branch_site_exec;         // [branch_site_cap]
  uint64_t* d_branch_site_taken_warp;   // [branch_site_cap]
  uint64_t* d_branch_site_fall_warp;    // [branch_site_cap]
  uint64_t* d_branch_site_taken_lanes;  // [branch_site_cap]
  uint64_t* d_branch_site_fall_lanes;   // [branch_site_cap]
  uint64_t* d_gmem_sector_hist;
  uint64_t* d_gmem_sectors;
  uint64_t* d_gmem_align_hist;
  uint64_t* d_gmem_stride_hist;
  uint32_t gmem_set_bins;
  uint64_t* d_gmem_set_hist;  // [max_regions * gmem_set_bins] (optional)
  uint64_t* d_smem_bank_hist;
  uint64_t* d_smem_span_hist;
  uint64_t* d_smem_broadcast;
  uint64_t* d_gmem_line_bits;
  PcMapEntry* d_pcmap;
  uint32_t* d_pcmap_count;
  uint32_t* d_mem_exec_sample_counter;
  uint32_t* d_mem_pattern_sample_counter;
  uint32_t* d_trace_sample_counter;
  MemTraceEntry* d_trace;
  uint32_t* d_trace_count;
};

