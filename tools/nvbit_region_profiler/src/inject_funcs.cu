#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common.h"

// NOTE: NVBit's reference tools include "utils/utils.h" (provided by NVBIT_CORE).
// In this repo, clangd does not automatically inherit that include path, which
// leads to missing declarations (get_laneid/get_warpid/get_global_warp_id).
// Provide small local equivalents to keep this TU self-contained.
__device__ __forceinline__ uint32_t ikp_nvbit_cta_linear();

__device__ __forceinline__ uint32_t ikp_nvbit_tid_linear() {
  return (uint32_t)(((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x);
}

__device__ __forceinline__ int get_laneid() { return (int)(ikp_nvbit_tid_linear() & 31u); }
__device__ __forceinline__ int get_warpid() { return (int)(ikp_nvbit_tid_linear() >> 5); }

__device__ __forceinline__ uint64_t get_global_warp_id() {
  const uint32_t threads_per_cta = (uint32_t)(blockDim.x * blockDim.y * blockDim.z);
  const uint32_t warps_per_cta = (threads_per_cta + 31u) >> 5;
  return (uint64_t)ikp_nvbit_cta_linear() * (uint64_t)warps_per_cta + (uint64_t)get_warpid();
}

__device__ __forceinline__ uint32_t ikp_nvbit_cta_linear() {
  return (uint32_t)(((blockIdx.z * gridDim.y) + blockIdx.y) * gridDim.x + blockIdx.x);
}

__device__ __forceinline__ bool ikp_nvbit_should_sample(const DeviceParams* p) {
  if (p->sample_cta != kIkpNvbitSampleAll && ikp_nvbit_cta_linear() != p->sample_cta) {
    return false;
  }
  if (p->sample_warp != kIkpNvbitSampleAll && get_warpid() != p->sample_warp) {
    return false;
  }
  return true;
}

__device__ __forceinline__ bool ikp_nvbit_should_record(const DeviceParams* p) {
  // IMPORTANT: always guard by active_warps.
  // The host clamps active_warps to g_max_warps; instrumentation still runs for all
  // warps in the launch, so we must avoid OOB accesses even when sampling "any region".
  uint32_t wid = (uint32_t)get_global_warp_id();
  if (wid >= p->active_warps) return false;
  if (p->target_region == kIkpNvbitAnyRegion) return true;
  return p->d_target_active[wid] != 0;
}

__device__ __forceinline__ int ikp_nvbit_warp_leader_lane(unsigned active_mask) {
  return __ffs((int)active_mask) - 1;
}

// Warp-uniform counter-based sampling:
// - only the leader lane increments the counter
// - decision is broadcast to all active lanes via shfl_sync
__device__ __forceinline__ bool ikp_nvbit_warp_sample_u32(unsigned active_mask, int leader_lane, uint32_t n,
                                                         uint32_t* counter_ptr) {
  if (n <= 1) return true;
  if (!counter_ptr) return true;
  uint32_t do_sample = 0;
  if (get_laneid() == leader_lane) {
    const uint32_t idx = atomicAdd(counter_ptr, 1u);
    if ((n & (n - 1)) == 0) do_sample = ((idx & (n - 1)) == 0) ? 1u : 0u;
    else do_sample = ((idx % n) == 0) ? 1u : 0u;
  }
  do_sample = __shfl_sync(active_mask, do_sample, leader_lane);
  return do_sample != 0;
}

__device__ __forceinline__ uint32_t ikp_nvbit_effective_mem_sample_n(const DeviceParams* p) {
  // sample_mem_every_n <= 1 (or 0) means "no sampling" in our implementation.
  uint32_t n = p ? p->sample_mem_every_n : 1u;
  return (n <= 1u) ? 1u : n;
}

extern "C" __device__ __noinline__ void ikp_nvbit_region_update(uint32_t is_push, uint32_t region_id,
                                                               uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!ikp_nvbit_should_sample(params)) return;
  const unsigned m = __activemask();
  const int leader_lane = ikp_nvbit_warp_leader_lane(m);
  if (get_laneid() != leader_lane) return;

  uint32_t wid = (uint32_t)get_global_warp_id();
  if (wid >= params->active_warps) return;
  uint32_t sp = params->d_sp[wid];
  const uint32_t md = params->max_depth;
  if (is_push) {
    if (sp < md) {
      params->d_stack[wid * md + sp] = region_id;
      params->d_curr[wid] = region_id;
    }
    // Always advance logical depth so pop pairing stays correct on overflow.
    params->d_sp[wid] = sp + 1;
  } else {
    if (sp == 0) return;
    const uint32_t orig_sp = sp;
    sp -= 1;
    params->d_sp[wid] = sp;
    // Optional debug: detect marker push/pop mismatch (only checkable when not overflowing).
    if (params->d_marker_mismatch_count && md > 0 && orig_sp <= md) {
      const uint32_t top = params->d_stack[wid * md + (orig_sp - 1)];
      if (top != region_id) {
        atomicAdd(reinterpret_cast<unsigned long long*>(params->d_marker_mismatch_count), 1ull);
      }
    }
    if (sp == 0) {
      params->d_curr[wid] = 0u;
    } else if (md > 0 && sp <= md) {
      params->d_curr[wid] = params->d_stack[wid * md + (sp - 1)];
    } else if (md > 0) {
      // Still in overflow: keep deepest visible entry.
      params->d_curr[wid] = params->d_stack[wid * md + (md - 1)];
    } else {
      params->d_curr[wid] = 0u;
    }
  }

  if (params->target_region != kIkpNvbitAnyRegion && region_id == params->target_region) {
    if (is_push) {
      uint32_t td = params->d_target_depth[wid];
      if (td == 0) {
        uint32_t iter = params->d_target_iter[wid]++;
        uint32_t active = (iter >= params->iter_begin && iter <= params->iter_end) ? 1u : 0u;
        params->d_target_active[wid] = active;
      }
      params->d_target_depth[wid] = td + 1;
    } else {
      uint32_t td = params->d_target_depth[wid];
      if (td > 0) {
        td -= 1;
        params->d_target_depth[wid] = td;
        if (td == 0) params->d_target_active[wid] = 0u;
      }
    }
  }
}

extern "C" __device__ __noinline__ void ikp_nvbit_inst_exec(int pred, int pred_is_neg, uint32_t class_id,
                                                           uint32_t pipe_id, uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;

  if (pred_is_neg) pred = !pred;
  const int active_mask = __ballot_sync(__activemask(), 1);
  const int pred_mask = __ballot_sync(__activemask(), pred);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;
  if (first_laneid != laneid) return;

  if ((pred_mask & active_mask) == 0) {
    const uint32_t wid = (uint32_t)get_global_warp_id();
    const uint32_t region = params->d_curr[wid];
    if (region < params->max_regions) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_counters[region * kCounterCount + kInstPredOff]), 1ull);
    }
    return;
  }

  const uint32_t wid = (uint32_t)get_global_warp_id();
  if (wid >= params->active_warps) return;
  const uint32_t region = params->d_curr[wid];
  if (region < params->max_regions) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_counters[region * kCounterCount + kInstTotal]), 1ull);
    if (params->enable_inst_class && params->d_inst_class && class_id < kInstClassCount) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_inst_class[region * kInstClassCount + class_id]), 1ull);
    }
    if (params->enable_inst_pipe && params->d_inst_pipe && pipe_id < kInstPipeCount) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_inst_pipe[region * kInstPipeCount + pipe_id]), 1ull);
    }
  }
}

extern "C" __device__ __noinline__ void ikp_nvbit_bb_exec(int pred, int pred_is_neg, uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!params->enable_bb_count) return;
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;
  if (pred_is_neg) pred = !pred;
  const int active_mask = __ballot_sync(__activemask(), 1);
  const int pred_mask = __ballot_sync(__activemask(), pred);
  const uint32_t active_pred_mask = static_cast<uint32_t>(pred_mask & active_mask);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;
  if (first_laneid != laneid) return;
  if (active_pred_mask == 0) return;

  const uint32_t wid = (uint32_t)get_global_warp_id();
  if (wid >= params->active_warps) return;
  const uint32_t region = params->d_curr[wid];
  if (region < params->max_regions && params->d_bb_exec) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_bb_exec[region]), 1ull);
  }
}

extern "C" __device__ __noinline__ void ikp_nvbit_bb_hot(uint32_t bb_id, uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!params->enable_bb_hot) return;
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;
  if (!params->d_bb_hot || params->bb_cap == 0) return;
  if (bb_id >= params->bb_cap) return;
  const unsigned m = __activemask();
  const unsigned active_mask = __ballot_sync(m, 1);
  const int laneid = get_laneid();
  const int leader_lane = ikp_nvbit_warp_leader_lane(active_mask);
  if (laneid != leader_lane) return;
  atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_bb_hot[bb_id]), 1ull);
}

extern "C" __device__ __noinline__ void ikp_nvbit_branch_div(int pred, int pred_is_neg, uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!params->enable_branch_div) return;
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;
  if (pred_is_neg) pred = !pred;
  const unsigned m = __activemask();
  const unsigned active_mask = __ballot_sync(m, 1);
  const unsigned pred_mask = __ballot_sync(m, pred);
  const int laneid = get_laneid();
  const int leader_lane = ikp_nvbit_warp_leader_lane(active_mask);
  if (leader_lane != laneid) return;
  const uint32_t active_lanes = static_cast<uint32_t>(__popc(static_cast<unsigned>(active_mask)));
  const uint32_t true_lanes = static_cast<uint32_t>(__popc(static_cast<unsigned>(pred_mask & active_mask)));
  if (true_lanes > 32) return;

  const uint32_t wid = (uint32_t)get_global_warp_id();
  if (wid >= params->active_warps) return;
  const uint32_t region = params->d_curr[wid];
  if (region < params->max_regions) {
    if (params->d_branch_div_hist) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_branch_div_hist[region * kBranchDivHistBins + true_lanes]), 1ull);
    }
    if (params->d_branch_active_hist && active_lanes <= 32) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_branch_active_hist[region * kBranchActiveHistBins + active_lanes]), 1ull);
    }
  }
}

extern "C" __device__ __noinline__ void ikp_nvbit_branch_site(int pred, int pred_is_neg, uint32_t site_id,
                                                             uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!params->enable_branch_sites) return;
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;
  if (!params->d_branch_site_exec || params->branch_site_cap == 0) return;
  if (site_id >= params->branch_site_cap) return;

  if (pred_is_neg) pred = !pred;
  const unsigned m = __activemask();
  const unsigned active_mask = __ballot_sync(m, 1);
  const unsigned pred_mask = __ballot_sync(m, pred);
  const int laneid = get_laneid();
  const int leader_lane = ikp_nvbit_warp_leader_lane(active_mask);
  if (laneid != leader_lane) return;

  const uint32_t active_lanes = static_cast<uint32_t>(__popc(active_mask));
  const uint32_t true_lanes = static_cast<uint32_t>(__popc(pred_mask & active_mask));
  const uint32_t false_lanes = (active_lanes >= true_lanes) ? (active_lanes - true_lanes) : 0u;

  atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_branch_site_exec[site_id]), 1ull);
  if (true_lanes) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_branch_site_taken_warp[site_id]), 1ull);
    atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_branch_site_taken_lanes[site_id]),
              static_cast<unsigned long long>(true_lanes));
  }
  if (false_lanes) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_branch_site_fall_warp[site_id]), 1ull);
    atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_branch_site_fall_lanes[site_id]),
              static_cast<unsigned long long>(false_lanes));
  }
}

__device__ __forceinline__ uint32_t ikp_nvbit_span_bucket(uint64_t span) {
  if (span < 4) return 0;
  if (span < 8) return 1;
  if (span < 16) return 2;
  if (span < 32) return 3;
  if (span < 64) return 4;
  if (span < 128) return 5;
  if (span < 256) return 6;
  return 7;
}

extern "C" __device__ __noinline__ void ikp_nvbit_mem_pattern(int pred, int pred_is_neg, int mem_space, int is_load,
                                                             int is_store, int access_size, uint64_t pc, uint64_t addr,
                                                             uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  (void)is_load;
  (void)is_store;
  (void)pc;
  if (!params->enable_mem_pattern) return;
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;
  if (pred_is_neg) pred = !pred;

  const unsigned m = __activemask();
  const unsigned active_mask = __ballot_sync(m, 1);
  const unsigned pred_mask = __ballot_sync(m, pred);
  const uint32_t exec_mask = static_cast<uint32_t>(pred_mask & active_mask);
  if (exec_mask == 0) return;

  const int laneid = get_laneid();
  const int leader_lane = ikp_nvbit_warp_leader_lane(active_mask);
  if (!ikp_nvbit_warp_sample_u32(active_mask, leader_lane, params->sample_mem_every_n, params->d_mem_pattern_sample_counter)) {
    return;
  }

  if (access_size <= 0) return;
  const uint32_t access_bytes = static_cast<uint32_t>(access_size);

  // local memory is often spill/local arrays; don't mix it into gmem pattern stats.
  if (mem_space == 3) return;

  // Collect addresses. IMPORTANT: all lanes in `active_mask` must participate in shfl_sync.
  uint64_t addrs[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    uint64_t a = 0ull;
    if (exec_mask & (1u << i)) a = __shfl_sync(active_mask, addr, i);
    addrs[i] = (exec_mask & (1u << i)) ? a : 0ull;
  }
  if (laneid != leader_lane) return;

  const uint32_t wid = (uint32_t)get_global_warp_id();
  if (wid >= params->active_warps) return;
  const uint32_t region = params->d_curr[wid];
  if (region >= params->max_regions) return;

  const uint64_t w = (params->reweight_mem_pattern != 0u) ? static_cast<uint64_t>(ikp_nvbit_effective_mem_sample_n(params)) : 1ull;

  if (mem_space == 1) { // global/generic
    // sectors (32B)
    uint64_t sectors[32];
    int sector_count = 0;
    uint64_t lines[32];
    int line_count = 0;
    int base_lane = __ffs((int)exec_mask) - 1;
    uint64_t base_addr = addrs[base_lane];

    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask & (1u << lane)) == 0) continue;
      uint64_t a = addrs[lane];
      uint64_t end = a + static_cast<uint64_t>(access_bytes - 1u);
      uint64_t sector = a >> 5;
      uint64_t end_sector = end >> 5;
      while (sector <= end_sector && sector_count < 32) {
        bool seen = false;
        for (int j = 0; j < sector_count; ++j) {
          if (sectors[j] == sector) { seen = true; break; }
        }
        if (!seen && sector_count < 32) sectors[sector_count++] = sector;
        sector++;
      }
      uint64_t line = a >> 7;
      uint64_t end_line = end >> 7;
      while (line <= end_line && line_count < 32) {
        bool seen = false;
        for (int j = 0; j < line_count; ++j) {
          if (lines[j] == line) { seen = true; break; }
        }
        if (!seen && line_count < 32) lines[line_count++] = line;
        line++;
      }
    }

    // Approximate cache set histogram using 128B line addresses.
    if (params->d_gmem_set_hist && params->gmem_set_bins) {
      const uint32_t bins = params->gmem_set_bins;
      for (int j = 0; j < line_count; ++j) {
        const uint64_t line = lines[j];
        // Cheap hash to spread line ids.
        uint64_t h = line ^ (line >> 17) ^ (line >> 31);
        uint32_t set = (bins & (bins - 1)) == 0 ? (uint32_t)(h & (bins - 1)) : (uint32_t)(h % bins);
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &params->d_gmem_set_hist[region * bins + set]),
                  static_cast<unsigned long long>(w));
      }
    }

    if (sector_count > 0 && params->d_gmem_sector_hist) {
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &params->d_gmem_sector_hist[region * kGmemSectorHistBins + sector_count]),
                static_cast<unsigned long long>(w));
      atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_gmem_sectors[region]),
                static_cast<unsigned long long>(w * static_cast<uint64_t>(sector_count)));
    }

    // alignment histogram (addr % 128, 16B bins)
    if (params->d_gmem_align_hist) {
      uint32_t align = static_cast<uint32_t>(base_addr & 0x7Fu);
      uint32_t bin = (align >> 4) & 0x7u;
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &params->d_gmem_align_hist[region * kGmemAlignHistBins + bin]),
                static_cast<unsigned long long>(w));
    }

    // stride class histogram
    if (params->d_gmem_stride_hist) {
      uint32_t stride_class = 2; // gather
      uint32_t other_mask = exec_mask & ~(1u << base_lane);
      int next_lane = __ffs(other_mask) - 1;
      if (next_lane < 0) {
        stride_class = 0;
      } else {
        int lane_delta = next_lane - base_lane;
        int64_t delta = static_cast<int64_t>(addrs[next_lane]) - static_cast<int64_t>(base_addr);
        if (lane_delta != 0 && (delta % lane_delta) == 0) {
          int64_t stride = delta / lane_delta;
          bool ok = true;
          for (int lane = 0; lane < 32; ++lane) {
            if ((exec_mask & (1u << lane)) == 0) continue;
            int64_t expected = static_cast<int64_t>(base_addr) +
                               stride * static_cast<int64_t>(lane - base_lane);
            if (static_cast<int64_t>(addrs[lane]) != expected) { ok = false; break; }
          }
          if (ok) {
            stride_class = (stride == access_size) ? 0u : 1u;
          }
        }
      }
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &params->d_gmem_stride_hist[region * kStrideClassBins + stride_class]),
                static_cast<unsigned long long>(w));
    }

    // approximate unique 128B lines (bitset)
    if (params->d_gmem_line_bits) {
      for (int j = 0; j < line_count; ++j) {
        uint64_t line = lines[j];
        uint64_t h = line ^ (line >> 32);
        uint32_t bit = static_cast<uint32_t>(h) & ((kGmemLineBitsWords * 64) - 1);
        uint32_t word = bit >> 6;
        uint64_t mask = 1ull << (bit & 63u);
        atomicOr(reinterpret_cast<unsigned long long*>(
                     &params->d_gmem_line_bits[region * kGmemLineBitsWords + word]),
                 mask);
      }
    }
  } else if (mem_space == 2) { // shared
    uint32_t bank_count[32];
    uint32_t bank_unique[32];
    uint64_t bank_addrs[32][32];
    #pragma unroll
    for (int b = 0; b < 32; ++b) {
      bank_count[b] = 0;
      bank_unique[b] = 0;
    }

    uint64_t min_addr = 0;
    uint64_t max_addr = 0;
    bool init = false;
    if (access_bytes <= 4) {
      for (int lane = 0; lane < 32; ++lane) {
        if ((exec_mask & (1u << lane)) == 0) continue;
        uint64_t a = addrs[lane];
        if (!init) { min_addr = max_addr = a; init = true; }
        if (a < min_addr) min_addr = a;
        if (a > max_addr) max_addr = a;
        uint32_t bank = static_cast<uint32_t>((a >> 2) & 31u);
        bank_count[bank]++;
        if (bank_count[bank] == 1) bank_unique[bank] = 1;
        else {
          bool seen = false;
          for (int lane2 = 0; lane2 < lane; ++lane2) {
            if ((exec_mask & (1u << lane2)) == 0) continue;
            uint64_t a2 = addrs[lane2];
            uint32_t b2 = static_cast<uint32_t>((a2 >> 2) & 31u);
            if (b2 == bank && a2 == a) { seen = true; break; }
          }
          if (!seen) bank_unique[bank]++;
        }
      }
    } else {
      for (int lane = 0; lane < 32; ++lane) {
        if ((exec_mask & (1u << lane)) == 0) continue;
        uint64_t base = addrs[lane];
        uint64_t end = base + static_cast<uint64_t>(access_bytes - 1u);
        if (!init) { min_addr = base; max_addr = end; init = true; }
        if (base < min_addr) min_addr = base;
        if (end > max_addr) max_addr = end;
        for (uint64_t off = 0; off < access_bytes; off += 4u) {
          uint64_t a = base + off;
          uint32_t bank = static_cast<uint32_t>((a >> 2) & 31u);
          bank_count[bank]++;
          bool seen = false;
          for (uint32_t j = 0; j < bank_unique[bank]; ++j) {
            if (bank_addrs[bank][j] == a) { seen = true; break; }
          }
          if (!seen && bank_unique[bank] < 32) {
            bank_addrs[bank][bank_unique[bank]++] = a;
          }
        }
      }
    }

    if (!init) return;
    uint32_t max_conflict = 1;
    uint32_t broadcast_banks = 0;
    for (int b = 0; b < 32; ++b) {
      if (bank_count[b] > 1 && bank_unique[b] == 1) {
        broadcast_banks++;
      }
      if (bank_unique[b] > max_conflict) max_conflict = bank_unique[b];
    }
    if (params->d_smem_bank_hist && max_conflict <= 32) {
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &params->d_smem_bank_hist[region * kSmemBankHistBins + max_conflict]),
                static_cast<unsigned long long>(w));
    }
    if (broadcast_banks > 0 && params->d_smem_broadcast) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&params->d_smem_broadcast[region]),
                static_cast<unsigned long long>(w * static_cast<uint64_t>(broadcast_banks)));
    }
    if (params->d_smem_span_hist) {
      uint64_t span = max_addr - min_addr;
      uint32_t bin = ikp_nvbit_span_bucket(span);
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &params->d_smem_span_hist[region * kSmemSpanHistBins + bin]),
                static_cast<unsigned long long>(w));
    }
  }
}

extern "C" __device__ __noinline__ void ikp_nvbit_mem_exec(int pred, int pred_is_neg, int mem_space, int is_load,
                                                          int is_store, int access_size, uint64_t pc, uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;

  if (pred_is_neg) pred = !pred;

  const unsigned m = __activemask();
  const unsigned active_mask = __ballot_sync(m, 1);
  const unsigned pred_mask = __ballot_sync(m, pred);
  const uint32_t exec_mask = static_cast<uint32_t>(pred_mask & active_mask);
  if (exec_mask == 0) return;

  const int laneid = get_laneid();
  const int leader_lane = ikp_nvbit_warp_leader_lane(active_mask);
  if (!ikp_nvbit_warp_sample_u32(active_mask, leader_lane, params->sample_mem_every_n, params->d_mem_exec_sample_counter)) {
    return;
  }
  if (laneid != leader_lane) return;

  const uint32_t wid = (uint32_t)get_global_warp_id();
  if (wid >= params->active_warps) return;
  const uint32_t region = params->d_curr[wid];
  if (region >= params->max_regions) return;

  const uint32_t lanes = static_cast<uint32_t>(__popc(static_cast<unsigned>(exec_mask)));
  const uint64_t bytes = static_cast<uint64_t>(lanes) * static_cast<uint64_t>(access_size);
  const uint64_t w = (params->reweight_mem_exec != 0u) ? static_cast<uint64_t>(ikp_nvbit_effective_mem_sample_n(params)) : 1ull;
  const unsigned long long one = static_cast<unsigned long long>(w);
  const unsigned long long wbytes = static_cast<unsigned long long>(bytes * w);
  uint64_t* base = &params->d_counters[region * kCounterCount];

  if (mem_space == 1) {
    if (is_load) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kGmemLoad]), one);
    if (is_store) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kGmemStore]), one);
    if (bytes) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kGmemBytes]), wbytes);
  } else if (mem_space == 2) {
    if (is_load) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kSmemLoad]), one);
    if (is_store) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kSmemStore]), one);
    if (bytes) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kSmemBytes]), wbytes);
  } else if (mem_space == 3) {
    if (is_load) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kLmemLoad]), one);
    if (is_store) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kLmemStore]), one);
    if (bytes) atomicAdd(reinterpret_cast<unsigned long long*>(&base[kLmemBytes]), wbytes);
  }
}

extern "C" __device__ __noinline__ void ikp_nvbit_mem_trace(int pred, int pred_is_neg, int mem_space, int is_load,
                                                           int is_store, int access_size, uint64_t pc, uint64_t addr,
                                                           uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;
  if (!params->d_trace || params->trace_cap == 0) return;

  if (pred_is_neg) pred = !pred;

  const unsigned m = __activemask();
  const unsigned active_mask = __ballot_sync(m, 1);
  const unsigned pred_mask = __ballot_sync(m, pred);
  const uint32_t exec_mask = static_cast<uint32_t>(pred_mask & active_mask);
  if (exec_mask == 0) return;

  const int laneid = get_laneid();
  const int leader_lane = ikp_nvbit_warp_leader_lane(active_mask);
  if (!ikp_nvbit_warp_sample_u32(active_mask, leader_lane, params->sample_mem_every_n, params->d_trace_sample_counter)) {
    return;
  }

  uint32_t out_idx = 0;
  uint32_t ok = 1;
  if (laneid == leader_lane) {
    out_idx = atomicAdd(params->d_trace_count, 1u);
    ok = (out_idx < params->trace_cap) ? 1u : 0u;
  }
  ok = __shfl_sync(active_mask, ok, leader_lane);
  out_idx = __shfl_sync(active_mask, out_idx, leader_lane);
  if (!ok) return;

  MemTraceEntry* dst = nullptr;
  if (laneid == leader_lane) {
    dst = &params->d_trace[out_idx];
    dst->pc = pc;
    const uint32_t wid = (uint32_t)get_global_warp_id();
    dst->region = params->d_curr[wid];
    dst->cta_linear = ikp_nvbit_cta_linear();
    dst->warp_id = get_warpid();
    dst->active_mask = exec_mask;
    dst->access_size = static_cast<uint32_t>(access_size);
    uint32_t flags = 0;
    if (is_load) flags |= 1u;
    if (is_store) flags |= 2u;
    if (mem_space == 1) flags |= 4u;
    if (mem_space == 2) flags |= 8u;
    if (mem_space == 3) flags |= 16u;
    dst->flags = flags;
  }

  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    uint64_t a = 0ull;
    if (exec_mask & (1u << i)) a = __shfl_sync(active_mask, addr, i);
    if (dst) dst->addrs[i] = (exec_mask & (1u << i)) ? a : 0ull;
  }
}

extern "C" __device__ __noinline__ void ikp_nvbit_pcmap_record(uint32_t function_id, uint64_t pc, uint64_t p_params) {
  DeviceParams* params = reinterpret_cast<DeviceParams*>(p_params);
  if (!ikp_nvbit_should_sample(params)) return;
  if (!ikp_nvbit_should_record(params)) return;
  const unsigned m = __activemask();
  const int leader_lane = ikp_nvbit_warp_leader_lane(m);
  if (get_laneid() != leader_lane) return;

  uint32_t idx = atomicAdd(params->d_pcmap_count, 1);
  if (idx < params->pcmap_cap) {
    const uint32_t wid = (uint32_t)get_global_warp_id();
    const uint32_t region = params->d_curr[wid];
    params->d_pcmap[idx] = {pc, function_id, region};
  }
}

