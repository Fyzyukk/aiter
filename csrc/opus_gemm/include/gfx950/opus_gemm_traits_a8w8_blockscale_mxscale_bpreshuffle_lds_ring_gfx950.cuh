// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_a8w8_blockscale_mxscale_bpreshuffle_ring_common_gfx950.cuh"

template <int BlockM, int BlockN, int WaveM, int WaveN, int Stages, int SplitK = 1,
          int Output = 2, int WGPerCU = 1, int CacheA = 0, int CacheB = 0, int StoreCache = 2,
          int MaxK = 16384, int FixedK = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_traits {
    static constexpr int B_M = BlockM, B_N = BlockN, B_K = 128;
    static constexpr int T_M = WaveM, T_N = WaveN, T_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128, WARP_SIZE = 64;
    static constexpr int NUM_WAVES = WaveM * WaveN, BLOCK_SIZE = NUM_WAVES * WARP_SIZE;
    static constexpr int WG_PER_CU = WGPerCU;
    static constexpr int E_M = BlockM / (WaveM * 16), E_N = BlockN / (WaveN * 16), E_K = 1;
    static constexpr int HALF_B_M = BlockM, HALF_B_N = BlockN;
    static constexpr int VEC_A = 16, VEC_B = 16, VEC_SCALE_A = 16;
    static constexpr int smem_linear_wave = 1024, smem_padding = 32;
    static constexpr int smem_m_rep = BlockM / 8, smem_n_rep = BlockN / 8;
    static constexpr int A_STAGE = BlockM / 8 * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = BlockN / 8 * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = Stages, OUTPUT = Output, SPLIT_K = SplitK;
    static constexpr int REDUCE_VEC = 4, REDUCE_BLOCK = 128;
    static constexpr int CACHECTL_A = CacheA, CACHECTL_B = CacheB, STORE_CACHE = StoreCache;
    static constexpr int MAX_K = MaxK, FIXED_K = FixedK;
    static constexpr int MAX_LOOPS = (MAX_K / B_K + SplitK - 1) / SplitK;
    static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= MaxK && FixedK % B_K == 0));
    static constexpr int B_GROUPS = (BlockN + 127) / 128;
    static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
    static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) + (BlockM + B_GROUPS) * MAX_LOOPS;
    // Some A-copy waves can be masked on a short tile; the lower count waits conservatively.
    static constexpr int VMEM_TILE = (BlockM + BlockN) * B_K / (BLOCK_SIZE * VEC_A);
    static_assert(WaveM > 0 && WaveM <= 4 && WaveN > 0 && WaveN <= 4);
    static_assert(NUM_WAVES == 4 || NUM_WAVES == 8);
    static_assert(BlockM % (WaveM * 16) == 0 && BlockN % (NUM_WAVES * 16) == 0);
    static_assert(E_M > 0 && E_M <= 8 && E_N > 0 && Stages >= 2 && Stages <= MAX_LOOPS);
    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4);
    static_assert(Output == 0 || Output == 2);
    static_assert(WGPerCU == 1 || WGPerCU == 2);
    static_assert(CacheA == 0 || CacheA == 1 || CacheA == 17);
    static_assert(CacheB == 0 || CacheB == 1 || CacheB == 17);
    static_assert(((MAX_LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
    static constexpr int lds_bytes(int k) {
        const int loops = (k / B_K + SplitK - 1) / SplitK;
        const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
        return stages * (A_STAGE + B_STAGE) + (BlockM + B_GROUPS) * loops;
    }
};
