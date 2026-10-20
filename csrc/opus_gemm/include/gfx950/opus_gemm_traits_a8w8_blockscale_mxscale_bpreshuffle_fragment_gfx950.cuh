// SPDX-License-Identifier: Apache-2.0
#pragma once

// One wave owns a rectangular array of 16x16 accumulators. Wk is one;
// workgroup shape, register queue depth and scale representation are independent.
template<int BM, int BN, int WM, int WN, int Depth = 2,
         bool PackScales = true, int CacheA = 0, int CacheB = 0>
struct opus_gemm_fragment_traits {
    static constexpr int B_M = BM, B_N = BN, B_K = 128;
    static constexpr int T_M = WM, T_N = WN, T_K = 1;
    static constexpr int E_M = BM / (16 * WM), E_N = BN / (16 * WN);
    static constexpr int BLOCK_SIZE = 64 * WM * WN, WG_PER_CU = 1;
    static constexpr int DEPTH = Depth, CACHE_A = CacheA, CACHE_B = CacheB;
    static constexpr bool PACK_SCALES = PackScales;
    static constexpr bool SCALES_PREPACKED = false;
    static constexpr int STAGE_K = 0;
    static constexpr int FIXED_K = 0;
    static constexpr int lds_bytes(int) { return 0; }
    // N repeats are interleaved between waves. Each consecutive group of
    // N_PER_SCALE repeats shares an E8M0 byte, including sub-128 N tiles.
    static constexpr int N_PER_SCALE = 8 / WN;
    static constexpr int N_SCALES = (E_N + N_PER_SCALE - 1) / N_PER_SCALE;
    static_assert(BM > 0 && BN > 0 && WM > 0 && WN > 0 && WM * WN <= 16);
    static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(BN <= 128 ? 128 % BN == 0 : BN % 128 == 0);
    static_assert(Depth >= 1 && Depth <= 4);
};

template<class Base, int K>
struct opus_gemm_fixed_k_fragment_traits : Base {
    static constexpr int FIXED_K = K;
    static_assert(K > 0 && K % 128 == 0);
};

// Fixed K makes every register-ring slot and its last refill known at compile
// time. Scale preloading removes late byte loads from the contraction queue.
template<int BM, int BN, int WM, int WN, int K, int Depth = 3,
         bool PreloadScales = true, int CacheB = 2>
struct opus_gemm_scheduled_fragment_traits
    : opus_gemm_fragment_traits<BM, BN, WM, WN, Depth, true, 0, CacheB> {
    static constexpr int FIXED_K = K;
    static constexpr int PANELS = K / 128;
    static constexpr bool PRELOAD_SCALES = PreloadScales;
    static_assert(K > 0 && K % 128 == 0);
};

// Cooperative transport uses two K panels and one compact scale tile. The
// allocation follows runtime K; short problems do not reserve a maximum-K tile.
template<int BM, int BN, int WM, int WN, int BK, bool PackScales = true, bool RegisterB = false>
struct opus_gemm_shared_fragment_traits : opus_gemm_fragment_traits<BM, BN, WM, WN, 2, PackScales> {
    using Base = opus_gemm_fragment_traits<BM, BN, WM, WN, 2, PackScales>;
    static constexpr int STAGE_K = BK;
    static constexpr bool SCALES_PREPACKED = PackScales;
    static constexpr bool REGISTER_B = RegisterB;
    static constexpr int M_WORDS = (Base::E_M + 3) / 4;
    static constexpr int SCALE_M_BYTES = PackScales ? WM * M_WORDS * 64 : BM;
    static constexpr int DMA = Base::BLOCK_SIZE * 16;
    static constexpr int A_BYTES = ((BM * BK + DMA - 1) / DMA) * DMA;
    static constexpr int B_BYTES = RegisterB ? 0 : ((BN * BK + DMA - 1) / DMA) * DMA;
    static constexpr int PANEL_BYTES = A_BYTES + B_BYTES;
    static constexpr int SCALE_A_OFFSET = 2 * PANEL_BYTES;
    static constexpr int SCALE_A_DMA = Base::BLOCK_SIZE * 4;
    static constexpr int scale_a_bytes(int k) {
        return ((SCALE_M_BYTES * (k / 128) + SCALE_A_DMA - 1) / SCALE_A_DMA) * SCALE_A_DMA;
    }
    static constexpr int scale_b_bytes(int k) {
        constexpr int rows = PackScales ? 1 : (BN + 127) / 128;
        return ((rows * (k / 128) + Base::BLOCK_SIZE - 1) / Base::BLOCK_SIZE) * Base::BLOCK_SIZE * 4;
    }
    static constexpr int lds_bytes(int k) {
        return SCALE_A_OFFSET + scale_a_bytes(k) + scale_b_bytes(k);
    }
    static_assert(BK == 128 || BK == 256 || BK == 512);
    static_assert(lds_bytes(16384) <= 160 * 1024);
};
