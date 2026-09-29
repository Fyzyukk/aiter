// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// Each candidate fixes its tile and prefetch queue; K is a runtime dimension.
// Output 3 packs MFMA16 results within a wave; output 0 uses ordinary stores.
template<int BlockM, int BlockN, int WaveM, int WaveN, int Prefetch,
         int WaveK = 1, int Output = 0, int BCache = 0>
struct opus_gemm_small_register_traits_gfx950 {
    static constexpr int B_M = BlockM, B_N = BlockN, B_K = 128;
    static constexpr int T_M = WaveM, T_N = WaveN, T_K = WaveK;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    static constexpr int BLOCK_SIZE = WaveM * WaveN * WaveK * 64;
    static constexpr int E_M = BlockM / (WaveM * W_M);
    static constexpr int E_N = BlockN / (WaveN * W_N);
    static constexpr int PREFETCH = Prefetch, WAVE_K = WaveK;
    static constexpr int B_CACHE = BCache, OUTPUT = Output;
    static_assert(BlockM % (WaveM * W_M) == 0 && BlockN % (WaveN * W_N) == 0);
    static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && Prefetch > 0);
    static_assert(BLOCK_SIZE <= 512 && WaveK > 0);
    static_assert(Output == 0 || Output == 3);
    static_assert(BCache >= 0 && BCache <= 3);
    static_assert((WaveK - 1) * BlockM * BlockN * sizeof(float) <= 160 * 1024);
};

// Short K fills only the LDS slots it needs; longer K reuses the same ring.
// A barrier covers Cluster successive K128 tiles, including a partial tail.
// Output 1 reuses matrix LDS for C; output 2 packs C within each wave.
template<int BlockM, int BlockN, int WaveM, int WaveN,
         int Stages, int Cluster = 1, int Output = 2,
         bool RegisterScales = false, bool XorLds = false, bool EarlyScaleLoads = false,
         bool PrefetchBeforeRead = false, bool ReadOnlyDrain = false>
struct opus_gemm_small_lds_traits_gfx950 {
    static constexpr int B_M = BlockM, B_N = BlockN, B_K = 128;
    static constexpr int T_M = WaveM, T_N = WaveN, T_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    static constexpr int BLOCK_SIZE = WaveM * WaveN * 64;
    static constexpr int WARP_SIZE = 64, NUM_WAVES = WaveM * WaveN;
    static constexpr int E_M = BlockM / (WaveM * 16), E_N = BlockN / (WaveN * 16), E_K = 1;
    static constexpr int HALF_B_M = BlockM, HALF_B_N = BlockN;
    static constexpr int VEC_A = 16, VEC_B = 16, VEC_SCALE_A = 16;
    static constexpr int smem_linear_wave = 1024, smem_padding = XorLds ? 0 : 32;
    static constexpr int smem_m_rep = BlockM / 8, smem_n_rep = BlockN / 8;
    static constexpr int A_STAGE = BlockM / 8 * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = BlockN / 8 * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = Stages, CLUSTER = Cluster;
    static constexpr int OUTPUT = Output;
    static constexpr bool REGISTER_SCALES = RegisterScales, XOR_LDS = XorLds;
    static constexpr bool EARLY_SCALE_LOADS = EarlyScaleLoads;
    static constexpr bool PREFETCH_BEFORE_READ = PrefetchBeforeRead;
    static constexpr bool READ_ONLY_DRAIN = ReadOnlyDrain;
    static constexpr int MAX_K = 16384, MAX_LOOPS = MAX_K / B_K;
    static constexpr int B_GROUPS = (BlockN + 127) / 128;
    static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
    static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) +
                                         (RegisterScales ? 0 : (BlockM + B_GROUPS) * MAX_LOOPS);
    static constexpr int VMEM_TILE = (BlockM + BlockN) * 128 / (BLOCK_SIZE * 16) +
                                     (RegisterScales ? E_M + B_GROUPS : 0);
    static_assert(BlockM % (BLOCK_SIZE / 8) == 0 && BlockN % (NUM_WAVES * 16) == 0);
    static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && MAX_LOOPS >= Stages);
    static_assert(Cluster > 0 && Stages % Cluster == 0 && Stages >= 2 * Cluster);
    static_assert(MAX_LDS_BYTES <= 160 * 1024 && B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
    static_assert(Output == 1 || Output == 2);
    static_assert(!XorLds || (NUM_WAVES == 4 && A_STAGE % 4096 == 0 && B_STAGE % 4096 == 0));
    static_assert(Output != 1 || (BlockM * (BlockN + 8) * 2 <= A_STAGE + B_STAGE &&
                                 BlockM * BlockN % (BLOCK_SIZE * 8) == 0));

    // The launcher allocates this dynamic LDS size for the actual K.
    static constexpr int lds_bytes(int k) {
        const int loops = k / B_K;
        const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
        return stages * (A_STAGE + B_STAGE) + (RegisterScales ? 0 : (BlockM + B_GROUPS) * loops);
    }
};

using opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 6, 1, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 2, 8>;
using opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 3, 4, 3>;

using opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 8, 2, 2, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 4, 1, 2, false, false, true, true, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<96, 64, 2, 2, 4, 1, 1, false, false, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<64, 128, 4, 2, 6, 2>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<32, 64, 2, 2, 4, 1, 2, true>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x64_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<32, 64, 2, 2, 4, 1, 2, true, true>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<32, 128, 2, 2, 4, 2, 2, true, true, false, true>;
