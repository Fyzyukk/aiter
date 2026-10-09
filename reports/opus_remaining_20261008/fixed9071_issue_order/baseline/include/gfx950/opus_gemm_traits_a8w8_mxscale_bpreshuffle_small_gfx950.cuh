// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// K is runtime unless FixedK selects a compiler specialization. Output 3 packs
// MFMA16 results; output 4 distributes the FP32 reduction among K waves and
// can mask N16 tails. The same pipeline handles all register configurations.
template<int BlockM, int BlockN, int WaveM, int WaveN, int Prefetch,
         int WaveK = 1, int Output = 0, int BCache = 0,
         int FixedK = 0, bool PadN = false, bool ReuseBScale = false>
struct opus_gemm_small_register_traits_gfx950 {
    static constexpr int B_M = BlockM, B_N = BlockN, B_K = 128;
    static constexpr int T_M = WaveM, T_N = WaveN, T_K = WaveK;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    static constexpr int BLOCK_SIZE = WaveM * WaveN * WaveK * 64;
    static constexpr int E_M = BlockM / (WaveM * W_M);
    static constexpr int E_N = BlockN / (WaveN * W_N);
    static constexpr int PREFETCH = Prefetch, WAVE_K = WaveK;
    static constexpr int B_CACHE = BCache, OUTPUT = Output;
    static constexpr int FIXED_K = FixedK;
    static constexpr bool N_TAIL = PadN, REUSE_B_SCALE = ReuseBScale;
    static_assert(BlockM % (WaveM * W_M) == 0 && BlockN % (WaveN * W_N) == 0);
    static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && Prefetch > 0);
    static_assert(BLOCK_SIZE <= 512 && WaveK > 0);
    static_assert(Output == 0 || Output == 3 || Output == 4);
    static_assert(BCache >= 0 && BCache <= 3);
    static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
    static_assert(!PadN || Output == 4);
    static_assert(!ReuseBScale || 128 % BlockN == 0);
    static_assert(Output != 4 || WaveK > 1);
    static_assert((WaveK - (Output == 4 ? 0 : 1)) * BlockM * BlockN * sizeof(float) <= 160 * 1024);
};

// Short K fills only the LDS slots it needs; longer K reuses the same ring.
// A barrier covers Cluster successive K128 tiles, including a partial tail.
// Output 1 reuses matrix LDS for C; output 2 packs C within each wave.
template<int BlockM, int BlockN, int WaveM, int WaveN,
         int Stages, int Cluster = 1, int Output = 2,
         bool RegisterScales = false, bool XorLds = false, bool EarlyScaleLoads = false,
         bool PrefetchBeforeRead = false, bool ReadOnlyDrain = false,
         int SplitK = 1, int ReduceVec = 4, int ReduceBlock = 128,
         int StoreCache = 0, int FixedK = 0, bool FineMLoads = false>
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
    static constexpr bool FINE_M_LOADS = FineMLoads;
    static constexpr int SPLIT_K = SplitK, REDUCE_VEC = ReduceVec, REDUCE_BLOCK = ReduceBlock;
    static constexpr int STORE_CACHE = StoreCache, FIXED_K = FixedK;
    static constexpr int MAX_K = 16384, MAX_LOOPS = (MAX_K / B_K + SplitK - 1) / SplitK;
    static constexpr int B_GROUPS = (BlockN + 127) / 128;
    static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
    static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) +
                                         (RegisterScales ? 0 : (BlockM + B_GROUPS) * MAX_LOOPS);
    static constexpr int VMEM_TILE = (BlockM + BlockN) * 128 / (BLOCK_SIZE * 16) +
                                     (RegisterScales ? E_M + B_GROUPS : 0);
    static_assert(BlockM % (T_M * 16) == 0 && BlockN % (NUM_WAVES * 16) == 0);
    static_assert(FineMLoads || BlockM % (BLOCK_SIZE / 8) == 0);
    static_assert(E_M > 0 && E_M <= (FineMLoads ? 8 : 4) && E_N > 0 && MAX_LOOPS >= Stages);
    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
    static_assert(ReduceVec >= 4 && ReduceVec % 4 == 0 && ReduceVec <= 16);
    static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= MAX_K && FixedK % 128 == 0));
    static_assert(Cluster > 0 && Stages % Cluster == 0 && Stages >= 2 * Cluster);
    static_assert(MAX_LDS_BYTES <= 160 * 1024 && B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
    static_assert(Output == 1 || Output == 2);
    static_assert(!XorLds || (NUM_WAVES == 4 && A_STAGE % 4096 == 0 && B_STAGE % 4096 == 0));
    static_assert(Output != 1 || (BlockM * (BlockN + 8) * 2 <= A_STAGE + B_STAGE &&
                                 BlockM * BlockN % (BLOCK_SIZE * 8) == 0));

    // The launcher allocates this dynamic LDS size for the actual K.
    static constexpr int lds_bytes(int k) {
        const int loops = (k / B_K + SplitK - 1) / SplitK;
        const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
        return stages * (A_STAGE + B_STAGE) + (RegisterScales ? 0 : (BlockM + B_GROUPS) * loops);
    }
};

using opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 6, 1, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 2, 8>;
using opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 3, 4, 3, 0, 0, false, true>;

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

// Additional exact-kid candidates retain the original configurations above.
// K waves reduce long reductions within one workgroup before BF16 conversion.
using opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x16_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 3, 8, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x32_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 3, 4, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_16x32_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 2, 8, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x32_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 2, 8, 3, 3, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x64_traits_gfx950 =
    opus_gemm_small_register_traits_gfx950<32, 64, 1, 1, 2, 4, 3, 3, 0, false, true>;

// Longer queues amortize barriers across four/two successive K128 tiles.
using opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 12, 4, 2, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950 =
    opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 8, 2, 2, false, false, true>;
