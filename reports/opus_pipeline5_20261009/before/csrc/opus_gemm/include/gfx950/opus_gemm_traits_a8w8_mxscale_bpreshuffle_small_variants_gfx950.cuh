#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"

constexpr int opus_gemm_mxscale_bpreshuffle_small_gcd(int a, int b) { while (b) { int r=a%b; a=b; b=r; } return a; }
template<class Baseline, int Ahead>
struct opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits : Baseline {
    static constexpr int B_AHEAD = Ahead, B_SLOTS = Ahead + 1;
    static constexpr int RING_PERIOD = Baseline::NUM_STAGES * B_SLOTS / opus_gemm_mxscale_bpreshuffle_small_gcd(Baseline::NUM_STAGES, B_SLOTS);
    static constexpr int B_STAGE = 0;
    static constexpr int C_BYTES = Baseline::OUTPUT == 1 ? Baseline::B_M * (Baseline::B_N + 8) * 2 : 0;
    static constexpr int MAX_LDS_BYTES = Baseline::NUM_STAGES * Baseline::A_STAGE +
        (Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * Baseline::MAX_LOOPS);
    static_assert(Ahead >= 1 && Ahead <= 3 && Baseline::T_K == 1);
    static_assert(MAX_LDS_BYTES <= 160 * 1024 && C_BYTES <= 160 * 1024);
    static constexpr int lds_bytes(int k) {
        const int loops = (k / Baseline::B_K + Baseline::SPLIT_K - 1) / Baseline::SPLIT_K;
        const int stages = loops < Baseline::NUM_STAGES ? loops : Baseline::NUM_STAGES;
        const int matrix_scale = stages * Baseline::A_STAGE +
            (Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * loops);
        return matrix_scale > C_BYTES ? matrix_scale : C_BYTES;
    }
};
template<int Actual> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits;
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9043> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9044> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9045> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9046> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9047> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9049> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9055> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9056> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950; };
template<int Actual, int Ahead> using opus_gemm_mxscale_bpreshuffle_small_direct_b_traits = opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits<typename opus_gemm_mxscale_bpreshuffle_small_baseline_traits<Actual>::type, Ahead>;

// K128 arithmetic and raw scales retained. Each global partition completes its
// local K-wave reduction before writing exactly one FP32 partial per output.
template<int BM, int BN, int WaveK>
struct opus_gemm_mxscale_bpreshuffle_register_split_traits
    : opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
    static constexpr int GLOBAL_SPLIT_K = 4;
    static_assert(WaveK == 1 || WaveK == 2);
    static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
                  (BM == 32 && (BN == 32 || BN == 64)));
};

// The existing common fine-M LDS loader, scale handoff, conservative waits,
// output packing and separate reducer are reused with smaller new geometry.
template<int BM, int BN, int WM, int WN, int SplitK>
struct opus_gemm_mxscale_bpreshuffle_narrow_fine_traits
    : opus_gemm_small_lds_traits_gfx950<BM, BN, WM, WN, 4, 1, 2,
        false, false, true, true, true, SplitK, 16, 128, 2, 0, true> {
    static_assert((BM == 48 && BN == 64 && WM == 1 && WN == 4) ||
                  ((BM == 64 || BM == 96) && BN == 128 && WM == 2 && WN == 2));
    static_assert(SplitK == 1 || SplitK == 2);
};
