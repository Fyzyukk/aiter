#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"

constexpr int opus_private_small_gcd(int a, int b) { while (b) { int r=a%b; a=b; b=r; } return a; }
template<class Baseline, int Ahead>
struct opus_private_small_direct_b_traits : Baseline {
    static constexpr int B_AHEAD = Ahead, B_SLOTS = Ahead + 1;
    static constexpr int RING_PERIOD = Baseline::NUM_STAGES * B_SLOTS / opus_private_small_gcd(Baseline::NUM_STAGES, B_SLOTS);
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
template<int Actual> struct opus_private_small_baseline;
template<> struct opus_private_small_baseline<9043> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950; };
template<> struct opus_private_small_baseline<9044> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950; };
template<> struct opus_private_small_baseline<9045> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950; };
template<> struct opus_private_small_baseline<9046> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950; };
template<> struct opus_private_small_baseline<9047> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950; };
template<> struct opus_private_small_baseline<9049> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950; };
template<> struct opus_private_small_baseline<9055> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950; };
template<> struct opus_private_small_baseline<9056> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950; };
template<int Actual, int Ahead> using opus_private_small_traits = opus_private_small_direct_b_traits<typename opus_private_small_baseline<Actual>::type, Ahead>;
