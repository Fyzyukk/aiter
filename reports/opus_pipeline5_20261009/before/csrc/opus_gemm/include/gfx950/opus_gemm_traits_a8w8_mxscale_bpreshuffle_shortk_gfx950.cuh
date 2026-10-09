#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"

// Fixed short-K variants preserve the audited panel8/SFA one-pass pipeline.
template<int FixedK>
struct opus_gemm_mxscale_bpreshuffle_shortk_traits
    : opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
    static constexpr int FIXED_K = FixedK;
    static constexpr int SCALE_PANEL = 8;
    static constexpr int MAX_K = FIXED_K;
    static constexpr int MAX_K_TILES = FIXED_K / B_K;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
    static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;

    static_assert(FIXED_K == 384 || FIXED_K == 768);
    static_assert(MAX_K_TILES == 3 || MAX_K_TILES == 6);
    static_assert(MAX_K_TILES <= SCALE_PANEL && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
    static_assert(SFA_BYTES == 1280 && SFB_BYTES == 8 && SFA_PASSES == 1);
    static_assert(SFA_VECTORS_PER_GROUP == 10 && B_M % VEC_SCALE_A == 0);
    static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 76032 && LDS_BYTES == 77320);
    static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
};
