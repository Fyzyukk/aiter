// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh"

// One geometry and one capacity for every supported runtime K. The frozen
// matrix layouts are reused; no fixed-K kernel is instantiated by this type.
struct opus_gemm_mxscale_bpreshuffle_main_traits_gfx950
    : opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128> {
    static constexpr int MIN_K = 128;
    static constexpr int MAX_K = 16384;
    static constexpr int LOOP_UNROLL = 2;
    static_assert(B_M == 192 && B_N == 256 && B_K == 128);
    static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
    static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
    static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
    static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
    static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 256);
    static_assert(LDS_BYTES == 143104 && LDS_BYTES <= 160 * 1024);
};
