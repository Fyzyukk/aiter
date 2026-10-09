// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// Single 192x224 layout with a maximum twelve-group resident scale panel.
// The kernel loads only its runtime K/128 groups; K is not a template parameter.
struct runtime_n224_loop_traits {
    static constexpr int MAX_K = 1536;
    static constexpr int BLOCK_SIZE = 512;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;
    static constexpr int B_M = 192, B_N = 224, B_K = 128;
    static constexpr int T_M = 4, T_N = 2, T_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = 1;
    static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
    static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
    static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
    static constexpr int MAX_K_TILES = MAX_K / B_K;
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int SCALE_PANEL = MAX_K_TILES;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = 3 * SCALE_PANEL;
    static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
    static constexpr int B_SCALE_PACKS = E_N;
    static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;

    // Invert the unchanged 9031 cooperative global-to-LDS A layout.
    OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
        const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
        constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
        const int pass = matrix_row / rows_per_pass;
        const int remainder = matrix_row % rows_per_pass;
        const int producer_n = remainder / (8 * T_M);
        const int producer_m = remainder % T_M;
        const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
        return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
                   (smem_linear_wave + smem_padding) +
               producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
    }

    static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
    static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
    static_assert(smem_m_rep % NUM_WAVES == 0);
    static_assert(E_M == 3 && E_N == 7 && E_K == 1 && C_REGS == 84);
    static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 109824);
    static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
    static_assert(SFB_BYTES <= BLOCK_SIZE);
    static_assert(LDS_BYTES <= 160 * 1024);
};
