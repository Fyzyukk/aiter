// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// One runtime flow with a shape-constant matrix ring: M128 uses three slots
// for single-CTA latency, M160 uses two to retain double-CTA residency.
// Both use the same refillable 32-group scale panel and runtime-K loop.
template<int TileM>
struct opus_gemm_mxscale_bpreshuffle_small_traits_gfx950 {
    static_assert(TileM == 128 || TileM == 160);
    static constexpr int MAX_K = 16384;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;
    static constexpr int B_M = TileM, B_N = 128, B_K = 128;
    static constexpr int T_M = 2, T_N = 2, T_K = 1;
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
    static constexpr int NUM_STAGES = B_M == 128 ? 3 : 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int SCALE_PANEL = 32;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
    static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
    static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
    static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
    // Every wave issues these fixed counts, including buffer-OOB A loads.
    static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;

    // Invert the 9030 four-wave cooperative global-to-LDS A layout.
    // This body owns its constants; it is not inherited from eight-wave traits.
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

    static_assert(NUM_WAVES == 4 && NUM_WAVES == T_M * T_N * T_K);
    static_assert(MAX_K_TILES == 128 && SCALE_PANEL == 32);
    static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
    static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
    static_assert((E_M == 4 || E_M == 5) && E_N == 4 && E_K == 1);
    static_assert(C_REGS + E_M * 8 <= 128, "C followed by A stays in AGPR0:127");
    static_assert(NUM_STAGES == (B_M == 128 ? 3 : 2));
    static_assert(A_VMEM_INSTRUCTIONS == (B_M == 128 ? 4 : 5));
    static_assert(B_VMEM_INSTRUCTIONS == 4);
    static_assert(VMEM_INSTRUCTIONS_PER_TILE == (B_M == 128 ? 8 : 9));
    static_assert(MATRIX_LDS_BYTES == (B_M == 128 ? 101376 : 76032));
    static_assert(LDS_BYTES == (B_M == 128 ? 105504 : 81184));
    static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == (B_M == 128 ? 1 : 2));
    static_assert(SFB_BYTES <= BLOCK_SIZE);
    static_assert(LDS_BYTES <= (B_M == 128 ? 160 : 80) * 1024);
};
