// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// Experimental wide-N family. Wave count is an explicit build-time choice;
// neither this family nor its tail contract changes kids 9000/9010/9011.
template<int Waves, int ScalePanel = 64>
struct opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950 {
    static_assert(Waves == 4 || Waves == 8);
    static_assert(ScalePanel == 64 || ScalePanel == 128);
    static constexpr int BLOCK_SIZE = Waves * 64;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = Waves;
    static constexpr int MIN_WGS_PER_CU = 1;
    static constexpr int B_M = 192, B_N = 256, B_K = 128;
    static constexpr int T_M = Waves / 2, T_N = 2, T_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = 1;
    static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
    static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
    static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int SCALE_PANEL = ScalePanel;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
    static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
    static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
    static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
    // Invert the cooperative global->LDS A layout. The imported four-wave
    // register layout assumes T_M==2 and cannot be reused with eight waves.
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
    static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
    static_assert(smem_m_rep % Waves == 0 && smem_n_rep % Waves == 0);
    static_assert(E_N == 8 && C_REGS <= 256);
    static_assert(LDS_BYTES <= 160 * 1024);
};
