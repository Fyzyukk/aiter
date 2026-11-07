// SPDX-License-Identifier: Apache-2.0
#pragma once

struct opus_gemm_mxfp8_tt_4wave_traits {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int B_M = 256, B_N = 256, B_K = 128;
    static constexpr int HALF_B_M = B_M / 2, HALF_B_N = B_N / 2;
    static constexpr int T_M = 2, T_N = 2, T_K = 1;
    static constexpr int E_M = 4, E_N = 4, E_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
    static constexpr int VEC_LDS_A = 8, VEC_LDS_B = 16;
    static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
    static constexpr int VEC_SF = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SCALE_SF = 4;
    static constexpr int VEC_SCALE_SF_PAIR = 8;
    static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
    static constexpr int SCALE_PANEL_K_CAPACITY = 64;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = B_M;
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
    static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 32;
    static constexpr int SMEM_HALF_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int SMEM_HALF_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int SMEM_A_ELEMS = SMEM_HALF_A_ELEMS * 4;
    static constexpr int SMEM_B_ELEMS = SMEM_HALF_B_ELEMS * 4;
    static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
    static_assert(NUM_WAVES == T_M * T_N * T_K && NUM_WAVES == 4);
    static_assert(SMEM_A_ELEMS + SMEM_B_ELEMS == 135168);
    static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};
