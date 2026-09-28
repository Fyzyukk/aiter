// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// Fixed 64x64 geometry. K is a runtime dimension, never a template.
struct opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950 {
    static constexpr int BLOCK_SIZE = 256, WARP_SIZE = 64, NUM_WAVES = 4;
    static constexpr int MIN_WGS_PER_CU = 1;
    static constexpr int B_M = 64, B_N = 64, B_K = 128;
    static constexpr int T_M = 2, T_N = 2, T_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    static constexpr int E_M = 2, E_N = 2, E_K = 1;
    static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4, VEC_SCALE_A = 16;
    static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
    static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    // Keep the measured four-stage DMA depth and two-workgroup LDS budget.
    static constexpr int NUM_STAGES = 4;
    static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int SCALE_PANEL = 64;
    static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
    static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(opus::u32_t);
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static constexpr int A_CHUNKS_PER_FRAGMENT = 2, B_CHUNKS_PER_FRAGMENT = 2;
    static constexpr int VMEM_INSTRUCTIONS_PER_TILE = (B_M + B_N) * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
    static_assert(LDS_BYTES == 71936);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * 2 <= 160 * 1024);
};
