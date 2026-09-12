// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../opus_gemm_utils.cuh"

// gfx950 blockscale bpreshuffle: A is row-major FP8; B has the aiter
// shuffle_weight(layout=(16,16)) byte layout. SFA is E8M0 [K/128,M]
// column-major for logical [M,K/128]; SFB is E8M0 [N/128,K/128].
struct opus_gemm_mxscale_bpreshuffle_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;

    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int stride_sfa;  // SFA K128-column stride in bytes (M for dense input).
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
};

// Runtime-K compact blockscale GEMM imported from the validated standalone
// pipeline. One workgroup computes one 256x256 output tile. Four Wave64s use
// a0:a255 for C and two LDS stages for K128 matrix blocks.
template<bool OUTPUT_BF16_ = false>
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = 256;
    static constexpr int B_N = 256;
    static constexpr int B_K = 128;

    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;

    static_assert(NUM_WAVES == 4);
    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr bool OUTPUT_BF16 = OUTPUT_BF16_;
    static constexpr int OUTPUT_TILES_PER_WG = 1;

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;

    // Compact input groups and the four hardware K32 lane groups are
    // different: each input K128 scale is broadcast to all four lane groups.
    static constexpr int MFMA_SCALE_GROUP_K = 32;
    static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
    static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 32;

    static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;

    static_assert(NUM_KGROUPS == 4);

    static constexpr int a_buffer_load_insts =
        HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts =
        HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts =
        E_M * W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int b_ds_read_insts =
        E_N * W_N * W_K / (WARP_SIZE * VEC_B);

    static constexpr int SMEM_A_ELEMS =
        smem_m_rep * (smem_linear_wave + smem_padding) * 4;
    static constexpr int SMEM_B_ELEMS =
        smem_n_rep * (smem_linear_wave + smem_padding) * 4;
    // Scale-panel capacity is independent of the runtime GEMM K extent.
    static constexpr int SCALE_PANEL_K_TILES = 64;
    static constexpr int SFA_PANEL_PITCH = B_M;
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_TILES;
    static constexpr int SFB_PANEL_BYTES =
        SCALE_N_HALVES * SCALE_PANEL_K_TILES * 4;
    static constexpr int LDS_BYTES =
        SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;

    static_assert(E_M == 4 && E_N == 4 && E_K == 1);
    static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
    static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
    static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};

__host__ __device__ inline int opus_mxscale_bpreshuffle_ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
