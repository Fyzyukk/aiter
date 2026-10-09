#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

template<int TileM>
struct opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950 {
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 2;

    static constexpr int B_M = TileM;
    static constexpr int B_N = 64;
    static constexpr int B_K = 128;

    static constexpr int T_M = 1;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int HALF_B_M = B_M;
    static constexpr int HALF_B_N = B_N;
    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = 1;

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
    static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);

    static constexpr int SCALE_PANEL = 32;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
    static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
    static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
    static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
    static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;

    static_assert(B_M == 16 || B_M == 32);
    static_assert(NUM_WAVES == T_M * T_N * T_K && NUM_WAVES == 2);
    static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
    static_assert(B_M % (BLOCK_SIZE / (B_K / VEC_A)) == 0);
    static_assert(B_N % (NUM_WAVES * W_N) == 0);
    static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
    static_assert(E_M <= 2 && E_N == 2 && E_K == 1);
    static_assert(NUM_STAGES == 2 && B_K == GROUP_K && B_N <= GROUP_N);
    static_assert(SFA_PASSES == 1 && B_SCALE_PACKS == 1);
    static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
    static_assert(LDS_BYTES == (B_M == 16 ? 21664 : 26400));
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * MIN_WGS_PER_CU <= 160 * 1024);
};
