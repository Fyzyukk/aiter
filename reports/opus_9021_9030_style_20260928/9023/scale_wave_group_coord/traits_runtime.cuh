// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

struct opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_gfx950 {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = 64;
    static constexpr int B_N = 128;
    static constexpr int B_K = 128;

    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M;
    static constexpr int HALF_B_N = B_N;

    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
    static_assert(B_M == 64 && B_N == 128 && B_K == 128);
    static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);

    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = 1;

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
    static constexpr int A_K_VECTORS = B_K / VEC_A;
    static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
    static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
    static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
    static constexpr int A_LANES_K = WARP_SIZE / W_M;
    static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
    static_assert(E_M == 2 && E_K == 1);
    static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
    static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
    static_assert(B_M % A_ROWS_PER_BLOCK == 0);

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static_assert(B_N <= GROUP_N && B_K == GROUP_K);

    static constexpr int SWIZZLE_PAIR_B_N = 64;
    static constexpr unsigned SWIZZLE_MAX_GRID_N = 16;
    static constexpr unsigned SWIZZLE_PAIR_MIN_GRID_M = 16;
    static constexpr unsigned SWIZZLE_PAIR_MAX_GRID_M = 32;
    static constexpr unsigned SWIZZLE_N_PARTITIONS = 4;
    static constexpr unsigned SWIZZLE_M_PAIR = 2;
    static constexpr unsigned SWIZZLE_PAIR_PARTITIONS = SWIZZLE_N_PARTITIONS * SWIZZLE_M_PAIR;
    static constexpr int SWIZZLE_PAIR_SHIFT = 3;
    static constexpr unsigned SWIZZLE_MIN_TILES = 256;
    static constexpr unsigned SWIZZLE_SMALL_GRID_N = 8;
    static constexpr unsigned SWIZZLE_SMALL_TILES = 768;
    static constexpr unsigned SWIZZLE_SHORT_GRID_M = 24;
    static constexpr unsigned SWIZZLE_LARGE_PARTITIONS = 16;
    static_assert(SWIZZLE_PAIR_PARTITIONS == (1u << SWIZZLE_PAIR_SHIFT));

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 3;
    static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int VMEM_INSTRUCTIONS_PER_TILE =
        B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
    static_assert(NUM_STAGES == 3 && VEC_A == VEC_B);

    static constexpr int SCALE_PANEL = 64;
    static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
    static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
    static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
    static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
    static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
    static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
    static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
    static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
    static constexpr unsigned SFB_REPLICATE = 0x01010101u;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(opus::u32_t);
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
    static_assert(SFA_PRODUCERS_PER_GROUP == 2 && SFA_PACK_CHUNKS == 2 && SFA_WORDS_PER_CHUNK == 2);
    static_assert(WARP_SIZE % SFA_PRODUCERS_PER_GROUP == 0);
    static_assert(SFA_ROWS_PER_REPEAT * E_M == B_M && E_M == sizeof(opus::u16_t));
    static_assert(SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE && SCALE_PANEL <= BLOCK_SIZE);
    static_assert(LDS_BYTES == 80384);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * 2 <= 160 * 1024);
};
