// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

struct opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = 160;
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
    static_assert(B_M == 160 && B_N == 128 && B_K == 128);
    static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);

    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = 1;
    static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
    static_assert(E_M == 5 && E_N == 4 && E_K == 1);

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_OUTPUT = 8;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
    static constexpr int A_REGS_PER_CHUNK = VEC_A / sizeof(unsigned);
    static_assert(C_REGS + E_M * A_CHUNKS_PER_FRAGMENT * A_REGS_PER_CHUNK <= 128,
                  "C followed by A stays in AGPR0:127");

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;

    static constexpr int MIN_K = 128;
    static constexpr int MAX_K = 16384;
    static constexpr int MAX_K_TILES = MAX_K / B_K;
    static_assert(MAX_K_TILES == 128);

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int smem_padding = 32;
    static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
    static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);

    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 76032);

    // Fixed per-wave request counts include buffer-OOB A loads.
    static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
    static_assert(A_VMEM_INSTRUCTIONS == 5 && B_VMEM_INSTRUCTIONS == 4);
    static_assert(VMEM_INSTRUCTIONS_PER_TILE == 9);

    static constexpr int SCALE_PANEL = 32;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
    static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
    static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
    static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
    static_assert(SCALE_PANEL == 32 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
    static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
    static_assert(SFA_PASSES == 2 && SFB_BYTES <= BLOCK_SIZE);

    static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
    static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static_assert(B_SCALE_PACKS == 1);
    static_assert(LDS_BYTES == 81184 && LDS_BYTES <= 80 * 1024);

    // Row stride in BF16 elements for the cooperative output staging tile.
    static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
    static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
    static_assert(C_LDS_ROW_STRIDE_ELEMS == 136);
    static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
    static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
