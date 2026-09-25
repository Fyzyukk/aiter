// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// Kid 9010: 128x128x128, three matrix LDS stages, one A/B register buffer,
// and a refillable panel of 64 K128 scale groups.
struct opus_gemm_mxscale_bpreshuffle_128x128_traits_gfx950 {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = 128;
    static constexpr int B_N = 128;
    static constexpr int B_K = 128;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = B_K / (T_K * W_K);

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;

    // The shared layout helpers call their operand tile HALF_B_M/N.
    // Here this is one complete tile, with no second M/N half.
    static constexpr int HALF_B_M = B_M;
    static constexpr int HALF_B_N = B_N;
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 3;
    static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);

    // Each K128 group has 128 A row scales and one shared B scale.
    static constexpr int SCALE_PANEL = 64;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL;
    static constexpr int SFA_VALUES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
    static constexpr int SFA_PASSES =
        (SFA_BYTES + SFA_VALUES_PER_PASS - 1) / SFA_VALUES_PER_PASS;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;

    // One operand buffer per thread. A, B and C use disjoint pinned ranges.
    static constexpr int REG_BUFFERS = 1;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
    static constexpr int A_REGS = E_M * W_M * W_K / (WARP_SIZE * sizeof(opus::u32_t));
    static constexpr int B_REGS = E_N * W_N * W_K / (WARP_SIZE * sizeof(opus::u32_t));
    static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
    static constexpr int A_AGPR_BASE = 0;
    static constexpr int B_AGPR_BASE = A_AGPR_BASE + A_REGS;
    static constexpr int C_AGPR_BASE = B_AGPR_BASE + B_REGS;
    static constexpr int PINNED_AGPRS = C_AGPR_BASE + C_REGS;

    // Four A and four B requests per tile. Wait for K+1 while leaving K+2
    // in flight; the drain step waits for all remaining matrix requests.
    static constexpr int a_buffer_load_insts = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int VMEM_INSTRUCTIONS_PER_TILE = a_buffer_load_insts + b_buffer_load_insts;
    static constexpr int VMEM_STEADY_WAIT = VMEM_INSTRUCTIONS_PER_TILE;

    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(E_M == 4 && E_N == 4 && E_K == 1);
    static_assert(B_K == GROUP_K && B_N == GROUP_N);
    static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
    static_assert(A_REGS == 32 && B_REGS == 32 && C_REGS == 64);
    static_assert(PINNED_AGPRS == 128);
    static_assert(VMEM_INSTRUCTIONS_PER_TILE == 8 && VMEM_STEADY_WAIT == 8);
    static_assert(SFA_PASSES == 2 && LDS_BYTES == 109632);
    static_assert(LDS_BYTES <= 160 * 1024);
};
