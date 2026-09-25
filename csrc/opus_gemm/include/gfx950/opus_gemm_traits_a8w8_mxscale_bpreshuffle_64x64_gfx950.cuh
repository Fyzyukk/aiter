// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// The single final kid 9012 configuration: 64x64x128, four matrix LDS stages,
// one A/B register buffer, and a refillable panel of 64 K128 scale groups.
struct opus_gemm_mxscale_bpreshuffle_64x64_traits_gfx950 {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;
    // Existing single-launch, direct BF16 output ABI (tuner splitK=0).
    static constexpr int SPLIT_K = 1;

    static constexpr int B_M = 64;
    static constexpr int B_N = 64;
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

    // Shared layout helpers call their operand tile HALF_B_M/N. This kernel
    // has one complete tile, without the 256x256 kernel's second M/N half.
    static constexpr int HALF_B_M = B_M;
    static constexpr int HALF_B_N = B_N;
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    static constexpr int LDS_WAVE_BYTES = smem_linear_wave;
    static constexpr int LDS_PADDING_BYTES = smem_padding;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 4;
    static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);

    // Scale panel: 64 A row scales and one shared B scale for each K128 group.
    static constexpr int SCALE_PANEL = 64;
    static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
    static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(opus::u32_t);
    static constexpr int SFA_VALUES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
    static constexpr int SFA_PASSES =
        (SFA_BYTES + SFA_VALUES_PER_PASS - 1) / SFA_VALUES_PER_PASS;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;

    // Fragment geometry; the current pipeline uses compiler register allocation.
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

    // Two A and two B buffer loads per K128 tile. While computing K, wait
    // for K+1 operands and leave K+2/K+3 outstanding; the epilogue drains them.
    static constexpr int a_buffer_load_insts = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int VMEM_INSTRUCTIONS_PER_TILE = a_buffer_load_insts + b_buffer_load_insts;
    static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
    static constexpr int LDS_ALLOC_GRANULE = 1280;
    static constexpr int LDS_ALLOC_BYTES =
        (LDS_BYTES + LDS_ALLOC_GRANULE - 1) / LDS_ALLOC_GRANULE * LDS_ALLOC_GRANULE;

    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(E_M == 2 && E_N == 2 && E_K == 1);
    static_assert(B_K == GROUP_K && GROUP_N % B_N == 0);
    static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
    static_assert(A_REGS == 16 && B_REGS == 16 && C_REGS == 16);
    static_assert(PINNED_AGPRS == 48);
    static_assert(VMEM_INSTRUCTIONS_PER_TILE == 4 && VMEM_STEADY_WAIT == 8);
    static_assert(LDS_BYTES == 71936);
    static_assert(LDS_ALLOC_BYTES * 2 <= 160 * 1024);
};
