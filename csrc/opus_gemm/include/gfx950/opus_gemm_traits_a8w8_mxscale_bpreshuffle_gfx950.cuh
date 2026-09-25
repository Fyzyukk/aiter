// SPDX-License-Identifier: Apache-2.0
// Source: Fyzyukk/gcnasm, commit 896ed09c4d52f47be92f53f34208a187533615e1
// Upstream directory: opus_gemm/mxfp8_gemm_16x16x128_1x32scale_blockscale_bpreshuffle_4wave
// Modified for AITER: local includes and symbol names; ABI size assertion added.
// See ../../licenses/gcnasm-LICENSE for the upstream license.
#pragma once

#include "../opus_gemm_utils.cuh"

struct opus_mxscale_bpreshuffle_common_gfx950 {
    // Scale vector widths in bytes: one packed operand, or both tile halves.
    static constexpr int VEC_SF = 4;
    static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};

// Runtime-K compact blockscale GEMM: one 256x256 output tile per workgroup.
// Four Wave64s use pinned A/B and selected accumulators with two LDS stages.
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 : opus_mxscale_bpreshuffle_common_gfx950 {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = 256;
    static constexpr int B_N = 256;
    static constexpr int B_K = 128;
    static constexpr bool PAD_M = false;

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
    static constexpr int OUTPUT_TILES_PER_WG = 1;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SCALE_SF = 4;
    static constexpr int VEC_SCALE_SF_PAIR = 8;

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

    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);

    static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
    static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
    // LDS capacity in K128 scale groups; the input has runtime K / GROUP_K groups.
    static constexpr int SCALE_PANEL_K_CAPACITY = 64;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M; // 64 / 16 = 4
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE; // 4 * 4 = 16
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS; // 64 / 16 = 4

    static constexpr int SFA_PANEL_PITCH = B_M; // 256
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY; // 256 * 64
    static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF; // 2 * 64
    static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;

    static_assert(E_M == 4 && E_N == 4 && E_K == 1);
    static_assert(VEC_SCALE_SF == E_M);
    static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
    static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
    static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
    static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
    static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};

// Standalone ABI for the gfx950 four-Wave64 blockscale bpreshuffle kernel.
// A/B/SFA/SFB are inputs and C is BF16 output; no global scratch/workspace is used.
// Keep the field order identical to the validated historical 96-byte ABI.
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
    // SFA is column-major [K/128,M]: stride_sfa is the K-column stride.
    // SFB is contiguous [N/128,K/128]: stride_sfb is the N-block row stride.
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
};

__host__ __device__ inline int opus_mxscale_bpreshuffle_ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static_assert(sizeof(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) == 96);
