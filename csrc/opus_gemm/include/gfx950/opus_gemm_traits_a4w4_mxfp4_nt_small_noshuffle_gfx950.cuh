// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kargs of the row-major small-tile MXFP4 x MXFP4 NT pipeline (gfx950):
// opus_gemm_pipeline_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh (1x4 waves).  Inputs as aiter Triton
// gemm_afp4wfp4, no padding:
//   A [batch, M, K/2] fp4x2;  B [batch, N, K/2] fp4x2;  SFA [rows, K/32] / SFB [rows, K/32] UE8M0;  C [batch, M, N] bf16.
//   Any M, N % 8 == 0, K % 256 == 0.  Non-persistent: grid = (N/B_N, M/B_M, batch * split_k).
// Independent of the shuffle small-tile traits (opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh): both can be included in
// one TU (opus_splitk_ws_handle is guarded by the same macro as in the a16w16 / shuffle traits).
#pragma once

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#include <opus/hip_minimal.hpp>
#endif
#include <opus/opus.hpp>

#ifndef OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
#define OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
// Indirection slot for the split-K fp32 workspace pointer (same struct as opus_gemm_traits_a16w16_gfx950.cuh): captured
// HIP graphs hold the slot address, so a later workspace grow does not leave a dangling pointer in the graph.
struct opus_splitk_ws_handle {
    void*         ptr;    // current backing workspace; null until first grow
    unsigned long bytes;  // current capacity in bytes
};
#endif

// Units: A / B in bytes (fp4x2), C in elements, scales in bytes.
// split_k = S > 1: block z = batch * S + k split writes its fp32 partial tile to ws_handle->ptr, laid out as
// [S][batch][padded_M][padded_N] (padded_M / padded_N = M / N rounded up to B_M / B_N), and splitk_reduce_kernel
// (splitk_reduce_gfx950.cuh) sums the S partials into C.  S = 1: C is written directly, ws_handle is not read.
struct opus_gemm_mxfp4_small_noshuffle_kargs_gfx950 {
    const void* ptr_a; const void* ptr_b; void* ptr_c; const void* ptr_sfa; const void* ptr_sfb;
    int m, n, k, batch;
    int stride_a, stride_c;                 // bytes per A row (K/2); elements per C row
    int stride_sfa, stride_sfb;             // bytes per scale row (K/32)
    long long stride_a_batch, stride_b_batch, stride_c_batch, stride_sfa_batch, stride_sfb_batch;   // bytes (C: elements)
    int split_k;                            // S >= 1 (grid.z = batch * S)
    const opus_splitk_ws_handle* ws_handle; // S > 1: fp32 partials (see above)
};

// Tiles B_M 16/32/64 x B_N 16/32/64/128/256, K tile B_K = 256.  STAGES (D) = ring depth: ring step t (KS K tiles) lives in
// LDS stage t % D.  KS = 1: the 4 waves split B_N (1x4); KS = 2 / 4: the waves split KS ways along K and 4/KS ways along N,
// and their partial tiles are summed through LDS once at the end.  Same structure as the shuffle small traits, except that
// B and the scales go through the LDS ring as well (row-major rows are not MFMA-operand shaped; LDS-DMA copies them
// coalesced and the ds_reads re-shape them).
template<int BM_, int BN_, int D_, int KS_>
struct opus_gemm_a4w4_mxfp4_nt_small_noshuffle_traits_gfx950 {
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;   // 4

    static constexpr int B_M = BM_;
    static constexpr int B_N = BN_;
    static constexpr int B_K = 256;                            // fp4 elements per K tile
    static constexpr int ELEMS_PER_BYTE = 2;
    static constexpr int B_K_BYTES = B_K / ELEMS_PER_BYTE;     // 128
    static constexpr int STAGES = D_;                          // ring depth
    static constexpr int KS = KS_;                             // in-block K split of the waves

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WAVES / KS;                 // waves along N
    static constexpr int T_K = KS;                             // waves along K
    static_assert(NUM_WAVES == T_M * T_N * T_K);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;                            // fp4 elements per MFMA
    static constexpr int W_K_BYTES = W_K / ELEMS_PER_BYTE;     // 64

    static constexpr int E_M = B_M / (W_M * T_M);              // M repeats per wave
    static constexpr int E_N = B_N / (W_N * T_N);              // N repeats per wave
    static constexpr int E_K = B_K / W_K;                      // 2 MFMAs along K per K tile
    static constexpr int WAVE_N = B_N / T_N;                   // columns per wave

    static constexpr int VEC_A = 16;                           // bytes = 32 fp4 = one MFMA operand
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;                            // fp32 per lane per 16x16 C fragment

    // A and B: global -> LDS by LDS-DMA, one issue = 64 lanes x 16 B = 8 rows x 128 B.  An 8-row chunk takes
    // smem_linear_wave + smem_padding = 1088 B (row r at r * 128, 16 B piece c at slot c ^ ((r >> 1) & 3)).
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;               // 1024
    static constexpr int smem_padding = 64;
    static constexpr int smem_sub = smem_linear_wave / B_K_BYTES;            // 8 rows per chunk
    static constexpr int CHUNK_BYTES = smem_linear_wave + smem_padding;      // 1088
    static constexpr int A_CHUNKS = B_M / smem_sub;                          // chunks per K tile
    static constexpr int B_CHUNKS = B_N / smem_sub;
    static constexpr int A_TILE_BYTES = A_CHUNKS * CHUNK_BYTES;
    static constexpr int B_TILE_BYTES = B_CHUNKS * CHUNK_BYTES;

    // Scales: row-major [rows][K/32], 8 bytes per row per K tile, global -> LDS by a block-wide 4-byte LDS-DMA (one issue =
    // 32 rows x 8 B, lane l copies row l / 2, bytes [4 (l % 2), +4)): a [row][8 B] image per K tile, max(B_M, 32) A rows and
    // max(B_N, 32) B rows.  Read back as one ds_read_b64 per 16-row block; one v_perm puts K groups lane / 16 and 4 + lane / 16
    // into bytes 0 / 1, picked by the MFMA opsel = kk.
    static constexpr int SF_GROUP = 32;                                      // fp4 elements per scale
    static constexpr int SF_K_TILE_BYTES = B_K / SF_GROUP;                   // 8
    static constexpr int VEC_SF = SF_K_TILE_BYTES;                           // one ds_read_b64 per 16-row block
    static constexpr int VEC_SF_DMA = 4;                                     // bytes per lane of the scale LDS-DMA
    static constexpr int SF_DMA_ROWS = WARP_SIZE * VEC_SF_DMA / SF_K_TILE_BYTES;   // 32 rows per issue
    static constexpr int SF_CHUNK_BYTES = SF_DMA_ROWS * SF_K_TILE_BYTES;    // 256
    static constexpr int SFA_ROWS = B_M < SF_DMA_ROWS ? SF_DMA_ROWS : B_M;
    static constexpr int SFB_ROWS = B_N < SF_DMA_ROWS ? SF_DMA_ROWS : B_N;
    static constexpr int SFA_CHUNKS = SFA_ROWS / SF_DMA_ROWS;
    static constexpr int SF_CHUNKS = SFA_CHUNKS + SFB_ROWS / SF_DMA_ROWS;    // per K tile, A rows first
    static constexpr int SFA_TILE_BYTES = SFA_ROWS * SF_K_TILE_BYTES;
    static constexpr int SFB_TILE_BYTES = SFB_ROWS * SF_K_TILE_BYTES;

    // Ring stage = KS K tiles: [A tiles][B tiles][A scale tiles][B scale tiles].
    static constexpr int STAGE_B_OFF = KS * A_TILE_BYTES;
    static constexpr int STAGE_SFA_OFF = STAGE_B_OFF + KS * B_TILE_BYTES;
    static constexpr int STAGE_SFB_OFF = STAGE_SFA_OFF + KS * SFA_TILE_BYTES;
    static constexpr int STAGE_BYTES = STAGE_SFB_OFF + KS * SFB_TILE_BYTES;
    // LDS-DMA issues per wave per ring step (spread over the waves); with fewer chunks than waves the last waves repeat the
    // last chunk (same data to the same LDS bytes), so every wave issues the same number of VMEM instructions.
    static constexpr int A_DMA_CHUNKS = KS * A_CHUNKS;
    static constexpr int B_DMA_CHUNKS = KS * B_CHUNKS;
    static constexpr int SF_DMA_CHUNKS = KS * SF_CHUNKS;
    static constexpr int a_buffer_load_insts = (A_DMA_CHUNKS + NUM_WAVES - 1) / NUM_WAVES;
    static constexpr int b_buffer_load_insts = (B_DMA_CHUNKS + NUM_WAVES - 1) / NUM_WAVES;
    static constexpr int sf_buffer_load_insts = (SF_DMA_CHUNKS + NUM_WAVES - 1) / NUM_WAVES;

    // VMEM instructions per wave per ring step (the counted vmcnt waits are in units of this).
    static constexpr int VMEM_PER_STEP = a_buffer_load_insts + b_buffer_load_insts + sf_buffer_load_insts;
    // LDS reads per wave per ring step (one step ahead; sched_group_barrier).
    static constexpr int DS_READS_PER_STEP = E_K * (E_M + E_N) + E_M + E_N;

    // KS > 1: groups 1..KS-1 park their partial tiles in LDS, group 0 adds them.
    static constexpr int RED_BYTES = KS > 1 ? (KS - 1) * T_N * E_M * E_N * WARP_SIZE * VEC_C * 4 : 0;
    static constexpr int LDS_BYTES = STAGES * STAGE_BYTES + RED_BYTES;

    static_assert(B_M == 16 || B_M == 32 || B_M == 64);
    static_assert(B_N == 16 || B_N == 32 || B_N == 64 || B_N == 128 || B_N == 256);
    static_assert(KS == 1 || KS == 2 || KS == 4, "KS = 1 / 2 / 4");
    static_assert(E_N >= 1 && WAVE_N % W_N == 0);
    static_assert(STAGES >= 2, "ring depth >= 2");
    static_assert(LDS_BYTES <= 160 * 1024, "LDS ring too deep for this tile");
    static_assert((STAGES - 1) * VMEM_PER_STEP <= 63, "vmcnt range: ring too deep for this tile");
    static_assert(VEC_A * ELEMS_PER_BYTE == W_K / (WARP_SIZE / W_M), "one b128 = one fp4 MFMA operand");
};
