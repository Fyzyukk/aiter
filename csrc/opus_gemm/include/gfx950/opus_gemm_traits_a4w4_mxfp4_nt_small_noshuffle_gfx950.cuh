// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Tile traits of the row-major small-tile MXFP4 x MXFP4 NT pipeline (gfx950):
// opus_gemm_pipeline_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh (B_DIRECT / B_LDS).  Inputs as aiter Triton
// gemm_afp4wfp4, no padding:
//   A [batch, M, K/2] fp4x2;  B [batch, N, K/2] fp4x2;  SFA [rows, K/32] / SFB [rows, K/32] UE8M0;  C [batch, M, N] bf16.
//   Any M, N % 8 == 0, K % 256 == 0.  Kernel arguments: opus_gemm_mxfp4_small_kargs_gfx950 with
//   stride_sfa / stride_sfb = bytes per scale row (K/32).  Non-persistent: grid = (N/BN, M/BM, batch * split).
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_common_gfx950.cuh"

// BMODE == B_DIRECT: tiles 16/32/64/128 x 32/64/128/256 (4 waves split KS ways along K and 4/KS ways along N).
// BMODE == B_LDS   : tiles 16/32/64 x 64/128/256 with a 1x4 wave layout and 128 x 128 with a 2x2 wave layout; KS = 1.
template<int BM_, int BN_, int D_, int KS_, int BMODE_>
struct opus_gemm_a4w4_mxfp4_nt_small_noshuffle_traits_gfx950 {
    static constexpr int BM = BM_, BN = BN_, D = D_, KS = KS_, BMODE = BMODE_;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int B_K = 256;                      // fp4 elements per K tile (128 bytes per row)
    static_assert(D >= 2, "ring depth >= 2");
    static_assert(BMODE == OPUS_MXFP4_SMALL_B_DIRECT || BMODE == OPUS_MXFP4_SMALL_B_LDS, "row-major: B_DIRECT / B_LDS only");
    static_assert(BMODE == OPUS_MXFP4_SMALL_B_DIRECT || KS == 1, "B_LDS splits work along M / N only");
};
