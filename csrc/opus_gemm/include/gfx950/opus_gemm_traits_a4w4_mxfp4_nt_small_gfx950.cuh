// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Tile traits of the shuffle small-tile MXFP4 x MXFP4 NT pipeline (gfx950):
// opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh (A_LDS only).  Inputs as aiter gemm_a4w4(bpreshuffle=True):
//   A [batch, M, K/2] fp4x2 row-major;  B shuffle_weight(16, 16);  scaleA / scaleB e8m0_shuffle (1x32 UE8M0);
//   C [batch, M, N] bf16.  Any M, N % 16 == 0, K % 256 == 0.  Kernel arguments: opus_gemm_mxfp4_small_kargs_gfx950 with
//   stride_sfa / stride_sfb = bytes per 32-row scale block.  Non-persistent: grid = (N/BN, M/BM, batch * split).
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_common_gfx950.cuh"

// Tiles 16/32/64 x 32/64/128/256.  KS = 1: 1x4 waves along N; KS = 2 / 4: the waves split KS ways along K and 4/KS ways
// along N.  Ring depth limited by (D - 1) * (VMEM per wave per ring step) <= 63.
template<int BM_, int BN_, int D_, int KS_, int BMODE_>
struct opus_gemm_a4w4_mxfp4_nt_small_traits_gfx950 {
    static constexpr int BM = BM_, BN = BN_, D = D_, KS = KS_, BMODE = BMODE_;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int B_K = 256;                      // fp4 elements per K tile (128 bytes per row)
    static_assert(D >= 2, "ring depth >= 2");
    static_assert(BMODE == OPUS_MXFP4_SMALL_A_LDS, "shuffle small-tile pipeline: A_LDS only");
    static_assert(KS == 1 || KS == 2 || KS == 4, "KS = 1 / 2 / 4");
};
