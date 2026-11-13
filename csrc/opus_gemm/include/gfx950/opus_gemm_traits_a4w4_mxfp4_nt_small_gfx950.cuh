// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel arguments / tile traits of the shuffle small-tile MXFP4 x MXFP4 NT pipeline (gfx950):
// opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh (A_LDS only).  Standard ROCm toolchain (no pin attributes, no opus).
//   A [batch, M, K/2] fp4x2 row-major;  B aiter shuffle_weight(16, 16);  scaleA / scaleB aiter e8m0_shuffle (1x32 UE8M0);
//   C [batch, M, N] bf16.  Any M, N % 16 == 0, K % 256 == 0.  Non-persistent: grid = (N/BN, M/BM, batch * split).
// The kargs and BMODE values are shared with the row-major variant (opus_gemm_traits_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

struct opus_gemm_mxfp4_small_kargs_gfx950 {
    const void* ptr_a; const void* ptr_b; void* ptr_c; const void* ptr_sfa; const void* ptr_sfb;
    int m, n, k, batch;
    int stride_a, stride_c;                 // bytes per A row (K/2); elements per C row
    int stride_sfa, stride_sfb;             // bytes per 32-row block of the shuffled scales (32 * K/32 padded to 8);
                                            //   row-major variant: bytes per scale row (K/32)
    long long stride_a_batch, stride_b_batch, stride_c_batch, stride_sfa_batch, stride_sfb_batch;   // bytes (C: elements)
    int splitk;                             // S >= 1: K split over S blocks (grid.z = batch * S)
    float* ws;                              // [batch][S][Mp][Np] fp32 partials (Mp, Np = M, N rounded up to the tile)
    int* cnt;                               // [batch][tiles] arrival counters, zero on entry, reset by the reducing block
};

// Operand path (BMODE): B_DIRECT / B_LDS in the row-major pipeline, A_LDS in the shuffle pipeline.
enum : int {
    OPUS_MXFP4_SMALL_B_DIRECT = 0,   // A, B, scales global -> registers (no LDS, no barrier in the K loop); KS = 1/2/4
    OPUS_MXFP4_SMALL_B_LDS    = 1,   // A, B, scales global -> LDS by LDS-DMA (D-deep ring, one barrier per K tile)
    OPUS_MXFP4_SMALL_A_LDS    = 2,   // A global -> LDS by LDS-DMA (D-deep ring), B + scales global -> a D-deep register ring,
                                     //   one barrier per ring step (the asm 32x128 structure); KS = 1/2/4
};

// BMODE must be A_LDS.  Tiles 16/32/64 x 32/64/128/256; KS = 1: 1x4 waves along N; KS = 2 / 4: the waves split KS
// ways along K and 4/KS ways along N.  Ring depth limited by (D - 1) * (VMEM per wave per ring step) <= 63.
template<int BM_, int BN_, int D_, int KS_, int BMODE_>
struct opus_gemm_a4w4_mxfp4_nt_small_traits_gfx950 {
    static constexpr int BM = BM_, BN = BN_, D = D_, KS = KS_, BMODE = BMODE_;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int B_K = 256;                      // fp4 elements per K tile (128 bytes per row)
    static_assert(D >= 2);
    static_assert(BMODE == OPUS_MXFP4_SMALL_A_LDS, "shuffle small-tile pipeline: A_LDS only");
    static_assert(KS == 1 || KS == 2 || KS == 4);
};
