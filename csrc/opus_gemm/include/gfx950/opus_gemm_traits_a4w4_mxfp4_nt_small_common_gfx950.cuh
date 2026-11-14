// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel arguments shared by the two small-tile MXFP4 x MXFP4 NT pipelines (gfx950), and the operand-path ids of the
// row-major one:
//   shuffle   opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh           (1x4 waves)
//   row-major opus_gemm_traits_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh (B_DIRECT / B_LDS)
// Standard ROCm toolchain (no pin attributes, no opus).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

struct opus_gemm_mxfp4_small_kargs_gfx950 {
    const void* ptr_a; const void* ptr_b; void* ptr_c; const void* ptr_sfa; const void* ptr_sfb;
    int m, n, k, batch;
    int stride_a, stride_c;                 // bytes per A row (K/2); elements per C row
    int stride_sfa, stride_sfb;             // shuffle: bytes per 32-row block of the e8m0_shuffle scales (32 * K/32 padded to 8);
                                            //   row-major: bytes per scale row (K/32)
    long long stride_a_batch, stride_b_batch, stride_c_batch, stride_sfa_batch, stride_sfb_batch;   // bytes (C: elements)
    int splitk;                             // S >= 1: K split over S blocks (grid.z = batch * S)
    float* ws;                              // [batch][S][Mp][Np] fp32 partials (Mp, Np = M, N rounded up to the tile)
    int* cnt;                               // [batch][tiles] arrival counters, zero on entry, reset by the reducing block
};

// Row-major operand path (noshuffle traits BMODE).  The values are part of the kid configs (opus_gemm_common.py), keep
// them stable.
enum : int {
    OPUS_MXFP4_SMALL_B_DIRECT = 0,   // A, B, scales global -> a D-deep register ring (no LDS, no barrier in the K loop);
                                     //   KS = 1/2/4
    OPUS_MXFP4_SMALL_B_LDS    = 1,   // A, B, scales global -> LDS by LDS-DMA (D-deep ring, one barrier per K tile)
};
