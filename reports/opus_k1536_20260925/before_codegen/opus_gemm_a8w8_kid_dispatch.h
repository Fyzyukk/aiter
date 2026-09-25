#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See gen_instances.py:gen_a8w8_kid_dispatch.
//
// Interfaces remain separate even when argument counts match. Missing kernels
// use a size-0 table.
#define GENERATE_A8W8_NOSCALE_KID_DISPATCH_GFX950_SIZE 0
#define GENERATE_A8W8_NOSCALE_KID_DISPATCH_GFX950

#define GENERATE_A8W8_BLOCKSCALE_KID_DISPATCH_GFX950_SIZE 0
#define GENERATE_A8W8_BLOCKSCALE_KID_DISPATCH_GFX950

#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16_SIZE 12
#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16 \
    { 9000, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t> },  \
    { 9010, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t> },  \
    { 9011, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t> },  \
    { 9012, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t> },  \
    { 9020, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1<bf16_t> },  \
    { 9030, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x192x256x128_2x2_16x16x128_1x128x128_tiles1_sfpanel64<bf16_t> },  \
    { 9031, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel64<bf16_t> },  \
    { 9032, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x192x256x128_2x2_16x16x128_1x128x128_tiles1_sfpanel128<bf16_t> },  \
    { 9033, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel128<bf16_t> },  \
    { 9040, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel3_fixedk384<bf16_t> },  \
    { 9041, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel6_fixedk768<bf16_t> },  \
    { 9042, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel8_fixedk1024<bf16_t> },

#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_FP32_SIZE 0
#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_FP32

#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX942_BF16_SIZE 0
#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX942_BF16

#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX942_FP32_SIZE 0
#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX942_FP32

#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_BF16_SIZE 0
#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_BF16

#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_FP32_SIZE 0
#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_FP32

