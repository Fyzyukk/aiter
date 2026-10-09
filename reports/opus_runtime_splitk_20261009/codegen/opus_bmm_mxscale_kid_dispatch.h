#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated. Do not edit.
#define GENERATE_BMM_MXSCALE_KID_DISPATCH_SIZE 45
#define GENERATE_BMM_MXSCALE_KID_DISPATCH(CTYPE) \
    { 8000, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x128x128_2x1_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8032, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x128x128_2x1_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8064, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x64x128x128_2x1_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8100, &opus_bmm_a8w8_mxscale_flatmm_fused_256x32x128x128_2x1_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8128, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8131, &opus_bmm_a8w8_mxscale_flatmm_mouter_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8132, &opus_bmm_a8w8_mxscale_flatmm_wave8n2_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8134, &opus_bmm_a8w8_mxscale_flatmm_wave4m2_selfload_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8137, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1_scaleprefetch<CTYPE> },  \
    { 8138, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x64x128x256_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8139, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x128x64x256_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8142, &opus_bmm_a8w8_mxscale_flatmm_wave4m2_selfload_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1_ssw<CTYPE> },  \
    { 8144, &opus_bmm_a8w8_mxscale_flatmm_mouter_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1_ssw<CTYPE> },  \
    { 8148, &opus_bmm_a8w8_mxscale_flatmm_wave4m2_selfload_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1_ssw_psod<CTYPE> },  \
    { 8149, &opus_bmm_a8w8_mxscale_pipeline_512x128x256x128_2x1_16x16x128_1x128x128<CTYPE> },  \
    { 8150, &opus_bmm_a8w8_mxscale_pipeline_512x256x256x128_2x1_16x16x128_1x128x128<CTYPE> },  \
    { 8151, &opus_bmm_a8w8_mxscale_pipeline_512x256x256x128_2x1_16x16x128_1x128x128_k1024<CTYPE> },  \
    { 8152, &opus_bmm_a8w8_mxscale_pipeline_512x256x256x128_2x1_16x16x128_1x128x128_k1024lb1<CTYPE> },  \
    { 8158, &opus_bmm_a8w8_mxscale_pipeline_512x256x256x128_2x1_16x16x128_1x128x128_preload_sf<CTYPE> },  \
    { 8160, &opus_bmm_a8w8_mxscale_flatmm_mouter_tunable_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8161, &opus_bmm_a8w8_mxscale_flatmm_mouter_tunable_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1_ssw<CTYPE> },  \
    { 8162, &opus_bmm_a8w8_mxscale_flatmm_minterleave_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8163, &opus_bmm_a8w8_mxscale_flatmm_minterleave_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1_skip_scale_wait<CTYPE> },  \
    { 8256, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x256x128_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8311, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x16x32x512_1x2_16x16x128_1x128x128_wgpcu2_scaleprefetch<CTYPE> },  \
    { 8312, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x16x128x256_1x2_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8313, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x16x64x256_1x2_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8314, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x16x32x512_1x2_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8316, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x16x32x256_1x2_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8317, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x16x32x256_1x2_16x16x128_1x128x128_wgpcu2_scaleprefetch<CTYPE> },  \
    { 8318, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x16x32x128_1x2_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8319, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x16x32x256_1x2_16x16x128_1x128x128_wgpcu4<CTYPE> },  \
    { 8320, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x64x32x256_2x1_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8321, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x32x256_2x1_16x16x128_1x128x128_wgpcu2_scaleprefetch<CTYPE> },  \
    { 8322, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x64x32x256_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8323, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x32x128_2x1_16x16x128_1x128x128_wgpcu2_scaleprefetch<CTYPE> },  \
    { 8324, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x64x32x256_2x1_16x16x128_1x128x128_wgpcu2_sfpreload<CTYPE> },  \
    { 8325, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x128x128x128_2x1_16x16x128_1x128x128_wgpcu1_sfpreload<CTYPE> },  \
    { 8326, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x128x64x256_2x1_16x16x128_1x128x128_wgpcu1_sfpreload<CTYPE> },  \
    { 8327, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x64x128x256_2x1_16x16x128_1x128x128_wgpcu1_sfpreload<CTYPE> },  \
    { 8640, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x64x256_2x1_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8642, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x64x256_2x1_16x16x128_1x128x128_wgpcu1<CTYPE> },  \
    { 8646, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x64x256_2x1_16x16x128_1x128x128_wgpcu2_selfload<CTYPE> },  \
    { 8650, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x64x64x128_2x1_16x16x128_1x128x128_wgpcu2<CTYPE> },  \
    { 8653, &opus_bmm_a8w8_mxscale_flatmm_splitk_256x64x64x128_2x1_16x16x128_1x128x128_wgpcu2_scaleprefetch<CTYPE> },

