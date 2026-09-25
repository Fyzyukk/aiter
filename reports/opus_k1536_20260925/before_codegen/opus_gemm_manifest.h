#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <cstdlib>
#include <optional>

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x192x256x128_2x2_16x16x128_1x128x128_tiles1_sfpanel64(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel64(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x192x256x128_2x2_16x16x128_1x128x128_tiles1_sfpanel128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel3_fixedk384(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel6_fixedk768(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel8_fixedk1024(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
