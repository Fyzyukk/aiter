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
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_scale_reset_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_unroll4(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_main(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_small(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_small(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_narrow(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x16x128_1x1_16x16x128_1x128x128_tiles1_small_register(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_small_lds(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_small_lds(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_small_regscale(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_small_regscale_xor(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_small_regscale_xor(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x16x128_1x1_16x16x128_1x128x128_tiles1_small_register_prefetch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_prefetch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_small_lds_deep(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds_deep(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk1_v4b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk1_v4b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk2_v4b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk2_v16b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b64(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x112x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p3_bc3_k7168_cA0cB3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x16x48x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p4_bc3_k7168_cA0cB3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p4_bc3_k7168_cA0cB3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x48x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p3_bc0_k7168_cA0cB0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p8_k384(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p8_k768(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k3072(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k7168(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k7168(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k3072(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_tile_order_gm8_k7168(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_tile_order_gm16_k7168(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_shortk160_p8_k384(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_shortk160_p8_k768(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k384_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k1536_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p32_reset0_k3072_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset0_k7168_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset0_k16384_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset1_k384_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset1_k1536_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p32_reset1_k3072_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset1_k7168_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset1_k16384_nooob(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pad_pin_fixed_k7168(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead3(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x16x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_128x16x16x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk2_sk4(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_128x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk2_sk4(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x32x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x32x64x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x64x128_1x4_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x64x128_1x4_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace, int split_k);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);

template <typename D_C>
void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
