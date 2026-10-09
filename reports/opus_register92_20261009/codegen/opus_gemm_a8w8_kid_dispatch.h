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

#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16_SIZE 105
#define GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16 \
    { 9000, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t> },  \
    { 9001, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_scale_reset_nooob<bf16_t> },  \
    { 9010, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1<bf16_t> },  \
    { 9011, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_unroll4<bf16_t> },  \
    { 9020, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_main<bf16_t> },  \
    { 9021, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_small<bf16_t> },  \
    { 9022, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_small<bf16_t> },  \
    { 9023, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow<bf16_t> },  \
    { 9024, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_narrow<bf16_t> },  \
    { 9030, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1<bf16_t> },  \
    { 9040, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register<bf16_t> },  \
    { 9041, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x16x128_1x1_16x16x128_1x128x128_tiles1_small_register<bf16_t> },  \
    { 9042, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register<bf16_t> },  \
    { 9043, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_small_lds<bf16_t> },  \
    { 9044, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds<bf16_t> },  \
    { 9045, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds<bf16_t> },  \
    { 9046, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_small_lds<bf16_t> },  \
    { 9047, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_small_regscale<bf16_t> },  \
    { 9048, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_small_regscale_xor<bf16_t> },  \
    { 9049, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_small_regscale_xor<bf16_t> },  \
    { 9050, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x16x128_1x1_16x16x128_1x128x128_tiles1_small_register_prefetch<bf16_t> },  \
    { 9051, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_prefetch<bf16_t> },  \
    { 9052, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek<bf16_t> },  \
    { 9053, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek<bf16_t> },  \
    { 9054, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek<bf16_t> },  \
    { 9055, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_small_lds_deep<bf16_t> },  \
    { 9056, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds_deep<bf16_t> },  \
    { 9060, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk1_v4b128<bf16_t> },  \
    { 9061, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk1_v4b128<bf16_t> },  \
    { 9062, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk2_v4b128<bf16_t> },  \
    { 9063, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b128<bf16_t> },  \
    { 9064, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk2_v16b128<bf16_t> },  \
    { 9065, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b128<bf16_t> },  \
    { 9066, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b64<bf16_t> },  \
    { 9067, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128<bf16_t> },  \
    { 9068, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x112x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128<bf16_t> },  \
    { 9069, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128<bf16_t> },  \
    { 9070, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p3_bc3_k7168_cA0cB3<bf16_t> },  \
    { 9071, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x16x48x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p4_bc3_k7168_cA0cB3<bf16_t> },  \
    { 9072, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p4_bc3_k7168_cA0cB3<bf16_t> },  \
    { 9073, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x48x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p3_bc0_k7168_cA0cB0<bf16_t> },  \
    { 92000, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0<bf16_t> },  \
    { 92001, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0<bf16_t> },  \
    { 92002, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0<bf16_t> },  \
    { 92003, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p8_k384<bf16_t> },  \
    { 92004, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p8_k768<bf16_t> },  \
    { 92005, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536<bf16_t> },  \
    { 92006, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536<bf16_t> },  \
    { 92007, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k3072<bf16_t> },  \
    { 92008, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k7168<bf16_t> },  \
    { 92009, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k7168<bf16_t> },  \
    { 92010, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k3072<bf16_t> },  \
    { 92011, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_tile_order_gm8_k7168<bf16_t> },  \
    { 92012, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_tile_order_gm16_k7168<bf16_t> },  \
    { 92020, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_shortk160_p8_k384<bf16_t> },  \
    { 92021, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_shortk160_p8_k768<bf16_t> },  \
    { 92100, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k384_nooob<bf16_t> },  \
    { 92101, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k1536_nooob<bf16_t> },  \
    { 92102, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p32_reset0_k3072_nooob<bf16_t> },  \
    { 92103, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset0_k7168_nooob<bf16_t> },  \
    { 92104, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset0_k16384_nooob<bf16_t> },  \
    { 92110, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset1_k384_nooob<bf16_t> },  \
    { 92111, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset1_k1536_nooob<bf16_t> },  \
    { 92112, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p32_reset1_k3072_nooob<bf16_t> },  \
    { 92113, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset1_k7168_nooob<bf16_t> },  \
    { 92114, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset1_k16384_nooob<bf16_t> },  \
    { 92120, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pad_pin_fixed_k7168<bf16_t> },  \
    { 92200, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead1<bf16_t> },  \
    { 92201, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead2<bf16_t> },  \
    { 92202, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead3<bf16_t> },  \
    { 92203, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead1<bf16_t> },  \
    { 92204, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead2<bf16_t> },  \
    { 92205, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead3<bf16_t> },  \
    { 92206, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead1<bf16_t> },  \
    { 92207, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead2<bf16_t> },  \
    { 92208, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead3<bf16_t> },  \
    { 92209, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead1<bf16_t> },  \
    { 92210, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead2<bf16_t> },  \
    { 92211, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead3<bf16_t> },  \
    { 92212, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead1<bf16_t> },  \
    { 92213, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead2<bf16_t> },  \
    { 92214, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead3<bf16_t> },  \
    { 92215, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead1<bf16_t> },  \
    { 92216, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead2<bf16_t> },  \
    { 92217, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead3<bf16_t> },  \
    { 92218, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead1<bf16_t> },  \
    { 92219, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead2<bf16_t> },  \
    { 92220, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead3<bf16_t> },  \
    { 92221, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead1<bf16_t> },  \
    { 92222, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead2<bf16_t> },  \
    { 92223, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead3<bf16_t> },  \
    { 92310, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x16x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4<bf16_t> },  \
    { 92311, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_128x16x16x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk2_sk4<bf16_t> },  \
    { 92320, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4<bf16_t> },  \
    { 92321, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_128x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk2_sk4<bf16_t> },  \
    { 92330, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x32x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4<bf16_t> },  \
    { 92340, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x32x64x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4<bf16_t> },  \
    { 92410, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x64x128_1x4_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1<bf16_t> },  \
    { 92411, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x64x128_1x4_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2<bf16_t> },  \
    { 92420, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1<bf16_t> },  \
    { 92421, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2<bf16_t> },  \
    { 92430, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1<bf16_t> },  \
    { 92431, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2<bf16_t> },  \
    { 92500, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536<bf16_t> },  \
    { 92501, &opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536<bf16_t> },

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

