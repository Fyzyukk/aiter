// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated per-arch host TU (gfx950). See gen_instances.py:_emit_fused_host_tu.
#ifndef __HIP_DEVICE_COMPILE__
#define OPUS_FUSED_HOST_TU 1
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
// Forward declaration only. Specialisations live in per-arch device TUs.
#include "gfx950/opus_gemm_traits_a16w16_gfx950.cuh"
template<int VEC_, int BLOCK_, typename D_OUT,
         bool HAS_BIAS_, typename D_BIAS_,
         bool HAS_OOB_>
__global__ void splitk_reduce_kernel(
    const void* ws_ptr, D_OUT* c_out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    const D_BIAS_* bias, int stride_bias_batch);
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_128x16x16x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk2_sk4.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_128x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk2_sk4.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x112x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k3072.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k7168.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_small.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_shortk160_p8_k384.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_shortk160_p8_k768.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_small.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_prefetch.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x16x48x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p4_bc3_k7168_cA0cB3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pad_pin_fixed_k7168.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k1536_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k384_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset1_k1536_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset1_k384_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p32_reset0_k3072_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p32_reset1_k3072_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset0_k16384_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset0_k7168_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset1_k16384_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset1_k7168_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_scale_reset_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_unroll4.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_small_regscale_xor.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p4_bc3_k7168_cA0cB3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x48x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p3_bc0_k7168_cA0cB0.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_small_lds.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_small_lds_deep.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_small_regscale.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_small_regscale_xor.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x64x128_1x4_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x64x128_1x4_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_narrow.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds_deep.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_tile_order_gm16_k7168.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_tile_order_gm8_k7168.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk1_v4b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk2_v4b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b64.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k3072.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k7168.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p8_k384.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p8_k768.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x16x128_1x1_16x16x128_1x128x128_tiles1_small_register.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x16x128_1x1_16x16x128_1x128x128_tiles1_small_register_prefetch.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p3_bc3_k7168_cA0cB3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_main.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead3.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_small_lds.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk1_v4b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk2_v16b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x16x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x32x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x32x64x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4.cuh"
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_scale_reset_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_unroll4<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_main<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_small<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_small<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_narrow<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x16x128_1x1_16x16x128_1x128x128_tiles1_small_register<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_small_lds<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_small_lds<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_small_regscale<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_small_regscale_xor<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_small_regscale_xor<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x16x128_1x1_16x16x128_1x128x128_tiles1_small_register_prefetch<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_prefetch<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x32x32x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x1_16x16x128_1x128x128_tiles1_small_register_wavek<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_small_lds_deep<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_small_lds_deep<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk1_v4b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk1_v4b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk2_v4b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x80x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk2_v16b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x96x128x128_2x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v16b64<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x112x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x128x128_1x4_16x16x128_1x128x128_tiles1_fine_lds_s4_c1_sk4_v4b128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p3_bc3_k7168_cA0cB3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x16x48x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p4_bc3_k7168_cA0cB3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x32x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p4_bc3_k7168_cA0cB3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x48x128_1x1_16x16x128_1x128x128_tiles1_register_tail_p3_bc0_k7168_cA0cB0<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k0<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p8_k384<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p8_k768<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k3072<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k7168<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k7168<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p32_k3072<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_tile_order_gm8_k7168<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_tile_order_gm16_k7168<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_shortk160_p8_k384<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_shortk160_p8_k768<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k384_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k1536_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p32_reset0_k3072_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset0_k7168_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset0_k16384_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset1_k384_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset1_k1536_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p32_reset1_k3072_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset1_k7168_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p64_reset1_k16384_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pad_pin_fixed_k7168<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9043_ahead3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9044_ahead3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9045_ahead3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x64x128x128_4x2_16x16x128_1x128x128_tiles1_direct_b_a9046_ahead3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9047_ahead3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x128x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9049_ahead3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x32x64x128_1x4_16x16x128_1x128x128_tiles1_direct_b_a9055_ahead3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_direct_b_a9056_ahead3<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x16x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_128x16x16x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk2_sk4<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_128x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk2_sk4<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x32x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x32x64x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x64x128_1x4_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x48x64x128_1x4_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x96x128x128_2x2_16x16x128_1x128x128_tiles1_narrow_fine_s4_c1_sk2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y, std::optional<aiter_tensor_t> workspace);
#endif // host pass only
