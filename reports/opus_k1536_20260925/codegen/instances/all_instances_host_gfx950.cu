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
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x192x256x128_2x2_16x16x128_1x128x128_tiles1_sfpanel128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x192x256x128_2x2_16x16x128_1x128x128_tiles1_sfpanel64.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_nooob.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel128.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel12_fixedk1536_kunroll12.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel12_fixedk1536_kunroll2.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel3_fixedk384.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel64.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel6_fixedk768.cuh"
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel8_fixedk1024.cuh"
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x128x128x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x128x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x64x64x128_2x2_16x16x128_1x128x128_tiles1_nooob<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x192x256x128_2x2_16x16x128_1x128x128_tiles1_sfpanel64<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel64<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x192x256x128_2x2_16x16x128_1x128x128_tiles1_sfpanel128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel128<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel3_fixedk384<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel6_fixedk768<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel8_fixedk1024<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel12_fixedk1536_kunroll12<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
template void
opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_sfpanel12_fixedk1536_kunroll2<bf16_t>(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
#endif // host pass only
