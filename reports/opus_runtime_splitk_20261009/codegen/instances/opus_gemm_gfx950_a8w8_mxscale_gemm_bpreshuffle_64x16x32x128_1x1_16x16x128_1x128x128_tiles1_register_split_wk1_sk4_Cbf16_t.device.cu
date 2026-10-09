// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See gen_instances.py:_emit_device_tus.
//
// Device-only translation unit for one (kid, dtype) pair.
// Keep both JIT and prebuild host passes on the minimal branch --
// no torch and no full HIP runtime.
#ifndef __HIPCC_RTC__
#define __HIPCC_RTC__ 1
#endif
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__)
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4.cuh"
template __global__ void opus_gemm_mxscale_bpreshuffle_register_runtime_kernel<
    opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_64x16x32x128_1x1_16x16x128_1x128x128_tiles1_register_split_wk1_sk4_Traits>(opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_reduce_runtime_kernel<16, 128>(const float*, opus::bf16_t*, int, int);
#endif // host pass or gfx950 device pass
