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
#include "impl/opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k1536_nooob.cuh"
template __global__ void opus_gemm_mxscale_bpreshuffle_pin_kernel<
    opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_pin_fixed_p16_reset0_k1536_nooob_Traits>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
#endif // host pass or gfx950 device pass
