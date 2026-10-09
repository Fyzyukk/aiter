// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <optional>
#endif
#ifdef OPUS_FUSED_HOST_TU
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"
template<typename Traits>
__global__ void opus_gemm_mxscale_bpreshuffle_tiled_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs);
#else
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_tiled_gfx950.cuh"
#endif
using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536_Traits = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_geometry_traits<160,2,16,1536>, 2>;

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536(
    aiter_tensor_t &XQ, aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> workspace)
{
    static_assert(std::is_same_v<D_C, bf16_t>);
    constexpr const char* entry =
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch";
    AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
                WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
                entry, ": XQ/WQ/Y must have matching rank 2 or 3");
    AITER_CHECK(XQ.dim() == 2 ||
                (XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
                entry, ": only batch=1 is supported");
    AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
                entry, ": XQ/WQ/Y must be contiguous");
    AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
                entry, ": XQ/WQ must be FP8");
    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
                entry, ": output dtype must be bf16");

    const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
    AITER_CHECK(m > 0 && n > 0 && k > 0 &&
                m % 16 == 0 && n % 128 == 0 && k % 128 == 0,
                entry, ": requires positive M multiple of 16, N multiple of 128 and K multiple of 128");
    AITER_CHECK(k <= 16384, entry, ": requires K <= 16384");
    AITER_CHECK(k == 1536, entry, ": requires K == 1536");
    AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
                entry, ": XQ/WQ/Y shapes do not match");
    // Bound before narrowing dimensions or multiplying the signed int kargs.
    constexpr int64_t byte_limit = 2147483647;
    AITER_CHECK(m <= byte_limit / k && n <= byte_limit / k &&
                m <= (byte_limit / sizeof(D_C)) / n,
                entry, ": tensor byte extent exceeds signed 32-bit addressing");

    const auto is_e8m0 = [](const aiter_tensor_t& t) {
        return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
    };
    AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
                entry, ": scales must contain one-byte E8M0 values");
    AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
                entry, ": x_scale must be logical [M,K/128]");
    AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
                entry, ": x_scale must have dense column-major storage");
    AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
                w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
                entry, ": w_scale must be row-major [N/128,K/128]");
    const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
    const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
    for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
        const uint64_t bytes = input->numel() * input->element_size();
        const bool overlap = output >= begin
            ? output - begin < bytes : begin - output < output_bytes;
        AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
    }
    AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
                entry, ": XQ/WQ must be 16-byte aligned; Y must be 16-byte aligned");

    AITER_CHECK(reinterpret_cast<uintptr_t>(x_scale.data_ptr()) % 16 == 0,
                entry, ": x_scale must be 16-byte aligned");

    AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
    args.m = m; args.n = n; args.k = k; args.batch = 1;
    args.stride_a = k; args.stride_b = k; args.stride_c = n;
    args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = m * n;
    args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
    args.stride_sfa = m; args.stride_sfb = k / 128;
    args.stride_sfa_batch = m * (k / 128);
    args.stride_sfb_batch = (n / 128) * (k / 128);
    const int tiles_m = (m + 159) / 160;
    // The imported pipeline uses block_id_x for N and block_id_y for M,
    // including its 2x2 tile swizzle when both dimensions are multiples of 512.
    const dim3 grid(n / 128, tiles_m);
    opus_gemm_mxscale_bpreshuffle_tiled_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x160x128x128_2x2_16x16x128_1x128x128_tiles1_geometry_s2_p16_k1536_Traits><<<
        grid, dim3(256), 0, aiter::getCurrentHIPStream()>>>(args);
}
#endif
