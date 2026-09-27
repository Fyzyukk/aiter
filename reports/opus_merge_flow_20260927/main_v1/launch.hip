// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include <cstdint>
#include "pipeline_runtime.cuh"

extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if (variant != 21000 || m <= 0 || n <= 0 || k < 128 || k > 16384 ||
        m % 64 || n % 256 || k % 128 ||
        int64_t(m) * k > limit || int64_t(n) * k > limit ||
        int64_t(m) * n * 2 > limit || !a || !b || !sfa || !sfb || !c ||
        reinterpret_cast<uintptr_t>(a) % 16 || reinterpret_cast<uintptr_t>(b) % 16 ||
        reinterpret_cast<uintptr_t>(c) % 16)
        return static_cast<int>(hipErrorInvalidValue);
    const uintptr_t output = reinterpret_cast<uintptr_t>(c);
    const uint64_t output_bytes = uint64_t(m) * n * 2;
    const void* inputs[] = {a, b, sfa, sfb};
    const uint64_t sizes[] = {uint64_t(m) * k, uint64_t(n) * k,
                              uint64_t(m) * (k / 128), uint64_t(n / 128) * (k / 128)};
    for (int i = 0; i < 4; ++i) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(inputs[i]);
        if (output >= begin ? output - begin < sizes[i] : begin - output < output_bytes)
            return static_cast<int>(hipErrorInvalidValue);
    }
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a = a; args.ptr_b = b; args.ptr_c = c;
    args.ptr_sfa = sfa; args.ptr_sfb = sfb;
    args.m = m; args.n = n; args.k = k; args.batch = 1;
    args.stride_a = k; args.stride_b = k; args.stride_c = n;
    args.stride_sfa = m; args.stride_sfb = k / 128;
    args.stride_a_batch = m * k; args.stride_b_batch = n * k;
    args.stride_c_batch = m * n;
    args.stride_sfa_batch = m * (k / 128); args.stride_sfb_batch = (n / 128) * (k / 128);
    const dim3 grid(n / 256, 1 + (m - 1) / 192);
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    gemm_a8w8_mxfp8_bpreshuffle_merged_main_kernel<<<grid, 512, 0, hip_stream>>>(args);
    return static_cast<int>(hipGetLastError());
}
