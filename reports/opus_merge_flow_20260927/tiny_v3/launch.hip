// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include <cstdint>
#include "pipeline.cuh"

extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if ((variant != 21220 && variant != 21221) || m <= 0 || n <= 0 ||
        k < 128 || k > 16384 || m % 64 || n % 128 || k % 128 ||
        int64_t(m) * k > limit || int64_t(n) * k > limit || int64_t(m) * n * 2 > limit ||
        !a || !b || !sfa || !sfb || !c ||
        reinterpret_cast<uintptr_t>(a) % 16 || reinterpret_cast<uintptr_t>(b) % 16 ||
        reinterpret_cast<uintptr_t>(c) % 16 || reinterpret_cast<uintptr_t>(sfa) % 16)
        return static_cast<int>(hipErrorInvalidValue);
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a = a; args.ptr_b = b; args.ptr_c = c;
    args.ptr_sfa = sfa; args.ptr_sfb = sfb;
    args.m = m; args.n = n; args.k = k; args.batch = 1;
    args.stride_a = k; args.stride_b = k; args.stride_c = n;
    args.stride_sfa = m; args.stride_sfb = k / 128;
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    if (variant == 21220)
        tiny_runtime_v3_kernel<tiny_runtime_traits<128>><<<dim3(n / 128, m / 64), 256, 0, hip_stream>>>(args);
    else
        tiny_runtime_v3_kernel<tiny_runtime_traits<64>><<<dim3(n / 64, m / 64), 256, 0, hip_stream>>>(args);
    return static_cast<int>(hipGetLastError());
}
