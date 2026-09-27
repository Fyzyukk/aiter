#include <hip/hip_runtime.h>
#include <cstdint>
#include "pipeline.cuh"

extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if (variant != 20124 || m <= 0 || n <= 0 || k <= 0 || k > 1536 || k % 128 ||
        m % 64 || n % 896 || int64_t(m) * k > limit || int64_t(n) * k > limit ||
        int64_t(m) * n * 2 > limit || !a || !b || !sfa || !sfb || !c ||
        reinterpret_cast<uintptr_t>(a) % 16 || reinterpret_cast<uintptr_t>(b) % 16 ||
        reinterpret_cast<uintptr_t>(c) % 16 || reinterpret_cast<uintptr_t>(sfa) % 16)
        return static_cast<int>(hipErrorInvalidValue);
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a = a; args.ptr_b = b; args.ptr_c = c;
    args.ptr_sfa = sfa; args.ptr_sfb = sfb;
    args.m = m; args.n = n; args.k = k; args.batch = 1;
    args.stride_a = k; args.stride_b = k; args.stride_c = n;
    args.stride_sfa = m; args.stride_sfb = k / 128;
    const dim3 grid(n / 224, 1 + (m - 1) / 192);
    runtime_n224_loop_kernel<<<grid, 512, 0, reinterpret_cast<hipStream_t>(stream)>>>(args);
    return static_cast<int>(hipGetLastError());
}
