#include <hip/hip_runtime.h>
#include <cstdint>
#include "pipeline_short_runtime.cuh"

// Exactly 2 retained kernel instantiation(s); K is always a runtime kargs field.
extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if ((variant != 20010 && variant != 20011) || m <= 0 || n <= 0 ||
        k < 128 || k > 1536 || k % 128 || m % 64 || n % 128 ||
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
    switch (variant) {
        case 20010:
            short_runtime_kernel<short_runtime_traits<128>>
                <<<dim3(n / 128, 1 + (m - 1) / 128), 256, 0,
                   reinterpret_cast<hipStream_t>(stream)>>>(args); break;
        case 20011:
            short_runtime_kernel<short_runtime_traits<160>>
                <<<dim3(n / 128, 1 + (m - 1) / 160), 256, 0,
                   reinterpret_cast<hipStream_t>(stream)>>>(args); break;
    }
    return static_cast<int>(hipGetLastError());
}
