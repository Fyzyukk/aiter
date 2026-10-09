#include <hip/hip_runtime.h>
#include <cstdint>
// Prevent host-only legacy FP8 runtime types from entering the OPUS traits.
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh"
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"

extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if ((variant != 9021 && variant != 9022) || m <= 0 || n <= 0 || k <= 0 ||
        k > 16384 || k % 128 || n % 128 || (variant == 9022 && m % 16) ||
        int64_t(m)*k > limit || int64_t(n)*k > limit || int64_t(m)*n*2 > limit ||
        !a || !b || !sfa || !sfb || !c ||
        reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
        reinterpret_cast<uintptr_t>(c)%16 || reinterpret_cast<uintptr_t>(sfa)%16)
        return static_cast<int>(hipErrorInvalidValue);
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a=a; args.ptr_b=b; args.ptr_sfa=sfa; args.ptr_sfb=sfb; args.ptr_c=c;
    args.m=m; args.n=n; args.k=k; args.batch=1;
    args.stride_a=k; args.stride_b=k; args.stride_c=n;
    args.stride_sfa=m; args.stride_sfb=k/128;
    args.stride_a_batch=m*k; args.stride_b_batch=n*k; args.stride_c_batch=m*n;
    args.stride_sfa_batch=m*(k/128); args.stride_sfb_batch=(n/128)*(k/128);
    const auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    if (variant==9021) {
        using T=opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950;
        gemm_a8w8_mxfp8_scale_4wave_128x128_kernel<T><<<dim3(n/128,(m+127)/128),256,0,hip_stream>>>(args);
    } else {
        using T=opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950;
        gemm_a8w8_mxfp8_scale_4wave_160x128_kernel<T><<<dim3(n/128,(m+159)/160),256,0,hip_stream>>>(args);
    }
    return static_cast<int>(hipGetLastError());
}
