#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh"

extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if ((variant != 9023 && variant != 9024) || m <= 0 || n <= 0 || k <= 0 ||
        k > 16384 || k % 128 || n % 128 ||
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
    if (variant==9023) {
        if (m>=1024 && n<=1024 && k==7168 && int64_t((m+63)/64)*(n/128)<=256) {
            using T=opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950<4,64,7168>;
            gemm_a8w8_mxfp8_scale_4wave_64x128_kernel<T><<<dim3(n/128,(m+63)/64),256,0,hip_stream>>>(args);
        } else {
            using T=opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_gfx950;
            gemm_a8w8_mxfp8_scale_4wave_64x128_kernel<T><<<dim3(n/128,(m+63)/64),256,0,hip_stream>>>(args);
        }
    } else {
        if (m>=1024 && m<=2048 && n<=1024 && k==7168) {
            using T=opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,4>;
            gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<T><<<dim3(n/64,(m+63)/64),256,0,hip_stream>>>(args);
        } else {
            using T=opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950;
            gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<T><<<dim3(n/64,(m+63)/64),256,0,hip_stream>>>(args);
        }
    }
    return static_cast<int>(hipGetLastError());
}
