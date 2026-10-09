#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "contract.h"
#if defined(SMALL_DIRECT_B_AHEAD)
#include "candidate/pipeline.cuh"
#else
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh"
#include "candidate/traits.cuh"
#endif

template<int Actual>
void opus_private_small_launch(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args, hipStream_t stream) {
#if defined(SMALL_DIRECT_B_AHEAD)
    using T = opus_private_small_traits<Actual, SMALL_DIRECT_B_AHEAD>;
#else
    using T = typename opus_private_small_baseline<Actual>::type;
#endif
    static_assert(T::SPLIT_K == 1 && T::FIXED_K == 0);
    const dim3 grid(args.n / T::B_N, 1 + (args.m - 1) / T::B_M, T::SPLIT_K);
#if defined(SMALL_DIRECT_B_AHEAD)
    opus_private_small_a_lds_direct_b_kernel<T><<<grid,T::BLOCK_SIZE,T::lds_bytes(args.k),stream>>>(args);
#else
    gemm_a8w8_mxfp8_scale_small_lds_kernel<T><<<grid,T::BLOCK_SIZE,T::lds_bytes(args.k),stream>>>(args);
#endif
}

extern "C" __attribute__((visibility("default")))
int launch(int actual, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    if (!opus_private_small_shape_valid(actual,m,n,k) || !opus_private_small_pointer_valid(a,b,sfa,sfb,c))
        return int(hipErrorInvalidValue);
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a=a; args.ptr_b=b; args.ptr_sfa=sfa; args.ptr_sfb=sfb; args.ptr_c=c;
    args.m=m; args.n=n; args.k=k; args.batch=1;
    args.stride_a=k; args.stride_b=k; args.stride_c=n;
    args.stride_sfa=m; args.stride_sfb=k/128;
    args.stride_a_batch=m*k; args.stride_b_batch=n*k; args.stride_c_batch=m*n;
    args.stride_sfa_batch=m*(k/128); args.stride_sfb_batch=(n/128)*(k/128);
    const auto hip_stream=reinterpret_cast<hipStream_t>(stream);
    switch(actual) {
        case 9043: opus_private_small_launch<9043>(args,hip_stream); break;
        case 9044: opus_private_small_launch<9044>(args,hip_stream); break;
        case 9045: opus_private_small_launch<9045>(args,hip_stream); break;
        case 9046: opus_private_small_launch<9046>(args,hip_stream); break;
        case 9047: opus_private_small_launch<9047>(args,hip_stream); break;
        case 9049: opus_private_small_launch<9049>(args,hip_stream); break;
        case 9055: opus_private_small_launch<9055>(args,hip_stream); break;
        case 9056: opus_private_small_launch<9056>(args,hip_stream); break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}
