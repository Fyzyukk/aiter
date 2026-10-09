// Private experiment: caller owns all buffers; no allocation is timed.
#include <opus/hip_minimal.hpp>
#include <cstdint>
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_fine_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh"
using R80S1 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 1, 4, 128, 0, 0>;
using F80S1K7168 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 1, 4, 128, 0, 7168>;
using F80S1K3072 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 1, 4, 128, 0, 3072>;
using R96W8S1 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96, 2, 4, 4, 1, 1, 4, 128, 0, 0>;
using R80S2 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 2, 4, 128, 0, 0>;
using F80S2K16384 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 2, 4, 128, 0, 16384>;
using R96W8S2 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96, 2, 4, 4, 1, 2, 16, 128, 2, 0>;
using R80S4 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 4, 16, 128, 2, 0>;
using F80S4K16384 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80, 1, 4, 4, 1, 4, 4, 128, 0, 16384>;
using R96W8S4 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96, 2, 4, 4, 1, 4, 16, 128, 2, 0>;
using R96W4S4 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96, 2, 2, 4, 1, 4, 16, 64, 0, 0>;
using R128W4S4 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<128, 2, 2, 4, 1, 4, 4, 128, 0, 0>;
using R112W4S4 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<112, 1, 4, 4, 1, 4, 4, 128, 0, 0>;
using R48W4S4 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<48, 1, 4, 4, 1, 4, 4, 128, 0, 0>;
using F48W4S4K7168 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<48, 1, 4, 6, 2, 4, 4, 128, 0, 7168>;
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
static int invoke(const void* a, const void* b, const void* sfa, const void* sfb,
                  void* c, void* workspace, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if (!a || !b || !sfa || !sfb || !c || m < 1 || m > 2048 || n < 128 || n % 128 ||
        k < 128 || k > 16384 || k % 128 || (T::FIXED_K && k != T::FIXED_K) ||
        int64_t(m)*k > limit || int64_t(n)*k > limit ||
        int64_t(m)*n*(T::SPLIT_K > 1 ? 4 : 2) > limit ||
        reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
        reinterpret_cast<uintptr_t>(c)%16 || reinterpret_cast<uintptr_t>(sfa)%16 ||
        (T::SPLIT_K > 1 && (!workspace || reinterpret_cast<uintptr_t>(workspace)%16))) return 1;
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a=a; args.ptr_b=b; args.ptr_c=T::SPLIT_K > 1 ? workspace : c;
    args.ptr_sfa=sfa; args.ptr_sfb=sfb; args.m=m; args.n=n; args.k=k; args.batch=1;
    args.stride_a=k; args.stride_b=k; args.stride_c=n;
    args.stride_a_batch=m*k; args.stride_b_batch=n*k; args.stride_c_batch=m*n;
    args.stride_sfa=m; args.stride_sfb=k/128;
    args.stride_sfa_batch=m*(k/128); args.stride_sfb_batch=(n/128)*(k/128);
    gemm_a8w8_mxfp8_scale_small_lds_kernel<T><<<
        dim3(n/128,(m+T::B_M-1)/T::B_M,T::SPLIT_K),dim3(T::BLOCK_SIZE),T::lds_bytes(k),
        reinterpret_cast<hipStream_t>(stream)>>>(args);
    int error=hipGetLastError();
    if (error) return error;
    if constexpr (T::SPLIT_K > 1) {
        opus_gemm_mxscale_bpreshuffle_reduce_kernel<T::SPLIT_K,T::REDUCE_VEC,T::REDUCE_BLOCK><<<
            dim3((m*n+T::REDUCE_VEC*T::REDUCE_BLOCK-1)/(T::REDUCE_VEC*T::REDUCE_BLOCK)),
            dim3(T::REDUCE_BLOCK),0,reinterpret_cast<hipStream_t>(stream)>>>(
                reinterpret_cast<const float*>(workspace), reinterpret_cast<opus::bf16_t*>(c),m*n);
    }
    return hipGetLastError();
}
#define INVOKE(T) return invoke<T>(a,b,sfa,sfb,c,workspace,m,n,k,stream)
static int dispatch(int variant,const void* a,const void* b,const void* sfa,const void* sfb,
                    void* c,void* workspace,int m,int n,int k,void* stream) {
    switch (variant) {
        case 9060: if(k==7168){INVOKE(F80S1K7168);} if(k==3072){INVOKE(F80S1K3072);} INVOKE(R80S1);
        case 9061: INVOKE(R96W8S1);
        case 9062:
            if((m+95)/96 < (m+79)/80){INVOKE(R96W8S2);}
            if(k==16384){INVOKE(F80S2K16384);} INVOKE(R80S2);
        case 9063:
            if(k>=8192 && m<=48){INVOKE(R48W4S4);}
            if(k>=8192 && m>80 && m<=96){INVOKE(R96W4S4);}
            if(k>=8192 && m>96 && m<=112){INVOKE(R112W4S4);}
            if(k>=8192 && m>112 && m<=128){INVOKE(R128W4S4);}
            if(!(k>=8192 && m<=128) && ((m+79)/80)*(n/128)*4>256){INVOKE(R96W8S4);}
            if(k==16384){INVOKE(F80S4K16384);} INVOKE(R80S4);
        case 9064: INVOKE(R96W8S2);
        case 9065: INVOKE(R96W8S4);
        case 9066: INVOKE(R96W4S4);
        case 9067: INVOKE(R128W4S4);
        case 9068: INVOKE(R112W4S4);
        case 9069: if(k==7168){INVOKE(F48W4S4K7168);} INVOKE(R48W4S4);
        case 29060: INVOKE(R80S1);
        case 29061: INVOKE(R96W8S1);
        case 29062: INVOKE(R80S2);
        case 29063: INVOKE(R80S4);
        case 29069: INVOKE(R48W4S4);
        default:return 1;
    }
}
extern "C" __attribute__((visibility("default")))
int launch_workspace(int variant,const void* a,const void* b,const void* sfa,const void* sfb,
                     void* c,void* workspace,int m,int n,int k,void* stream) {
    return dispatch(variant,a,b,sfa,sfb,c,workspace,m,n,k,stream);
}
extern "C" __attribute__((visibility("default")))
int launch(int variant,const void* a,const void* b,const void* sfa,const void* sfb,
           void* c,int m,int n,int k,void* stream) {
    return dispatch(variant,a,b,sfa,sfb,c,nullptr,m,n,k,stream);
}
extern "C" __attribute__((visibility("default")))
int split_count(int variant) {
    switch(variant) {
        case 9060: case 9061: case 29060: case 29061:return 1;
        case 9062: case 9064: case 29062:return 2;
        case 9063: case 9065: case 9066: case 9067: case 9068: case 9069:
        case 29063: case 29069:return 4;
        default:return 0;
    }
}
#endif

template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R80S1>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<F80S1K7168>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<F80S1K3072>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R96W8S1>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R80S2>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<F80S2K16384>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R96W8S2>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R80S4>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<F80S4K16384>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R96W8S4>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R96W4S4>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R128W4S4>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R112W4S4>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<R48W4S4>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<F48W4S4K7168>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_reduce_kernel<2,4,128>(const float*,opus::bf16_t*,int);
template __global__ void opus_gemm_mxscale_bpreshuffle_reduce_kernel<2,16,128>(const float*,opus::bf16_t*,int);
template __global__ void opus_gemm_mxscale_bpreshuffle_reduce_kernel<4,4,128>(const float*,opus::bf16_t*,int);
template __global__ void opus_gemm_mxscale_bpreshuffle_reduce_kernel<4,16,64>(const float*,opus::bf16_t*,int);
template __global__ void opus_gemm_mxscale_bpreshuffle_reduce_kernel<4,16,128>(const float*,opus::bf16_t*,int);
