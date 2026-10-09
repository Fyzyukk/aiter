// Isolated fixed-K fine-M geometry experiment; caller preallocates workspace.
#include <opus/hip_minimal.hpp>
#include <cstdint>
#include <climits>
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh"
using ExperimentTraits = opus_gemm_small_lds_traits_gfx950<80, 128, 1, 4, 4, 1, 2,
    false, false, true, true, true, 2, 4, 128, 0, 16384, true>;
#if !defined(__HIP_DEVICE_COMPILE__)
extern "C" __attribute__((visibility("default")))
int launch_workspace(int variant,const void* a,const void* b,const void* sfa,const void* sfb,
                     void* c,void* workspace,int m,int n,int k,void* stream) {
    using T=ExperimentTraits;
    constexpr int64_t limit=INT32_MAX;
    if(variant!=9062 || !a || !b || !sfa || !sfb || !c || !workspace ||
       m<1 || m>2048 || n<128 || n%128 || k!=16384 ||
       int64_t(m)*k>limit || int64_t(n)*k>limit || int64_t(m)*n*4>limit ||
       reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
       reinterpret_cast<uintptr_t>(c)%16 || reinterpret_cast<uintptr_t>(sfa)%16 ||
       reinterpret_cast<uintptr_t>(workspace)%16) return 1;
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a=a; args.ptr_b=b; args.ptr_c=workspace; args.ptr_sfa=sfa; args.ptr_sfb=sfb;
    args.m=m; args.n=n; args.k=k; args.batch=1;
    args.stride_a=k; args.stride_b=k; args.stride_c=n;
    args.stride_a_batch=m*k; args.stride_b_batch=n*k; args.stride_c_batch=m*n;
    args.stride_sfa=m; args.stride_sfb=k/128;
    args.stride_sfa_batch=m*(k/128); args.stride_sfb_batch=(n/128)*(k/128);
    const auto hs=reinterpret_cast<hipStream_t>(stream);
    gemm_a8w8_mxfp8_scale_small_lds_kernel<T><<<
        dim3(n/T::B_N,(m+T::B_M-1)/T::B_M,T::SPLIT_K),dim3(T::BLOCK_SIZE),T::lds_bytes(k),hs>>>(args);
    int error=hipGetLastError(); if(error)return error;
    opus_gemm_mxscale_bpreshuffle_reduce_kernel<2,4,128><<<
        dim3((m*n+4*128-1)/(4*128)),dim3(128),0,hs>>>(
            reinterpret_cast<const float*>(workspace),reinterpret_cast<opus::bf16_t*>(c),m*n);
    return hipGetLastError();
}
extern "C" __attribute__((visibility("default")))
int split_count(int variant){return variant==9062 ? 2 : 0;}
#endif
// Instantiate host stubs and device entries, then verify fatbin before use.
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<ExperimentTraits>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_reduce_kernel<2,4,128>(const float*,opus::bf16_t*,int);
