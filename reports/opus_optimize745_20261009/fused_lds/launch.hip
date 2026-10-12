// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <opus/hip_minimal.hpp>
#include "fused_kargs.cuh"
#include "pipeline.cuh"
using FusedTraits0 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<48, 64, 1, 4, 3, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
__global__ __launch_bounds__(FusedTraits0::BLOCK_SIZE,1)
void fused_lds_kernel0(opus745_fused_lds_kargs args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_lds_detail::run<FusedTraits0,false>(args);
#endif
}
using FusedTraits1 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<48, 64, 1, 4, 4, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
__global__ __launch_bounds__(FusedTraits1::BLOCK_SIZE,1)
void fused_lds_kernel1(opus745_fused_lds_kargs args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_lds_detail::run<FusedTraits1,false>(args);
#endif
}
using FusedTraits2 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 3, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
__global__ __launch_bounds__(FusedTraits2::BLOCK_SIZE,1)
void fused_lds_kernel2(opus745_fused_lds_kargs args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_lds_detail::run<FusedTraits2,false>(args);
#endif
}
using FusedTraits3 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 4, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
__global__ __launch_bounds__(FusedTraits3::BLOCK_SIZE,1)
void fused_lds_kernel3(opus745_fused_lds_kargs args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_lds_detail::run<FusedTraits3,false>(args);
#endif
}
using FusedTraits4 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<64, 128, 2, 2, 3, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
__global__ __launch_bounds__(FusedTraits4::BLOCK_SIZE,1)
void fused_lds_kernel4(opus745_fused_lds_kargs args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_lds_detail::run<FusedTraits4,false>(args);
#endif
}
using FusedTraits5 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<64, 128, 2, 2, 4, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
__global__ __launch_bounds__(FusedTraits5::BLOCK_SIZE,1)
void fused_lds_kernel5(opus745_fused_lds_kargs args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus745_fused_lds_detail::run<FusedTraits5,false>(args);
#endif
}
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> int launch_one(opus745_fused_lds_kargs args, void* stream, const void* kernel) {
    const int tiles = ((args.m + T::B_M - 1) / T::B_M) * (args.n / T::B_N);
    const dim3 grid(((tiles + 7) / 8) * 8 * args.split_k);
    void* argv[] = {&args};
    return hipLaunchKernel(kernel,grid,dim3(T::BLOCK_SIZE),argv,T::lds_bytes(args.k,args.split_k),stream);
}
extern "C" __attribute__((visibility("default")))
int launch(int variant,const void* a,const void* b,const void* sfa,const void* sfb,
           void* c,float* workspace,int* counters,int m,int n,int k,int split_k,void* stream) {
    if (m<=0 || m>2048 || n<=0 || k<128 || k>16384 || n%128 || k%128 ||
        split_k<1 || split_k>16 || split_k>k/128 || int64_t(m)*k>INT32_MAX ||
        int64_t(n)*k>INT32_MAX || int64_t(m)*n*2>INT32_MAX ||
        !a || !b || !sfa || !sfb || !c || (split_k>1 && (!workspace || !counters)) ||
        reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
        reinterpret_cast<uintptr_t>(sfa)%16 || reinterpret_cast<uintptr_t>(sfb)%4 ||
        reinterpret_cast<uintptr_t>(c)%16 || reinterpret_cast<uintptr_t>(workspace)%16 ||
        reinterpret_cast<uintptr_t>(counters)%4) return 1;
    opus745_fused_lds_kargs args{};
    args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
    args.ptr_workspace=workspace;args.ptr_counters=counters;args.split_k=split_k;
    args.m=m;args.n=n;args.k=k;args.batch=1;
    args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
    switch (variant) {

    case 0:return launch_one<FusedTraits0>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel0));
    case 1:return launch_one<FusedTraits1>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel1));
    case 2:return launch_one<FusedTraits2>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel2));
    case 3:return launch_one<FusedTraits3>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel3));
    case 4:return launch_one<FusedTraits4>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel4));
    case 5:return launch_one<FusedTraits5>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel5));
    default:return 1;
    }
}
#endif
