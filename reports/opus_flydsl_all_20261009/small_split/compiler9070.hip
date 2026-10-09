#include <opus/hip_minimal.hpp>
#include "contract.h"
#ifndef __HIPCC_RTC__
#define __HIPCC_RTC__ 1
#endif
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh"
using Actual9070=opus_gemm_small_register_traits_gfx950<16,32,1,1,3,8,4,3,7168,false,true>;

// Same exact actual9070 producer in both compiler libraries, including fixed K.
// Raw ABI matches launch.hip. This single-producer control takes no workspace.
extern "C" __attribute__((visibility("default")))
int launch(int id,const void* a,const void* b,const void* sfa,const void* sfb,
           void* c,void* workspace,uint64_t capacity,int m,int n,int k,void* stream) {
    PrivateConfig cfg{};
    if(id!=9052 || k!=7168 || !private_config(id,false,m,n,k,cfg) ||
       !private_pointers_valid(cfg,a,b,sfa,sfb,c,workspace,capacity,m,n,k))
        return 1; // HIP invalid-value status, using only the minimal declaration header.
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
    args.m=m;args.n=n;args.k=k;args.batch=1;
    args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
    args.stride_a_batch=m*k;args.stride_b_batch=n*k;args.stride_c_batch=m*n;
    args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
    gemm_a8w8_mxfp8_scale_small_register_kernel<Actual9070>
        <<<dim3((n+31)/32,(m+15)/16),512,0,reinterpret_cast<hipStream_t>(stream)>>>(args);
    return int(hipGetLastError());
}
