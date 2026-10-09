#define __HIPCC_RTC__ 1
#include <opus/hip_minimal.hpp>
#include <cstdint>
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"
template<class T> __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
using R9051=opus_gemm_small_register_traits_gfx950<16,32,1,1,3,4,3,3,0,false,false>;
#if !defined(__HIP_DEVICE_COMPILE__)
extern "C" __attribute__((visibility("default"))) int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream){
 constexpr int64_t limit=INT32_MAX;
 if(kid!=9051||!a||!b||!sfa||!sfb||!c||m<1||m>512||n<128||n%128||k<128||k>16384||k%128||int64_t(m)*k>limit||int64_t(n)*k>limit||int64_t(m)*n*2>limit||reinterpret_cast<uintptr_t>(a)%16||reinterpret_cast<uintptr_t>(b)%16||reinterpret_cast<uintptr_t>(sfa)%16||reinterpret_cast<uintptr_t>(c)%16)return 1;
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;args.m=m;args.n=n;args.k=k;args.batch=1;
 args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
 args.stride_a_batch=m*k;args.stride_b_batch=n*k;args.stride_c_batch=m*n;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 gemm_a8w8_mxfp8_scale_small_register_kernel<R9051><<<dim3(n/32,(m+15)/16),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args);
 return hipGetLastError();
}
#endif
