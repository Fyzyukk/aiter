#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"
extern "C" __attribute__((visibility("default")))
int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream) {
 constexpr int64_t input_limit=INT32_MAX;
 constexpr int64_t output_limit=INT64_MAX;
 if(kid!=9030||m<=0||n<=0||k<=0||k>16384||k%128||m%64||n%256||
    int64_t(m)>input_limit/k||int64_t(n)>input_limit/k||
    int64_t(m)<=(input_limit/2)/n||int64_t(m)>(output_limit/2)/n||
    int64_t(191)*n+256>input_limit/2||
    !a||!b||!sfa||!sfb||!c||reinterpret_cast<uintptr_t>(a)%16||
    reinterpret_cast<uintptr_t>(b)%16||reinterpret_cast<uintptr_t>(sfa)%16||
    reinterpret_cast<uintptr_t>(c)%16) return int(hipErrorInvalidValue);
 const uint64_t output_begin=reinterpret_cast<uintptr_t>(c);
 const uint64_t output_bytes=uint64_t(m)*n*2;
 const auto overlaps=[&](const void* p,uint64_t bytes) {
   const uint64_t begin=reinterpret_cast<uintptr_t>(p);
   return output_begin>=begin?output_begin-begin<bytes:begin-output_begin<output_bytes;
 };
 if(overlaps(a,uint64_t(m)*k)||overlaps(b,uint64_t(n)*k)||
    overlaps(sfa,uint64_t(m)*(k/128))||overlaps(sfb,uint64_t(n/128)*(k/128)))
   return int(hipErrorInvalidValue);
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
 args.m=m;args.n=n;args.k=k;args.batch=1;args.stride_a=k;args.stride_b=k;args.stride_c=n;
 args.stride_sfa=m;args.stride_sfb=k/128;args.stride_a_batch=m*k;args.stride_b_batch=n*k;
 args.stride_c_batch=0;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 using T=opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950;
 gemm_a8w8_mxfp8_scale_8wave_192x256_large_output_kernel<T>
   <<<dim3(n/256,(m+191)/192),512,0,reinterpret_cast<hipStream_t>(stream)>>>(args);
 return int(hipGetLastError());
}
