#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh"
extern "C" __attribute__((visibility("default")))
int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream) {
 if((kid!=9000&&kid!=9010)||m<=0||n<=0||k<=0||k>16384||k%128||n%256||m%(kid==9000?256:64)||int64_t(m)*k>INT32_MAX||int64_t(n)*k>INT32_MAX||int64_t(m)*n*2>INT32_MAX) return int(hipErrorInvalidValue);
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
 args.m=m;args.n=n;args.k=k;args.batch=1;args.stride_a=k;args.stride_b=k;args.stride_c=n;
 args.stride_sfa=m;args.stride_sfb=k/128;args.stride_a_batch=m*k;args.stride_b_batch=n*k;args.stride_c_batch=m*n;
 args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 auto hs=reinterpret_cast<hipStream_t>(stream);
 if(kid==9000) gemm_a8w8_mxfp8_scale_kernel<opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950><<<dim3(n/256,m/256),256,0,hs>>>(args);
 else gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel<opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950><<<dim3(n/256,(m+255)/256),256,0,hs>>>(args);
 return int(hipGetLastError());
}
