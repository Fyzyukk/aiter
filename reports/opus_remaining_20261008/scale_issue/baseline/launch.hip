#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh"
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"
template<class T> void launch_9020(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args, hipStream_t stream) {
 gemm_a8w8_mxfp8_scale_8wave_192x256_kernel<T><<<dim3(args.n/T::B_N,(args.m+T::B_M-1)/T::B_M),T::BLOCK_SIZE,0,stream>>>(args);
}
extern "C" __attribute__((visibility("default")))
int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream) {
 constexpr int64_t limit=INT32_MAX;
 if ((kid!=9020&&kid!=9022)||m<=0||n<=0||k<=0||k>16384||k%128||m%16||
     n%(kid==9020?256:128)||int64_t(m)*k>limit||int64_t(n)*k>limit||int64_t(m)*n*2>limit||
     !a||!b||!sfa||!sfb||!c||reinterpret_cast<uintptr_t>(a)%16||reinterpret_cast<uintptr_t>(b)%16||
     reinterpret_cast<uintptr_t>(sfa)%16||reinterpret_cast<uintptr_t>(c)%16) return int(hipErrorInvalidValue);
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
 args.m=m;args.n=n;args.k=k;args.batch=1;args.stride_a=k;args.stride_b=k;args.stride_c=n;
 args.stride_sfa=m;args.stride_sfb=k/128;args.stride_a_batch=m*k;args.stride_b_batch=n*k;
 args.stride_c_batch=m*n;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 auto hip_stream=reinterpret_cast<hipStream_t>(stream);
 if(kid==9022) {using T=opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950;
 gemm_a8w8_mxfp8_scale_4wave_160x128_kernel<T><<<dim3(n/128,(m+159)/160),256,0,hip_stream>>>(args);}
 else if(m>=8192&&n<=1024&&k<=8192&&(((m+127)/128)*(n/128)<=512||((m+191)/192)*(n/256)>256))
 launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<128,128,64>>(args,hip_stream);
 else if(m>=1024&&k==384) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,8,384>>(args,hip_stream);
 else if(m>=1024&&k==768) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,8,768>>(args,hip_stream);
 else if(m>=1024&&k==1536) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,32,1536>>(args,hip_stream);
 else if(m>=1024&&k==3072) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,32,3072>>(args,hip_stream);
 else if(m>=1024&&k==7168) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,64,7168>>(args,hip_stream);
 else launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_192x256_traits_gfx950>(args,hip_stream);
 return int(hipGetLastError());
}
