#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "contract.h"
#include "candidate/traits.cuh"
#include "candidate/geometry.cuh"
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh"
extern "C" __attribute__((visibility("default")))
int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream) {
 if(!main_pointers_valid(a,b,sfa,sfb,c) || !main_shape_valid(m,n,k,128,0))return int(hipErrorInvalidValue);
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
 args.m=m;args.n=n;args.k=k;args.batch=1;
 args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
 args.stride_a_batch=int64_t(m)*k;args.stride_b_batch=int64_t(n)*k;args.stride_c_batch=int64_t(m)*n;
 args.stride_sfa_batch=int64_t(m)*(k/128);args.stride_sfb_batch=int64_t(n/128)*(k/128);
 // Extent guard occurs before these values are consumed by a kernel.
 const auto s=reinterpret_cast<hipStream_t>(stream);
 switch(kid) {
 case 92000:{
 if(!main_shape_valid(m,n,k,128,0)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<64,2,32>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92001:{
 if(!main_shape_valid(m,n,k,128,0)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<96,2,32>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92002:{
 if(!main_shape_valid(m,n,k,128,0)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<128,2,32>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92003:{
 if(!main_shape_valid(m,n,k,128,384)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<96,2,8,384>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92004:{
 if(!main_shape_valid(m,n,k,128,768)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<96,2,8,768>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92005:{
 if(!main_shape_valid(m,n,k,128,1536)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<128,2,16,1536>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92006:{
 if(!main_shape_valid(m,n,k,128,1536)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<160,2,16,1536>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92007:{
 if(!main_shape_valid(m,n,k,128,3072)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<96,2,32,3072>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92008:{
 if(!main_shape_valid(m,n,k,128,7168)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<96,2,32,7168>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92009:{
 if(!main_shape_valid(m,n,k,128,7168)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<128,2,32,7168>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92010:{
 if(!main_shape_valid(m,n,k,128,3072)){return int(hipErrorInvalidValue);}
 using T=opus_private_geometry_traits<128,2,32,3072>;
 opus_private_main_geometry_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92011:{
 if(!main_shape_valid(m,n,k,64,7168)){return int(hipErrorInvalidValue);}
 using T=opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,8>;
 gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92012:{
 if(!main_shape_valid(m,n,k,64,7168)){return int(hipErrorInvalidValue);}
 using T=opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,16>;
 gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 9022:{
 if(!main_shape_valid(m,n,k,128,0)){return int(hipErrorInvalidValue);}
 using T=opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950;
 gemm_a8w8_mxfp8_scale_4wave_160x128_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 9021:{
 if(!main_shape_valid(m,n,k,128,0)){return int(hipErrorInvalidValue);}
 using T=opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950;
 gemm_a8w8_mxfp8_scale_4wave_128x128_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 9020:{
 if(!main_shape_valid(m,n,k,256,0)){return int(hipErrorInvalidValue);}
 using T=opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<>;
 gemm_a8w8_mxfp8_scale_8wave_192x256_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 91920:{
 if(!main_shape_valid(m,n,k,128,0)){return int(hipErrorInvalidValue);}
 using T=opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<128,128,64>;
 gemm_a8w8_mxfp8_scale_8wave_192x256_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 9024:{
 if(!main_shape_valid(m,n,k,64,7168)){return int(hipErrorInvalidValue);}
 using T=opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,4>;
 gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 default:return int(hipErrorInvalidValue);
 }return int(hipGetLastError());
}
