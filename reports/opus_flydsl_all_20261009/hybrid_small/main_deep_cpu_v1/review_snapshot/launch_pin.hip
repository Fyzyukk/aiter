#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "contract.h"
#include "candidate/traits.cuh"
#include "candidate/pin.cuh"
#include "candidate/pad_pin.cuh"
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh"
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
 case 92100:{
 if(!main_shape_valid(m,n,k,256,384)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<384,16,false>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92101:{
 if(!main_shape_valid(m,n,k,256,1536)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<1536,16,false>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92102:{
 if(!main_shape_valid(m,n,k,256,3072)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<3072,32,false>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92103:{
 if(!main_shape_valid(m,n,k,256,7168)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<7168,64,false>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92104:{
 if(!main_shape_valid(m,n,k,256,16384)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<16384,64,false>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92110:{
 if(!main_shape_valid(m,n,k,256,384)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<384,16,true>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92111:{
 if(!main_shape_valid(m,n,k,256,1536)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<1536,16,true>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92112:{
 if(!main_shape_valid(m,n,k,256,3072)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<3072,32,true>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92113:{
 if(!main_shape_valid(m,n,k,256,7168)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<7168,64,true>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92114:{
 if(!main_shape_valid(m,n,k,256,16384)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_private_pin_fixed_traits<16384,64,true>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 92120:{
 if(!main_shape_valid(m,n,k,256,7168)){return int(hipErrorInvalidValue);}
 using T=opus_private_pad_pin_fixed_traits<7168>;
 opus_private_pin::gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel_private_fixed<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 9000:{
 if(!main_shape_valid(m,n,k,256,0)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950;
 gemm_a8w8_mxfp8_scale_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 9001:{
 if(!main_shape_valid(m,n,k,256,0)){return int(hipErrorInvalidValue);}
 if(m%256)return int(hipErrorInvalidValue);
 using T=opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950;
 gemm_a8w8_mxfp8_scale_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 case 9011:{
 if(!main_shape_valid(m,n,k,256,0)){return int(hipErrorInvalidValue);}
 using T=opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950;
 gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}
 default:return int(hipErrorInvalidValue);
 }return int(hipGetLastError());
}
