#define __HIPCC_RTC__ 1
#include <opus/hip_minimal.hpp>
#include <cstdint>
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_fine_gfx950.cuh"
template<class T> __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template<int Split, int Vec, int Block> __global__ void opus_gemm_mxscale_bpreshuffle_reduce_kernel(const float*, opus::bf16_t*, int);
using Runtime = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,2,4,128,0,0>;
using Fixed = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,2,4,128,0,16384>;
using M96 = opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96,2,4,4,1,2,16,128,2,0>;
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> int invoke(const void* a,const void* b,const void* sfa,const void* sfb,void* c,void* workspace,int m,int n,int k,void* stream) {
 constexpr int64_t limit=INT32_MAX;
 if(!a||!b||!sfa||!sfb||!c||!workspace||m<1||m>2048||n<128||n%128||k<128||k>16384||k%128||
    (T::FIXED_K&&k!=T::FIXED_K)||int64_t(m)*k>limit||int64_t(n)*k>limit||int64_t(m)*n*4>limit||
    reinterpret_cast<uintptr_t>(a)%16||reinterpret_cast<uintptr_t>(b)%16||reinterpret_cast<uintptr_t>(sfa)%16||
    reinterpret_cast<uintptr_t>(c)%16||reinterpret_cast<uintptr_t>(workspace)%16) return 1;
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=workspace;
 args.m=m;args.n=n;args.k=k;args.batch=1;args.stride_a=k;args.stride_b=k;args.stride_c=n;
 args.stride_sfa=m;args.stride_sfb=k/128;args.stride_a_batch=m*k;args.stride_b_batch=n*k;
 args.stride_c_batch=m*n;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 auto hip_stream=reinterpret_cast<hipStream_t>(stream);
 gemm_a8w8_mxfp8_scale_small_lds_kernel<T><<<dim3(n/128,(m+T::B_M-1)/T::B_M,2),dim3(T::BLOCK_SIZE),T::lds_bytes(k),hip_stream>>>(args);
 auto error=hipGetLastError();if(error)return error;
 opus_gemm_mxscale_bpreshuffle_reduce_kernel<2,T::REDUCE_VEC,T::REDUCE_BLOCK><<<dim3((m*n+T::REDUCE_VEC*T::REDUCE_BLOCK-1)/(T::REDUCE_VEC*T::REDUCE_BLOCK)),dim3(T::REDUCE_BLOCK),0,hip_stream>>>(reinterpret_cast<const float*>(workspace),reinterpret_cast<opus::bf16_t*>(c),m*n);
 return hipGetLastError();
}
extern "C" __attribute__((visibility("default"))) int launch_workspace(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,void* workspace,int m,int n,int k,void* stream) {
 if(kid!=9062)return 1;
 if((m+95)/96<(m+79)/80)return invoke<M96>(a,b,sfa,sfb,c,workspace,m,n,k,stream);
 if(k==16384)return invoke<Fixed>(a,b,sfa,sfb,c,workspace,m,n,k,stream);
 return invoke<Runtime>(a,b,sfa,sfb,c,workspace,m,n,k,stream);
}
extern "C" __attribute__((visibility("default"))) int split_count(int kid){return kid==9062?2:0;}
#endif
