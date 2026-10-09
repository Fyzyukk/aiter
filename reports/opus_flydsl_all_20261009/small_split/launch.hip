#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "contract.h"
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_fine_gfx950.cuh"
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh"
#if defined(SMALL_SPLIT_CANDIDATE)
#include "candidate/traits.cuh"
#include "candidate/register_global_split.cuh"
#endif

template<class T> void register_launch(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args, hipStream_t stream) {
    const dim3 grid((args.n + T::B_N - 1) / T::B_N, (args.m + T::B_M - 1) / T::B_M);
    gemm_a8w8_mxfp8_scale_small_register_kernel<T><<<grid, T::BLOCK_SIZE, 0, stream>>>(args);
}

template<class T> void lds_launch(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args, void* c, hipStream_t stream) {
    const dim3 grid(args.n / T::B_N, (args.m + T::B_M - 1) / T::B_M, T::SPLIT_K);
    gemm_a8w8_mxfp8_scale_small_lds_kernel<T><<<grid, T::BLOCK_SIZE, T::lds_bytes(args.k), stream>>>(args);
    if constexpr (T::SPLIT_K > 1) {
        opus_gemm_mxscale_bpreshuffle_reduce_kernel<T::SPLIT_K,T::REDUCE_VEC,T::REDUCE_BLOCK>
            <<<dim3((args.m * args.n + T::REDUCE_VEC * T::REDUCE_BLOCK - 1) / (T::REDUCE_VEC * T::REDUCE_BLOCK)),
               T::REDUCE_BLOCK, 0, stream>>>(reinterpret_cast<const float*>(args.ptr_c), reinterpret_cast<opus::bf16_t*>(c), args.m * args.n);
    }
}

#if defined(SMALL_SPLIT_CANDIDATE)
template<class T> void register_split_launch(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args, void* c, hipStream_t stream) {
    const dim3 grid(args.n / T::B_N, (args.m + T::B_M - 1) / T::B_M, T::GLOBAL_SPLIT_K);
    opus_private_register_global_split_kernel<T><<<grid,T::BLOCK_SIZE,0,stream>>>(args);
    opus_gemm_mxscale_bpreshuffle_reduce_kernel<T::GLOBAL_SPLIT_K,16,128>
        <<<dim3((args.m * args.n + 2047) / 2048),128,0,stream>>>(reinterpret_cast<const float*>(args.ptr_c), reinterpret_cast<opus::bf16_t*>(c), args.m * args.n);
}
#endif

// Complete-call raw ABI: workspace capacity in bytes, allocated by caller.
// No allocation, memset, or host synchronization occurs inside launch.
extern "C" __attribute__((visibility("default")))
int launch(int id, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, void* workspace, uint64_t workspace_bytes, int m, int n, int k, void* stream) {
#if defined(SMALL_SPLIT_CANDIDATE)
    constexpr bool candidate = true;
#else
    constexpr bool candidate = false;
#endif
    PrivateConfig cfg{};
    if (!private_config(id,candidate,m,n,k,cfg) ||
        !private_pointers_valid(cfg,a,b,sfa,sfb,c,workspace,workspace_bytes,m,n,k)) return int(hipErrorInvalidValue);
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=cfg.split>1?workspace:c;
    args.m=m;args.n=n;args.k=k;args.batch=1;
    args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
    args.stride_a_batch=m*k;args.stride_b_batch=n*k;args.stride_c_batch=m*n;
    args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
    const auto st=reinterpret_cast<hipStream_t>(stream);
#if defined(SMALL_SPLIT_CANDIDATE)
    switch(id) {
    case 110: register_split_launch<opus_private_register_split4_traits<16,16,1>>(args,c,st);break;
    case 111: register_split_launch<opus_private_register_split4_traits<16,16,2>>(args,c,st);break;
    case 120: register_split_launch<opus_private_register_split4_traits<16,32,1>>(args,c,st);break;
    case 121: register_split_launch<opus_private_register_split4_traits<16,32,2>>(args,c,st);break;
    case 130: register_split_launch<opus_private_register_split4_traits<32,32,1>>(args,c,st);break;
    case 140: register_split_launch<opus_private_register_split4_traits<32,64,1>>(args,c,st);break;
    case 210: lds_launch<opus_private_narrow_fine_traits<48,64,1,4,1>>(args,c,st);break;
    case 211: lds_launch<opus_private_narrow_fine_traits<48,64,1,4,2>>(args,c,st);break;
    case 220: lds_launch<opus_private_narrow_fine_traits<64,128,2,2,1>>(args,c,st);break;
    case 221: lds_launch<opus_private_narrow_fine_traits<64,128,2,2,2>>(args,c,st);break;
    case 230: lds_launch<opus_private_narrow_fine_traits<96,128,2,2,1>>(args,c,st);break;
    case 231: lds_launch<opus_private_narrow_fine_traits<96,128,2,2,2>>(args,c,st);break;
    default:return int(hipErrorInvalidValue);
    }
#else
    switch(id) {
    case 9040: register_launch<opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950>(args,st);break;
    case 9041:
        if (k>=8192 || (k>=4096 && ((m+15)/16)*(n/16)<256))
            register_launch<opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x16_traits_gfx950>(args,st);
        else register_launch<opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950>(args,st);break;
    case 9042:
        if(k==7168 && ((m+15)/16)*((n+47)/48)<=256) register_launch<opus_gemm_small_register_traits_gfx950<16,48,1,1,4,4,4,3,7168,true,false>>(args,st);
        else if(k==7168) register_launch<opus_gemm_small_register_traits_gfx950<32,48,1,1,3,4,4,0,7168,true,false>>(args,st);
        else register_launch<opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950>(args,st);break;
    case 9051: register_launch<opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x32_traits_gfx950>(args,st);break;
    case 9052:
        if(k==7168) register_launch<opus_gemm_small_register_traits_gfx950<16,32,1,1,3,8,4,3,7168,false,true>>(args,st);
        else register_launch<opus_gemm_mxscale_bpreshuffle_small_register_wavek_16x32_traits_gfx950>(args,st);break;
    case 9053:
        if(k==7168) register_launch<opus_gemm_small_register_traits_gfx950<32,32,1,1,4,4,4,3,7168,false,true>>(args,st);
        else register_launch<opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x32_traits_gfx950>(args,st);break;
    case 9054: register_launch<opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x64_traits_gfx950>(args,st);break;
    case 9060:
        if(k==7168) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,1,4,128,0,7168>>(args,c,st);
        else if(k==3072) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,1,4,128,0,3072>>(args,c,st);
        else lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,1>>(args,c,st);break;
    case 9061: lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96,2,4,4,1,1>>(args,c,st);break;
    case 9062:
        if(cfg.bm==96) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96,2,4,4,1,2,16,128,2>>(args,c,st);
        else if(k==16384) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,2,4,128,0,16384>>(args,c,st);
        else lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,2>>(args,c,st);break;
    case 9063:
        if(cfg.bm==48) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<48,1,4,4,1,4>>(args,c,st);
        else if(cfg.bm==112) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<112,1,4,4,1,4>>(args,c,st);
        else if(cfg.bm==128) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<128,2,2,4,1,4>>(args,c,st);
        else if(cfg.bm==96 && cfg.waves==4) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96,2,2,4,1,4,16,64>>(args,c,st);
        else if(cfg.bm==96) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<96,2,4,4,1,4,16,128,2>>(args,c,st);
        else if(k==16384) lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,4,4,128,0,16384>>(args,c,st);
        else lds_launch<opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950<80,1,4,4,1,4,16,128,2>>(args,c,st);break;
    default:return int(hipErrorInvalidValue);
    }
#endif
    return int(hipGetLastError());
}
