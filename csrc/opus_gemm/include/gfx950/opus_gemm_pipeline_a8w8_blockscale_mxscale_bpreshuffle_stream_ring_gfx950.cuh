// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_xcd_ring_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_stream_ring_detail {
template<class Traits>
__device__ __forceinline__ void body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    using namespace opus; using T=Traits;
    static_assert(T::SPLIT_K == 1);
    static_assert(T::SCALE_COPY == 1);
    auto* final_output=reinterpret_cast<bf16_t*>(args.ptr_c);
    constexpr int BM=T::B_M,BN=T::B_N,BK=T::B_K,KS=T::KS,MR=T::E_M,NR=T::E_N;
    const int shape_m=args.m,stride_c=args.stride_c;
    const int row=block_id_y()*BM,col=block_id_x()*BN;
    constexpr int split=0;
    const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
    const int wave=__builtin_amdgcn_readfirstlane(tid/64),wm=wave/T::T_N,wn=wave%T::T_N;
    const int valid_rows=shape_m-row<BM?shape_m-row:BM;
    auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*args.stride_a,valid_rows*args.stride_a);
    auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*args.stride_b,BN*args.stride_b);
    auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),(T::TOTAL_K/128-1)*args.stride_sfa+shape_m);
    auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),(args.n/128-1)*args.stride_sfb+T::TOTAL_K/128);
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    auto sm=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
    auto mma=make_tiled_mma<fp8_t,fp8_t,fp32_t>(seq<MR,NR,1>{},seq<T::T_M,T::T_N,1>{},seq<16,16,128>{},mfma_adaptor_swap_ab{});
    using Base=typename decltype(mma)::MMA;
    array<typename Base::vtype_c,MR*NR> c{};
    auto issue_scales=[&](int base,int kt) {
        static_for<T::SA_COPIES>([&](auto ii) {
            constexpr int i=decltype(ii)::value;
            const int index=(i*T::BLOCK_SIZE+tid)*T::SCALE_COPY,r=index%BM,sc=index/BM;
            int source=(sc<T::SC && row+r<shape_m)?(split*T::SC+kt*KS+sc)*args.stride_sfa+row+r:(T::TOTAL_K/128-1)*args.stride_sfa+shape_m;
            asm volatile("" : "+v"(source));
            auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SA+(i*T::BLOCK_SIZE+wave*64)*4));
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,T::SCALE_COPY,source,0,0,0);
        });
        auto* sbdst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SB+wave*64*4));
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,1,(col/128)*args.stride_sfb+split*T::SC+kt*KS+tid%T::SC,0,0,0);
    };
    auto issue=[&](int slot,int kt) {
        static_for<T::A_COPIES>([&](auto ii) {
            constexpr int i=decltype(ii)::value;
            const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/BK,kc=(lin%BK)^((r%(BK/16))*16);
            auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*args.stride_a+kc,split*T::K+kt*BK,0,0);
        });
        static_for<T::B_COPIES>([&](auto ii) {
            constexpr int i=decltype(ii)::value;
            const int lin=(i*T::BLOCK_SIZE+tid)*16,ng=lin/(16*BK),within=lin%(16*BK);
            auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+T::A_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,ng*16*args.stride_b+within,(split*T::K+kt*BK)*16,0,T::B_CACHE);
        });
        __builtin_amdgcn_asyncmark();__builtin_amdgcn_sched_barrier(0);
    };
    // The first wait publishes the scale panel together with data stage zero.
    issue(0,0);
    issue_scales(0,0);
    static_for<T::INITIAL-1>([&](auto ii){constexpr int i=decltype(ii)::value+1;issue(i,i);});
    auto step=[&](int slot,int t,auto pi,auto refill) {
        constexpr int issues=T::MATRIX_ISSUES;
        constexpr int wait=decltype(pi)::value*issues;
        s_waitcnt_vmcnt(number<(wait>63?63:wait)>{});s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
        array<array<typename Base::vtype_a,MR>,KS> af;
        array<array<typename Base::vtype_b,NR>,KS> bf;
        array<array<unsigned,MR>,KS> as;array<unsigned,KS> bs;
        static_for<KS>([&](auto ki){constexpr int ks=decltype(ki)::value;
            bs[ks]=load<1>(ss,T::OFF_SB+(t*KS+ks)*4)[0];
            static_for<MR>([&](auto mi){constexpr int m=decltype(mi)::value;const int r=(m*T::T_M+wm)*16+lane16;
                const int lo=slot*T::STAGE_BYTES+r*BK+(((ks*8+kg)^(r%(BK/16)))*16);
                const int hi=slot*T::STAGE_BYTES+r*BK+(((ks*8+kg+4)^(r%(BK/16)))*16);
                set_slice(af[ks][m],load<16>(sm,lo),0_I,16_I);set_slice(af[ks][m],load<16>(sm,hi),16_I,32_I);
                as[ks][m]=load<1>(ss,T::OFF_SA+((t*KS+ks)*BM+r)*(4/T::SCALE_COPY))[0];
            });
            static_for<NR>([&](auto ni){constexpr int n=decltype(ni)::value;
                const int ng=n*T::T_N+wn,lo=slot*T::STAGE_BYTES+T::A_BYTES+ng*16*BK+ks*2048+lane*16;
                set_slice(bf[ks][n],load<16>(sm,lo),0_I,16_I);set_slice(bf[ks][n],load<16>(sm,lo+1024),16_I,32_I);
            });
        });
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(decltype(refill)::value) issue((slot+T::INITIAL)%T::STAGES,t+T::INITIAL);
        s_waitcnt_lgkmcnt(0_I);
        static_for<KS>([&](auto ki){constexpr int ks=decltype(ki)::value;
            static_for<MR>([&](auto mi){constexpr int m=decltype(mi)::value;
                static_for<NR>([&](auto ni){constexpr int n=decltype(ni)::value;
                    c[m*NR+n]=Base{}(af[ks][m],bf[ks][n],c[m*NR+n],int(as[ks][m]),int(bs[ks]),0_I,0_I);
                });
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    };
    // Only one stage is compiled in the steady loop; the drain has fixed waits.
    constexpr int steady=T::PANELS-T::INITIAL;
    #pragma clang loop unroll(disable)
    for (int t=0;t<steady;++t) {
        step(t%T::STAGES,t,number<T::INITIAL-1>{},1_I);
    }
    static_for<T::INITIAL>([&](auto ji) {
        constexpr int j=decltype(ji)::value,t=steady+j;
        step(number<t%T::STAGES>{},t,number<T::INITIAL-1-j>{},0_I);
    });
    const auto pc = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
    const auto uc = partition_layout_c<4>(mma, opus::make_tuple(stride_c, 1_I), pc);
    const auto offsets = layout_to_offsets<4>(uc);
    auto output = make_gmem(final_output + static_cast<int64_t>(row) * stride_c + col,
        ((valid_rows - 1) * stride_c + BN) * sizeof(bf16_t));
    static_for<MR * NR>([&](auto fi) {
        constexpr int f = decltype(fi)::value;
        store<4>(output, cast<bf16_t>(c[f]), offsets[f], 0, number<2>{});
    });
}
} // namespace opus_gemm_stream_ring_detail
#endif

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_stream_ring_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_stream_ring_detail::body<Traits>(args);
#endif
}
