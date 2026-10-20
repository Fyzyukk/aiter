// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_composable_gfx950.cuh"

struct opus_gemm_bpreshuffle_fused_split_kargs {
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args;
    opus::bf16_t* output;
    unsigned* counters;
};

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_composable_detail {
template<class Traits>
__device__ __forceinline__ void body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args, opus::bf16_t* final_output, unsigned* counters) {
    using namespace opus; using T=Traits;
    if constexpr(T::FIXED_N>0) args.n=T::FIXED_N;
    args.k=T::TOTAL_K; args.stride_a=T::TOTAL_K; args.stride_b=T::TOTAL_K;
    args.stride_c=args.n; args.stride_sfa=args.m; args.stride_sfb=T::TOTAL_K/128;
    constexpr int BM=T::B_M,BN=T::B_N,BK=T::B_K,KS=T::KS,MR=T::E_M,NR=T::E_N;
    const int shape_m=args.m,stride_c=args.stride_c;
    const unsigned gm=(args.m+BM-1)/BM,gn=args.n/BN,tiles=gm*gn;
    unsigned tile=0,split=0;
    int row,col;
    constexpr bool m_first=[] {
        if constexpr(requires { T::M_FIRST; }) return T::M_FIRST;
        else return false;
    }();
    if constexpr (m_first) {
        row=block_id_x()*BM;
        col=block_id_y()*BN;
    } else if constexpr (T::SPLIT_K == 1 && !T::XCD_ORDER) {
        row=block_id_y()*BM;
        col=block_id_x()*BN;
    } else {
        const unsigned pid=block_id_x();
        tile=pid/(8*T::SPLIT_K)*8+pid%8;
        split=(pid/8)%T::SPLIT_K;
        if(tile>=tiles) return;
        row=(tile%gm)*BM;
        col=(tile/gm)*BN;
    }
    const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
    __builtin_assume(static_cast<unsigned>(tid)<T::BLOCK_SIZE);
    const int wave=__builtin_amdgcn_readfirstlane(tid/64);
    __builtin_assume(static_cast<unsigned>(wave)<T::T_M*T::T_N);
    const int wm=wave/T::T_N, wn=wave%T::T_N;
    auto mfrag=[&](auto mi) { return T::CONTIGUOUS?wm*MR+int(mi):int(mi)*T::T_M+wm; };
    auto nfrag=[&](auto ni) { return T::CONTIGUOUS?wn*NR+int(ni):int(ni)*T::T_N+wn; };
    const int valid_rows=shape_m-row<BM?shape_m-row:BM;
    auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*args.stride_a,valid_rows*args.stride_a);
    auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*args.stride_b,BN*args.stride_b);
    auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),(T::TOTAL_K/128-1)*args.stride_sfa+shape_m);
    auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),(args.n/128-1)*args.stride_sfb+T::TOTAL_K/128);
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    auto sm=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
    auto mma=make_tiled_mma<fp8_t,fp8_t,fp32_t>(seq<1,1,1>{},seq<1,1,1>{},seq<16,16,128>{},std::conditional_t<T::NORMAL,mfma_adaptor,mfma_adaptor_swap_ab>{});
    using Base=typename decltype(mma)::MMA;
    using AFragments=array<array<typename Base::vtype_a,MR>,KS>;
    using BFragments=array<array<typename Base::vtype_b,NR>,KS>;
    struct Stage { AFragments a; BFragments b; array<array<unsigned,MR>,KS> sa; array<array<unsigned,NR>,KS> sb; };
    array<typename Base::vtype_c,MR*NR> c{};
    array<BFragments,T::B_SETS> b_regs;
    auto prefetch_b=[&](int t,auto phase) {
        if constexpr(T::B_DIRECT) {
            static_for<KS>([&](auto ki){constexpr int ks=decltype(ki)::value;
                static_for<NR>([&](auto ni){constexpr int n=decltype(ni)::value;
                    const int off=nfrag(ni)*16*T::TOTAL_K+(split*T::K+t*BK)*16+ks*2048+lane*16;
                    auto& frag=b_regs[decltype(phase)::value][ks][n];
                    set_slice(frag,load<16>(gb,off,0,number<T::B_CACHE>{}),0_I,16_I);
                    set_slice(frag,load<16>(gb,off+1024,0,number<T::B_CACHE>{}),16_I,32_I);
                });
            });
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    auto issue_scales=[&](int base,int kt) {
    static_for<T::SA_COPIES>([&](auto ii) {
        constexpr int i=decltype(ii)::value;
        const int index=(i*T::BLOCK_SIZE+tid)*T::SCALE_COPY,r=index%BM,sc=index/BM;
        int source=(sc<T::SCALE_COLS && row+r<shape_m)?(split*T::SC+kt*KS+sc)*args.stride_sfa+row+r:(T::TOTAL_K/128-1)*args.stride_sfa+shape_m;
        // A already masks invalid rows. Reading a neighbouring finite scale for
        // those rows is harmless; the global descriptor still bounds the tensor.
        if constexpr(T::SIMPLE_SCALE) source=(split*T::SC+kt*KS+(sc<T::SCALE_COLS?sc:T::SCALE_COLS-1))*args.stride_sfa+row+r;
        asm volatile("" : "+v"(source));
        auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SA+(i*T::BLOCK_SIZE+wave*64)*4));
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,T::SCALE_COPY,source,0,0,0);
    });
    static_for<T::SB_COPIES>([&](auto ii) {
        constexpr int i=decltype(ii)::value;
        const unsigned ix=i*T::BLOCK_SIZE+tid;
        const unsigned bg=ix/T::SCALE_COLS<T::SB_ROWS?ix/T::SCALE_COLS:T::SB_ROWS-1;
        auto* sbdst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SB+(i*T::BLOCK_SIZE+wave*64)*4));
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,1,(col/128+bg)*args.stride_sfb+split*T::SC+kt*KS+ix%T::SCALE_COLS,0,0,0);
    });
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
        if constexpr (T::PER_STAGE) issue_scales(slot*T::STAGE_BYTES,kt);
        __builtin_amdgcn_asyncmark();

    };
    static_for<T::B_DIRECT?T::B_AHEAD:0>([&](auto ti) {
        prefetch_b(decltype(ti)::value,number<decltype(ti)::value%T::B_SETS>{});
    });
    if constexpr(!T::PER_STAGE) issue_scales(0,0);
    static_for<T::INITIAL>([&](auto si){issue(decltype(si)::value,decltype(si)::value);});
    // Read all operands before issuing a later stage's DMA. The LLVM backend
    // must see those reads before that DMA to avoid an unnecessary vmcnt(0).
    auto read_stage=[&](int slot,int t,auto phase) {
        Stage f;
        auto& af=f.a; auto& bf=f.b; auto& as=f.sa; auto& bs=f.sb;
        static_for<KS>([&](auto ki){constexpr int ks=decltype(ki)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n=decltype(ni)::value;
                const unsigned bg=nfrag(ni)*16/128;
                bs[ks][n]=load<1>(ss,(T::PER_STAGE?slot*T::STAGE_BYTES:0)+T::OFF_SB+(bg*T::SCALE_COLS+(T::PER_STAGE?0:t*KS)+ks)*4)[0];
            });
            // All M repeats differ by whole 16-row panels. Keep their static
            // displacement outside the dynamic address, including the swizzle.
            const unsigned r0=mfrag(0_I)*16+lane16;
            unsigned alo=slot*T::STAGE_BYTES+r0*BK+((ks*8+kg)^(r0%(BK/16)))*16;
            unsigned ahi=slot*T::STAGE_BYTES+r0*BK+((ks*8+kg+4)^(r0%(BK/16)))*16;
            if constexpr(T::ADDRESS==2) {
                asm volatile("" : "+v"(alo), "+v"(ahi));
            }
            static_for<MR>([&](auto mi){constexpr int m=decltype(mi)::value;const int r=mfrag(mi)*16+lane16;
                constexpr unsigned delta=m*(T::CONTIGUOUS?1:T::T_M)*16;
                if constexpr(T::ADDRESS>0 && delta%(BK/16)==0) {
                    set_slice(af[ks][m],load<16>(sm,alo+delta*BK),0_I,16_I);
                    set_slice(af[ks][m],load<16>(sm,ahi+delta*BK),16_I,32_I);
                } else {
                    const int lo=slot*T::STAGE_BYTES+r*BK+(((ks*8+kg)^(r%(BK/16)))*16);
                    const int hi=slot*T::STAGE_BYTES+r*BK+(((ks*8+kg+4)^(r%(BK/16)))*16);
                    set_slice(af[ks][m],load<16>(sm,lo),0_I,16_I);set_slice(af[ks][m],load<16>(sm,hi),16_I,32_I);
                }
                as[ks][m]=load<1>(ss,(T::PER_STAGE?slot*T::STAGE_BYTES:0)+T::OFF_SA+(((T::PER_STAGE?0:t*KS)+ks)*BM+r)*(4/T::SCALE_COPY))[0];
            });
            static_for<NR>([&](auto ni){constexpr int n=decltype(ni)::value;
                if constexpr(T::B_DIRECT) bf[ks][n]=b_regs[decltype(phase)::value][ks][n];
                else {
                    const int ng=nfrag(ni),lo=slot*T::STAGE_BYTES+T::A_BYTES+ng*16*BK+ks*2048+lane*16;
                    set_slice(bf[ks][n],load<16>(sm,lo),0_I,16_I);
                    set_slice(bf[ks][n],load<16>(sm,lo+1024),16_I,32_I);
                }
            });
        });
        return f;
    };
    auto compute=[&](const Stage& f) {
        static_for<KS>([&](auto ki){constexpr int ks=decltype(ki)::value;
            static_for<MR>([&](auto mi){constexpr int m=decltype(mi)::value;
                static_for<NR>([&](auto ni){constexpr int n=decltype(ni)::value;
                    c[m*NR+n]=Base{}(f.a[ks][m],f.b[ks][n],c[m*NR+n],int(f.sa[ks][m]),int(f.sb[ks][n]),0_I,0_I);
                });
            });
        });
    };
    auto step=[&](int t,auto phase,auto pending,auto refill,auto refill_b) {
        if constexpr(T::ASYNC_GROUPS) __builtin_amdgcn_wait_asyncmark(decltype(pending)::value);
        else {
            constexpr int wait=decltype(pending)::value*(T::MATRIX_ISSUES+T::SA_COPIES+T::SB_COPIES);
            s_waitcnt_vmcnt(number<(wait>63?63:wait)>{});
        }
        __builtin_amdgcn_s_barrier();
        if constexpr(T::B_DIRECT && decltype(refill_b)::value)
            prefetch_b(t+T::B_AHEAD,number<(decltype(phase)::value+T::B_AHEAD)%T::B_SETS>{});
        auto f=read_stage(t%T::STAGES,t,phase);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(decltype(refill)::value) {
            issue((t+T::INITIAL)%T::STAGES,t+T::INITIAL);
            __builtin_amdgcn_sched_barrier(0);
        }
        compute(f);
    };
    constexpr int steady=T::PANELS-T::INITIAL;
    #pragma clang loop unroll(disable)
    for(int i=0;i<steady/T::UNROLL;++i) {
        static_for<T::UNROLL>([&](auto pi) {
            step(i*T::UNROLL+decltype(pi)::value,pi,number<T::INITIAL-1>{},1_I,1_I);
        });
    }
    static_for<steady%T::UNROLL>([&](auto ii) {
        constexpr int t=steady-steady%T::UNROLL+decltype(ii)::value;
        step(t,number<t%T::B_SETS>{},number<T::INITIAL-1>{},1_I,1_I);
    });
    static_for<T::INITIAL>([&](auto ji) {
        constexpr int j=decltype(ji)::value,t=steady+j;
        step(t,number<t%T::B_SETS>{},number<T::INITIAL-1-j>{},0_I,number<(t+T::B_AHEAD<T::PANELS)>{});
    });
    auto store_output=[&]() {
        auto output=make_gmem(final_output+static_cast<int64_t>(row)*stride_c+col,
            ((valid_rows-1)*stride_c+BN)*sizeof(bf16_t));
        if constexpr(T::OUTPUT<2) {
            static_for<MR*NR>([&](auto fi) {
                constexpr int f=decltype(fi)::value,mi=f/NR,ni=f%NR;
                auto values=cast<bf16_t>(c[f]);
                if constexpr(T::NORMAL) {
                    const int r=mfrag(number<mi>{})*16+kg*4;
                    const int col=nfrag(number<ni>{})*16+lane16;
                    static_for<4>([&](auto ei) {
                        constexpr int e=decltype(ei)::value;
                        store<1>(output,slice(values,number<e>{},number<e+1>{}),(r+e)*stride_c+col,0,0_I);
                    });
                } else {
                    const int r=mfrag(number<mi>{})*16+lane16;
                    const int col=nfrag(number<ni>{})*16+kg*4;
                    store<4>(output,values,r*stride_c+col,0,0_I);
                }
            });
        } else {
        constexpr int pitch=BN+8;
        auto staged=make_smem(reinterpret_cast<bf16_t*>(arena));
        __builtin_amdgcn_s_barrier();
        static_for<MR*NR>([&](auto fi) {
            constexpr int f=decltype(fi)::value,mi=f/NR,ni=f%NR;
            auto values=cast<bf16_t>(c[f]);
            if constexpr(T::NORMAL) {
                const int r=mfrag(number<mi>{})*16+kg*4;
                const int col=nfrag(number<ni>{})*16+lane16;
                static_for<4>([&](auto ei) {
                    constexpr int e=decltype(ei)::value;
                    store<1>(staged,slice(values,number<e>{},number<e+1>{}),(r+e)*pitch+col);
                });
            } else {
                const int r=mfrag(number<mi>{})*16+lane16;
                const int col=nfrag(number<ni>{})*16+kg*4;
                store<4>(staged,values,r*pitch+col);
            }
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        constexpr int chunks=BN/8;
        static_for<(BM*chunks+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto ii) {
            const int index=decltype(ii)::value*T::BLOCK_SIZE+tid;
            const int r=index/chunks,col=(index%chunks)*8;
            if(index<BM*chunks) {
                auto values=load<8>(staged,r*pitch+col);
                store<8>(output,values,r*stride_c+col,0,number<T::OUTPUT==2?2:0>{});
            }
        });
    }
    };
    if constexpr(T::SPLIT_K==1) {
        store_output();
    } else {
        // Continuous fragment slots keep both the producer and final gather
        // coalesced, including partial M tiles. Only final C is bounds-masked.
        constexpr int tile_elements = T::B_M * T::B_N;
        auto partials = make_gmem(reinterpret_cast<float *>(args.ptr_c) +
            static_cast<int64_t>(tile) * T::SPLIT_K * tile_elements,
            T::SPLIT_K * tile_elements * sizeof(float));
        static_for<T::E_M * T::E_N>([&](auto fi) {
            constexpr int f = decltype(fi)::value;
            store<4>(partials, c[f], split * tile_elements + (f * T::BLOCK_SIZE + thread_id_x()) * 4, 0, 0_I);
        });
        // Local-L2 mode follows the gfx950 XCD-grouped publication protocol:
        // every wave drains its stores, all waves arrive, then the counter updates.
        s_waitcnt_vmcnt(0_I);
        asm volatile("" ::: "memory");
        __builtin_amdgcn_s_barrier();
        __shared__ unsigned arrived;
        if (thread_id_x() == 0) {
            arrived = __hip_atomic_fetch_add(counters + tile, 1u, __ATOMIC_ACQ_REL, 3);
        }
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        if (arrived == T::SPLIT_K - 1) {
            const int valid_rows = shape_m - row < T::B_M ? shape_m - row : T::B_M;
            auto output = make_gmem(final_output + static_cast<int64_t>(row) * stride_c + col,
                ((valid_rows - 1) * stride_c + T::B_N) * sizeof(bf16_t));
            static_for<T::E_M * T::E_N>([&](auto fi) {
                constexpr int f = decltype(fi)::value;
                typename Base::vtype_c sum{};
                static_for<T::SPLIT_K>([&](auto si) {
                    constexpr int sk = decltype(si)::value;
                    sum += load<4>(partials, sk * tile_elements + (f * T::BLOCK_SIZE + thread_id_x()) * 4, 0, number<17>{});
                });
                c[f]=sum;
            });
            store_output();
            if (thread_id_x() == 0) {
                __hip_atomic_store(counters + tile, 0u, __ATOMIC_RELAXED, 3);
            }
        }
    }
}
template<class Traits>
__device__ __forceinline__ void direct(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    static_assert(Traits::SPLIT_K == 1);
    body<Traits>(args,reinterpret_cast<opus::bf16_t*>(args.ptr_c),nullptr);
}
} // namespace opus_gemm_composable_detail
#endif

// The existing tile and fused split launchers share this pipeline body.
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_composable_split_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,
    opus::bf16_t* output, unsigned* counters) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    static_assert(Traits::SPLIT_K>1);
    opus_gemm_composable_detail::body<Traits>(args,output,counters);
#endif
}
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_composable_direct_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_composable_detail::direct<Traits>(args);
#endif
}
