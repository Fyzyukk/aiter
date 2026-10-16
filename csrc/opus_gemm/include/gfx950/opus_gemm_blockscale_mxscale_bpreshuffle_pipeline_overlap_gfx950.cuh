#ifndef OPUS_BP_OVERLAP_PIPELINE_BASE
#define OPUS_BP_OVERLAP_PIPELINE_BASE
// SPDX-License-Identifier: Apache-2.0

#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_overlap_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_tiled_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_panel_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_pipeline_panel_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)

namespace opus_bpreshuffle_gfx950::overlap::schedule_0 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "pinned accumulator candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void medium270_pinned_a_async_b_multiahead_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
const int M=args.m,N=T::STATIC_N ? T::STATIC_N : args.n;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
const int index=(decltype(pi)::value*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<4>(sfb,load<4>(gsb,source),index);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=nr*K+lane*16;
unsigned b_offset=static_cast<unsigned>(offset)+static_cast<unsigned>(kt*2048);
asm volatile("" : "+v"(b_offset));
set_slice(bregs[set][n],load<16>(gb,b_offset,0),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,b_offset+1024,0),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
unsigned a_offset=static_cast<unsigned>(r<BM ? r*K+kc : valid_rows*K)+static_cast<unsigned>(kt*128);
asm volatile("" : "+v"(a_offset));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,a_offset,0,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto ai) {
issue_b(ai,decltype(ai)::value);
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
ascale[m]=load<1>(sfa,kt*BM+m*16+lane16)[0];
});
static_assert(BN==128 && T::T_N==2 && NR==4);
const unsigned shared_bscale=load<1>(sfb,kt)[0];
static_for<NR>([&](auto ni) {bscale[decltype(ni)::value]=shared_bscale;});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_b_i)::value)
issue_b_column(number<set>{},ni,kt+BD);
});
if constexpr(decltype(future_a_i)::value) {
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_1 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "pinned accumulator candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void medium270_pinned_a_async_b_multiahead_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
const int M=args.m,N=T::STATIC_N ? T::STATIC_N : args.n;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
const int index=(decltype(pi)::value*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<4>(sfb,load<4>(gsb,source),index);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=nr*K+lane*16;
set_slice(bregs[set][n],load<16>(gb,offset,kt*2048),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,kt*2048),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto ai) {
issue_b(ai,decltype(ai)::value);
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
ascale[m]=load<1>(sfa,kt*BM+m*16+lane16)[0];
});
static_assert(BN==128 && T::T_N==2 && NR==4);
const unsigned shared_bscale=load<1>(sfb,kt)[0];
static_for<NR>([&](auto ni) {bscale[decltype(ni)::value]=shared_bscale;});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_b_i)::value)
issue_b_column(number<set>{},ni,kt+BD);
});
if constexpr(decltype(future_a_i)::value) {
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_2 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "pinned accumulator candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void medium270_pinned_a_async_b_multiahead_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
const int M=args.m,N=T::STATIC_N ? T::STATIC_N : args.n;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a),
static_cast<unsigned>(M*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b),
static_cast<unsigned>(N*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
const int index=(decltype(pi)::value*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<4>(sfb,load<4>(gsb,source),index);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=(col+nr)*K+lane*16;
set_slice(bregs[set][n],load<16>(gb,offset,kt*2048),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,kt*2048),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<valid_rows ? (row+r)*K+kc : M*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto ai) {
issue_b(ai,decltype(ai)::value);
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,scale_group*LOOPS+kt)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_b_i)::value)
issue_b_column(number<set>{},ni,kt+BD);
});
if constexpr(decltype(future_a_i)::value) {
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_3 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "A operand AGPR candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void large_A_AGPR_bdirect_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
const int M=args.m,N=T::STATIC_N ? T::STATIC_N : args.n;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
const int index=(decltype(pi)::value*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<4>(sfb,load<4>(gsb,source),index);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<AD>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(decltype(future_b_i)::value)
issue_b(number<(phase+BD)%BS>{},kt+BD);
array<i32x8_t,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
i32x4_t al,ah;
[[clang::amdgpu_pin_agpr(m*8)]]
al=*reinterpret_cast<OPUS_LDS_ADDR i32x4_t*>(sa.ptr+lo);
[[clang::amdgpu_pin_agpr(m*8+4)]]
ah=*reinterpret_cast<OPUS_LDS_ADDR i32x4_t*>(sa.ptr+hi);
[[clang::amdgpu_pin_agpr(m*8)]]
afrags[m]=__builtin_shufflevector(al,ah,0,1,2,3,4,5,6,7);
ascale[m]=load<1>(sfa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,scale_group*LOOPS+kt)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
if constexpr(decltype(future_a_i)::value)
__builtin_amdgcn_asyncmark();
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_4 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "pinned accumulator candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void medium260_splitA_pinned_a_async_b_multiahead_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
const int M=args.m,N=T::STATIC_N ? T::STATIC_N : args.n;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int wm=wave%T::T_M,wn=wave/T::T_M;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
const int index=(decltype(pi)::value*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<4>(sfb,load<4>(gsb,source),index);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wn*(BN/T::T_N)+n*16;
const int offset=nr*K+lane*16;
set_slice(bregs[set][n],load<16>(gb,offset,kt*2048),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,kt*2048),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto ai) {
issue_b(ai,decltype(ai)::value);
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=(m*T::T_M+wm)*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(col+wn*(BN/T::T_N)+n*16)/128-col/128;
bscale[n]=load<1>(sfb,scale_group*LOOPS+kt)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_b_i)::value)
issue_b_column(number<set>{},ni,kt+BD);
});
if constexpr(decltype(future_a_i)::value) {
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=(m*T::T_M+wm)*16+kgroup*4+e,nc=wn*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=(m*T::T_M+wm)*16+kgroup*4+e,nc=wn*(BN/T::T_N)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_5 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "pinned accumulator candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void medium270_pinned_a_async_b_multiahead_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
const int M=args.m,N=T::STATIC_N ? T::STATIC_N : args.n;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
const int index=(decltype(pi)::value*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<4>(sfb,load<4>(gsb,source),index);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=nr*K+lane*16;
set_slice(bregs[set][n],load<16>(gb,offset,kt*2048),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,kt*2048),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto ai) {
issue_b(ai,decltype(ai)::value);
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,scale_group*LOOPS+kt)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_b_i)::value)
issue_b_column(number<set>{},ni,kt+BD);
});
if constexpr(decltype(future_a_i)::value) {
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_6 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "pinned accumulator candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void actual_medium_pin_160_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
constexpr int M=1856,N=7168;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(sfb.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,source,0,0,0);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=nr*K+lane*16;
set_slice(bregs[set][n],load<16>(gb,offset,kt*2048),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,kt*2048),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto ai) {
issue_b(ai,decltype(ai)::value);
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,scale_group*LOOPS+kt)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_b_i)::value)
issue_b_column(number<set>{},ni,kt+BD);
});
if constexpr(decltype(future_a_i)::value) {
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_7 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "pinned accumulator candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void actual_medium_pin_197_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
constexpr int M=896,N=7168;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(sfb.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,source,0,0,0);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=nr*K+lane*16;
set_slice(bregs[set][n],load<16>(gb,offset,kt*2048),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,kt*2048),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto ai) {
issue_b(ai,decltype(ai)::value);
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
ascale[m]=load<1>(sfa,kt*BM+m*16+lane16)[0];
});
static_assert(BN==128 && T::T_N==2 && NR==4);
const unsigned shared_bscale=load<1>(sfb,kt)[0];
static_for<NR>([&](auto ni) {bscale[decltype(ni)::value]=shared_bscale;});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_b_i)::value)
issue_b_column(number<set>{},ni,kt+BD);
});
if constexpr(decltype(future_a_i)::value) {
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_9 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_0;
using namespace opus_bpreshuffle_gfx950::overlap::schedule_8;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class Traits>
__device__ __forceinline__ 
void short_runtime_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = remove_cvref_t<Traits>;
using D_A = fp8_t;
using D_B = fp8_t;
using D_C = bf16_t;
using D_SF = unsigned char;
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const int row = block_id_y() * T::B_M;
const int col = block_id_x() * T::B_N;
const int k_tiles = kargs.k / T::B_K;
const int active_sfa_bytes = T::B_M * k_tiles;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a,
static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b);
auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + row * kargs.stride_c + col,
static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb));
alignas(16) __shared__ char smem[T::LDS_BYTES];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem + T::NUM_STAGES * T::A_STAGE));
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem + T::MATRIX_LDS_BYTES));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem + T::MATRIX_LDS_BYTES + T::SFA_BYTES));
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
auto mma = make_tiled_mma<D_A, D_B, fp32_t>(
seq<T::E_M, T::E_N, 1>{}, seq<T::T_M, T::T_N, 1>{},
seq<16, 16, 128>{}, mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using Chunk = vector_t<fp8_t, 16>;
array<AFragment, T::E_M> v_a;
array<BFragment, T::E_N> v_b;
array<AccFragment, T::E_M * T::E_N> c00{};
array<unsigned, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
array<unsigned, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
auto* a_chunks = reinterpret_cast<Chunk*>(&v_a);
auto* b_chunks = reinterpret_cast<Chunk*>(&v_b);
auto issue_matrix_prefetch = [&](auto stage, int group) {
constexpr int slot = decltype(stage)::value;
static_assert(slot >= 0 && slot < T::NUM_STAGES);
async_load<16>(g_a, s_a.ptr, u_ga, u_sa + slot * T::A_STAGE, group * T::B_K);
async_load<16>(g_b, s_b.ptr, u_gb, u_sb + slot * T::B_STAGE, group * T::B_K * T::W_N);
};
auto load_scale_panel = [&]() {
static_for<T::SFA_PASSES>([&](auto i) {
const int index = (thread_id_x() + decltype(i)::value * T::BLOCK_SIZE) * 16;
const int group = index / T::B_M;
const int local_row = index % T::B_M;
if (index < active_sfa_bytes) {
vector_t<D_SF, 16> values;
if (row + local_row < kargs.m)
values = load<16>(g_sfa, group * kargs.stride_sfa + row + local_row);
else
static_for<16>([&](auto j) { values[decltype(j)::value] = 0x7f; });
store<16>(s_sfa, values, index);
}
});
if (thread_id_x() < k_tiles) {
const int group = thread_id_x();
store<1>(s_sfb, load<1>(g_sfb, (col / T::GROUP_N) * kargs.stride_sfb + group),
thread_id_x());
}
};
auto read_scales = [&](int group, auto& scale_a, auto& scale_b) {
static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
static_for<T::E_M>([&](auto i) {
constexpr int repeat = decltype(i)::value;
const int local_row = wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
const unsigned value = load<1>(s_sfa, group * T::B_M + local_row)[0];
scale_a[repeat / 4] |= value << ((repeat % 4) * 8);
});
static_for<T::B_SCALE_PACKS>([&](auto i) {
constexpr int half_n = decltype(i)::value;
scale_b[half_n] = load<1>(s_sfb, half_n * T::SCALE_PANEL + group)[0];
});
};
auto load_a = [&](auto m, auto stage) {
constexpr int repeat = decltype(m)::value;
constexpr int slot = decltype(stage)::value;
static_for<2>([&](auto c) {
constexpr int index = repeat * 2 + decltype(c)::value;
const int offset = T::a_lds_offset(wave_id_m, lane_id, repeat, decltype(c)::value) + slot * T::A_STAGE;
[[clang::amdgpu_pin_agpr(T::C_REGS + index * 4)]]
a_chunks[index] = load<16>(s_a, offset);
});
};
auto load_b = [&](auto n, auto stage) {
constexpr int repeat = decltype(n)::value;
constexpr int slot = decltype(stage)::value;
const auto offsets = layout_to_offsets<16>(u_rb + slot * T::B_STAGE);
static_for<2>([&](auto c) {
constexpr int index = repeat * 2 + decltype(c)::value;
b_chunks[index] = load<16>(s_b, offsets[index]);
});
};
auto compute = [&](auto m, auto n) {
constexpr int mi = decltype(m)::value, ni = decltype(n)::value;
constexpr int ci = mi * T::E_N + ni;
constexpr int half_n = ni / (T::GROUP_N / (T::T_N * T::W_N));
[[clang::amdgpu_pin_agpr(ci * sizeof(AccFragment) / sizeof(u32_t))]]
c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
number<mi % 4>{}, number<0>{});
};
load_scale_panel();
issue_matrix_prefetch(number<0>{}, 0);
if (k_tiles > 1)
issue_matrix_prefetch(number<1>{}, 1);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(0, v_sfa, v_sfb);
static_for<T::E_M>([&](auto m) { load_a(m, number<0>{}); });
static_for<T::E_N>([&](auto n) { load_b(n, number<0>{}); });
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
auto advance = [&](auto current_stage, int group) {
constexpr int stage = decltype(current_stage)::value;
constexpr int next_stage = 1 - stage;
if (group + 2 < k_tiles)
issue_matrix_prefetch(number<stage>{}, group + 2);
read_scales(group + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m) {
static_for<T::E_N>([&](auto n) {
compute(m, n);
if constexpr (decltype(m)::value == T::E_M - 1)
load_b(n, number<next_stage>{});
});
load_a(m, number<next_stage>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
v_sfa = v_sfa_next;
v_sfb = v_sfb_next;
};
int group = 0;
#pragma unroll 1
for (; group + 2 < k_tiles; group += 2) {
advance(number<0>{}, group);
advance(number<1>{}, group + 1);
}
if (group + 1 < k_tiles)
advance(number<0>{}, group);
static_for<T::E_M>([&](auto m) {
static_for<T::E_N>([&](auto n) { compute(m, n); });
});
constexpr int C_PITCH = T::B_N + 8;
constexpr int C_BYTES = T::B_M * C_PITCH * sizeof(D_C);
constexpr int OUTPUT_VEC = 8;
constexpr int OUTPUT_PASSES = T::B_M * T::B_N / (T::BLOCK_SIZE * OUTPUT_VEC);
static_assert(C_PITCH == 136 && C_BYTES <= T::LDS_BYTES);
static_assert(T::B_M * T::B_N % (T::BLOCK_SIZE * OUTPUT_VEC) == 0);
auto s_c = make_smem(reinterpret_cast<D_C*>(smem));
const auto p_coord = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
const auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(C_PITCH, 1_I), p_coord);
const auto c_offsets = layout_to_offsets<T::VEC_C>(u_sc);
static_for<T::E_M * T::E_N>([&](auto i) {
constexpr int index = decltype(i)::value;
store<T::VEC_C>(s_c, cast<D_C>(c00[index]), c_offsets[index]);
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<OUTPUT_PASSES>([&](auto pass) {
const int linear = thread_id_x() * OUTPUT_VEC + decltype(pass)::value * T::BLOCK_SIZE * OUTPUT_VEC;
const int output_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<OUTPUT_VEC>(s_c, output_row * C_PITCH + output_col);
store<OUTPUT_VEC>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
});
}
#endif
}
#endif

#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_10) && !defined(OPUS_BP_INCLUDE_OVERLAP_10_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_10_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_10 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a),
static_cast<unsigned>(M*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b),
static_cast<unsigned>(N*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*16,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
const int sb_source=tid<T::SFB_BYTES ? (col/128+tid/LOOPS)*LOOPS+tid%LOOPS:(N/128)*LOOPS;
auto* sb_dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfb.ptr+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sb_dst,1,sb_source,0,0,0);
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=(col+nr)*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<valid_rows ? (row+r)*K+kc : M*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto scale_byte=[&]<int OFFSET>(unsigned base,number<OFFSET>) {
const unsigned addr=static_cast<unsigned>(reinterpret_cast<__UINTPTR_TYPE__>(arena))+base;
unsigned value;
asm volatile("ds_read_u8 %0, %1 offset:%2\n" : "=v"(value) : "v"(addr),"n"(OFFSET) : "memory");
return value;
};
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=scale_byte(lane16,number<T::OFF_SFA+t*BM+m*16>{});
});
static_assert(BN==128 && T::T_N==4 && NR==2);
const unsigned shared_bscale=scale_byte(0,number<T::OFF_SFB+t*4>{});
static_for<NR>([&](auto ni){bscale[decltype(ni)::value]=shared_bscale;});
s_waitcnt_lgkmcnt(0_I);
asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_11) && !defined(OPUS_BP_INCLUDE_OVERLAP_11_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_11_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_11 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_10;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_12) && !defined(OPUS_BP_INCLUDE_OVERLAP_12_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_12_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_12 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a),
static_cast<unsigned>(M*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b),
static_cast<unsigned>(N*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*16,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
const int sb_source=tid<T::SFB_BYTES ? (col/128+tid/LOOPS)*LOOPS+tid%LOOPS:(N/128)*LOOPS;
auto* sb_dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfb.ptr+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sb_dst,1,sb_source,0,0,0);
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=(col+nr)*K+kt*2048;
unsigned b_offset=static_cast<unsigned>(offset+soffset);
asm volatile("" : "+v"(b_offset));
set_slice(bregs[set][n],load<16>(gb,b_offset,0,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,b_offset+1024,0,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
unsigned dma_offset=(r<valid_rows ? (row+r)*K+kc : M*K)+kt*128;
asm volatile("" : "+v"(dma_offset));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,dma_offset,0,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,(scale_group*LOOPS+t)*4)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_13) && !defined(OPUS_BP_INCLUDE_OVERLAP_13_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_13_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_13 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_12;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_14) && !defined(OPUS_BP_INCLUDE_OVERLAP_14_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_14_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_14 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a),
static_cast<unsigned>(M*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b),
static_cast<unsigned>(N*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*16,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
const int sb_source=tid<T::SFB_BYTES ? (col/128+tid/LOOPS)*LOOPS+tid%LOOPS:(N/128)*LOOPS;
auto* sb_dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfb.ptr+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sb_dst,1,sb_source,0,0,0);
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=(col+nr)*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<valid_rows ? (row+r)*K+kc : M*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,(scale_group*LOOPS+t)*4)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_15) && !defined(OPUS_BP_INCLUDE_OVERLAP_15_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_15_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_15 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_14;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_16) && !defined(OPUS_BP_INCLUDE_OVERLAP_16_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_16_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_16 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct medium192_allLDS_traits {
static constexpr int B_M=192,B_N=128,B_K=128,T_M=4,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=512,MR=3,NR=4,STAGES=3,GLOBAL_SPLIT=1;
static constexpr int K=7168,PANELS=56,B_CACHE=0;
static constexpr int A_BYTES=B_M*B_K,B_BYTES=B_N*B_K,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int SA_BYTES=((B_M*PANELS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int SB_BYTES=BLOCK_SIZE*4,OFF_SA=STAGES*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=1,ISSUES=A_COPIES+B_COPIES;
static constexpr int PIPELINE_BYTES=OFF_SB+SB_BYTES,OUTPUT_BYTES=B_M*(B_N+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static_assert(PANELS%4==0 && A_COPIES==3 && B_COPIES==2 && SA_COPIES==6);
static_assert(LDS_BYTES==137216 && LDS_BYTES<=160*1024);
};
template<class T> __global__ void medium192_allLDS_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

struct medium192_allLDS_traits {
static constexpr int B_M=192,B_N=128,B_K=128,T_M=4,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=512,MR=3,NR=4,STAGES=3,GLOBAL_SPLIT=1;
static constexpr int K=7168,PANELS=56,B_CACHE=0;
static constexpr int A_BYTES=B_M*B_K,B_BYTES=B_N*B_K,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int SA_BYTES=((B_M*PANELS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int SB_BYTES=BLOCK_SIZE*4,OFF_SA=STAGES*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=1,ISSUES=A_COPIES+B_COPIES;
static constexpr int PIPELINE_BYTES=OFF_SB+SB_BYTES,OUTPUT_BYTES=B_M*(B_N+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static_assert(PANELS%4==0 && A_COPIES==3 && B_COPIES==2 && SA_COPIES==6);
static_assert(LDS_BYTES==137216 && LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void medium192_allLDS_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(T::BLOCK_SIZE,1)
void medium192_allLDS_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,P=T::PANELS,MR=T::MR,NR=T::NR;
const int M=args.m,N=args.n,tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wm=wave/2,wn=wave%2;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=N==6144?pid%8*(total/8)+pid/8:pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN,valid_rows=M-row<BM?M-row:BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*P));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*P));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
static_for<T::SA_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,sr=index%(BM/4)*4,scol=index/(BM/4);
const int sc=scol<P?scol:P-1;
const int source=row+sr<M?sc*M+row+sr:M*P;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SA+(i*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
static_for<T::SB_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int word=(i*T::BLOCK_SIZE+tid)%(P/4);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SB+(i*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,(col/128)*P+word*4,0,0,0);
});
auto issue=[&](int slot,int kt) {
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/128,kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,kt*128,0,0);
});
static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,group=lin/2048,within=lin%2048;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+T::A_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,group*(T::K*16)+within,kt*2048,0,T::B_CACHE);
});
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
static_for<NB-1>([&](auto si){issue(decltype(si)::value,decltype(si)::value);});
auto step=[&](int t,int slot,auto pi,auto fi) {
constexpr int pending=decltype(pi)::value;
constexpr bool future=decltype(fi)::value;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::ISSUES*pending>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> af;
array<typename Base::vtype_b,NR> bf;
array<unsigned,MR> as;
const unsigned bs=load<1>(ss,T::OFF_SB+t)[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=wm*48+m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*128+((kg^(ar%8))*16);
const int hi=slot*T::STAGE_BYTES+ar*128+(((kg+4)^(ar%8))*16);
set_slice(af[m],load<16>(sa,lo),0_I,16_I);
set_slice(af[m],load<16>(sa,hi),16_I,32_I);
as[m]=load<1>(ss,T::OFF_SA+t*BM+ar)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*64+n*16;
const int bo=slot*T::STAGE_BYTES+T::A_BYTES+(br/16)*2048+kg*256+lane16*16;
set_slice(bf[n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[n],load<16>(sa,bo+1024),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(future)issue(slot==0?NB-1:slot-1,t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(af[m],bf[n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=P-(NB-1);
int ring_slot=0;
#pragma clang loop unroll(disable)
for(int t=0;t<STEADY;++t) {
step(t,ring_slot,number<NB-2>{},number<true>{});
ring_slot=ring_slot+1==NB?0:ring_slot+1;
}
static_for<NB-1>([&](auto ti) {
constexpr int t=STEADY+decltype(ti)::value;
step(t,t%NB,number<P-t-1>{},number<false>{});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+kg*4,nn=wn*64+(ci%NR)*16+lane16;
const auto value=cast<bf16_t>(c[ci]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
if(rr+e<valid_rows)store<1>(sc,vector_t<bf16_t,1>{value[e]},(rr+e)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto ii) {
const int index=decltype(ii)::value*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn);
});
s_waitcnt_vmcnt(0_I);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_17) && !defined(OPUS_BP_INCLUDE_OVERLAP_17_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_17_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_17 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_16;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium192_allLDS_traits;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
const bool supported=(n==7168 && (m==672 || m==704 || m==736 || m==768)) ||
(n==6144 && (m==864 || m==896));
if(selector!=0 || literal!=1 || !supported || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,4,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
medium192_allLDS_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium192_allLDS_traits;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
const bool supported=(n==7168 && (m==672 || m==704 || m==736 || m==768)) ||
(n==6144 && (m==864 || m==896));
if(selector!=0 || literal!=1 || !supported || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,4,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
medium192_allLDS_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_18) && !defined(OPUS_BP_INCLUDE_OVERLAP_18_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_18_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_18 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_16;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium192_allLDS_traits;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
const bool supported=n==6144 && (m==832 || m==928 || m==960);
if(selector!=0 || literal!=1 || !supported || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,4,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
medium192_allLDS_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium192_allLDS_traits;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
const bool supported=n==6144 && (m==832 || m==928 || m==960);
if(selector!=0 || literal!=1 || !supported || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,4,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
medium192_allLDS_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_19) && !defined(OPUS_BP_INCLUDE_OVERLAP_19_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_19_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_19 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0>
struct medium274_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=128;
static constexpr int T_M=1,T_N=2,E_M=BM/16,E_N=BN/32;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_STAGE=BM*128,A_PASSES=(A_STAGE+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_BYTES=BM*LOOPS;
static constexpr int SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=(SFB_BYTES+511)/512*512;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=Output==1 && OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES : PIPELINE_BYTES;
static constexpr int gcd(int a,int b) { return b ? gcd(b,a%b) : a; }
static constexpr int UNROLL=Stages*B_SETS/gcd(Stages,B_SETS);
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%32==0);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
template<class T>
__global__ void medium274_a_async_b_multiahead_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0>
struct medium274_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=128;
static constexpr int T_M=1,T_N=2,E_M=BM/16,E_N=BN/32;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_STAGE=BM*128,A_PASSES=(A_STAGE+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_BYTES=BM*LOOPS;
static constexpr int SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=(SFB_BYTES+511)/512*512;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=Output==1 && OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES : PIPELINE_BYTES;
static constexpr int gcd(int a,int b) { return b ? gcd(b,a%b) : a; }
static constexpr int UNROLL=Stages*B_SETS/gcd(Stages,B_SETS);
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%32==0);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void medium274_a_async_b_multiahead_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void medium274_a_async_b_multiahead_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD,U=T::UNROLL;
const int M=args.m,N=T::STATIC_N ? T::STATIC_N : args.n;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_MAPPING ? (pid%8u)*group+pid/8u : pid;
if(tile>=total) return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r : M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if constexpr(T::SCALE_PACKET==1) {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
const int index=(decltype(pi)::value*T::BLOCK_SIZE+tid)*4;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<4>(sfb,load<4>(gsb,source),index);
}
});
} else {
static_for<(T::SFB_BYTES+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid;
if(index<T::SFB_BYTES) {
const int source=(col/128+index/LOOPS)*LOOPS+index%LOOPS;
store<1>(sfb,load<1>(gsb,source),index);
}
});
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/2)+n*16;
const int offset=nr*K+lane*16;
set_slice(bregs[set][n],load<16>(gb,offset,kt*2048),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,kt*2048),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_assert(AD==BD);
static_for<AD>([&](auto ai) {
issue_b(ai,decltype(ai)::value);
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto phase_i,int kt,auto future_a_i,auto future_b_i,auto pending_i) {
constexpr int phase=decltype(phase_i)::value,slot=phase%STAGES,set=phase%BS;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/2)+n*16)/128;
bscale[n]=load<1>(sfb,scale_group*LOOPS+kt)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_a_i)::value)
issue_a(number<(phase+AD)%STAGES>{},kt+AD);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(decltype(future_b_i)::value)
issue_b_column(number<set>{},ni,kt+BD);
});
if constexpr(decltype(future_a_i)::value) {
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=LOOPS-AD,FULL=(STEADY/U)*U;
#pragma clang loop unroll(disable)
for(int base=0;base<FULL;base+=U) {
static_for<U>([&](auto pi) { step(pi,base+decltype(pi)::value,1_I,1_I,number<STAGES-2>{}); });
}
static_for<STEADY-FULL>([&](auto pi) {
constexpr int t=FULL+decltype(pi)::value;
step(number<t%U>{},t,1_I,1_I,number<STAGES-2>{});
});
static_for<AD>([&](auto ji) {
constexpr int j=decltype(ji)::value,t=STEADY+j;
step(number<t%U>{},t,0_I,number<(t+BD<LOOPS)>{},number<STAGES-2-j>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/2)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
} else {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/2)+n*16+lane16;
if(r<valid_rows) store<1>(gc,vector_t<bf16_t,1>{value[e]},r*N+nc);
});
});
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_20) && !defined(OPUS_BP_INCLUDE_OVERLAP_20_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_20_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_20 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,2,1,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,2,1,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_21) && !defined(OPUS_BP_INCLUDE_OVERLAP_21_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_21_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_21 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,2,1,384,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,2,1,384,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_22) && !defined(OPUS_BP_INCLUDE_OVERLAP_22_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_22_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_22 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,2,1,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,2,1,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_23) && !defined(OPUS_BP_INCLUDE_OVERLAP_23_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_23_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_23 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,1536,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,1536,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_24) && !defined(OPUS_BP_INCLUDE_OVERLAP_24_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_24_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_24 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,16384,0,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,16384,0,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_25) && !defined(OPUS_BP_INCLUDE_OVERLAP_25_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_25_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_25 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_26) && !defined(OPUS_BP_INCLUDE_OVERLAP_26_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_26_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_26 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,3072,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,3072,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_27) && !defined(OPUS_BP_INCLUDE_OVERLAP_27_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_27_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_27 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,7168,0,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,7168,0,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_28) && !defined(OPUS_BP_INCLUDE_OVERLAP_28_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_28_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_28 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,7168,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,7168,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_29) && !defined(OPUS_BP_INCLUDE_OVERLAP_29_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_29_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_29 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_30) && !defined(OPUS_BP_INCLUDE_OVERLAP_30_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_30_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_30 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,768,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,3,2,768,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_31) && !defined(OPUS_BP_INCLUDE_OVERLAP_31_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_31_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_31 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,4,3,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,4,3,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_32) && !defined(OPUS_BP_INCLUDE_OVERLAP_32_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_32_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_32 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,4,3,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<112,256,4,3,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_33) && !defined(OPUS_BP_INCLUDE_OVERLAP_33_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_33_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_33 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,2,1,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,2,1,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_34) && !defined(OPUS_BP_INCLUDE_OVERLAP_34_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_34_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_34 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,2,1,384,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,2,1,384,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_35) && !defined(OPUS_BP_INCLUDE_OVERLAP_35_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_35_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_35 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,2,1,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,2,1,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_36) && !defined(OPUS_BP_INCLUDE_OVERLAP_36_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_36_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_36 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,1536,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,1536,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_37) && !defined(OPUS_BP_INCLUDE_OVERLAP_37_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_37_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_37 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,16384,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,16384,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_38) && !defined(OPUS_BP_INCLUDE_OVERLAP_38_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_38_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_38 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_39) && !defined(OPUS_BP_INCLUDE_OVERLAP_39_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_39_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_39 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,3072,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,3072,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_40) && !defined(OPUS_BP_INCLUDE_OVERLAP_40_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_40_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_40 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_41) && !defined(OPUS_BP_INCLUDE_OVERLAP_41_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_41_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_41 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,768,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,3,2,768,1,1,0,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_42) && !defined(OPUS_BP_INCLUDE_OVERLAP_42_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_42_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_42 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,4,3,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,4,3,16384,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_43) && !defined(OPUS_BP_INCLUDE_OVERLAP_43_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_43_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_43 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,4,3,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<128,256,4,3,7168,1,1,1,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_44) && !defined(OPUS_BP_INCLUDE_OVERLAP_44_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_44_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_44 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<32,128,3,2,7168,1,1,1,2048>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<32,128,3,2,7168,1,1,1,2048>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_45) && !defined(OPUS_BP_INCLUDE_OVERLAP_45_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_45_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_45 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<32,128,3,2,7168,1,1,1,768>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<32,128,3,2,7168,1,1,1,768>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_46) && !defined(OPUS_BP_INCLUDE_OVERLAP_46_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_46_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_46 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<32,128,3,2,7168,1,2,1,2048>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<32,128,3,2,7168,1,2,1,2048>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_47) && !defined(OPUS_BP_INCLUDE_OVERLAP_47_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_47_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_47 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<32,128,3,2,7168,1,2,1,768>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<32,128,3,2,7168,1,2,1,768>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_48) && !defined(OPUS_BP_INCLUDE_OVERLAP_48_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_48_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_48 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<64,128,3,2,7168,1,1,1,2048>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<64,128,3,2,7168,1,1,1,2048>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_49) && !defined(OPUS_BP_INCLUDE_OVERLAP_49_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_49_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_49 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<64,128,3,2,7168,1,1,1,768>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<64,128,3,2,7168,1,1,1,768>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_50) && !defined(OPUS_BP_INCLUDE_OVERLAP_50_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_50_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_50 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<64,128,3,2,7168,1,2,1,2048>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<64,128,3,2,7168,1,2,1,2048>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_51) && !defined(OPUS_BP_INCLUDE_OVERLAP_51_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_51_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_51 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_19;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<64,128,3,2,7168,1,2,1,768>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=medium274_a_async_b_ahead_traits<64,128,3,2,7168,1,2,1,768>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m<257 || m>4096 || m%16 || n%Traits::B_N || n%128 ||
k!=Traits::FIXED_K || (Traits::STATIC_N && n!=Traits::STATIC_N)) return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked) return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
const unsigned workers=Traits::XCD_MAPPING ? (total+7u)/8u*8u : total;
medium274_a_async_b_multiahead_kernel<Traits><<<dim3(workers,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_52) && !defined(OPUS_BP_INCLUDE_OVERLAP_52_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_52_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_52 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=2;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=BN*256,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
template<class T> __global__ void mediumK256_fromBMM_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=2;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=BN*256,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void mediumK256_fromBMM_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(256,1)
void mediumK256_fromBMM_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,float* partials,int* counters) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,SK=T::GLOBAL_SPLIT,P=T::PANELS,MR=T::MR,NR=T::NR;
const int M=args.m,N=args.n,tid=thread_id_x(),lane=tid%64;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wm=wave/2,wn=wave%2,kg=lane/16,lane16=lane%16;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=pid/(8*SK)*8+pid%8,split=pid/8%SK;
if(tile>=total)return;
const unsigned mtile=tile%gm,ntile=tile/gm;
const int row=mtile*BM,col=ntile*BN,valid_rows=M-row<BM?M-row:BM;
const int begin=split*P;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::TOTAL_SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::TOTAL_SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
auto issue_scales=[&](auto si,int kt) {
constexpr int slot=decltype(si)::value;
const int base=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
static_for<T::SA_COPIES>([&](auto ii) {
const int index=decltype(ii)::value*256+tid;
const int sr=index%(BM/4)*4,scol=index/(BM/4);
const int sc=scol<T::SA_COLS?scol:T::SA_COLS-1;
const int source=row+sr<M?(kt*2+sc)*M+row+sr:M*T::TOTAL_SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SA+(decltype(ii)::value*256+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
static_for<T::SB_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SB+(i*256+wave*64)*4));
if constexpr(T::SB_SLOTS) {
const int scol=(i*256+tid)%T::SA_COLS;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,(col/128)*T::TOTAL_SCALES+kt*2+scol,0,0,0);
} else {
const int dword=(i*256+tid)%(T::SA_COLS/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,(col/128)*T::TOTAL_SCALES+kt*2+dword*4,0,0,0);
}
});
};
auto issue=[&](auto si,int kt) {
constexpr int slot=decltype(si)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*256+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,kt*256,0,0);
});
static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+T::A_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,i*16*T::K+tid*16,kt*4096,0,T::B_CACHE);
});
if constexpr(!T::SCALE_PRELOAD)issue_scales(si,kt);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::SCALE_PRELOAD)issue_scales(0_I,begin);
static_for<NB-1>([&](auto si){issue(si,begin+decltype(si)::value);});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%NB,pending=P-t-1<NB-2?P-t-1:NB-2;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::ISSUES*pending>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<array<unsigned,NR>,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
const int sbase=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
const int scale_col=T::SCALE_PRELOAD?t*2+ks:ks;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=wm*48+m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,sbase+T::OFF_SA+scale_col*BM+ar)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/2)+n*16;
const int bo=slot*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
bs[ks][n]=load<1>(ss,sbase+T::OFF_SB+scale_col*(T::SB_SLOTS?4:1))[0];
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+NB-1<P)issue(number<(slot+NB-1)%NB>{},begin+t+NB-1);
__builtin_amdgcn_sched_barrier(0);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks][n]),0_I,0_I);
});
});
});
};
static_for<P>([&](auto ti){step(ti);});
const int plane=valid_rows*BN;
const int64_t tile_base=(int64_t(row)*N+int64_t(ntile)*plane)*SK;
auto partial=make_gmem(partials+tile_base+int64_t(split)*plane,static_cast<unsigned>(plane*4));
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)store<4>(partial,c[ci],rr*BN+nn,0,number<17>{});
});
auto* flag=reinterpret_cast<OPUS_LDS_ADDR unsigned*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::FLAG_OFF));
s_waitcnt_vmcnt(0_I);
__builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
__builtin_amdgcn_s_barrier();
if(tid==0) {
const int old=__hip_atomic_fetch_add(counters+tile,1,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);
*flag=old==SK-1;
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
const unsigned last=*flag;
s_waitcnt_lgkmcnt(0_I);
if(last) {
__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
array<vector_t<float,4>,MR*NR> sums{};
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows) {
static_for<SK>([&](auto ski) {
auto src=make_gmem(partials+tile_base+int64_t(decltype(ski)::value)*plane,static_cast<unsigned>(plane*4));
sums[ci]+=load<4>(src,rr*BN+nn,0,number<17>{});
});
}
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)store<4>(sc,cast<bf16_t>(sums[ci]),rr*(BN+8)+nn);
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+255)/256>([&](auto ii) {
const int index=decltype(ii)::value*256+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn,0,number<2>{});
});
s_waitcnt_vmcnt(0_I);
__builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
__builtin_amdgcn_s_barrier();
if(tid==0) {
__hip_atomic_store(counters+tile,0,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
s_waitcnt_vmcnt(0_I);
}
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_53) && !defined(OPUS_BP_INCLUDE_OVERLAP_53_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_53_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_53 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_52;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,3,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,3,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_54) && !defined(OPUS_BP_INCLUDE_OVERLAP_54_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_54_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_54 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_52;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_55) && !defined(OPUS_BP_INCLUDE_OVERLAP_55_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_55_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_55 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_52;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_56) && !defined(OPUS_BP_INCLUDE_OVERLAP_56_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_56_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_56 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_52;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_57) && !defined(OPUS_BP_INCLUDE_OVERLAP_57_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_57_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_57 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_52;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_58) && !defined(OPUS_BP_INCLUDE_OVERLAP_58_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_58_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_58 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_52;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_fromBMM_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_59) && !defined(OPUS_BP_INCLUDE_OVERLAP_59_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_59_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_59 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=2;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=BN*256,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
template<class T> __global__ void mediumK256_AV_agent_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=2;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=BN*256,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void mediumK256_AV_agent_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(256,1)
void mediumK256_AV_agent_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,float* partials,int* counters) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,SK=T::GLOBAL_SPLIT,P=T::PANELS,MR=T::MR,NR=T::NR;
const int M=args.m,N=args.n,tid=thread_id_x(),lane=tid%64;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wm=wave/2,wn=wave%2,kg=lane/16,lane16=lane%16;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=pid/(8*SK)*8+pid%8,split=pid/8%SK;
if(tile>=total)return;
const unsigned mtile=tile%gm,ntile=tile/gm;
const int row=mtile*BM,col=ntile*BN,valid_rows=M-row<BM?M-row:BM;
const int begin=split*P;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::TOTAL_SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::TOTAL_SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
auto issue_scales=[&](auto si,int kt) {
constexpr int slot=decltype(si)::value;
const int base=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
static_for<T::SA_COPIES>([&](auto ii) {
const int index=decltype(ii)::value*256+tid;
const int sr=index%(BM/4)*4,scol=index/(BM/4);
const int sc=scol<T::SA_COLS?scol:T::SA_COLS-1;
const int source=row+sr<M?(kt*2+sc)*M+row+sr:M*T::TOTAL_SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SA+(decltype(ii)::value*256+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
static_for<T::SB_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SB+(i*256+wave*64)*4));
if constexpr(T::SB_SLOTS) {
const int scol=(i*256+tid)%T::SA_COLS;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,(col/128)*T::TOTAL_SCALES+kt*2+scol,0,0,0);
} else {
const int dword=(i*256+tid)%(T::SA_COLS/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,(col/128)*T::TOTAL_SCALES+kt*2+dword*4,0,0,0);
}
});
};
auto issue=[&](auto si,int kt) {
constexpr int slot=decltype(si)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*256+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,kt*256,0,0);
});
static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+T::A_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,i*16*T::K+tid*16,kt*4096,0,T::B_CACHE);
});
if constexpr(!T::SCALE_PRELOAD)issue_scales(si,kt);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::SCALE_PRELOAD)issue_scales(0_I,begin);
static_for<NB-1>([&](auto si){issue(si,begin+decltype(si)::value);});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%NB,pending=P-t-1<NB-2?P-t-1:NB-2;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::ISSUES*pending>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<array<unsigned,NR>,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
const int sbase=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
const int scale_col=T::SCALE_PRELOAD?t*2+ks:ks;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=wm*48+m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,sbase+T::OFF_SA+scale_col*BM+ar)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/2)+n*16;
const int bo=slot*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
bs[ks][n]=load<1>(ss,sbase+T::OFF_SB+scale_col*(T::SB_SLOTS?4:1))[0];
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+NB-1<P)issue(number<(slot+NB-1)%NB>{},begin+t+NB-1);
__builtin_amdgcn_sched_barrier(0);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks][n]),0_I,0_I);
});
});
});
};
static_for<P>([&](auto ti){step(ti);});
const int plane=valid_rows*BN;
const int64_t tile_base=(int64_t(row)*N+int64_t(ntile)*plane)*SK;
using av_u32x4 = unsigned __attribute__((ext_vector_type(4)));
float* partial=partials+tile_base+int64_t(split)*plane;
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)
__builtin_amdgcn_av_store_b128(reinterpret_cast<av_u32x4*>(partial+rr*BN+nn),
__builtin_bit_cast(av_u32x4,c[ci]),__MEMORY_SCOPE_DEVICE);
});
auto* flag=reinterpret_cast<OPUS_LDS_ADDR unsigned*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::FLAG_OFF));
s_waitcnt_vmcnt(0_I);
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
if(tid==0) {
const int old=__scoped_atomic_fetch_add(counters+tile,1,__ATOMIC_ACQ_REL,__MEMORY_SCOPE_DEVICE);
*flag=old==SK-1;
}
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
const unsigned last=*flag;
s_waitcnt_lgkmcnt(0_I);
if(last) {
array<vector_t<float,4>,MR*NR> sums{};
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows) {
static_for<SK>([&](auto ski) {
float* src=partials+tile_base+int64_t(decltype(ski)::value)*plane+rr*BN+nn;
sums[ci]+=__builtin_bit_cast(vector_t<float,4>,
__builtin_amdgcn_av_load_b128(reinterpret_cast<av_u32x4*>(src),__MEMORY_SCOPE_DEVICE));
});
}
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)store<4>(sc,cast<bf16_t>(sums[ci]),rr*(BN+8)+nn);
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+255)/256>([&](auto ii) {
const int index=decltype(ii)::value*256+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn,0,number<2>{});
});
s_waitcnt_vmcnt(0_I);
__builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
if(tid==0) {
[[clang::amdgpu_av("none")]] __scoped_atomic_store_n(counters+tile,0,__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
s_waitcnt_vmcnt(0_I);
}
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_60) && !defined(OPUS_BP_INCLUDE_OVERLAP_60_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_60_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_60 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_59;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,3,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,3,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_61) && !defined(OPUS_BP_INCLUDE_OVERLAP_61_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_61_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_61 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_59;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_62) && !defined(OPUS_BP_INCLUDE_OVERLAP_62_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_62_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_62 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_59;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_63) && !defined(OPUS_BP_INCLUDE_OVERLAP_63_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_63_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_63 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_59;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_64) && !defined(OPUS_BP_INCLUDE_OVERLAP_64_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_64_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_64 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_59;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,2>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_65) && !defined(OPUS_BP_INCLUDE_OVERLAP_65_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_65_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_65 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_59;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_agent_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_66) && !defined(OPUS_BP_INCLUDE_OVERLAP_66_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_66_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_66 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=0;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=BN*256,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
template<class T> __global__ void mediumK256_AV_B0_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=0;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=BN*256,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void mediumK256_AV_B0_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(256,1)
void mediumK256_AV_B0_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,float* partials,int* counters) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,SK=T::GLOBAL_SPLIT,P=T::PANELS,MR=T::MR,NR=T::NR;
const int M=args.m,N=args.n,tid=thread_id_x(),lane=tid%64;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wm=wave/2,wn=wave%2,kg=lane/16,lane16=lane%16;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=pid/(8*SK)*8+pid%8,split=pid/8%SK;
if(tile>=total)return;
const unsigned mtile=tile%gm,ntile=tile/gm;
const int row=mtile*BM,col=ntile*BN,valid_rows=M-row<BM?M-row:BM;
const int begin=split*P;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::TOTAL_SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::TOTAL_SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
auto issue_scales=[&](auto si,int kt) {
constexpr int slot=decltype(si)::value;
const int base=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
static_for<T::SA_COPIES>([&](auto ii) {
const int index=decltype(ii)::value*256+tid;
const int sr=index%(BM/4)*4,scol=index/(BM/4);
const int sc=scol<T::SA_COLS?scol:T::SA_COLS-1;
const int source=row+sr<M?(kt*2+sc)*M+row+sr:M*T::TOTAL_SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SA+(decltype(ii)::value*256+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
static_for<T::SB_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SB+(i*256+wave*64)*4));
if constexpr(T::SB_SLOTS) {
const int scol=(i*256+tid)%T::SA_COLS;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,(col/128)*T::TOTAL_SCALES+kt*2+scol,0,0,0);
} else {
const int dword=(i*256+tid)%(T::SA_COLS/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,(col/128)*T::TOTAL_SCALES+kt*2+dword*4,0,0,0);
}
});
};
auto issue=[&](auto si,int kt) {
constexpr int slot=decltype(si)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*256+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,kt*256,0,0);
});
static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+T::A_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,i*16*T::K+tid*16,kt*4096,0,T::B_CACHE);
});
if constexpr(!T::SCALE_PRELOAD)issue_scales(si,kt);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::SCALE_PRELOAD)issue_scales(0_I,begin);
static_for<NB-1>([&](auto si){issue(si,begin+decltype(si)::value);});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%NB,pending=P-t-1<NB-2?P-t-1:NB-2;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::ISSUES*pending>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<array<unsigned,NR>,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
const int sbase=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
const int scale_col=T::SCALE_PRELOAD?t*2+ks:ks;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=wm*48+m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,sbase+T::OFF_SA+scale_col*BM+ar)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/2)+n*16;
const int bo=slot*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
bs[ks][n]=load<1>(ss,sbase+T::OFF_SB+scale_col*(T::SB_SLOTS?4:1))[0];
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+NB-1<P)issue(number<(slot+NB-1)%NB>{},begin+t+NB-1);
__builtin_amdgcn_sched_barrier(0);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks][n]),0_I,0_I);
});
});
});
};
static_for<P>([&](auto ti){step(ti);});
const int plane=valid_rows*BN;
const int64_t tile_base=(int64_t(row)*N+int64_t(ntile)*plane)*SK;
using av_u32x4 = unsigned __attribute__((ext_vector_type(4)));
float* partial=partials+tile_base+int64_t(split)*plane;
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)
__builtin_amdgcn_av_store_b128(reinterpret_cast<av_u32x4*>(partial+rr*BN+nn),
__builtin_bit_cast(av_u32x4,c[ci]),__MEMORY_SCOPE_DEVICE);
});
auto* flag=reinterpret_cast<OPUS_LDS_ADDR unsigned*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::FLAG_OFF));
s_waitcnt_vmcnt(0_I);
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
if(tid==0) {
const int old=__scoped_atomic_fetch_add(counters+tile,1,__ATOMIC_ACQ_REL,__MEMORY_SCOPE_DEVICE);
*flag=old==SK-1;
}
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
const unsigned last=*flag;
s_waitcnt_lgkmcnt(0_I);
if(last) {
array<vector_t<float,4>,MR*NR> sums{};
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows) {
static_for<SK>([&](auto ski) {
float* src=partials+tile_base+int64_t(decltype(ski)::value)*plane+rr*BN+nn;
sums[ci]+=__builtin_bit_cast(vector_t<float,4>,
__builtin_amdgcn_av_load_b128(reinterpret_cast<av_u32x4*>(src),__MEMORY_SCOPE_DEVICE));
});
}
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)store<4>(sc,cast<bf16_t>(sums[ci]),rr*(BN+8)+nn);
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+255)/256>([&](auto ii) {
const int index=decltype(ii)::value*256+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn,0,number<2>{});
});
s_waitcnt_vmcnt(0_I);
__builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
if(tid==0) {
[[clang::amdgpu_av("none")]] __scoped_atomic_store_n(counters+tile,0,__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
s_waitcnt_vmcnt(0_I);
}
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_67) && !defined(OPUS_BP_INCLUDE_OVERLAP_67_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_67_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_67 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_66;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_B0_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_B0_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_68) && !defined(OPUS_BP_INCLUDE_OVERLAP_68_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_68_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_68 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_66;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_B0_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_B0_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_69) && !defined(OPUS_BP_INCLUDE_OVERLAP_69_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_69_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_69 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=0;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=BN*256,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
template<class T> __global__ void mediumK256_AV_loop_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=0;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=BN*256,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void mediumK256_AV_loop_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(256,1)
void mediumK256_AV_loop_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,float* partials,int* counters) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,SK=T::GLOBAL_SPLIT,P=T::PANELS,MR=T::MR,NR=T::NR;
const int M=args.m,N=args.n,tid=thread_id_x(),lane=tid%64;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wm=wave/2,wn=wave%2,kg=lane/16,lane16=lane%16;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=pid/(8*SK)*8+pid%8,split=pid/8%SK;
if(tile>=total)return;
const unsigned mtile=tile%gm,ntile=tile/gm;
const int row=mtile*BM,col=ntile*BN,valid_rows=M-row<BM?M-row:BM;
const int begin=split*P;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::TOTAL_SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::TOTAL_SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
auto issue_scales=[&](auto si,int kt) {
const int slot=si;
const int base=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
static_for<T::SA_COPIES>([&](auto ii) {
const int index=decltype(ii)::value*256+tid;
const int sr=index%(BM/4)*4,scol=index/(BM/4);
const int sc=scol<T::SA_COLS?scol:T::SA_COLS-1;
const int source=row+sr<M?(kt*2+sc)*M+row+sr:M*T::TOTAL_SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SA+(decltype(ii)::value*256+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
static_for<T::SB_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SB+(i*256+wave*64)*4));
if constexpr(T::SB_SLOTS) {
const int scol=(i*256+tid)%T::SA_COLS;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,(col/128)*T::TOTAL_SCALES+kt*2+scol,0,0,0);
} else {
const int dword=(i*256+tid)%(T::SA_COLS/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,(col/128)*T::TOTAL_SCALES+kt*2+dword*4,0,0,0);
}
});
};
auto issue=[&](auto si,int kt) {
const int slot=si;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*256+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,kt*256,0,0);
});
static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+T::A_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,i*16*T::K+tid*16,kt*4096,0,T::B_CACHE);
});
if constexpr(!T::SCALE_PRELOAD)issue_scales(si,kt);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::SCALE_PRELOAD)issue_scales(0_I,begin);
static_for<NB-1>([&](auto si){issue(si,begin+decltype(si)::value);});
auto step=[&](int t,int slot,auto pi,auto fi) {
constexpr int pending=decltype(pi)::value;
constexpr bool future=decltype(fi)::value;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::ISSUES*pending>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<array<unsigned,NR>,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
const int sbase=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
const int scale_col=T::SCALE_PRELOAD?t*2+ks:ks;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=wm*48+m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,sbase+T::OFF_SA+scale_col*BM+ar)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/2)+n*16;
const int bo=slot*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
bs[ks][n]=load<1>(ss,sbase+T::OFF_SB+scale_col*(T::SB_SLOTS?4:1))[0];
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(future)issue(slot==0?NB-1:slot-1,begin+t+NB-1);
__builtin_amdgcn_sched_barrier(0);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks][n]),0_I,0_I);
});
});
});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int STEADY=P-(NB-1);
int ring_slot=0;
#pragma clang loop unroll(disable)
for(int t=0;t<STEADY;++t) {
step(t,ring_slot,number<NB-2>{},number<true>{});
ring_slot=ring_slot+1==NB?0:ring_slot+1;
}
static_for<NB-1>([&](auto ti) {
constexpr int t=STEADY+decltype(ti)::value;
constexpr int pending=P-t-1;
step(t,t%NB,number<pending>{},number<false>{});
});
const int plane=valid_rows*BN;
const int64_t tile_base=(int64_t(row)*N+int64_t(ntile)*plane)*SK;
using av_u32x4 = unsigned __attribute__((ext_vector_type(4)));
float* partial=partials+tile_base+int64_t(split)*plane;
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)
__builtin_amdgcn_av_store_b128(reinterpret_cast<av_u32x4*>(partial+rr*BN+nn),
__builtin_bit_cast(av_u32x4,c[ci]),__MEMORY_SCOPE_DEVICE);
});
auto* flag=reinterpret_cast<OPUS_LDS_ADDR unsigned*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::FLAG_OFF));
s_waitcnt_vmcnt(0_I);
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
if(tid==0) {
const int old=__scoped_atomic_fetch_add(counters+tile,1,__ATOMIC_ACQ_REL,__MEMORY_SCOPE_DEVICE);
*flag=old==SK-1;
}
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
const unsigned last=*flag;
s_waitcnt_lgkmcnt(0_I);
if(last) {
array<vector_t<float,4>,MR*NR> sums{};
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows) {
static_for<SK>([&](auto ski) {
float* src=partials+tile_base+int64_t(decltype(ski)::value)*plane+rr*BN+nn;
sums[ci]+=__builtin_bit_cast(vector_t<float,4>,
__builtin_amdgcn_av_load_b128(reinterpret_cast<av_u32x4*>(src),__MEMORY_SCOPE_DEVICE));
});
}
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)store<4>(sc,cast<bf16_t>(sums[ci]),rr*(BN+8)+nn);
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+255)/256>([&](auto ii) {
const int index=decltype(ii)::value*256+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn,0,number<2>{});
});
s_waitcnt_vmcnt(0_I);
__builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
if(tid==0) {
[[clang::amdgpu_av("none")]] __scoped_atomic_store_n(counters+tile,0,__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
s_waitcnt_vmcnt(0_I);
}
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_70) && !defined(OPUS_BP_INCLUDE_OVERLAP_70_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_70_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_70 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_69;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_loop_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_loop_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_71) && !defined(OPUS_BP_INCLUDE_OVERLAP_71_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_71_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_71 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_69;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_loop_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
mediumK256_AV_loop_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_72) && !defined(OPUS_BP_INCLUDE_OVERLAP_72_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_72_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_72 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=0;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=0,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static constexpr int B_AHEAD=1,B_SETS=2,B_ISSUES=4*NR;
static constexpr int initial_issues=(SCALE_PRELOAD?SA_COPIES+SB_COPIES:0)+B_ISSUES+(NB-1)*ISSUES;
static constexpr int issued_before(int t) {
return initial_issues+(t<PANELS-B_AHEAD?t:PANELS-B_AHEAD)*B_ISSUES+
(t<PANELS-(NB-1)?t:PANELS-(NB-1))*ISSUES;
}
static constexpr int a_end(int t) {
return t<NB-1?(SCALE_PRELOAD?SA_COPIES+SB_COPIES:0)+B_ISSUES+(t+1)*ISSUES:
issued_before(t-(NB-1))+B_ISSUES+ISSUES;
}
static constexpr int b_end(int t) {return t==0?B_ISSUES:issued_before(t-1)+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)+(t+1<PANELS?B_ISSUES:0)+
(t+NB-1<PANELS?ISSUES:0)-b_end(t);}
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
template<class T> __global__ void medium_K256_BdirectAV_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BN,int NB,int SK>
struct mediumK256_traits {
static constexpr int B_M=96,B_N=BN,B_K=256,T_M=2,T_N=2,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=3,NR=BN/32,STAGES=NB,GLOBAL_SPLIT=SK,B_CACHE=0;
static constexpr int K=7168,TOTAL_SCALES=56,PANELS=28/SK,SCALE_COLS=56/SK;
static constexpr bool SCALE_PRELOAD=BN==64;
static constexpr int A_BYTES=B_M*256,B_BYTES=0,A_COPIES=A_BYTES/4096,B_COPIES=B_BYTES/4096;
static constexpr int SA_COLS=SCALE_PRELOAD ? SCALE_COLS:2;
static constexpr int SA_BYTES=((B_M*SA_COLS+1023)/1024)*1024;
static constexpr bool SB_SLOTS=SA_COLS%4!=0;
static constexpr int SB_BYTES=((SA_COLS*(SB_SLOTS?4:1)+1023)/1024)*1024;
static constexpr int STAGE_BYTES=A_BYTES+B_BYTES+(SCALE_PRELOAD?0:SA_BYTES+SB_BYTES);
static constexpr int OFF_SA=SCALE_PRELOAD?NB*STAGE_BYTES:A_BYTES+B_BYTES;
static constexpr int OFF_SB=OFF_SA+SA_BYTES;
static constexpr int SA_COPIES=SA_BYTES/1024,SB_COPIES=SB_BYTES/1024;
static constexpr int ISSUES=A_COPIES+B_COPIES+(SCALE_PRELOAD?0:SA_COPIES+SB_COPIES);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES+(SCALE_PRELOAD?SA_BYTES+SB_BYTES:0);
static constexpr int FLAG_OFF=PIPELINE_BYTES,LDS_BYTES=FLAG_OFF+16;
static_assert(BN==32 || BN==64);
static_assert(NB==3 || NB==4);
static_assert(SK==2 || SK==4);
static constexpr int B_AHEAD=1,B_SETS=2,B_ISSUES=4*NR;
static constexpr int initial_issues=(SCALE_PRELOAD?SA_COPIES+SB_COPIES:0)+B_ISSUES+(NB-1)*ISSUES;
static constexpr int issued_before(int t) {
return initial_issues+(t<PANELS-B_AHEAD?t:PANELS-B_AHEAD)*B_ISSUES+
(t<PANELS-(NB-1)?t:PANELS-(NB-1))*ISSUES;
}
static constexpr int a_end(int t) {
return t<NB-1?(SCALE_PRELOAD?SA_COPIES+SB_COPIES:0)+B_ISSUES+(t+1)*ISSUES:
issued_before(t-(NB-1))+B_ISSUES+ISSUES;
}
static constexpr int b_end(int t) {return t==0?B_ISSUES:issued_before(t-1)+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)+(t+1<PANELS?B_ISSUES:0)+
(t+NB-1<PANELS?ISSUES:0)-b_end(t);}
static_assert(PANELS>=NB-1 && 7168%(256*SK)==0);
static_assert(LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void medium_K256_BdirectAV_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950,float*,int*) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(256,1)
void medium_K256_BdirectAV_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,float* partials,int* counters) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,SK=T::GLOBAL_SPLIT,P=T::PANELS,MR=T::MR,NR=T::NR;
const int M=args.m,N=args.n,tid=thread_id_x(),lane=tid%64;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wm=wave/2,wn=wave%2,kg=lane/16,lane16=lane%16;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=pid/(8*SK)*8+pid%8,split=pid/8%SK;
if(tile>=total)return;
const unsigned mtile=tile%gm,ntile=tile/gm;
const int row=mtile*BM,col=ntile*BN,valid_rows=M-row<BM?M-row:BM;
const int begin=split*P;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::TOTAL_SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::TOTAL_SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
auto issue_scales=[&](auto si,int kt) {
constexpr int slot=decltype(si)::value;
const int base=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
static_for<T::SA_COPIES>([&](auto ii) {
const int index=decltype(ii)::value*256+tid;
const int sr=index%(BM/4)*4,scol=index/(BM/4);
const int sc=scol<T::SA_COLS?scol:T::SA_COLS-1;
const int source=row+sr<M?(kt*2+sc)*M+row+sr:M*T::TOTAL_SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SA+(decltype(ii)::value*256+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
static_for<T::SB_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+base+T::OFF_SB+(i*256+wave*64)*4));
if constexpr(T::SB_SLOTS) {
const int scol=(i*256+tid)%T::SA_COLS;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,(col/128)*T::TOTAL_SCALES+kt*2+scol,0,0,0);
} else {
const int dword=(i*256+tid)%(T::SA_COLS/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,(col/128)*T::TOTAL_SCALES+kt*2+dword*4,0,0,0);
}
});
};
auto issue=[&](auto si,int kt) {
constexpr int slot=decltype(si)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*256+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,kt*256,0,0);
});
static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+T::A_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,i*16*T::K+tid*16,kt*4096,0,T::B_CACHE);
});
if constexpr(!T::SCALE_PRELOAD)issue_scales(si,kt);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
array<array<array<typename Base::vtype_b,NR>,2>,T::B_SETS> bregs;
auto issue_b=[&](auto bi,int kt) {
constexpr int set=decltype(bi)::value;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/2)+n*16;
const int offset=ks*2048+kg*256+lane16*16,soffset=br*T::K+kt*4096;
set_slice(bregs[set][ks][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][ks][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
issue_b(0_I,begin);
if constexpr(T::SCALE_PRELOAD)issue_scales(0_I,begin);
static_for<NB-1>([&](auto si){issue(si,begin+decltype(si)::value);});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%NB,pending=P-t-1<NB-2?P-t-1:NB-2;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(t+1<P)issue_b(number<(t+1)%T::B_SETS>{},begin+t+1);
array<array<typename Base::vtype_a,MR>,2> af;
array<array<unsigned,MR>,2> as;
array<array<unsigned,NR>,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
const int sbase=T::SCALE_PRELOAD?0:slot*T::STAGE_BYTES;
const int scale_col=T::SCALE_PRELOAD?t*2+ks:ks;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=wm*48+m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=slot*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,sbase+T::OFF_SA+scale_col*BM+ar)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/2)+n*16;
bs[ks][n]=load<1>(ss,sbase+T::OFF_SB+scale_col*(T::SB_SLOTS?4:1))[0];
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+NB-1<P)issue(number<(slot+NB-1)%NB>{},begin+t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(af[ks][m],bregs[t%T::B_SETS][ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks][n]),0_I,0_I);
});
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<P>([&](auto ti){step(ti);});
const int plane=valid_rows*BN;
const int64_t tile_base=(int64_t(row)*N+int64_t(ntile)*plane)*SK;
using av_u32x4 = unsigned __attribute__((ext_vector_type(4)));
float* partial=partials+tile_base+int64_t(split)*plane;
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)
__builtin_amdgcn_av_store_b128(reinterpret_cast<av_u32x4*>(partial+rr*BN+nn),
__builtin_bit_cast(av_u32x4,c[ci]),__MEMORY_SCOPE_DEVICE);
});
auto* flag=reinterpret_cast<OPUS_LDS_ADDR unsigned*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::FLAG_OFF));
s_waitcnt_vmcnt(0_I);
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
if(tid==0) {
const int old=__scoped_atomic_fetch_add(counters+tile,1,__ATOMIC_ACQ_REL,__MEMORY_SCOPE_DEVICE);
*flag=old==SK-1;
}
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
const unsigned last=*flag;
s_waitcnt_lgkmcnt(0_I);
if(last) {
array<vector_t<float,4>,MR*NR> sums{};
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows) {
static_for<SK>([&](auto ski) {
float* src=partials+tile_base+int64_t(decltype(ski)::value)*plane+rr*BN+nn;
sums[ci]+=__builtin_bit_cast(vector_t<float,4>,
__builtin_amdgcn_av_load_b128(reinterpret_cast<av_u32x4*>(src),__MEMORY_SCOPE_DEVICE));
});
}
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
const int rr=wm*48+(ci/NR)*16+lane16,nn=wn*(BN/2)+(ci%NR)*16+kg*4;
if(rr<valid_rows)store<4>(sc,cast<bf16_t>(sums[ci]),rr*(BN+8)+nn);
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+255)/256>([&](auto ii) {
const int index=decltype(ii)::value*256+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn,0,number<2>{});
});
s_waitcnt_vmcnt(0_I);
__builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
__builtin_amdgcn_s_barrier();
[[clang::amdgpu_av("none")]] __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__MEMORY_SCOPE_DEVICE);
if(tid==0) {
[[clang::amdgpu_av("none")]] __scoped_atomic_store_n(counters+tile,0,__ATOMIC_RELEASE,__MEMORY_SCOPE_DEVICE);
s_waitcnt_vmcnt(0_I);
}
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_73) && !defined(OPUS_BP_INCLUDE_OVERLAP_73_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_73_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_73 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_72;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
medium_K256_BdirectAV_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<32,4,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
medium_K256_BdirectAV_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_74) && !defined(OPUS_BP_INCLUDE_OVERLAP_74_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_74_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_74 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_72;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
medium_K256_BdirectAV_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=mediumK256_traits<64,3,4>;
int launch_config(
int selector,int launchSplit,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* stream) {
if(selector!=0 || launchSplit!=0 || k!=7168 || !((m==80 && n==2048 && Traits::B_N==32) ||
((m==160 && n==2048)||(m==352 && n==768)) && Traits::B_N==64))return -11;
FixedMContract contract{m,n,k,Traits::GLOBAL_SPLIT,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);if(checked)return checked;
const uint64_t flags_offset=(uint64_t(Traits::GLOBAL_SPLIT)*m*n*4+15)&~uint64_t(15);
const uint64_t flag_count=uint64_t((m+95)/96)*(n/Traits::B_N);
const uint64_t required=flags_offset+((flag_count*4+15)&~uint64_t(15));
if(bytes!=required)return -12;
const void* inputs[]={a,b,out,sfa,sfb};
const uint64_t extents[]={uint64_t(m)*k,uint64_t(n)*k,uint64_t(m)*n*2,uint64_t(m)*(k/128),uint64_t(n/128)*(k/128)};
for(int i=0;i<5;++i)if(fixed_m_overlap(ws,required,inputs[i],extents[i]))return -13;
contract.split=1;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
auto partials=reinterpret_cast<float*>(ws);
auto counters=reinterpret_cast<int*>(reinterpret_cast<char*>(ws)+flags_offset);
const unsigned grid=(flag_count+7)/8*8*Traits::GLOBAL_SPLIT;
medium_K256_BdirectAV_kernel<Traits><<<dim3(grid,1,1),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args,partials,counters);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_75) && !defined(OPUS_BP_INCLUDE_OVERLAP_75_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_75_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_75 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC>
struct shortK_K256_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=3,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int SFB_BYTES=BLOCK_SIZE*4,OFF_SFA=STAGES*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)+(BD && t==0?B_ISSUES:0)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
template<class T> __global__ void shortK_K256_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC>
struct shortK_K256_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=3,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int SFB_BYTES=BLOCK_SIZE*4,OFF_SFA=STAGES*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)+(BD && t==0?B_ISSUES:0)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void shortK_K256_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_K256_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,MR=T::MR,NR=T::NR;
const int M=args.m,N=args.n,tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64);
const unsigned pid=block_id_x(),gn=N/BN;
const int row=(pid/gn)*BM,col=(pid%gn)*BN,valid_rows=M-row<BM?M-row:BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<array<typename Base::vtype_b,NR>,2>,T::B_SETS> bregs;
auto issue_b=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int off=ks*2048+kg*256+lane16*16,soff=br*T::K+t*4096;
set_slice(bregs[t][ks][n],load<16>(gb,off,soff,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[t][ks][n],load<16>(gb,off+1024,soff,number<T::B_CACHE>{}),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::B_DIRECT)static_for<2>([&](auto ti){issue_b(ti);});
static_for<T::SFA_PASSES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,sr=index%(BM/16)*16,scol=index/(BM/16);
const int sc=scol<T::SCALES?scol:T::SCALES-1;
const int source=row+sr<M?sc*M+row+sr:M*T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFA+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
{
const int sc=tid%T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFB+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,(col/128)*T::SCALES+sc,0,0,0);
}
auto issue=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,t*256,0,0);
});
if constexpr(!T::B_DIRECT)static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,group=lin/4096,within=lin%4096;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+T::A_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,group*(T::K*16)+within,t*4096,0,T::B_CACHE);
});
__builtin_amdgcn_asyncmark();__builtin_amdgcn_sched_barrier(0);
};
static_for<T::PANELS>([&](auto ti){issue(ti);});
static_for<T::PANELS>([&](auto ti) {
constexpr int t=decltype(ti)::value;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(T::PANELS-t-1);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::B_DIRECT && t==0)issue_b(2_I);
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<unsigned,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
bs[ks]=load<1>(ss,T::OFF_SFB+(t*2+ks)*4)[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=t*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=t*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,T::OFF_SFA+(t*2+ks)*BM+ar)[0];
});
if constexpr(!T::B_DIRECT)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int bo=t*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(T::B_DIRECT)s_waitcnt_vmcnt(number<T::wait_b(t)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::B_DIRECT)c[ci]=Base{}(af[ks][m],bregs[t][ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
else c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
});
});
});
__builtin_amdgcn_sched_barrier(0);
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto ci) {
constexpr int i=decltype(ci)::value;
const auto value=cast<bf16_t>(c[i]);
const int rr=(i/NR)*16+kg*4,nn=wave*(BN/T::T_N)+(i%NR)*16+lane16;
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
if(rr+e<valid_rows)store<1>(sc,vector_t<bf16_t,1>{value[e]},(rr+e)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto ii) {
const int index=decltype(ii)::value*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn);
});
s_waitcnt_vmcnt(0_I);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_76) && !defined(OPUS_BP_INCLUDE_OVERLAP_76_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_76_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_76 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_75;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_traits<48,32,2,false,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || !(m==48) || n!=7168 || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_K256_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_traits<48,32,2,false,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || !(m==48) || n!=7168 || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_K256_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_77) && !defined(OPUS_BP_INCLUDE_OVERLAP_77_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_77_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_77 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_75;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_traits<64,32,2,true,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || !(m==64) || n!=7168 || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_K256_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_traits<64,32,2,true,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || !(m==64) || n!=7168 || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_K256_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_78) && !defined(OPUS_BP_INCLUDE_OVERLAP_78_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_78_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_78 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_75;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_traits<64,64,4,true,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || !(m==112 || m==128) || n!=7168 || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_K256_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_traits<64,64,4,true,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || !(m==112 || m==128) || n!=7168 || k!=Traits::K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_K256_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_79) && !defined(OPUS_BP_INCLUDE_OVERLAP_79_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_79_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_79 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC,int FM>
struct shortK_K256_fixedMN_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int M=FM,N=7168;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=3,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int SFB_BYTES=BLOCK_SIZE*4,OFF_SFA=STAGES*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)+(BD && t==0?B_ISSUES:0)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
template<class T> __global__ void shortK_K256_fixedMN_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC,int FM>
struct shortK_K256_fixedMN_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int M=FM,N=7168;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=3,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int SFB_BYTES=BLOCK_SIZE*4,OFF_SFA=STAGES*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)+(BD && t==0?B_ISSUES:0)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void shortK_K256_fixedMN_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_K256_fixedMN_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,MR=T::MR,NR=T::NR;
constexpr int M=T::M,N=T::N;
const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64);
const unsigned pid=block_id_x(),gn=N/BN;
const int row=M<=BM?0:(pid/gn)*BM,col=M<=BM?pid*BN:(pid%gn)*BN;
const int valid_rows=M<=BM?M:(M-row<BM?M-row:BM);
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<array<typename Base::vtype_b,NR>,2>,T::B_SETS> bregs;
auto issue_b=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int off=ks*2048+kg*256+lane16*16,soff=br*T::K+t*4096;
set_slice(bregs[t][ks][n],load<16>(gb,off,soff,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[t][ks][n],load<16>(gb,off+1024,soff,number<T::B_CACHE>{}),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::B_DIRECT)static_for<2>([&](auto ti){issue_b(ti);});
static_for<T::SFA_PASSES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,sr=index%(BM/16)*16,scol=index/(BM/16);
const int sc=scol<T::SCALES?scol:T::SCALES-1;
const int source=row+sr<M?sc*M+row+sr:M*T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFA+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
{
const int sc=tid%T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFB+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,(col/128)*T::SCALES+sc,0,0,0);
}
auto issue=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,t*256,0,0);
});
if constexpr(!T::B_DIRECT)static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,group=lin/4096,within=lin%4096;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+T::A_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,group*(T::K*16)+within,t*4096,0,T::B_CACHE);
});
__builtin_amdgcn_asyncmark();__builtin_amdgcn_sched_barrier(0);
};
static_for<T::PANELS>([&](auto ti){issue(ti);});
static_for<T::PANELS>([&](auto ti) {
constexpr int t=decltype(ti)::value;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(T::PANELS-t-1);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::B_DIRECT && t==0)issue_b(2_I);
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<unsigned,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
bs[ks]=load<1>(ss,T::OFF_SFB+(t*2+ks)*4)[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=t*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=t*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,T::OFF_SFA+(t*2+ks)*BM+ar)[0];
});
if constexpr(!T::B_DIRECT)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int bo=t*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(T::B_DIRECT)s_waitcnt_vmcnt(number<T::wait_b(t)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::B_DIRECT)c[ci]=Base{}(af[ks][m],bregs[t][ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
else c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
});
});
});
__builtin_amdgcn_sched_barrier(0);
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto ci) {
constexpr int i=decltype(ci)::value;
const auto value=cast<bf16_t>(c[i]);
const int rr=(i/NR)*16+kg*4,nn=wave*(BN/T::T_N)+(i%NR)*16+lane16;
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
if(rr+e<valid_rows)store<1>(sc,vector_t<bf16_t,1>{value[e]},(rr+e)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto ii) {
const int index=decltype(ii)::value*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn);
});
s_waitcnt_vmcnt(0_I);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_80) && !defined(OPUS_BP_INCLUDE_OVERLAP_80_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_80_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_80 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_79;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<48,32,2,false,2,48>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
shortK_K256_fixedMN_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<48,32,2,false,2,48>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
shortK_K256_fixedMN_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_81) && !defined(OPUS_BP_INCLUDE_OVERLAP_81_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_81_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_81 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_79;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,32,2,true,2,64>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
shortK_K256_fixedMN_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,32,2,true,2,64>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
shortK_K256_fixedMN_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_82) && !defined(OPUS_BP_INCLUDE_OVERLAP_82_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_82_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_82 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_79;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,64,4,true,0,112>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
shortK_K256_fixedMN_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,64,4,true,0,112>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
shortK_K256_fixedMN_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_83) && !defined(OPUS_BP_INCLUDE_OVERLAP_83_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_83_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_83 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_79;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,64,4,true,0,128>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
shortK_K256_fixedMN_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,64,4,true,0,128>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
shortK_K256_fixedMN_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_84) && !defined(OPUS_BP_INCLUDE_OVERLAP_84_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_84_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_84 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=64*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==3 || Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void actual_short_058_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=64*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==3 || Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void actual_short_058_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void actual_short_058_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if(wave==0) {
constexpr int WORDS=T::LOOPS/4;
const int source=lane<T::B_GROUPS*WORDS ? (col/128+lane/WORDS)*LOOPS+(lane%WORDS)*4:(N/128)*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(sfb.ptr));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,source,0,0,0);
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,scale_group*LOOPS+t)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_85) && !defined(OPUS_BP_INCLUDE_OVERLAP_85_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_85_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_85 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_84;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,96,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=58 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
actual_short_058_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,96,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=58 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
actual_short_058_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_86) && !defined(OPUS_BP_INCLUDE_OVERLAP_86_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_86_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_86 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=64*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==3 || Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void actual_short_059_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=64*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==3 || Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void actual_short_059_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void actual_short_059_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if(wave==0) {
constexpr int WORDS=T::LOOPS/4;
const int source=lane<T::B_GROUPS*WORDS ? (col/128+lane/WORDS)*LOOPS+(lane%WORDS)*4:(N/128)*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(sfb.ptr));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,4,source,0,0,0);
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,scale_group*LOOPS+t)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_87) && !defined(OPUS_BP_INCLUDE_OVERLAP_87_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_87_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_87 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_86;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,80,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=59 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
actual_short_059_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,80,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=59 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
actual_short_059_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_88) && !defined(OPUS_BP_INCLUDE_OVERLAP_88_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_88_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_88 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC,int FM>
struct shortK_K256_fixedMN_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int M=FM,N=7168;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=2,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int SFB_BYTES=64*4,OFF_SFA=PANELS*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
template<class T> __global__ void actual_short_118_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC,int FM>
struct shortK_K256_fixedMN_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int M=FM,N=7168;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=2,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int SFB_BYTES=64*4,OFF_SFA=PANELS*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void actual_short_118_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(T::BLOCK_SIZE,1)
void actual_short_118_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,MR=T::MR,NR=T::NR;
constexpr int M=T::M,N=T::N;
const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64);
const unsigned pid=block_id_x(),gn=N/BN;
const int row=M<=BM?0:(pid/gn)*BM,col=M<=BM?pid*BN:(pid%gn)*BN;
const int valid_rows=M<=BM?M:(M-row<BM?M-row:BM);
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<array<typename Base::vtype_b,NR>,2>,T::B_SETS> bregs;
auto issue_b=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int off=ks*2048+kg*256+lane16*16,soff=br*T::K+t*4096;
set_slice(bregs[t%T::B_SETS][ks][n],load<16>(gb,off,soff,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[t%T::B_SETS][ks][n],load<16>(gb,off+1024,soff,number<T::B_CACHE>{}),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::B_DIRECT)static_for<2>([&](auto ti){issue_b(ti);});
static_for<T::SFA_PASSES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*T::BLOCK_SIZE+tid)*4,sr=index%BM,sc=index/BM;
const int source=index<BM*T::SCALES && sr<valid_rows?sc*M+row+sr:M*T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFA+(i*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if(wave==0) {
const int source=lane<T::SCALES?(col/128)*T::SCALES+lane:(N/128)*T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFB));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,source,0,0,0);
}
auto issue=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,t*256,0,0);
});
if constexpr(!T::B_DIRECT)static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,group=lin/4096,within=lin%4096;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+T::A_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,group*(T::K*16)+within,t*4096,0,T::B_CACHE);
});
__builtin_amdgcn_asyncmark();__builtin_amdgcn_sched_barrier(0);
};
static_for<T::PANELS>([&](auto ti){issue(ti);});
static_for<T::PANELS>([&](auto ti) {
constexpr int t=decltype(ti)::value;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(T::PANELS-t-1);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<unsigned,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
bs[ks]=load<1>(ss,T::OFF_SFB+(t*2+ks)*4)[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=t*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=t*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,T::OFF_SFA+(t*2+ks)*BM+ar)[0];
});
if constexpr(!T::B_DIRECT)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int bo=t*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(T::B_DIRECT)s_waitcnt_vmcnt(number<(t<2?T::wait_b(t)-1:T::wait_b(t))>{});
s_waitcnt_lgkmcnt(0_I);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::B_DIRECT)c[ci]=Base{}(af[ks][m],bregs[t%T::B_SETS][ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
else c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
});
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(T::B_DIRECT && t==0)issue_b(2_I);
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto ci) {
constexpr int i=decltype(ci)::value;
const auto value=cast<bf16_t>(c[i]);
const int rr=(i/NR)*16+kg*4,nn=wave*(BN/T::T_N)+(i%NR)*16+lane16;
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
if(rr+e<valid_rows)store<1>(sc,vector_t<bf16_t,1>{value[e]},(rr+e)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto ii) {
const int index=decltype(ii)::value*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn);
});
s_waitcnt_vmcnt(0_I);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_89) && !defined(OPUS_BP_INCLUDE_OVERLAP_89_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_89_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_89 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_88;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,64,4,true,0,112>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=118 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
actual_short_118_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,64,4,true,0,112>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=118 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
actual_short_118_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_90) && !defined(OPUS_BP_INCLUDE_OVERLAP_90_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_90_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_90 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC,int FM>
struct shortK_K256_fixedMN_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int M=FM,N=7168;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=2,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int SFB_BYTES=64*4,OFF_SFA=PANELS*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
template<class T> __global__ void actual_short_138_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC,int FM>
struct shortK_K256_fixedMN_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int M=FM,N=7168;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=2,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int SFB_BYTES=64*4,OFF_SFA=PANELS*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void actual_short_138_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(T::BLOCK_SIZE,1)
void actual_short_138_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,MR=T::MR,NR=T::NR;
constexpr int M=T::M,N=T::N;
const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64);
const unsigned pid=block_id_x(),gn=N/BN;
const int row=M<=BM?0:(pid/gn)*BM,col=M<=BM?pid*BN:(pid%gn)*BN;
const int valid_rows=M<=BM?M:(M-row<BM?M-row:BM);
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<array<typename Base::vtype_b,NR>,2>,T::B_SETS> bregs;
auto issue_b=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int off=ks*2048+kg*256+lane16*16,soff=br*T::K+t*4096;
set_slice(bregs[t%T::B_SETS][ks][n],load<16>(gb,off,soff,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[t%T::B_SETS][ks][n],load<16>(gb,off+1024,soff,number<T::B_CACHE>{}),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::B_DIRECT)static_for<2>([&](auto ti){issue_b(ti);});
static_for<T::SFA_PASSES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*T::BLOCK_SIZE+tid)*4,sr=index%BM,sc=index/BM;
const int source=index<BM*T::SCALES && sr<valid_rows?sc*M+row+sr:M*T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFA+(i*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if(wave==0) {
const int source=lane<T::SCALES?(col/128)*T::SCALES+lane:(N/128)*T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFB));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,source,0,0,0);
}
auto issue=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,t*256,0,0);
});
if constexpr(!T::B_DIRECT)static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,group=lin/4096,within=lin%4096;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+T::A_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,group*(T::K*16)+within,t*4096,0,T::B_CACHE);
});
__builtin_amdgcn_asyncmark();__builtin_amdgcn_sched_barrier(0);
};
static_for<T::PANELS>([&](auto ti){issue(ti);});
static_for<T::PANELS>([&](auto ti) {
constexpr int t=decltype(ti)::value;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(T::PANELS-t-1);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<unsigned,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
bs[ks]=load<1>(ss,T::OFF_SFB+(t*2+ks)*4)[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=t*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=t*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,T::OFF_SFA+(t*2+ks)*BM+ar)[0];
});
if constexpr(!T::B_DIRECT)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int bo=t*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(T::B_DIRECT)s_waitcnt_vmcnt(number<(t<2?T::wait_b(t)-1:T::wait_b(t))>{});
s_waitcnt_lgkmcnt(0_I);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::B_DIRECT)c[ci]=Base{}(af[ks][m],bregs[t%T::B_SETS][ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
else c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
});
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(T::B_DIRECT && t==0)issue_b(2_I);
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto ci) {
constexpr int i=decltype(ci)::value;
const auto value=cast<bf16_t>(c[i]);
const int rr=(i/NR)*16+kg*4,nn=wave*(BN/T::T_N)+(i%NR)*16+lane16;
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
if(rr+e<valid_rows)store<1>(sc,vector_t<bf16_t,1>{value[e]},(rr+e)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto ii) {
const int index=decltype(ii)::value*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn);
});
s_waitcnt_vmcnt(0_I);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_91) && !defined(OPUS_BP_INCLUDE_OVERLAP_91_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_91_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_91 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_90;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,32,2,true,2,64>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=138 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
actual_short_138_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<64,32,2,true,2,64>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=138 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
actual_short_138_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_92) && !defined(OPUS_BP_INCLUDE_OVERLAP_92_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_92_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_92 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_7;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC,int FM>
struct shortK_K256_fixedMN_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int M=FM,N=7168;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=2,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int SFB_BYTES=64*4,OFF_SFA=PANELS*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
template<class T> __global__ void actual_short_139_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int WN,bool BD,int BC,int FM>
struct shortK_K256_fixedMN_traits {
static constexpr int B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=WN,T_K=1,BLOCK_SIZE=WN*64;
static constexpr int M=FM,N=7168;
static constexpr int MR=BM/16,NR=BN/(WN*16),K=768,PANELS=3,SCALES=6,STAGES=4;
static constexpr bool B_DIRECT=BD;
static constexpr int B_CACHE=BC,B_AHEAD=2,B_SETS=2,B_ISSUES=4*NR;
static constexpr int A_BYTES=BM*256,B_BYTES=BD?0:BN*256,STAGE_BYTES=A_BYTES+B_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int ISSUES=A_COPIES+B_COPIES;
static constexpr int SFA_PASSES=(BM*SCALES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4),SFA_BYTES=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int SFB_BYTES=64*4,OFF_SFA=PANELS*STAGE_BYTES,OFF_SFB=OFF_SFA+SFA_BYTES;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_BYTES,OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int SCALE_ISSUES=SFA_PASSES+1;
static constexpr int INITIAL_ISSUES=(BD?2*B_ISSUES:0)+SCALE_ISSUES+PANELS*ISSUES;
static constexpr int issued_before(int t) {return INITIAL_ISSUES+(BD && t>0?B_ISSUES:0);}
static constexpr int a_end(int t) {return (BD?2*B_ISSUES:0)+SCALE_ISSUES+(t+1)*ISSUES;}
static constexpr int b_end(int t) {return t<2?(t+1)*B_ISSUES:INITIAL_ISSUES+B_ISSUES;}
static constexpr int wait_a(int t) {return issued_before(t)-a_end(t);}
static constexpr int wait_b(int t) {return issued_before(t)-b_end(t);}
static_assert((BM==48 && BN==32 && WN==2 && !BD) || (BM==64 && BN==32 && WN==2 && BD) || (BM==64 && BN==64 && WN==4 && BD));
static_assert(BM%16==0 && NR==1 && SFA_PASSES==1 && SCALES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> __global__ void actual_short_139_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T> __global__ __launch_bounds__(T::BLOCK_SIZE,1)
void actual_short_139_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,MR=T::MR,NR=T::NR;
constexpr int M=T::M,N=T::N;
const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64);
const unsigned pid=block_id_x(),gn=N/BN;
const int row=M<=BM?0:(pid/gn)*BM,col=M<=BM?pid*BN:(pid%gn)*BN;
const int valid_rows=M<=BM?M:(M-row<BM?M-row:BM);
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*T::K,static_cast<unsigned>(valid_rows*T::K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*T::K,static_cast<unsigned>(BN*T::K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*T::SCALES));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*T::SCALES));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<array<typename Base::vtype_b,NR>,2>,T::B_SETS> bregs;
auto issue_b=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int off=ks*2048+kg*256+lane16*16,soff=br*T::K+t*4096;
set_slice(bregs[t%T::B_SETS][ks][n],load<16>(gb,off,soff,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[t%T::B_SETS][ks][n],load<16>(gb,off+1024,soff,number<T::B_CACHE>{}),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
if constexpr(T::B_DIRECT)static_for<2>([&](auto ti){issue_b(ti);});
static_for<T::SFA_PASSES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*T::BLOCK_SIZE+tid)*4,sr=index%BM,sc=index/BM;
const int source=index<BM*T::SCALES && sr<valid_rows?sc*M+row+sr:M*T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFA+(i*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if(wave==0) {
const int source=lane<T::SCALES?(col/128)*T::SCALES+lane:(N/128)*T::SCALES;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SFB));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,source,0,0,0);
}
auto issue=[&](auto ti) {
constexpr int t=decltype(ti)::value;
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/256,kc=(lin%256)^((r%16)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*T::K+kc,t*256,0,0);
});
if constexpr(!T::B_DIRECT)static_for<T::B_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,group=lin/4096,within=lin%4096;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+t*T::STAGE_BYTES+T::A_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gb.cached_rsrc,dst,16,group*(T::K*16)+within,t*4096,0,T::B_CACHE);
});
__builtin_amdgcn_asyncmark();__builtin_amdgcn_sched_barrier(0);
};
static_for<T::PANELS>([&](auto ti){issue(ti);});
static_for<T::PANELS>([&](auto ti) {
constexpr int t=decltype(ti)::value;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(T::PANELS-t-1);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<array<typename Base::vtype_a,MR>,2> af;
array<array<typename Base::vtype_b,NR>,2> bf;
array<array<unsigned,MR>,2> as;
array<unsigned,2> bs;
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
bs[ks]=load<1>(ss,T::OFF_SFB+(t*2+ks)*4)[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=t*T::STAGE_BYTES+ar*256+(((ks*8+kg)^(ar%16))*16);
const int hi=t*T::STAGE_BYTES+ar*256+(((ks*8+kg+4)^(ar%16))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
as[ks][m]=load<1>(ss,T::OFF_SFA+(t*2+ks)*BM+ar)[0];
});
if constexpr(!T::B_DIRECT)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wave*(BN/T::T_N)+n*16;
const int bo=t*T::STAGE_BYTES+T::A_BYTES+(br/16)*4096+ks*2048+kg*256+lane16*16;
set_slice(bf[ks][n],load<16>(sa,bo),0_I,16_I);
set_slice(bf[ks][n],load<16>(sa,bo+1024),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(T::B_DIRECT)s_waitcnt_vmcnt(number<(t<2?T::wait_b(t)-1:T::wait_b(t))>{});
s_waitcnt_lgkmcnt(0_I);
static_for<2>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::B_DIRECT)c[ci]=Base{}(af[ks][m],bregs[t%T::B_SETS][ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
else c[ci]=Base{}(af[ks][m],bf[ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
});
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(T::B_DIRECT && t==0)issue_b(2_I);
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR*NR>([&](auto ci) {
constexpr int i=decltype(ci)::value;
const auto value=cast<bf16_t>(c[i]);
const int rr=(i/NR)*16+kg*4,nn=wave*(BN/T::T_N)+(i%NR)*16+lane16;
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
if(rr+e<valid_rows)store<1>(sc,vector_t<bf16_t,1>{value[e]},(rr+e)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto ii) {
const int index=decltype(ii)::value*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<valid_rows)store<8>(gc,load<8>(sc,rr*(BN+8)+nn),rr*N+nn);
});
s_waitcnt_vmcnt(0_I);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_93) && !defined(OPUS_BP_INCLUDE_OVERLAP_93_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_93_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_93 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_92;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<48,32,2,false,2,48>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=139 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
actual_short_139_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_K256_fixedMN_traits<48,32,2,false,2,48>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=139 || literal!=1 || m!=Traits::M || n!=Traits::N || k!=Traits::K)return -11;
FixedMContract contract{Traits::M,Traits::N,Traits::K,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
constexpr unsigned total=((Traits::M+Traits::B_M-1)/Traits::B_M)*(Traits::N/Traits::B_N);
actual_short_139_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_94) && !defined(OPUS_BP_INCLUDE_OVERLAP_94_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_94_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_94 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_splitM2_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=2,T_N=2;
static constexpr int E_M=BM/32,E_N=BN/32;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=64*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==128);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==5);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void actual_short_143_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_splitM2_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=2,T_N=2;
static constexpr int E_M=BM/32,E_N=BN/32;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=64*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==128);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==5);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void actual_short_143_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void actual_short_143_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int wm=wave%T::T_M,wn=wave/T::T_M;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*4,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
if(wave==0) {
const int source=lane<T::SFB_BYTES ? (col/128+lane/LOOPS)*LOOPS+lane%LOOPS:(N/128)*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(sfb.ptr));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,1,source,0,0,0);
}
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wn*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=(m*T::T_M+wm)*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wn*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,(scale_group*LOOPS+t)*4)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=(m*T::T_M+wm)*16+kgroup*4+e,nc=wn*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_95) && !defined(OPUS_BP_INCLUDE_OVERLAP_95_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_95_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_95 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_94;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_splitM2_traits<128,64,5,2,208,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=143 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
actual_short_143_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_splitM2_traits<128,64,5,2,208,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=143 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
actual_short_143_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_96) && !defined(OPUS_BP_INCLUDE_OVERLAP_96_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_96_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_96 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*16,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
const int sb_source=tid<T::SFB_BYTES ? (col/128+tid/LOOPS)*LOOPS+tid%LOOPS:(N/128)*LOOPS;
auto* sb_dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfb.ptr+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sb_dst,1,sb_source,0,0,0);
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,(scale_group*LOOPS+t)*4)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_97) && !defined(OPUS_BP_INCLUDE_OVERLAP_97_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_97_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_97 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,160,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,160,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_98) && !defined(OPUS_BP_INCLUDE_OVERLAP_98_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_98_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_98 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,1856,7168,384,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,1856,7168,384,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_99) && !defined(OPUS_BP_INCLUDE_OVERLAP_99_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_99_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_99 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,4,1,320,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,4,1,320,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_100) && !defined(OPUS_BP_INCLUDE_OVERLAP_100_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_100_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_100 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_101) && !defined(OPUS_BP_INCLUDE_OVERLAP_101_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_101_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_101 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_LDSoverlap_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void shortK_LDSoverlap_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_LDSoverlap_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void shortK_LDSoverlap_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_LDSoverlap_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*16,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
const int sb_source=tid<T::SFB_BYTES ? (col/128+tid/LOOPS)*LOOPS+tid%LOOPS:(N/128)*LOOPS;
auto* sb_dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfb.ptr+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sb_dst,1,sb_source,0,0,0);
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,(scale_group*LOOPS+t)*4)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_102) && !defined(OPUS_BP_INCLUDE_OVERLAP_102_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_102_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_102 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_101;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_LDSoverlap_traits<96,64,6,2,144,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_LDSoverlap_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_LDSoverlap_traits<96,64,6,2,144,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_LDSoverlap_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_103) && !defined(OPUS_BP_INCLUDE_OVERLAP_103_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_103_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_103 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_101;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_LDSoverlap_traits<96,64,6,2,160,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_LDSoverlap_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_LDSoverlap_traits<96,64,6,2,160,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_LDSoverlap_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_104) && !defined(OPUS_BP_INCLUDE_OVERLAP_104_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_104_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_104 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_101;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_LDSoverlap_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_LDSoverlap_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_LDSoverlap_traits<64,128,4,1,64,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_LDSoverlap_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_105) && !defined(OPUS_BP_INCLUDE_OVERLAP_105_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_105_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_105 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,144,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,144,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_106) && !defined(OPUS_BP_INCLUDE_OVERLAP_106_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_106_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_106 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,176,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,176,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_107) && !defined(OPUS_BP_INCLUDE_OVERLAP_107_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_107_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_107 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,192,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,192,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_108) && !defined(OPUS_BP_INCLUDE_OVERLAP_108_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_108_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_108 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,4,1,288,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,4,1,288,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_109) && !defined(OPUS_BP_INCLUDE_OVERLAP_109_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_109_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_109 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,4,1,352,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,4,1,352,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_110) && !defined(OPUS_BP_INCLUDE_OVERLAP_110_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_110_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_110 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,4,1,384,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,4,1,384,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_111) && !defined(OPUS_BP_INCLUDE_OVERLAP_111_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_111_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_111 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==3 || Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==3 || Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*16,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
const int sb_source=tid<T::SFB_BYTES ? (col/128+tid/LOOPS)*LOOPS+tid%LOOPS:(N/128)*LOOPS;
auto* sb_dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfb.ptr+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sb_dst,1,sb_source,0,0,0);
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,(scale_group*LOOPS+t)*4)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_112) && !defined(OPUS_BP_INCLUDE_OVERLAP_112_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_112_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_112 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_111;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,2,48,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<64,128,4,2,48,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_113) && !defined(OPUS_BP_INCLUDE_OVERLAP_113_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_113_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_113 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_111;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,80,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,80,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_114) && !defined(OPUS_BP_INCLUDE_OVERLAP_114_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_114_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_114 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_111;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,96,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,96,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_115) && !defined(OPUS_BP_INCLUDE_OVERLAP_115_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_115_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_115 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,96,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,64,6,2,96,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_116) && !defined(OPUS_BP_INCLUDE_OVERLAP_116_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_116_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_116 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_splitM2_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=2,T_N=2;
static constexpr int E_M=BM/32,E_N=BN/32;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==128);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==5);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void shortK_splitM2_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_splitM2_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=2,T_N=2;
static constexpr int E_M=BM/32,E_N=BN/32;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==128);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==5);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void shortK_splitM2_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_splitM2_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int wm=wave%T::T_M,wn=wave/T::T_M;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*16,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
const int sb_source=tid<T::SFB_BYTES ? (col/128+tid/LOOPS)*LOOPS+tid%LOOPS:(N/128)*LOOPS;
auto* sb_dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfb.ptr+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sb_dst,1,sb_source,0,0,0);
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wn*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=(m*T::T_M+wm)*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wn*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,(scale_group*LOOPS+t)*4)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=(m*T::T_M+wm)*16+kgroup*4+e,nc=wn*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_117) && !defined(OPUS_BP_INCLUDE_OVERLAP_117_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_117_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_117 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_116;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_splitM2_traits<128,64,5,2,208,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_splitM2_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_splitM2_traits<128,64,5,2,208,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_splitM2_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_118) && !defined(OPUS_BP_INCLUDE_OVERLAP_118_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_118_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_118 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_116;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_splitM2_traits<128,64,5,2,256,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_splitM2_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_splitM2_traits<128,64,5,2,256,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_splitM2_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_119) && !defined(OPUS_BP_INCLUDE_OVERLAP_119_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_119_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_119 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==3 || Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct shortK_directB_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=256,T_M=1,T_N=4;
static constexpr int E_M=BM/16,E_N=BN/64;
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_M=FixedM,STATIC_N=FixedN,FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int A_PASSES=BM*128/(BLOCK_SIZE*16),A_STAGE=BM*128;
static constexpr int INITIAL_A=LOOPS<Stages-1 ? LOOPS:Stages-1;
static constexpr int SFA_BYTES=BM*LOOPS,SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*16;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=BLOCK_SIZE*4;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES:PIPELINE_BYTES;
static constexpr int OUTPUT=1;
static_assert(BM==64 || BM==96);
static_assert(BN==64 || BN==128);
static_assert(FixedM%16==0 && FixedN%BN==0 && FixedK%128==0);
static_assert(BAhead>=1 && BAhead<Stages && BAhead<LOOPS);
static_assert(Stages==3 || Stages==4 || Stages==6);
static_assert(SFB_BYTES<=BLOCK_SIZE && LDS_BYTES<=160*1024);
static constexpr int B_ISSUES=E_N*2,SCALE_ISSUES=SFA_PASSES+1;
static constexpr int issued_before(int t) {
return SCALE_ISSUES+B_AHEAD*B_ISSUES+INITIAL_A*A_PASSES+
(t<LOOPS-B_AHEAD ? t:LOOPS-B_AHEAD)*B_ISSUES+
(t<LOOPS-(Stages-1) ? t:(LOOPS>Stages-1 ? LOOPS-(Stages-1):0))*A_PASSES;
}
static constexpr int a_end(int t) {
return t<INITIAL_A ? SCALE_ISSUES+B_AHEAD*B_ISSUES+(t+1)*A_PASSES:
issued_before(t-(Stages-1))+B_ISSUES+A_PASSES;
}
static constexpr int b_end(int t) {
return t<B_AHEAD ? SCALE_ISSUES+(t+1)*B_ISSUES:
issued_before(t-B_AHEAD)+B_ISSUES;
}
static constexpr int wait_a(int t) { return issued_before(t)-a_end(t); }
static constexpr int after_future(int t) {
return issued_before(t)+(t+B_AHEAD<LOOPS ? B_ISSUES:0)+(t+Stages-1<LOOPS ? A_PASSES:0);
}
static constexpr int wait_b(int t) { return after_future(t)-b_end(t); }
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void shortK_directB_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,K=T::FIXED_K,LOOPS=T::LOOPS;
constexpr int MR=T::E_M,NR=T::E_N,STAGES=T::NUM_STAGES,BS=T::B_SETS;
constexpr int AD=STAGES-1,BD=T::B_AHEAD;
constexpr int M=T::FIXED_M,N=T::STATIC_N;
const unsigned gn=N/BN,gm=(M+BM-1)/BM,total=gn*gm,pid=block_id_x();
const unsigned tile=pid;
const int row=(tile/gn)*BM,col=(tile%gn)*BN;
const unsigned tid=thread_id_x();
__builtin_assume(tid<T::BLOCK_SIZE);
const unsigned wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const int valid_rows=M-row<BM ? M-row : BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,
static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
static_cast<unsigned>(M*LOOPS));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
static_cast<unsigned>((N/128)*LOOPS));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,
static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto sfa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFA));
auto sfb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SFB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
static_for<T::SFA_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+tid)*16,r=index%BM,kt=index/BM;
const int source=index<T::SFA_BYTES && row+r<M ? kt*M+row+r:M*LOOPS;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfa.ptr+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
const int sb_source=tid<T::SFB_BYTES ? (col/128+tid/LOOPS)*LOOPS+tid%LOOPS:(N/128)*LOOPS;
auto* sb_dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sfb.ptr+wave*64*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sb_dst,1,sb_source,0,0,0);
__builtin_amdgcn_sched_barrier(0);
auto issue_b_column=[&](auto set_i,auto ni,int kt) {
constexpr int set=decltype(set_i)::value;
constexpr int n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16;
const int offset=lane*16,soffset=nr*K+kt*2048;
set_slice(bregs[set][n],load<16>(gb,offset,soffset,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,offset+1024,soffset,number<T::B_CACHE>{}),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto set_i,int kt) {
static_for<NR>([&](auto ni) { issue_b_column(set_i,ni,kt); });
};
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int lin=(p*T::BLOCK_SIZE+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(
sa.ptr+slot*T::A_STAGE+(p*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r<BM ? r*K+kc : valid_rows*K,kt*128,0,0);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<BD>([&](auto bi) { issue_b(bi,decltype(bi)::value); });
static_for<T::INITIAL_A>([&](auto ai) {
issue_a(ai,decltype(ai)::value);
__builtin_amdgcn_asyncmark();
});
auto step=[&](auto ti) {
constexpr int t=decltype(ti)::value,slot=t%STAGES,set=t%BS;
constexpr int pending=LOOPS-t-1<AD-1 ? LOOPS-t-1:AD-1;
static_assert(T::wait_a(t)>=0 && T::wait_a(t)<64 && T::wait_b(t)>=0 && T::wait_b(t)<64);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_wait_asyncmark(pending);
s_waitcnt_vmcnt(number<T::wait_a(t)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+BD<LOOPS) issue_b(number<(t+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=slot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=slot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(sfa,t*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int scale_group=(wave*(BN/T::T_N)+n*16)/128;
bscale[n]=load<1>(sfb,(scale_group*LOOPS+t)*4)[0];
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
if constexpr(t+AD<LOOPS) {
issue_a(number<(t+AD)%STAGES>{},t+AD);
__builtin_amdgcn_asyncmark();
}
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::wait_b(t)>{});
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[set][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<LOOPS>([&](auto ti) { step(ti); });
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(T::OUTPUT==1) {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto value=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ei) {
constexpr int e=decltype(ei)::value;
const int r=m*16+kgroup*4+e,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{value[e]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(BM*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE>([&](auto pi) {
const int index=decltype(pi)::value*T::BLOCK_SIZE+tid,r=index/(BN/8),nc=index%(BN/8)*8;
if(r<valid_rows)
store<8>(gc,load<8>(sc,r*(BN+8)+nc),r*N+nc);
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_OVERLAP_120) && !defined(OPUS_BP_INCLUDE_OVERLAP_120_DONE)
#define OPUS_BP_INCLUDE_OVERLAP_120_DONE
namespace opus_bpreshuffle_gfx950::overlap::schedule_120 {
using namespace opus_bpreshuffle_gfx950::overlap::schedule_119;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,80,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=shortK_directB_traits<96,128,3,1,80,65536,1536,2>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
shortK_directB_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif
