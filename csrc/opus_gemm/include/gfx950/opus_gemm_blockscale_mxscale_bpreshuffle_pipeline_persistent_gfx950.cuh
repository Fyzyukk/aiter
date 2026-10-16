#ifndef OPUS_BP_PERSISTENT_PIPELINE_BASE
#define OPUS_BP_PERSISTENT_PIPELINE_BASE
// SPDX-License-Identifier: Apache-2.0

#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_persistent_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)

namespace opus_bpreshuffle_gfx950::persistent::schedule_0 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "A operand AGPR candidate requires amdgpu-pin-op-dst toolchain"
#endif
template<class T>
__device__ __forceinline__ 
void large_C_AGPR_B3_bdirect_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
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
i32x4_t al=*reinterpret_cast<OPUS_LDS_ADDR i32x4_t*>(sa.ptr+lo);
i32x4_t ah=*reinterpret_cast<OPUS_LDS_ADDR i32x4_t*>(sa.ptr+hi);
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
[[clang::amdgpu_pin_agpr(ci*4)]]
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

namespace opus_bpreshuffle_gfx950::persistent::schedule_1 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T>
__device__ __forceinline__ 
void large_pinc270_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int K=T::FIXED_K,BM=T::B_M,BN=T::B_N,THREADS=T::BLOCK_SIZE;
constexpr int MR=T::E_M,NR=T::E_N,BSETS=T::B_SETS,ASTAGES=T::NUM_STAGES;
constexpr int TILES=T::LOOPS;
const int M=args.m,N=T::FIXED_N ? T::FIXED_N:args.n;
const unsigned tiles_n=static_cast<unsigned>(N)/BN;
const unsigned total=((static_cast<unsigned>(M)+BM-1)/BM)*tiles_n;
const unsigned pid=static_cast<unsigned>(block_id_x());
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_ORDER ? (pid%8u)*group+pid/8u:pid;
if (tile>=total) return;
const int row=(tile/tiles_n)*BM,col=(tile%tiles_n)*BN;
const int valid_rows=M-row<BM ? M-row:BM;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const uint64_t a_extent=static_cast<uint64_t>(valid_rows)*K;
const uint64_t sa_extent=static_cast<uint64_t>(M)*TILES;
const uint64_t sb_extent=static_cast<uint64_t>(N/128)*TILES;
const uint64_t c_extent=(static_cast<uint64_t>(valid_rows-1)*N+BN)*2;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+static_cast<uint64_t>(row)*K,
static_cast<uint32_t>(a_extent));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+static_cast<uint64_t>(col)*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<uint32_t>(sa_extent));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<uint32_t>(sb_extent));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+static_cast<uint64_t>(row)*N+col,
static_cast<uint32_t>(c_extent));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto ssa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SA));
auto ssb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BSETS> bregs;
static_for<T::SA_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int index=(pass*THREADS+tid)*4,r=index%BM,kt=index/BM;
const uint64_t src64=index<BM*TILES && row+r<M ? static_cast<uint64_t>(kt)*M+row+r:sa_extent;
const int src=__builtin_bit_cast(int,static_cast<uint32_t>(src64));
async_load<4>(gsa,ssa.ptr+(pass*THREADS+wave*64)*4,src);
});
static_for<T::SB_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int index=pass*THREADS+tid;
const uint64_t src64=index<(BN/128)*TILES
? static_cast<uint64_t>(col/128+index/TILES)*TILES+index%TILES:sb_extent;
const int src=__builtin_bit_cast(int,static_cast<uint32_t>(src64));
async_load<1>(gsb,ssb.ptr+(pass*THREADS+wave*64)*4,src);
});
__builtin_amdgcn_sched_barrier(0);
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int lin=(pass*THREADS+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(
reinterpret_cast<__UINTPTR_TYPE__>(sa.ptr+slot*T::A_STAGE+(pass*THREADS+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
};
auto issue_b_column=[&](auto slot_i,auto ni,int kt) {
constexpr int slot=decltype(slot_i)::value,n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16,offset=nr*K+lane*16;
set_slice(bregs[slot][n],load<16>(gb,offset,kt*128*16),0_I,16_I);
set_slice(bregs[slot][n],load<16>(gb,offset+1024,kt*128*16),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto slot,int kt) {
static_for<NR>([&](auto ni){issue_b_column(slot,ni,kt);});
};
static_for<BSETS>([&](auto i) {
issue_b(i,decltype(i)::value);
issue_a(i,decltype(i)::value);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
});
auto step=[&](auto a_i,auto b_i,int kt,auto future_i,auto pending_i) {
constexpr int aslot=decltype(a_i)::value,bslot=decltype(b_i)::value;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=aslot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=aslot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(ssa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int nrow=wave*(BN/T::T_N)+n*16;
bscale[n]=load<1>(ssb,((nrow/128)*TILES+kt)*4)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr (decltype(future_i)::value)
issue_a(number<(aslot+BSETS)%ASTAGES>{},kt+BSETS);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[m],bregs[bslot][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr (decltype(future_i)::value) issue_b_column(b_i,ni,kt+BSETS);
});
if constexpr (decltype(future_i)::value) {
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
}
};
constexpr int STEADY=TILES-BSETS,MAIN=STEADY/T::UNROLL*T::UNROLL;
#pragma clang loop unroll(disable)
for (int base=0;base<MAIN;base+=T::UNROLL) {
static_for<T::UNROLL>([&](auto phase_i) {
constexpr int phase=decltype(phase_i)::value;
step(number<phase%ASTAGES>{},number<phase%BSETS>{},base+phase,1_I,number<BSETS-1>{});
});
}
static_for<STEADY-MAIN>([&](auto pi) {
constexpr int kt=MAIN+decltype(pi)::value;
step(number<kt%ASTAGES>{},number<kt%BSETS>{},kt,1_I,number<BSETS-1>{});
});
static_for<BSETS>([&](auto pi) {
constexpr int kt=STEADY+decltype(pi)::value,pending=TILES-1-kt;
step(number<kt%ASTAGES>{},number<kt%BSETS>{},kt,0_I,number<pending>{});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr (T::DIRECT_OUTPUT) {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto values=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ii) {
constexpr int element=decltype(ii)::value;
const int r=m*16+kgroup*4+element,nc=wave*(BN/T::T_N)+n*16+lane16;
const uint64_t byte=(static_cast<uint64_t>(r)*N+nc)*2;
gc._store<1>(vector_t<bf16_t,1>{values[element]},
__builtin_bit_cast(int,static_cast<uint32_t>(byte)),0,number<T::OUTPUT_CACHE>{});
});
});
});
} else {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<BM/T::C_ROWS>([&](auto chunk_i) {
constexpr int chunk=decltype(chunk_i)::value;
static_for<T::C_ROWS/16>([&](auto mi) {
constexpr int m=decltype(mi)::value,cm=chunk*(T::C_ROWS/16)+m;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto values=cast<bf16_t>(c[cm*NR+n]);
static_for<4>([&](auto ii) {
constexpr int element=decltype(ii)::value;
const int r=m*16+kgroup*4+element,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{values[element]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<T::OUTPUT_PASSES>([&](auto pi) {
const int linear=(decltype(pi)::value*THREADS+tid)*8,r=linear/BN,nc=linear%BN;
if (r<T::C_ROWS && chunk*T::C_ROWS+r<valid_rows) {
const uint64_t byte=(static_cast<uint64_t>(chunk*T::C_ROWS+r)*N+nc)*2;
gc._store<8>(load<8>(sc,r*(BN+8)+nc),
__builtin_bit_cast(int,static_cast<uint32_t>(byte)),0,number<T::OUTPUT_CACHE>{});
}
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
});
}
}
#endif
}

namespace opus_bpreshuffle_gfx950::persistent::schedule_2 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T>
__device__ __forceinline__ 
void large_slice260_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int K=T::FIXED_K,BM=T::B_M,BN=T::B_N,THREADS=T::BLOCK_SIZE;
constexpr int MR=T::E_M,NR=T::E_N,BSETS=T::B_SETS,ASTAGES=T::NUM_STAGES;
constexpr int TILES=T::LOOPS;
const int M=args.m,N=T::FIXED_N ? T::FIXED_N:args.n;
const unsigned tiles_n=static_cast<unsigned>(N)/BN;
const unsigned tiles_m=(static_cast<unsigned>(M)+BM-1)/BM;
const unsigned total=tiles_m*tiles_n;
const unsigned pid=static_cast<unsigned>(block_id_x());
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_ORDER ? (pid%8u)*group+pid/8u:pid;
const unsigned n_tile=T::WORKERS_M ? pid%tiles_n:tile%tiles_n;
const unsigned first_m=T::WORKERS_M ? pid/tiles_n:tile/tiles_n;
if ((!T::WORKERS_M && tile>=total) || first_m>=tiles_m) return;
const int col=n_tile*BN;
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto ssb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SB));
const uint64_t sb_extent=static_cast<uint64_t>(N/128)*TILES;
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<uint32_t>(sb_extent));
static_for<T::SB_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int index=pass*THREADS+thread_id_x();
const uint64_t src64=index<(BN/128)*TILES
? static_cast<uint64_t>(col/128+index/TILES)*TILES+index%TILES:sb_extent;
const int src=__builtin_bit_cast(int,static_cast<uint32_t>(src64));
const int scale_wave=__builtin_amdgcn_readfirstlane(thread_id_x()/64);
async_load<1>(gsb,ssb.ptr+(pass*THREADS+scale_wave*64)*4,src);
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
auto process_m_tile=[&](unsigned m_tile) {
const int row=m_tile*BM;
const int valid_rows=M-row<BM ? M-row:BM;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const uint64_t a_extent=static_cast<uint64_t>(valid_rows)*K;
const uint64_t sa_extent=static_cast<uint64_t>(M)*TILES;
const uint64_t c_extent=(static_cast<uint64_t>(valid_rows-1)*N+BN)*2;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+static_cast<uint64_t>(row)*K,
static_cast<uint32_t>(a_extent));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+static_cast<uint64_t>(col)*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<uint32_t>(sa_extent));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+static_cast<uint64_t>(row)*N+col,
static_cast<uint32_t>(c_extent));
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto ssa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SA));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BSETS> bregs;
static_for<T::SA_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int index=(pass*THREADS+tid)*4,r=index%BM,kt=index/BM;
const uint64_t src64=index<BM*TILES && row+r<M ? static_cast<uint64_t>(kt)*M+row+r:sa_extent;
const int src=__builtin_bit_cast(int,static_cast<uint32_t>(src64));
async_load<4>(gsa,ssa.ptr+(pass*THREADS+wave*64)*4,src);
});
__builtin_amdgcn_sched_barrier(0);
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int lin=(pass*THREADS+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(
reinterpret_cast<__UINTPTR_TYPE__>(sa.ptr+slot*T::A_STAGE+(pass*THREADS+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
};
auto issue_b_column=[&](auto slot_i,auto ni,int kt) {
constexpr int slot=decltype(slot_i)::value,n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16,offset=nr*K+lane*16;
set_slice(bregs[slot][n],load<16>(gb,offset,kt*128*16),0_I,16_I);
set_slice(bregs[slot][n],load<16>(gb,offset+1024,kt*128*16),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto slot,int kt) {
static_for<NR>([&](auto ni){issue_b_column(slot,ni,kt);});
};
static_for<BSETS>([&](auto i) {
issue_b(i,decltype(i)::value);
issue_a(i,decltype(i)::value);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
});
auto step=[&](auto a_i,auto b_i,int kt,auto future_i,auto pending_i) {
constexpr int aslot=decltype(a_i)::value,bslot=decltype(b_i)::value;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<unsigned,NR> bscale;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int nrow=wave*(BN/T::T_N)+n*16;
bscale[n]=load<1>(ssb,((nrow/128)*TILES+kt)*4)[0];
});
static_for<T::M_SLICES>([&](auto slice_i) {
constexpr int first=decltype(slice_i)::value*T::SLICE_M;
constexpr int count=MR-first<T::SLICE_M ? MR-first:T::SLICE_M;
array<typename Base::vtype_a,count> afrags;
array<unsigned,count> ascale;
static_for<count>([&](auto local_i) {
constexpr int local=decltype(local_i)::value,m=first+local;
const int r=m*16+lane16,swz=r%8;
const int lo=aslot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=aslot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[local],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[local],load<16>(sa,hi),16_I,32_I);
ascale[local]=load<1>(ssa,kt*BM+r)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr (first+count==MR && decltype(future_i)::value)
issue_a(number<(aslot+BSETS)%ASTAGES>{},kt+BSETS);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<count>([&](auto local_i) {
constexpr int local=decltype(local_i)::value,m=first+local,ci=m*NR+n;
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(afrags[local],bregs[bslot][n],c[ci],
static_cast<int>(ascale[local]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr (first+count==MR && decltype(future_i)::value)
issue_b_column(b_i,ni,kt+BSETS);
});
});
if constexpr (decltype(future_i)::value) {
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
}
};
constexpr int STEADY=TILES-BSETS,MAIN=STEADY/T::UNROLL*T::UNROLL;
#pragma clang loop unroll(disable)
for (int base=0;base<MAIN;base+=T::UNROLL) {
static_for<T::UNROLL>([&](auto phase_i) {
constexpr int phase=decltype(phase_i)::value;
step(number<phase%ASTAGES>{},number<phase%BSETS>{},base+phase,1_I,number<BSETS-1>{});
});
}
static_for<STEADY-MAIN>([&](auto pi) {
constexpr int kt=MAIN+decltype(pi)::value;
step(number<kt%ASTAGES>{},number<kt%BSETS>{},kt,1_I,number<BSETS-1>{});
});
static_for<BSETS>([&](auto pi) {
constexpr int kt=STEADY+decltype(pi)::value,pending=TILES-1-kt;
step(number<kt%ASTAGES>{},number<kt%BSETS>{},kt,0_I,number<pending>{});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr (T::DIRECT_OUTPUT) {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto values=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ii) {
constexpr int element=decltype(ii)::value;
const int r=m*16+kgroup*4+element,nc=wave*(BN/T::T_N)+n*16+lane16;
const uint64_t byte=(static_cast<uint64_t>(r)*N+nc)*2;
gc._store<1>(vector_t<bf16_t,1>{values[element]},
__builtin_bit_cast(int,static_cast<uint32_t>(byte)),0,number<T::OUTPUT_CACHE>{});
});
});
});
} else {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<BM/T::C_ROWS>([&](auto chunk_i) {
constexpr int chunk=decltype(chunk_i)::value;
static_for<T::C_ROWS/16>([&](auto mi) {
constexpr int m=decltype(mi)::value,cm=chunk*(T::C_ROWS/16)+m;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto values=cast<bf16_t>(c[cm*NR+n]);
static_for<4>([&](auto ii) {
constexpr int element=decltype(ii)::value;
const int r=m*16+kgroup*4+element,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{values[element]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<T::OUTPUT_PASSES>([&](auto pi) {
const int linear=(decltype(pi)::value*THREADS+tid)*8,r=linear/BN,nc=linear%BN;
if (r<T::C_ROWS && chunk*T::C_ROWS+r<valid_rows) {
const uint64_t byte=(static_cast<uint64_t>(chunk*T::C_ROWS+r)*N+nc)*2;
gc._store<8>(load<8>(sc,r*(BN+8)+nc),
__builtin_bit_cast(int,static_cast<uint32_t>(byte)),0,number<T::OUTPUT_CACHE>{});
}
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
});
}
};
if constexpr (T::WORKERS_M>0) {
for (unsigned m_tile=first_m;m_tile<tiles_m;m_tile+=T::WORKERS_M) {
process_m_tile(m_tile);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
} else {
process_m_tile(first_m);
}
}
#endif
}
#endif

#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_3) && !defined(OPUS_BP_INCLUDE_PERSISTENT_3_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_3_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_3 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0




template<int BM,int BN,int WavesN,int FixedK,int FixedN,int Ahead,bool Direct,bool XCD>
struct large_bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=WavesN;
static constexpr int BLOCK_SIZE=WavesN*64,NUM_WAVES=WavesN,WARP_SIZE=64;
static constexpr int FIXED_K=FixedK,FIXED_N=FixedN,LOOPS=FixedK/128;
static constexpr int E_M=BM/16,E_N=BN/(WavesN*16);
static constexpr int B_AHEAD=Ahead,B_SETS=Ahead+1,NUM_STAGES=B_SETS+1;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
static constexpr int SA_PASSES=(BM*LOOPS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SA_BYTES=SA_PASSES*BLOCK_SIZE*4;
static constexpr int SB_PASSES=((BN/128)*LOOPS+BLOCK_SIZE-1)/BLOCK_SIZE;
static constexpr int SB_BYTES=SB_PASSES*BLOCK_SIZE*4;
static constexpr int OFF_SA=NUM_STAGES*A_STAGE,OFF_SB=OFF_SA+SA_BYTES;
static constexpr int MATRIX_BYTES=OFF_SB+SB_BYTES;
static constexpr int C_ROWS=(BM>128 || BN>256) ? BM/2 : BM;
static constexpr int C_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=Direct || MATRIX_BYTES>C_BYTES ? MATRIX_BYTES:C_BYTES;
static constexpr int OUTPUT_PASSES=(C_ROWS*BN+BLOCK_SIZE*8-1)/(BLOCK_SIZE*8);
static constexpr int UNROLL=NUM_STAGES*B_SETS;
static constexpr int OUTPUT_CACHE=2;
static constexpr bool DIRECT_OUTPUT=Direct,XCD_ORDER=XCD;
static_assert(BM%16==0 && BN%128==0 && BN%(WavesN*16)==0);
static_assert(WavesN==2 || WavesN==4);
static_assert(Ahead==1 || Ahead==2);
static_assert(LOOPS>=B_SETS && FixedK%128==0);
static_assert(C_ROWS%16==0 && BM%C_ROWS==0);
static_assert(LDS_BYTES<=160*1024);
};
template<class T>
__global__ void large_bdirect_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0




template<int BM,int BN,int WavesN,int FixedK,int FixedN,int Ahead,bool Direct,bool XCD>
struct large_bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=WavesN;
static constexpr int BLOCK_SIZE=WavesN*64,NUM_WAVES=WavesN,WARP_SIZE=64;
static constexpr int FIXED_K=FixedK,FIXED_N=FixedN,LOOPS=FixedK/128;
static constexpr int E_M=BM/16,E_N=BN/(WavesN*16);
static constexpr int B_AHEAD=Ahead,B_SETS=Ahead+1,NUM_STAGES=B_SETS+1;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
static constexpr int SA_PASSES=(BM*LOOPS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SA_BYTES=SA_PASSES*BLOCK_SIZE*4;
static constexpr int SB_PASSES=((BN/128)*LOOPS+BLOCK_SIZE-1)/BLOCK_SIZE;
static constexpr int SB_BYTES=SB_PASSES*BLOCK_SIZE*4;
static constexpr int OFF_SA=NUM_STAGES*A_STAGE,OFF_SB=OFF_SA+SA_BYTES;
static constexpr int MATRIX_BYTES=OFF_SB+SB_BYTES;
static constexpr int C_ROWS=(BM>128 || BN>256) ? BM/2 : BM;
static constexpr int C_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=Direct || MATRIX_BYTES>C_BYTES ? MATRIX_BYTES:C_BYTES;
static constexpr int OUTPUT_PASSES=(C_ROWS*BN+BLOCK_SIZE*8-1)/(BLOCK_SIZE*8);
static constexpr int UNROLL=NUM_STAGES*B_SETS;
static constexpr int OUTPUT_CACHE=2;
static constexpr bool DIRECT_OUTPUT=Direct,XCD_ORDER=XCD;
static_assert(BM%16==0 && BN%128==0 && BN%(WavesN*16)==0);
static_assert(WavesN==2 || WavesN==4);
static_assert(Ahead==1 || Ahead==2);
static_assert(LOOPS>=B_SETS && FixedK%128==0);
static_assert(C_ROWS%16==0 && BM%C_ROWS==0);
static_assert(LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void large_bdirect_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void large_bdirect_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int K=T::FIXED_K,BM=T::B_M,BN=T::B_N,THREADS=T::BLOCK_SIZE;
constexpr int MR=T::E_M,NR=T::E_N,BSETS=T::B_SETS,ASTAGES=T::NUM_STAGES;
constexpr int TILES=T::LOOPS;
const int M=args.m,N=T::FIXED_N ? T::FIXED_N:args.n;
const unsigned tiles_n=static_cast<unsigned>(N)/BN;
const unsigned total=((static_cast<unsigned>(M)+BM-1)/BM)*tiles_n;
const unsigned pid=static_cast<unsigned>(block_id_x());
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_ORDER ? (pid%8u)*group+pid/8u:pid;
if (tile>=total) return;
const int row=(tile/tiles_n)*BM,col=(tile%tiles_n)*BN;
const int valid_rows=M-row<BM ? M-row:BM;
const int tid=thread_id_x(),wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const uint64_t a_extent=static_cast<uint64_t>(valid_rows)*K;
const uint64_t sa_extent=static_cast<uint64_t>(M)*TILES;
const uint64_t sb_extent=static_cast<uint64_t>(N/128)*TILES;
const uint64_t c_extent=(static_cast<uint64_t>(valid_rows-1)*N+BN)*2;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+static_cast<uint64_t>(row)*K,
static_cast<uint32_t>(a_extent));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+static_cast<uint64_t>(col)*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<uint32_t>(sa_extent));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<uint32_t>(sb_extent));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+static_cast<uint64_t>(row)*N+col,
static_cast<uint32_t>(c_extent));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto ssa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SA));
auto ssb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BSETS> bregs;
static_for<T::SA_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int index=(pass*THREADS+tid)*4,r=index%BM,kt=index/BM;
const uint64_t src64=index<BM*TILES && row+r<M ? static_cast<uint64_t>(kt)*M+row+r:sa_extent;
const int src=__builtin_bit_cast(int,static_cast<uint32_t>(src64));
async_load<4>(gsa,ssa.ptr+(pass*THREADS+wave*64)*4,src);
});
static_for<T::SB_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int index=pass*THREADS+tid;
const uint64_t src64=index<(BN/128)*TILES
? static_cast<uint64_t>(col/128+index/TILES)*TILES+index%TILES:sb_extent;
const int src=__builtin_bit_cast(int,static_cast<uint32_t>(src64));
async_load<1>(gsb,ssb.ptr+(pass*THREADS+wave*64)*4,src);
});
__builtin_amdgcn_sched_barrier(0);
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int lin=(pass*THREADS+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(
reinterpret_cast<__UINTPTR_TYPE__>(sa.ptr+slot*T::A_STAGE+(pass*THREADS+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
};
auto issue_b_column=[&](auto slot_i,auto ni,int kt) {
constexpr int slot=decltype(slot_i)::value,n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16,offset=nr*K+lane*16;
set_slice(bregs[slot][n],load<16>(gb,offset,kt*128*16),0_I,16_I);
set_slice(bregs[slot][n],load<16>(gb,offset+1024,kt*128*16),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto slot,int kt) {
static_for<NR>([&](auto ni){issue_b_column(slot,ni,kt);});
};
static_for<BSETS>([&](auto i) {
issue_b(i,decltype(i)::value);
issue_a(i,decltype(i)::value);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
});
auto step=[&](auto a_i,auto b_i,int kt,auto future_i,auto pending_i) {
constexpr int aslot=decltype(a_i)::value,bslot=decltype(b_i)::value;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=aslot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=aslot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(ssa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int nrow=wave*(BN/T::T_N)+n*16;
bscale[n]=load<1>(ssb,((nrow/128)*TILES+kt)*4)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr (decltype(future_i)::value)
issue_a(number<(aslot+BSETS)%ASTAGES>{},kt+BSETS);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[bslot][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr (decltype(future_i)::value) issue_b_column(b_i,ni,kt+BSETS);
});
if constexpr (decltype(future_i)::value) {
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
}
};
constexpr int STEADY=TILES-BSETS,MAIN=STEADY/T::UNROLL*T::UNROLL;
#pragma clang loop unroll(disable)
for (int base=0;base<MAIN;base+=T::UNROLL) {
static_for<T::UNROLL>([&](auto phase_i) {
constexpr int phase=decltype(phase_i)::value;
step(number<phase%ASTAGES>{},number<phase%BSETS>{},base+phase,1_I,number<BSETS-1>{});
});
}
static_for<STEADY-MAIN>([&](auto pi) {
constexpr int kt=MAIN+decltype(pi)::value;
step(number<kt%ASTAGES>{},number<kt%BSETS>{},kt,1_I,number<BSETS-1>{});
});
static_for<BSETS>([&](auto pi) {
constexpr int kt=STEADY+decltype(pi)::value,pending=TILES-1-kt;
step(number<kt%ASTAGES>{},number<kt%BSETS>{},kt,0_I,number<pending>{});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr (T::DIRECT_OUTPUT) {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto values=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ii) {
constexpr int element=decltype(ii)::value;
const int r=m*16+kgroup*4+element,nc=wave*(BN/T::T_N)+n*16+lane16;
const uint64_t byte=(static_cast<uint64_t>(r)*N+nc)*2;
gc._store<1>(vector_t<bf16_t,1>{values[element]},
__builtin_bit_cast(int,static_cast<uint32_t>(byte)),0,number<T::OUTPUT_CACHE>{});
});
});
});
} else {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<BM/T::C_ROWS>([&](auto chunk_i) {
constexpr int chunk=decltype(chunk_i)::value;
static_for<T::C_ROWS/16>([&](auto mi) {
constexpr int m=decltype(mi)::value,cm=chunk*(T::C_ROWS/16)+m;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto values=cast<bf16_t>(c[cm*NR+n]);
static_for<4>([&](auto ii) {
constexpr int element=decltype(ii)::value;
const int r=m*16+kgroup*4+element,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{values[element]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<T::OUTPUT_PASSES>([&](auto pi) {
const int linear=(decltype(pi)::value*THREADS+tid)*8,r=linear/BN,nc=linear%BN;
if (r<T::C_ROWS && chunk*T::C_ROWS+r<valid_rows) {
const uint64_t byte=(static_cast<uint64_t>(chunk*T::C_ROWS+r)*N+nc)*2;
gc._store<8>(load<8>(sc,r*(BN+8)+nc),
__builtin_bit_cast(int,static_cast<uint32_t>(byte)),0,number<T::OUTPUT_CACHE>{});
}
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_4) && !defined(OPUS_BP_INCLUDE_PERSISTENT_4_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_4_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_4 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_long_m112n256_w1x4_k7168_ba1_lds_xcd : large_bdirect_traits<112,256,4,7168,0,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 21 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 7168 || !(N >= 1024 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_long_m112n256_w1x4_k7168_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_long_m112n256_w1x4_k7168_ba1_lds_xcd : large_bdirect_traits<112,256,4,7168,0,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 21 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 7168 || !(N >= 1024 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_long_m112n256_w1x4_k7168_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_5) && !defined(OPUS_BP_INCLUDE_PERSISTENT_5_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_5_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_5 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_long_m112n256_w1x4_k7168_ba2_lds_xcd : large_bdirect_traits<112,256,4,7168,0,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 23 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 7168 || !(N >= 1024 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_long_m112n256_w1x4_k7168_ba2_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_long_m112n256_w1x4_k7168_ba2_lds_xcd : large_bdirect_traits<112,256,4,7168,0,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 23 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 7168 || !(N >= 1024 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_long_m112n256_w1x4_k7168_ba2_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_6) && !defined(OPUS_BP_INCLUDE_PERSISTENT_6_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_6_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_6 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_narrow_m128n384_w1x4_k7168_ba1_lds_xcd : large_bdirect_traits<128,384,4,7168,768,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 20 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/128)*(N/384);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_narrow_m128n384_w1x4_k7168_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_narrow_m128n384_w1x4_k7168_ba1_lds_xcd : large_bdirect_traits<128,384,4,7168,768,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 20 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/128)*(N/384);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_narrow_m128n384_w1x4_k7168_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_7) && !defined(OPUS_BP_INCLUDE_PERSISTENT_7_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_7_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_7 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_narrow_m128n384_w1x4_k7168_ba2_lds_xcd : large_bdirect_traits<128,384,4,7168,768,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 22 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/128)*(N/384);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_narrow_m128n384_w1x4_k7168_ba2_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_narrow_m128n384_w1x4_k7168_ba2_lds_xcd : large_bdirect_traits<128,384,4,7168,768,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 22 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/128)*(N/384);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_narrow_m128n384_w1x4_k7168_ba2_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_8) && !defined(OPUS_BP_INCLUDE_PERSISTENT_8_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_8_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_8 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m64n128_w1x4_k384_ba1_direct : large_bdirect_traits<64,128,4,384,7168,1,true,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 13 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/128);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m64n128_w1x4_k384_ba1_direct>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m64n128_w1x4_k384_ba1_direct : large_bdirect_traits<64,128,4,384,7168,1,true,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 13 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/128);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m64n128_w1x4_k384_ba1_direct>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_9) && !defined(OPUS_BP_INCLUDE_PERSISTENT_9_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_9_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_9 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m64n128_w1x4_k384_ba1_lds : large_bdirect_traits<64,128,4,384,7168,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 12 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/128);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m64n128_w1x4_k384_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m64n128_w1x4_k384_ba1_lds : large_bdirect_traits<64,128,4,384,7168,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 12 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/128);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m64n128_w1x4_k384_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_10) && !defined(OPUS_BP_INCLUDE_PERSISTENT_10_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_10_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_10 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m64n128_w1x4_k768_ba1_direct : large_bdirect_traits<64,128,4,768,7168,1,true,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 17 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/128);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m64n128_w1x4_k768_ba1_direct>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m64n128_w1x4_k768_ba1_direct : large_bdirect_traits<64,128,4,768,7168,1,true,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 17 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/128);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m64n128_w1x4_k768_ba1_direct>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_11) && !defined(OPUS_BP_INCLUDE_PERSISTENT_11_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_11_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_11 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m64n128_w1x4_k768_ba1_lds : large_bdirect_traits<64,128,4,768,7168,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 16 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/128);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m64n128_w1x4_k768_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m64n128_w1x4_k768_ba1_lds : large_bdirect_traits<64,128,4,768,7168,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 16 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/128);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m64n128_w1x4_k768_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_12) && !defined(OPUS_BP_INCLUDE_PERSISTENT_12_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_12_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_12 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m96n256_w1x4_k384_ba1_direct : large_bdirect_traits<96,256,4,384,7168,1,true,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 15 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/96)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m96n256_w1x4_k384_ba1_direct>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m96n256_w1x4_k384_ba1_direct : large_bdirect_traits<96,256,4,384,7168,1,true,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 15 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/96)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m96n256_w1x4_k384_ba1_direct>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_13) && !defined(OPUS_BP_INCLUDE_PERSISTENT_13_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_13_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_13 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m96n256_w1x4_k384_ba1_lds : large_bdirect_traits<96,256,4,384,7168,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 14 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/96)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m96n256_w1x4_k384_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m96n256_w1x4_k384_ba1_lds : large_bdirect_traits<96,256,4,384,7168,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 14 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/96)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m96n256_w1x4_k384_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_14) && !defined(OPUS_BP_INCLUDE_PERSISTENT_14_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_14_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_14 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m96n256_w1x4_k768_ba1_direct : large_bdirect_traits<96,256,4,768,7168,1,true,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 19 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/96)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m96n256_w1x4_k768_ba1_direct>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m96n256_w1x4_k768_ba1_direct : large_bdirect_traits<96,256,4,768,7168,1,true,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 19 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/96)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m96n256_w1x4_k768_ba1_direct>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_15) && !defined(OPUS_BP_INCLUDE_PERSISTENT_15_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_15_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_15 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m96n256_w1x4_k768_ba1_lds : large_bdirect_traits<96,256,4,768,7168,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 18 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/96)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m96n256_w1x4_k768_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_short_m96n256_w1x4_k768_ba1_lds : large_bdirect_traits<96,256,4,768,7168,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 18 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/96)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_short_m96n256_w1x4_k768_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_16) && !defined(OPUS_BP_INCLUDE_PERSISTENT_16_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_16_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_16 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m112n256_w1x4_k1536_ba1_lds : large_bdirect_traits<112,256,4,1536,65536,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 8 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m112n256_w1x4_k1536_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m112n256_w1x4_k1536_ba1_lds : large_bdirect_traits<112,256,4,1536,65536,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 8 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m112n256_w1x4_k1536_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_17) && !defined(OPUS_BP_INCLUDE_PERSISTENT_17_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_17_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_17 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m112n256_w1x4_k1536_ba1_lds_xcd : large_bdirect_traits<112,256,4,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 4 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m112n256_w1x4_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m112n256_w1x4_k1536_ba1_lds_xcd : large_bdirect_traits<112,256,4,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 4 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m112n256_w1x4_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_18) && !defined(OPUS_BP_INCLUDE_PERSISTENT_18_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_18_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_18 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m112n256_w1x4_k1536_ba2_lds : large_bdirect_traits<112,256,4,1536,65536,2,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 9 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m112n256_w1x4_k1536_ba2_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m112n256_w1x4_k1536_ba2_lds : large_bdirect_traits<112,256,4,1536,65536,2,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 9 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m112n256_w1x4_k1536_ba2_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_19) && !defined(OPUS_BP_INCLUDE_PERSISTENT_19_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_19_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_19 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m112n256_w1x4_k1536_ba2_lds_xcd : large_bdirect_traits<112,256,4,1536,65536,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 5 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m112n256_w1x4_k1536_ba2_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m112n256_w1x4_k1536_ba2_lds_xcd : large_bdirect_traits<112,256,4,1536,65536,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 5 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/112)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m112n256_w1x4_k1536_ba2_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_20) && !defined(OPUS_BP_INCLUDE_PERSISTENT_20_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_20_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_20 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds : large_bdirect_traits<224,256,4,1536,65536,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 10 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds : large_bdirect_traits<224,256,4,1536,65536,1,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 10 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_21) && !defined(OPUS_BP_INCLUDE_PERSISTENT_21_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_21_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_21 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds_xcd : large_bdirect_traits<224,256,4,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds_xcd : large_bdirect_traits<224,256,4,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_22) && !defined(OPUS_BP_INCLUDE_PERSISTENT_22_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_22_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_22 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba2_lds : large_bdirect_traits<224,256,4,1536,65536,2,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 11 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba2_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba2_lds : large_bdirect_traits<224,256,4,1536,65536,2,false,false> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 11 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=total;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba2_lds>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_23) && !defined(OPUS_BP_INCLUDE_PERSISTENT_23_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_23_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_23 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba2_lds_xcd : large_bdirect_traits<224,256,4,1536,65536,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 7 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba2_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba2_lds_xcd : large_bdirect_traits<224,256,4,1536,65536,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 7 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba2_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_24) && !defined(OPUS_BP_INCLUDE_PERSISTENT_24_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_24_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_24 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m32n256_w1x2_k1536_ba1_lds_xcd : large_bdirect_traits<32,256,2,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 0 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/32)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m32n256_w1x2_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(128),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m32n256_w1x2_k1536_ba1_lds_xcd : large_bdirect_traits<32,256,2,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 0 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/32)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m32n256_w1x2_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(128),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_25) && !defined(OPUS_BP_INCLUDE_PERSISTENT_25_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_25_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_25 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m32n256_w1x2_k1536_ba2_lds_xcd : large_bdirect_traits<32,256,2,1536,65536,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 1 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/32)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m32n256_w1x2_k1536_ba2_lds_xcd>),dim3(grid,1,1),
dim3(128),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m32n256_w1x2_k1536_ba2_lds_xcd : large_bdirect_traits<32,256,2,1536,65536,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 1 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/32)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m32n256_w1x2_k1536_ba2_lds_xcd>),dim3(grid,1,1),
dim3(128),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_26) && !defined(OPUS_BP_INCLUDE_PERSISTENT_26_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_26_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_26 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m64n256_w1x2_k1536_ba1_lds_xcd : large_bdirect_traits<64,256,2,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 2 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m64n256_w1x2_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(128),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m64n256_w1x2_k1536_ba1_lds_xcd : large_bdirect_traits<64,256,2,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 2 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m64n256_w1x2_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(128),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_27) && !defined(OPUS_BP_INCLUDE_PERSISTENT_27_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_27_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_27 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_3;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m64n256_w1x2_k1536_ba2_lds_xcd : large_bdirect_traits<64,256,2,1536,65536,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 3 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m64n256_w1x2_k1536_ba2_lds_xcd>),dim3(grid,1,1),
dim3(128),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m64n256_w1x2_k1536_ba2_lds_xcd : large_bdirect_traits<64,256,2,1536,65536,2,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 3 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/64)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m64n256_w1x2_k1536_ba2_lds_xcd>),dim3(grid,1,1),
dim3(128),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_29) && !defined(OPUS_BP_INCLUDE_PERSISTENT_29_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_29_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_29 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_7;
using namespace opus_bpreshuffle_gfx950::persistent::schedule_28;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0





namespace large_geometry_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
template<class Traits>
__global__ void large_geometry_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0





namespace large_geometry_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void large_geometry_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void large_geometry_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
namespace panel_layout = large_geometry_layout;
const int stride_a = T::FIXED_K ? T::FIXED_K : kargs.stride_a;
const int stride_b = T::FIXED_K ? T::FIXED_K : kargs.stride_b;
const int stride_c = T::FIXED_N ? T::FIXED_N : kargs.stride_c;
const int stride_sfb = T::FIXED_K ? T::FIXED_K / 128 : kargs.stride_sfb;
const int output_n = T::FIXED_N ? T::FIXED_N : kargs.n;
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const int tiles_m = 1 + (kargs.m - 1) / T::B_M;
const int tiles_n = output_n / T::B_N;
const int tile_count = tiles_m * tiles_n;
alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
const int first_tile = T::RESIDENT_WORKERS ? block_id_x()
: block_id_y() * tiles_n + block_id_x();
auto process_output_tile = [&](int logical_tile) {
int block_m, block_n;
if constexpr (T::FIXED_N == 65536) {
const int group_size = tiles_m * T::SWIZZLE_GROUP_N;
const int first_n = (logical_tile / group_size) * T::SWIZZLE_GROUP_N;
const int actual_n = tiles_n - first_n < T::SWIZZLE_GROUP_N ? tiles_n - first_n : T::SWIZZLE_GROUP_N;
const int within_group = logical_tile % group_size;
block_m = within_group / actual_n;
block_n = first_n + within_group % actual_n;
} else {
const int group_size = T::SWIZZLE_GROUP_M * tiles_n;
const int first_m = (logical_tile / group_size) * T::SWIZZLE_GROUP_M;
const int actual_m = tiles_m - first_m < T::SWIZZLE_GROUP_M ? tiles_m - first_m : T::SWIZZLE_GROUP_M;
const int within_group = logical_tile % group_size;
block_m = first_m + within_group % actual_m;
block_n = within_group / actual_m;
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : kargs.k / T::B_K;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * stride_a, static_cast<unsigned>((kargs.m - row) * stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * stride_b);
const int remaining_rows = kargs.m - row;
const int valid_rows = remaining_rows < T::B_M ? remaining_rows : T::B_M;
const int64_t c_base_offset = static_cast<int64_t>(batch_id) * kargs.stride_c_batch
+ static_cast<int64_t>(row) * stride_c + col;
const unsigned c_bytes = static_cast<unsigned>(
((static_cast<int64_t>(valid_rows) - 1) * stride_c + T::B_N) * sizeof(D_C));
auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + c_base_offset, c_bytes);
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * stride_sfb);
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = panel_layout::make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = panel_layout::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_ssfa = panel_layout::make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
const auto u_rsfa = panel_layout::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_gsfb = panel_layout::make_layout_gsfb_scale<T>(lane_id, wave_id, stride_sfb);
const auto u_ssfb = panel_layout::make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = panel_layout::make_layout_rsfb_scale();
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
constexpr int ELEM_A = decltype(mma)::elem_a;
constexpr int ELEM_B = decltype(mma)::elem_b;
constexpr int ELEM_C = decltype(mma)::elem_c;
typename decltype(mma)::vtype_a v_a;
typename decltype(mma)::vtype_b v_b;
typename decltype(mma)::vtype_c v_c;
clear(v_c);
const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
const auto u_gc_direct = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(stride_c, 1_I), p_coord_c);
const auto gc_direct_offsets = layout_to_offsets<T::VEC_C>(u_gc_direct);
array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::B_M>{}; };
auto ssfb_offset = [&](int tile_k) { return tile_k; };
auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };
auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
static_assert(stage >= 0 && stage < T::NUM_STAGES);
if constexpr (T::B_FIRST) {
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
__builtin_amdgcn_sched_barrier(0);
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
}
};
const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : kargs.k / T::GROUP_K;
const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);
auto load_sfa_panel = [&](int panel_k_begin) {
const int local_k_group = wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_id / T::SFA_THREADS_PER_GROUP;
const int k_group = panel_k_begin + local_k_group;
static_for<T::SFA_PASSES>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int local_row = sfa_smem_offsets[pass] - local_k_group * T::B_M;
const int smem_offset = sfa_smem_offsets[pass] + (T::INTERLEAVED ? panel_k_begin * T::B_M : 0);
if ((!T::INTERLEAVED || local_row < T::B_M) && smem_offset < T::SFA_BYTES && k_group < scale_k_groups) {
vector_t<D_SF, T::VEC_SCALE_A> raw;
if (row + local_row < kargs.m)
raw = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
else
static_for<T::VEC_SCALE_A>([&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
}
});
};
auto load_sfb_panel = [&](int panel_k_begin) {
const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
if (k_group < scale_k_groups) {
unsigned packed_scales = 0;
static_for<T::SCALE_N_HALVES>([&](auto half_i) {
constexpr int half = decltype(half_i)::value;
const unsigned value = load<1>(g_sfb, sfb_gmem_offsets[half] + gsfb_offset(panel_k_begin))[0];
packed_scales |= value << (half * 8);
});
const vector_t<D_SF_PACK, 1> packed{packed_scales};
store<1>(s_sfb, packed, sfb_smem_offsets[0]);
}
};
auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
static_for<T::E_M>([&](auto m_i) {
constexpr int m_repeat = decltype(m_i)::value;
const unsigned value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
});
scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
};
auto load_a_fragment = [&](auto m_i, auto stage_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
});
};
auto load_b_fragment = [&](auto n_i, auto stage_i) {
constexpr int n_repeat = decltype(n_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
});
};
auto mma_scale_fragment = [&](auto m_i, auto n_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int n_repeat = decltype(n_i)::value;
constexpr int c_index = m_repeat * T::E_N + n_repeat;
constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
c = typename decltype(mma)::MMA{}(a, b, c,
static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[0]),
number<m_repeat % 4>{}, number<scale_n_index>{});
set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
};
auto advance_tile = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
constexpr int next_stage = (stage + 1) % T::NUM_STAGES;
if constexpr (T::SUPPLY_POLICY == 2) {
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<next_stage>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<next_stage>{}); });
} else {
if constexpr (T::SUPPLY_POLICY == 1) {
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
__builtin_amdgcn_sched_barrier(0);
}
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
mma_scale_fragment(m_i, n_i);
if constexpr (decltype(m_i)::value == T::E_M - 1)
load_b_fragment(n_i, number<next_stage>{});
});
if constexpr (T::SUPPLY_POLICY == 0 && decltype(m_i)::value == 0) {
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
__builtin_amdgcn_sched_barrier(0);
}
load_a_fragment(m_i, number<next_stage>{});
});
}
s_waitcnt_lgkmcnt(0_I);
v_sfa = v_sfa_next;
v_sfb = v_sfb_next;
};
if constexpr (T::INTERLEAVED) {
issue_matrix_prefetch(number<0>{}, 0);
__builtin_amdgcn_sched_barrier(0);
static_for<T::SFA_K_PASSES>([&](auto i) {
const int first_k = decltype(i)::value * T::SFA_K_PANEL;
if (first_k < scale_k_groups) load_sfa_panel(first_k);
});
load_sfb_panel(0);
__builtin_amdgcn_sched_barrier(0);
if (loops > 1) {
issue_matrix_prefetch(number<1>{}, 1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::MATRIX_VMEM_INSTRUCTIONS>{});
} else {
s_waitcnt_vmcnt(0_I);
}
} else {
load_sfa_panel(0);
load_sfb_panel(0);
issue_matrix_prefetch(number<0>{}, 0);
if (loops > 1)
issue_matrix_prefetch(number<1>{}, 1);
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(0, v_sfa, v_sfb);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<0>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<0>{}); });
s_waitcnt_lgkmcnt(0_I);
if constexpr (!T::INTERLEAVED)
__builtin_amdgcn_s_barrier();
for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
advance_tile(number<0>{}, tile_k);
if (tile_k + 2 < loops)
advance_tile(number<1>{}, tile_k + 1);
}
auto stage_output_fragment = [&](auto c_i) {
constexpr int c_index = decltype(c_i)::value;
const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
if constexpr (T::DIRECT_OUTPUT) {
store<T::VEC_C>(g_c, cast<D_C>(c), gc_direct_offsets[c_index], 0, number<T::OUTPUT_CACHE>{});
} else {
store<T::VEC_C>(s_c, cast<D_C>(c), gc_offsets[c_index]);
}
};
if constexpr (T::INTERLEAVED) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
constexpr int c_index = decltype(m_i)::value * T::E_N + decltype(n_i)::value;
mma_scale_fragment(m_i, n_i);
stage_output_fragment(number<c_index>{});
});
});
if constexpr (!T::DIRECT_OUTPUT) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
if constexpr (T::INTERLEAVED) {
auto copy_output_bf16 = [&](int copy_index) {
const int linear = output_thread_id * T::VEC_OUTPUT + copy_index * T::BLOCK_SIZE * T::VEC_OUTPUT;
const int output_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
if (row + output_row < kargs.m)
store<T::VEC_OUTPUT>(g_c, value, output_row * stride_c + output_col,
0, opus::number<T::OUTPUT_CACHE>{});
};
static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(decltype(copy_i)::value); });
} else {
auto copy_output_bf16 = [&](auto copy_i) {
constexpr int pass = decltype(copy_i)::value;
const int linear = output_thread_id * T::VEC_OUTPUT + pass * T::BLOCK_SIZE * T::VEC_OUTPUT;
const int output_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
if (row + output_row < kargs.m)
store<T::VEC_OUTPUT>(g_c, value, output_row * stride_c + output_col,
0, opus::number<T::OUTPUT_CACHE>{});
};
static_for<T::OUTPUT_PASSES>(copy_output_bf16);
}
}
};
if constexpr (T::RESIDENT_WORKERS > 0) {
for (int logical_tile = first_tile; logical_tile < tile_count; logical_tile += T::RESIDENT_WORKERS) {
process_output_tile(logical_tile);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
} else {
process_output_tile(first_tile);
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_30) && !defined(OPUS_BP_INCLUDE_PERSISTENT_30_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_30_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_30 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_long_w4_m64n256_k7168_lds : large_geometry_traits<64,256,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,0,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 7168 || !(N >= 1024 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_long_w4_m64n256_k7168_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_long_w4_m64n256_k7168_lds : large_geometry_traits<64,256,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,0,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 7168 || !(N >= 1024 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_long_w4_m64n256_k7168_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_31) && !defined(OPUS_BP_INCLUDE_PERSISTENT_31_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_31_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_31 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_long_w8_m128n256_k7168_lds : large_geometry_traits<128,256,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,0,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 7 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 7168 || !(N >= 1024 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_long_w8_m128n256_k7168_lds>),
dim3(N/256,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_long_w8_m128n256_k7168_lds : large_geometry_traits<128,256,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,0,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 7 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 7168 || !(N >= 1024 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_long_w8_m128n256_k7168_lds>),
dim3(N/256,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_32) && !defined(OPUS_BP_INCLUDE_PERSISTENT_32_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_32_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_32 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w4_m64n384_k7168_lds : large_geometry_traits<64,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 0 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/384);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w4_m64n384_k7168_lds>),
dim3(N/384,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w4_m64n384_k7168_lds : large_geometry_traits<64,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 0 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/384);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w4_m64n384_k7168_lds>),
dim3(N/384,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_33) && !defined(OPUS_BP_INCLUDE_PERSISTENT_33_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_33_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_33 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w4_m96n384_k7168_lds : large_geometry_traits<96,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 1 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/384);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w4_m96n384_k7168_lds>),
dim3(N/384,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w4_m96n384_k7168_lds : large_geometry_traits<96,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 1 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/384);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w4_m96n384_k7168_lds>),
dim3(N/384,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_34) && !defined(OPUS_BP_INCLUDE_PERSISTENT_34_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_34_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_34 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w4_m96n384_k7168_lds_p256 : large_geometry_traits<96,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,256,2,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 5 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/384);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w4_m96n384_k7168_lds_p256>),
dim3(256,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w4_m96n384_k7168_lds_p256 : large_geometry_traits<96,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,256,2,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 5 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/384);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w4_m96n384_k7168_lds_p256>),
dim3(256,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_35) && !defined(OPUS_BP_INCLUDE_PERSISTENT_35_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_35_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_35 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w8_m128n384_k7168_lds : large_geometry_traits<128,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 3 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/384);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w8_m128n384_k7168_lds>),
dim3(N/384,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w8_m128n384_k7168_lds : large_geometry_traits<128,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 3 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/384);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w8_m128n384_k7168_lds>),
dim3(N/384,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_36) && !defined(OPUS_BP_INCLUDE_PERSISTENT_36_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_36_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_36 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w8_m64n384_k7168_lds : large_geometry_traits<64,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 2 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/384);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w8_m64n384_k7168_lds>),
dim3(N/384,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w8_m64n384_k7168_lds : large_geometry_traits<64,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 2 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/384);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w8_m64n384_k7168_lds>),
dim3(N/384,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_37) && !defined(OPUS_BP_INCLUDE_PERSISTENT_37_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_37_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_37 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w8_m64n384_k7168_lds_p256 : large_geometry_traits<64,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,256,4,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 4 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/384);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w8_m64n384_k7168_lds_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_narrow_w8_m64n384_k7168_lds_p256 : large_geometry_traits<64,384,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,256,4,768,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 4 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 384 || K != 7168 || !(N == 768)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/384);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_narrow_w8_m64n384_k7168_lds_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_38) && !defined(OPUS_BP_INCLUDE_PERSISTENT_38_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_38_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_38 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m64n256_k384_direct : large_geometry_traits<64,256,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 9 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m64n256_k384_direct>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m64n256_k384_direct : large_geometry_traits<64,256,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 9 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m64n256_k384_direct>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_39) && !defined(OPUS_BP_INCLUDE_PERSISTENT_39_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_39_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_39 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m64n256_k384_lds : large_geometry_traits<64,256,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 8 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m64n256_k384_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m64n256_k384_lds : large_geometry_traits<64,256,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 8 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m64n256_k384_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_40) && !defined(OPUS_BP_INCLUDE_PERSISTENT_40_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_40_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_40 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m64n256_k768_direct : large_geometry_traits<64,256,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 13 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m64n256_k768_direct>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m64n256_k768_direct : large_geometry_traits<64,256,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 13 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m64n256_k768_direct>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_41) && !defined(OPUS_BP_INCLUDE_PERSISTENT_41_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_41_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_41 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m64n256_k768_lds : large_geometry_traits<64,256,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 12 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m64n256_k768_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m64n256_k768_lds : large_geometry_traits<64,256,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 12 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m64n256_k768_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_42) && !defined(OPUS_BP_INCLUDE_PERSISTENT_42_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_42_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_42 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m96n128_k384_direct : large_geometry_traits<96,128,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 25 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m96n128_k384_direct>),
dim3(N/128,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m96n128_k384_direct : large_geometry_traits<96,128,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 25 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m96n128_k384_direct>),
dim3(N/128,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_43) && !defined(OPUS_BP_INCLUDE_PERSISTENT_43_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_43_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_43 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m96n128_k384_lds : large_geometry_traits<96,128,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 24 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m96n128_k384_lds>),
dim3(N/128,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m96n128_k384_lds : large_geometry_traits<96,128,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 24 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m96n128_k384_lds>),
dim3(N/128,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_44) && !defined(OPUS_BP_INCLUDE_PERSISTENT_44_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_44_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_44 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m96n128_k768_direct : large_geometry_traits<96,128,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 29 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m96n128_k768_direct>),
dim3(N/128,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m96n128_k768_direct : large_geometry_traits<96,128,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 29 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m96n128_k768_direct>),
dim3(N/128,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_45) && !defined(OPUS_BP_INCLUDE_PERSISTENT_45_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_45_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_45 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m96n128_k768_lds : large_geometry_traits<96,128,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 28 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m96n128_k768_lds>),
dim3(N/128,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w4_m96n128_k768_lds : large_geometry_traits<96,128,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 28 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w4_m96n128_k768_lds>),
dim3(N/128,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_46) && !defined(OPUS_BP_INCLUDE_PERSISTENT_46_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_46_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_46 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m128n128_k384_direct : large_geometry_traits<128,128,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 27 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m128n128_k384_direct>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m128n128_k384_direct : large_geometry_traits<128,128,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 27 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m128n128_k384_direct>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_47) && !defined(OPUS_BP_INCLUDE_PERSISTENT_47_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_47_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_47 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m128n128_k384_lds : large_geometry_traits<128,128,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 26 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m128n128_k384_lds>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m128n128_k384_lds : large_geometry_traits<128,128,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 26 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m128n128_k384_lds>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_48) && !defined(OPUS_BP_INCLUDE_PERSISTENT_48_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_48_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_48 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m128n128_k768_direct : large_geometry_traits<128,128,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 31 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m128n128_k768_direct>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m128n128_k768_direct : large_geometry_traits<128,128,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 31 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m128n128_k768_direct>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_49) && !defined(OPUS_BP_INCLUDE_PERSISTENT_49_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_49_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_49 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m128n128_k768_lds : large_geometry_traits<128,128,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 30 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m128n128_k768_lds>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m128n128_k768_lds : large_geometry_traits<128,128,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 30 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 128 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m128n128_k768_lds>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_50) && !defined(OPUS_BP_INCLUDE_PERSISTENT_50_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_50_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_50 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m64n256_k384_direct : large_geometry_traits<64,256,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 11 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m64n256_k384_direct>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m64n256_k384_direct : large_geometry_traits<64,256,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 11 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m64n256_k384_direct>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_51) && !defined(OPUS_BP_INCLUDE_PERSISTENT_51_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_51_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_51 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m64n256_k384_lds : large_geometry_traits<64,256,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 10 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m64n256_k384_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m64n256_k384_lds : large_geometry_traits<64,256,16,384,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 10 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 384 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m64n256_k384_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_52) && !defined(OPUS_BP_INCLUDE_PERSISTENT_52_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_52_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_52 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m64n256_k768_direct : large_geometry_traits<64,256,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 15 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m64n256_k768_direct>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m64n256_k768_direct : large_geometry_traits<64,256,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 15 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m64n256_k768_direct>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_53) && !defined(OPUS_BP_INCLUDE_PERSISTENT_53_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_53_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_53 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m64n256_k768_lds : large_geometry_traits<64,256,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 14 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m64n256_k768_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_short_w8_m64n256_k768_lds : large_geometry_traits<64,256,16,768,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,7168,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 14 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 4096) || M % 16 || N <= 0 || N % 256 || K != 768 || !(N == 7168)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_short_w8_m64n256_k768_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_54) && !defined(OPUS_BP_INCLUDE_PERSISTENT_54_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_54_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_54 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w4_m32n256_k1536_lds : large_geometry_traits<32,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 16 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/32)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w4_m32n256_k1536_lds>),
dim3(N/256,1+(M-1)/32,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w4_m32n256_k1536_lds : large_geometry_traits<32,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 16 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/32)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w4_m32n256_k1536_lds>),
dim3(N/256,1+(M-1)/32,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_55) && !defined(OPUS_BP_INCLUDE_PERSISTENT_55_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_55_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_55 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w4_m64n256_k1536_lds : large_geometry_traits<64,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 17 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w4_m64n256_k1536_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w4_m64n256_k1536_lds : large_geometry_traits<64,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 17 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w4_m64n256_k1536_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_56) && !defined(OPUS_BP_INCLUDE_PERSISTENT_56_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_56_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_56 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w4_m96n256_k1536_direct : large_geometry_traits<96,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,65536,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 22 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w4_m96n256_k1536_direct>),
dim3(N/256,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w4_m96n256_k1536_direct : large_geometry_traits<96,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,65536,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 22 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w4_m96n256_k1536_direct>),
dim3(N/256,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_57) && !defined(OPUS_BP_INCLUDE_PERSISTENT_57_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_57_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_57 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w4_m96n256_k1536_lds : large_geometry_traits<96,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 18 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w4_m96n256_k1536_lds>),
dim3(N/256,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w4_m96n256_k1536_lds : large_geometry_traits<96,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,2,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 18 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/96)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w4_m96n256_k1536_lds>),
dim3(N/256,1+(M-1)/96,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_58) && !defined(OPUS_BP_INCLUDE_PERSISTENT_58_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_58_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_58 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w8_m128n256_k1536_direct : large_geometry_traits<128,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,65536,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 23 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w8_m128n256_k1536_direct>),
dim3(N/256,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w8_m128n256_k1536_direct : large_geometry_traits<128,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,65536,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 23 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w8_m128n256_k1536_direct>),
dim3(N/256,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_59) && !defined(OPUS_BP_INCLUDE_PERSISTENT_59_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_59_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_59 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w8_m128n256_k1536_lds : large_geometry_traits<128,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 20 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w8_m128n256_k1536_lds>),
dim3(N/256,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w8_m128n256_k1536_lds : large_geometry_traits<128,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 20 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w8_m128n256_k1536_lds>),
dim3(N/256,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_60) && !defined(OPUS_BP_INCLUDE_PERSISTENT_60_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_60_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_60 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w8_m192n256_k1536_lds : large_geometry_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 21 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w8_m192n256_k1536_lds>),
dim3(N/256,1+(M-1)/192,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w8_m192n256_k1536_lds : large_geometry_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 21 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w8_m192n256_k1536_lds>),
dim3(N/256,1+(M-1)/192,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_61) && !defined(OPUS_BP_INCLUDE_PERSISTENT_61_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_61_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_61 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_29;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w8_m64n256_k1536_lds : large_geometry_traits<64,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 19 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w8_m64n256_k1536_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_g281_wide_w8_m64n256_k1536_lds : large_geometry_traits<64,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0,4,65536,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 19 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/64)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N; a.stride_sfa=M; a.stride_sfb=K/128;
hipLaunchKernelGGL((large_geometry_kernel<traits_large_g281_wide_w8_m64n256_k1536_lds>),
dim3(N/256,1+(M-1)/64,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_63) && !defined(OPUS_BP_INCLUDE_PERSISTENT_63_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_63_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_63 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_7;
using namespace opus_bpreshuffle_gfx950::persistent::schedule_62;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0





namespace large_persistent_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
template<class Traits>
__global__ void large_group_persistent_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0





namespace large_persistent_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void large_group_persistent_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void large_group_persistent_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
namespace panel_layout = large_persistent_layout;
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const int tiles_m = 1 + (kargs.m - 1) / T::B_M;
const int tiles_n = kargs.n / T::B_N;
const int tile_count = tiles_m * tiles_n;
alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
const int first_tile = T::RESIDENT_WORKERS ? block_id_x()
: block_id_y() * tiles_n + block_id_x();
auto process_output_tile = [&](int logical_tile) {
int block_m, block_n;
if constexpr (T::FIXED_K == 1536) {
const int group_size = tiles_m * T::SWIZZLE_GROUP_N;
const int first_n = (logical_tile / group_size) * T::SWIZZLE_GROUP_N;
const int remaining_n = tiles_n - first_n;
const int actual_n = remaining_n < T::SWIZZLE_GROUP_N ? remaining_n : T::SWIZZLE_GROUP_N;
const int within_group = logical_tile % group_size;
block_m = within_group / actual_n;
block_n = first_n + within_group % actual_n;
} else {
const int group_size = T::SWIZZLE_GROUP_M * tiles_n;
const int first_m = (logical_tile / group_size) * T::SWIZZLE_GROUP_M;
const int remaining_m = tiles_m - first_m;
const int actual_m = remaining_m < T::SWIZZLE_GROUP_M ? remaining_m : T::SWIZZLE_GROUP_M;
const int within_group = logical_tile % group_size;
block_m = first_m + within_group % actual_m;
block_n = within_group / actual_m;
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : kargs.k / T::B_K;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
const int remaining_rows = kargs.m - row;
const int valid_rows = remaining_rows < T::B_M ? remaining_rows : T::B_M;
const int64_t c_base_offset = static_cast<int64_t>(batch_id) * kargs.stride_c_batch
+ static_cast<int64_t>(row) * kargs.stride_c + col;
const unsigned c_bytes = static_cast<unsigned>(
((static_cast<int64_t>(valid_rows) - 1) * kargs.stride_c + T::B_N) * sizeof(D_C));
auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + c_base_offset, c_bytes);
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = panel_layout::make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = panel_layout::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_ssfa = panel_layout::make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
const auto u_rsfa = panel_layout::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_gsfb = panel_layout::make_layout_gsfb_scale<T>(lane_id, wave_id, kargs.stride_sfb);
const auto u_ssfb = panel_layout::make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = panel_layout::make_layout_rsfb_scale();
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
constexpr int ELEM_A = decltype(mma)::elem_a;
constexpr int ELEM_B = decltype(mma)::elem_b;
constexpr int ELEM_C = decltype(mma)::elem_c;
typename decltype(mma)::vtype_a v_a;
typename decltype(mma)::vtype_b v_b;
typename decltype(mma)::vtype_c v_c;
clear(v_c);
const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::B_M>{}; };
auto ssfb_offset = [&](int tile_k) { return tile_k; };
auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };
auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
static_assert(stage >= 0 && stage < T::NUM_STAGES);
if constexpr (T::B_FIRST) {
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
__builtin_amdgcn_sched_barrier(0);
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
}
};
const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : kargs.k / T::GROUP_K;
const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);
auto load_sfa_panel = [&](int panel_k_begin) {
const int local_k_group = wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_id / T::SFA_THREADS_PER_GROUP;
const int k_group = panel_k_begin + local_k_group;
static_for<T::SFA_PASSES>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int local_row = sfa_smem_offsets[pass] - local_k_group * T::B_M;
const int smem_offset = sfa_smem_offsets[pass] + (T::INTERLEAVED ? panel_k_begin * T::B_M : 0);
if ((!T::INTERLEAVED || local_row < T::B_M) && smem_offset < T::SFA_BYTES && k_group < scale_k_groups) {
vector_t<D_SF, T::VEC_SCALE_A> raw;
if (row + local_row < kargs.m)
raw = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
else
static_for<T::VEC_SCALE_A>([&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
}
});
};
auto load_sfb_panel = [&](int panel_k_begin) {
const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
if (k_group < scale_k_groups) {
const unsigned lo = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin))[0];
unsigned hi = 0;
if constexpr (T::SCALE_N_HALVES == 2)
hi = load<1>(g_sfb, sfb_gmem_offsets[1] + gsfb_offset(panel_k_begin))[0];
const vector_t<D_SF_PACK, 1> packed{lo | (hi << 8)};
store<1>(s_sfb, packed, sfb_smem_offsets[0]);
}
};
auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
static_for<T::E_M>([&](auto m_i) {
constexpr int m_repeat = decltype(m_i)::value;
const unsigned value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
});
scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
};
auto load_a_fragment = [&](auto m_i, auto stage_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
});
};
auto load_b_fragment = [&](auto n_i, auto stage_i) {
constexpr int n_repeat = decltype(n_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
});
};
auto mma_scale_fragment = [&](auto m_i, auto n_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int n_repeat = decltype(n_i)::value;
constexpr int c_index = m_repeat * T::E_N + n_repeat;
constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
c = typename decltype(mma)::MMA{}(a, b, c,
static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[0]),
number<m_repeat % 4>{}, number<scale_n_index>{});
set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
};
auto advance_tile = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
constexpr int next_stage = (stage + 1) % T::NUM_STAGES;
if constexpr (T::SUPPLY_POLICY == 2) {
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<next_stage>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<next_stage>{}); });
} else {
if constexpr (T::SUPPLY_POLICY == 1) {
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
__builtin_amdgcn_sched_barrier(0);
}
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
mma_scale_fragment(m_i, n_i);
if constexpr (decltype(m_i)::value == T::E_M - 1)
load_b_fragment(n_i, number<next_stage>{});
});
if constexpr (T::SUPPLY_POLICY == 0 && decltype(m_i)::value == 0) {
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
__builtin_amdgcn_sched_barrier(0);
}
load_a_fragment(m_i, number<next_stage>{});
});
}
s_waitcnt_lgkmcnt(0_I);
v_sfa = v_sfa_next;
v_sfb = v_sfb_next;
};
if constexpr (T::INTERLEAVED) {
issue_matrix_prefetch(number<0>{}, 0);
__builtin_amdgcn_sched_barrier(0);
static_for<T::SFA_K_PASSES>([&](auto i) {
const int first_k = decltype(i)::value * T::SFA_K_PANEL;
if (first_k < scale_k_groups) load_sfa_panel(first_k);
});
load_sfb_panel(0);
__builtin_amdgcn_sched_barrier(0);
if (loops > 1) {
issue_matrix_prefetch(number<1>{}, 1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::MATRIX_VMEM_INSTRUCTIONS>{});
} else {
s_waitcnt_vmcnt(0_I);
}
} else {
load_sfa_panel(0);
load_sfb_panel(0);
issue_matrix_prefetch(number<0>{}, 0);
if (loops > 1)
issue_matrix_prefetch(number<1>{}, 1);
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(0, v_sfa, v_sfb);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<0>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<0>{}); });
s_waitcnt_lgkmcnt(0_I);
if constexpr (!T::INTERLEAVED)
__builtin_amdgcn_s_barrier();
for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
advance_tile(number<0>{}, tile_k);
if (tile_k + 2 < loops)
advance_tile(number<1>{}, tile_k + 1);
}
auto stage_output_fragment = [&](auto c_i) {
constexpr int c_index = decltype(c_i)::value;
const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
store<T::VEC_C>(s_c, cast<D_C>(c), gc_offsets[c_index]);
};
if constexpr (T::INTERLEAVED) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
constexpr int c_index = decltype(m_i)::value * T::E_N + decltype(n_i)::value;
mma_scale_fragment(m_i, n_i);
stage_output_fragment(number<c_index>{});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
if constexpr (T::INTERLEAVED) {
auto copy_output_bf16 = [&](int copy_index) {
const int linear = output_thread_id * T::VEC_OUTPUT + copy_index * T::BLOCK_SIZE * T::VEC_OUTPUT;
const int output_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
if (row + output_row < kargs.m)
store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col,
0, opus::number<2>{});
};
static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(decltype(copy_i)::value); });
} else {
auto copy_output_bf16 = [&](auto copy_i) {
constexpr int pass = decltype(copy_i)::value;
const int linear = output_thread_id * T::VEC_OUTPUT + pass * T::BLOCK_SIZE * T::VEC_OUTPUT;
const int output_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
if (row + output_row < kargs.m)
store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col,
0, opus::number<2>{});
};
static_for<T::OUTPUT_PASSES>(copy_output_bf16);
}
};
if constexpr (T::RESIDENT_WORKERS > 0) {
for (int logical_tile = first_tile; logical_tile < tile_count; logical_tile += T::RESIDENT_WORKERS) {
process_output_tile(logical_tile);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
} else {
process_output_tile(first_tile);
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_64) && !defined(OPUS_BP_INCLUDE_PERSISTENT_64_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_64_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_64 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_group_only_q16 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,16,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 9 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_group_only_q16>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_group_only_q16 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,16,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 9 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_group_only_q16>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_65) && !defined(OPUS_BP_INCLUDE_PERSISTENT_65_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_65_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_65 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_group_only_q32 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,32,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 10 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_group_only_q32>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_group_only_q32 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,32,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 10 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_group_only_q32>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_66) && !defined(OPUS_BP_INCLUDE_PERSISTENT_66_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_66_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_66 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_group_only_q8 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 8 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_group_only_q8>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_group_only_q8 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 8 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_group_only_q8>),
dim3(N/128,1+(M-1)/128,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_67) && !defined(OPUS_BP_INCLUDE_PERSISTENT_67_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_67_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_67 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q16_p128 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,16,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 14 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q16_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q16_p128 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,16,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 14 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q16_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_68) && !defined(OPUS_BP_INCLUDE_PERSISTENT_68_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_68_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_68 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q16_p256 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,16,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 15 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q16_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q16_p256 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,16,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 15 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q16_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_69) && !defined(OPUS_BP_INCLUDE_PERSISTENT_69_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_69_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_69 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q16_p512 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,16,8192,1,true,512> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 16 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (512 > 0 && tile_count <= 512)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q16_p512>),
dim3(512,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q16_p512 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,16,8192,1,true,512> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 16 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (512 > 0 && tile_count <= 512)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q16_p512>),
dim3(512,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_70) && !defined(OPUS_BP_INCLUDE_PERSISTENT_70_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_70_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_70 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q32_p128 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,32,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 17 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q32_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q32_p128 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,32,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 17 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q32_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_71) && !defined(OPUS_BP_INCLUDE_PERSISTENT_71_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_71_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_71 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q32_p256 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,32,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 18 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q32_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q32_p256 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,32,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 18 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q32_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_72) && !defined(OPUS_BP_INCLUDE_PERSISTENT_72_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_72_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_72 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q32_p512 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,32,8192,1,true,512> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 19 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (512 > 0 && tile_count <= 512)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q32_p512>),
dim3(512,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q32_p512 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,32,8192,1,true,512> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 19 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (512 > 0 && tile_count <= 512)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q32_p512>),
dim3(512,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_73) && !defined(OPUS_BP_INCLUDE_PERSISTENT_73_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_73_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_73 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q8_p128 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 11 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q8_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q8_p128 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 11 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q8_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_74) && !defined(OPUS_BP_INCLUDE_PERSISTENT_74_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_74_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_74 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q8_p256 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 12 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q8_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q8_p256 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 12 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q8_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_75) && !defined(OPUS_BP_INCLUDE_PERSISTENT_75_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_75_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_75 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q8_p512 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,512> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 13 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (512 > 0 && tile_count <= 512)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q8_p512>),
dim3(512,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_runtime_persistent_q8_p512 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,512> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 13 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 128 || K <= 0 || K % 128 || !(N <= 8192 && N >= 128 && K <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (512 > 0 && tile_count <= 512)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_runtime_persistent_q8_p512>),
dim3(512,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_76) && !defined(OPUS_BP_INCLUDE_PERSISTENT_76_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_76_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_76 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_group_only_q16 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,16,4,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 1 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_group_only_q16>),
dim3(N/256,1+(M-1)/192,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_group_only_q16 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,16,4,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 1 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_group_only_q16>),
dim3(N/256,1+(M-1)/192,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_77) && !defined(OPUS_BP_INCLUDE_PERSISTENT_77_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_77_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_77 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_group_only_q8 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 0 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_group_only_q8>),
dim3(N/256,1+(M-1)/192,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_group_only_q8 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,0> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 0 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (0 > 0 && tile_count <= 0)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_group_only_q8>),
dim3(N/256,1+(M-1)/192,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_78) && !defined(OPUS_BP_INCLUDE_PERSISTENT_78_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_78_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_78 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q16_p128 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,16,4,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 4 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q16_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q16_p128 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,16,4,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 4 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q16_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_79) && !defined(OPUS_BP_INCLUDE_PERSISTENT_79_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_79_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_79 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q16_p256 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,16,4,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 5 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q16_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q16_p256 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,16,4,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 5 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q16_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_80) && !defined(OPUS_BP_INCLUDE_PERSISTENT_80_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_80_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_80 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q32_p128 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q32_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q32_p128 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q32_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_81) && !defined(OPUS_BP_INCLUDE_PERSISTENT_81_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_81_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_81 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q32_p256 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 7 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q32_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q32_p256 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 7 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q32_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_82) && !defined(OPUS_BP_INCLUDE_PERSISTENT_82_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_82_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_82 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q8_p128 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 2 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q8_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q8_p128 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,128> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 2 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (128 > 0 && tile_count <= 128)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q8_p128>),
dim3(128,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_83) && !defined(OPUS_BP_INCLUDE_PERSISTENT_83_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_83_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_83 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_63;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q8_p256 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 3 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q8_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_gp_wide_persistent_q8_p256 : large_persistent_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,8,4,8192,1,true,256> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 3 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 128 || !(N == 65536 && K == 1536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/192)*(N/256);
if (tile_count > 2147483647LL || (256 > 0 && tile_count <= 256)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_group_persistent_kernel<traits_large_gp_wide_persistent_q8_p256>),
dim3(256,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_85) && !defined(OPUS_BP_INCLUDE_PERSISTENT_85_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_85_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_85 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_7;
using namespace opus_bpreshuffle_gfx950::persistent::schedule_84;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0





namespace large_m_8wave_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
template<class Traits>
__global__ void large_m_supply_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0





namespace large_m_8wave_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void large_m_supply_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void large_m_supply_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
namespace panel_layout = large_m_8wave_layout;
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
int block_m = block_id_y();
int block_n = block_id_x();
if constexpr (T::INTERLEAVED) {
if (T::ENABLE_N_SWIZZLE && kargs.n / T::B_N >= T::SWIZZLE_MIN_N_TILES && kargs.m > T::B_M) {
const int grid_m = 1 + (kargs.m - 1) / T::B_M;
const int grid_n = kargs.n / T::B_N;
const int linear = block_id_y() * grid_n + block_id_x();
const int group_size = grid_m * T::SWIZZLE_GROUP_N;
const int first_n = (linear / group_size) * T::SWIZZLE_GROUP_N;
const int remaining_n = grid_n - first_n;
const int actual_n = remaining_n < T::SWIZZLE_GROUP_N ? remaining_n : T::SWIZZLE_GROUP_N;
const int within_group = linear % group_size;
block_m = within_group / actual_n;
block_n = first_n + within_group % actual_n;
} else if (kargs.n <= T::SWIZZLE_MAX_N && kargs.m >= T::SWIZZLE_MIN_M) {
const int grid_m = 1 + (kargs.m - 1) / T::B_M;
const int grid_n = kargs.n / T::B_N;
const int linear = block_id_y() * grid_n + block_id_x();
const int group_size = T::SWIZZLE_GROUP_M * grid_n;
const int first_m = (linear / group_size) * T::SWIZZLE_GROUP_M;
const int remaining_m = grid_m - first_m;
const int actual_m = remaining_m < T::SWIZZLE_GROUP_M ? remaining_m : T::SWIZZLE_GROUP_M;
const int within_group = linear % group_size;
block_m = first_m + within_group % actual_m;
block_n = within_group / actual_m;
}
} else {
if constexpr (T::ENABLE_N_SWIZZLE) {
if (kargs.n / T::B_N >= T::SWIZZLE_MIN_N_TILES && kargs.m > T::B_M) {
const int grid_m = 1 + (kargs.m - 1) / T::B_M;
const int grid_n = kargs.n / T::B_N;
const int linear = block_id_y() * grid_n + block_id_x();
const int group_size = grid_m * T::SWIZZLE_GROUP_N;
const int first_n = (linear / group_size) * T::SWIZZLE_GROUP_N;
const int remaining_n = grid_n - first_n;
const int actual_n = remaining_n < T::SWIZZLE_GROUP_N ? remaining_n : T::SWIZZLE_GROUP_N;
const int within_group = linear % group_size;
block_m = within_group / actual_n;
block_n = first_n + within_group % actual_n;
}
}
if (kargs.n <= T::SWIZZLE_MAX_N && kargs.m >= T::SWIZZLE_MIN_M) {
const int grid_m = 1 + (kargs.m - 1) / T::B_M;
const int grid_n = kargs.n / T::B_N;
const int linear = block_id_y() * grid_n + block_id_x();
const int group_size = T::SWIZZLE_GROUP_M * grid_n;
const int first_m = (linear / group_size) * T::SWIZZLE_GROUP_M;
const int remaining_m = grid_m - first_m;
const int actual_m = remaining_m < T::SWIZZLE_GROUP_M ? remaining_m : T::SWIZZLE_GROUP_M;
const int within_group = linear % group_size;
block_m = first_m + within_group % actual_m;
block_n = within_group / actual_m;
}
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : kargs.k / T::B_K;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
const int remaining_rows = kargs.m - row;
const int valid_rows = remaining_rows < T::B_M ? remaining_rows : T::B_M;
const int64_t c_base_offset = static_cast<int64_t>(batch_id) * kargs.stride_c_batch
+ static_cast<int64_t>(row) * kargs.stride_c + col;
const unsigned c_bytes = static_cast<unsigned>(
((static_cast<int64_t>(valid_rows) - 1) * kargs.stride_c + T::B_N) * sizeof(D_C));
auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + c_base_offset, c_bytes);
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = panel_layout::make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = panel_layout::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_ssfa = panel_layout::make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
const auto u_rsfa = panel_layout::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_gsfb = panel_layout::make_layout_gsfb_scale<T>(lane_id, wave_id, kargs.stride_sfb);
const auto u_ssfb = panel_layout::make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = panel_layout::make_layout_rsfb_scale();
alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
constexpr int ELEM_A = decltype(mma)::elem_a;
constexpr int ELEM_B = decltype(mma)::elem_b;
constexpr int ELEM_C = decltype(mma)::elem_c;
typename decltype(mma)::vtype_a v_a;
typename decltype(mma)::vtype_b v_b;
typename decltype(mma)::vtype_c v_c;
clear(v_c);
const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::B_M>{}; };
auto ssfb_offset = [&](int tile_k) { return tile_k; };
auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };
auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
static_assert(stage >= 0 && stage < T::NUM_STAGES);
if constexpr (T::B_FIRST) {
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
__builtin_amdgcn_sched_barrier(0);
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
}
};
const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : kargs.k / T::GROUP_K;
const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);
auto load_sfa_panel = [&](int panel_k_begin) {
const int local_k_group = wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_id / T::SFA_THREADS_PER_GROUP;
const int k_group = panel_k_begin + local_k_group;
static_for<T::SFA_PASSES>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int local_row = sfa_smem_offsets[pass] - local_k_group * T::B_M;
const int smem_offset = sfa_smem_offsets[pass] + (T::INTERLEAVED ? panel_k_begin * T::B_M : 0);
if ((!T::INTERLEAVED || local_row < T::B_M) && smem_offset < T::SFA_BYTES && k_group < scale_k_groups) {
vector_t<D_SF, T::VEC_SCALE_A> raw;
if (row + local_row < kargs.m)
raw = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
else
static_for<T::VEC_SCALE_A>([&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
}
});
};
auto load_sfb_panel = [&](int panel_k_begin) {
const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
if (k_group < scale_k_groups) {
const unsigned lo = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin))[0];
unsigned hi = 0;
if constexpr (T::SCALE_N_HALVES == 2)
hi = load<1>(g_sfb, sfb_gmem_offsets[1] + gsfb_offset(panel_k_begin))[0];
const vector_t<D_SF_PACK, 1> packed{lo | (hi << 8)};
store<1>(s_sfb, packed, sfb_smem_offsets[0]);
}
};
auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
static_for<T::E_M>([&](auto m_i) {
constexpr int m_repeat = decltype(m_i)::value;
const unsigned value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
});
scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
};
auto load_a_fragment = [&](auto m_i, auto stage_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
});
};
auto load_b_fragment = [&](auto n_i, auto stage_i) {
constexpr int n_repeat = decltype(n_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
});
};
auto mma_scale_fragment = [&](auto m_i, auto n_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int n_repeat = decltype(n_i)::value;
constexpr int c_index = m_repeat * T::E_N + n_repeat;
constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
c = typename decltype(mma)::MMA{}(a, b, c,
static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[0]),
number<m_repeat % 4>{}, number<scale_n_index>{});
set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
};
auto advance_tile = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
constexpr int next_stage = (stage + 1) % T::NUM_STAGES;
if constexpr (T::SUPPLY_POLICY == 2) {
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<next_stage>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<next_stage>{}); });
} else {
if constexpr (T::SUPPLY_POLICY == 1) {
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
__builtin_amdgcn_sched_barrier(0);
}
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
mma_scale_fragment(m_i, n_i);
if constexpr (decltype(m_i)::value == T::E_M - 1)
load_b_fragment(n_i, number<next_stage>{});
});
if constexpr (T::SUPPLY_POLICY == 0 && decltype(m_i)::value == 0) {
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
__builtin_amdgcn_sched_barrier(0);
}
load_a_fragment(m_i, number<next_stage>{});
});
}
s_waitcnt_lgkmcnt(0_I);
v_sfa = v_sfa_next;
v_sfb = v_sfb_next;
};
if constexpr (T::INTERLEAVED) {
issue_matrix_prefetch(number<0>{}, 0);
__builtin_amdgcn_sched_barrier(0);
static_for<T::SFA_K_PASSES>([&](auto i) {
const int first_k = decltype(i)::value * T::SFA_K_PANEL;
if (first_k < scale_k_groups) load_sfa_panel(first_k);
});
load_sfb_panel(0);
__builtin_amdgcn_sched_barrier(0);
if (loops > 1) {
issue_matrix_prefetch(number<1>{}, 1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::MATRIX_VMEM_INSTRUCTIONS>{});
} else {
s_waitcnt_vmcnt(0_I);
}
} else {
load_sfa_panel(0);
load_sfb_panel(0);
issue_matrix_prefetch(number<0>{}, 0);
if (loops > 1)
issue_matrix_prefetch(number<1>{}, 1);
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(0, v_sfa, v_sfb);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<0>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<0>{}); });
s_waitcnt_lgkmcnt(0_I);
if constexpr (!T::INTERLEAVED)
__builtin_amdgcn_s_barrier();
for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
advance_tile(number<0>{}, tile_k);
if (tile_k + 2 < loops)
advance_tile(number<1>{}, tile_k + 1);
}
auto stage_output_fragment = [&](auto c_i) {
constexpr int c_index = decltype(c_i)::value;
const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
store<T::VEC_C>(s_c, cast<D_C>(c), gc_offsets[c_index]);
};
if constexpr (T::INTERLEAVED) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
constexpr int c_index = decltype(m_i)::value * T::E_N + decltype(n_i)::value;
mma_scale_fragment(m_i, n_i);
stage_output_fragment(number<c_index>{});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
if constexpr (T::INTERLEAVED) {
auto copy_output_bf16 = [&](int copy_index) {
const int linear = output_thread_id * T::VEC_OUTPUT + copy_index * T::BLOCK_SIZE * T::VEC_OUTPUT;
const int output_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
if (row + output_row < kargs.m)
store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col,
0, opus::number<2>{});
};
static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(decltype(copy_i)::value); });
} else {
auto copy_output_bf16 = [&](auto copy_i) {
constexpr int pass = decltype(copy_i)::value;
const int linear = output_thread_id * T::VEC_OUTPUT + pass * T::BLOCK_SIZE * T::VEC_OUTPUT;
const int output_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
if (row + output_row < kargs.m)
store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col,
0, opus::number<2>{});
};
static_for<T::OUTPUT_PASSES>(copy_output_bf16);
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_86) && !defined(OPUS_BP_INCLUDE_PERSISTENT_86_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_86_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_86 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_narrow128_ab_late : large_m_8wave_traits<128,128,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K == 7168 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_narrow128_ab_late>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_narrow128_ab_late : large_m_8wave_traits<128,128,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K == 7168 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_narrow128_ab_late>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_87) && !defined(OPUS_BP_INCLUDE_PERSISTENT_87_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_87_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_87 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_narrow128_bfirst_early : large_m_8wave_traits<128,128,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 5 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K == 7168 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_narrow128_bfirst_early>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_narrow128_bfirst_early : large_m_8wave_traits<128,128,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 5 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K == 7168 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_narrow128_bfirst_early>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_88) && !defined(OPUS_BP_INCLUDE_PERSISTENT_88_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_88_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_88 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_narrow128_bfirst_mid : large_m_8wave_traits<128,128,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,0,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 4 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K == 7168 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_narrow128_bfirst_mid>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_narrow128_bfirst_mid : large_m_8wave_traits<128,128,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,0,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 4 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K == 7168 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_narrow128_bfirst_mid>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_89) && !defined(OPUS_BP_INCLUDE_PERSISTENT_89_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_89_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_89 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_narrow192_bfirst_late : large_m_8wave_traits<192,128,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 7 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K == 7168 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_narrow192_bfirst_late>),
dim3(N/128,1+(M-1)/192,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_narrow192_bfirst_late : large_m_8wave_traits<192,128,64,7168,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 7 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K == 7168 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_narrow192_bfirst_late>),
dim3(N/128,1+(M-1)/192,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_90) && !defined(OPUS_BP_INCLUDE_PERSISTENT_90_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_90_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_90 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_runtime128_ab_late : large_m_8wave_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 9 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K <= 8192 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_runtime128_ab_late>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_runtime128_ab_late : large_m_8wave_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 9 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K <= 8192 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_runtime128_ab_late>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_91) && !defined(OPUS_BP_INCLUDE_PERSISTENT_91_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_91_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_91 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_runtime128_bfirst_early : large_m_8wave_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 8 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K <= 8192 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_runtime128_bfirst_early>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_runtime128_bfirst_early : large_m_8wave_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 8 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 128 || !(K <= 8192 && N >= 128 && N <= 8192)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_runtime128_bfirst_early>),
dim3(N/128,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_92) && !defined(OPUS_BP_INCLUDE_PERSISTENT_92_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_92_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_92 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_wide128_bfirst_late : large_m_8wave_traits<128,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 3 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 256 || !(K == 1536 && N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_wide128_bfirst_late>),
dim3(N/256,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_wide128_bfirst_late : large_m_8wave_traits<128,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 3 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 256 || !(K == 1536 && N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_wide128_bfirst_late>),
dim3(N/256,1+(M-1)/128,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_93) && !defined(OPUS_BP_INCLUDE_PERSISTENT_93_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_93_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_93 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_wide192_ab_late : large_m_8wave_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 2 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 256 || !(K == 1536 && N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_wide192_ab_late>),
dim3(N/256,1+(M-1)/192,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_wide192_ab_late : large_m_8wave_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,2,false> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 2 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 256 || !(K == 1536 && N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_wide192_ab_late>),
dim3(N/256,1+(M-1)/192,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_94) && !defined(OPUS_BP_INCLUDE_PERSISTENT_94_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_94_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_94 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_wide192_bfirst_early : large_m_8wave_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 1 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 256 || !(K == 1536 && N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_wide192_bfirst_early>),
dim3(N/256,1+(M-1)/192,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_wide192_bfirst_early : large_m_8wave_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,1,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 1 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 256 || !(K == 1536 && N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_wide192_bfirst_early>),
dim3(N/256,1+(M-1)/192,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_95) && !defined(OPUS_BP_INCLUDE_PERSISTENT_95_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_95_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_95 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_85;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_wide192_bfirst_mid : large_m_8wave_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,0,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 0 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 256 || !(K == 1536 && N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_wide192_bfirst_mid>),
dim3(N/256,1+(M-1)/192,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_m_wide192_bfirst_mid : large_m_8wave_traits<192,256,16,1536,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,4,8192,0,true> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 0 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M < 4096 || M % 16 || K <= 0 || K % 128 || N % 256 || !(K == 1536 && N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((large_m_supply_kernel<traits_large_m_wide192_bfirst_mid>),
dim3(N/256,1+(M-1)/192,1),dim3(512),0,
reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_96) && !defined(OPUS_BP_INCLUDE_PERSISTENT_96_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_96_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_96 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct large_shortK_pipeline_traits {
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
static_assert(BN==256);
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
__global__ void large_shortK_pipeline_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedM,int FixedN,int FixedK,int BCache=0>
struct large_shortK_pipeline_traits {
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
static_assert(BN==256);
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
__global__ void large_shortK_pipeline_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void large_shortK_pipeline_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
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

#if defined(OPUS_BP_INCLUDE_PERSISTENT_97) && !defined(OPUS_BP_INCLUDE_PERSISTENT_97_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_97_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_97 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=large_shortK_pipeline_traits<96,256,4,1,10240,7168,384,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
large_shortK_pipeline_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=large_shortK_pipeline_traits<96,256,4,1,10240,7168,384,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
large_shortK_pipeline_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
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

#if defined(OPUS_BP_INCLUDE_PERSISTENT_98) && !defined(OPUS_BP_INCLUDE_PERSISTENT_98_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_98_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_98 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=large_shortK_pipeline_traits<96,256,6,1,57344,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
large_shortK_pipeline_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=large_shortK_pipeline_traits<96,256,6,1,57344,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
large_shortK_pipeline_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
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

#if defined(OPUS_BP_INCLUDE_PERSISTENT_99) && !defined(OPUS_BP_INCLUDE_PERSISTENT_99_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_99_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_99 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_96;
using namespace opus_bpreshuffle_gfx950::host_contract::schedule_0;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


using Traits=large_shortK_pipeline_traits<96,256,3,1,65536,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
large_shortK_pipeline_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
0,reinterpret_cast<hipStream_t>(raw_stream)>>>(args);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


using Traits=large_shortK_pipeline_traits<96,256,3,1,65536,7168,768,0>;
int launch_config(
int selector,int literal,const void* a,const void* b,const void* sfa,const void* sfb,
void* out,void* ws,uint64_t bytes,int m,int n,int k,void* raw_stream) {
if(selector!=0 || literal!=1 || m!=Traits::FIXED_M || n!=Traits::STATIC_N || k!=Traits::FIXED_K)return -11;
FixedMContract contract{m,n,k,1,4,128,1,false};
const int checked=fixed_m_validate(contract,0,a,b,out,sfa,sfb,ws,bytes,m,n,k);
if(checked)return checked;
auto args=fixed_m_args(contract,a,b,out,sfa,sfb,ws);
const unsigned total=((m+Traits::B_M-1)/Traits::B_M)*(n/Traits::B_N);
large_shortK_pipeline_kernel<Traits><<<dim3(total,1,1),dim3(Traits::BLOCK_SIZE),
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

#if defined(OPUS_BP_INCLUDE_PERSISTENT_100) && !defined(OPUS_BP_INCLUDE_PERSISTENT_100_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_100_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_100 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_7;
using namespace opus_bpreshuffle_gfx950::support::schedule_2;
using namespace opus_bpreshuffle_gfx950::persistent::schedule_62;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0






namespace large_persistent_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
template<class Traits>
__global__ void actual_persistent_100_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0






namespace large_persistent_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void actual_persistent_100_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void actual_persistent_100_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
namespace panel_layout = large_persistent_layout;
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const int tiles_m = 1 + (6144 - 1) / T::B_M;
const int tiles_n = 7168 / T::B_N;
const int tile_count = tiles_m * tiles_n;
constexpr int EXACT_SFA_BYTES=T::B_M*(384/128);
constexpr int EXACT_LDS_BYTES=(T::MATRIX_LDS_BYTES+EXACT_SFA_BYTES+T::SFB_BYTES+255)/256*256;
alignas(16) __shared__ char smem_matrix[EXACT_LDS_BYTES];
const int first_tile = T::RESIDENT_WORKERS ? block_id_x()
: block_id_y() * tiles_n + block_id_x();
auto process_output_tile = [&](int logical_tile) {
int block_m, block_n;
if constexpr (T::FIXED_K == 1536) {
const int group_size = tiles_m * T::SWIZZLE_GROUP_N;
const int first_n = (logical_tile / group_size) * T::SWIZZLE_GROUP_N;
const int remaining_n = tiles_n - first_n;
const int actual_n = remaining_n < T::SWIZZLE_GROUP_N ? remaining_n : T::SWIZZLE_GROUP_N;
const int within_group = logical_tile % group_size;
block_m = within_group / actual_n;
block_n = first_n + within_group % actual_n;
} else {
const int group_size = T::SWIZZLE_GROUP_M * tiles_n;
const int first_m = (logical_tile / group_size) * T::SWIZZLE_GROUP_M;
const int remaining_m = tiles_m - first_m;
const int actual_m = remaining_m < T::SWIZZLE_GROUP_M ? remaining_m : T::SWIZZLE_GROUP_M;
const int within_group = logical_tile % group_size;
block_m = first_m + within_group % actual_m;
block_n = within_group / actual_m;
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : 384 / T::B_K;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * 0 + row * 384, static_cast<unsigned>((6144 - row) * 384));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * 0 + col * 384);
const int remaining_rows = 6144 - row;
const int valid_rows = remaining_rows < T::B_M ? remaining_rows : T::B_M;
const int64_t c_base_offset = static_cast<int64_t>(batch_id) * 0
+ static_cast<int64_t>(row) * 7168 + col;
const unsigned c_bytes = static_cast<unsigned>(
((static_cast<int64_t>(valid_rows) - 1) * 7168 + T::B_N) * sizeof(D_C));
auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + c_base_offset, c_bytes);
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * 0 + row, static_cast<unsigned>(6144*(384/128)-row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * 0 + (col / T::GROUP_N) * 3);
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, 384);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = panel_layout::make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, 384);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = panel_layout::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, 6144);
const auto u_ssfa = panel_layout::make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
const auto u_rsfa = panel_layout::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_gsfb = panel_layout::make_layout_gsfb_scale<T>(lane_id, wave_id, 3);
const auto u_ssfb = panel_layout::make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = panel_layout::make_layout_rsfb_scale();
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + T::MATRIX_LDS_BYTES + EXACT_SFA_BYTES));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
constexpr int ELEM_A = decltype(mma)::elem_a;
constexpr int ELEM_B = decltype(mma)::elem_b;
constexpr int ELEM_C = decltype(mma)::elem_c;
typename decltype(mma)::vtype_a v_a;
typename decltype(mma)::vtype_b v_b;
typename decltype(mma)::vtype_c v_c;
clear(v_c);
const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * 6144; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::B_M>{}; };
auto ssfb_offset = [&](int tile_k) { return tile_k; };
auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };
auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
static_assert(stage >= 0 && stage < T::NUM_STAGES);
if constexpr (T::B_FIRST) {
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
__builtin_amdgcn_sched_barrier(0);
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
}
};
const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : 384 / T::GROUP_K;
const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);
auto load_sfa_panel = [&](int panel_k_begin) {
static_for<(T::B_M*(384/128)+T::BLOCK_SIZE*4-1)/(T::BLOCK_SIZE*4)>([&](auto pi) {
constexpr int p=decltype(pi)::value;
const int index=(p*T::BLOCK_SIZE+thread_id_x())*4;
if(index<T::B_M*(384/128)) {
const int r=index%T::B_M,kg=index/T::B_M;
const int source=row+r<6144?(panel_k_begin+kg)*6144+r:6144*(384/128)-row;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(s_sfa.ptr+(p*T::BLOCK_SIZE+wave_id*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(g_sfa.cached_rsrc,dst,4,source,0,0,0);
}
});
};
auto load_sfb_panel = [&](int panel_k_begin) {
const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
if (k_group < scale_k_groups) {
const unsigned lo = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin))[0];
unsigned hi = 0;
if constexpr (T::SCALE_N_HALVES == 2)
hi = load<1>(g_sfb, sfb_gmem_offsets[1] + gsfb_offset(panel_k_begin))[0];
const vector_t<D_SF_PACK, 1> packed{lo | (hi << 8)};
store<1>(s_sfb, packed, sfb_smem_offsets[0]);
}
};
auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
static_for<T::E_M>([&](auto m_i) {
constexpr int m_repeat = decltype(m_i)::value;
const unsigned value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
});
scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
};
auto load_a_fragment = [&](auto m_i, auto stage_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
});
};
auto load_b_fragment = [&](auto n_i, auto stage_i) {
constexpr int n_repeat = decltype(n_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
});
};
auto mma_scale_fragment = [&](auto m_i, auto n_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int n_repeat = decltype(n_i)::value;
constexpr int c_index = m_repeat * T::E_N + n_repeat;
constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
c = typename decltype(mma)::MMA{}(a, b, c,
static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[0]),
number<m_repeat % 4>{}, number<scale_n_index>{});
set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
};
auto advance_tile = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
constexpr int next_stage = (stage + 1) % T::NUM_STAGES;
if constexpr (T::SUPPLY_POLICY == 2) {
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<next_stage>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<next_stage>{}); });
} else {
if constexpr (T::SUPPLY_POLICY == 1) {
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
__builtin_amdgcn_sched_barrier(0);
}
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
mma_scale_fragment(m_i, n_i);
if constexpr (decltype(m_i)::value == T::E_M - 1)
load_b_fragment(n_i, number<next_stage>{});
});
if constexpr (T::SUPPLY_POLICY == 0 && decltype(m_i)::value == 0) {
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (tile_k + 2 < loops) issue_matrix_prefetch(stage_i, tile_k + 2);
__builtin_amdgcn_sched_barrier(0);
}
load_a_fragment(m_i, number<next_stage>{});
});
}
s_waitcnt_lgkmcnt(0_I);
v_sfa = v_sfa_next;
v_sfb = v_sfb_next;
};
if constexpr (T::INTERLEAVED) {
issue_matrix_prefetch(number<0>{}, 0);
__builtin_amdgcn_sched_barrier(0);
static_for<T::SFA_K_PASSES>([&](auto i) {
const int first_k = decltype(i)::value * T::SFA_K_PANEL;
if (first_k < scale_k_groups) load_sfa_panel(first_k);
});
load_sfb_panel(0);
__builtin_amdgcn_sched_barrier(0);
if (loops > 1) {
issue_matrix_prefetch(number<1>{}, 1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<T::MATRIX_VMEM_INSTRUCTIONS>{});
} else {
s_waitcnt_vmcnt(0_I);
}
} else {
load_sfa_panel(0);
load_sfb_panel(0);
issue_matrix_prefetch(number<0>{}, 0);
if (loops > 1)
issue_matrix_prefetch(number<1>{}, 1);
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(0, v_sfa, v_sfb);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<0>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<0>{}); });
s_waitcnt_lgkmcnt(0_I);
if constexpr (!T::INTERLEAVED)
__builtin_amdgcn_s_barrier();
for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
advance_tile(number<0>{}, tile_k);
if (tile_k + 2 < loops)
advance_tile(number<1>{}, tile_k + 1);
}
s_waitcnt_lgkmcnt(0_I);
static_for<T::E_M>([&](auto mi) {
static_for<T::E_N>([&](auto ni) { mma_scale_fragment(mi,ni); });
});
array<vector_t<float,4>,T::E_M*T::E_N> output;
static_for<T::E_M*T::E_N>([&](auto fi) {
constexpr int ci=decltype(fi)::value;
output[ci]=slice(v_c,number<ci*4>{},number<(ci+1)*4>{});
});
actual188_mfma16_output::store_mfma16_packed<T>(output,g_c,lane_id,wave_id_m,wave_id_n,7168);
};
if constexpr (T::RESIDENT_WORKERS > 0) {
for (int logical_tile = first_tile; logical_tile < tile_count; logical_tile += T::RESIDENT_WORKERS) {
process_output_tile(logical_tile);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
} else {
process_output_tile(first_tile);
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_101) && !defined(OPUS_BP_INCLUDE_PERSISTENT_101_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_101_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_101 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_100;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0



struct traits_large_gp_runtime_persistent_q8_p512 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,512> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 100 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M != 6144 || N != 7168 || K != 384) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (512 > 0 && tile_count <= 512)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((actual_persistent_100_kernel<traits_large_gp_runtime_persistent_q8_p512>),
dim3(512,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0



struct traits_large_gp_runtime_persistent_q8_p512 : large_persistent_traits<128,128,64,0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
true,32,8,8192,1,true,512> {};
int launch_config(
int selector, int literalSplit, const void* A, const void* B,
const void* SFA, const void* SFB, void* C, void* WS, uint64_t bytes,
int M, int N, int K, void* stream) {
if (selector != 100 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (M != 6144 || N != 7168 || K != 384) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)
| reinterpret_cast<uintptr_t>(SFA) | reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const int64_t tile_count=int64_t(1+(M-1)/128)*(N/128);
if (tile_count > 2147483647LL || (512 > 0 && tile_count <= 512)) return -6;
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A; a.ptr_b=B; a.ptr_sfa=SFA; a.ptr_sfb=SFB; a.ptr_c=C;
a.m=M; a.n=N; a.k=K; a.batch=1;
a.stride_a=K; a.stride_b=K; a.stride_c=N;
a.stride_a_batch=0; a.stride_b_batch=0; a.stride_c_batch=0;
a.stride_sfa=M; a.stride_sfb=K/128;
a.stride_sfa_batch=0; a.stride_sfb_batch=0;
hipLaunchKernelGGL((actual_persistent_100_kernel<traits_large_gp_runtime_persistent_q8_p512>),
dim3(512,1,1),
dim3(512),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_102) && !defined(OPUS_BP_INCLUDE_PERSISTENT_102_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_102_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_102 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_69;
using namespace opus_bpreshuffle_gfx950::tiled::schedule_70;
using namespace opus_bpreshuffle_gfx950::tiled::schedule_71;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0







template<class Traits>
__global__ void opus_gemm_mxscale_bpreshuffle_large_output_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0







#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void opus_gemm_mxscale_bpreshuffle_large_output_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void opus_gemm_mxscale_bpreshuffle_large_output_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
static_assert(T::SCHEDULE == 0 || T::SCHEDULE == 1);
constexpr bool DIRECT_B = T::SCHEDULE == 1;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
namespace layout_9020 = opus_gemm_8wave_192x256_layout;
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
int block_m = block_id_y();
int block_n = block_id_x();
if (kargs.n <= T::SWIZZLE_MAX_N && kargs.m >= T::SWIZZLE_MIN_M) {
const int grid_m = 1 + (kargs.m - 1) / T::B_M;
const int grid_n = kargs.n / T::B_N;
const int linear = block_id_y() * grid_n + block_id_x();
const int group_size = T::SWIZZLE_GROUP_M * grid_n;
const int first_m = (linear / group_size) * T::SWIZZLE_GROUP_M;
const int remaining_m = grid_m - first_m;
const int actual_m = remaining_m < T::SWIZZLE_GROUP_M ? remaining_m : T::SWIZZLE_GROUP_M;
const int within_group = linear % group_size;
block_m = first_m + within_group % actual_m;
block_n = within_group / actual_m;
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : kargs.k / T::B_K;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
const int remaining_rows = kargs.m - row;
const int valid_rows = remaining_rows < T::B_M ? remaining_rows : T::B_M;
const int64_t c_base_offset = static_cast<int64_t>(batch_id) * kargs.stride_c_batch + static_cast<int64_t>(row) * kargs.stride_c + col;
const unsigned c_bytes = static_cast<unsigned>(
((static_cast<int64_t>(valid_rows) - 1) * kargs.stride_c + T::B_N) * sizeof(D_C));
auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + c_base_offset, c_bytes);
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = layout_9020::make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = layout_9020::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_ssfa = layout_9020::make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
const auto u_rsfa = layout_9020::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_gsfb = layout_9020::make_layout_gsfb_scale<T>(lane_id, wave_id, kargs.stride_sfb);
const auto u_ssfb = layout_9020::make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = layout_9020::make_layout_rsfb_scale();
alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
constexpr int ELEM_A = decltype(mma)::elem_a;
constexpr int ELEM_B = decltype(mma)::elem_b;
constexpr int ELEM_C = decltype(mma)::elem_c;
typename decltype(mma)::vtype_a v_a;
using BRegisters = std::conditional_t<DIRECT_B, vector_t<D_B, ELEM_B>, typename decltype(mma)::vtype_b>;
BRegisters v_b;
vector_t<D_B, ELEM_B> v_b_next;
typename decltype(mma)::vtype_c v_c;
clear(v_c);
const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::B_M>{}; };
auto ssfb_offset = [&](int tile_k) { return tile_k; };
auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };
auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {
const int stage = [&] {
if constexpr (DIRECT_B) return stage_i;
else return decltype(stage_i)::value;
}();
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
if constexpr (!DIRECT_B) {
static_assert(decltype(stage_i)::value >= 0 && decltype(stage_i)::value < T::NUM_STAGES);
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
}
};
const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : kargs.k / T::GROUP_K;
const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);
auto load_sfa_panel = [&](int panel_k_begin) {
const int local_k_group = wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_id / T::SFA_THREADS_PER_GROUP;
const int k_group = panel_k_begin + local_k_group;
static_for<T::SFA_PASSES>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int smem_offset = sfa_smem_offsets[pass];
const int local_row = smem_offset - local_k_group * T::B_M;
if ((!T::FIXED_K || local_row < T::B_M) && smem_offset < T::SFA_BYTES && k_group < scale_k_groups) {
vector_t<D_SF, T::VEC_SCALE_A> raw;
if (row + local_row < kargs.m)
raw = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
else
static_for<T::VEC_SCALE_A>([&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
}
});
};
auto load_sfb_panel = [&](int panel_k_begin) {
const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
if (k_group < scale_k_groups) {
const unsigned lo = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin))[0];
const unsigned hi = load<1>(g_sfb, sfb_gmem_offsets[1] + gsfb_offset(panel_k_begin))[0];
const vector_t<D_SF_PACK, 1> packed{lo | (hi << 8)};
store<1>(s_sfb, packed, sfb_smem_offsets[0]);
}
};
auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
static_for<T::A_SCALE_PACKS>([&](auto pack_i) { scale_a[decltype(pack_i)::value] = 0; });
static_for<T::E_M>([&](auto m_i) {
constexpr int m_repeat = decltype(m_i)::value;
const unsigned value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
});
scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
};
auto load_a_fragment = [&](auto m_i, auto stage_i) {
constexpr int m_repeat = decltype(m_i)::value;
const int stage = [&] {
if constexpr (DIRECT_B) return stage_i;
else return decltype(stage_i)::value;
}();
const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
});
};
auto load_b_fragment = [&](auto n_i, auto stage_i) {
constexpr int n_repeat = decltype(n_i)::value;
constexpr int stage = decltype(stage_i)::value;
const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
});
};
auto load_b_direct = [&](auto& target, int tile_k, auto n_i) {
constexpr int n_repeat = decltype(n_i)::value;
const int group = n_repeat * T::T_N + wave_id_n;
const int base = group * T::W_N * kargs.stride_b + tile_k * T::B_K * T::W_N + lane_id * T::VEC_B;
static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int chunk = decltype(chunk_i)::value;
set_slice(target, load<T::VEC_B>(g_b, base + chunk * T::WARP_SIZE * T::VEC_B),
number<chunk * T::VEC_B>{}, number<(chunk + 1) * T::VEC_B>{});
});
};
auto mma_scale_fragment = [&](auto m_i, auto n_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int n_repeat = decltype(n_i)::value;
constexpr int c_index = m_repeat * T::E_N + n_repeat;
constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
const auto b = [&] {
if constexpr (DIRECT_B) return v_b;
else return slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
}();
auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
c = typename decltype(mma)::MMA{}(a, b, c,
static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[0]),
number<m_repeat % 4>{}, number<scale_n_index>{});
set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
};
if constexpr (DIRECT_B) {
auto compute_tile = [&](int tile_k) {
static_for<T::E_N>([&](auto n_i) {
constexpr int n_repeat = decltype(n_i)::value;
static_for<T::E_M>([&](auto m_i) { mma_scale_fragment(m_i, n_i); });
if constexpr (n_repeat + 1 < T::E_N) {
s_waitcnt_vmcnt(0_I);
v_b = v_b_next;
if constexpr (n_repeat + 2 < T::E_N)
load_b_direct(v_b_next, tile_k, number<n_repeat + 2>{});
else if (tile_k + 1 < loops)
load_b_direct(v_b_next, tile_k + 1, number<0>{});
} else if (tile_k + 1 < loops) {
s_waitcnt_vmcnt(0_I);
v_b = v_b_next;
load_b_direct(v_b_next, tile_k + 1, number<1>{});
}
});
};
load_sfa_panel(0);
load_sfb_panel(0);
issue_matrix_prefetch(0, 0);
issue_matrix_prefetch(1, 1);
load_b_direct(v_b, 0, number<0>{});
load_b_direct(v_b_next, 0, number<1>{});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(0, v_sfa, v_sfb);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
#pragma unroll 1
for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
if (tile_k + 2 < loops)
issue_matrix_prefetch((tile_k + 2) % T::NUM_STAGES, tile_k + 2);
compute_tile(tile_k);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, (tile_k + 1) % T::NUM_STAGES); });
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
v_sfa = v_sfa_next;
v_sfb = v_sfb_next;
}
compute_tile(loops - 1);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<T::C_CHUNKS>([&](auto chunk_i) {
constexpr int first_row = decltype(chunk_i)::value * T::C_CHUNK_ROWS;
static_for<T::E_M * T::E_N>([&](auto c_i) {
constexpr int c_index = decltype(c_i)::value;
const int offset = gc_offsets[c_index];
const int output_row = offset / T::C_LDS_ROW_STRIDE_ELEMS;
if (output_row >= first_row && output_row < first_row + T::C_CHUNK_ROWS) {
const auto value = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
store<T::VEC_C>(s_c, cast<D_C>(value), offset - first_row * T::C_LDS_ROW_STRIDE_ELEMS);
}
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
static_for<T::CHUNK_OUTPUT_PASSES>([&](auto pass_i) {
const int linear = output_thread_id * T::VEC_OUTPUT + decltype(pass_i)::value * T::BLOCK_SIZE * T::VEC_OUTPUT;
const int local_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(local_row, output_col));
const int output_row = first_row + local_row;
if (row + output_row < kargs.m)
store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col, 0, number<2>{});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
});
} else {
auto advance_tile = [&](auto stage_i, int tile_k) {
constexpr int stage = decltype(stage_i)::value;
constexpr int next_stage = (stage + 1) % T::NUM_STAGES;
if (tile_k + 2 < loops)
issue_matrix_prefetch(stage_i, tile_k + 2);
read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
mma_scale_fragment(m_i, n_i);
if constexpr (decltype(m_i)::value == T::E_M - 1)
load_b_fragment(n_i, number<next_stage>{});
});
load_a_fragment(m_i, number<next_stage>{});
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
v_sfa = v_sfa_next;
v_sfb = v_sfb_next;
};
load_sfa_panel(0);
load_sfb_panel(0);
issue_matrix_prefetch(number<0>{}, 0);
if (loops > 1)
issue_matrix_prefetch(number<1>{}, 1);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(0, v_sfa, v_sfb);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<0>{}); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<0>{}); });
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
advance_tile(number<0>{}, tile_k);
if (tile_k + 2 < loops)
advance_tile(number<1>{}, tile_k + 1);
}
auto stage_output_fragment = [&](auto c_i) {
constexpr int c_index = decltype(c_i)::value;
const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
store<T::VEC_C>(s_c, cast<D_C>(c), gc_offsets[c_index]);
};
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) {
constexpr int c_index = decltype(m_i)::value * T::E_N + decltype(n_i)::value;
mma_scale_fragment(m_i, n_i);
stage_output_fragment(number<c_index>{});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
auto copy_output_bf16 = [&](auto copy_i) {
constexpr int pass = decltype(copy_i)::value;
const int linear = output_thread_id * T::VEC_OUTPUT + pass * T::BLOCK_SIZE * T::VEC_OUTPUT;
const int output_row = linear / T::B_N;
const int output_col = linear % T::B_N;
const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
if (row + output_row < kargs.m)
store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col,
0, opus::number<2>{});
};
static_for<T::OUTPUT_PASSES>(copy_output_bf16);
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_103) && !defined(OPUS_BP_INCLUDE_PERSISTENT_103_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_103_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_103 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_69;
using namespace opus_bpreshuffle_gfx950::persistent::schedule_102;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0








using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536_Traits = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_large_config_traits<16, 1536, false, 0>, 0>;
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(k <= 16384, entry, ": requires K <= 16384");
AITER_CHECK(k == 1536, entry, ": requires K == 1536");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t input_byte_limit = 2147483647;
constexpr int64_t output_byte_limit = 9223372036854775807;
AITER_CHECK(m <= input_byte_limit / k && n <= input_byte_limit / k,
entry, ": A/B tensor byte extent exceeds signed 32-bit addressing");
AITER_CHECK(m > (input_byte_limit / sizeof(D_C)) / n &&
m <= (output_byte_limit / sizeof(D_C)) / n,
entry, ": large-output kernel requires C beyond signed 32-bit byte extent");
AITER_CHECK(191 * n + 256 <= input_byte_limit / sizeof(D_C),
entry, ": output tile byte span exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ must be 16-byte aligned; Y must be 16-byte aligned");
AITER_CHECK(reinterpret_cast<uintptr_t>(x_scale.data_ptr()) % 16 == 0,
entry, ": x_scale must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = 0;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 191) / 192;
const dim3 grid(n / 256, tiles_m);
opus_gemm_mxscale_bpreshuffle_large_output_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536_Traits><<<
grid, dim3(512), 0, aiter::getCurrentHIPStream()>>>(args);
}
#else

// SPDX-License-Identifier: Apache-2.0
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)






#endif
#ifdef OPUS_FUSED_HOST_TU


#else


#endif
using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536_Traits = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_large_config_traits<16, 1536, false, 0>, 0>;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(k <= 16384, entry, ": requires K <= 16384");
AITER_CHECK(k == 1536, entry, ": requires K == 1536");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t input_byte_limit = 2147483647;
constexpr int64_t output_byte_limit = 9223372036854775807;
AITER_CHECK(m <= input_byte_limit / k && n <= input_byte_limit / k,
entry, ": A/B tensor byte extent exceeds signed 32-bit addressing");
AITER_CHECK(m > (input_byte_limit / sizeof(D_C)) / n &&
m <= (output_byte_limit / sizeof(D_C)) / n,
entry, ": large-output kernel requires C beyond signed 32-bit byte extent");
AITER_CHECK(191 * n + 256 <= input_byte_limit / sizeof(D_C),
entry, ": output tile byte span exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ must be 16-byte aligned; Y must be 16-byte aligned");
AITER_CHECK(reinterpret_cast<uintptr_t>(x_scale.data_ptr()) % 16 == 0,
entry, ": x_scale must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = 0;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 191) / 192;
const dim3 grid(n / 256, tiles_m);
opus_gemm_mxscale_bpreshuffle_large_output_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536_Traits><<<
grid, dim3(512), 0, aiter::getCurrentHIPStream()>>>(args);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_104) && !defined(OPUS_BP_INCLUDE_PERSISTENT_104_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_104_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_104 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_103;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 103, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M % 64 == 0, "requires m_align");
AITER_CHECK(splitK == 0, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
#if !defined(__HIP_DEVICE_COMPILE__)
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 103, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M % 64 == 0, "requires m_align");
AITER_CHECK(splitK == 0, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_panel16_k1536<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif

#endif
}
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template __global__ void opus_bpreshuffle_gfx950::persistent::schedule_102::opus_gemm_mxscale_bpreshuffle_large_output_kernel<
opus_bpreshuffle_gfx950::tiled::schedule_69::opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_bpreshuffle_gfx950::tiled::schedule_69::opus_gemm_mxscale_bpreshuffle_large_config_traits<16, 1536, false, 0>, 0>>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
#endif

#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_105) && !defined(OPUS_BP_INCLUDE_PERSISTENT_105_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_105_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_105 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_69;
using namespace opus_bpreshuffle_gfx950::persistent::schedule_102;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0








using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536_Traits = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_large_config_traits<16, 1536, true, 96>, 1>;
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(k <= 16384, entry, ": requires K <= 16384");
AITER_CHECK(k == 1536, entry, ": requires K == 1536");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t input_byte_limit = 2147483647;
constexpr int64_t output_byte_limit = 9223372036854775807;
AITER_CHECK(m <= input_byte_limit / k && n <= input_byte_limit / k,
entry, ": A/B tensor byte extent exceeds signed 32-bit addressing");
AITER_CHECK(m > (input_byte_limit / sizeof(D_C)) / n &&
m <= (output_byte_limit / sizeof(D_C)) / n,
entry, ": large-output kernel requires C beyond signed 32-bit byte extent");
AITER_CHECK(191 * n + 256 <= input_byte_limit / sizeof(D_C),
entry, ": output tile byte span exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ must be 16-byte aligned; Y must be 16-byte aligned");
AITER_CHECK(reinterpret_cast<uintptr_t>(x_scale.data_ptr()) % 16 == 0,
entry, ": x_scale must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = 0;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 191) / 192;
const dim3 grid(n / 256, tiles_m);
opus_gemm_mxscale_bpreshuffle_large_output_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536_Traits><<<
grid, dim3(512), 0, aiter::getCurrentHIPStream()>>>(args);
}
#else

// SPDX-License-Identifier: Apache-2.0
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)






#endif
#ifdef OPUS_FUSED_HOST_TU


#else


#endif
using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536_Traits = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_large_config_traits<16, 1536, true, 96>, 1>;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(k <= 16384, entry, ": requires K <= 16384");
AITER_CHECK(k == 1536, entry, ": requires K == 1536");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t input_byte_limit = 2147483647;
constexpr int64_t output_byte_limit = 9223372036854775807;
AITER_CHECK(m <= input_byte_limit / k && n <= input_byte_limit / k,
entry, ": A/B tensor byte extent exceeds signed 32-bit addressing");
AITER_CHECK(m > (input_byte_limit / sizeof(D_C)) / n &&
m <= (output_byte_limit / sizeof(D_C)) / n,
entry, ": large-output kernel requires C beyond signed 32-bit byte extent");
AITER_CHECK(191 * n + 256 <= input_byte_limit / sizeof(D_C),
entry, ": output tile byte span exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ must be 16-byte aligned; Y must be 16-byte aligned");
AITER_CHECK(reinterpret_cast<uintptr_t>(x_scale.data_ptr()) % 16 == 0,
entry, ": x_scale must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = 0;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 191) / 192;
const dim3 grid(n / 256, tiles_m);
opus_gemm_mxscale_bpreshuffle_large_output_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536_Traits><<<
grid, dim3(512), 0, aiter::getCurrentHIPStream()>>>(args);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_106) && !defined(OPUS_BP_INCLUDE_PERSISTENT_106_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_106_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_106 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_105;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 104, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M % 64 == 0, "requires m_align");
AITER_CHECK(splitK == 0, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
#if !defined(__HIP_DEVICE_COMPILE__)
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 104, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M % 64 == 0, "requires m_align");
AITER_CHECK(splitK == 0, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_large_output_direct_b_k1536<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif

#endif
}
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template __global__ void opus_bpreshuffle_gfx950::persistent::schedule_102::opus_gemm_mxscale_bpreshuffle_large_output_kernel<
opus_bpreshuffle_gfx950::tiled::schedule_69::opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_bpreshuffle_gfx950::tiled::schedule_69::opus_gemm_mxscale_bpreshuffle_large_config_traits<16, 1536, true, 96>, 1>>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
#endif

#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_107) && !defined(OPUS_BP_INCLUDE_PERSISTENT_107_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_107_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_107 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_69;
using namespace opus_bpreshuffle_gfx950::persistent::schedule_102;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0








using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_Traits = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_large_config_traits<128, 0, false, 0>, 0>;
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(k <= 16384, entry, ": requires K <= 16384");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t input_byte_limit = 2147483647;
constexpr int64_t output_byte_limit = 9223372036854775807;
AITER_CHECK(m <= input_byte_limit / k && n <= input_byte_limit / k,
entry, ": A/B tensor byte extent exceeds signed 32-bit addressing");
AITER_CHECK(m > (input_byte_limit / sizeof(D_C)) / n &&
m <= (output_byte_limit / sizeof(D_C)) / n,
entry, ": large-output kernel requires C beyond signed 32-bit byte extent");
AITER_CHECK(191 * n + 256 <= input_byte_limit / sizeof(D_C),
entry, ": output tile byte span exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ/Y must be 16-byte aligned");
AITER_CHECK(reinterpret_cast<uintptr_t>(x_scale.data_ptr()) % 16 == 0,
entry, ": x_scale must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = 0;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 191) / 192;
const dim3 grid(n / 256, tiles_m);
opus_gemm_mxscale_bpreshuffle_large_output_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_Traits><<<
grid, dim3(512), 0, aiter::getCurrentHIPStream()>>>(args);
}
#else

// SPDX-License-Identifier: Apache-2.0
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)






#endif
#ifdef OPUS_FUSED_HOST_TU


#else


#endif
using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_Traits = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_large_config_traits<128, 0, false, 0>, 0>;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(k <= 16384, entry, ": requires K <= 16384");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t input_byte_limit = 2147483647;
constexpr int64_t output_byte_limit = 9223372036854775807;
AITER_CHECK(m <= input_byte_limit / k && n <= input_byte_limit / k,
entry, ": A/B tensor byte extent exceeds signed 32-bit addressing");
AITER_CHECK(m > (input_byte_limit / sizeof(D_C)) / n &&
m <= (output_byte_limit / sizeof(D_C)) / n,
entry, ": large-output kernel requires C beyond signed 32-bit byte extent");
AITER_CHECK(191 * n + 256 <= input_byte_limit / sizeof(D_C),
entry, ": output tile byte span exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ/Y must be 16-byte aligned");
AITER_CHECK(reinterpret_cast<uintptr_t>(x_scale.data_ptr()) % 16 == 0,
entry, ": x_scale must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = 0;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 191) / 192;
const dim3 grid(n / 256, tiles_m);
opus_gemm_mxscale_bpreshuffle_large_output_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1_Traits><<<
grid, dim3(512), 0, aiter::getCurrentHIPStream()>>>(args);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_108) && !defined(OPUS_BP_INCLUDE_PERSISTENT_108_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_108_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_108 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_107;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 9, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M % 64 == 0, "requires m_align");
AITER_CHECK(splitK == 0, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
#if !defined(__HIP_DEVICE_COMPILE__)
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 9, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M % 64 == 0, "requires m_align");
AITER_CHECK(splitK == 0, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_512x192x256x128_4x2_16x16x128_1x128x128_tiles1<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif

#endif
}
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template __global__ void opus_bpreshuffle_gfx950::persistent::schedule_102::opus_gemm_mxscale_bpreshuffle_large_output_kernel<
opus_bpreshuffle_gfx950::tiled::schedule_69::opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_bpreshuffle_gfx950::tiled::schedule_69::opus_gemm_mxscale_bpreshuffle_large_config_traits<128, 0, false, 0>, 0>>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
#endif

#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_109) && !defined(OPUS_BP_INCLUDE_PERSISTENT_109_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_109_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_109 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0




template<int BM,int BN,int WavesN,int FixedK,int FixedN,int Ahead,bool Direct,bool XCD>
struct large_bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=WavesN;
static constexpr int BLOCK_SIZE=WavesN*64,NUM_WAVES=WavesN,WARP_SIZE=64;
static constexpr int FIXED_K=FixedK,FIXED_N=FixedN,LOOPS=FixedK/128;
static constexpr int E_M=BM/16,E_N=BN/(WavesN*16);
static constexpr int B_AHEAD=Ahead,B_SETS=Ahead+1,NUM_STAGES=B_SETS+1;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
static constexpr int SA_PASSES=(BM*LOOPS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SA_BYTES=SA_PASSES*BLOCK_SIZE*4;
static constexpr int SB_PASSES=((BN/128)*LOOPS+BLOCK_SIZE-1)/BLOCK_SIZE;
static constexpr int SB_BYTES=SB_PASSES*BLOCK_SIZE*4;
static constexpr int OFF_SA=NUM_STAGES*A_STAGE,OFF_SB=OFF_SA+SA_BYTES;
static constexpr int MATRIX_BYTES=OFF_SB+SB_BYTES;
static constexpr int C_ROWS=(BM>128 || BN>256) ? BM/2 : BM;
static constexpr int C_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=Direct || MATRIX_BYTES>C_BYTES ? MATRIX_BYTES:C_BYTES;
static constexpr int OUTPUT_PASSES=(C_ROWS*BN+BLOCK_SIZE*8-1)/(BLOCK_SIZE*8);
static constexpr int UNROLL=NUM_STAGES*B_SETS;
static constexpr int OUTPUT_CACHE=2;
static constexpr bool DIRECT_OUTPUT=Direct,XCD_ORDER=XCD;
static_assert(BM%16==0 && BN%128==0 && BN%(WavesN*16)==0);
static_assert(WavesN==2 || WavesN==4);
static_assert(Ahead==1 || Ahead==2);
static_assert(LOOPS>=B_SETS && FixedK%128==0);
static_assert(C_ROWS%16==0 && BM%C_ROWS==0);
static_assert(LDS_BYTES<=160*1024);
};
template<class T>
__global__ void large_bdirect_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0




template<int BM,int BN,int WavesN,int FixedK,int FixedN,int Ahead,bool Direct,bool XCD>
struct large_bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=WavesN;
static constexpr int BLOCK_SIZE=WavesN*64,NUM_WAVES=WavesN,WARP_SIZE=64;
static constexpr int FIXED_K=FixedK,FIXED_N=FixedN,LOOPS=FixedK/128;
static constexpr int E_M=BM/16,E_N=BN/(WavesN*16);
static constexpr int B_AHEAD=Ahead,B_SETS=Ahead+1,NUM_STAGES=B_SETS+1;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
static constexpr int SA_PASSES=(BM*LOOPS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SA_BYTES=SA_PASSES*BLOCK_SIZE*4;
static constexpr int SB_PASSES=((BN/128)*LOOPS+BLOCK_SIZE-1)/BLOCK_SIZE;
static constexpr int SB_BYTES=SB_PASSES*BLOCK_SIZE*4;
static constexpr int OFF_SA=NUM_STAGES*A_STAGE,OFF_SB=OFF_SA+SA_BYTES;
static constexpr int MATRIX_BYTES=OFF_SB+SB_BYTES;
static constexpr int C_ROWS=(BM>128 || BN>256) ? BM/2 : BM;
static constexpr int C_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=Direct || MATRIX_BYTES>C_BYTES ? MATRIX_BYTES:C_BYTES;
static constexpr int OUTPUT_PASSES=(C_ROWS*BN+BLOCK_SIZE*8-1)/(BLOCK_SIZE*8);
static constexpr int UNROLL=NUM_STAGES*B_SETS;
static constexpr int OUTPUT_CACHE=2;
static constexpr bool DIRECT_OUTPUT=Direct,XCD_ORDER=XCD;
static_assert(BM%16==0 && BN%128==0 && BN%(WavesN*16)==0);
static_assert(WavesN==2 || WavesN==4);
static_assert(Ahead==1 || Ahead==2);
static_assert(LOOPS>=B_SETS && FixedK%128==0);
static_assert(C_ROWS%16==0 && BM%C_ROWS==0);
static_assert(LDS_BYTES<=160*1024);
};
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void large_bdirect_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE,1)
void large_bdirect_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int K=T::FIXED_K,BM=T::B_M,BN=T::B_N,THREADS=T::BLOCK_SIZE;
constexpr int MR=T::E_M,NR=T::E_N,BSETS=T::B_SETS,ASTAGES=T::NUM_STAGES;
constexpr int TILES=T::LOOPS;
const int M=args.m,N=T::FIXED_N ? T::FIXED_N:args.n;
const unsigned tiles_n=static_cast<unsigned>(N)/BN;
const unsigned total=((static_cast<unsigned>(M)+BM-1)/BM)*tiles_n;
const unsigned pid=static_cast<unsigned>(block_id_x());
const unsigned group=(total+7u)/8u;
const unsigned tile=T::XCD_ORDER ? (pid%8u)*group+pid/8u:pid;
if (tile>=total) return;
const int row=(tile/tiles_n)*BM,col=(tile%tiles_n)*BN;
const int valid_rows=M-row<BM ? M-row:BM;
const unsigned tid=thread_id_x();
__builtin_assume(tid<T::BLOCK_SIZE);
const unsigned wave=__builtin_amdgcn_readfirstlane(tid/64);
const int lane=tid%64,lane16=lane%16,kgroup=lane/16;
const uint64_t a_extent=static_cast<uint64_t>(valid_rows)*K;
const uint64_t sa_extent=static_cast<uint64_t>(M)*TILES;
const uint64_t sb_extent=static_cast<uint64_t>(N/128)*TILES;
const uint64_t c_extent=(static_cast<uint64_t>(valid_rows-1)*N+BN)*2;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+static_cast<uint64_t>(row)*K,
static_cast<uint32_t>(a_extent));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+static_cast<uint64_t>(col)*K,
static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<uint32_t>(sa_extent));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<uint32_t>(sb_extent));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+static_cast<uint64_t>(row)*N+col,
static_cast<uint32_t>(c_extent));
alignas(16) __shared__ char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));
auto ssa=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SA));
auto ssb=make_smem(reinterpret_cast<unsigned char*>(arena+T::OFF_SB));
using Base=mfma<fp8_t,fp8_t,fp32_t,16,16,128>;
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BSETS> bregs;
static_for<T::SA_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int index=(pass*THREADS+tid)*4,r=index%BM,kt=index/BM;
const uint64_t src64=index<BM*TILES && row+r<M ? static_cast<uint64_t>(kt)*M+row+r:sa_extent;
const int src=__builtin_bit_cast(int,static_cast<uint32_t>(src64));
async_load<4>(gsa,ssa.ptr+(pass*THREADS+wave*64)*4,src);
});
static_for<T::SB_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int index=pass*THREADS+tid;
const uint64_t src64=index<(BN/128)*TILES
? static_cast<uint64_t>(col/128+index/TILES)*TILES+index%TILES:sb_extent;
const int src=__builtin_bit_cast(int,static_cast<uint32_t>(src64));
async_load<1>(gsb,ssb.ptr+(pass*THREADS+wave*64)*4,src);
});
__builtin_amdgcn_sched_barrier(0);
auto issue_a=[&](auto slot_i,int kt) {
constexpr int slot=decltype(slot_i)::value;
static_for<T::A_PASSES>([&](auto pi) {
constexpr int pass=decltype(pi)::value;
const int lin=(pass*THREADS+tid)*16,r=lin/128;
const int kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(
reinterpret_cast<__UINTPTR_TYPE__>(sa.ptr+slot*T::A_STAGE+(pass*THREADS+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
};
auto issue_b_column=[&](auto slot_i,auto ni,int kt) {
constexpr int slot=decltype(slot_i)::value,n=decltype(ni)::value;
const int nr=wave*(BN/T::T_N)+n*16,offset=nr*K+lane*16;
set_slice(bregs[slot][n],load<16>(gb,offset,kt*128*16),0_I,16_I);
set_slice(bregs[slot][n],load<16>(gb,offset+1024,kt*128*16),16_I,32_I);
__builtin_amdgcn_sched_barrier(0);
};
auto issue_b=[&](auto slot,int kt) {
static_for<NR>([&](auto ni){issue_b_column(slot,ni,kt);});
};
static_for<BSETS>([&](auto i) {
issue_b(i,decltype(i)::value);
issue_a(i,decltype(i)::value);
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
});
auto step=[&](auto a_i,auto b_i,int kt,auto future_i,auto pending_i) {
constexpr int aslot=decltype(a_i)::value,bslot=decltype(b_i)::value;
__builtin_amdgcn_wait_asyncmark(decltype(pending_i)::value);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
array<typename Base::vtype_a,MR> afrags;
array<unsigned,MR> ascale;
array<unsigned,NR> bscale;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int r=m*16+lane16,swz=r%8;
const int lo=aslot*T::A_STAGE+r*128+((kgroup^swz)*16);
const int hi=aslot*T::A_STAGE+r*128+(((kgroup+4)^swz)*16);
set_slice(afrags[m],load<16>(sa,lo),0_I,16_I);
set_slice(afrags[m],load<16>(sa,hi),16_I,32_I);
ascale[m]=load<1>(ssa,kt*BM+r)[0];
});
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int nrow=wave*(BN/T::T_N)+n*16;
bscale[n]=load<1>(ssb,((nrow/128)*TILES+kt)*4)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr (decltype(future_i)::value)
issue_a(number<(aslot+BSETS)%ASTAGES>{},kt+BSETS);
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value,ci=m*NR+n;
c[ci]=Base{}(afrags[m],bregs[bslot][n],c[ci],
static_cast<int>(ascale[m]),static_cast<int>(bscale[n]),0_I,0_I);
});
__builtin_amdgcn_sched_barrier(0);
if constexpr (decltype(future_i)::value) issue_b_column(b_i,ni,kt+BSETS);
});
if constexpr (decltype(future_i)::value) {
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
}
};
constexpr int STEADY=TILES-BSETS,MAIN=STEADY/T::UNROLL*T::UNROLL;
#pragma clang loop unroll(disable)
for (int base=0;base<MAIN;base+=T::UNROLL) {
static_for<T::UNROLL>([&](auto phase_i) {
constexpr int phase=decltype(phase_i)::value;
step(number<phase%ASTAGES>{},number<phase%BSETS>{},base+phase,1_I,number<BSETS-1>{});
});
}
static_for<STEADY-MAIN>([&](auto pi) {
constexpr int kt=MAIN+decltype(pi)::value;
step(number<kt%ASTAGES>{},number<kt%BSETS>{},kt,1_I,number<BSETS-1>{});
});
static_for<BSETS>([&](auto pi) {
constexpr int kt=STEADY+decltype(pi)::value,pending=TILES-1-kt;
step(number<kt%ASTAGES>{},number<kt%BSETS>{},kt,0_I,number<pending>{});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr (T::DIRECT_OUTPUT) {
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto values=cast<bf16_t>(c[m*NR+n]);
static_for<4>([&](auto ii) {
constexpr int element=decltype(ii)::value;
const int r=m*16+kgroup*4+element,nc=wave*(BN/T::T_N)+n*16+lane16;
const uint64_t byte=(static_cast<uint64_t>(r)*N+nc)*2;
gc._store<1>(vector_t<bf16_t,1>{values[element]},
__builtin_bit_cast(int,static_cast<uint32_t>(byte)),0,number<T::OUTPUT_CACHE>{});
});
});
});
} else {
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<BM/T::C_ROWS>([&](auto chunk_i) {
constexpr int chunk=decltype(chunk_i)::value;
static_for<T::C_ROWS/16>([&](auto mi) {
constexpr int m=decltype(mi)::value,cm=chunk*(T::C_ROWS/16)+m;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const auto values=cast<bf16_t>(c[cm*NR+n]);
static_for<4>([&](auto ii) {
constexpr int element=decltype(ii)::value;
const int r=m*16+kgroup*4+element,nc=wave*(BN/T::T_N)+n*16+lane16;
store<1>(sc,vector_t<bf16_t,1>{values[element]},r*(BN+8)+nc);
});
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<T::OUTPUT_PASSES>([&](auto pi) {
const int linear=(decltype(pi)::value*THREADS+tid)*8,r=linear/BN,nc=linear%BN;
if (r<T::C_ROWS && chunk*T::C_ROWS+r<valid_rows) {
const uint64_t byte=(static_cast<uint64_t>(chunk*T::C_ROWS+r)*N+nc)*2;
gc._store<8>(load<8>(sc,r*(BN+8)+nc),
__builtin_bit_cast(int,static_cast<uint32_t>(byte)),0,number<T::OUTPUT_CACHE>{});
}
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
});
}
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_110) && !defined(OPUS_BP_INCLUDE_PERSISTENT_110_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_110_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_110 {
using namespace opus_bpreshuffle_gfx950::persistent::schedule_109;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds_xcd : large_bdirect_traits<224,256,4,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
// SPDX-License-Identifier: Apache-2.0


struct traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds_xcd : large_bdirect_traits<224,256,4,1536,65536,1,false,true> {};
int launch_config(
int selector,int literalSplit,const void* A,const void* B,const void* SFA,const void* SFB,
void* C,void* WS,uint64_t bytes,int M,int N,int K,void* stream) {
if (selector != 6 || literalSplit != 1) return -1;
if (!A || !B || !SFA || !SFB || !C || WS || bytes) return -2;
if (!(M >= 16 && M <= 4096) || M % 16 || N <= 0 || N % 256 || K != 1536 || !(N == 65536)) return -3;
if (uint64_t(M)*K > 2147483647ULL || uint64_t(N)*K > 2147483647ULL) return -4;
if ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(SFA)|reinterpret_cast<uintptr_t>(C)) & 15) return -5;
const uint64_t total=uint64_t(1+(M-1)/224)*(N/256);
if (total > 2147483647ULL) return -6;
const unsigned grid=8*((total+7)/8);
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 a{};
a.ptr_a=A;a.ptr_b=B;a.ptr_sfa=SFA;a.ptr_sfb=SFB;a.ptr_c=C;
a.m=M;a.n=N;a.k=K;a.batch=1;
a.stride_a=K;a.stride_b=K;a.stride_c=N;a.stride_sfa=M;a.stride_sfb=K/128;
hipLaunchKernelGGL((large_bdirect_kernel<traits_large_bd274_wide_m224n256_w1x4_k1536_ba1_lds_xcd>),dim3(grid,1,1),
dim3(256),0,reinterpret_cast<hipStream_t>(stream),a);
return static_cast<int>(hipGetLastError());
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif
}
#endif
