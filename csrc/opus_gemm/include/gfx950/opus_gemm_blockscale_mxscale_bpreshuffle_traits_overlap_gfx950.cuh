// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_tiled_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_panel_gfx950.cuh"

namespace opus_bpreshuffle_gfx950::overlap::schedule_0 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2>
struct medium270_pinned_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=WaveN*64;
static constexpr int T_M=1,T_N=WaveN,E_M=BM/16,E_N=BN/(WaveN*16);
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
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
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%(WaveN*16)==0);
static_assert(WaveN==2 || WaveN==4);
static_assert(E_M*E_N*4<=256);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_1 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2>
struct medium270_pinned_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=WaveN*64;
static constexpr int T_M=1,T_N=WaveN,E_M=BM/16,E_N=BN/(WaveN*16);
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
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
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%(WaveN*16)==0);
static_assert(WaveN==2 || WaveN==4);
static_assert(E_M*E_N*4<=256);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_2 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2>
struct medium270_pinned_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=WaveN*64;
static constexpr int T_M=1,T_N=WaveN,E_M=BM/16,E_N=BN/(WaveN*16);
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
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
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%(WaveN*16)==0);
static_assert(WaveN==2 || WaveN==4);
static_assert(E_M*E_N*4<=256);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_3 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2,int BCache=0>
struct large_A_AGPR_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=WaveN*64;
static constexpr int T_M=1,T_N=WaveN,E_M=BM/16,E_N=BN/(WaveN*16);
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead+1,B_CACHE=BCache;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
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
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%(WaveN*16)==0);
static_assert(WaveN==2 || WaveN==4);
static_assert(E_M*E_N*4<=256);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(BCache==0 || BCache==2);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_4 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2,int WaveM=1>
struct medium260_splitA_pinned_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=WaveM*WaveN*64;
static constexpr int T_M=WaveM,T_N=WaveN,E_M=BM/(WaveM*16),E_N=BN/(WaveN*16);
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
static constexpr int SFA_BYTES=BM*LOOPS;
static constexpr int SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=BN%128==0 ? BN/128 : (BN+126)/128+1;
static constexpr int SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=(SFB_BYTES+511)/512*512;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=Output==1 && OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES : PIPELINE_BYTES;
static constexpr int gcd(int a,int b) { return b ? gcd(b,a%b) : a; }
static constexpr int UNROLL=Stages*B_SETS/gcd(Stages,B_SETS);
static_assert(BM>0 && BM%(WaveM*16)==0 && BN>=64 && BN%(WaveN*16)==0);
static_assert(WaveN==2 || WaveN==4);
static_assert(WaveM==1 || WaveM==2);
static_assert(E_M*E_N*4<=256);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_5 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2>
struct medium270_pinned_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=WaveN*64;
static constexpr int T_M=1,T_N=WaveN,E_M=BM/16,E_N=BN/(WaveN*16);
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
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
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%(WaveN*16)==0);
static_assert(WaveN==2 || WaveN==4);
static_assert(E_M*E_N*4<=256);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_6 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2>
struct medium270_pinned_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=WaveN*64;
static constexpr int T_M=1,T_N=WaveN,E_M=BM/16,E_N=BN/(WaveN*16);
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
static constexpr int SFA_BYTES=BM*LOOPS;
static constexpr int SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=(SFB_BYTES+255)/256*256;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=Output==1 && OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES : PIPELINE_BYTES;
static constexpr int gcd(int a,int b) { return b ? gcd(b,a%b) : a; }
static constexpr int UNROLL=Stages*B_SETS/gcd(Stages,B_SETS);
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%(WaveN*16)==0);
static_assert(WaveN==2 || WaveN==4);
static_assert(E_M*E_N*4<=256);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_7 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2>
struct medium270_pinned_a_async_b_ahead_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,BLOCK_SIZE=WaveN*64;
static constexpr int T_M=1,T_N=WaveN,E_M=BM/16,E_N=BN/(WaveN*16);
static constexpr int NUM_STAGES=Stages,B_AHEAD=BAhead,B_SETS=BAhead;
static constexpr int FIXED_K=FixedK,LOOPS=FixedK/128;
static constexpr int XCD_MAPPING=XCD,OUTPUT=Output,SCALE_PACKET=ScalePacket,STATIC_N=StaticN;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
static constexpr int SFA_BYTES=BM*LOOPS;
static constexpr int SFA_PASSES=(SFA_BYTES+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SFA_ALLOCATION=SFA_PASSES*BLOCK_SIZE*4;
static constexpr int B_GROUPS=(BN+127)/128,SFB_BYTES=B_GROUPS*LOOPS;
static constexpr int SFB_ALLOCATION=(SFB_BYTES+255)/256*256;
static constexpr int OFF_SFA=Stages*A_STAGE,OFF_SFB=OFF_SFA+SFA_ALLOCATION;
static constexpr int PIPELINE_BYTES=OFF_SFB+SFB_ALLOCATION;
static constexpr int OUTPUT_BYTES=BM*(BN+8)*2;
static constexpr int LDS_BYTES=Output==1 && OUTPUT_BYTES>PIPELINE_BYTES ? OUTPUT_BYTES : PIPELINE_BYTES;
static constexpr int gcd(int a,int b) { return b ? gcd(b,a%b) : a; }
static constexpr int UNROLL=Stages*B_SETS/gcd(Stages,B_SETS);
static_assert(BM>0 && BM%16==0 && BN>=64 && BN%(WaveN*16)==0);
static_assert(WaveN==2 || WaveN==4);
static_assert(E_M*E_N*4<=256);
static_assert(Stages>=2 && Stages<=4 && BAhead>=1 && BAhead<Stages);
static_assert(FixedK>0 && FixedK<=16384 && FixedK%128==0 && LOOPS>=Stages-1);
static_assert(XCD==0 || XCD==1);
static_assert(Output==1 || Output==2);
static_assert(ScalePacket==0 || (ScalePacket==1 && LOOPS%4==0));
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_8 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_5;
// SPDX-License-Identifier: Apache-2.0

template<int TileM>
struct short_runtime_traits {
static_assert(TileM == 128 || TileM == 160);
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = TileM, B_N=128, B_K=128;
static constexpr int T_M=2, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 4 && NUM_WAVES == T_M * T_N * T_K);
static_assert(MAX_K_TILES == 12);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert((E_M == 4 || E_M == 5) && E_N == 4 && E_K == 1);
static_assert(C_REGS + E_M * 8 <= 128, "C followed by A stays in AGPR0:127");
static_assert(NUM_STAGES == 2);
static_assert(MATRIX_LDS_BYTES == (B_M == 128 ? 67584 : 76032));
static_assert(LDS_BYTES == (B_M == 128 ? 69132 : 77964));
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};
}

namespace opus_bpreshuffle_gfx950::overlap::schedule_9 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_0;
using namespace opus_bpreshuffle_gfx950::overlap::schedule_8;
// SPDX-License-Identifier: Apache-2.0
}
