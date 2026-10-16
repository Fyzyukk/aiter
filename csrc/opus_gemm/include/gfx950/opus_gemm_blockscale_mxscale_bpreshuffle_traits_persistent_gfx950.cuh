#ifndef OPUS_BP_PERSISTENT_TRAITS_BASE
#define OPUS_BP_PERSISTENT_TRAITS_BASE
// SPDX-License-Identifier: Apache-2.0

#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"

namespace opus_bpreshuffle_gfx950::persistent::schedule_0 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int Stages,int BAhead,int FixedK,int XCD,int Output,int ScalePacket,
int StaticN=0,int WaveN=2,int BCache=0>
struct large_C_AGPR_B3_traits {
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

namespace opus_bpreshuffle_gfx950::persistent::schedule_1 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0




template<int BM,int BN,int WavesN,int FixedK,int FixedN,int Ahead,bool Direct,bool XCD>
struct large_pinc270_traits {
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
static_assert(Ahead==0 || Ahead==1);
static_assert(E_M*E_N*4<=256);
static_assert(LOOPS>=B_SETS && FixedK%128==0);
static_assert(C_ROWS%16==0 && BM%C_ROWS==0);
static_assert(LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::persistent::schedule_2 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0




template<int BM,int BN,int WavesN,int FixedK,int FixedN,int Ahead,bool Direct,bool XCD,int SliceM,int WorkersM>
struct large_slice260_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=WavesN;
static constexpr int BLOCK_SIZE=WavesN*64,NUM_WAVES=WavesN,WARP_SIZE=64;
static constexpr int FIXED_K=FixedK,FIXED_N=FixedN,LOOPS=FixedK/128;
static constexpr int E_M=BM/16,E_N=BN/(WavesN*16);
static constexpr int SLICE_M=SliceM,WORKERS_M=WorkersM;
static constexpr int M_SLICES=(E_M+SLICE_M-1)/SLICE_M;
static_assert(SLICE_M==2 || SLICE_M==4);
static constexpr int B_AHEAD=Ahead,B_SETS=Ahead+1,NUM_STAGES=B_SETS+1;
static constexpr int A_PASSES=(BM*128+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16);
static constexpr int A_STAGE=A_PASSES*BLOCK_SIZE*16;
static constexpr int SA_PASSES=(BM*LOOPS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4);
static constexpr int SA_BYTES=SA_PASSES*BLOCK_SIZE*4;
static constexpr int SB_PASSES=((BN/128)*LOOPS+BLOCK_SIZE-1)/BLOCK_SIZE;
static constexpr int SB_BYTES=SB_PASSES*BLOCK_SIZE*4;
static constexpr int C_ROWS=(BM>128 || BN>256) ? BM/2 : BM;
static constexpr int OFF_SA=NUM_STAGES*A_STAGE;
static constexpr int ARENA_WITHOUT_SB=OFF_SA+SA_BYTES>C_ROWS*(BN+8)*2 ? OFF_SA+SA_BYTES:C_ROWS*(BN+8)*2;
static constexpr int OFF_SB=ARENA_WITHOUT_SB;
static constexpr int MATRIX_BYTES=OFF_SB+SB_BYTES;
static constexpr int C_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=Direct || MATRIX_BYTES>C_BYTES ? MATRIX_BYTES:C_BYTES;
static constexpr int OUTPUT_PASSES=(C_ROWS*BN+BLOCK_SIZE*8-1)/(BLOCK_SIZE*8);
static constexpr int UNROLL=NUM_STAGES*B_SETS;
static constexpr int OUTPUT_CACHE=2;
static constexpr bool DIRECT_OUTPUT=Direct,XCD_ORDER=XCD;
static_assert(BM%16==0 && BN%128==0 && BN%(WavesN*16)==0);
static_assert(WavesN==2 || WavesN==4);
static_assert(Ahead==0 || Ahead==1);
static_assert(E_M*E_N*4<=256);
static_assert(LOOPS>=B_SETS && FixedK%128==0);
static_assert(C_ROWS%16==0 && BM%C_ROWS==0);
static_assert(LDS_BYTES<=160*1024);
};
}

#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_28) && !defined(OPUS_BP_INCLUDE_PERSISTENT_28_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_28_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_28 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048, int SupplyPolicy = 0, bool BFirst = false, int ResidentWorkers = 0, int WavesM = 4, int FixedN = 0, bool DirectOutput = false>
struct large_geometry_traits {
static constexpr int SUPPLY_POLICY = SupplyPolicy;
static constexpr int RESIDENT_WORKERS = ResidentWorkers;
static_assert(RESIDENT_WORKERS == 0 || RESIDENT_WORKERS == 128 || RESIDENT_WORKERS == 256 || RESIDENT_WORKERS == 512);
static constexpr bool B_FIRST = BFirst;
static_assert(SUPPLY_POLICY >= 0 && SUPPLY_POLICY <= 2);
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
static constexpr int BLOCK_SIZE = WavesM * 2 * 64;
static constexpr int FIXED_N = FixedN;
static constexpr bool DIRECT_OUTPUT = DirectOutput;
static constexpr int OUTPUT_CACHE = 2;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M = WavesM;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M >= 32 && B_M <= 192 && (B_N == 128 || B_N == 256 || B_N == 384));
static_assert(B_M % (BLOCK_SIZE / (B_K / 16)) == 0);
static_assert(B_N % (NUM_WAVES * W_N) == 0);
static_assert((NUM_WAVES == 4 || NUM_WAVES == 8) && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048, int SupplyPolicy = 0, bool BFirst = false, int ResidentWorkers = 0, int WavesM = 4, int FixedN = 0, bool DirectOutput = false>
struct large_geometry_traits {
static constexpr int SUPPLY_POLICY = SupplyPolicy;
static constexpr int RESIDENT_WORKERS = ResidentWorkers;
static_assert(RESIDENT_WORKERS == 0 || RESIDENT_WORKERS == 128 || RESIDENT_WORKERS == 256 || RESIDENT_WORKERS == 512);
static constexpr bool B_FIRST = BFirst;
static_assert(SUPPLY_POLICY >= 0 && SUPPLY_POLICY <= 2);
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
static constexpr int BLOCK_SIZE = WavesM * 2 * 64;
static constexpr int FIXED_N = FixedN;
static constexpr bool DIRECT_OUTPUT = DirectOutput;
static constexpr int OUTPUT_CACHE = 2;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M = WavesM;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M >= 32 && B_M <= 192 && (B_N == 128 || B_N == 256 || B_N == 384));
static_assert(B_M % (BLOCK_SIZE / (B_K / 16)) == 0);
static_assert(B_N % (NUM_WAVES * W_N) == 0);
static_assert((NUM_WAVES == 4 || NUM_WAVES == 8) && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_62) && !defined(OPUS_BP_INCLUDE_PERSISTENT_62_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_62_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_62 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048, int SupplyPolicy = 0, bool BFirst = false, int ResidentWorkers = 0>
struct large_persistent_traits {
static constexpr int SUPPLY_POLICY = SupplyPolicy;
static constexpr int RESIDENT_WORKERS = ResidentWorkers;
static_assert(RESIDENT_WORKERS == 0 || RESIDENT_WORKERS == 128 || RESIDENT_WORKERS == 256 || RESIDENT_WORKERS == 512);
static constexpr bool B_FIRST = BFirst;
static_assert(SUPPLY_POLICY >= 0 && SUPPLY_POLICY <= 2);
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
static constexpr int BLOCK_SIZE=512;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert((B_M == 128 || B_M == 192) && (B_N == 128 || B_N == 256));
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048, int SupplyPolicy = 0, bool BFirst = false, int ResidentWorkers = 0>
struct large_persistent_traits {
static constexpr int SUPPLY_POLICY = SupplyPolicy;
static constexpr int RESIDENT_WORKERS = ResidentWorkers;
static_assert(RESIDENT_WORKERS == 0 || RESIDENT_WORKERS == 128 || RESIDENT_WORKERS == 256 || RESIDENT_WORKERS == 512);
static constexpr bool B_FIRST = BFirst;
static_assert(SUPPLY_POLICY >= 0 && SUPPLY_POLICY <= 2);
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
static constexpr int BLOCK_SIZE=512;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert((B_M == 128 || B_M == 192) && (B_N == 128 || B_N == 256));
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PERSISTENT_84) && !defined(OPUS_BP_INCLUDE_PERSISTENT_84_DONE)
#define OPUS_BP_INCLUDE_PERSISTENT_84_DONE
namespace opus_bpreshuffle_gfx950::persistent::schedule_84 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048, int SupplyPolicy = 0, bool BFirst = false>
struct large_m_8wave_traits {
static constexpr int SUPPLY_POLICY = SupplyPolicy;
static constexpr bool B_FIRST = BFirst;
static_assert(SUPPLY_POLICY >= 0 && SUPPLY_POLICY <= 2);
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
static constexpr int BLOCK_SIZE=512;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert((B_M == 128 || B_M == 192) && (B_N == 128 || B_N == 256));
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048, int SupplyPolicy = 0, bool BFirst = false>
struct large_m_8wave_traits {
static constexpr int SUPPLY_POLICY = SupplyPolicy;
static constexpr bool B_FIRST = BFirst;
static_assert(SUPPLY_POLICY >= 0 && SUPPLY_POLICY <= 2);
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
static constexpr int BLOCK_SIZE=512;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert((B_M == 128 || B_M == 192) && (B_N == 128 || B_N == 256));
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif
