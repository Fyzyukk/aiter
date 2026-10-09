#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh"
// Geometry controls are independent of the accepted current 9021 scheduler.
template<int BM,int Stages,int Panel,int FixedK=0>
struct opus_private_geometry_traits : opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
 static constexpr int B_M=BM, HALF_B_M=BM, FIXED_K=FixedK;
 static constexpr int E_M=B_M/(T_M*W_M);
 static constexpr int smem_m_rep=B_M*B_K/smem_linear_wave;
 static constexpr int A_STAGE=smem_m_rep*(smem_linear_wave+smem_padding);
 static constexpr int NUM_STAGES=Stages;
 static constexpr int MATRIX_LDS_BYTES=NUM_STAGES*(A_STAGE+B_STAGE);
 static constexpr int A_VMEM_INSTRUCTIONS=B_M*B_K/(BLOCK_SIZE*VEC_A);
 static constexpr int VMEM_INSTRUCTIONS_PER_TILE=A_VMEM_INSTRUCTIONS+B_VMEM_INSTRUCTIONS;
 static constexpr int SCALE_PANEL=Panel;
 static constexpr int SFA_BYTES=B_M*SCALE_PANEL;
 static constexpr int SFB_BYTES=SCALE_PANEL;
 static constexpr int SFA_VECTORS_PER_GROUP=B_M/VEC_SCALE_A;
 static constexpr int SFA_PASSES=(SFA_BYTES+SFA_BYTES_PER_PASS-1)/SFA_BYTES_PER_PASS;
 static constexpr int A_SCALE_PACKS=(E_M+3)/4;
 static constexpr int LDS_BYTES=MATRIX_LDS_BYTES+SFA_BYTES+SFB_BYTES;
 static constexpr int OUTPUT_PASSES=B_M*B_N/(BLOCK_SIZE*VEC_OUTPUT);
 static_assert(BM==64||BM==96||BM==128||BM==160);
 static_assert(Stages==2||Stages==3);
 static_assert((Panel&(Panel-1))==0 && Panel>=8 && Panel<=32);
 static_assert(FixedK==0 || (FixedK%128==0 && FixedK<=16384));
 static_assert(B_M%(T_M*W_M)==0 && smem_m_rep%NUM_WAVES==0);
 static_assert(B_M*B_N%(BLOCK_SIZE*VEC_OUTPUT)==0);
 static_assert(B_M*C_LDS_ROW_STRIDE_ELEMS*2<=LDS_BYTES && LDS_BYTES<=160*1024);
};
template<int K,int Panel,bool Reset=false>
struct opus_private_pin_fixed_traits : opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
 static constexpr int FIXED_K=K;
 static constexpr bool RESET_SFA_BEFORE_LOAD=Reset;
 static constexpr int SCALE_PANEL_K_CAPACITY=Panel;
 static constexpr int SFA_PASSES_PER_CACHE_PANEL=Panel/SFA_K_COLUMNS_PER_PASS;
 static constexpr int SFA_PANEL_BYTES=B_M*Panel;
 static constexpr int SFB_PANEL_BYTES=SCALE_N_HALVES*Panel*VEC_SF;
 static constexpr int LDS_BYTES=SMEM_A_ELEMS+SMEM_B_ELEMS+SFA_PANEL_BYTES+SFB_PANEL_BYTES;
 static_assert(K%128==0 && K<=16384 && Panel>=16 && Panel<=64);
 static_assert((Panel&(Panel-1))==0 && SFA_PASSES_PER_CACHE_PANEL>0);
};
template<int K> struct opus_private_pad_pin_fixed_traits : opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950 {
 static constexpr int FIXED_K=K;
};
