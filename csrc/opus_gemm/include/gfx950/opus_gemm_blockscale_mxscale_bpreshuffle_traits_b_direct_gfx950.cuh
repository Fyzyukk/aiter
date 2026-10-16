// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_register_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"

namespace opus_bpreshuffle_gfx950::b_direct::schedule_0 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_POWER_ROWS=[]() constexpr {int r=16;while(r<BM)r*=2;return r;}();
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_PADDED_BYTES=((SA_POWER_ROWS*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_ROWS=SA_PADDED_BYTES==SA_BYTES?SA_POWER_ROWS:BM;
static constexpr int SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_1 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_2 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BM>=192 && BN==256;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=BM/32,B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+4095)/4096)*4096,SB_ROWS=(BN+127)/128,SB_BYTES=1024;
static constexpr int SA_COPIES=SA_BYTES/4096,SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%32==0 && BM>=64 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_3 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=true;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=112,OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_4 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=true;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=16,OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_5 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=true,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_6 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=true,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_7 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BM>=192 && BN==256;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=BM/32,B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+4095)/4096)*4096,SB_ROWS=(BN+127)/128,SB_BYTES=1024;
static constexpr int SA_COPIES=SA_BYTES/4096,SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%32==0 && BM>=64 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_8 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_9 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_10 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_11 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_12 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_13 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_14 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int M=MM,B_M=BM,B_N=BN,B_K=256,T_M=1,T_N=4,T_K=1;
static constexpr int BLOCK_SIZE=256,MR=BM/(16*T_M),NR=BN/(16*T_N),STAGES=NB,GLOBAL_SPLIT=1;
static constexpr int N=NN,K=KK,B_CACHE=BC,TOTAL_SCALES=KK/128,PANELS=KK/(256*GLOBAL_SPLIT),SCALE_COLS=KK/(128*GLOBAL_SPLIT),KSTEPS=2;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true,PERSTAGE=false;
static constexpr int A_BYTES=BM*B_K,B_BYTES=0;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_COPIES=B_BYTES/(BLOCK_SIZE*16);
static constexpr int SA_ROWS=BM,SB_ROWS=(BN+127)/128,SA_BYTES=3072,SB_BYTES=1024;
static constexpr int OFF_SA=PERSTAGE?A_BYTES+B_BYTES:NB*12288,OFF_SB=OFF_SA+SA_BYTES;
static constexpr int STAGE_BYTES=12288,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4);
static constexpr int ISSUES=A_COPIES+B_COPIES+(PERSTAGE?SA_COPIES+SB_COPIES:0);
static constexpr int PIPELINE_BYTES=NB*STAGE_BYTES,OUTPUT_BYTES=0,FLAG_OFF=PIPELINE_BYTES+(PERSTAGE?0:SA_BYTES+SB_BYTES),LDS_BYTES=FLAG_OFF+(GLOBAL_SPLIT>1?16:0);
static_assert(MM>0 && MM%4==0 && NN%BN==0 && KK%(B_K*GLOBAL_SPLIT)==0 && PANELS>=NB-1);
static_assert(BM%(16*T_M)==0 && BN%(16*T_N)==0 && LDS_BYTES<=160*1024);
static constexpr int B_AHEAD=BD,B_SETS=BD+1,INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int B_ISSUES=NR*2*KSTEPS,SCALE_ISSUES=PERSTAGE?0:SA_COPIES+SB_COPIES;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=ISSUES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=ISSUES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*ISSUES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=ISSUES;
}
return issued-be;
}
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_15 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_16 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=true;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=16,OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_17 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=true;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=112,OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_18 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (2==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_POWER_ROWS=[]() constexpr {int r=16;while(r<BM)r*=2;return r;}();
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_PADDED_BYTES=((SA_POWER_ROWS*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_ROWS=SA_PADDED_BYTES==SA_BYTES?SA_POWER_ROWS:BM;
static constexpr int SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_19 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_POWER_ROWS=[]() constexpr {int r=16;while(r<BM)r*=2;return r;}();
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_PADDED_BYTES=((SA_POWER_ROWS*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_ROWS=SA_PADDED_BYTES==SA_BYTES?SA_POWER_ROWS:BM;
static constexpr int SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_20 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (2==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_POWER_ROWS=[]() constexpr {int r=16;while(r<BM)r*=2;return r;}();
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_PADDED_BYTES=((SA_POWER_ROWS*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_ROWS=SA_PADDED_BYTES==SA_BYTES?SA_POWER_ROWS:BM;
static constexpr int SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_21 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (2==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_22 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
__device__ __forceinline__ unsigned bulk_sfa_dword_byte(unsigned word, unsigned shift) {
unsigned byte;
asm("v_bfe_u32 %0, %1, %2, 8" : "=v"(byte) : "v"(word), "v"(shift));
return byte;
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_23 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (2==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_24 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_25 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_26 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (2==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_27 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (2==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_28 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_29 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_30 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=true;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=16,OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
__device__ __forceinline__ unsigned bulk_sfa_dword_byte(unsigned word, unsigned shift) {
unsigned byte;
asm("v_bfe_u32 %0, %1, %2, 8" : "=v"(byte) : "v"(word), "v"(shift));
return byte;
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_31 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=true;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=16,OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
__device__ __forceinline__ unsigned bulk_sfb_scalar_word(uint64_t base, unsigned offset) {
unsigned ready;
asm("s_load_dword %0, %1, %2"
: "=s"(ready) : "s"(base), "s"(offset));
return ready;
}
__device__ __forceinline__ unsigned bulk_sfb_scalar_complete(unsigned pending) {
unsigned ready;
asm volatile("s_waitcnt lgkmcnt(0)\n\ts_mov_b32 %0, %1"
: "=s"(ready) : "s"(pending) : "memory");
return ready;
}
__device__ __forceinline__ unsigned bulk_sfb_scalar_byte(unsigned word, unsigned shift) {
unsigned ready;
asm("s_lshr_b32 %0, %1, %2\n\ts_and_b32 %0, %0, 255"
: "=&s"(ready) : "s"(word), "s"(shift) : "scc");
return ready;
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_32 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_33 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_34 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int j=0;j<t;++j){
if(j+NB-1<PANELS){issued+=A_COPIES;if(j+NB-1==t)ae=issued;}
if(j+BD<PANELS)issued+=B_ISSUES;
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int j=0;j<t;++j){
if(j+NB-1<PANELS)issued+=A_COPIES;
if(j+BD<PANELS){issued+=B_ISSUES;if(j+BD==t)be=issued;}
}
if(t+NB-1<PANELS)issued+=A_COPIES;
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_35 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int j=0;j<t;++j){
if(j+NB-1<PANELS){issued+=A_COPIES;if(j+NB-1==t)ae=issued;}
if(j+BD<PANELS)issued+=B_ISSUES;
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int j=0;j<t;++j){
if(j+NB-1<PANELS)issued+=A_COPIES;
if(j+BD<PANELS){issued+=B_ISSUES;if(j+BD==t)be=issued;}
}
if(t+NB-1<PANELS)issued+=A_COPIES;
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_36 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD,N=NN,K=KK,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BM>=192 && BN==256;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=BM/32,B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+4095)/4096)*4096,SB_ROWS=(BN+127)/128,SB_BYTES=1024;
static constexpr int SA_COPIES=SA_BYTES/4096,SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int j=0;j<t;++j){
if(j+NB-1<PANELS){issued+=A_COPIES;if(j+NB-1==t)ae=issued;}
if(j+BD<PANELS)issued+=B_ISSUES;
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int j=0;j<t;++j){
if(j+NB-1<PANELS)issued+=A_COPIES;
if(j+BD<PANELS){issued+=B_ISSUES;if(j+BD==t)be=issued;}
}
if(t+NB-1<PANELS)issued+=A_COPIES;
return issued-be;
}
static_assert(BM%32==0 && BM>=64 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_37 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int j=0;j<t;++j){
if(j+NB-1<PANELS){issued+=A_COPIES;if(j+NB-1==t)ae=issued;}
if(j+BD<PANELS)issued+=B_ISSUES;
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int j=0;j<t;++j){
if(j+NB-1<PANELS)issued+=A_COPIES;
if(j+BD<PANELS){issued+=B_ISSUES;if(j+BD==t)be=issued;}
}
if(t+NB-1<PANELS)issued+=A_COPIES;
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_38 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int j=0;j<t;++j){
if(j+NB-1<PANELS){issued+=A_COPIES;if(j+NB-1==t)ae=issued;}
if(j+BD<PANELS)issued+=B_ISSUES;
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int j=0;j<t;++j){
if(j+NB-1<PANELS)issued+=A_COPIES;
if(j+BD<PANELS){issued+=B_ISSUES;if(j+BD==t)be=issued;}
}
if(t+NB-1<PANELS)issued+=A_COPIES;
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_39 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD>
struct bulk_Bdirect_traits {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD,N=NN,K=KK,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BM>=192 && BN==256;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=BM/32,B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+4095)/4096)*4096,SB_ROWS=(BN+127)/128,SB_BYTES=1024;
static constexpr int SA_COPIES=SA_BYTES/4096,SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int j=0;j<t;++j){
if(j+NB-1<PANELS){issued+=A_COPIES;if(j+NB-1==t)ae=issued;}
if(j+BD<PANELS)issued+=B_ISSUES;
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int j=0;j<t;++j){
if(j+NB-1<PANELS)issued+=A_COPIES;
if(j+BD<PANELS){issued+=B_ISSUES;if(j+BD==t)be=issued;}
}
if(t+NB-1<PANELS)issued+=A_COPIES;
return issued-be;
}
static_assert(BM%32==0 && BM>=64 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_40 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_2wave_pin_agpr_traits_gfx950 {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=2,T_K=1,BLOCK_SIZE=128;
static constexpr int MR=BM/16,NR=BN/32,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=true;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=16,OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T> __device__ __forceinline__
void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_2wave_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,BD=T::B_AHEAD,BS=T::B_SETS,P=T::PANELS;
constexpr int MR=T::MR,NR=T::NR,N=T::N,K=T::K;
constexpr int M=T::M;
const unsigned tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const unsigned wave=__builtin_amdgcn_readfirstlane(tid/64),wn=wave;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=T::XCD_ORDER?pid%8*((total+7)/8)+pid/8:pid;
if constexpr(T::XCD_ORDER)if(tile>=total)return;
const int row=M<=BM?0:(tile/gn)*BM,col=M<=BM?tile*BN:(tile%gn)*BN;
const int valid_rows=M%BM==0?BM:(M<=BM?M:(row==(gm-1)*BM?M%BM:BM));
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*P));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*P));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
auto issue_b=[&](auto bi,int kt) {
constexpr int set=decltype(bi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/2)+n*16,off=kg*256+lane16*16;
set_slice(bregs[set][n],load<16>(gb,off,br*K+kt*2048,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,off+1024,br*K+kt*2048,number<T::B_CACHE>{}),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_B>([&](auto i){issue_b(i,decltype(i)::value);});
static_for<T::SA_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*T::BLOCK_SIZE+tid)*16,sr=index%BM,scol=index/BM,sc=scol<P?scol:P-1;
const int source=row+sr<M?sc*M+row+sr:M*P;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SA+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
auto* sbdst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SB+wave*64*4));
if constexpr(T::SB_SLOTS) {
const int index=tid%(T::SB_ROWS*P),sr=index/P,sc=index%P;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,1,(col/128+sr)*P+sc,0,0,0);
} else {
const int index=tid%(T::SB_ROWS*(P/4)),sr=index/(P/4),word=index%(P/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,4,(col/128+sr)*P+word*4,0,0,0);
}
auto issue_a=[&](int slot,int kt) {
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/128,kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_A>([&](auto i){issue_a(decltype(i)::value,decltype(i)::value);});
auto step=[&](int t,int slot,auto bi,auto wai,auto wbi,auto fai,auto fbi) {
constexpr int set=decltype(bi)::value,wa=decltype(wai)::value,wb=decltype(wbi)::value;
constexpr bool future_a=decltype(fai)::value,future_b=decltype(fbi)::value;
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wa>63?63:wa)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(future_b)issue_b(number<(set+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> af;
array<unsigned,MR> as;
const int sbrow=wn*(BN/2)/128;
const unsigned bs=load<1>(ss,T::OFF_SB+(sbrow*P+t)*(T::SB_SLOTS?4:1))[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*128+((kg^(ar%8))*16);
const int hi=slot*T::STAGE_BYTES+ar*128+(((kg+4)^(ar%8))*16);
set_slice(af[m],load<16>(sa,lo),0_I,16_I);
set_slice(af[m],load<16>(sa,hi),16_I,32_I);
as[m]=load<1>(ss,T::OFF_SA+t*BM+ar)[0];
});
if constexpr(future_a)issue_a(slot==0?NB-1:slot-1,t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wb>63?63:wb)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::C_AGPR) {
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
} else c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
});
});
if constexpr(!T::C_AGPR)static_for<MR*NR>([&](auto ci) {auto& v=c[decltype(ci)::value];asm volatile("" : "+v"(v));});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int START=T::STEADY<NB-1?T::STEADY:NB-1;
static_for<START>([&](auto ti) {
constexpr int t=decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<true>{},number<(t+BD<P)>{});
});
constexpr int PHASES=[]() constexpr {int a=NB,b=BS;while(b){int r=a%b;a=b;b=r;}return NB/a*BS;}();
constexpr int GROUPS=(T::STEADY-START)/PHASES,REGULAR_END=START+GROUPS*PHASES;
if constexpr(GROUPS>0) {
#pragma clang loop unroll(disable)
for(int group=0;group<GROUPS;++group) {
static_for<PHASES>([&](auto phase) {
constexpr int ph=decltype(phase)::value,t0=START+ph;
step(START+group*PHASES+ph,t0%NB,number<t0%BS>{},number<T::wait_a(t0)>{},number<T::wait_b(t0)>{},number<true>{},number<true>{});
});
}
}
static_for<P-REGULAR_END>([&](auto ti) {
constexpr int t=REGULAR_END+decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<(t+NB-1<P)>{},number<(t+BD<P)>{});
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
constexpr int MAX_ROWS=M<BM?M:BM;
static_for<(MAX_ROWS+T::C_ROWS-1)/T::C_ROWS>([&](auto parti) {
constexpr int part=decltype(parti)::value;
constexpr int PART_ROWS=MAX_ROWS-part*T::C_ROWS<T::C_ROWS?MAX_ROWS-part*T::C_ROWS:T::C_ROWS;
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
if constexpr(m*16/T::C_ROWS==part && m*16<MAX_ROWS)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
const int rr=m*16+lane16,nn=wn*(BN/2)+n*16+kg*4;
store<4>(sc,cast<bf16_t>(c[ci]),(rr-part*T::C_ROWS)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
constexpr int CHUNKS=(PART_ROWS*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE;
array<vector_t<bf16_t,8>,CHUNKS> output;
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if constexpr(PART_ROWS*(BN/8)%T::BLOCK_SIZE==0)output[i]=load<8>(sc,rr*(BN+8)+nn);
else if(rr<PART_ROWS)output[i]=load<8>(sc,rr*(BN+8)+nn);
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<PART_ROWS && part*T::C_ROWS+rr<valid_rows)
store<8>(gc,output[i],(part*T::C_ROWS+rr)*N+nn,0,number<2>{});
});
});
s_waitcnt_vmcnt(0_I);
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_41 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_first_a_pin_agpr_traits_gfx950 {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_POWER_ROWS=[]() constexpr {int r=16;while(r<BM)r*=2;return r;}();
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_PADDED_BYTES=((SA_POWER_ROWS*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16);
static constexpr int SA_ROWS=SA_PADDED_BYTES==SA_BYTES?SA_POWER_ROWS:BM;
static constexpr int SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T> __device__ __forceinline__
void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_first_a_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,BD=T::B_AHEAD,BS=T::B_SETS,P=T::PANELS;
constexpr int MR=T::MR,NR=T::NR,N=T::N,K=T::K;
constexpr int M=T::M;
const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wn=wave;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=T::XCD_ORDER?pid%8*((total+7)/8)+pid/8:pid;
if constexpr(T::XCD_ORDER)if(tile>=total)return;
const int row=M<=BM?0:(tile/gn)*BM,col=M<=BM?tile*BN:(tile%gn)*BN;
const int valid_rows=M%BM==0?BM:(M<=BM?M:(row==(gm-1)*BM?M%BM:BM));
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*P));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*P));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
auto issue_b=[&](auto bi,int kt) {
constexpr int set=decltype(bi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/4)+n*16,off=kg*256+lane16*16;
set_slice(bregs[set][n],load<16>(gb,off,br*K+kt*2048,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,off+1024,br*K+kt*2048,number<T::B_CACHE>{}),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
};
auto issue_a=[&](int slot,int kt) {
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/128,kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_B>([&](auto i){issue_b(i,decltype(i)::value);});
issue_a(0,0);
static_for<T::SA_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*T::BLOCK_SIZE+tid)*16,sr=index%T::SA_ROWS,scol=index/T::SA_ROWS,sc=scol<P?scol:P-1;
int source=sr<BM && row+sr<M?sc*M+row+sr:M*P;
asm volatile("" : "+v"(source));
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SA+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
auto* sbdst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SB+wave*64*4));
if constexpr(T::SB_SLOTS) {
const int index=tid%(T::SB_ROWS*P),sr=index/P,sc=index%P;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,1,(col/128+sr)*P+sc,0,0,0);
} else {
const int index=tid%(T::SB_ROWS*(P/4)),sr=index/(P/4),word=index%(P/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,4,(col/128+sr)*P+word*4,0,0,0);
}
static_for<T::INITIAL_A-1>([&](auto i){issue_a(decltype(i)::value+1,decltype(i)::value+1);});
auto step=[&](int t,int slot,auto bi,auto wai,auto wbi,auto fai,auto fbi) {
constexpr int set=decltype(bi)::value,wa=decltype(wai)::value,wb=decltype(wbi)::value;
constexpr bool future_a=decltype(fai)::value,future_b=decltype(fbi)::value;
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wa>63?63:wa)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(future_b)issue_b(number<(set+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> af;
array<unsigned,MR> as;
const int sbrow=wn*(BN/4)/128;
const unsigned bs=load<1>(ss,T::OFF_SB+(sbrow*P+t)*(T::SB_SLOTS?4:1))[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*128+((kg^(ar%8))*16);
const int hi=slot*T::STAGE_BYTES+ar*128+(((kg+4)^(ar%8))*16);
set_slice(af[m],load<16>(sa,lo),0_I,16_I);
set_slice(af[m],load<16>(sa,hi),16_I,32_I);
as[m]=load<1>(ss,T::OFF_SA+t*T::SA_ROWS+ar)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(future_a)issue_a(slot==0?NB-1:slot-1,t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wb>63?63:wb)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::C_AGPR) {
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
} else c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
});
});
if constexpr(!T::C_AGPR)static_for<MR*NR>([&](auto ci) {auto& v=c[decltype(ci)::value];asm volatile("" : "+v"(v));});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int START=T::STEADY<NB-1?T::STEADY:NB-1;
static_for<START>([&](auto ti) {
constexpr int t=decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<true>{},number<(t+BD<P)>{});
});
constexpr int PHASES=[]() constexpr {int a=NB,b=BS;while(b){int r=a%b;a=b;b=r;}return NB/a*BS;}();
constexpr int GROUPS=(T::STEADY-START)/PHASES,REGULAR_END=START+GROUPS*PHASES;
if constexpr(GROUPS>0) {
#pragma clang loop unroll(disable)
for(int group=0;group<GROUPS;++group) {
static_for<PHASES>([&](auto phase) {
constexpr int ph=decltype(phase)::value,t0=START+ph;
step(START+group*PHASES+ph,t0%NB,number<t0%BS>{},number<T::wait_a(t0)>{},number<T::wait_b(t0)>{},number<true>{},number<true>{});
});
}
}
static_for<P-REGULAR_END>([&](auto ti) {
constexpr int t=REGULAR_END+decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<(t+NB-1<P)>{},number<(t+BD<P)>{});
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
constexpr int MAX_ROWS=M<BM?M:BM;
static_for<(MAX_ROWS+T::C_ROWS-1)/T::C_ROWS>([&](auto parti) {
constexpr int part=decltype(parti)::value;
constexpr int PART_ROWS=MAX_ROWS-part*T::C_ROWS<T::C_ROWS?MAX_ROWS-part*T::C_ROWS:T::C_ROWS;
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
if constexpr(m*16/T::C_ROWS==part && m*16<MAX_ROWS)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
const int rr=m*16+lane16,nn=wn*(BN/4)+n*16+kg*4;
store<4>(sc,cast<bf16_t>(c[ci]),(rr-part*T::C_ROWS)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
constexpr int CHUNKS=(PART_ROWS*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE;
array<vector_t<bf16_t,8>,CHUNKS> output;
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if constexpr(PART_ROWS*(BN/8)%T::BLOCK_SIZE==0)output[i]=load<8>(sc,rr*(BN+8)+nn);
else if(rr<PART_ROWS)output[i]=load<8>(sc,rr*(BN+8)+nn);
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<PART_ROWS && part*T::C_ROWS+rr<valid_rows)
store<8>(gc,output[i],(part*T::C_ROWS+rr)*N+nn,0,number<2>{});
});
});
s_waitcnt_vmcnt(0_I);
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_42 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_fixed_m_pin_agpr_traits_gfx950 {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BN==256 && (BM>=192 || (4==2 && BM>=112 && BD>=2));
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T> __device__ __forceinline__
void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_fixed_m_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,BD=T::B_AHEAD,BS=T::B_SETS,P=T::PANELS;
constexpr int MR=T::MR,NR=T::NR,N=T::N,K=T::K;
constexpr int M=T::M;
const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wn=wave;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=T::XCD_ORDER?pid%8*((total+7)/8)+pid/8:pid;
if constexpr(T::XCD_ORDER)if(tile>=total)return;
const int row=M<=BM?0:(tile/gn)*BM,col=M<=BM?tile*BN:(tile%gn)*BN;
const int valid_rows=M%BM==0?BM:(M<=BM?M:(row==(gm-1)*BM?M%BM:BM));
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*P));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*P));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
auto issue_b=[&](auto bi,int kt) {
constexpr int set=decltype(bi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/4)+n*16,off=kg*256+lane16*16;
set_slice(bregs[set][n],load<16>(gb,off,br*K+kt*2048,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,off+1024,br*K+kt*2048,number<T::B_CACHE>{}),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_B>([&](auto i){issue_b(i,decltype(i)::value);});
static_for<T::SA_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*T::BLOCK_SIZE+tid)*16,sr=index%BM,scol=index/BM,sc=scol<P?scol:P-1;
const int source=row+sr<M?sc*M+row+sr:M*P;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SA+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
auto* sbdst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SB+wave*64*4));
if constexpr(T::SB_SLOTS) {
const int index=tid%(T::SB_ROWS*P),sr=index/P,sc=index%P;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,1,(col/128+sr)*P+sc,0,0,0);
} else {
const int index=tid%(T::SB_ROWS*(P/4)),sr=index/(P/4),word=index%(P/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,4,(col/128+sr)*P+word*4,0,0,0);
}
auto issue_a=[&](int slot,int kt) {
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/128,kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_A>([&](auto i){issue_a(decltype(i)::value,decltype(i)::value);});
auto step=[&](int t,int slot,auto bi,auto wai,auto wbi,auto fai,auto fbi) {
constexpr int set=decltype(bi)::value,wa=decltype(wai)::value,wb=decltype(wbi)::value;
constexpr bool future_a=decltype(fai)::value,future_b=decltype(fbi)::value;
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wa>63?63:wa)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(future_b)issue_b(number<(set+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> af;
array<unsigned,MR> as;
const int sbrow=wn*(BN/4)/128;
const unsigned bs=load<1>(ss,T::OFF_SB+(sbrow*P+t)*(T::SB_SLOTS?4:1))[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*128+((kg^(ar%8))*16);
const int hi=slot*T::STAGE_BYTES+ar*128+(((kg+4)^(ar%8))*16);
set_slice(af[m],load<16>(sa,lo),0_I,16_I);
set_slice(af[m],load<16>(sa,hi),16_I,32_I);
as[m]=load<1>(ss,T::OFF_SA+t*BM+ar)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(future_a)issue_a(slot==0?NB-1:slot-1,t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wb>63?63:wb)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::C_AGPR) {
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
} else c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
});
});
if constexpr(!T::C_AGPR)static_for<MR*NR>([&](auto ci) {auto& v=c[decltype(ci)::value];asm volatile("" : "+v"(v));});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int START=T::STEADY<NB-1?T::STEADY:NB-1;
static_for<START>([&](auto ti) {
constexpr int t=decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<true>{},number<(t+BD<P)>{});
});
constexpr int PHASES=[]() constexpr {int a=NB,b=BS;while(b){int r=a%b;a=b;b=r;}return NB/a*BS;}();
constexpr int GROUPS=(T::STEADY-START)/PHASES,REGULAR_END=START+GROUPS*PHASES;
if constexpr(GROUPS>0) {
#pragma clang loop unroll(disable)
for(int group=0;group<GROUPS;++group) {
static_for<PHASES>([&](auto phase) {
constexpr int ph=decltype(phase)::value,t0=START+ph;
step(START+group*PHASES+ph,t0%NB,number<t0%BS>{},number<T::wait_a(t0)>{},number<T::wait_b(t0)>{},number<true>{},number<true>{});
});
}
}
static_for<P-REGULAR_END>([&](auto ti) {
constexpr int t=REGULAR_END+decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<(t+NB-1<P)>{},number<(t+BD<P)>{});
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
constexpr int MAX_ROWS=M<BM?M:BM;
static_for<(MAX_ROWS+T::C_ROWS-1)/T::C_ROWS>([&](auto parti) {
constexpr int part=decltype(parti)::value;
constexpr int PART_ROWS=MAX_ROWS-part*T::C_ROWS<T::C_ROWS?MAX_ROWS-part*T::C_ROWS:T::C_ROWS;
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
if constexpr(m*16/T::C_ROWS==part && m*16<MAX_ROWS)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
const int rr=m*16+lane16,nn=wn*(BN/4)+n*16+kg*4;
store<4>(sc,cast<bf16_t>(c[ci]),(rr-part*T::C_ROWS)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
constexpr int CHUNKS=(PART_ROWS*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE;
array<vector_t<bf16_t,8>,CHUNKS> output;
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if constexpr(PART_ROWS*(BN/8)%T::BLOCK_SIZE==0)output[i]=load<8>(sc,rr*(BN+8)+nn);
else if(rr<PART_ROWS)output[i]=load<8>(sc,rr*(BN+8)+nn);
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<PART_ROWS && part*T::C_ROWS+rr<valid_rows)
store<8>(gc,output[i],(part*T::C_ROWS+rr)*N+nn,0,number<2>{});
});
});
s_waitcnt_vmcnt(0_I);
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_43 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_pin_agpr_traits_gfx950 {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BM>=192 && BN==256;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=BM/32,B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+4095)/4096)*4096,SB_ROWS=(BN+127)/128,SB_BYTES=1024;
static constexpr int SA_COPIES=SA_BYTES/4096,SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%32==0 && BM>=64 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T> __device__ __forceinline__
void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,BD=T::B_AHEAD,BS=T::B_SETS,P=T::PANELS;
constexpr int MR=T::MR,NR=T::NR,N=T::N,K=T::K;
const int M=args.m,tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wn=wave;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=T::XCD_ORDER?pid%8*((total+7)/8)+pid/8:pid;
if constexpr(T::XCD_ORDER)if(tile>=total)return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN,valid_rows=M-row<BM?M-row:BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*P));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*P));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
auto issue_b=[&](auto bi,int kt) {
constexpr int set=decltype(bi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/4)+n*16,off=kg*256+lane16*16;
set_slice(bregs[set][n],load<16>(gb,off,br*K+kt*2048,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,off+1024,br*K+kt*2048,number<T::B_CACHE>{}),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_B>([&](auto i){issue_b(i,decltype(i)::value);});
static_for<T::SA_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*256+tid)*16,sr=index%BM,scol=index/BM,sc=scol<P?scol:P-1;
const int source=row+sr<M?sc*M+row+sr:M*P;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SA+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
auto* sbdst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SB+wave*64*4));
if constexpr(T::SB_SLOTS) {
const int index=tid%(T::SB_ROWS*P),sr=index/P,sc=index%P;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,1,(col/128+sr)*P+sc,0,0,0);
} else {
const int index=tid%(T::SB_ROWS*(P/4)),sr=index/(P/4),word=index%(P/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,4,(col/128+sr)*P+word*4,0,0,0);
}
auto issue_a=[&](int slot,int kt) {
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*256+tid)*16,r=lin/128,kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_A>([&](auto i){issue_a(decltype(i)::value,decltype(i)::value);});
auto step=[&](int t,int slot,auto bi,auto wai,auto wbi,auto fai,auto fbi) {
constexpr int set=decltype(bi)::value,wa=decltype(wai)::value,wb=decltype(wbi)::value;
constexpr bool future_a=decltype(fai)::value,future_b=decltype(fbi)::value;
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wa>63?63:wa)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(future_b)issue_b(number<(set+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> af;
array<unsigned,MR> as;
const int sbrow=wn*(BN/4)/128;
const unsigned bs=load<1>(ss,T::OFF_SB+(sbrow*P+t)*(T::SB_SLOTS?4:1))[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*128+((kg^(ar%8))*16);
const int hi=slot*T::STAGE_BYTES+ar*128+(((kg+4)^(ar%8))*16);
set_slice(af[m],load<16>(sa,lo),0_I,16_I);
set_slice(af[m],load<16>(sa,hi),16_I,32_I);
as[m]=load<1>(ss,T::OFF_SA+t*BM+ar)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(future_a)issue_a(slot==0?NB-1:slot-1,t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wb>63?63:wb)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::C_AGPR) {
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
} else c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
});
});
if constexpr(!T::C_AGPR)static_for<MR*NR>([&](auto ci) {auto& v=c[decltype(ci)::value];asm volatile("" : "+v"(v));});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int START=T::STEADY<NB-1?T::STEADY:NB-1;
static_for<START>([&](auto ti) {
constexpr int t=decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<true>{},number<(t+BD<P)>{});
});
constexpr int GROUPS=(T::STEADY-START)/BS,REGULAR_END=START+GROUPS*BS;
if constexpr(GROUPS>0) {
int ring_slot=START%NB;
#pragma clang loop unroll(disable)
for(int group=0;group<GROUPS;++group) {
static_for<BS>([&](auto phase) {
constexpr int ph=decltype(phase)::value,t0=START+ph;
step(START+group*BS+ph,ring_slot,number<t0%BS>{},number<T::wait_a(t0)>{},number<T::wait_b(t0)>{},number<true>{},number<true>{});
ring_slot=ring_slot+1==NB?0:ring_slot+1;
});
}
}
static_for<P-REGULAR_END>([&](auto ti) {
constexpr int t=REGULAR_END+decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<(t+NB-1<P)>{},number<(t+BD<P)>{});
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<BM/T::C_ROWS>([&](auto parti) {
constexpr int part=decltype(parti)::value;
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
if constexpr(m*16/T::C_ROWS==part)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
const int rr=m*16+lane16,nn=wn*(BN/4)+n*16+kg*4;
if(rr<valid_rows)store<4>(sc,cast<bf16_t>(c[ci]),(rr-part*T::C_ROWS)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<(T::C_ROWS*(BN/8)+255)/256>([&](auto ii) {
const int index=decltype(ii)::value*256+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<T::C_ROWS && part*T::C_ROWS+rr<valid_rows)
store<8>(gc,load<8>(sc,rr*(BN+8)+nn),(part*T::C_ROWS+rr)*N+nn,0,number<2>{});
});
});
s_waitcnt_vmcnt(0_I);
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_44 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_stage_pin_agpr_traits_gfx950 {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=true;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+BLOCK_SIZE*16-1)/(BLOCK_SIZE*16))*(BLOCK_SIZE*16),SB_ROWS=(BN+127)/128,SB_BYTES=BLOCK_SIZE*4;
static constexpr int SA_COPIES=SA_BYTES/(BLOCK_SIZE*16),SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=112,OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%16==0 && BM>=16 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(MM>0 && MM%16==0);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T> __device__ __forceinline__
void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_stage_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,BD=T::B_AHEAD,BS=T::B_SETS,P=T::PANELS;
constexpr int MR=T::MR,NR=T::NR,N=T::N,K=T::K;
constexpr int M=T::M;
const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wn=wave;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=T::XCD_ORDER?pid%8*((total+7)/8)+pid/8:pid;
if constexpr(T::XCD_ORDER)if(tile>=total)return;
const int row=M<=BM?0:(tile/gn)*BM,col=M<=BM?tile*BN:(tile%gn)*BN;
const int valid_rows=M%BM==0?BM:(M<=BM?M:(row==(gm-1)*BM?M%BM:BM));
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*P));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*P));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
auto issue_b=[&](auto bi,int kt) {
constexpr int set=decltype(bi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/4)+n*16,off=kg*256+lane16*16;
set_slice(bregs[set][n],load<16>(gb,off,br*K+kt*2048,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,off+1024,br*K+kt*2048,number<T::B_CACHE>{}),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_B>([&](auto i){issue_b(i,decltype(i)::value);});
static_for<T::SA_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*T::BLOCK_SIZE+tid)*16,sr=index%BM,scol=index/BM,sc=scol<P?scol:P-1;
const int source=row+sr<M?sc*M+row+sr:M*P;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SA+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
auto* sbdst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SB+wave*64*4));
if constexpr(T::SB_SLOTS) {
const int index=tid%(T::SB_ROWS*P),sr=index/P,sc=index%P;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,1,(col/128+sr)*P+sc,0,0,0);
} else {
const int index=tid%(T::SB_ROWS*(P/4)),sr=index/(P/4),word=index%(P/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,4,(col/128+sr)*P+word*4,0,0,0);
}
auto issue_a=[&](int slot,int kt) {
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/128,kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_A>([&](auto i){issue_a(decltype(i)::value,decltype(i)::value);});
auto step=[&](int t,int slot,auto bi,auto wai,auto wbi,auto fai,auto fbi) {
constexpr int set=decltype(bi)::value,wa=decltype(wai)::value,wb=decltype(wbi)::value;
constexpr bool future_a=decltype(fai)::value,future_b=decltype(fbi)::value;
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wa>63?63:wa)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(future_b)issue_b(number<(set+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> af;
array<unsigned,MR> as;
const int sbrow=wn*(BN/4)/128;
const unsigned bs=load<1>(ss,T::OFF_SB+(sbrow*P+t)*(T::SB_SLOTS?4:1))[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*128+((kg^(ar%8))*16);
const int hi=slot*T::STAGE_BYTES+ar*128+(((kg+4)^(ar%8))*16);
set_slice(af[m],load<16>(sa,lo),0_I,16_I);
set_slice(af[m],load<16>(sa,hi),16_I,32_I);
as[m]=load<1>(ss,T::OFF_SA+t*BM+ar)[0];
});
if constexpr(future_a)issue_a(slot==0?NB-1:slot-1,t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wb>63?63:wb)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::C_AGPR) {
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
} else c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
});
});
if constexpr(!T::C_AGPR)static_for<MR*NR>([&](auto ci) {auto& v=c[decltype(ci)::value];asm volatile("" : "+v"(v));});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int START=T::STEADY<NB-1?T::STEADY:NB-1;
static_for<START>([&](auto ti) {
constexpr int t=decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<true>{},number<(t+BD<P)>{});
});
constexpr int PHASES=[]() constexpr {int a=NB,b=BS;while(b){int r=a%b;a=b;b=r;}return NB/a*BS;}();
constexpr int GROUPS=(T::STEADY-START)/PHASES,REGULAR_END=START+GROUPS*PHASES;
if constexpr(GROUPS>0) {
#pragma clang loop unroll(disable)
for(int group=0;group<GROUPS;++group) {
static_for<PHASES>([&](auto phase) {
constexpr int ph=decltype(phase)::value,t0=START+ph;
step(START+group*PHASES+ph,t0%NB,number<t0%BS>{},number<T::wait_a(t0)>{},number<T::wait_b(t0)>{},number<true>{},number<true>{});
});
}
}
static_for<P-REGULAR_END>([&](auto ti) {
constexpr int t=REGULAR_END+decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<(t+NB-1<P)>{},number<(t+BD<P)>{});
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
constexpr int MAX_ROWS=M<BM?M:BM;
static_for<(MAX_ROWS+T::C_ROWS-1)/T::C_ROWS>([&](auto parti) {
constexpr int part=decltype(parti)::value;
constexpr int PART_ROWS=MAX_ROWS-part*T::C_ROWS<T::C_ROWS?MAX_ROWS-part*T::C_ROWS:T::C_ROWS;
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
if constexpr(m*16/T::C_ROWS==part && m*16<MAX_ROWS)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
const int rr=m*16+lane16,nn=wn*(BN/4)+n*16+kg*4;
store<4>(sc,cast<bf16_t>(c[ci]),(rr-part*T::C_ROWS)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
constexpr int CHUNKS=(PART_ROWS*(BN/8)+T::BLOCK_SIZE-1)/T::BLOCK_SIZE;
array<vector_t<bf16_t,8>,CHUNKS> output;
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if constexpr(PART_ROWS*(BN/8)%T::BLOCK_SIZE==0)output[i]=load<8>(sc,rr*(BN+8)+nn);
else if(rr<PART_ROWS)output[i]=load<8>(sc,rr*(BN+8)+nn);
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*T::BLOCK_SIZE+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<PART_ROWS && part*T::C_ROWS+rr<valid_rows)
store<8>(gc,output[i],(part*T::C_ROWS+rr)*N+nn,0,number<2>{});
});
});
s_waitcnt_vmcnt(0_I);
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_45 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_static_ring_pin_agpr_traits_gfx950 {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=256;
static constexpr int MR=BM/16,NR=BN/64,STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=(KK/128)%4!=0,C_AGPR=BM>=192 && BN==256;
static constexpr int PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES,A_COPIES=BM/32,B_ISSUES=NR*2;
static constexpr int SA_BYTES=((BM*PANELS+4095)/4096)*4096,SB_ROWS=(BN+127)/128,SB_BYTES=1024;
static constexpr int SA_COPIES=SA_BYTES/4096,SCALE_ISSUES=SA_COPIES+1;
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES;
static constexpr int c_rows() {
for(int rows=BM;rows>=16;rows-=16)
if(BM%rows==0 && rows*(BN+8)*2<=PIPELINE_BYTES)return rows;
return 16;
}
static constexpr int C_ROWS=c_rows(),OUTPUT_BYTES=C_ROWS*(BN+8)*2;
static constexpr int LDS_BYTES=PIPELINE_BYTES>OUTPUT_BYTES?PIPELINE_BYTES:OUTPUT_BYTES;
static constexpr int INITIAL_A=PANELS<NB-1?PANELS:NB-1;
static constexpr int INITIAL_B=PANELS<BD?PANELS:BD;
static constexpr int STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=0,ae=-1;
for(int i=0;i<INITIAL_B;++i)issued+=B_ISSUES;
issued+=SCALE_ISSUES;
for(int i=0;i<INITIAL_A;++i) {issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s) {
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS) {issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i) {issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s) {
if(s+BD<PANELS) {issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert(BM%32==0 && BM>=64 && BM<=224 && (BN==64 || BN==128 || BN==256));
static_assert(NN%128==0 && KK%128==0 && NB>=2 && NB<=8 && BD>=1 && BD<=4 && NB-1>=BD);
static_assert(PANELS>=INITIAL_B && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T> __device__ __forceinline__
void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_static_ring_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,BD=T::B_AHEAD,BS=T::B_SETS,P=T::PANELS;
constexpr int MR=T::MR,NR=T::NR,N=T::N,K=T::K;
const int M=args.m,tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wn=wave;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=T::XCD_ORDER?pid%8*((total+7)/8)+pid/8:pid;
if constexpr(T::XCD_ORDER)if(tile>=total)return;
const int row=(tile/gn)*BM,col=(tile%gn)*BN,valid_rows=M-row<BM?M-row:BM;
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*P));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*P));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
array<array<typename Base::vtype_b,NR>,BS> bregs;
auto issue_b=[&](auto bi,int kt) {
constexpr int set=decltype(bi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const int br=wn*(BN/4)+n*16,off=kg*256+lane16*16;
set_slice(bregs[set][n],load<16>(gb,off,br*K+kt*2048,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][n],load<16>(gb,off+1024,br*K+kt*2048,number<T::B_CACHE>{}),16_I,32_I);
});
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_B>([&](auto i){issue_b(i,decltype(i)::value);});
static_for<T::SA_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=(i*256+tid)*16,sr=index%BM,scol=index/BM,sc=scol<P?scol:P-1;
const int source=row+sr<M?sc*M+row+sr:M*P;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SA+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,16,source,0,0,0);
});
auto* sbdst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SB+wave*64*4));
if constexpr(T::SB_SLOTS) {
const int index=tid%(T::SB_ROWS*P),sr=index/P,sc=index%P;
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,1,(col/128+sr)*P+sc,0,0,0);
} else {
const int index=tid%(T::SB_ROWS*(P/4)),sr=index/(P/4),word=index%(P/4);
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,sbdst,4,(col/128+sr)*P+word*4,0,0,0);
}
auto issue_a=[&](int slot,int kt) {
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int lin=(i*256+tid)*16,r=lin/128,kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*256+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_A>([&](auto i){issue_a(decltype(i)::value,decltype(i)::value);});
auto step=[&](int t,int slot,auto bi,auto wai,auto wbi,auto fai,auto fbi) {
constexpr int set=decltype(bi)::value,wa=decltype(wai)::value,wb=decltype(wbi)::value;
constexpr bool future_a=decltype(fai)::value,future_b=decltype(fbi)::value;
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wa>63?63:wa)>{});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if constexpr(future_b)issue_b(number<(set+BD)%BS>{},t+BD);
array<typename Base::vtype_a,MR> af;
array<unsigned,MR> as;
const int sbrow=wn*(BN/4)/128;
const unsigned bs=load<1>(ss,T::OFF_SB+(sbrow*P+t)*(T::SB_SLOTS?4:1))[0];
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
const int ar=m*16+lane16;
const int lo=slot*T::STAGE_BYTES+ar*128+((kg^(ar%8))*16);
const int hi=slot*T::STAGE_BYTES+ar*128+(((kg+4)^(ar%8))*16);
set_slice(af[m],load<16>(sa,lo),0_I,16_I);
set_slice(af[m],load<16>(sa,hi),16_I,32_I);
as[m]=load<1>(ss,T::OFF_SA+t*BM+ar)[0];
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(future_a)issue_a(slot==0?NB-1:slot-1,t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wb>63?63:wb)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::C_AGPR) {
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
} else c[ci]=Base{}(af[m],bregs[set][n],c[ci],static_cast<int>(as[m]),static_cast<int>(bs),0_I,0_I);
});
});
if constexpr(!T::C_AGPR)static_for<MR*NR>([&](auto ci) {auto& v=c[decltype(ci)::value];asm volatile("" : "+v"(v));});
__builtin_amdgcn_sched_barrier(0);
};
constexpr int START=T::STEADY<NB-1?T::STEADY:NB-1;
static_for<START>([&](auto ti) {
constexpr int t=decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<true>{},number<(t+BD<P)>{});
});
constexpr int PHASES=[]() constexpr {int a=NB,b=BS;while(b){int r=a%b;a=b;b=r;}return NB/a*BS;}();
constexpr int GROUPS=(T::STEADY-START)/PHASES,REGULAR_END=START+GROUPS*PHASES;
if constexpr(GROUPS>0) {
#pragma clang loop unroll(disable)
for(int group=0;group<GROUPS;++group) {
static_for<PHASES>([&](auto phase) {
constexpr int ph=decltype(phase)::value,t0=START+ph;
step(START+group*PHASES+ph,t0%NB,number<t0%BS>{},number<T::wait_a(t0)>{},number<T::wait_b(t0)>{},number<true>{},number<true>{});
});
}
}
static_for<P-REGULAR_END>([&](auto ti) {
constexpr int t=REGULAR_END+decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<(t+NB-1<P)>{},number<(t+BD<P)>{});
});
auto sc=make_smem(reinterpret_cast<bf16_t*>(arena));
static_for<BM/T::C_ROWS>([&](auto parti) {
constexpr int part=decltype(parti)::value;
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
if constexpr(m*16/T::C_ROWS==part)static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
const int rr=m*16+lane16,nn=wn*(BN/4)+n*16+kg*4;
store<4>(sc,cast<bf16_t>(c[ci]),(rr-part*T::C_ROWS)*(BN+8)+nn);
});
});
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
constexpr int CHUNKS=(T::C_ROWS*(BN/8)+255)/256;
array<vector_t<bf16_t,8>,CHUNKS> output;
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*256+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if constexpr(T::C_ROWS*(BN/8)%256==0)output[i]=load<8>(sc,rr*(BN+8)+nn);
else if(rr<T::C_ROWS)output[i]=load<8>(sc,rr*(BN+8)+nn);
});
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
static_for<CHUNKS>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const int index=i*256+tid,rr=index/(BN/8),nn=index%(BN/8)*8;
if(rr<T::C_ROWS && part*T::C_ROWS+rr<valid_rows)
store<8>(gc,output[i],(part*T::C_ROWS+rr)*N+nn,0,number<2>{});
});
});
s_waitcnt_vmcnt(0_I);
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_46 {
using namespace opus_bpreshuffle_gfx950::registers::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM,int BN,int NB,int BD,int NN,int KK,int BC,bool XCD,int MM>
struct opus_gemm_blockscale_mxscale_bpreshuffle_bmm_direct_early_scale_pin_agpr_traits_gfx950 {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=4,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=false,C_AGPR=true;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && (BN==128 || BN==256) && (T_N==2 || T_N==4));
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM%16==0 && KK%128==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class T> __device__ __forceinline__
void opus_gemm_blockscale_mxscale_bpreshuffle_bmm_direct_early_scale_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
using namespace opus;
constexpr int BM=T::B_M,BN=T::B_N,NB=T::STAGES,BD=T::B_AHEAD,BS=T::B_SETS,P=T::PANELS;
constexpr int MR=T::MR,NR=T::NR,N=T::N,K=T::K,SC=T::SCALE_COLS;
constexpr int M=T::M;
const int tid=thread_id_x(),lane=tid%64,lane16=lane%16,kg=lane/16;
const int wave=__builtin_amdgcn_readfirstlane(tid/64),wm=0,wn=wave;
const unsigned pid=block_id_x(),gm=(M+BM-1)/BM,gn=N/BN,total=gm*gn;
const unsigned tile=T::XCD_ORDER?pid%8*((total+7)/8)+pid/8:pid;
if constexpr(T::XCD_ORDER)if(tile>=total)return;
const unsigned row=(T::XCD_ORDER?tile/gn:pid)*BM,col=(T::XCD_ORDER?tile%gn:block_id_y())*BN;
const unsigned valid_rows=M%BM==0?BM:(M<=BM?M:(row==(gm-1)*BM?M%BM:BM));
auto ga=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a)+row*K,static_cast<unsigned>(valid_rows*K));
auto gb=make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b)+col*K,static_cast<unsigned>(BN*K));
auto gsa=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),static_cast<unsigned>(M*SC));
auto gsb=make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),static_cast<unsigned>((N/128)*SC));
auto gc=make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c)+row*N+col,static_cast<unsigned>(((valid_rows-1)*N+BN)*2));
alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
auto sa=make_smem(reinterpret_cast<fp8_t*>(arena));auto ss=make_smem(arena);
using Base=decltype(make_mfma<fp8_t,fp8_t,fp32_t>(16_I,16_I,128_I,mfma_adaptor_swap_ab{}));
array<typename Base::vtype_c,MR*NR> c{};
array<array<array<typename Base::vtype_b,NR>,1>,BS> bregs;
auto issue_b=[&](auto bi,unsigned kt) {
constexpr int set=decltype(bi)::value;
static_for<1>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value;
const unsigned br=wn*(BN/T::T_N)+n*16,off=ks*2048+kg*256+lane16*16;
set_slice(bregs[set][ks][n],load<16>(gb,off,br*K+kt*2048,number<T::B_CACHE>{}),0_I,16_I);
set_slice(bregs[set][ks][n],load<16>(gb,off+1024,br*K+kt*2048,number<T::B_CACHE>{}),16_I,32_I);
});
});
__builtin_amdgcn_sched_barrier(0);
};
auto issue_a=[&](unsigned slot,unsigned kt) {
static_for<T::A_COPIES>([&](auto ii) {
constexpr int i=decltype(ii)::value;
const unsigned lin=(i*T::BLOCK_SIZE+tid)*16,r=lin/128,kc=(lin%128)^((r%8)*16);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+slot*T::STAGE_BYTES+(i*T::BLOCK_SIZE+wave*64)*16));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(ga.cached_rsrc,dst,16,r*K+kc,kt*128,0,0);
});
__builtin_amdgcn_asyncmark();
__builtin_amdgcn_sched_barrier(0);
};
static_for<T::INITIAL_B>([&](auto i){issue_b(i,decltype(i)::value);});
static_for<T::SA_COPIES>([&](auto ii){
constexpr int i=decltype(ii)::value;
const unsigned index=(i*T::BLOCK_SIZE+tid)*4,sr=index%BM,sc=index/BM<SC?index/BM:SC-1;
const unsigned source=sr<valid_rows?sc*M+row+sr:M*SC;
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SA+(i*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc,dst,4,source,0,0,0);
});
static_for<T::SB_COPIES>([&](auto ii){
constexpr int i=decltype(ii)::value;
constexpr int WORDS=T::SB_SLOTS?SC:SC/4;
const unsigned c=i*T::BLOCK_SIZE+tid,sr=c/WORDS<T::SB_ROWS?c/WORDS:T::SB_ROWS-1;
const unsigned source=(col/128+sr)*SC+(T::SB_SLOTS?c%WORDS:((c%WORDS+WORDS-sr%WORDS)%WORDS)*4);
auto* dst=reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(arena+T::OFF_SB+(i*T::BLOCK_SIZE+wave*64)*4));
__builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsb.cached_rsrc,dst,T::SB_SLOTS?1:4,source,0,0,0);
});
static_for<T::INITIAL_A>([&](auto i){issue_a(decltype(i)::value,decltype(i)::value);});
auto step=[&](unsigned t,unsigned slot,auto bi,auto wai,auto wbi,auto fai,auto fbi) {
constexpr int set=decltype(bi)::value,wa=decltype(wai)::value,wb=decltype(wbi)::value;
constexpr bool future_a=decltype(fai)::value,future_b=decltype(fbi)::value;
s_waitcnt_vmcnt(number<(wa>63?63:wa)>{});
__builtin_amdgcn_s_barrier();
if constexpr(future_b)issue_b(number<(set+BD)%BS>{},t+BD);
array<array<typename Base::vtype_a,MR>,1> af;
array<array<unsigned,MR>,1> as;
array<unsigned,1> bs;
static_for<1>([&](auto ki){
constexpr int ks=decltype(ki)::value;
const unsigned sr=wn*(BN/T::T_N)/128;
const unsigned sb_off=T::SB_SLOTS?(sr*SC+t)*4:sr*SC+((t/4+sr)%(SC/4))*4+t%4;
bs[ks]=load<1>(ss,T::OFF_SB+sb_off)[0];
__builtin_amdgcn_sched_barrier(0);
static_for<MR>([&](auto mi){
constexpr int m=decltype(mi)::value;
const unsigned ar=m*16+lane16;
const unsigned lo=slot*T::STAGE_BYTES+ar*128+((kg^(ar%8))*16);
const unsigned hi=slot*T::STAGE_BYTES+ar*128+(((kg+4)^(ar%8))*16);
set_slice(af[ks][m],load<16>(sa,lo),0_I,16_I);
set_slice(af[ks][m],load<16>(sa,hi),16_I,32_I);
});
static_for<MR>([&](auto mi){
constexpr int m=decltype(mi)::value;
as[ks][m]=load<1>(ss,T::OFF_SA+t*BM+m*16+lane16)[0];
});
});
__builtin_amdgcn_sched_barrier(0);
if constexpr(future_a)issue_a(slot==0?NB-1:slot-1,t+NB-1);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(number<(wb>63?63:wb)>{});
s_waitcnt_lgkmcnt(0_I);
static_for<1>([&](auto ki) {
constexpr int ks=decltype(ki)::value;
static_for<MR>([&](auto mi) {
constexpr int m=decltype(mi)::value;
static_for<NR>([&](auto ni) {
constexpr int n=decltype(ni)::value,ci=m*NR+n;
if constexpr(T::C_AGPR) {
[[clang::amdgpu_pin_agpr(ci*4)]]
c[ci]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
__builtin_bit_cast(vector_t<int,8>,bregs[set][ks][n]),
__builtin_bit_cast(vector_t<int,8>,af[ks][m]),c[ci],
0,0,0,static_cast<int>(bs[ks]),0,static_cast<int>(as[ks][m]));
} else {
c[ci]=Base{}(af[ks][m],bregs[set][ks][n],c[ci],static_cast<int>(as[ks][m]),static_cast<int>(bs[ks]),0_I,0_I);
}
});
});
});
};
if constexpr(BD==1) {
static_for<P>([&](auto ti){
constexpr int t=decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},
number<(t<T::STEADY)>{},number<(t+BD<P)>{});
});
} else {
constexpr int GROUPS=T::STEADY/BS,REGULAR_END=GROUPS*BS;
if constexpr(GROUPS>0) {
static_for<BS>([&](auto phase) {
constexpr int t=decltype(phase)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<true>{},number<true>{});
});
#pragma clang loop unroll(disable)
for(unsigned group=1;group<GROUPS;++group) {
static_for<BS>([&](auto phase) {
constexpr int ph=decltype(phase)::value;
const unsigned t=group*BS+ph;
const unsigned slot=__builtin_amdgcn_readfirstlane(t%NB);
step(t,slot,number<ph>{},number<T::wait_a(BS+ph)>{},number<T::wait_b(BS+ph)>{},number<true>{},number<true>{});
});
}
}
static_for<T::STEADY-REGULAR_END>([&](auto ti) {
constexpr int t=REGULAR_END+decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<true>{},number<(t+BD<P)>{});
});
static_for<NB-1>([&](auto ti) {
constexpr int t=T::STEADY+decltype(ti)::value;
step(t,t%NB,number<t%BS>{},number<T::wait_a(t)>{},number<T::wait_b(t)>{},number<false>{},number<(t+BD<P)>{});
});
}
constexpr int FRAGS=MR*NR;
const int side=(lane/16)%2;
static_for<FRAGS/2>([&](auto pi) {
constexpr int ci=decltype(pi)::value*2;
const auto first=__builtin_bit_cast(vector_t<unsigned,2>,cast<bf16_t>(c[ci]));
const auto second=__builtin_bit_cast(vector_t<unsigned,2>,cast<bf16_t>(c[ci+1]));
const auto swap0=__builtin_amdgcn_permlane16_swap(first[0],second[0],false,true);
const auto swap1=__builtin_amdgcn_permlane16_swap(first[1],second[1],false,true);
const vector_t<unsigned,4> packed{swap0[0],swap1[0],swap0[1],swap1[1]};
const int frag=ci+side,rr=wm*(BM/T::T_M)+(frag/NR)*16+lane16;
const int nn=wn*(BN/T::T_N)+(frag%NR)*16+(lane/32)*8;
if(rr<valid_rows) {
store<8>(gc,__builtin_bit_cast(vector_t<bf16_t,8>,packed),rr*N+nn);
}
});
if constexpr(FRAGS%2!=0) {
constexpr int ci=FRAGS-1;
const auto words=__builtin_bit_cast(vector_t<unsigned,2>,cast<bf16_t>(c[ci]));
const auto swap0=__builtin_amdgcn_permlane16_swap(words[0],words[0],false,true);
const auto swap1=__builtin_amdgcn_permlane16_swap(words[1],words[1],false,true);
if(side==0) {
const vector_t<unsigned,4> packed{swap0[0],swap1[0],swap0[1],swap1[1]};
const int rr=wm*(BM/T::T_M)+(ci/NR)*16+lane16;
const int nn=wn*(BN/T::T_N)+(ci%NR)*16+(lane/32)*8;
if(rr<valid_rows) {
store<8>(gc,__builtin_bit_cast(vector_t<bf16_t,8>,packed),rr*N+nn);
}
}
}
s_waitcnt_vmcnt(0_I);
}
#endif
}

namespace opus_bpreshuffle_gfx950::b_direct::schedule_47 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BM, int BN, int NB, int BD, int NN, int KK, int BC, bool XCD,
int MM, int WaveN, bool BScaleSlots = true,
bool EarlyBScale = false, bool TensorBase = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_bmm_direct_pin_agpr_traits_gfx950 {
static constexpr int B_M=BM,B_N=BN,B_K=128,T_M=1,T_N=WaveN,T_K=1,BLOCK_SIZE=T_N*64;
static constexpr int MR=BM/16,NR=BN/(T_N*16),STAGES=NB,B_AHEAD=BD,B_SETS=BD+1,N=NN,K=KK,M=MM,B_CACHE=BC;
static constexpr bool XCD_ORDER=XCD,SB_SLOTS=BScaleSlots,C_AGPR=true;
static constexpr bool EARLY_B_SCALE=EarlyBScale,TENSOR_BASE=TensorBase;
static constexpr int SCALE_COLS=KK/128,PANELS=KK/128,A_BYTES=BM*128,STAGE_BYTES=A_BYTES;
static constexpr int SA_ROWS=BM,SA_BYTES=((BM*SCALE_COLS+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4),SB_ROWS=(BN+127)/128,SB_BYTES=((SB_ROWS*SCALE_COLS*(SB_SLOTS?4:1)+BLOCK_SIZE*4-1)/(BLOCK_SIZE*4))*(BLOCK_SIZE*4);
static constexpr int OFF_SA=NB*STAGE_BYTES,OFF_SB=OFF_SA+SA_BYTES,PIPELINE_BYTES=OFF_SB+SB_BYTES,LDS_BYTES=PIPELINE_BYTES;
static constexpr int A_COPIES=A_BYTES/(BLOCK_SIZE*16),B_ISSUES=NR*2,SA_COPIES=SA_BYTES/(BLOCK_SIZE*4),SB_COPIES=SB_BYTES/(BLOCK_SIZE*4),SCALE_ISSUES=SA_COPIES+SB_COPIES;
static constexpr int INITIAL_A=NB-1,INITIAL_B=BD,STEADY=PANELS-INITIAL_A;
static constexpr int wait_a(int t) {
int issued=INITIAL_B*B_ISSUES+SCALE_ISSUES,ae=-1;
for(int i=0;i<INITIAL_A;++i){issued+=A_COPIES;if(i==t)ae=issued;}
for(int s=0;s<t;++s){
if(s+BD<PANELS)issued+=B_ISSUES;
if(s+NB-1<PANELS){issued+=A_COPIES;if(s+NB-1==t)ae=issued;}
}
return issued-ae;
}
static constexpr int wait_b(int t) {
int issued=0,be=-1;
for(int i=0;i<INITIAL_B;++i){issued+=B_ISSUES;if(i==t)be=issued;}
issued+=SCALE_ISSUES+INITIAL_A*A_COPIES;
for(int s=0;s<=t;++s){
if(s+BD<PANELS){issued+=B_ISSUES;if(s+BD==t)be=issued;}
if(s+NB-1<PANELS)issued+=A_COPIES;
}
return issued-be;
}
static_assert((BM==64 || BM==112 || BM==128 || BM==224) && BN==256 && (T_N==2 || T_N==4));
static_assert(A_BYTES%(BLOCK_SIZE*16)==0 && MR*NR*4<=256);
static_assert((NB==2 || NB==3 || NB==4) && (BD==1 || BD==2) && MM>0 && MM%16==0 && KK%128==0 && NN%BN==0);
static_assert(SB_SLOTS || SCALE_COLS%4==0);
static_assert(PANELS>=NB-1 && LDS_BYTES<=160*1024);
};
}
