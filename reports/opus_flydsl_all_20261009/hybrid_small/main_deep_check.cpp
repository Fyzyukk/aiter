#include "helpers.h"
#include "main_traits.cuh"
#include <cassert>
#include <iostream>
#include <vector>
#include <array>

long long matrix_bytes=0,sfa_bytes=0,sfb_bytes=0;
template<class T> void matrix() {
 for(int k:{128,384,768,1024,1536,3072,7168,16384}) {
  std::vector<int> a(T::A_STAGE,-1),b(T::B_STAGE,-1);
  for(int wave=0;wave<T::NUM_WAVES;wave++)for(int lane=0;lane<64;lane++) {
   int wm=wave%T::T_M,wn=wave/T::T_M;
   auto ga=opus::layout_to_offsets<16>(checked::make_layout_ga_scale<T>(lane,wm,wn,k));
   auto sa=opus::layout_to_offsets<16>(checked::make_layout_sa_scale<T>(wm,wn));
   auto gb=opus::layout_to_offsets<16>(checked::make_layout_gb_scale<T>(lane,wm,wn,k));
   auto sb=opus::layout_to_offsets<16>(checked::make_layout_sb_scale<T>(wm,wn));
   assert(int(ga.size())==T::A_VMEM_INSTRUCTIONS && int(gb.size())==T::B_VMEM_INSTRUCTIONS);
   for(int i=0;i<int(ga.size());i++)for(int byte=0;byte<16;byte++) {
    int dest=sa[i]+lane*16+byte;assert(dest<T::A_STAGE && a[dest]==-1);a[dest]=ga[i]+byte;
   }
   for(int i=0;i<int(gb.size());i++)for(int byte=0;byte<16;byte++) {
    int dest=sb[i]+lane*16+byte;assert(dest<T::B_STAGE && b[dest]==-1);b[dest]=gb[i]+byte;
   }
  }
  for(int wave=0;wave<T::NUM_WAVES;wave++)for(int lane=0;lane<64;lane++) {
   int wm=wave%T::T_M,wn=wave/T::T_M;
   auto ra=opus::layout_to_offsets<16>(checked::make_layout_ra_scale<T>(lane,wm));
   auto rb=opus::layout_to_offsets<16>(checked::make_layout_rb_scale<T>(lane,wn));
   for(int mi=0;mi<T::E_M;mi++)for(int half=0;half<2;half++)for(int byte=0;byte<16;byte++) {
    int r=(mi*T::T_M+wm)*16+lane%16;
    int expected=r*k+(lane/16)*16+half*64+byte;
    assert(a[ra[mi*2+half]+byte]==expected);matrix_bytes++;
   }
   for(int ni=0;ni<T::E_N;ni++)for(int half=0;half<2;half++)for(int byte=0;byte<16;byte++) {
    int nr=(ni*T::T_N+wn)*16;
    assert(b[rb[ni*2+half]+byte]==nr*k+lane*16+half*1024+byte);matrix_bytes++;
   }
  }
 }
}

using Word=std::array<int,4>;
Word perm(Word first,Word second,unsigned mask) {
 std::array<int,8> source{second[0],second[1],second[2],second[3],first[0],first[1],first[2],first[3]};
 return {source[(mask>>0)&7],source[(mask>>8)&7],source[(mask>>16)&7],source[(mask>>24)&7]};
}
template<class T> void pin_scales(int m=512,int row=256) {
 constexpr int loops=T::FIXED_K/128,Panel=T::SCALE_PANEL_K_CAPACITY;
 const int gsa_bytes=m*loops-row;
 assert(gsa_bytes>0 && gsa_bytes==m*loops-row);
 for(int begin=0;begin<loops;begin+=Panel) {
  std::vector<int> sfa(T::SFA_PANEL_BYTES,-1),coverage(T::SFA_PANEL_BYTES);
  std::vector<int> sfb(T::SFB_PANEL_BYTES,-1),bcoverage(T::SFB_PANEL_BYTES);
  for(int pass=0;pass<T::SFA_PASSES_PER_CACHE_PANEL;pass++)for(int wave=0;wave<4;wave++) {
   int first=begin+pass*T::SFA_K_COLUMNS_PER_PASS+wave*T::SFA_K_COLUMNS_PER_WAVE;
   if(first>=loops)continue;
   int wm=wave%T::T_M,wn=wave/T::T_M;
   // Symbolic byte identity retains global SFA address through DPP/perm.
   std::array<std::array<int,16>,64> raw{};
   std::array<int,64> destinations{};
   for(int lane=0;lane<64;lane++) {
    auto ga=opus::layout_to_offsets<16>(checked::make_layout_gsfa_scale<T>(lane,wm,wn,m));
    auto ss=opus::layout_to_offsets<4>(checked::make_layout_ssfa_scale<T>(lane,wm,wn));
    destinations[lane]=ss[pass];
    for(int byte=0;byte<16;byte++) {
     int rel=ga[pass]+begin*m+byte;
     raw[lane][byte]=row+rel< m*loops ? row+rel : -2;
     // Descriptor bounds block all K-tail accesses, even when a pass has
     // fewer than four remaining K columns. No out-of-range host load occurs.
     assert(raw[lane][byte]==-2 || (rel>=0 && rel<gsa_bytes));
    }
   }
   for(int word=0;word<4;word++) {
    std::array<Word,64> z{},pair{},out{};
    for(int lane=0;lane<64;lane++)for(int byte=0;byte<4;byte++)z[lane][byte]=raw[lane][word*4+byte];
    for(int lane=0;lane<64;lane++)pair[lane]=perm(z[lane^1],z[lane],lane%2 ? 0x03070105u:0x06020400u);
    for(int lane=0;lane<64;lane++)out[lane]=perm(pair[lane^2],pair[lane],lane%4/2 ? 0x03020706u:0x05040100u);
    for(int lane=0;lane<64;lane++)for(int byte=0;byte<4;byte++) {
     int dest=destinations[lane]+word*32+byte;
     assert(dest>=0 && dest<T::SFA_PANEL_BYTES);sfa[dest]=out[lane][byte];coverage[dest]++;
    }
   }
  }
  for(int wave=0;wave<4;wave++)for(int lane=0;lane<64;lane++) {
   bool producer=wave<T::SCALE_N_HALVES && lane<Panel;
   auto gb=opus::layout_to_offsets<1>(checked::make_layout_gsfb_scale<T>(lane,wave));
   auto sb=opus::layout_to_offsets<4>(checked::make_layout_ssfb_scale<T>(lane,wave));
   if(producer) {
    assert(gb[0]==lane);
    for(int byte=0;byte<4;byte++) {
     int dest=sb[0]+byte;assert(dest>=0 && dest<T::SFB_PANEL_BYTES);
     sfb[dest]=begin+lane<loops ? wave*loops+begin+lane:-2;bcoverage[dest]++;
    }
   } else if(wave<T::SCALE_N_HALVES) {
    assert(sb[0]>=T::SFB_PANEL_BYTES); // Missing lane<Panel guard would overrun.
   }
  }
  for(int wave=0;wave<4;wave++)for(int lane=0;lane<64;lane++) {
   int wm=wave%T::T_M;
   auto rsa=opus::layout_to_offsets<4>(checked::make_layout_rsfa_scale<T>(lane,wm));
   for(int kt=begin;kt<loops && kt<begin+Panel;kt++)for(int half=0;half<2;half++)for(int mi=0;mi<T::E_M;mi++) {
    int dest=rsa[0]+(kt&(Panel-1))*T::SFA_PANEL_PITCH+half*4+mi;
    int expected=kt*m+row+half*T::HALF_B_M+(mi*T::T_M+wm)*16+lane%16;
    assert(coverage[dest]==1 && sfa[dest]==expected);sfa_bytes++;
   }
   for(int kt=begin;kt<loops && kt<begin+Panel;kt++)for(int half=0;half<2;half++) {
    int dest=(kt&(Panel-1))*T::SCALE_N_HALVES*4+half*4;
    assert(bcoverage[dest]==1 && sfb[dest]==half*loops+kt);sfb_bytes++;
   }
  }
 }
}

int main() {
 matrix<opus_private_geometry_traits<64,2,32>>();matrix<opus_private_geometry_traits<96,2,32>>();
 matrix<opus_private_geometry_traits<128,2,32>>();matrix<opus_private_geometry_traits<160,2,32>>();
 pin_scales<opus_private_pin_fixed_traits<384,16,false>>();pin_scales<opus_private_pin_fixed_traits<1536,16,false>>();
 pin_scales<opus_private_pin_fixed_traits<3072,32,false>>();pin_scales<opus_private_pin_fixed_traits<7168,64,false>>();
 pin_scales<opus_private_pin_fixed_traits<16384,64,false>>();
 pin_scales<opus_private_pin_fixed_traits<384,16,true>>();pin_scales<opus_private_pin_fixed_traits<1536,16,true>>();
 pin_scales<opus_private_pin_fixed_traits<3072,32,true>>();pin_scales<opus_private_pin_fixed_traits<7168,64,true>>();
 pin_scales<opus_private_pin_fixed_traits<16384,64,true>>();
 std::cout<<"{\"status\":\"passed\",\"geometry_matrix_address_bytes\":"<<matrix_bytes
          <<",\"pin_sfa_consumer_bytes\":"<<sfa_bytes<<",\"pin_sfb_consumer_bytes\":"<<sfb_bytes
          <<",\"BM_variants\":4,\"K_strides\":8,\"pin_fixed_reset_variants\":10,\"gpu_operations\":0}\n";
}
