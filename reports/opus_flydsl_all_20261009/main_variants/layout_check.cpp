#include "helpers.h"
#include "candidate/traits.cuh"
#include "contract.h"
#include <cassert>
#include <vector>
#include <iostream>
template<class T>void check(int loops) {
 for(int m:{16,144,160,176,320,336,544,672,864,992,1088,4096})for(int row=0;row<m;row+=T::B_M){
  for(int begin=0;begin<loops;begin+=T::SCALE_PANEL){
   std::vector<int> coverage(T::SFA_BYTES);
   opus::static_for<T::SFA_PASSES>([&](auto pass){
    for(int wave=0;wave<T::NUM_WAVES;++wave)for(int lane=0;lane<64;++lane){
     auto g=checked::make_layout_gsfa_scale<T,decltype(pass)::value>(lane,wave%T::T_M,wave/T::T_M,m);
     auto s=checked::make_layout_ssfa_scale<T,decltype(pass)::value>(lane,wave%T::T_M,wave/T::T_M);
     int offset=opus::layout_to_offsets<T::VEC_SCALE_A>(s)[0];int local_k=offset/T::B_M,r=offset%T::B_M;
     if(offset<T::SFA_BYTES&&begin+local_k<loops){
      assert(opus::layout_to_offsets<T::VEC_SCALE_A>(g)[0]==local_k*m+r);
      assert(r%16==0&&r+16<=T::B_M);
      if(row+r<m)assert((begin+local_k)*m+row+r+15<m*loops);
      for(int b=0;b<16;++b)++coverage[offset+b];
     }
    }
   });
   for(int k=0;k<T::SCALE_PANEL&&begin+k<loops;++k)for(int r=0;r<T::B_M;++r)assert(coverage[k*T::B_M+r]==1);
   std::vector<int> bcoverage(T::SCALE_PANEL);
   for(int w=0;w<T::NUM_WAVES;++w)for(int l=0;l<64;++l){
    int offset=opus::layout_to_offsets<1>(checked::make_layout_ssfb_scale<T>(l,w%T::T_M,w/T::T_M))[0];
    if(offset<T::SCALE_PANEL&&begin+offset<loops)++bcoverage[offset];
   }
   for(int k=0;k<T::SCALE_PANEL&&begin+k<loops;++k)assert(bcoverage[k]==1);
  }
 }
 // Current matrix ring: first tiles0/1; every future tile is issued once.
 std::vector<int> issued(loops);issued[0]++;if(loops>1)issued[1]++;
 for(int kt=0;kt+1<loops;++kt)if(kt+2<loops)issued[kt+2]++;
 for(int count:issued)assert(count==1);
 static_assert(T::B_M*T::C_LDS_ROW_STRIDE_ELEMS*2<=T::LDS_BYTES);
 // C copies are contiguous vectors wholly within each row and tile.
 std::vector<int> c(T::B_M*T::B_N);
 for(int pass=0;pass<T::OUTPUT_PASSES;++pass)for(int tid=0;tid<T::BLOCK_SIZE;++tid)for(int b=0;b<T::VEC_OUTPUT;++b)++c[pass*T::BLOCK_SIZE*T::VEC_OUTPUT+tid*T::VEC_OUTPUT+b];
 for(int count:c)assert(count==1);
}
int main(){
 for(int loops:{1,3,6,8,12,24,32,56,64,128}){
  check<opus_private_geometry_traits<64,2,32>>(loops);check<opus_private_geometry_traits<96,2,32>>(loops);check<opus_private_geometry_traits<128,2,32>>(loops);
 }
 check<opus_private_geometry_traits<96,2,8,384>>(3);check<opus_private_geometry_traits<96,2,8,768>>(6);
 check<opus_private_geometry_traits<128,2,16,1536>>(12);check<opus_private_geometry_traits<160,2,16,1536>>(12);
 assert(main_shape_valid(672,7168,384,128,384));assert(!main_shape_valid(15,7168,384,128,384));assert(!main_shape_valid(16,7168,768,128,384));assert(!main_shape_valid(65536,65536,1536,128,0));
 std::cout<<"passed actual SFA/SFB layout coverage, 12 M/tail classes, panels8/16/32, loops1..128, matrix issue once, C copy once and launcher dimension guards; no HIP runtime calls\n";
}
