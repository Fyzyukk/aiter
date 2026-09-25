#include <hip/hip_runtime.h>
#include <opus/opus.hpp>
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh"
#include <cassert>
#include <iostream>
#include <vector>
using opus::operator""_I;
#include "host_layout_helpers.inc"

template<int Waves> void check() {
    using T = opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<Waves>;
    std::vector<int> a(T::A_STAGE,-1), b(T::B_STAGE,-1), covered(T::B_M*T::B_N,0);
    for(int tid=0;tid<T::BLOCK_SIZE;++tid) {
        const int lane=tid%64, wm=(tid/64)%T::T_M, wn=(tid/64)/T::T_M;
        auto ga=opus::layout_to_offsets<16>(make_layout_ga_scale<T>(lane,wm,wn,128));
        auto sa=opus::layout_to_offsets<16>(make_layout_sa_scale<T>(wm,wn));
        auto gb=opus::layout_to_offsets<16>(make_layout_gb_scale<T>(lane,wm,wn,128));
        auto sb=opus::layout_to_offsets<16>(make_layout_sb_scale<T>(wm,wn));
        for(int i=0;i<ga.size();++i) for(int j=0;j<16;++j) {
            int dst=sa[i]+lane*16+j;
            assert(dst>=0 && dst<T::A_STAGE && a[dst]==-1);
            a[dst]=ga[i]+j;
        }
        for(int i=0;i<gb.size();++i) for(int j=0;j<16;++j) {
            int dst=sb[i]+lane*16+j;
            assert(dst>=0 && dst<T::B_STAGE && b[dst]==-1);
            b[dst]=gb[i]+j;
        }
    }
    auto mma=opus::make_tiled_mma<opus::fp8_t,opus::fp8_t,opus::fp32_t>(
        opus::seq<T::E_M,T::E_N,1>{},opus::seq<T::T_M,T::T_N,1>{},
        opus::seq<16,16,128>{},opus::mfma_adaptor_swap_ab{});
    int old_a_mismatches=0;
    for(int tid=0;tid<T::BLOCK_SIZE;++tid) {
        const int lane=tid%64, wm=(tid/64)%T::T_M, wn=(tid/64)/T::T_M;
        for(int mi=0;mi<T::E_M;++mi) for(int c=0;c<2;++c) for(int j=0;j<16;++j) {
            int dst=T::a_lds_offset(wm,lane,mi,c)+j;
            int row=mi*T::T_M*16+wm*16+lane%16;
            int k=(lane/16)*16+c*64+j;
            assert(dst>=0 && dst<T::A_STAGE && a[dst]==row*128+k);
        }
        auto old_ra=opus::layout_to_offsets<16>(make_layout_ra_scale<T>(lane,wm));
        for(int mi=0;mi<T::E_M;++mi) for(int c=0;c<2;++c)
            old_a_mismatches += old_ra[mi*2+c]!=T::a_lds_offset(wm,lane,mi,c);
        auto rb=opus::layout_to_offsets<16>(make_layout_rb_scale<T>(lane,wn));
        for(int ni=0;ni<T::E_N;++ni) for(int c=0;c<2;++c) for(int j=0;j<16;++j) {
            int dst=rb[ni*2+c]+j;
            int n_group=ni*T::T_N+wn;
            int expected=n_group*16*128+c*64*16+lane*16+j;
            assert(dst>=0 && dst<T::B_STAGE && b[dst]==expected);
        }
        auto coord=opus::make_tuple(wm,lane%mma.grpn_c,wn,lane/mma.grpn_c);
        auto uc=opus::partition_layout_c<4>(mma,opus::make_tuple(256,1_I),coord);
        auto offsets=opus::layout_to_offsets<4>(uc);
        assert(offsets.size()==T::E_M*T::E_N);
        for(int i=0;i<offsets.size();++i) for(int j=0;j<4;++j) {
            int index=offsets[i]+j;
            assert(index>=0 && index<T::B_M*T::B_N);
            ++covered[index];
        }
    }
    for(int c:covered) assert(c==1);
    assert((Waves==4 && old_a_mismatches==0) || (Waves==8 && old_a_mismatches>0));
    for(int stride:{256,6144,7168}) {
        std::fill(covered.begin(),covered.end(),0);
        for(int tid=0;tid<T::BLOCK_SIZE;++tid) {
            int lane=tid%64,wm=(tid/64)%T::T_M,wn=(tid/64)/T::T_M;
            auto coord=opus::make_tuple(wm,lane%mma.grpn_c,wn,lane/mma.grpn_c);
            auto offsets=opus::layout_to_offsets<4>(opus::partition_layout_c<4>(mma,opus::make_tuple(stride,1_I),coord));
            for(int i=0;i<offsets.size();++i) for(int j=0;j<4;++j) {
                int off=offsets[i]+j, row=off/stride, col=off%stride;
                assert(row>=0 && row<192 && col>=0 && col<256);
                ++covered[row*256+col];
                for(int valid_rows:{64,128,192}) for(int origin_col:{0,stride-256})
                    assert((off<(valid_rows*stride-origin_col))==(row<valid_rows));
            }
        }
        for(int c:covered) assert(c==1);
    }
    std::cout<<Waves<<" waves: A/B bytes match coordinates; output covered once; M tails bounded; legacy A offset differences="<<old_a_mismatches<<"\n";
}
int main() { check<4>();check<8>(); }
