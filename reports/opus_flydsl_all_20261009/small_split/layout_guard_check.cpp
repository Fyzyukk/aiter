#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "frozen_layout_helpers.h"
#include "candidate/traits.cuh"
#include "contract.h"
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

using namespace opus;

// These checks call the actual frozen C and B layout helpers with host
// attributes. No MFMA arithmetic, buffer operation, or HIP API is called.
template<class T> void output_layout(int m, int n) {
    auto mma = make_tiled_mma<fp8_t,fp8_t,fp32_t>(seq<T::E_M,T::E_N,1>{},
        seq<T::T_M,T::T_N,1>{},seq<16,16,128>{},mfma_adaptor_swap_ab{});
    for (int row=0;row<m;row+=T::B_M) {
        std::vector<int> covered(T::B_M*T::B_N);
        for (int wave=0;wave<T::T_M*T::T_N;++wave)
            for (int lane=0;lane<64;++lane) {
                const int wm=wave%T::T_M,wn=wave/T::T_M;
                auto u=partition_layout_c<4>(mma,make_tuple(n,1_I),
                    make_tuple(wm,lane%mma.grpn_c,wn,lane/mma.grpn_c));
                auto offsets=layout_to_offsets<4>(u);
                for(int i=0;i<T::E_M*T::E_N;++i)
                    for(int elem=0;elem<4;++elem) {
                        const int address=offsets[i]+elem,r=address/n,c=address%n;
                        assert(r>=0&&r<T::B_M&&c>=0&&c<T::B_N);
                        ++covered[r*T::B_N+c];
                        if(row+r<m) assert((row+r)*n+c<m*n);
                    }
            }
        for(int count:covered) assert(count==1);
    }
}

template<class T> void fine_operands(int m) {
    constexpr int k=7168,pad=1024+T::smem_padding;
    std::vector<int> bmap(T::B_STAGE,-1),acover(T::B_M*128);
    for(int wave=0;wave<T::NUM_WAVES;++wave)
        for(int lane=0;lane<64;++lane) {
            const int wm=wave%T::T_M,wn=wave/T::T_M;
            const auto g=checked_layout::make_layout_gb_scale<T>(lane,wm,wn,k);
            const auto s=checked_layout::make_layout_sb_scale<T>(wm,wn);
            const auto go=layout_to_offsets<16>(g),so=layout_to_offsets<16>(s);
            for(int i=0;i<T::B_N*128/(T::BLOCK_SIZE*16);++i)
                for(int e=0;e<16;++e) {
                    const int target=so[i]+lane*16+e;
                    assert(target>=0&&target<T::B_STAGE&&bmap[target]==-1);
                    bmap[target]=go[i]+e;
                }
            for(int pass=0;pass<(T::B_M/8+T::NUM_WAVES-1)/T::NUM_WAVES;++pass) {
                const int group=pass*T::NUM_WAVES+wave;
                if(group>=T::B_M/8) continue;
                const int r=(group/T::T_M)*8*T::T_M+(lane/8)*T::T_M+wm;
                for(int e=0;e<16;++e) ++acover[r*128+(lane%8)*16+e];
            }
        }
    for(int count:acover) assert(count==1);
    for(int wave=0;wave<T::NUM_WAVES;++wave)
        for(int lane=0;lane<64;++lane) {
            const int wn=wave/T::T_M;
            const auto r=checked_layout::make_layout_rb_scale<T>(lane,wn);
            const auto ro=layout_to_offsets<16>(r);
            for(int i=0;i<T::E_N*2;++i)
                for(int e=0;e<16;++e) {
                    const int ni=i/2,nr=(ni*T::T_N+wn)*16;
                    const int expected=nr*k+lane*16+(i%2)*1024+e;
                    assert(bmap[ro[i]+e]==expected);
                }
        }
    for(int row=0;row<m;row+=T::B_M)
        for(int loops=1;loops<=T::MAX_LOOPS;++loops) {
            std::vector<int> scale(T::B_M*loops);
            for(int index=0;index<T::B_M*loops;index+=16) {
                const int r=index%T::B_M,kt=index/T::B_M;
                for(int e=0;e<16;++e) {
                    ++scale[index+e];
                    if(row+r+e<m) assert(kt*m+row+r+e<m*loops);
                }
            }
            for(int count:scale) assert(count==1);
        }
    assert(T::lds_bytes(16384)<=160*1024);
    (void)pad;
}

int main() {
    const int mvalues[]={1,15,16,17,31,48,64,80,96,112,128,144,512,2048};
    for(int m:mvalues) {
        output_layout<opus_private_register_split4_traits<16,16,1>>(m,256);
        output_layout<opus_private_register_split4_traits<16,16,2>>(m,256);
        output_layout<opus_private_register_split4_traits<16,32,1>>(m,256);
        output_layout<opus_private_register_split4_traits<16,32,2>>(m,256);
        output_layout<opus_private_register_split4_traits<32,32,1>>(m,256);
        output_layout<opus_private_register_split4_traits<32,64,1>>(m,256);
        output_layout<opus_private_narrow_fine_traits<48,64,1,4,1>>(m,256);
        output_layout<opus_private_narrow_fine_traits<48,64,1,4,2>>(m,256);
        output_layout<opus_private_narrow_fine_traits<64,128,2,2,1>>(m,256);
        output_layout<opus_private_narrow_fine_traits<64,128,2,2,2>>(m,256);
        output_layout<opus_private_narrow_fine_traits<96,128,2,2,1>>(m,256);
        output_layout<opus_private_narrow_fine_traits<96,128,2,2,2>>(m,256);
    }
    fine_operands<opus_private_narrow_fine_traits<48,64,1,4,1>>(113);
    fine_operands<opus_private_narrow_fine_traits<64,128,2,2,1>>(113);
    fine_operands<opus_private_narrow_fine_traits<96,128,2,2,1>>(113);
    PrivateConfig cfg{};
    const int ids[]={110,111,120,121,130,140,210,211,220,221,230,231};
    const void* a=reinterpret_cast<const void*>(uintptr_t(0x100000000));
    const void* b=reinterpret_cast<const void*>(uintptr_t(0x200000000));
    const void* sa=reinterpret_cast<const void*>(uintptr_t(0x300000000));
    const void* sb=reinterpret_cast<const void*>(uintptr_t(0x400000001));
    void* c=reinterpret_cast<void*>(uintptr_t(0x500000000));
    void* ws=reinterpret_cast<void*>(uintptr_t(0x600000000));
    for(int id:ids) {
        assert(private_config(id,true,112,768,7168,cfg));
        const auto bytes=private_workspace_bytes(cfg,112,768);
        assert(private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,112,768,7168));
        assert(!private_pointers_valid(cfg,a,b,sa,sb,const_cast<void*>(a),bytes?ws:nullptr,bytes,112,768,7168));
        assert(!private_pointers_valid(cfg,nullptr,b,sa,sb,c,bytes?ws:nullptr,bytes,112,768,7168));
        if(bytes) {
            assert(!private_pointers_valid(cfg,a,b,sa,sb,c,ws,bytes-1,112,768,7168));
            assert(!private_pointers_valid(cfg,a,b,sa,sb,c,c,bytes,112,768,7168));
            assert(!private_pointers_valid(cfg,a,b,sa,sb,c,reinterpret_cast<void*>(uintptr_t(0x600000001)),bytes,112,768,7168));
        } else assert(!private_pointers_valid(cfg,a,b,sa,sb,c,ws,16,112,768,7168));
        assert(!private_config(id,true,INT_MAX,INT_MAX,INT_MAX,cfg));
        assert(!private_config(id,true,0,128,128,cfg));
        assert(!private_config(id,true,16,129,128,cfg));
        assert(!private_config(id,true,16,128,129,cfg));
        assert(!private_config(id,true,16,INT_MAX-127,16384,cfg));
        assert(!private_config(id,true,16,128,16512,cfg));
    }
    assert(!private_config(999,true,16,128,128,cfg));
    std::cout<<"passed: frozen C layout coverage for 12 variants and 14 M tails; frozen B producer/consumer layout and fine A/scales for 3 geometries; actual workspace/alignment/overlap/dimension guards; no HIP runtime calls\n";
}
