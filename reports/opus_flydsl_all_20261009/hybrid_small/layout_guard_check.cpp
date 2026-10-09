#include "frozen_layout_helpers.h"
#include "candidate/traits.cuh"
#include "contract.h"
#include <cassert>
#include <iostream>
#include <map>
#include <vector>

long long checked_bytes=0, ring_consumptions=0, scale_consumptions=0, checked_outputs=0;

template<int Actual>
void addresses() {
    using T=typename opus_private_small_baseline<Actual>::type;
    for(const int stride : {128,384,768,1024,1536,3072,7168,16384}) {
    std::vector<int> a(T::A_STAGE,-1),b(T::B_STAGE,-1);
    for(int wave=0;wave<T::NUM_WAVES;wave++) for(int lane=0;lane<64;lane++) {
        int wm=wave%T::T_M,wn=wave/T::T_M;
        int load_lane=T::XOR_LDS ? lane ^ ((wave%4)*2) : lane;
        const auto ga=opus::layout_to_offsets<16>(checked_layout::make_layout_ga_scale<T>(load_lane,wm,wn,stride));
        const auto sa=opus::layout_to_offsets<16>(checked_layout::make_layout_sa_scale<T>(wm,wn));
        const auto gb=opus::layout_to_offsets<16>(checked_layout::make_layout_gb_scale<T>(lane,wm,wn,stride));
        const auto sb=opus::layout_to_offsets<16>(checked_layout::make_layout_sb_scale<T>(wm,wn));
        for(int i=0;i<int(ga.size());i++) for(int byte=0;byte<16;byte++) {
            int dest=sa[i]+lane*16+byte;
            assert(dest>=0 && dest<T::A_STAGE && a[dest]==-1);
            a[dest]=ga[i]+byte;
        }
        for(int i=0;i<int(gb.size());i++) for(int byte=0;byte<16;byte++) {
            int mask=T::XOR_LDS ? ((sb[i]>>10)&3)<<5 : 0;
            int dest=sb[i]+lane*16+byte;
            assert(dest>=0 && dest<T::B_STAGE && b[dest]==-1);
            b[dest]=(gb[i]^mask)+byte;
        }
    }
    for(int wave=0;wave<T::NUM_WAVES;wave++) for(int lane=0;lane<64;lane++) {
        int wm=wave%T::T_M,wn=wave/T::T_M;
        auto rb=opus::layout_to_offsets<16>(checked_layout::make_layout_rb_scale<T>(lane,wn));
        for(int mi=0;mi<T::E_M;mi++) {
            int r=(mi*T::T_M+wm)*16+lane%16;
            int offset=((r/(8*T::T_M))*T::T_M+r%T::T_M)*(1024+T::smem_padding)+
                ((r/T::T_M)%8)*128+(lane/16)*16;
            for(int half=0;half<2;half++) for(int byte=0;byte<16;byte++) {
                int read=offset+half*64;
                if(T::XOR_LDS) read^=((read>>10)&3)<<5;
                assert(a[read+byte]==r*stride+(lane/16)*16+half*64+byte);
                checked_bytes++;
            }
        }
        for(int ni=0;ni<T::E_N;ni++) for(int half=0;half<2;half++) for(int byte=0;byte<16;byte++) {
            int read=rb[ni*2+half];
            if(T::XOR_LDS) read^=((read>>10)&3)<<5;
            int direct=(ni*T::T_N+wn)*16*stride+lane*16+half*1024+byte;
            assert(read>=0 && read+byte<T::B_STAGE && b[read+byte]==direct);
            checked_bytes++;
            for(int kt : {0,1,2,3,5,7,8,11,12,23,55,127}) {
                if(kt>=stride/128) continue;
                int global=direct+kt*2048;
                assert(global>=0 && global< T::B_N*stride);
            }
        }
    }
    }
}

template<int Actual,int Ahead>
void rings_and_sizes() {
    using T=opus_private_small_traits<Actual,Ahead>;
    using B=typename opus_private_small_baseline<Actual>::type;
    for(int loops=1;loops<=128;loops++) {
        std::vector<int> a(T::NUM_STAGES,-1),b(T::B_SLOTS,-1);
        for(int i=0;i<T::NUM_STAGES && i<loops;i++) a[i]=i;
        for(int i=0;i<T::B_SLOTS && i<loops;i++) b[i]=i;
        for(int base=0;base<loops;base+=T::RING_PERIOD) for(int i=0;i<T::RING_PERIOD && base+i<loops;i++) {
            int kt=base+i,si=i%T::NUM_STAGES,bi=i%T::B_SLOTS;
            assert(si==kt%T::NUM_STAGES && bi==kt%T::B_SLOTS);
            assert(a[si]==kt && b[bi]==kt);
            ring_consumptions++;
            // The implementation copies A/B/scales locally, completes all LDS
            // reads, barriers, then reuses the A and B slots before MFMA.
            a[si]=kt+T::NUM_STAGES<loops ? kt+T::NUM_STAGES : -1;
            b[bi]=kt+T::B_SLOTS<loops ? kt+T::B_SLOTS : -1;
        }
        int k=loops*128,stages=loops<T::NUM_STAGES ? loops:T::NUM_STAGES;
        int matrix=stages*T::A_STAGE,scale=T::REGISTER_SCALES ? 0:(T::B_M+T::B_GROUPS)*loops;
        assert(T::lds_bytes(k)>=matrix+scale && T::lds_bytes(k)>=T::C_BYTES);
        assert(T::lds_bytes(k)<=160*1024 && T::lds_bytes(k)<=B::lds_bytes(k));
        assert(T::B_M==B::B_M && T::B_N==B::B_N && T::T_M==B::T_M && T::T_N==B::T_N &&
               T::SPLIT_K==B::SPLIT_K && T::OUTPUT==B::OUTPUT && T::XOR_LDS==B::XOR_LDS &&
               T::REGISTER_SCALES==B::REGISTER_SCALES && T::FIXED_K==B::FIXED_K);
    }
}

template<int Actual>
void scales_outputs_and_guards() {
    using T=typename opus_private_small_baseline<Actual>::type;
    const int max_m=Actual==9047 || Actual==9049 ? 512:2048;
    for(int m : {1,2,4,8,16,17,31,32,48,63,64,80,96,112,128,144,256,512}) {
        assert(opus_private_small_shape_valid(Actual,m,128,128));
        for(int row=0;row<m;row+=T::B_M) for(int loops : {1,2,3,6,8,9,12,13,24,56,128}) {
            std::vector<int> valid(T::B_M*loops),identity(T::B_M*loops);
            // Existing scale producer performs aligned vectors when possible
            // and per-byte fallback/0x7f identity for arbitrary M tails.
            for(int index=0;index<T::B_M*loops;index+=16) {
                int r=index%T::B_M,kt=index/T::B_M;
                for(int byte=0;byte<16;byte++) {
                    int rr=r+byte;
                    if(row+rr<m) { assert(kt*m+row+rr<m*loops); valid[index+byte]++; }
                    else identity[index+byte]++;
                }
            }
            for(int count=0;count<T::B_M*loops;count++) assert(valid[count]+identity[count]==1);
            for(int wave=0;wave<T::NUM_WAVES;wave++) for(int lane=0;lane<64;lane++) {
                int wm=wave%T::T_M,wn=wave/T::T_M;
                for(int kt=0;kt<loops;kt++) for(int mi=0;mi<T::E_M;mi++) {
                    int r=(mi*T::T_M+wm)*16+lane%16,index=kt*T::B_M+r;
                    assert(valid[index]+identity[index]==1);
                    int reg_source=row+r<m ? kt*m+r:-1;
                    assert(reg_source==-1 || row+reg_source<m*loops);
                    scale_consumptions++;
                }
                for(int kt=0;kt<loops;kt++) for(int ni=0;ni<T::E_N;ni++) {
                    int nr=(ni*T::T_N+wn)*16;
                    int sg=ni/(128/(T::T_N*16));
                    assert(sg==nr/128 && sg<T::B_GROUPS);
                    assert(sg*loops+kt<T::B_GROUPS*loops);
                    scale_consumptions++;
                }
            }
        }
    }
    // Exact packed output offsets from the frozen helper; each valid output
    // byte has one producer even when E_M*E_N has an unpaired fragment.
    if(T::OUTPUT==2) {
        std::vector<int> coverage(T::B_M*T::B_N);
        for(int wave=0;wave<T::NUM_WAVES;wave++) for(int lane=0;lane<64;lane++) {
            int wm=wave%T::T_M,wn=wave/T::T_M,side=(lane/16)%2;
            for(int pair=0;pair<T::E_M*T::E_N/2;pair++) {
                int ci=pair*2,mi=(ci+side)/T::E_N,ni=(ci+side)%T::E_N;
                int r=(mi*T::T_M+wm)*16+lane%16,n=(ni*T::T_N+wn)*16+(lane/32)*8;
                for(int byte=0;byte<8;byte++) coverage[r*T::B_N+n+byte]++;
            }
            if(T::E_M*T::E_N%2 && side==0) {
                int ci=T::E_M*T::E_N-1,r=((ci/T::E_N)*T::T_M+wm)*16+lane%16;
                int n=((ci%T::E_N)*T::T_N+wn)*16+(lane/32)*8;
                for(int byte=0;byte<8;byte++) coverage[r*T::B_N+n+byte]++;
            }
        }
        for(int count:coverage) {assert(count==1);checked_outputs++;}
    } else {
        assert(T::OUTPUT==1);
        constexpr int stride=T::B_N+8;
        for(int thread=0;thread<T::BLOCK_SIZE;thread++) for(int pass=0;pass<T::B_M*T::B_N/(T::BLOCK_SIZE*8);pass++) {
            int index=(thread+pass*T::BLOCK_SIZE)*8,r=index/T::B_N,n=index%T::B_N;
            assert((r*stride+n+8)*2<=T::B_M*stride*2);checked_outputs+=8;
        }
    }
    assert(!opus_private_small_shape_valid(Actual,max_m+1,128,128));
    assert(!opus_private_small_shape_valid(Actual,1,64,128));
    assert(!opus_private_small_shape_valid(Actual,1,128,127));
    assert(!opus_private_small_shape_valid(Actual,1,128,16512));
}

template<int Actual> void family() {
    addresses<Actual>();
    rings_and_sizes<Actual,1>();rings_and_sizes<Actual,2>();rings_and_sizes<Actual,3>();
    scales_outputs_and_guards<Actual>();
}
int main() {
    family<9043>();family<9044>();family<9045>();family<9046>();
    family<9047>();family<9049>();family<9055>();family<9056>();
    assert(!opus_private_small_shape_valid(0,1,128,128));
    const void* p=reinterpret_cast<const void*>(uintptr_t(4096));
    const void* unaligned=reinterpret_cast<const void*>(uintptr_t(4097));
    assert(opus_private_small_pointer_valid(p,p,unaligned,unaligned,p));
    assert(!opus_private_small_pointer_valid(unaligned,p,p,p,p));
    assert(!opus_private_small_pointer_valid(p,p,p,p,nullptr));
    std::cout << "{\"status\":\"passed\",\"frozen_layout_address_bytes\":"<<checked_bytes
              <<",\"ring_consumptions\":"<<ring_consumptions
              <<",\"scale_consumptions\":"<<scale_consumptions
              <<",\"output_elements\":"<<checked_outputs
              <<",\"actual_families\":8,\"b_ahead_variants\":3,\"k128_loop_lengths\":128,\"gpu_operations\":0}\n";
}
