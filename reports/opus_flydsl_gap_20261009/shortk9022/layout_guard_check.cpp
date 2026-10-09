#include "frozen_layout_helpers.h"
#include "candidate/traits.cuh"
#include "contract.h"
#include <cassert>
#include <iostream>
#include <vector>

template<class T, int Pass>
void producers(int m, int row, std::vector<int>& coverage, std::vector<int>& identity_tail) {
    constexpr int loops = T::FIXED_K / T::B_K;
    for (int wave = 0; wave < T::NUM_WAVES; ++wave) {
        for (int lane = 0; lane < T::WARP_SIZE; ++lane) {
            const auto g = checked_layout::make_layout_gsfa_scale<T, Pass>(lane, wave % T::T_M, wave / T::T_M, m);
            const auto s = checked_layout::make_layout_ssfa_scale<T, Pass>(lane, wave % T::T_M, wave / T::T_M);
            const int smem_offset = opus::layout_to_offsets<T::VEC_SCALE_A>(s)[0];
            const int local_k = smem_offset / T::B_M;
            const int local_row = smem_offset % T::B_M;
            if (smem_offset >= T::SFA_BYTES || local_k >= loops)
                continue;
            const int gmem_offset = opus::layout_to_offsets<T::VEC_SCALE_A>(g)[0];
            assert(gmem_offset == local_k * m + local_row);
            assert(local_row % 16 == 0 && local_row + 16 <= T::B_M);
            for (int byte = 0; byte < 16; ++byte) {
                const int address = smem_offset + byte;
                assert(address < T::SFA_BYTES);
                if (row + local_row < m) {
                    assert(gmem_offset + row + byte < m * loops);
                    ++coverage[address];
                } else {
                    ++identity_tail[address]; // Original aligned-M tail writes 0x7f.
                }
            }
        }
    }
}

template<class T>
void check_layout() {
    constexpr int loops = T::FIXED_K / T::B_K;
    const int shapes[] = {16, 144, 160, 176, 304, 320, 336, 544, 672, 800, 864, 992, 1024, 1088};
    for (const int m : shapes) {
        for (int row = 0; row < m; row += T::B_M) {
            std::vector<int> coverage(T::SFA_BYTES), identity_tail(T::SFA_BYTES);
            producers<T, 0>(m, row, coverage, identity_tail);
            for (int k = 0; k < loops; ++k)
                for (int local_row = 0; local_row < T::B_M; ++local_row) {
                    const int address = k * T::B_M + local_row;
                    assert(coverage[address] + identity_tail[address] == 1);
                    assert(coverage[address] == (row + local_row < m ? 1 : 0));
                }
            std::vector<int> b_coverage(loops);
            for (int wave = 0; wave < T::NUM_WAVES; ++wave)
                for (int lane = 0; lane < T::WARP_SIZE; ++lane) {
                    const auto ra = checked_layout::make_layout_rsfa_scale<T>(lane, wave % T::T_M);
                    const auto offsets = opus::layout_to_offsets<1>(ra);
                    for (int repeat = 0; repeat < T::E_M; ++repeat) {
                        const int expected_row = repeat * T::T_M * T::W_M + (wave % T::T_M) * T::W_M + lane % T::W_M;
                        assert(offsets[repeat] == expected_row);
                        for (int k = 0; k < loops; ++k)
                            assert(coverage[k * T::B_M + offsets[repeat]] + identity_tail[k * T::B_M + offsets[repeat]] == 1);
                    }
                    const auto gb = checked_layout::make_layout_gsfb_scale<T>(lane, wave % T::T_M, wave / T::T_M, loops);
                    const auto sb = checked_layout::make_layout_ssfb_scale<T>(lane, wave % T::T_M, wave / T::T_M);
                    const int offset = opus::layout_to_offsets<1>(sb)[0];
                    if (offset < loops) {
                        assert(opus::layout_to_offsets<1>(gb)[0] == offset);
                        ++b_coverage[offset];
                    }
                }
            for (const int count : b_coverage)
                assert(count == 1);
        }
    }
}

int main() {
    check_layout<opus_private_shortk9022_panel8_traits<384>>();
    check_layout<opus_private_shortk9022_panel8_traits<768>>();
    assert(shortk9022_shape_valid(16, 128, 384, true));
    assert(shortk9022_shape_valid(1088, 7168, 768, true));
    assert(shortk9022_shape_valid(1088, 7168, 16384, false));
    assert(!shortk9022_shape_valid(1088, 7168, 16384, true));
    assert(!shortk9022_shape_valid(15, 128, 384, true));
    assert(!shortk9022_shape_valid(16, 64, 384, true));
    assert(!shortk9022_shape_valid(0, 128, 384, true));
    assert(!shortk9022_shape_valid(16, 128, 512, true));
    assert(!shortk9022_shape_valid(32768, 65536, 384, true));
    assert(!shortk9022_shape_valid(16, 2147483520, 768, true));
    const void* aligned = reinterpret_cast<const void*>(uintptr_t(0x1000));
    const void* unaligned = reinterpret_cast<const void*>(uintptr_t(0x1001));
    assert(shortk9022_pointer_valid(aligned, aligned, aligned, unaligned, aligned));
    assert(!shortk9022_pointer_valid(unaligned, aligned, aligned, aligned, aligned));
    assert(!shortk9022_pointer_valid(aligned, aligned, unaligned, aligned, aligned));
    assert(!shortk9022_pointer_valid(aligned, aligned, aligned, nullptr, aligned));
    std::cout << "passed: actual frozen SFA producer/consumer layouts, scale tails and SFB offsets; two fixed K variants; 14 M shapes; actual launcher guards; no HIP runtime calls\n";
}
