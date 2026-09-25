// CPU-only coordinate validation; no HIP runtime functions are called.
#include <hip/hip_runtime.h>
#include <opus/opus.hpp>
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
using opus::operator""_I;
#include "host_layout_helpers.inc"

template<int FixedK> void check_matrices_and_output() {
    using T = opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950<FixedK>;
    static_assert(T::FIXED_K == FixedK && T::K_TILES == FixedK / 128);
    static_assert(T::B_M == 192 && T::B_N == 256 && T::B_K == 128);
    static_assert(T::NUM_WAVES == 8 && T::NUM_STAGES == 2);
    static_assert(T::SFA_BYTES == T::B_M * T::K_TILES);
    static_assert(T::SFB_BYTES == (T::B_N / T::GROUP_N) * T::K_TILES);
    static_assert(T::MATRIX_LDS_BYTES == T::NUM_STAGES * (T::A_STAGE + T::B_STAGE));
    static_assert(T::LDS_BYTES == T::MATRIX_LDS_BYTES + T::SFA_BYTES + T::SFB_BYTES);
    std::vector<int> a(T::A_STAGE), b(T::B_STAGE);
    std::vector<int> ga_coverage(T::B_M * FixedK, 0), gb_coverage(T::B_N * FixedK, 0);
    std::uint64_t matrix_producer_bytes = 0, matrix_consumer_bytes = 0;
    for (int tile = 0; tile < T::K_TILES; ++tile) {
        std::fill(a.begin(), a.end(), -1);
        std::fill(b.begin(), b.end(), -1);
        for (int tid = 0; tid < T::BLOCK_SIZE; ++tid) {
            const int lane = tid % 64, wm = (tid / 64) % T::T_M, wn = (tid / 64) / T::T_M;
            auto ga = opus::layout_to_offsets<16>(make_layout_ga_scale<T>(lane, wm, wn, FixedK));
            auto sa = opus::layout_to_offsets<16>(make_layout_sa_scale<T>(wm, wn));
            auto gb = opus::layout_to_offsets<16>(make_layout_gb_scale<T>(lane, wm, wn, FixedK));
            auto sb = opus::layout_to_offsets<16>(make_layout_sb_scale<T>(wm, wn));
            assert(ga.size() == sa.size() && gb.size() == sb.size());
            for (int i = 0; i < ga.size(); ++i) for (int j = 0; j < 16; ++j) {
                const int dst = sa[i] + lane * 16 + j;
                const int src = ga[i] + tile * T::B_K + j;
                assert(dst >= 0 && dst < T::A_STAGE && a[dst] == -1);
                assert(src >= 0 && src < static_cast<int>(ga_coverage.size()));
                a[dst] = src;
                ++ga_coverage[src];
                ++matrix_producer_bytes;
            }
            for (int i = 0; i < gb.size(); ++i) for (int j = 0; j < 16; ++j) {
                const int dst = sb[i] + lane * 16 + j;
                const int src = gb[i] + tile * T::B_K * T::W_N + j;
                assert(dst >= 0 && dst < T::B_STAGE && b[dst] == -1);
                assert(src >= 0 && src < static_cast<int>(gb_coverage.size()));
                b[dst] = src;
                ++gb_coverage[src];
                ++matrix_producer_bytes;
            }
        }
        assert(std::count_if(a.begin(), a.end(), [](int x) { return x != -1; }) == T::B_M * T::B_K);
        assert(std::count_if(b.begin(), b.end(), [](int x) { return x != -1; }) == T::B_N * T::B_K);
        for (int tid = 0; tid < T::BLOCK_SIZE; ++tid) {
            const int lane = tid % 64, wm = (tid / 64) % T::T_M, wn = (tid / 64) / T::T_M;
            for (int mi = 0; mi < T::E_M; ++mi) for (int chunk = 0; chunk < 2; ++chunk)
                for (int j = 0; j < 16; ++j) {
                    const int dst = T::a_lds_offset(wm, lane, mi, chunk) + j;
                    const int row = mi * T::T_M * 16 + wm * 16 + lane % 16;
                    const int k = tile * 128 + (lane / 16) * 16 + chunk * 64 + j;
                    assert(dst >= 0 && dst < T::A_STAGE && a[dst] == row * FixedK + k);
                    for (int valid_rows : {64, 128, 192})
                        assert((a[dst] < valid_rows * FixedK) == (row < valid_rows));
                    ++matrix_consumer_bytes;
                }
            auto rb = opus::layout_to_offsets<16>(make_layout_rb_scale<T>(lane, wn));
            for (int ni = 0; ni < T::E_N; ++ni) for (int chunk = 0; chunk < 2; ++chunk)
                for (int j = 0; j < 16; ++j) {
                    const int dst = rb[ni * 2 + chunk] + j;
                    const int n_group = ni * T::T_N + wn;
                    const int expected = n_group * 16 * FixedK + tile * 128 * 16 + chunk * 64 * 16 + lane * 16 + j;
                    assert(dst >= 0 && dst < T::B_STAGE && b[dst] == expected);
                    const int global_n = n_group * 16 + lane % 16;
                    const int scale_half = ni / (T::GROUP_N / (T::T_N * T::W_N));
                    assert(global_n / T::GROUP_N == scale_half);
                    ++matrix_consumer_bytes;
                }
        }
    }
    for (int count : ga_coverage) assert(count == 1);
    for (int count : gb_coverage) assert(count == 1);
    // Track physical allocation ownership, including padding, for both matrix
    // slots and the two exact-size scale panels.
    std::vector<int> lds_owner(T::LDS_BYTES, 0);
    auto claim = [&](int begin, int size) {
        for (int i = begin; i < begin + size; ++i) {
            assert(i >= 0 && i < T::LDS_BYTES);
            assert(++lds_owner[i] == 1);
        }
    };
    for (int stage = 0; stage < 2; ++stage) {
        claim(stage * T::A_STAGE, T::A_STAGE);
        claim(2 * T::A_STAGE + stage * T::B_STAGE, T::B_STAGE);
    }
    claim(T::MATRIX_LDS_BYTES, T::SFA_BYTES);
    claim(T::MATRIX_LDS_BYTES + T::SFA_BYTES, T::SFB_BYTES);
    for (int count : lds_owner) assert(count == 1);

    auto mma = opus::make_tiled_mma<opus::fp8_t, opus::fp8_t, opus::fp32_t>(
        opus::seq<T::E_M, T::E_N, 1>{}, opus::seq<T::T_M, T::T_N, 1>{},
        opus::seq<16, 16, 128>{}, opus::mfma_adaptor_swap_ab{});
    std::uint64_t output_coordinates = 0, bounded_vectors = 0;
    for (int stride : {256, 512, 6144, 7168, 8192}) {
        std::vector<int> covered(T::B_M * T::B_N, 0);
        for (int tid = 0; tid < T::BLOCK_SIZE; ++tid) {
            const int lane = tid % 64, wm = (tid / 64) % T::T_M, wn = (tid / 64) / T::T_M;
            auto coord = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
            auto offsets = opus::layout_to_offsets<4>(opus::partition_layout_c<4>(mma, opus::make_tuple(stride, 1_I), coord));
            assert(offsets.size() == T::E_M * T::E_N);
            for (int i = 0; i < offsets.size(); ++i) {
                for (int j = 0; j < 4; ++j) {
                    const int off = offsets[i] + j, row = off / stride, col = off % stride;
                    assert(row >= 0 && row < T::B_M && col >= 0 && col < T::B_N);
                    ++covered[row * T::B_N + col];
                    ++output_coordinates;
                }
                for (int valid_rows : {64, 128, 192}) for (int origin_col : {0, stride - T::B_N}) {
                    const int last_valid = valid_rows * stride - origin_col;
                    const int local_row = offsets[i] / stride;
                    const bool first_in_bounds = offsets[i] < last_valid;
                    const bool last_in_bounds = offsets[i] + 3 < last_valid;
                    assert(first_in_bounds == (local_row < valid_rows));
                    assert(first_in_bounds == last_in_bounds);
                    ++bounded_vectors;
                }
            }
        }
        for (int count : covered) assert(count == 1);
    }
    std::cout << "K=" << FixedK << " matrix_producer_bytes=" << matrix_producer_bytes
              << " matrix_consumer_bytes=" << matrix_consumer_bytes
              << " output_coordinates=" << output_coordinates
              << " M64_bounded_vectors=" << bounded_vectors
              << " LDS_bytes=" << T::LDS_BYTES << '\n';
}

#include "scale_schedule_check.inc"

int main() {
    check_matrices_and_output<384>();
    check_matrices_and_output<768>();
    check_matrices_and_output<1024>();
    check_scales<384>();
    check_scales<768>();
    check_scales<1024>();
    check_schedule<384>();
    check_schedule<768>();
    check_schedule<1024>();
    std::cout << "PASS: CPU-only coordinates, exact scales, M64 tails, and fixed-K schedule.\n";
}
