#include "frozen_layout_helpers.h"
#include "traits.cuh"
#include "contract.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

using T = opus_private_9030_directb_a3_chunk96_traits;
constexpr int loops = T::FIXED_K / T::B_K;

void check_matrix_layouts() {
    std::vector<int> amap(T::A_STAGE, -1), bmap(T::B_STAGE, -1);
    std::vector<int> aglobal(T::B_M * T::B_K), bglobal(T::B_N * T::B_K);
    for (int wave = 0; wave < T::NUM_WAVES; ++wave)
        for (int lane = 0; lane < T::WARP_SIZE; ++lane) {
            const int wm = wave % T::T_M, wn = wave / T::T_M;
            const auto ga = opus::layout_to_offsets<T::VEC_A>(checked_matrix::make_layout_ga_scale<T>(lane, wm, wn, T::FIXED_K));
            const auto sa = opus::layout_to_offsets<T::VEC_A>(checked_matrix::make_layout_sa_scale<T>(wm, wn));
            const auto gb = opus::layout_to_offsets<T::VEC_B>(checked_matrix::make_layout_gb_scale<T>(lane, wm, wn, T::FIXED_K));
            const auto sb = opus::layout_to_offsets<T::VEC_B>(checked_matrix::make_layout_sb_scale<T>(wm, wn));
            static_assert(decltype(ga)::size() == T::A_VMEM_INSTRUCTIONS);
            for (int i = 0; i < decltype(ga)::size(); ++i)
                for (int byte = 0; byte < T::VEC_A; ++byte) {
                    // CDNA buffer-to-LDS adds the lane's vec-byte offset to the uniform LDS base.
                    const int dst = sa[i] + lane * T::VEC_A + byte, src = ga[i] + byte;
                    assert(dst < T::A_STAGE && amap[dst] == -1);
                    amap[dst] = src;
                    const int linear = (src / T::FIXED_K) * T::B_K + src % T::FIXED_K;
                    assert(src % T::FIXED_K < T::B_K && linear < int(aglobal.size()));
                    ++aglobal[linear];
                }
            for (int i = 0; i < decltype(gb)::size(); ++i)
                for (int byte = 0; byte < T::VEC_B; ++byte) {
                    const int dst = sb[i] + lane * T::VEC_B + byte, src = gb[i] + byte;
                    assert(dst < T::B_STAGE && bmap[dst] == -1);
                    bmap[dst] = src;
                    const int group = src / (T::W_N * T::FIXED_K), offset = src % (T::W_N * T::FIXED_K);
                    const int linear = group * T::W_N * T::B_K + offset;
                    assert(offset < T::W_N * T::B_K && linear < int(bglobal.size()));
                    ++bglobal[linear];
                }
        }
    assert(std::all_of(aglobal.begin(), aglobal.end(), [](int v) { return v == 1; }));
    assert(std::all_of(bglobal.begin(), bglobal.end(), [](int v) { return v == 1; }));
    for (int wave = 0; wave < T::NUM_WAVES; ++wave)
        for (int lane = 0; lane < T::WARP_SIZE; ++lane) {
            const int wm = wave % T::T_M, wn = wave / T::T_M;
            const auto ra = opus::layout_to_offsets<T::VEC_A>(checked_layout::make_layout_ra_scale<T>(lane, wm));
            const auto rb = opus::layout_to_offsets<T::VEC_B>(checked_matrix::make_layout_rb_scale<T>(lane, wn));
            for (int mr = 0; mr < T::E_M; ++mr)
                for (int chunk = 0; chunk < T::A_CHUNKS_PER_FRAGMENT; ++chunk)
                    for (int byte = 0; byte < T::VEC_A; ++byte) {
                        const int dst = ra[mr * T::A_CHUNKS_PER_FRAGMENT + chunk] + byte;
                        const int matrix_row = mr * T::T_M * T::W_M + wm * T::W_M + lane % T::W_M;
                        const int matrix_k = chunk * 64 + (lane / T::W_M) * T::VEC_A + byte;
                        assert(amap[dst] == matrix_row * T::FIXED_K + matrix_k);
                        for (int stage = 0; stage < T::NUM_STAGES; ++stage)
                            assert(stage * T::A_STAGE + dst < T::MATRIX_LDS_BYTES);
                    }
            for (int nr = 0; nr < T::E_N; ++nr)
                for (int chunk = 0; chunk < T::B_CHUNKS_PER_FRAGMENT; ++chunk)
                    for (int byte = 0; byte < T::VEC_B; ++byte) {
                        const int dst = rb[nr * T::B_CHUNKS_PER_FRAGMENT + chunk] + byte;
                        const int group = nr * T::T_N + wn;
                        const int direct = checked_layout::direct_b_address<T>(0, nr, wn, lane, chunk, T::FIXED_K) + byte;
                        assert(direct == group * T::W_N * T::FIXED_K + lane * T::VEC_B + chunk * T::WARP_SIZE * T::VEC_B + byte);
                        assert(bmap[dst] == direct);
                        for (int tile = 0; tile < loops; ++tile)
                            assert(direct + tile * T::W_N * T::B_K < T::B_N * T::FIXED_K);
                    }
        }
}

void check_scales() {
    for (const int valid_rows : {64, 128, 192}) {
        const int m = 192 + valid_rows, row = 192;
        std::vector<int> coverage(T::SFA_BYTES), identity(T::SFA_BYTES), bcoverage(loops);
        for (int wave = 0; wave < T::NUM_WAVES; ++wave)
            for (int lane = 0; lane < T::WARP_SIZE; ++lane) {
                const int wm = wave % T::T_M, wn = wave / T::T_M;
                const auto gs = opus::layout_to_offsets<T::VEC_SCALE_A>(checked_layout::make_layout_gsfa_scale<T>(lane, wm, wn, m));
                const auto ss = opus::layout_to_offsets<T::VEC_SCALE_A>(checked_layout::make_layout_ssfa_scale<T>(lane, wm, wn));
                const int local_k = wave * T::SFA_K_COLUMNS_PER_WAVE + lane / T::SFA_THREADS_PER_GROUP;
                for (int pass = 0; pass < T::SFA_PASSES; ++pass) {
                    const int local_row = ss[pass] - local_k * T::B_M;
                    if (local_row < T::B_M && ss[pass] < T::SFA_BYTES && local_k < loops) {
                        assert(local_row >= 0 && local_row + T::VEC_SCALE_A <= T::B_M);
                        assert(gs[pass] == local_k * m + local_row);
                        for (int byte = 0; byte < T::VEC_SCALE_A; ++byte) {
                            if (row + local_row < m) {
                                assert(row + gs[pass] + byte < m * loops);
                                ++coverage[ss[pass] + byte];
                            } else ++identity[ss[pass] + byte];
                        }
                    }
                }
                const auto ra = opus::layout_to_offsets<1>(checked_layout::make_layout_rsfa_scale<T>(lane, wm));
                for (int mr = 0; mr < T::E_M; ++mr)
                    assert(ra[mr] == mr * T::T_M * T::W_M + wm * T::W_M + lane % T::W_M);
                const auto gb = opus::layout_to_offsets<1>(checked_layout::make_layout_gsfb_scale<T>(lane, wave, loops));
                const auto sb = opus::layout_to_offsets<1>(checked_layout::make_layout_ssfb_scale<T>(lane, wave));
                const int k = wave * T::WARP_SIZE + lane;
                if (k < loops) {
                    assert(gb[0] == k && gb[1] == loops + k && sb[0] == k);
                    ++bcoverage[k];
                }
                for (int nr = 0; nr < T::E_N; ++nr)
                    assert(((nr * T::T_N + wn) * T::W_N) / T::GROUP_N == (nr < 4 ? 0 : 1));
            }
        for (int k = 0; k < loops; ++k)
            for (int local_row = 0; local_row < T::B_M; ++local_row) {
                const int address = k * T::B_M + local_row;
                assert(coverage[address] + identity[address] == 1);
                assert(coverage[address] == (local_row < valid_rows ? 1 : 0));
            }
        for (const int count : bcoverage) assert(count == 1);
    }
}

void check_chunk_output() {
    auto mma = opus::make_tiled_mma<opus::fp8_t, opus::fp8_t, opus::fp32_t>(
        opus::seq<T::E_M, T::E_N, T::E_K>{}, opus::seq<T::T_M, T::T_N, T::T_K>{},
        opus::seq<T::W_M, T::W_N, T::W_K>{}, opus::mfma_adaptor_swap_ab{});
    static_assert(decltype(mma)::elem_c == T::VEC_C);
    std::vector<int> coverage(T::B_M * T::B_N), writes(T::B_M * T::B_N);
    for (int chunk = 0; chunk < T::C_CHUNKS; ++chunk) {
        const int first_row = chunk * T::C_CHUNK_ROWS;
        std::vector<int> arena(T::C_CHUNK_BYTES / sizeof(opus::bf16_t), -1);
        for (int wave = 0; wave < T::NUM_WAVES; ++wave)
            for (int lane = 0; lane < T::WARP_SIZE; ++lane) {
                const auto pc = opus::make_tuple(wave % T::T_M, lane % mma.grpn_c, wave / T::T_M, lane / mma.grpn_c);
                const auto offsets = opus::layout_to_offsets<T::VEC_C>(opus::partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), pc));
                for (int i = 0; i < T::E_M * T::E_N; ++i) {
                    const int row = offsets[i] / T::C_LDS_ROW_STRIDE_ELEMS, col = offsets[i] % T::C_LDS_ROW_STRIDE_ELEMS;
                    assert(row < T::B_M && col + T::VEC_C <= T::B_N);
                    if (row >= first_row && row < first_row + T::C_CHUNK_ROWS)
                        for (int elem = 0; elem < T::VEC_C; ++elem) {
                            const int dst = offsets[i] - first_row * T::C_LDS_ROW_STRIDE_ELEMS + elem;
                            const int logical = row * T::B_N + col + elem;
                            assert(dst < int(arena.size()) && arena[dst] == -1);
                            arena[dst] = logical;
                            ++coverage[logical];
                        }
                }
            }
        for (int thread = 0; thread < T::BLOCK_SIZE; ++thread)
            for (int pass = 0; pass < T::CHUNK_OUTPUT_PASSES; ++pass) {
                const int linear = thread * T::VEC_OUTPUT + pass * T::BLOCK_SIZE * T::VEC_OUTPUT;
                const int row = linear / T::B_N, col = linear % T::B_N;
                for (int elem = 0; elem < T::VEC_OUTPUT; ++elem) {
                    const int logical = (first_row + row) * T::B_N + col + elem;
                    assert(arena[row * T::C_LDS_ROW_STRIDE_ELEMS + col + elem] == logical);
                    ++writes[logical];
                }
            }
    }
    for (const int v : coverage) assert(v == 1);
    for (const int v : writes) assert(v == 1);
}

void check_lifecycle() {
    std::array<int, 3> a{-1, -1, -1};
    std::array<bool, 3> retired{true, true, true};
    auto issue = [&](int tile) {
        const int stage = tile % 3;
        assert(retired[stage]);
        a[stage] = tile;
    };
    auto read = [&](int tile) {
        assert(a[tile % 3] == tile);
        retired[tile % 3] = false;
        // Source's lgkmcnt(0) plus block barrier retires all A reads.
        retired[tile % 3] = true;
    };
    issue(0); issue(1); read(0);
    int b = 0, bnext = 1;
    for (int tile = 0; tile < loops; ++tile) {
        if (tile + 2 < loops) issue(tile + 2);
        assert(a[tile % 3] == tile);
        for (int nr = 0; nr < T::E_N; ++nr) {
            assert(b == tile * T::E_N + nr);
            const int next = tile * T::E_N + nr + 1;
            if (next < loops * T::E_N) {
                assert(bnext == next);
                b = bnext; // Source waits for the pending B read, copies, then refills.
                if (next + 1 < loops * T::E_N) bnext = next + 1;
            }
        }
        if (tile + 1 < loops) read(tile + 1);
    }
    assert(b == loops * T::E_N - 1 && a[(loops - 1) % 3] == loops - 1);
    assert(std::all_of(retired.begin(), retired.end(), [](bool v) { return v; }));
}

void check_guards_and_large_addresses() {
    const int ms[] = {16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536};
    for (const int m : ms) {
        assert(private9030_shape_valid(m, 65536, 1536, true));
        const int row = ((m - 1) / T::B_M) * T::B_M, n = 65536;
        const int valid = m - row;
        assert(valid == 64 || valid == 128 || valid == 192);
        const int64_t base = (int64_t(row) * n + n - T::B_N) * 2;
        const int64_t resource = ((int64_t(valid) - 1) * n + T::B_N) * 2;
        assert(base + resource == int64_t(m) * n * 2);
        assert(resource <= INT_MAX && int64_t(m) * n * 2 > INT_MAX);
        for (int chunk = 0; chunk < T::C_CHUNKS; ++chunk)
            for (int local_row = 0; local_row < T::C_CHUNK_ROWS; ++local_row) {
                const int out_row = chunk * T::C_CHUNK_ROWS + local_row;
                if (out_row < valid)
                    assert((int64_t(out_row) * n + T::B_N) * 2 <= resource);
            }
    }
    assert(private9030_shape_valid(65536, 16384, 1536, true));
    assert(private9030_shape_valid(65536, 16384, 384, false));
    assert(private9030_shape_valid(65536, 16384, 16384, false));
    assert(!private9030_shape_valid(65536, 16384, 384, true));
    assert(!private9030_shape_valid(65536, 16384, 16384, true));
    assert(private9030_shape_valid(1398016, 1398016, 1536, true));
    assert(!private9030_shape_valid(1398272, 1398016, 1536, true));
    assert(!private9030_shape_valid(1398016, 1398272, 1536, true));
    assert(!private9030_shape_valid(INT_MAX, INT_MAX, 1536, true));
    assert(!private9030_shape_valid(16384, 65536, 512, true));
    assert(!private9030_shape_valid(16320, 65536, 1536, true));
    assert(!private9030_shape_valid(16384, 65535, 1536, true));
    assert(!private9030_shape_valid(0, 65536, 1536, true));
    const void* p = reinterpret_cast<const void*>(uintptr_t(0x1000));
    const void* unaligned = reinterpret_cast<const void*>(uintptr_t(0x1001));
    assert(private9030_pointer_valid(p, p, p, unaligned, p));
    assert(!private9030_pointer_valid(unaligned, p, p, p, p));
    assert(!private9030_pointer_valid(p, unaligned, p, p, p));
    assert(!private9030_pointer_valid(p, p, unaligned, p, p));
    assert(!private9030_pointer_valid(p, p, p, nullptr, p));
    assert(!private9030_pointer_valid(p, p, p, p, unaligned));
}

int main() {
    check_matrix_layouts(); check_scales(); check_chunk_output(); check_lifecycle();
    check_guards_and_large_addresses();
    std::cout << "passed: actual frozen A/B producer and fragment layouts; direct-B byte equivalence for all waves/lanes/repeats/chunks and 12 K tiles; panel16 SFA/SFB coverage and 64/128/192 row tails; actual MMA C partition with two chunk96 output passes; A3/B2 lifecycle model; all10 historical shape guards and signed32/signed64 C boundaries; no HIP runtime calls\n";
}
