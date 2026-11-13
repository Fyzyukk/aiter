# Generate the no-shuffle (row-major B / scales) pipeline from the parameterized shuffle pipeline.
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
def rep(a, b, n=1):
    global s
    c = s.count(a)
    assert c == n, (c, n, a[:120])
    s = s.replace(a, b)
def cut(start, end_marker, repl, include_end=False):
    global s
    i0 = s.index(start); i1 = s.index(end_marker, i0)
    if include_end: i1 += len(end_marker)
    s = s[:i0] + repl + s[i1:]

# --- header / namespace ---
rep('#include "opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh"\n',
    '#include "opus_gemm_traits_a4w4_mxfp4_nt_noshuffle_gfx950.cuh"\n')
rep('using opus::operator""_I;\n',
    'using opus::operator""_I;\n\n'
    '// Everything below lives in its own namespace so this header and the shuffle pipeline header can share a TU.\n'
    'namespace opus_mxfp4_ns {\n')

# --- B layouts: row-major K tiles, same scheme as A (chunk (wave_m * 2 + wave_n) * insts + issue) ---
GB = '''template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    // Row-major B (as A, roles of wave_m / wave_n swapped): wave (P0 = wave_m, P1 = wave_n) copies chunks
    // (P0 * 2 + P1) * insts + i; chunk c holds rows (c / 2) * 16 + 2 * (lane / 8) + c % 2, 16 B XOR swizzle along K.
    constexpr int threads_k = T::B_K_BYTES / T::VEC_B;                 // 8
    static_assert(threads_k == T::B_K / T::VEC_B_ELEMS && T::T_M == 2 && T::T_N == 2);
    static_assert(T::HALF_B_N / (T::BLOCK_SIZE / threads_k) == T::b_buffer_load_insts);
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0
        opus::number<2>{},                                             // p P1
        opus::number<T::b_buffer_load_insts / 2>{},                   // y ih
        opus::number<T::WARP_SIZE / threads_k>{},                      // p lane / 8
        opus::number<2>{},                                             // y il
        opus::number<threads_k>{},                                     // p lane % 8
        opus::number<T::VEC_B>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_b, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n, lane_id / threads_k, (lane_id % threads_k) ^ ((((lane_id / threads_k) >> 1) & 1) << 1)}));
}

template<class T>
__device__ inline auto make_layout_sb(int wave_id_m, int wave_id_n) {
    // Chunk (P0 * 2 + P1) * insts + issue (see make_layout_gb).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0
        opus::number<2>{},                                             // p P1
        opus::number<T::b_buffer_load_insts>{},                        // y issue
        opus::number<T::VEC_B>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::smem_linear_wave_b + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    static_assert(T::smem_sub_b * T::T_N == T::W_N && T::T_N == 2);
    constexpr int reads_per_operand = T::W_N * T::W_K_BYTES / T::WARP_SIZE / T::VEC_B;   // 1 b128 per fp4 operand
    static_assert(reads_per_operand == 1);
    static_assert(T::WARP_SIZE / T::W_N * T::VEC_B_ELEMS == T::W_K);

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},                                        // y N repeat
        opus::number<T::T_N>{},                                        // p wave_n
        opus::number<T::T_N>{},                                        // p r % 2
        opus::number<T::W_N / T::T_N>{},                               // p r / 2
        opus::number<T::E_K>{},                                        // p kk, fixed to 0; kk -> + kk * W_K_BYTES
        opus::number<reads_per_operand>{},                             // y b128 per operand
        opus::number<T::WARP_SIZE / T::W_N>{},                         // p g = lane / 16
        opus::number<T::VEC_B>{});                                     // y bytes
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_n = lane_id % T::W_N;
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave_b + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_n % T::T_N, lane_n / T::T_N, 0_I, (lane_id / T::W_N) ^ (((lane_n >> 2) & 1) << 1)}));
}

'''
cut('template<class T>\n__device__ inline auto make_layout_gb(', '// Like opus async_load(layout)', GB)

# --- m8 C11 schedule (scale transposes + e1: panel ds_write after the last ds_read, then the panel buffer_load) ---
rep('''// C00 (32 MFMAs): 16 DS (A1 / B1 of t+1) in the first 16 MFMAs''',
'''// m8 C11 (32 MFMAs): 8 DS (A0 of t+1) + 24 VALU (scale transposes), the panel ds_write after the last ds_read, then
// the next panel's global load.
__device__ inline void sched_c11_ns() {
    opus::static_for<7>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>(); });
    sched_m<1, 1>();
    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);   // e1: scale ds_write after the last ds_read
    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);   // e1: scale panel buffer_load
    sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>();
}
// C00 (32 MFMAs): 16 DS (A1 / B1 of t+1) in the first 16 MFMAs''')

# --- names ---
rep('gemm_a4w4_mxfp4_nt_body', 'gemm_a4w4_mxfp4_nt_noshuffle_body', 2)
rep('void gemm_a4w4_mxfp4_nt_kernel(', 'void gemm_a4w4_mxfp4_nt_noshuffle_kernel(')

# --- global bases (row-major B / scales) ---
rep('const int a_k0 = kt0 * T::B_K_BYTES, b_k0 = kt0 * T::B_K_BYTES * T::W_N, sf_k0 = kt0 * 256;',
    'const int a_k0 = kt0 * T::B_K_BYTES, b_k0 = kt0 * T::B_K_BYTES, sf_k0 = kt0 * T::NUM_KGROUPS;')
i0 = s.index('    auto g_sfa = make_gmem('); i1 = s.index('\n', s.index('    auto g_sfb = make_gmem(', i0)) + 1
s = s[:i0] + '''    // Row-major scales [rows, K/32]: rows past the matrix are outside the range (read as 0).
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row * kargs.stride_sfa + sf_k0,
                           static_cast<unsigned int>((kargs.m - row) * kargs.stride_sfa - sf_k0));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + col * kargs.stride_sfb + sf_k0,
                           static_cast<unsigned int>((kargs.n - col) * kargs.stride_sfb - sf_k0));
''' + s[i1:]

# --- scale operand macros: bytes = repeats (opsel m / n), no shifts ---
cut('    D_SF_PACK v_ssa[2][T::E_M], v_ssb[2][T::E_N];\n', '    D_SF_PACK v_sfa[2][T::E_K];\n',
'''#undef SFA
#undef SFB
#undef SOA
#undef SOB
#define SFA(h, kk, m) v_sfa[h][kk]
#define SFB(h, kk, n) v_sfb[h][kk]
#define SOA(kk, m) (m)
#define SOB(kk, n) (n)
''')

rep('auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * (T::B_K_BYTES * T::W_N); };   // bs: K tile = 2 KB per 16-row block',
    'auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K_BYTES; };')

# --- scale global offsets ---
cut('    // sc: e8m0_shuffle layout = [32-row block]', '    auto ssfa_offset = ',
'''    // Row-major scales: writer wave (A: h = wave_n, w = wave_m; B: h = wave_m, w = wave_n), lane 4 r + e reads the 16 B
    // (16 groups = 2 K tiles) of row h * HALF_B + e * 32 + w * 16 + r.  e >= E rows feed bytes no MFMA selects.
    const int sf_e = lane_id % 4, sf_r = lane_id / 4;
    const int sfa_g_lane = (wave_id_n * T::HALF_B_M + sf_e * 32 + wave_id_m * 16 + sf_r) * kargs.stride_sfa;
    const int sfb_g_lane = (wave_id_m * T::HALF_B_N + sf_e * 32 + wave_id_n * 16 + sf_r) * kargs.stride_sfb;
    auto gsfa_offset = [&](int panel_k_begin) { return sfa_g_lane + panel_k_begin * T::NUM_KGROUPS; };
    auto gsfb_offset = [&](int panel_k_begin) { return sfb_g_lane + panel_k_begin * T::NUM_KGROUPS; };
''')

# --- odd B DMA: row-major chunks (as A) ---
cut('    auto dma_odd_b = [&](auto& g, int lds_off, int s_os) {', '    auto prefetch_a = [&]',
'''    auto dma_odd_b = [&](auto& g, int lds_off, int s_os) {
        constexpr int NI = T::b_buffer_load_insts;
        const int w = wave_id_m * 2 + wave_id_n;
        char* const base = reinterpret_cast<char*>(s_b.ptr) + lds_off + w * NI * DMA_CHUNK;
        opus::static_for<NI>([&](auto i_) {
            constexpr int i = decltype(i_)::value, io = i * DMA_CHUNK;
            const int c = w * NI + i;
            const int goff = ((c >> 1) * 16 + 2 * (lane_id / 8) + (c & 1)) * kargs.stride_b + a_kcol;
            g.template async_load<T::VEC_B>(base, goff - io, s_os, opus::number<io>{}, opus::number<AB_AUX>{});
        });
    };
''')

# --- scale panel path: global -> VGPR -> 4x4 transpose -> LDS (m8) ---
cut('    // Scale panels go global -> VGPR -> 4x4 byte transpose -> LDS', '    // ---- LDS -> VGPR ----',
'''    // Scale panels go global -> VGPR -> 4x4 byte transpose -> LDS, so LDS holds the MFMA scale
    // layout and the main loop reads it as is.  A panel is loaded in iteration t and written in t + 1
    // (its load has landed by B_b of t + 1).
    using SF_PANEL = opus::vector_t<D_SF, T::VEC_GLOBAL_SCALE>;
    using SF_DW4 = opus::vector_t<opus::u32_t, 4>;
    SF_PANEL v_sf_pend;
    auto fetch_sfa = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfa, gsfa_offset(panel_k_begin)); };
    auto fetch_sfb = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfb, gsfb_offset(panel_k_begin)); };
    // 4x4 byte transpose inside each lane quad (lane = 4 r + e, byte g = K group), as PR 6696:
    //   step 1, partner e ^ 1: even e -> [own0, nb0, own2, nb2], odd e -> [nb1, own1, nb3, own3]
    //   step 2, partner e ^ 2: e < 2 -> [own0, own1, nb0, nb1], e >= 2 -> [nb2, nb3, own2, own3]
    // v_perm(S0 = neighbour, S1 = own): selector bytes 0-3 pick S1, 4-7 pick S0.
    const unsigned int sf_q = static_cast<unsigned int>(lane_id % 4);
    const unsigned int sf_sel1 = (sf_q & 1) ? 0x03070105u : 0x06020400u;
    const unsigned int sf_sel2 = (sf_q & 2) ? 0x03020706u : 0x05040100u;
    auto quad_transpose = [&](unsigned int x) {
        const unsigned int y = static_cast<unsigned int>(__builtin_amdgcn_mov_dpp(static_cast<int>(x), 0xB1, 0xF, 0xF, true));   // quad_perm [1,0,3,2]
        const unsigned int t = __builtin_amdgcn_perm(y, x, sf_sel1);
        const unsigned int z = static_cast<unsigned int>(__builtin_amdgcn_mov_dpp(static_cast<int>(t), 0x4E, 0xF, 0xF, true));   // quad_perm [2,3,0,1]
        return __builtin_amdgcn_perm(z, t, sf_sel2);
    };
    // Lane 4 r + g writes the 16 B its reader lane g * 16 + r reads.  Panel image [K tile][half][w][slot][8 B]
    // (2 x 2048 B per buffer), slots XOR-swizzled within a wave's 64: s -> s ^ (((s >> 4) & 3) << 2) (bank-conflict free).
    const int sf_put_off = ((lane_id % 4) * 16 + ((lane_id / 4) ^ ((lane_id % 4) << 2))) * T::VEC_LDS_SCALE;
    auto put_sf = [&](auto& s_sf_, int wave_base, const SF_PANEL& raw, int buf) {
        const auto dw = __builtin_bit_cast(SF_DW4, raw);
        SF_DW4 t;
        opus::static_for<4>([&](auto d) { t[d.value] = quad_transpose(dw[d.value]); });
        using SF_DW2 = opus::vector_t<opus::u32_t, 2>;
        using SF_B8 = opus::vector_t<D_SF, T::VEC_LDS_SCALE>;
        const int o = buf * T::SFA_PANEL_BYTES + wave_base + sf_put_off;
        store<T::VEC_LDS_SCALE>(s_sf_, __builtin_bit_cast(SF_B8, SF_DW2{t[0], t[1]}), o);
        store<T::VEC_LDS_SCALE>(s_sf_, __builtin_bit_cast(SF_B8, SF_DW2{t[2], t[3]}), o + T::SFA_PANEL_BYTES / 2);
    };
    const int ssfa_wbase = (wave_id_n * T::T_M + wave_id_m) * T::WARP_SIZE * T::VEC_LDS_SCALE;
    const int ssfb_wbase = (wave_id_m * T::T_N + wave_id_n) * T::WARP_SIZE * T::VEC_LDS_SCALE;
#undef SF_STEP_EVEN
#undef SF_STEP_ODD
#define SF_STEP_EVEN() do { put_sf(rd_sfa, ssfa_wbase, v_sf_pend, sf_wr); v_sf_pend = fetch_sfb(tile + 4); } while (0)   // sfa of panel tile + 2; load sfb of panel tile + 4
#define SF_STEP_ODD()  do { put_sf(rd_sfb, ssfb_wbase, v_sf_pend, sf_wr); v_sf_pend = fetch_sfa(tile + 3); } while (0)   // sfb of panel tile + 3; load sfa of panel tile + 3

''')

rep('load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + kk.value * (T::smem_linear_wave_b + T::smem_padding)));   // bs: kk = next chunk',
    'load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + kk.value * T::W_K_BYTES));')
rep('#define LOAD_B(rd, h, kt) load_b(rd, opus::number<h>{})', '#undef LOAD_B\n#define LOAD_B(rd, h, kt) load_b(rd, opus::number<h>{})')

# --- scale LDS reads (m8): one b64 per half per tensor, read as is ---
cut('    // Scale reads: lane reads the 2 dwords', '    // LDS reads go through __restrict__ copies of the views',
'''    // Scale reads: lane reads the 2 dwords (kk = 0, 1) of half h at LDS slot (h * 2 + w) * 64 + lane (+ 2 KB for the
    // second tile of the panel), early (C01) into v_sf*_next; read_sf moves half h into v_sfa / v_sfb when the previous
    // values are no longer used.
    using SF_KK = opus::vector_t<D_SF_PACK, T::E_K>;
    const int sf_rd_lane = lane_id ^ (((lane_id >> 4) & 3) << 2);
    const int sfa_rd_base = (wave_id_m * T::WARP_SIZE + sf_rd_lane) * T::VEC_LDS_SCALE;
    const int sfb_rd_base = (wave_id_n * T::WARP_SIZE + sf_rd_lane) * T::VEC_LDS_SCALE;
    constexpr int sf_half_stride = 2 * T::WARP_SIZE * T::VEC_LDS_SCALE;
    SF_KK v_sfa_next[2], v_sfb_next[2];
    int sf_read_off = 0;
    auto load_sfa = [&](auto& rd_sfa_, int buf, int tile_in_panel) {
        sf_read_off = buf * T::SFA_PANEL_BYTES + tile_in_panel * (T::SFA_PANEL_BYTES / 2);
        opus::static_for<2>([&](auto h) { v_sfa_next[h.value] = __builtin_bit_cast(SF_KK, load<T::VEC_LDS_SCALE>(rd_sfa_, sfa_rd_base + h.value * sf_half_stride + sf_read_off)); });
    };
    auto load_sfb = [&](auto& rd_sfb_, int, int) {
        opus::static_for<2>([&](auto h) { v_sfb_next[h.value] = __builtin_bit_cast(SF_KK, load<T::VEC_LDS_SCALE>(rd_sfb_, sfb_rd_base + h.value * sf_half_stride + sf_read_off)); });
    };
    auto read_sf = [&](auto&, auto&, auto h_) {
        constexpr int h = decltype(h_)::value;
        opus::static_for<T::E_K>([&](auto kk) { v_sfa[h][kk.value] = v_sfa_next[h][kk.value]; v_sfb[h][kk.value] = v_sfb_next[h][kk.value]; });
    };

''')

# --- schedule constants ---
rep('    constexpr int SFR = T::E_M + T::E_N;                                       // scale ds_read2st64 per K tile\n',
    '    constexpr int SFR = 4;                                                     // scale ds_read_b64 per K tile (2 halves x A, B)\n')
rep('    constexpr int SV = (T::E_M + 1) / 2 + (T::E_N + 1) / 2;                    // scale shifts per half\n',
    '    constexpr int SV = 0;                                                      // read_sf: plain register moves\n'
    '    constexpr int SVT = 16;                                                    // put_sf: 4 transposes x (2 DPP + 2 v_perm)\n')

# --- prologue (m8): panel 0 / sfb of panel 1 via VGPR, written after the first barrier ---
cut('    // SF_DMA2: panel 0 (sfa, sfb) and sfb of panel 1 go global -> LDS by DMA', '    views(s_a.ptr, s_b.ptr, s_sfa.ptr, s_sfb.ptr, [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {\n        // Tile 0 (copied by every wave)',
'''    // Panel 0 (sfa, sfb) and sfb of panel 1 are loaded into VGPRs here (a prefetched tile's panel 0 was loaded in the
    // previous tile's C11) and transposed into LDS after the first barrier; sfa of panel 1 stays in v_sf_pend and is
    // written in iteration 0.
    if (!prefetched) {
        prefetch_a(0, 0, 0);
        prefetch_b(0, 0, 0);
        prefetch_a(0, 0, 1);
        prefetch_b(0, 0, 1);
        v_nsfa0 = fetch_sfa(0);
        v_nsfb0 = fetch_sfb(0);
    }
    SF_PANEL v_sfb1;
    if (loops > T::SCALE_PANEL_TILES) {
        v_sfb1 = fetch_sfb(T::SCALE_PANEL_TILES);
        v_sf_pend = fetch_sfa(T::SCALE_PANEL_TILES);
    }
    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
        s_waitcnt_vmcnt(number<2 * (T::a_buffer_load_insts + T::b_buffer_load_insts) + 1>{});   // tile 1 + pending sfa may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
''')
rep('''        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, T::E_K * (T::E_M + T::E_N), 0);   // A0, B0 first''',
'''        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        put_sf(rd_sfa, ssfa_wbase, v_nsfa0, 0);
        put_sf(rd_sfb, ssfb_wbase, v_nsfb0, 0);
        if (loops > T::SCALE_PANEL_TILES) put_sf(rd_sfb, ssfb_wbase, v_sfb1, 1);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, T::E_K * (T::E_M + T::E_N), 0);   // A0, B0 first''')

# --- main loop (even / odd): m8 placement -- A1 / B1 copies in C01, panel put + fetch in C11 ---
rep('''                LOAD_B(rd_b, 0, tile + 1);
                opus::static_for<frags>''',
'''                LOAD_B(rd_b, 0, tile + 1);
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>''', 2)
rep('                SCHED(sched_c01_nov(), RB + SFR, 0, 0, 0);', '                SCHED(SCHED_C01(), RB + SFR, QA + QB, 0, 0);', 2)
rep('''                read_sf(rd_sfa, rd_sfb, number<0>{});
                prefetch_a(stage, tile + 2, 1);
''', '''                read_sf(rd_sfa, rd_sfb, number<0>{});
''', 2)
rep('                SCHED(sched_c11_pf(), RA, QA + 1, SV, 0);', '                SCHED(sched_c11_ns(), RA, 1, SVT, 0);', 2)
rep('''                read_sf(rd_sfa, rd_sfb, number<1>{});
                prefetch_b(stage, tile + 2, 1);
''', '''                read_sf(rd_sfa, rd_sfb, number<1>{});
''', 2)
rep('                SCHED(sched_c00_pf(), RA + RB, QB, SV, 0);', '                SCHED(SCHED_C00(), RA + RB, 0, SV, 0);', 2)
# tail loop / epilogue C11: no panel put in the quadrant
rep('SCHED(sched_c11(), RA, 1, SV, 0);', 'SCHED(sched_c11_ns(), RA, 1, SV, 0);', 2)

# --- last tile: next tile's panel 0 into VGPRs (row-major) ---
i0 = s.index('        auto ng_sfa = make_gmem('); i1 = s.index('\n', s.index('        auto ng_sfb = make_gmem(', i0)) + 1
s = s[:i0] + '''        auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + n_batch * kargs.stride_sfa_batch + n_row * kargs.stride_sfa,
                                has_next ? static_cast<unsigned int>((kargs.m - n_row) * kargs.stride_sfa) : 0u);
        auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + n_batch * kargs.stride_sfb_batch + n_col * kargs.stride_sfb,
                                has_next ? static_cast<unsigned int>((kargs.n - n_col) * kargs.stride_sfb) : 0u);
''' + s[i1:]
cut('        if constexpr (T::PERSIST) {   // sc4: lane offsets re-derived here', '        opus::static_for<frags>([&](auto i) {\n            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;\n            mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 1, m, n>()>',
'''        if constexpr (T::PERSIST) {   // next tile's panel 0; lane offsets re-derived here (opaque lane id) so the per-tile
                                      // ones need not stay live through the tile (they would spill)
            unsigned int lm = ~0u; asm volatile("" : "+s"(lm));
            const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lm, __builtin_amdgcn_mbcnt_lo(lm, 0u)));
            const int e = ln % 4, r = ln / 4;
            v_nsfa0 = load<T::VEC_GLOBAL_SCALE>(ng_sfa, (wave_id_n * T::HALF_B_M + e * 32 + wave_id_m * 16 + r) * kargs.stride_sfa);
            v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, (wave_id_m * T::HALF_B_N + e * 32 + wave_id_n * 16 + r) * kargs.stride_sfb);
        }
''')
rep('SCHED((sched_x4<2, 0, 1, 0>(), sched_x4<6, 0, 0, 0>()), 0, 3, 0, 0);', 'SCHED((sched_x4<2, 0, 1, 0>(), sched_x4<6, 0, 0, 0>()), 0, 2, 0, 0);')

# --- macro hygiene: #undef before the file-scope defines ---
for m in ['SCHED_C01', 'SCHED_C00', 'SCHED_C00_STORE', 'BB_VMCNT']:
    s = re.sub(r'(?m)^#define %s\b' % m, '#undef %s\n#define %s' % (m, m), s, count=1)
s = re.sub(r'(?m)^#define SCHED\(', '#undef SCHED\n#define SCHED(', s, count=1)

# --- close namespace + undef macros at the end ---
s = s.rstrip() + '''

}   // namespace opus_mxfp4_ns
#undef SCHED_C01
#undef SCHED_C00
#undef SCHED_C00_STORE
#undef BB_VMCNT
#undef SCHED
#undef SFA
#undef SFB
#undef SOA
#undef SOB
#undef SF_STEP_EVEN
#undef SF_STEP_ODD
#undef LOAD_B
'''
s = s.replace('#pragma once\n', '#pragma once\n// MXFP4 x MXFP4 NT 2x2 4-wave pipeline, no-shuffle inputs (row-major A / B / scales).  Generated from the shuffle\n// pipeline (opus_gemm_pipeline_a4w4_mxfp4_nt_gfx950.cuh) with the m8 B layout and scale path; see mxfp4_nt_noshuffle_dev.\n', 1)
s = s.replace('    // Split-K: this block\'s K segment [k0, k0 + k / S); bases below are moved to k0 (A: bytes, B: 2 KB per K tile in each\n    // 16-row block of shuffle_weight(16,16), scales: 256 B per K tile in each 32-row block of e8m0_shuffle).\n',
              '    // K segment [k0, k0 + k / S): split-K is not supported by the no-shuffle traits (S = 1, k0 = 0); kept for the shared code.\n')
open(dst, 'w').write(s)
print("ok", len(s.splitlines()), "lines")
