#!/usr/bin/env python3
"""Freeze current sources and create private fixed-panel and direct-B 9030 kernels."""
import difflib, hashlib, json, shutil
from fragment_b import fragment_b
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPELINE = "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"
OLD = "gemm_a8w8_mxfp8_scale_8wave_192x256_large_output_kernel"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if (HERE / "frozen").exists(): raise SystemExit("Refusing to replace frozen input")
    for source, destination in [(ROOT / "csrc/opus_gemm/include", HERE / "frozen/gemm_include"),
                                 (ROOT / "csrc/include/opus", HERE / "frozen/opus")]:
        shutil.copytree(source, destination)
    original = (HERE / "frozen/gemm_include" / PIPELINE).read_text()
    fixed = original.replace(OLD, "opus_private_9030_fixed1536_panel16_kernel")
    fixed = fixed.replace('#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"', '#include "traits.cuh"')
    fixed = fixed.replace("const int loops = kargs.k / T::B_K;", "constexpr int loops = T::FIXED_K / T::B_K;")
    fixed = fixed.replace("const int scale_k_groups = kargs.k / T::GROUP_K;", "constexpr int scale_k_groups = T::FIXED_K / T::GROUP_K;")
    # A shortened panel uses ceil M passes; inactive producers need the row guard.
    fixed = fixed.replace("smem_offset < T::SFA_BYTES && k_group < scale_k_groups", "local_row < T::B_M && smem_offset < T::SFA_BYTES && k_group < scale_k_groups")
    direct = fixed.replace("opus_private_9030_fixed1536_panel16_kernel", "opus_private_9030_directb_a3_chunk96_kernel")
    direct = "".join(line for line in direct.splitlines(keepends=True) if not any(t in line for t in [
        "const auto u_gb =", "const auto u_sb =", "const auto u_rb =", "auto s_b =", "auto sb_offset =",
        "async_load<T::VEC_B>(g_b",]))
    direct = direct.replace("typename decltype(mma)::vtype_b v_b;", "typename decltype(mma)::vtype_b v_b, v_b_next;")
    direct = direct.replace("auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {\n        constexpr int stage = decltype(stage_i)::value;", "auto issue_matrix_prefetch = [&](int stage, int tile_k) {")
    direct = direct.replace("        static_assert(stage >= 0 && stage < T::NUM_STAGES);\n", "")
    direct = direct.replace("auto load_a_fragment = [&](auto m_i, auto stage_i) {\n        constexpr int m_repeat = decltype(m_i)::value;\n        constexpr int stage = decltype(stage_i)::value;", "auto load_a_fragment = [&](auto m_i, int stage) {\n        constexpr int m_repeat = decltype(m_i)::value;")
    start = direct.index("    auto load_b_fragment =")
    stop = direct.index("    auto mma_scale_fragment =", start)
    direct = direct[:start] + '''    // Two register sets: B(t+1) is loaded globally while B(t) is consumed.
    // Standard (16,16) preshuffle: each N16/K128 block consists of two K64 halves.
    auto load_b_direct = [&](auto& target, int tile_k) {
        static_for<T::E_N>([&](auto n_i) {
            constexpr int n_repeat = decltype(n_i)::value;
            const int group = n_repeat * T::T_N + wave_id_n;
            const int base = group * T::W_N * kargs.stride_b + tile_k * T::B_K * T::W_N + lane_id * T::VEC_B;
            static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
                constexpr int chunk = decltype(chunk_i)::value;
                constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + chunk;
                set_slice(target, load<T::VEC_B>(g_b, base + chunk * T::WARP_SIZE * T::VEC_B),
                          number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
            });
        });
    };
''' + direct[stop:]
    start = direct.index("    auto advance_tile =")
    direct = direct[:start] + '''    // Fill two of three A slots; the third is free while current A is consumed.
    load_sfa_panel(0);
    load_sfb_panel(0);
    issue_matrix_prefetch(0, 0);
    issue_matrix_prefetch(1, 1);
    load_b_direct(v_b, 0);
    load_b_direct(v_b_next, 1);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    // Three-slot A ring, B ahead one. Keep explicit completion and reader retirement.
    #pragma unroll 1
    for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
        if (tile_k + 2 < loops)
            issue_matrix_prefetch((tile_k + 2) % T::NUM_STAGES, tile_k + 2);
        static_for<T::E_M>([&](auto m_i) {
            static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
        });
        // Completes B(t+1) and future A. Barrier publishes the A copies.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_b = v_b_next;
        read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
        static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, (tile_k + 1) % T::NUM_STAGES); });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        if (tile_k + 2 < loops)
            load_b_direct(v_b_next, tile_k + 2);
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    }
    static_for<T::E_M>([&](auto m_i) {
        static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
    });
    // All A readers finish before the matrix arena is aliased for BF16 C.
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    static_for<T::C_CHUNKS>([&](auto chunk_i) {
        constexpr int first_row = decltype(chunk_i)::value * T::C_CHUNK_ROWS;
        static_for<T::E_M * T::E_N>([&](auto c_i) {
            constexpr int c_index = decltype(c_i)::value;
            const int offset = gc_offsets[c_index];
            const int output_row = offset / T::C_LDS_ROW_STRIDE_ELEMS;
            if (output_row >= first_row && output_row < first_row + T::C_CHUNK_ROWS) {
                const auto value = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
                store<T::VEC_C>(s_c, cast<D_C>(value), offset - first_row * T::C_LDS_ROW_STRIDE_ELEMS);
            }
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
        static_for<T::CHUNK_OUTPUT_PASSES>([&](auto pass_i) {
            const int linear = output_thread_id * T::VEC_OUTPUT + decltype(pass_i)::value * T::BLOCK_SIZE * T::VEC_OUTPUT;
            const int local_row = linear / T::B_N;
            const int output_col = linear % T::B_N;
            const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(local_row, output_col));
            const int output_row = first_row + local_row;
            if (row + output_row < kargs.m)
                store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col, 0, number<2>{});
        });
        // Retire each chunk's LDS reads before the next chunk overwrites the arena.
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    });
}
#endif
'''
    direct = fragment_b(direct)
    for name, content in [("panel16", fixed), ("directb", direct)]:
        (HERE / name).mkdir()
        (HERE / name / "pipeline.cuh").write_text(content)
        (HERE / name / "pipeline.diff").write_text("".join(difflib.unified_diff(original.splitlines(keepends=True), content.splitlines(keepends=True), fromfile="frozen/" + PIPELINE, tofile=name + "/pipeline.cuh")))
    files = []
    for source, frozen in [(ROOT / "csrc/opus_gemm/include", HERE / "frozen/gemm_include"), (ROOT / "csrc/include/opus", HERE / "frozen/opus")]:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                copy = frozen / path.relative_to(source)
                assert sha(path) == sha(copy)
                files.append({"original": str(path.relative_to(ROOT)), "frozen": str(copy.relative_to(HERE)), "sha256": sha(path)})
    library = ROOT / "reports/opus_clang23_mixed_retune_20261008/jit/module_deepgemm_opus.so"
    report = {"status": "prepared_unvalidated_numerics", "cpu_only": True, "registered": False,
              "frozen_files": files, "fixed_k": 1536,
              "formal_control": {"path": str(library), "sha256": sha(library), "action": "hash only, not loaded"},
              "private_pipelines": {name: sha(HERE / name / "pipeline.cuh") for name in ["panel16", "directb"]}}
    (HERE / "source_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("prepared three private libraries: runtime, panel16, directb A3/chunk96; no GPU")

if __name__ == "__main__": main()
