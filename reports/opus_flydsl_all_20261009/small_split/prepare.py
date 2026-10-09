#!/usr/bin/env python3
"""Freeze current sources and make private producer variants; no GPU imports."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPE = "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if (HERE / "frozen").exists():
        raise SystemExit("Refusing to overwrite frozen inputs")
    inputs = [(ROOT / "csrc/opus_gemm/include", HERE / "frozen/gemm_include"),
              (ROOT / "csrc/include/opus", HERE / "frozen/opus")]
    frozen = []
    for original, dest in inputs:
        shutil.copytree(original, dest)
        for p in sorted(original.rglob("*")):
            if p.is_file():
                copy = dest / p.relative_to(original)
                assert sha(p) == sha(copy)
                frozen.append({"original": str(p.relative_to(ROOT)), "frozen": str(copy.relative_to(HERE)), "sha256": sha(p)})
    old = (HERE / "frozen/gemm_include" / PIPE).read_text()
    new = old.replace("gemm_a8w8_mxfp8_scale_small_register_kernel", "opus_private_register_global_split_kernel")
    new = new.replace('    const int tiles_per_wave = total_tiles / T::WAVE_K;\n    const int extra_tiles = total_tiles % T::WAVE_K;',
                      '    const int split = block_id_z();\n'
                      '    const int global_per = total_tiles / T::GLOBAL_SPLIT_K;\n'
                      '    const int global_extra = total_tiles % T::GLOBAL_SPLIT_K;\n'
                      '    const int global_loops = global_per + (split < global_extra);\n'
                      '    const int global_begin = split * global_per + (split < global_extra ? split : global_extra);\n'
                      '    const int tiles_per_wave = global_loops / T::WAVE_K;\n'
                      '    const int extra_tiles = global_loops % T::WAVE_K;')
    new = new.replace('    const int tile_begin = wk * tiles_per_wave + (wk < extra_tiles ? wk : extra_tiles);',
                      '    const int tile_begin = global_begin + wk * tiles_per_wave + (wk < extra_tiles ? wk : extra_tiles);')
    new = new.replace('        if constexpr (T::WAVE_K > 1) kt += tile_begin;', '        kt += tile_begin;')
    new = new.replace('    auto gc = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) + row * args.stride_c + col,\n'
                      '                        static_cast<unsigned>(((args.m - row) * args.stride_c - col) * 2));',
                      '    auto gc = make_gmem(reinterpret_cast<float*>(args.ptr_c) + static_cast<int64_t>(split) * args.m * args.stride_c + row * args.stride_c + col,\n'
                      '                        static_cast<unsigned>(((args.m - row) * args.stride_c - col) * sizeof(float)));')
    begin = new.index('        if constexpr (T::OUTPUT == 3) {')
    end = new.index('\n    }\n}\n#endif', begin)
    new = new[:begin] + '''        // Sole global partial store after the unchanged local FP32 reduction.
        // BF16 conversion occurs once in the matching frozen reducer.
        const auto pc = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
        const auto uc = partition_layout_c<4>(mma, opus::make_tuple(args.stride_c, 1_I), pc);
        const auto offsets = layout_to_offsets<4>(uc);
        static_for<T::E_M * T::E_N>([&](auto i) {
            store<4>(gc, c[decltype(i)::value], offsets[decltype(i)::value]);
        });''' + new[end:]
    assert 'reinterpret_cast<bf16_t*>(args.ptr_c)' not in new
    assert new.count('T::GLOBAL_SPLIT_K') == 2
    (HERE / "candidate/register_global_split.cuh").write_text(new)
    (HERE / "candidate/register_global_split.diff").write_text(''.join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile='frozen/gemm_include/' + PIPE,
        tofile='candidate/register_global_split.cuh')))
    comparison = ROOT / "reports/opus_flydsl_gap_20261009/losers294.csv"
    shutil.copy2(comparison, HERE / "frozen/losers294.csv")
    manifest = {"status": "private_sources_prepared_no_gpu_validation", "gpu_operations": 0,
                "registered": False, "production_modified": False, "frozen_files": frozen,
                "comparison": {"path": str(comparison), "sha256": sha(comparison)},
                "variants": [110, 111, 120, 121, 130, 140, 210, 211, 220, 221, 230, 231],
                "scope": "52 register historical losers and all 38 fine historical losers; pools overlap other experiments"}
    (HERE / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({"prepared": True, "frozen_files": len(frozen), "gpu_operations": 0}))

if __name__ == '__main__':
    main()
