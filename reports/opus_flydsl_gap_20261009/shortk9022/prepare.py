#!/usr/bin/env python3
"""Freeze CPU inputs and create a private, unregistered short-K 9022 variant."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = ROOT / "csrc/opus_gemm/include"
PIPELINE = "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"
OLD_SYMBOL = "gemm_a8w8_mxfp8_scale_4wave_160x128_kernel"
NEW_SYMBOL = "opus_private_shortk9022_panel8_kernel"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Refuse to overwrite a prepared experiment or its frozen control inputs.
    for target in [HERE / "frozen", HERE / "candidate/pipeline.cuh", HERE / "source_manifest.json"]:
        if target.exists():
            raise SystemExit(f"Refusing to overwrite {target}")
    (HERE / "candidate").mkdir(exist_ok=True)
    shutil.copytree(ORIGINAL, HERE / "frozen/gemm_include")
    shutil.copytree(ROOT / "csrc/include/opus", HERE / "frozen/opus")
    source = (ORIGINAL / PIPELINE).read_text()
    private = source.replace(OLD_SYMBOL, NEW_SYMBOL)
    private = private.replace(
        '#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"',
        '#include "traits.cuh"',
    )
    private = private.replace("    const int loops = kargs.k / T::B_K;", "    constexpr int loops = T::FIXED_K / T::B_K;")
    # Pass 1 must never be instantiated: the reused helper has a Pass<SFA_PASSES assertion.
    lines = private.splitlines(keepends=True)
    private = "".join(line for line in lines if "const auto u_gsfa_1 =" not in line and "const auto u_ssfa_1 =" not in line)
    private = private.replace(
        "opus::make_tuple(layout_to_offsets<T::VEC_SCALE_A>(u_gsfa_0), layout_to_offsets<T::VEC_SCALE_A>(u_gsfa_1))",
        "opus::make_tuple(layout_to_offsets<T::VEC_SCALE_A>(u_gsfa_0))",
    )
    private = private.replace(
        "opus::make_tuple(layout_to_offsets<T::VEC_SCALE_A>(u_ssfa_0), layout_to_offsets<T::VEC_SCALE_A>(u_ssfa_1))",
        "opus::make_tuple(layout_to_offsets<T::VEC_SCALE_A>(u_ssfa_0))",
    )
    assert source.count(OLD_SYMBOL) == 2 and private.count(NEW_SYMBOL) == 2
    assert "u_gsfa_1" not in private and "u_ssfa_1" not in private
    assert private.count("constexpr int loops = T::FIXED_K / T::B_K;") == 1
    (HERE / "candidate/pipeline.cuh").write_text(private)
    (HERE / "candidate/pipeline.diff").write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), private.splitlines(keepends=True),
        fromfile=f"frozen/gemm_include/{PIPELINE}", tofile="candidate/pipeline.cuh")))
    frozen_files = []
    for prefix, destination in [(ORIGINAL, HERE / "frozen/gemm_include"),
                                (ROOT / "csrc/include/opus", HERE / "frozen/opus")]:
        for original in sorted(prefix.rglob("*")):
            if original.is_file():
                frozen = destination / original.relative_to(prefix)
                assert sha(original) == sha(frozen)
                frozen_files.append({"original": str(original.relative_to(ROOT)),
                                     "frozen": str(frozen.relative_to(HERE)), "sha256": sha(original)})
    library = ROOT / "reports/opus_clang23_mixed_retune_20261008/jit/module_deepgemm_opus.so"
    manifest = {"status": "prepared_unvalidated_numerics", "cpu_only": True,
                "registered": False, "production_source_modified": False,
                "fixed_k": [384, 768], "scale_panel": 8,
                "original_pipeline_sha256": sha(ORIGINAL / PIPELINE),
                "private_pipeline_sha256": sha(HERE / "candidate/pipeline.cuh"),
                "frozen_files": frozen_files,
                "formal_control": {"path": str(library), "sha256": sha(library),
                                   "action": "read-only hash; not loaded or rebuilt"}}
    (HERE / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "frozen_file_count": len(frozen_files),
                      "private_pipeline_sha256": manifest["private_pipeline_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
