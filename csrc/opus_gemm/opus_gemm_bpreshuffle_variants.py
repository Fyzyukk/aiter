# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Parameter tables for retained MXFP8 B-preshuffle pipeline variants.

One entry is an exact configuration, not a copied kernel body. This module is
scalar-only so selection, code generation and offline tools use the same table.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BpreshuffleVariant:
    kid: int
    label: str
    family: str
    pipeline_header: str
    traits_header: str
    kernel: str
    traits: str
    tile_m: int
    tile_n: int
    wave_m: int
    wave_n: int
    wave_k: int = 1
    m_align: int = 1
    max_m: int | None = None
    fixed_k: int = 0
    split_k: int = 1
    dynamic_lds: bool = False
    pin_agpr: bool = False
    reduce_vec: int = 16
    reduce_block: int = 128
    sfa_alignment: int = 16
    c_alignment: int = 16
    runtime_split_k: bool = False
    schedule: int = 0


_HEADER_PREFIX = "gfx950/opus_gemm_"
_MAIN_TRAITS = _HEADER_PREFIX + "traits_a8w8_mxscale_bpreshuffle_main_variants_gfx950.cuh"
_SMALL_TRAITS = _HEADER_PREFIX + "traits_a8w8_mxscale_bpreshuffle_small_variants_gfx950.cuh"
_LARGE_TRAITS = _HEADER_PREFIX + "traits_a8w8_mxscale_bpreshuffle_large_variants_gfx950.cuh"
_SHORT_TRAITS = _HEADER_PREFIX + "traits_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh"
_SYMBOL = "opus_gemm_mxscale_bpreshuffle_"


def _pipeline(name):
    return _HEADER_PREFIX + f"pipeline_a8w8_mxscale_bpreshuffle_{name}_gfx950.cuh"


def _variants():
    entries = []
    # BM, panel and fixed-K parameters share one two-stage geometry body.
    geometry = (
        (92000, 64, 32, 0), (92001, 96, 32, 0), (92002, 128, 32, 0),
        (92003, 96, 8, 384), (92004, 96, 8, 768),
        (92005, 128, 16, 1536), (92006, 160, 16, 1536),
        (92007, 96, 32, 3072), (92008, 96, 32, 7168),
        (92009, 128, 32, 7168), (92010, 128, 32, 3072),
    )
    for kid, bm, panel, fixed in geometry:
        args = f"{bm},2,{panel}" + (f",{fixed}" if fixed else "")
        entries.append(BpreshuffleVariant(
            kid, f"geometry_s2_p{panel}_k{fixed}", "geometry", _pipeline("geometry"),
            _MAIN_TRAITS, _SYMBOL + "geometry_kernel", _SYMBOL + f"geometry_traits<{args}>",
            bm, 128, 2, 2, m_align=16, fixed_k=fixed))
    # Scheduling order is a traits parameter of the existing narrow body.
    for kid, group_m in ((92011, 8), (92012, 16)):
        entries.append(BpreshuffleVariant(
            kid, f"tile_order_gm{group_m}_k7168", "tile_order", _pipeline("4wave_64x64"),
            _HEADER_PREFIX + "traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh",
            "gemm_a8w8_mxfp8_scale_4wave_64x64_kernel",
            f"opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,{group_m}>",
            64, 64, 2, 2, m_align=16, fixed_k=7168))
    for reset, first_id in ((False, 92100), (True, 92110)):
        for offset, (fixed, panel) in enumerate(((384, 16), (1536, 16), (3072, 32),
                                               (7168, 64), (16384, 64))):
            entries.append(BpreshuffleVariant(
                first_id + offset, f"pin_fixed_p{panel}_reset{int(reset)}_k{fixed}",
                "pin_fixed", _pipeline("pin_fixed"), _MAIN_TRAITS,
                _SYMBOL + "pin_fixed_kernel",
                _SYMBOL + f"pin_fixed_traits<{fixed},{panel},{str(reset).lower()}>",
                256, 256, 2, 2, m_align=256, fixed_k=fixed, pin_agpr=True))
    entries.append(BpreshuffleVariant(
        92120, "pad_pin_fixed_k7168", "pad_pin_fixed", _pipeline("pad_pin_fixed"),
        _MAIN_TRAITS, _SYMBOL + "pad_pin_fixed_kernel",
        _SYMBOL + "pad_pin_fixed_traits<7168>",
        256, 256, 2, 2, m_align=16, fixed_k=7168, pin_agpr=True))

    # The original short-K body is kept distinct to preserve its audited load
    # layout. The BM96 short-K entries above share the geometry body instead.
    for kid, fixed in ((92020, 384), (92021, 768)):
        entries.append(BpreshuffleVariant(
            kid, f"shortk160_p8_k{fixed}", "shortk", _pipeline("shortk"), _SHORT_TRAITS,
            _SYMBOL + "shortk_kernel", _SYMBOL + f"shortk_traits<{fixed}>",
            160, 128, 2, 2, m_align=16, fixed_k=fixed))

    # Baseline actual ID encodes tile, A stages, scale and XOR policy. Only
    # B's movement changes; ahead is an independent register-queue parameter.
    hybrid = ((9043, 32, 64, 1, 4), (9044, 64, 64, 2, 2),
              (9045, 96, 64, 2, 2), (9046, 64, 128, 4, 2),
              (9047, 32, 64, 2, 2), (9049, 32, 128, 2, 2),
              (9055, 32, 64, 1, 4), (9056, 64, 64, 2, 2))
    for index, (actual, bm, bn, wm, wn) in enumerate(hybrid):
        for ahead in (1, 2, 3):
            entries.append(BpreshuffleVariant(
                92200 + index * 3 + ahead - 1, f"direct_b_a{actual}_ahead{ahead}",
                "small_direct_b", _pipeline("small_direct_b"), _SMALL_TRAITS,
                _SYMBOL + "small_direct_b_kernel",
                _SYMBOL + f"small_direct_b_traits<{actual},{ahead}>",
                bm, bn, wm, wn, max_m=512 if actual in (9047, 9049) else 2048,
                dynamic_lds=True, sfa_alignment=1, c_alignment=8))
    register = ((92310, 16, 16, 1), (92311, 16, 16, 2),
                (92320, 16, 32, 1), (92321, 16, 32, 2),
                (92330, 32, 32, 1), (92340, 32, 64, 1))
    for kid, bm, bn, wk in register:
        entries.append(BpreshuffleVariant(
            kid, f"register_split_wk{wk}_sk4", "register_split", _pipeline("register_runtime"),
            _HEADER_PREFIX + "traits_a8w8_mxscale_bpreshuffle_runtime_splitk_gfx950.cuh",
            _SYMBOL + "register_runtime_kernel",
            _SYMBOL + f"register_runtime_traits<{bm},{bn},{wk}>",
            bm, bn, 1, 1, wave_k=wk, max_m=512, split_k=4, sfa_alignment=1,
            runtime_split_k=True))
    fine = ((92410, 48, 64, 1, 4, 1), (92411, 48, 64, 1, 4, 2),
            (92420, 64, 128, 2, 2, 1), (92421, 64, 128, 2, 2, 2),
            (92430, 96, 128, 2, 2, 1), (92431, 96, 128, 2, 2, 2))
    for kid, bm, bn, wm, wn, split in fine:
        runtime = split == 1
        entries.append(BpreshuffleVariant(
            kid, f"narrow_fine_s4_c1_sk{split}", "fine_lds", _pipeline("small_lds"),
            _SMALL_TRAITS, "gemm_a8w8_mxfp8_scale_small_lds_kernel",
            _SYMBOL + f"narrow_fine_traits<{bm},{bn},{wm},{wn},{split}>",
            bm, bn, wm, wn, max_m=2048, split_k=split, dynamic_lds=True))
        if runtime:
            entries[-1] = replace(entries[-1],
                pipeline_header=_pipeline("small_lds_runtime"),
                traits_header=_HEADER_PREFIX + "traits_a8w8_mxscale_bpreshuffle_runtime_splitk_gfx950.cuh",
                kernel="gemm_a8w8_mxfp8_scale_small_lds_runtime_kernel",
                traits=_SYMBOL + f"narrow_fine_runtime_traits<{bm},{bn},{wm},{wn}>",
                runtime_split_k=True)
    for kid, name in ((92500, "panel16"), (92501, "direct_b")):
        entries.append(BpreshuffleVariant(
            kid, f"large_output_{name}_k1536", "large_output",
            _pipeline("large_output_" + name), _LARGE_TRAITS,
            _SYMBOL + f"large_output_{name}_kernel",
            _SYMBOL + f"large_output_{name}_traits",
            192, 256, 4, 2, m_align=64, fixed_k=1536))
    # Five producer templates own all retained configurations. Schedule selects
    # a compile-time load/wait policy inside a shared body, not another kernel.
    policies = {
        "geometry": ("tiled", 2), "tile_order": ("tiled", 4),
        "shortk": ("tiled", 2), "pin_fixed": ("pin", 0),
        "pad_pin_fixed": ("pin", 2), "register_split": ("register", 1),
        "fine_lds": ("lds", 0), "small_direct_b": ("lds", 2),
        "large_output": ("large_output", 0),
    }
    result = []
    for entry in entries:
        pipeline, schedule = policies[entry.family]
        if entry.kid == 92501:
            schedule = 1
        result.append(replace(entry, pipeline_header=_pipeline(pipeline),
                              traits_header=_HEADER_PREFIX + "traits_a8w8_mxscale_bpreshuffle_gfx950.cuh",
                              kernel=_SYMBOL + pipeline + "_kernel", schedule=schedule))
    return tuple(sorted(result, key=lambda entry: entry.kid))


NEW_BPRESHUFFLE_VARIANTS = _variants()
NEW_BPRESHUFFLE_VARIANTS_BY_KID = {entry.kid: entry for entry in NEW_BPRESHUFFLE_VARIANTS}
assert len(NEW_BPRESHUFFLE_VARIANTS_BY_KID) == len(NEW_BPRESHUFFLE_VARIANTS) == 64
assert len({(entry.kernel, entry.traits) for entry in NEW_BPRESHUFFLE_VARIANTS}) == 64
