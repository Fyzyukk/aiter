#!/usr/bin/env python3
"""Generate a bounded gfx950 MXFP8 CO search space with consecutive local IDs.

GEMM shapes supply only compile-time K values. Tile legality is determined by
the pipeline traits, independently of a shape's M and N. Only explicit AGPR pin
configurations are emitted, numbered from zero before compilation. Unavailable
CO candidates keep their assigned IDs.
"""
import argparse
from collections import Counter
import csv
from itertools import product
import json
from pathlib import Path

ROOT = "opus_gemm_blockscale_mxscale_bpreshuffle_"
HEADER = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_"
LDS_LIMIT = 160 * 1024
STANDARD_FLAGS = [
    "-fgpu-flush-denormals-to-zero", "-fno-offload-uniform-block", "-fno-gpu-rdc",
    "-fvisibility=hidden", "-fvisibility-inlines-hidden",
    "-mllvm", "--amdgpu-kernarg-preload-count=32",
    "-mllvm", "--amdgpu-mfma-vgpr-form", "-mllvm", "--lsr-drop-solution=1",
    "-mllvm", "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false",
    "-mllvm", "-enable-post-misched=0",
]
# BK=128 requires BM*BK to divide evenly into the wave64 A-copy lanes.
# The BM=48 tile therefore uses two waves rather than four.
B_VGPR_GEOMETRIES = ((48, 128, 2), (64, 128, 4), (96, 256, 4), (128, 256, 4))


def m_repeat_rejection(bm, bn, wm, panel, pa=False, pb=False, pc=False):
    if bm % (wm * 16) or bn % 32:
        return "tile_wave_divisibility"
    em, en, block = bm // (wm * 16), bn // 32, wm * 2 * 64
    pitch = ((em + 3) // 4) * wm * 16 * 4
    matrix = 2 * ((bm * 128 // 1024 + bn * 128 // 1024) * 1056)
    lds = matrix + pitch * panel + 4 * panel
    if em < 2 or bn < (wm * 2) * 16:
        return "tile_wave_scale_lanes"
    if block % panel or 64 % (((em + 3) // 4) * wm * 4):
        return "scale_panel_divisibility"
    if lds > LDS_LIMIT:
        return "lds_capacity"
    if bm * (bn + 8) * 2 > lds or bm * bn % (block * 8):
        return "output_lds_capacity"
    if pb and en * 8 > 64:
        return "pin_b_capacity"
    if pc and 128 + em * en * 4 > 256:
        return "pin_c_capacity"
    return None


def arguments(values):
    return [str(v).lower() if isinstance(v, bool) else str(v) for v in values]


def generate(k_values):
    rows, suffixes = [], {}
    rejected = Counter()

    def add(pipeline, tile, waves, stages, traits, args, body, suffix,
            *, pin_mask, grid="linear", max_k=None, fixed_k=0,
            wide=True, body_args=None, partial_c=False, pad_m=True, m_align=None):
        if not pin_mask:
            raise ValueError("non-pin configurations must use JIT, not CO")
        kid = len(rows)
        row = {
            "kid": kid, "pipeline": pipeline, "block_size": 64 * waves[0] * waves[1],
            "tile": list(tile), "stages": stages, "wave_layout": list(waves),
            "launch_bounds": [64 * waves[0] * waves[1], 1],
            "device_flags": "standard",
            "pad_m": pad_m, "wide_output": wide, "max_k": max_k, "fixed_k": fixed_k,
            "traits": {"name": traits, "args": arguments(args)},
            "body_func": body, "body_args": body_args or ["CoTraits"],
            "grid_layout": grid,
            "pipeline_header": HEADER + {
                "pin_agpr": "4wave_pin_agpr", "pingpong_lds": "m_repeat",
                "prefetch_b_vgpr": "b_vgpr", "prefetch_b_vgpr_overlap": "b_vgpr",
            }[pipeline] + "_gfx950.cuh",
            # A search variant can be rejected after compilation without
            # stopping compilation of the remaining independent variants.
            "pin_variant": True, "pin_mask": pin_mask,
        }
        if partial_c:
            row["partial_pin_c"] = {"base": 148, "dwords": 64}
        if m_align is not None:
            row["m_align"] = m_align
        if pipeline == "prefetch_b_vgpr_overlap":
            row["traits_header"] = HEADER.replace("_pipeline_", "_traits_", 1) + "b_vgpr_overlap_gfx950.cuh"
        rows.append(row)
        suffixes[str(kid)] = suffix

    def add_pin_agpr(pad_m):
        for panel, unroll in product((16, 32, 64), (2, 4)):
            wrapper = ROOT + "4wave_pin_agpr_traits<CoTraits>"
            suffix = f"sp{panel}_u{unroll}"
            if not pad_m:
                suffix = "pm0_" + suffix
            add("pin_agpr", (256, 256, 128), (2, 2), 2,
                ROOT + "4wave_pin_agpr_panel_traits", (panel, 0, False, unroll, pad_m),
                ROOT + "4wave_pin_agpr", suffix, grid="n_m",
                pin_mask=3, wide=False, body_args=[wrapper], partial_c=True,
                pad_m=pad_m, m_align=16 if pad_m else 256)

    for pad_m in (False, True):
        add_pin_agpr(pad_m)

    selected_panels = []
    for bm, bn, wm in product((128, 192, 256), (128, 256), (2, 4)):
        panel = next((p for p in (128, 64)
                      if m_repeat_rejection(bm, bn, wm, p) is None), None)
        if panel is None:
            raise ValueError(f"no legal scale panel for m_repeat {bm=} {bn=} {wm=}")
        selected_panels.append({"BM": bm, "BN": bn, "TM": wm, "TN": 2,
                                "scale_panel": panel})
        for pa, pb, pc in product((False, True), repeat=3):
            mask = int(pa) | (int(pb) << 1) | (int(pc) << 2)
            if not mask:
                continue
            reason = m_repeat_rejection(bm, bn, wm, panel, pa, pb, pc)
            if reason:
                rejected["m_repeat:" + reason] += 1
                continue
            add("pingpong_lds", (bm, bn, 128), (wm, 2), 2,
                "opus_gemm_bpreshuffle_pingpong_lds_traits",
                (bm, bn, panel, 0, wm, 0, 0, 0, True, pa, pb, pc, False, False, 0, 8),
                "opus_gemm_bpreshuffle_pingpong_lds_body",
                f"sp{panel}_pinA{int(pa)}B{int(pb)}C{int(pc)}_frag",
                grid="n_m", max_k=panel * 128, pin_mask=mask)

    for pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        for (bm, bn, wn), k, pa, pc in product(
            B_VGPR_GEOMETRIES, k_values, (False, True), (False, True),
        ):
            mask = int(pa) | (int(pc) << 2)
            if not mask:
                continue
            bk, nb, bd, block = 128, 4, 1, 64 * wn
            mr, nr, sc = bm // 16, bn // (wn * 16), k // 128
            sa = (bm * sc + block * 16 - 1) // (block * 16) * block * 16
            pitch = ((mr + 3) // 4) * 64
            lds = nb * bm * bk + sa + block * 4
            reason = None
            if k <= 0 or k % bk:
                reason = "k_divisibility"
            elif k // bk < nb - 1:
                reason = "stage_priming"
            elif bm * bk % (block * 16):
                reason = "tile_wave_divisibility"
            elif pitch * sc > sa:
                reason = "packed_a_scale_capacity"
            elif ((bn + 127) // 128) * (sc if sc % 4 else sc // 4) > block:
                reason = "b_scale_lane_capacity"
            elif ((lds + 1279) // 1280) * 1280 > LDS_LIMIT:
                reason = "lds_capacity"
            elif mr * nr * 4 > 256:
                reason = "accumulator_capacity"
            elif pa and (bk // 128) * mr * 8 > 128:
                reason = "pin_a_capacity"
            elif pc and mr * nr * 4 > 128:
                reason = "pin_c_capacity"
            if reason:
                rejected[pipeline + ":" + reason] += 1
                continue
            args = (bm, bn, nb, bd, 0, False, k, wn, 1, 0, 2, 0, bk, False, pa, pc, 0)
            if pipeline.endswith("overlap"):
                args += (0,)
            add(pipeline, (bm, bn, bk), (1, wn), nb,
                ROOT + pipeline + "_traits", args,
                "opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body",
                f"p{nb}_bd{bd}_k{k}_pinA{int(pa)}B0C{int(pc)}",
                max_k=k, fixed_k=k, pin_mask=mask,
                body_args=["CoTraits", "CoTraits::PUBLISH_M"] if pipeline.endswith("overlap") else None)

    counts = dict(Counter(row["pipeline"] for row in rows))
    by_k = Counter(row["fixed_k"] for row in rows if row["pipeline"] == "prefetch_b_vgpr")
    summary = {
        "total_candidates": len(rows), "pipeline_counts": counts, "input_k_values": k_values,
        "local_kid_count": len(rows),
        "local_kid_range": [0, len(rows) - 1],
        "pin_agpr_candidates_by_pad_m": {
            str(pad_m).lower(): sum(row["pipeline"] == "pin_agpr" and row["pad_m"] == pad_m
                                   for row in rows)
            for pad_m in (False, True)
        },
        "b_vgpr_candidates_per_k": dict(sorted(by_k.items())), "static_rejections": dict(rejected),
        "b_vgpr_geometries": [{"BM": bm, "BN": bn, "TM": 1, "TN": wn}
                              for bm, bn, wn in B_VGPR_GEOMETRIES],
        "m_repeat_selected_panels": selected_panels,
        "fixed_b_vgpr": {"BK": 128, "stages": 4, "B_distance": 1, "WG_per_CU": 1,
                           "cache_A": 0, "cache_B": 0, "store_cache": 2, "output": 0,
                           "contiguous": False, "XCD": False, "group_M": 0, "publish_M": 0},
        "fixed_m_repeat": {"FixedK": 0, "cache_A": 0, "cache_B": 0, "group_M": 0,
                           "fragment_arrays": True, "XCD": False, "bounded_output": False,
                           "publish_M": 0, "group_N": 8},
    }
    document = {
        "_generation": summary,
        "defaults": {"tag": "a8w8_blockscale_mxscale_gemm_bpreshuffle",
                     "dtypes": {"a": "fp8_t", "b": "fp8_t", "c": "bf16_t", "acc": "fp32_t"}},
        "device_flag_sets": {"standard": STANDARD_FLAGS},
        "image_suffixes": suffixes, "gfx950": rows,
    }
    return document, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        k_values = sorted({int(row["K"]) for row in csv.DictReader(stream)})
    if not k_values or any(k <= 0 for k in k_values):
        parser.error("input must contain positive K values")
    document, summary = generate(k_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
