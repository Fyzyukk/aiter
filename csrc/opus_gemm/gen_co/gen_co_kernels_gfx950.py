#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate pin-only gfx950 COs, preserving symbols and remote configurations.

The local sweep precedes missing original remote configurations. Candidates
are deduplicated and numbered contiguously; pruning options preserves symbols.
"""

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

LDS_LIMIT = 160 * 1024
WARP_SIZE = 64
MFMA_M = MFMA_N = 16
MFMA_K = 128
VECTOR_SIZE = 16
B_GEOMETRIES = ((64, 128, 2), (96, 128, 4), (96, 256, 4), (128, 128, 4))
# Original fyz/gfx950-co-spill-restore-20261029 (97ae7fbf259c) configurations
# follow the local sweep. Keep their complete traits/body and image names.
REMOTE_B_GEOMETRIES = ((48, 128, 2), (64, 128, 4), (96, 256, 4), (128, 256, 4))
PIPELINE_PREFIX = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_"
TRAITS_PREFIX = "opus_gemm_blockscale_mxscale_bpreshuffle_"
DEVICE_FLAGS = [
    "-fgpu-flush-denormals-to-zero",
    "-fno-offload-uniform-block",
    "-fno-gpu-rdc",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
    "-mllvm",
    "--amdgpu-kernarg-preload-count=32",
    "-mllvm",
    "--amdgpu-mfma-vgpr-form",
    "-mllvm",
    "--lsr-drop-solution=1",
    "-mllvm",
    "-amdgpu-early-inline-all=true",
    "-mllvm",
    "-amdgpu-function-calls=false",
    "-mllvm",
    "-enable-post-misched=0",
]


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def cpp_values(values):
    return [str(value).lower() for value in values]


def record(
    pipeline,
    tile,
    waves,
    stages,
    traits_name,
    traits_args,
    body_func,
    header,
    body_args,
    max_k,
    fixed_k=0,
    pin_mask=0,
    grid_layout="n_m",
    pad_m=True,
    wide_output=True,
):
    if not pin_mask:
        raise ValueError("CO candidates must declare their register pinning")
    threads = waves[0] * waves[1] * WARP_SIZE
    return {
        "pipeline": pipeline,
        "block_size": threads,
        "tile": list(tile),
        "stages": stages,
        "wave_layout": list(waves),
        "launch_bounds": [threads, 1],
        "device_flags": "flags_00",
        "pad_m": pad_m,
        "wide_output": wide_output,
        "max_k": max_k,
        "fixed_k": fixed_k,
        "traits": {"name": traits_name, "args": cpp_values(traits_args)},
        "body_func": body_func,
        "body_args": list(body_args),
        "pipeline_header": PIPELINE_PREFIX + header + "_gfx950.cuh",
        "grid_layout": grid_layout,
        "pin_variant": pipeline == "pin_agpr" or bool(pin_mask),
        "pin_mask": pin_mask,
    }


def pin_agpr_candidates(pad_m_values=(True, False), *, remote_original=False):
    unroll = 2
    for pad_m, panel in itertools.product(pad_m_values, (16, 32, 64)):
        wrapper = (
            TRAITS_PREFIX + "4wave_pin_agpr_traits<CoTraits>"
            if remote_original else
            TRAITS_PREFIX + "4wave_pin_agpr_traits<CoTraits, "
            "opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile, "
            "opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds, false, true, 0>"
        )
        candidate = record(
            "pin_agpr",
            (256, 256, 128),
            (2, 2),
            2,
            TRAITS_PREFIX + "4wave_pin_agpr_panel_traits",
            (panel, unroll, pad_m),
            TRAITS_PREFIX + "4wave_pin_agpr",
            "4wave_pin_agpr",
            (wrapper,),
            None if remote_original else 16384,
            pin_mask=3,
            pad_m=pad_m,
            wide_output=not remote_original,
        )
        # The padded scale load guards the start of each 16-byte M vector;
        # the unpadded path loads and stores a complete 256-row tile.
        candidate["m_align"] = 16 if pad_m else candidate["tile"][0]
        # C01 is tagged with C01_AGPR_BASE=148; its sixteen MFMA
        # accumulators occupy 64 dwords. The other C halves are not all pinned.
        candidate["partial_pin_c"] = {"base": 148, "dwords": 64}
        # Keep existing CO image names; k0/reset0 are only legacy name tokens.
        suffix = (
            ("" if pad_m else "pm0_") + f"sp{panel}_u{unroll}"
            if remote_original else
            f"pm{int(pad_m)}_sp{panel}_k0_reset0_u{unroll}_gm0_wide_nt0"
        )
        yield candidate, suffix


def b_vgpr_lds(bm, bn, wn, bk, nb, fixed_k, pin_a, pin_c):
    threads = wn * WARP_SIZE
    mr, nr = bm // MFMA_M, bn // (wn * MFMA_N)
    if bn % (wn * MFMA_N) or mr * nr * 4 > 256:
        return None
    if bm * bk % (threads * VECTOR_SIZE) or fixed_k % bk:
        return None
    if fixed_k // bk < nb - 1:
        return None
    if pin_a and bk // MFMA_K * mr * 8 > 128:
        return None
    if pin_c and mr * nr * 4 > 128:
        return None
    scales = fixed_k // MFMA_K
    sa_bytes = ceil_div(bm * scales, threads * VECTOR_SIZE) * threads * VECTOR_SIZE
    if ceil_div(mr, 4) * 64 * scales > sa_bytes:
        return None
    b_scale_rows = ceil_div(bn, 128)
    b_scale_slots = scales if scales % 4 else scales // 4
    if b_scale_rows * b_scale_slots > threads:
        return None
    lds = nb * bm * bk + sa_bytes + threads * 4
    if ceil_div(lds, 1280) * 1280 > LDS_LIMIT:
        return None
    return lds


def b_vgpr_candidates(k_values, *, remote_original=False, geometries=None):
    if geometries is None:
        # Keep each original sweep first, then cover all seven geometries with
        # that sweep's BD/XCD/Contiguous bundle.
        original = REMOTE_B_GEOMETRIES if remote_original else B_GEOMETRIES
        geometries = tuple(dict.fromkeys((*original, *B_GEOMETRIES, *REMOTE_B_GEOMETRIES)))
    ahead = 1 if remote_original else 2
    for (bm, bn, wn), fixed_k in itertools.product(geometries, k_values):
        bk, nb = 128, 4
        for pin_a, pin_c in itertools.product((False, True), repeat=2):
            if not (pin_a or pin_c):
                continue
            if b_vgpr_lds(bm, bn, wn, bk, nb, fixed_k, pin_a, pin_c) is None:
                continue
            mask = int(pin_a) + 4 * int(pin_c)
            args = (
                bm,
                bn,
                nb,
                ahead,
                0,
                not remote_original,  # XCD tile order.
                fixed_k,
                wn,
                1,
                0,
                2,
                0,
                bk,
                not remote_original,  # A/B strides specialized to K when true.
                pin_a,
                pin_c,
                0,
            )
            candidate = record(
                "prefetch_b_vgpr",
                (bm, bn, bk),
                (1, wn),
                nb,
                TRAITS_PREFIX + "prefetch_b_vgpr_traits",
                args,
                "opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body",
                "b_vgpr",
                ("CoTraits",),
                fixed_k,
                fixed_k=fixed_k,
                pin_mask=mask,
                grid_layout="linear" if remote_original else "xcd",
            )
            suffix = (
                f"p{nb}_bd{ahead}_k{fixed_k}_pinA{int(pin_a)}B0C{int(pin_c)}"
                if remote_original else
                f"p{nb}_a2_k{fixed_k}_c0x0_gm0_o0_contig_"
                f"pinA{int(pin_a)}B0C{int(pin_c)}_nt0"
            )
            yield candidate, suffix


def b_vgpr_overlap_candidates(k_values, *, remote_original=False):
    for candidate, suffix in b_vgpr_candidates(k_values, remote_original=remote_original):
        candidate["pipeline"] = "prefetch_b_vgpr_overlap"
        candidate["traits"]["name"] = TRAITS_PREFIX + "prefetch_b_vgpr_overlap_traits"
        candidate["traits"]["args"].append("0")
        candidate["traits_header"] = (
            "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_overlap_gfx950.cuh"
        )
        if remote_original:
            candidate["body_args"] = ["CoTraits", "CoTraits::PUBLISH_M"]
        else:
            candidate["body_func"] = (
                "opus_gemm_prefetch_b_vgpr_detail::bulk_panel_overlap_body"
            )
            suffix = suffix.removesuffix("_nt0") + "_pub0_nt0"
        yield candidate, suffix


def generate_document(k_values):
    k_values = sorted(set(k_values))
    local_candidates = itertools.chain(
        pin_agpr_candidates((True,)),
        b_vgpr_candidates(k_values),
        b_vgpr_overlap_candidates(k_values),
        pin_agpr_candidates((False,)),
    )
    remote_candidates = itertools.chain(
        pin_agpr_candidates((False, True), remote_original=True),
        b_vgpr_candidates(k_values, remote_original=True),
        b_vgpr_overlap_candidates(k_values, remote_original=True),
    )
    records, suffixes = [], {}
    configs = set()
    sources = {}
    for source, candidates in (("local", local_candidates), ("remote_original", remote_candidates)):
        first_kid, duplicates = len(records), 0
        for candidate, suffix in candidates:
            config = json.dumps(candidate, sort_keys=True)
            if config in configs:
                duplicates += 1
                continue
            configs.add(config)
            kid = len(records)
            candidate["kid"] = kid
            candidate["configKey"] = f"bpreshuffle_co_{candidate['pipeline']}_{kid}"
            records.append(candidate)
            suffixes[str(kid)] = suffix
        sources[source] = {
            "added": len(records) - first_kid,
            "duplicates_skipped": duplicates,
            "kid_range": [first_kid, len(records) - 1] if len(records) > first_kid else [],
        }
    return {
        "_generation": {"candidate_sources": sources},
        "defaults": {
            "tag": "a8w8_blockscale_mxscale_gemm_bpreshuffle",
            "dtypes": {"a": "fp8_t", "b": "fp8_t", "c": "bf16_t", "acc": "fp32_t"},
        },
        "device_flag_sets": {"flags_00": list(DEVICE_FLAGS)},
        "image_suffixes": suffixes,
        "gfx950": records,
    }


def validate_document(document, shapes):
    records = document["gfx950"]
    kids = {record["kid"] for record in records}
    if len(kids) != len(records):
        raise ValueError("Duplicate local CO kid")
    if any(type(kid) is not int or kid < 0 for kid in kids):
        raise ValueError("Local CO kids must be nonnegative integers")
    if kids != set(range(len(records))):
        raise ValueError("Local CO kids must be contiguous starting at zero")
    symbols = set()
    for candidate in records:
        if not candidate.get("pin_mask", 0):
            raise ValueError("CO candidates must declare their register pinning")
        geometry = (tuple(candidate["tile"]), tuple(candidate["wave_layout"]))
        symbol = (
            candidate["pipeline"],
            geometry,
            document["image_suffixes"][str(candidate["kid"])],
        )
        if symbol in symbols:
            raise ValueError("Two generated COs share an image/symbol")
        symbols.add(symbol)
        if candidate["launch_bounds"][0] != candidate["block_size"]:
            raise ValueError("Inconsistent block size and launch bounds")
        if candidate["pipeline"] == "pin_agpr":
            if candidate["pin_mask"] != 3 or candidate.get("partial_pin_c") != {"base": 148, "dwords": 64}:
                raise ValueError("4wave pin metadata must describe A/B and the C01 AGPR range")
            pad_m = candidate["pad_m"]
            if candidate["m_align"] != (16 if pad_m else candidate["tile"][0]):
                raise ValueError("Pin M alignment differs from its PAD_M path")
            if candidate["traits"]["args"][-1] != str(pad_m).lower():
                raise ValueError("Pin PAD_M differs between metadata and traits")
            suffix = document["image_suffixes"][str(candidate["kid"])]
            if not (suffix.startswith(f"pm{int(pad_m)}_")
                    or pad_m and suffix.startswith("sp")):
                raise ValueError("Pin PAD_M differs between metadata and image name")
        if candidate["pipeline"] == "prefetch_b_vgpr_overlap":
            publish = int(candidate["traits"]["args"][-1])
            if not 0 <= publish < candidate["tile"][0] // MFMA_M - 1:
                raise ValueError("Overlap publication must precede the final M repeat")
    uncovered = []
    for m, n, k in shapes:
        if not any(
            m % candidate.get("m_align", MFMA_M) == 0
            and (candidate["pad_m"] or m % candidate["tile"][0] == 0)
            and n % candidate["tile"][1] == 0
            and k % candidate["tile"][2] == 0
            and (candidate["max_k"] is None or k <= candidate["max_k"])
            and (not candidate["fixed_k"] or candidate["fixed_k"] == k)
            and m * k <= 2147483647
            and n * k <= 2147483647
            and (candidate["wide_output"] or 2 * m * n <= 2147483647)
            and 2 * ((min(m, candidate["tile"][0]) - 1) * n + candidate["tile"][1])
            <= 2147483647
            for candidate in records
        ):
            uncovered.append((m, n, k))
    if uncovered:
        raise ValueError(f"No declared candidate covers input shapes: {uncovered}")
    return dict(Counter(candidate["pipeline"] for candidate in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_large421_untuned_gemm.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("co_kernels_gfx950.json"),
    )
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    shapes = [(int(row["M"]), int(row["N"]), int(row["K"])) for row in rows]
    k_values = sorted({k for _, _, k in shapes})
    if any(k <= 0 or k % MFMA_K for k in k_values):
        raise ValueError("Input K values must be positive multiples of 128")
    document = generate_document(k_values)
    counts = validate_document(document, shapes)
    print(
        json.dumps(
            {
                "input_shapes": len(shapes),
                "fixed_k_values": k_values,
                "co_counts": counts,
                "total_co": sum(counts.values()),
                "candidate_sources": document["_generation"]["candidate_sources"],
                "local_co_range": [
                    min(candidate["kid"] for candidate in document["gfx950"]),
                    max(candidate["kid"] for candidate in document["gfx950"]),
                ],
            },
            indent=2,
        )
    )
    if not args.count_only:
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
