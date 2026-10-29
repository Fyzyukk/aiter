# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only gfx950 MXFP8 candidate selection shared with the GPU tuner."""

import argparse
import csv
import math
from collections import Counter, defaultdict

from csrc.opus_gemm.opus_gemm_common import (
    a8w8_mxscale_bpreshuffle_tune_candidates,
    candidate_kids_for_shape,
    get_registry,
)

PIPELINES = (
    "pin_agpr",
    "pingpong_lds",
    "prefetch_b_vgpr",
    "prefetch_b_vgpr_overlap",
    "wave_interleave",
)
ROUTES = ("jit", "co")


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return number


def opus_candidates(
    m,
    n,
    k,
    seeds=(),
    *,
    cu_num,
    top_geometries=8,
    pipelines=None,
    routes=None,
    verbose=True,
):
    if top_geometries < 0:
        raise ValueError(
            "top_geometries must be nonnegative; zero keeps all geometries"
        )
    if pipelines is not None and set(pipelines) - set(PIPELINES):
        raise ValueError(f"unknown OPUS pipelines: {set(pipelines) - set(PIPELINES)}")
    if routes is not None and set(routes) - set(ROUTES):
        raise ValueError(f"unknown OPUS routes: {set(routes) - set(ROUTES)}")

    registry = get_registry("a8w8_blockscale_bpreshuffle", "gfx950", cu_num)
    candidates = []
    for kid in candidate_kids_for_shape("gfx950", m, n, k, cu_num=cu_num):
        instance = registry[kid]
        entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
        route = "co" if instance.bpreshuffle_co else "jit"
        if pipelines is not None and entry["pipeline"] not in pipelines:
            continue
        if routes is not None and route not in routes:
            continue
        candidates.append((kid, instance))
    seed_kids = set(seeds) & {kid for kid, _ in candidates}
    candidate_count = len(candidates)
    candidates = a8w8_mxscale_bpreshuffle_tune_candidates(candidates, seed_kids)
    groups = defaultdict(list)
    for kid, instance in candidates:
        entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
        geometry = (
            entry["pipeline"],
            instance.B_M,
            instance.B_N,
            instance.B_K,
            instance.T_M,
            instance.T_N,
        )
        groups[geometry].append(kid)
    by_pipeline = defaultdict(list)
    for geometry in groups:
        pipeline, bm, bn, _, _, _ = geometry
        tiles = math.ceil(m / bm) * math.ceil(n / bn)
        padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
        occupancy = tiles / max(cu_num, 1)
        cost = (
            padding
            - 1
            + 2 * max(0, 1 - occupancy)
            + 0.1 * (math.ceil(occupancy) / occupancy - 1)
            + 8 / bm
            + 8 / bn
        )
        by_pipeline[pipeline].append((cost, geometry))
    retained = set()
    for geometries in by_pipeline.values():
        ranked = sorted(geometries)
        if top_geometries:
            ranked = ranked[:top_geometries]
        retained.update(geometry for _, geometry in ranked)
    selected = {kid for geometry in retained for kid in groups[geometry]}
    selected.update(seed_kids)
    if verbose:
        route_counts = Counter(
            "co" if registry[kid].bpreshuffle_co else "jit" for kid in selected
        )
        print(
            f"[OPUS search] shape={(m, n, k)} legal={candidate_count} "
            f"geometries={len(retained)}/{len(groups)} top={top_geometries} "
            f"candidates={len(selected)} jit={route_counts['jit']} "
            f"co={route_counts['co']} seeds={len(seed_kids)}",
            flush=True,
        )
    return [(kid, registry[kid]) for kid in sorted(selected)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--untune_file", required=True, help="shape CSV")
    parser.add_argument("-o", "--output", required=True, help="candidate CSV")
    parser.add_argument(
        "--cu_num",
        type=int,
        default=256,
        help="CU count when the input does not declare one",
    )
    parser.add_argument(
        "--opus_top_geometries",
        type=nonnegative_int,
        default=8,
        help="geometries per pipeline; zero keeps all",
    )
    parser.add_argument("--opus_pipelines", choices=PIPELINES, nargs="+")
    parser.add_argument("--opus_routes", choices=ROUTES, nargs="+")
    args = parser.parse_args()
    with open(args.untune_file) as stream:
        shapes = list(csv.DictReader(stream))
    fields = (
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "w_scale_block",
        "kernelId",
        "pipeline",
        "route",
        "B_M",
        "B_N",
        "B_K",
        "T_M",
        "T_N",
        "kernelName",
    )
    shape_count = candidate_count = 0
    seen = set()
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in shapes:
            gfx = row.get("gfx", "gfx950")
            block = row.get("w_scale_block", "128x128")
            if gfx != "gfx950" or block != "128x128":
                parser.error("candidate export supports gfx950 / 128x128 only")
            cu = int(row.get("cu_num") or args.cu_num)
            m, n, k = (int(row[key]) for key in ("M", "N", "K"))
            key = (cu, m, n, k)
            if key in seen:
                continue
            seen.add(key)
            shape_count += 1
            for kid, instance in opus_candidates(
                m,
                n,
                k,
                cu_num=cu,
                top_geometries=args.opus_top_geometries,
                pipelines=args.opus_pipelines,
                routes=args.opus_routes,
            ):
                entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
                writer.writerow(
                    {
                        "gfx": gfx,
                        "cu_num": cu,
                        "M": m,
                        "N": n,
                        "K": k,
                        "w_scale_block": block,
                        "kernelId": kid,
                        "pipeline": entry["pipeline"],
                        "route": "co" if instance.bpreshuffle_co else "jit",
                        **{
                            field: getattr(instance, field)
                            for field in ("B_M", "B_N", "B_K", "T_M", "T_N")
                        },
                        "kernelName": instance.name,
                    }
                )
                candidate_count += 1
    print(
        f"[OPUS export] shapes={shape_count} candidates={candidate_count} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
