# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only gfx950 MXFP8 selection and historical kernel identity."""

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict

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


def _opus_geometry(instance):
    entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
    return (
        entry["pipeline"],
        instance.B_M,
        instance.B_N,
        instance.B_K,
        instance.T_M,
        instance.T_N,
    )


def _normalize_opus_geometry(value):
    """Accept [pipeline, BM, BN, BK, TM, TN] or named geometry fields."""
    if isinstance(value, dict):
        fields = {str(key).lower().replace("_", ""): val for key, val in value.items()}
        tile = fields.get("tile")
        waves = fields.get("wavelayout")
        if tile is not None and waves is not None:
            value = (fields["pipeline"], *tile, *waves)
        else:
            value = tuple(
                fields[key] for key in ("pipeline", "bm", "bn", "bk", "tm", "tn")
            )
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise ValueError("geometry must contain pipeline, BM, BN, BK, TM, TN")
    pipeline = str(value[0]).strip()
    if not pipeline:
        raise ValueError("geometry pipeline must not be empty")
    dimensions = tuple(int(str(dimension).strip()) for dimension in value[1:])
    if min(dimensions) <= 0:
        raise ValueError("geometry dimensions must be positive integers")
    return (pipeline, *dimensions)


def load_opus_geometry_file(path):
    """Read shape-specific geometries without importing any GPU code.

    CSV rows have M,N,K and pipeline,BM,BN,BK,TM,TN (B_M etc. also work),
    or a JSON old_opus_geometry/opus_geometry field containing the same tuple
    or named fields. Historical tuple literals are accepted too. Repeated rows
    select multiple geometries; blank geometry fields in analysis tables skip
    shapes for which no historical geometry was recorded.
    """
    by_shape = defaultdict(set)
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                fields = {
                    key.lower().replace("_", ""): value for key, value in row.items()
                }
                shape = tuple(int(fields[key]) for key in ("m", "n", "k"))
                if min(shape) <= 0:
                    raise ValueError("shape dimensions must be positive integers")
                encoded = fields.get("oldopusgeometry") or fields.get("opusgeometry")
                if not encoded and (
                    "oldopusgeometry" in fields or "opusgeometry" in fields
                ):
                    continue
                if encoded:
                    try:
                        value = json.loads(encoded)
                    except json.JSONDecodeError:
                        value = ast.literal_eval(encoded)
                    singular = isinstance(value, dict) or (
                        isinstance(value, (list, tuple))
                        and len(value) == 6
                        and isinstance(value[0], str)
                    )
                    values = [value] if singular else value
                else:
                    values = [fields]
                if not values:
                    raise ValueError("shape has no requested OPUS geometry")
                by_shape[shape].update(
                    _normalize_opus_geometry(value) for value in values
                )
            except (ValueError, TypeError, KeyError, SyntaxError) as exc:
                raise ValueError(
                    f"{path}:{line}: invalid OPUS geometry: {exc}"
                ) from exc
    if not by_shape:
        raise ValueError(f"{path}: no OPUS geometry rows")
    return dict(by_shape)


def _opus_jit_full_name(instance):
    """Historical JIT names encode every C++ trait, including default arguments.

    CO symbols receive no geometry alias: e.g. WideOutput and BoundedOutput
    differences cannot be inferred from a tile and wave layout.
    """
    entry = instance.bpreshuffle_entry
    if entry is None or not isinstance(entry.get("traits"), str):
        return None
    root = "opus_gemm_blockscale_mxscale_bpreshuffle_"
    # These tails mirror the respective traits headers. Required leading
    # arguments remain None so an incomplete or unfamiliar trait is rejected.
    traits_defaults = {
        root
        + "prefetch_b_vgpr_traits": (
            "prefetch_b_vgpr",
            (None,) * 4
            + (
                "0",
                "false",
                "7168",
                "4",
                "1",
                "0",
                "2",
                "0",
                "256",
                "false",
                "false",
                "false",
                "0",
            ),
        ),
        root
        + "prefetch_b_vgpr_overlap_traits": (
            "prefetch_b_vgpr_overlap",
            (None,) * 4
            + (
                "0",
                "false",
                "7168",
                "4",
                "1",
                "0",
                "2",
                "0",
                "256",
                "false",
                "false",
                "false",
                "0",
                "0",
            ),
        ),
        "opus_gemm_bpreshuffle_pingpong_lds_traits": (
            "pingpong_lds",
            (
                "192",
                "256",
                "128",
                "0",
                "4",
                "0",
                "0",
                "0",
                "false",
                "false",
                "false",
                "0",
                "8",
            ),
        ),
        "opus_gemm_bpreshuffle_wave_interleave_traits": (
            "wave_interleave",
            ("256", "0", "4", "true", "0", "0", "4"),
        ),
    }
    match = re.fullmatch(r"\s*(\w+)\s*<([^<>]*)>\s*", entry["traits"])
    if not match or match[1] not in traits_defaults:
        return None
    pipeline, defaults = traits_defaults[match[1]]
    args = [arg.strip().lower() for arg in match[2].split(",") if arg.strip()]
    if pipeline != entry["pipeline"] or len(args) > len(defaults):
        return None
    args.extend(defaults[len(args) :])
    if any(arg is None or not re.fullmatch(r"-?\d+|true|false", arg) for arg in args):
        return None
    if pipeline == "pingpong_lds":
        # Historical names include three disabled pin switches after FragmentArrays.
        # Current m_repeat traits removed these slots and always use unpinned JIT.
        args[9:9] = ("false", "false", "false")
    return (
        "opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_"
        f"{pipeline}_{instance.B_M}x{instance.B_N}x{instance.B_K}"
        f"_w{instance.T_M}x{instance.T_N}_" + "_".join(args)
    )


def opus_kids_by_name(registry):
    """Map proven complete names to every local ID for that configuration."""
    names = defaultdict(set)
    for kid, instance in registry.items():
        names[instance.name].add(kid)
        alias = _opus_jit_full_name(instance)
        if alias is not None:
            names[alias].add(kid)
    return dict(names)


def resolve_opus_record(kernel_id, kernel_name, names, *, context):
    """Never restore an external numeric ID without matching its full name."""
    kids = names.get(str(kernel_name), ())
    if not kids:
        print(
            f"[OPUS identity] {context}: skipping kernelId={kernel_id!r} "
            f"kernelName={kernel_name!r}; no available local kernel has a proven "
            "equivalent complete name/configuration",
            flush=True,
        )
        return None
    try:
        original_kid = int(str(kernel_id))
    except (TypeError, ValueError):
        original_kid = None
    kid = original_kid if original_kid in kids else min(kids)
    if kid != original_kid:
        print(
            f"[OPUS identity] {context}: mapped kernelId={kernel_id!r} to local "
            f"kernelId={kid} by complete kernelName",
            flush=True,
        )
    return kid


def opus_candidates(
    m,
    n,
    k,
    seeds=(),
    *,
    cu_num,
    top_geometries=0,
    geometries=None,
    pipelines=None,
    routes=None,
    verbose=True,
):
    """Search every legal JIT/CO; >0 shortlists that many geometries per pipeline."""
    from csrc.opus_gemm.opus_gemm_common import (
        candidate_kids_for_shape,
        get_registry,
    )

    if top_geometries is None:  # Compatibility with callers of the former API.
        top_geometries = 0
    if type(top_geometries) is not int or top_geometries < 0:
        raise ValueError(
            "opus_top_geometries must be an integer >= 0 (0 means full search)"
        )
    if pipelines is not None and set(pipelines) - set(PIPELINES):
        raise ValueError(f"unknown OPUS pipelines: {set(pipelines) - set(PIPELINES)}")
    if routes is not None and set(routes) - set(ROUTES):
        raise ValueError(f"unknown OPUS routes: {set(routes) - set(ROUTES)}")
    registry = get_registry("a8w8_blockscale_bpreshuffle", "gfx950", cu_num)
    candidates = [
        (kid, registry[kid])
        for kid in candidate_kids_for_shape("gfx950", m, n, k, cu_num=cu_num)
    ]
    if pipelines is not None:
        candidates = [
            (kid, instance)
            for kid, instance in candidates
            if (instance.bpreshuffle_co or instance.bpreshuffle_entry)["pipeline"]
            in pipelines
        ]
    if routes is not None:
        candidates = [
            (kid, instance)
            for kid, instance in candidates
            if ("co" if instance.bpreshuffle_co else "jit") in routes
        ]
    candidate_count = len(candidates)
    if geometries is not None:
        requested = {_normalize_opus_geometry(geometry) for geometry in geometries}
        if not requested:
            raise ValueError(f"No requested OPUS geometry for shape {(m, n, k)}")
        missing = requested - {_opus_geometry(instance) for _, instance in candidates}
        if missing:
            raise ValueError(
                f"Requested OPUS geometries are unavailable or not legal for shape "
                f"{(m, n, k)}: {sorted(missing)}; check geometry definitions and CO images"
            )
        candidates = [
            (kid, instance)
            for kid, instance in candidates
            if _opus_geometry(instance) in requested
        ]
    if not candidates:
        return []
    seed_kids = set(seeds) & {kid for kid, _ in candidates}
    if top_geometries == 0:
        if verbose:
            print(
                f"[OPUS search] shape={(m, n, k)} legal={candidate_count} "
                f"candidates={len(candidates)} seeds={len(seed_kids)} "
                f"search={'geometry_file' if geometries is not None else 'full'}",
                flush=True,
            )
        return candidates
    groups = defaultdict(list)
    for kid, instance in candidates:
        geometry = _opus_geometry(instance)
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
    retained = {
        geometry
        for geometries in by_pipeline.values()
        for _, geometry in sorted(geometries)[:top_geometries]
    }
    legal = {kid for kid, _ in candidates}
    selected = {kid for geometry in retained for kid in groups[geometry]}
    selected.update(seed_kids & legal)
    if verbose:
        print(
            f"[OPUS search] shape={(m, n, k)} legal={candidate_count} "
            f"geometries={len(retained)} candidates={len(selected)} seeds={len(seed_kids)}",
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
        default=0,
        help="geometries per pipeline; zero keeps all",
    )
    parser.add_argument("--opus_geometry_file", help="shape-specific geometry CSV")
    parser.add_argument("--opus_pipelines", choices=PIPELINES, nargs="+")
    parser.add_argument("--opus_routes", choices=ROUTES, nargs="+")
    args = parser.parse_args()
    geometry_by_shape = (
        load_opus_geometry_file(args.opus_geometry_file)
        if args.opus_geometry_file
        else None
    )
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
        "traits",
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
            shape = (m, n, k)
            if geometry_by_shape is not None and shape not in geometry_by_shape:
                raise ValueError(
                    f"{args.opus_geometry_file}: no requested OPUS geometry for shape {shape}"
                )
            shape_count += 1
            for kid, instance in opus_candidates(
                m,
                n,
                k,
                cu_num=cu,
                top_geometries=args.opus_top_geometries,
                geometries=(
                    geometry_by_shape[shape] if geometry_by_shape is not None else None
                ),
                pipelines=args.opus_pipelines,
                routes=args.opus_routes,
            ):
                entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
                traits = entry["traits"]
                if isinstance(traits, dict):
                    traits = traits["name"] + "<" + ", ".join(traits["args"]) + ">"
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
                        "traits": traits,
                    }
                )
                candidate_count += 1
    print(
        f"[OPUS export] shapes={shape_count} candidates={candidate_count} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
