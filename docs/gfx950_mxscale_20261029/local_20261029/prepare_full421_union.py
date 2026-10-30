#!/usr/bin/env python3
"""Prepare the exact Top8 + directed Top0 kernel union on CPU.

Default operation never imports torch/aiter or compiles anything. After stage3
finishes, root may explicitly pass --compile to invoke the actual tuner's
ensure_opus_kids_compiled(union, cu_num=256) once before full and fallback runs.
--from-json reuses a prepared union only if source/input/registry identities
still match. Newly integrated sparse CO IDs are read dynamically.
"""

import argparse
import ast
import contextlib
import csv
import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
DOC = ROOT / "docs/gfx950_mxscale_20261029"
NAMESPACE = {"family": "a8w8_blockscale_bpreshuffle", "gfx": "gfx950", "cu_num": 256}
SHAPE_FIELDS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")
PURE_FUNCTIONS = {"_opus_geometry", "_normalize_opus_geometry", "load_opus_geometry_file",
                  "opus_candidates", "_opus_jit_full_name", "opus_kids_by_name"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_digest(registry, kids=None):
    digest = hashlib.sha256()
    for kid in sorted(registry if kids is None else kids):
        digest.update(str(kid).encode() + b":")
        digest.update(json.dumps(dataclasses.asdict(registry[kid]), sort_keys=True,
                                 separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def source_files(repo):
    return [repo / "csrc/opus_gemm/opus_gemm_common.py",
            repo / "csrc/opus_gemm/gen_co/co_kernels_gfx950.json",
            repo / "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py"]


def pure_tuner(repo):
    sys.path.insert(0, str(repo))
    common = importlib.import_module("csrc.opus_gemm.opus_gemm_common")
    path = source_files(repo)[2]
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in PURE_FUNCTIONS]
    if {node.name for node in nodes} != PURE_FUNCTIONS:
        raise ValueError("The tuner pure function set changed; inspect the actual source")
    scope = dict(math=math, defaultdict=defaultdict, ast=ast, csv=csv, json=json, re=re)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return common, scope


def shape_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    shapes = []
    for line, row in enumerate(rows, 2):
        if any(field not in row for field in SHAPE_FIELDS):
            raise ValueError(f"Missing complete shape key at {path}:{line}")
        if str(row["gfx"]) != "gfx950" or int(row["cu_num"]) != 256 or row["w_scale_block"] != "128x128":
            raise ValueError(f"Unexpected namespace/scale key at {path}:{line}")
        shape = tuple(int(row[field]) for field in ("M", "N", "K"))
        if min(shape) <= 0:
            raise ValueError(f"Invalid M/N/K at {path}:{line}")
        shapes.append(shape)
    if len(shapes) != len(set(shapes)):
        raise ValueError(f"Duplicate shape keys in {path}")
    return shapes


def read_seeds(paths, registry, names, shapes, common):
    seed_kids, skipped = defaultdict(set), []
    for path in paths:
        if not path.exists():
            skipped.append({"path": str(path), "reason": "optional seed CSV absent"})
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                if row.get("libtype") != "opus":
                    continue
                try:
                    if (row.get("gfx", "gfx950") != "gfx950" or int(row.get("cu_num", 256)) != 256
                            or row.get("w_scale_block", "128x128") != "128x128"):
                        continue
                    shape = tuple(int(row[field]) for field in ("M", "N", "K"))
                    us, err = float(row["us"]), float(row["errRatio"])
                    if shape not in shapes or not math.isfinite(us) or us <= 0 or not math.isfinite(err) or not 0 <= err <= .05:
                        continue
                except (TypeError, ValueError, KeyError):
                    continue  # Same validity gate as the actual tuner.
                matched = names.get(row.get("kernelName"), set())
                if not matched:
                    skipped.append({"path": str(path), "line": line, "kernelName": row.get("kernelName"),
                                    "reason": "No proven complete current name/traits mapping"})
                seed_kids[shape].update(kid for kid in matched
                                        if common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], *shape))
    return seed_kids, skipped


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def prepare(args):
    repo = args.repo.resolve()
    common, tuner = pure_tuner(repo)
    registry = common.get_registry(**NAMESPACE)
    if not registry:
        raise ValueError("No available gfx950/256 bpreshuffle kernels")
    sources = {str(path): file_hash(path) for path in source_files(repo)}
    full_shapes = shape_rows(args.shapes)
    fallback_shapes = shape_rows(args.fallback_shapes)
    if set(fallback_shapes) - set(full_shapes):
        raise ValueError("Fallback shape list extends outside full421")
    geometries = tuner["load_opus_geometry_file"](args.fallback_geometries)
    if set(geometries) != set(fallback_shapes):
        raise ValueError(f"Fallback shapes/geometries differ: missing={len(set(fallback_shapes)-set(geometries))}, extra={len(set(geometries)-set(fallback_shapes))}")
    directed_shapes, directed_geometries = [], {}
    if args.directed_shapes:
        directed_shapes = shape_rows(args.directed_shapes)
        directed_geometries = tuner["load_opus_geometry_file"](args.directed_geometries)
        if set(directed_shapes) - set(full_shapes):
            raise ValueError("Directed subset extends outside full421")
        if set(directed_geometries) != set(directed_shapes):
            raise ValueError("Directed shapes/geometries must contain exactly the same M/N/K keys")
    default_seed = Path(os.environ.get("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
                                      str(repo / "aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv")))
    seed_paths = list(dict.fromkeys([default_seed.resolve()] + [path.resolve() for path in args.seed_csv]))
    seed_kids, seed_skipped = read_seeds(seed_paths, registry, tuner["opus_kids_by_name"](registry),
                                       set(full_shapes), common)
    inputs = {str(path.resolve()): file_hash(path) for path in
              (args.shapes, args.fallback_shapes, args.fallback_geometries)}
    inputs.update({str(path): file_hash(path) for path in seed_paths if path.exists()})
    if args.directed_shapes:
        inputs.update({str(path.resolve()): file_hash(path) for path in (args.directed_shapes, args.directed_geometries)})
    full, fallback, directed, seeds, per_shape = set(), set(), set(), set(), []
    for source, shapes in (("full_top8", full_shapes), ("winner_fallback_top0", fallback_shapes), ("directed_top0", directed_shapes)):
        for shape in shapes:
            with contextlib.redirect_stdout(io.StringIO()):
                candidates = tuner["opus_candidates"](
                    *shape, seeds=seed_kids.get(shape, ()), cu_num=256,
                    top_geometries=8 if source == "full_top8" else 0,
                    geometries=(None if source == "full_top8" else
                                geometries[shape] if source == "winner_fallback_top0" else directed_geometries[shape]))
            kids = {kid for kid, _ in candidates}
            if not kids:
                raise ValueError(f"No legal candidates for {source} shape {shape}")
            (full if source == "full_top8" else fallback if source == "winner_fallback_top0" else directed).update(kids)
            seeds.update(seed_kids.get(shape, ()))
            by_pipeline = Counter(tuner["_opus_geometry"](instance)[0] for _, instance in candidates)
            co_count = sum(instance.bpreshuffle_co is not None for _, instance in candidates)
            per_shape.append({"source": source, "M": shape[0], "N": shape[1], "K": shape[2],
                              "candidate_count": len(kids), "co_count": co_count,
                              "jit_count": len(kids)-co_count,
                              "geometry_count": len({tuner["_opus_geometry"](instance) for _, instance in candidates}),
                              "by_pipeline": dict(by_pipeline), "seed_kids": sorted(set(seed_kids.get(shape, ())) & kids)})
    if sources != {str(path): file_hash(path) for path in source_files(repo)}:
        raise ValueError("Core/manifest/tuner source changed while preparing the union; rerun against the finished source")
    if inputs != {path: file_hash(path) for path in inputs}:
        raise ValueError("Shape/geometry/seed inputs changed during preparation; rerun against completed inputs")
    union = full | fallback | directed | seeds
    co_union = [kid for kid in sorted(union) if registry[kid].bpreshuffle_co is not None]
    document = {"version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "repo": str(repo), "namespace": NAMESPACE, "top_geometries": 8,
                "full_shape_count": len(full_shapes), "fallback_shape_count": len(fallback_shapes),
                "directed_shape_count": len(directed_shapes),
                "full_kids": sorted(full), "fallback_kids": sorted(fallback), "seed_kids": sorted(seeds),
                "directed_kids": sorted(directed),
                "union_kids": sorted(union), "union_co_kids": co_union,
                "registry_available_count": len(registry),
                "registry_available_co_count": sum(instance.bpreshuffle_co is not None for instance in registry.values()),
                "registry_identity_sha256": identity_digest(registry),
                "union_identity_sha256": identity_digest(registry, union),
                "source_sha256": sources, "input_sha256": inputs,
                "seed_inputs": list(map(str, seed_paths)), "seed_skipped": seed_skipped,
                "summary": {"full_unique_kids": len(full), "fallback_unique_kids": len(fallback),
                            "additional_fallback_kids": len(fallback-full), "seed_unique_kids": len(seeds),
                            "directed_unique_kids": len(directed),
                            "additional_directed_kids": len(directed-(full|fallback|seeds)),
                            "union_unique_kids": len(union), "union_co_count": len(co_union),
                            "union_jit_count": len(union)-len(co_union),
                            "full_total_candidates": sum(row["candidate_count"] for row in per_shape if row["source"]=="full_top8"),
                            "fallback_total_candidates": sum(row["candidate_count"] for row in per_shape if row["source"]=="winner_fallback_top0"),
                            "directed_total_candidates": sum(row["candidate_count"] for row in per_shape if row["source"]=="directed_top0")},
                "prepared_on_cpu": True, "compiled": False,
                "limitations": ["New COs must be built before prepare/compile so they appear in available get_registry.",
                                "Additional tune-file seeds must be supplied with --seed-csv; full candidates use the actual Top8+seed logic.",
                                "Any source/input/available-registry change invalidates --from-json and requires a fresh preparation."]}
    output = args.output_dir.resolve()
    atomic_json(output / "union.json", document)
    atomic_json(output / "per_shape_receipt.json", {"namespace": NAMESPACE, "shapes": per_shape})
    atomic_json(output / "receipt.json", {key: value for key, value in document.items()
                                          if key not in ("full_kids", "fallback_kids", "directed_kids", "seed_kids", "union_kids", "union_co_kids")})
    if "torch" in sys.modules or "aiter" in sys.modules:
        raise ValueError("CPU preparation unexpectedly imported torch or aiter")
    return document


def compile_union(document):
    if document.get("namespace") != NAMESPACE:
        raise ValueError("Prepared union namespace differs from gfx950/256 bpreshuffle")
    repo = Path(document["repo"])
    common, _ = pure_tuner(repo)
    registry = common.get_registry(**NAMESPACE)
    union = set(map(int, document["union_kids"]))
    if not union or union - registry.keys():
        raise ValueError("Prepared union is empty or now contains unavailable kernels; prepare again after building COs")
    for label in ("source_sha256", "input_sha256"):
        if document[label] != {path: file_hash(path) for path in document[label]}:
            raise ValueError(f"Prepared {label} no longer matches; rerun CPU preparation")
    if identity_digest(registry) != document["registry_identity_sha256"] or identity_digest(registry, union) != document["union_identity_sha256"]:
        raise ValueError("Available registry identities changed; regenerate the union")
    for variable in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ.setdefault(variable, "0,1,2,3")
        if os.environ[variable] != "0,1,2,3":
            raise ValueError(f"{variable} must be 0,1,2,3 for this workflow")
    # Extract this one actual function so compilation does not instantiate all
    # FlyDSL pipelines or run tuner preprocessing. This is the explicit build
    # path; its imports are intentionally deferred until --compile is passed.
    path = source_files(repo)[2]
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "ensure_opus_kids_compiled"]
    if len(nodes) != 1:
        raise ValueError("Cannot locate actual ensure_opus_kids_compiled")
    scope = {"Path": Path, "sys": sys, "__file__": str(path)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    rebuilt = scope["ensure_opus_kids_compiled"](union, cu_num=256)
    return {"compiled_utc": datetime.now(timezone.utc).isoformat(), "namespace": NAMESPACE,
            "union_unique_kids": len(union), "rebuilt": bool(rebuilt),
            "union_identity_sha256": document["union_identity_sha256"],
            "source_sha256": document["source_sha256"], "compiled": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--shapes", type=Path, default=DOC / "shapes_421.csv")
    parser.add_argument("--fallback-shapes", type=Path, default=LOCAL / "top8_current_audit/winner_fallback_shapes.csv")
    parser.add_argument("--fallback-geometries", type=Path, default=LOCAL / "top8_current_audit/winner_fallback_geometries.csv")
    parser.add_argument("--directed-shapes", type=Path, help="Optional directed Top0 subset of the full shape list; pair with --directed-geometries")
    parser.add_argument("--directed-geometries", type=Path, help="Requested geometries for --directed-shapes; exact M/N/K coverage required")
    parser.add_argument("--seed-csv", type=Path, nargs="*", default=[], help="Additional exact-name tune-file seeds the full tuner will read")
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "full421_union")
    parser.add_argument("--from-json", type=Path, help="With --compile, reuse a prepared union after strict identity revalidation")
    parser.add_argument("--compile", action="store_true", help="Explicitly compile the complete union once; root should use after stage3 finishes")
    args = parser.parse_args()
    if args.from_json and not args.compile:
        parser.error("--from-json requires --compile; default preparation always reads the current registry")
    if bool(args.directed_shapes) != bool(args.directed_geometries):
        parser.error("--directed-shapes and --directed-geometries must be supplied together")
    try:
        document = json.loads(args.from_json.read_text()) if args.from_json else prepare(args)
        if args.compile:
            receipt = compile_union(document)
            output = args.from_json.parent if args.from_json else args.output_dir.resolve()
            atomic_json(output / "compile_receipt.json", receipt)
        print(json.dumps({"output_dir": str(args.from_json.parent if args.from_json else args.output_dir.resolve()),
                          "full_shape_count": document["full_shape_count"],
                          "fallback_shape_count": document["fallback_shape_count"], **document["summary"],
                          "directed_shape_count": document.get("directed_shape_count", 0),
                          "compiled": bool(args.compile)}, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
