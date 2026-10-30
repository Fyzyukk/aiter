#!/usr/bin/env python3
"""Prepare a small second CO round against the existing isolated 804-CO repo.

Default/check-only writes local proposals and launchers without extending the
repo. Explicit --apply only replaces the isolated manifest, after backup; no
build or GPU runs here. Original 48212 + first-round 24 identities stay exact.
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import shlex
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from prepare_co_experiment import (LOCAL, ROOT, COMMON_REL, MANIFEST_REL,
                                   FROZEN_BASE_LINE, TRAITS_ROOT, load_common,
                                   registry, fingerprint, transform as first_transform,
                                   pin_wrapper, write_inputs, b_metadata)
from prepare_opus_repeat import DATA, DOC, csv_rows, key, shape_row, number
from integrate_co_winners import (ORIGINAL_DIGEST, dump, digest, registry_metadata,
                                  audit_repeat, resolve_files, atomic_write)

BASE_SPARSE = set(range(100000, 100024))
PAIR_FIELDS = ("tile", "wave_layout", "launch_bounds", "device_flags", "grid_layout",
               "pin_mask", "pad_m", "wide_output", "fixed_k", "max_k")


def device_identity(entry):
    """Canonical device/host contract; scope and provenance do not create work."""
    value = copy.deepcopy(entry)
    for field in ("kid", "configKey", "supported_shapes"):
        value.pop(field, None)
    if value["pipeline"] == "pin_agpr":
        root, args = pin_wrapper(value)
        value["body_args"][0] = root + "<" + ", ".join(args) + ">"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def transform(entry, vary, shapes, seed_kid, allow_equal=False):
    if len(vary) != 1:
        raise ValueError("Every second-round point must change one explicit variable")
    axis, value = next(iter(vary.items()))
    if entry["pipeline"] != "pin_agpr" or axis not in ("GroupM", "Panel", "Unroll"):
        return first_transform(entry, vary, shapes, seed_kid, allow_equal)
    root, wrapper = pin_wrapper(entry)
    traits = entry["traits"]
    if traits["name"] != TRAITS_ROOT + "4wave_pin_agpr_panel_traits" or len(traits["args"]) != 3:
        raise ValueError("Finite 4wave axes require the existing three-parameter panel traits")
    panel, unroll, pad = traits["args"]
    if int(panel) not in (16, 32, 64) or int(unroll) not in (2, 4) or (str(pad).lower() == "true") != bool(entry["pad_m"]):
        raise ValueError("Seed 4wave panel traits are outside the reviewed C++ contract")
    allowed = {"GroupM": (0, 4, 8, 16), "Panel": (16, 32, 64), "Unroll": (2, 4)}
    if type(value) is not int or value not in allowed[axis]:
        raise ValueError(f"{axis} only allows {allowed[axis]}; no source changes are inferred")
    before = {"GroupM": int(wrapper[-1]), "Panel": int(panel), "Unroll": int(unroll),
              "PadM": bool(entry["pad_m"]), "WideOutput": bool(entry.get("wide_output", False)), "runtime_K": True}
    if before[axis] == value:
        raise ValueError(f"Requested {axis} is unchanged")
    new = copy.deepcopy(entry)
    new.pop("kid", None)
    new.pop("configKey", None)
    if axis == "GroupM":
        wrapper[-1] = str(value)
        new["body_args"][0] = root + "<" + ", ".join(wrapper) + ">"
    else:
        new["traits"]["args"][0 if axis == "Panel" else 1] = str(value)
    new["supported_shapes"] = [list(shape) for shape in sorted({tuple(shape) for shape in shapes})]
    after = {**before, axis: value}
    # Unroll is a traits field, not an LDS stage; retain entry.stages unchanged.
    suffix = (f"exp2_nt_seed{seed_kid}_sp{after['Panel']}_u{after['Unroll']}_pm{int(after['PadM'])}"
              f"_wide{int(after['WideOutput'])}_gm{after['GroupM']}_h"
              + hashlib.sha256(json.dumps(new, sort_keys=True).encode()).hexdigest()[:12])
    return new, suffix, {"axis": axis, "before": before, "after": after, "strict_pair_note": None}


def prepare_document(base_doc, plan, max_new=24):
    seeds = {row["kid"]: row for row in base_doc["gfx950"]}
    if set(seeds) != set(range(780)) | BASE_SPARSE or len(seeds) != len(base_doc["gfx950"]):
        raise ValueError("Second round requires exactly original780 + first sparse24 CO entries")
    expanded = [(copy.deepcopy(point), None) for point in plan.get("experiments", [])]
    for index, pair in enumerate(plan.get("paired_experiments", [])):
        b, overlap = seeds[int(pair["b_seed_kid"])], seeds[int(pair["overlap_seed_kid"])]
        if b["pipeline"] != "prefetch_b_vgpr" or overlap["pipeline"] != "prefetch_b_vgpr_overlap" or b["traits"]["args"] != overlap["traits"]["args"][:17]:
            raise ValueError("Second-round pair must preserve all 17 B/overlap base traits")
        if any(b.get(field) != overlap.get(field) for field in PAIR_FIELDS):
            raise ValueError("Second-round B/overlap pair has different host/pin/geometry contracts")
        for seed in (pair["b_seed_kid"], pair["overlap_seed_kid"]):
            expanded.append(({**copy.deepcopy(pair), "seed_kid": seed}, f"round2_pair_{index}"))
    if not 1 <= len(expanded) <= max_new:
        raise ValueError(f"Plan requires 1..{max_new} explicit new COs, found {len(expanded)}")
    if "source_report" not in plan:
        raise ValueError("Second-round plan must identify its completed source_report")
    document, points = copy.deepcopy(base_doc), []
    owners = defaultdict(list)
    for kid, entry in seeds.items():
        owners[device_identity(entry)].append(kid)
    for index, (point, pair) in enumerate(expanded):
        seed = int(point["seed_kid"])
        shapes = point.get("shapes", [])
        if seed not in seeds or not point.get("reason") or not shapes or any(len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape) for shape in shapes):
            raise ValueError("Every point requires an existing seed, measured reason and explicit positive integer M,N,K")
        new, suffix, metadata = transform(seeds[seed], point["vary"], shapes, seed, point.get("allow_equal_b_ahead", False))
        if new["pipeline"] == "pin_agpr" and (int(new.get("fixed_k", 0)) != 0 or
                int(seeds[seed]["traits"]["args"][1]) != 2 or int(new["traits"]["args"][1]) != 2):
            raise ValueError("This bounded second round preserves runtime K and Unroll2; Unroll4 is outside this round's selected scope")
        identity = device_identity(new)
        if identity in owners:
            raise ValueError(f"New device configuration already exists as CO IDs {owners[identity]}; do not spend a CO on only a new scope/name")
        kid = 100024 + index
        owners[identity].append(kid)
        new["kid"] = kid
        document["gfx950"].append(new)
        document["image_suffixes"][str(kid)] = suffix
        points.append({"new_kid": kid, "seed_kid": seed, "pipeline": new["pipeline"],
                       "paired_point": pair, "reason": point["reason"],
                       "shapes": shapes, "round": 2, "gpu_validation_pending": True, **metadata})
    old_extensions = copy.deepcopy(document.get("_experimental_extensions", {}))
    history = document.setdefault("_experimental_extension_history", [])
    history.append({"round": 1, "metadata": old_extensions})
    # Active metadata only identifies this new round; all old entries remain.
    document["_experimental_extensions"] = {"sparse_id_start": 100024, "source_co_count": 804,
                                             "round": 2, "points": points}
    document["jit_kid_base"] = 780
    if document["gfx950"][:804] != base_doc["gfx950"] or any(document["image_suffixes"][str(k)] != base_doc["image_suffixes"][str(k)] for k in seeds):
        raise ValueError("Existing804 entries/suffixes changed")
    return document, points


def check_identity(base, extended, new_ids):
    left, right = registry(base), registry(extended)
    if set(left) != set(range(48212)) | BASE_SPARSE or set(right) != set(left) | set(new_ids):
        raise ValueError("Extension shifted/dropped old IDs or unexpectedly registered new IDs")
    previous, original = hashlib.sha256(), hashlib.sha256()
    for kid, instance in left.items():
        if instance.name != right[kid].name or dataclasses.asdict(instance) != dataclasses.asdict(right[kid]):
            raise ValueError(f"Existing identity changed for kid={kid}")
        line = f"{kid}:{fingerprint(instance)}\n".encode()
        previous.update(line)
        if kid < 48212:
            original.update(line)
    if original.hexdigest() != ORIGINAL_DIGEST or base._bpreshuffle_jit_kid_base != 780 or extended._bpreshuffle_jit_kid_base != 780:
        raise ValueError("Original48212 digest or frozen JIT780 base differs")
    names = [entry.name for entry in right.values()]
    if len(names) != len(set(names)):
        raise ValueError("New complete CO names are not unique")
    return {"previous_ids_verified": len(left), "original_ids_verified": 48212,
            "first_round_sparse_ids_verified": 24, "original_config_digest": original.hexdigest(),
            "previous_config_digest": previous.hexdigest(), "jit_kid_base": 780,
            "new_ids": list(new_ids), "complete_names_unique": True}


def make_repeat_plan(points, common, baseline, report_path):
    metadata = registry_metadata(common)
    candidates = defaultdict(lambda: defaultdict(set))
    for point in points:
        for shape in point["shapes"]:
            full = ("gfx950", 256, *shape, "128x128")
            candidates[full][point["new_kid"]].add("experimental_co")
            candidates[full][point["seed_kid"]].add("exact_experiment_parent")
    items = []
    for shape, kids in sorted(candidates.items()):
        rows = []
        for kid, roles in sorted(kids.items()):
            meta = metadata[kid]
            rows.append({"kid": kid, **meta, "bounded": None, "is_co": True,
                         "bpreshuffle_co": meta["kernelName"], "full_name_aliases": [],
                         "roles": sorted(roles), "exact_opposite_kids": [],
                         "us": None, "errRatio": None, "correct": False})
        items.append({"shape": shape_row(shape), "csv_baseline_us": baseline[shape],
                      "remote_winner": None, "candidates": rows})
    return {"version": 2, "family": "a8w8_blockscale_bpreshuffle", "metadata_module": "prepare_opus_repeat",
            "correctness_limit": .05, "selection": "Every second-round new CO and its exact existing seed; six alternated tuner-timing rounds",
            "source_report": str(report_path), "experimental_points": points, "integrated_points": [], "shapes": items}


def validate_points(base, extended, points, baseline, audited):
    for point in points:
        point["kernelName"] = registry(extended)[point["new_kid"]].name
        point["seed_kernelName"] = registry(base)[point["seed_kid"]].name
        point["seed_evidence"] = []
        probe = copy.deepcopy(registry(extended)[point["new_kid"]])
        probe.bpreshuffle_co.pop("supported_shapes", None)
        for mnk in point["shapes"]:
            shape = ("gfx950", 256, *mnk, "128x128")
            if shape not in baseline or not extended.a8w8_mxscale_bpreshuffle_supports_shape(probe, *mnk):
                raise ValueError(f"New shape outside fixed cohort or illegal without scope bypass: {mnk}")
            if not base.a8w8_mxscale_bpreshuffle_supports_shape(registry(base)[point["seed_kid"]], *mnk):
                raise ValueError(f"Seed {point['seed_kid']} is not legal for {mnk}; never widen its old supported_shapes")
            stat = audited.get((shape, point["seed_kid"]))
            point["seed_evidence"].append({"shape": list(mnk), "six_round_correct_in_source": bool(stat and stat["valid"]),
                                           "source_median_us": stat["median_us"] if stat else None,
                                           "gpu_validation_pending_for_new": True})


def write_launchers(out, repo, new_ids, previous_build_report):
    prefix = (f"#!/usr/bin/env bash\nset -euo pipefail\nsource {shlex.quote(str(out / 'experiment_env.sh'))}\n")
    env = (f"#!/usr/bin/env bash\nsource {shlex.quote(str(LOCAL / 'env.sh'))}\n"
           f"export OPUS_RUN_REPO={shlex.quote(str(repo))}\nexport OPUS_RUN_DIR={shlex.quote(str(out))}\n"
           'export PYTHONPATH="$OPUS_RUN_REPO"\nexport AITER_JIT_DIR="$OPUS_RUN_DIR/.cache/aiter_jit"\n'
           'export FLYDSL_RUNTIME_CACHE_DIR="$OPUS_RUN_DIR/.cache/flydsl"\nexport TRITON_CACHE_DIR="$OPUS_RUN_DIR/.cache/triton"\n'
           'unset OPUS_GEN_CO_DIR\ncd "$OPUS_RUN_REPO" || return 1\n')
    (out / "experiment_env.sh").write_text(env)
    build = (prefix + '"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" '
             + "--kids " + " ".join(map(str, new_ids)) + ' --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" '
             '--report "$OPUS_RUN_DIR/new_co_build_info.json" --jobs 8\n'
             + f'"$OPUS_RUN_PYTHON" {shlex.quote(str(Path(__file__).resolve()))} --merge-build-reports '
             + shlex.quote(str(previous_build_report)) + ' "$OPUS_RUN_DIR/new_co_build_info.json" '
             '--merged-report "$OPUS_RUN_DIR/merged_co_build_info.json"\n')
    (out / "build_new_co.sh").write_text(build)
    tune = (prefix + 'export AITER_REBUILD=1\nexport OPUS_TUNE_PROGRESS_FILE="$OPUS_RUN_DIR/progress.jsonl"\n'
            'export OPUS_TUNE_CANDIDATE_PROGRESS_DIR="$OPUS_RUN_DIR/candidates"\n'
            '"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py '
            '--mxscale_table --libtype opus --mp 4 --shape_grouped --opus_top_geometries 0 '
            '--opus_geometry_file "$OPUS_RUN_DIR/experiment_geometries.csv" -i "$OPUS_RUN_DIR/experiment_shapes.csv" '
            '-o "$OPUS_RUN_DIR/tuned.csv" -o2 "$OPUS_RUN_DIR/profile.csv"\n')
    (out / "run_tune.sh").write_text(tune)
    repeat = (prefix + '"$OPUS_RUN_PYTHON" "$OPUS_RUN_REPO/docs/gfx950_mxscale_20261029/local_20261029/run_repeat_shards.py" '
              '--plan "$OPUS_RUN_DIR/helper_repeat_plan.json" --out-dir "$OPUS_RUN_DIR/repeats"\n')
    # Check runner flags before calling: use the existing root-managed shard
    # launcher if its interface changes, keeping helper_repeat_plan.json.
    (out / "run_helper_repeat.sh").write_text(repeat)
    for name in ("experiment_env.sh", "build_new_co.sh", "run_tune.sh", "run_helper_repeat.sh"):
        (out / name).chmod(0o755)


def merge_reports(paths, output):
    kernels, sources = {}, []
    for path in paths:
        path = path.resolve()
        doc = json.loads(path.read_text())
        sources.append({"path": str(path), "sha256": digest(path), "llvm_bin": doc.get("llvm_bin")})
        for row in doc["kernels"]:
            kid = int(row["kid"])
            if kid in kernels:
                raise ValueError(f"Build history repeats kid={kid}; retain rounds separately rather than silently replacing evidence")
            kernels[kid] = row
    if output.exists():
        raise ValueError("Choose a fresh merged build report path")
    dump(output, {"llvm_bin": sources[-1]["llvm_bin"], "source_reports": sources,
                  "scope": "Sparse experimental COs only; original780 report remains unchanged",
                  "kernels": [kernels[kid] for kid in sorted(kernels)]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment-dir", type=Path, default=LOCAL / "co_round1")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-new-co", type=int, default=24)
    parser.add_argument("--merge-build-reports", nargs="+", type=Path)
    parser.add_argument("--merged-report", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Default: proposals only, do not extend clone")
    mode.add_argument("--apply", action="store_true", help="Explicitly append only to existing isolated clone, with manifest backup")
    args = parser.parse_args()
    if args.merge_build_reports:
        if not args.merged_report:
            parser.error("Merge mode needs --merged-report")
        merge_reports(args.merge_build_reports, args.merged_report)
        return
    if not args.plan or not args.out_dir:
        parser.error("An actual selected --plan and fresh local --out-dir are required")
    base_dir, out = args.base_experiment_dir.resolve(), args.out_dir.resolve()
    if not base_dir.is_relative_to(LOCAL) or not out.is_relative_to(LOCAL) or out == LOCAL or out.exists():
        parser.error("Base and fresh output must stay below local_20261029")
    repo = (base_dir / ".cache/experiment_repo").resolve()
    if not repo.is_relative_to(LOCAL) or repo == ROOT:
        parser.error("Only the existing isolated experiment repo may be extended")
    common_source = (repo / COMMON_REL).read_text()
    if common_source.count(FROZEN_BASE_LINE) != 1:
        parser.error("Existing isolated common must already freeze JIT base780 exactly")
    plan = json.loads(args.plan.read_text())
    report_path = Path(plan["source_report"]).resolve()
    report = json.loads(report_path.read_text())
    if report["summary"].get("rounds") != 6 or report["summary"].get("pending_shapes") != 0:
        parser.error("Second-round source evidence must complete six rounds")
    base_doc = json.loads((repo / MANIFEST_REL).read_text())
    base = load_common(repo, label="extension_base")
    full_rows = csv_rows(DOC / "shapes_421.csv")
    full_shapes = {key(row) for row in full_rows}
    if len(full_shapes) != 421 or len(full_rows) != 421:
        raise ValueError("Expected the unique complete421 NT cohort")
    baseline = {}
    for row in csv_rows(DATA / "csv_baseline.csv"):
        shape = key(row)
        if shape not in full_shapes:
            continue
        us = number(row.get("us"))
        if shape in baseline or us is None or us <= 0:
            raise ValueError("Ambiguous or invalid cohort baseline")
        baseline[shape] = us
    if set(baseline) != full_shapes:
        raise ValueError("Fixed baseline lacks an exact match for all421 cohort shapes")
    audited = audit_repeat(report, resolve_files(report["summary"]["files"]), baseline, registry_metadata(base), .05)
    document, points = prepare_document(base_doc, plan, args.max_new_co)
    ids = [point["new_kid"] for point in points]
    with tempfile.TemporaryDirectory(prefix="co_extension_check_", dir=LOCAL) as temp:
        manifest = Path(temp) / MANIFEST_REL.name
        dump(manifest, document)
        extended = load_common(repo, manifest=manifest, label="extension_proposed")
        identity = check_identity(base, extended, ids)
        validate_points(base, extended, points, baseline, audited)
        repeat_plan = make_repeat_plan(points, extended, baseline, report_path)
    base_build_report = base_dir / "new_co_build_info.json"
    build = json.loads(base_build_report.read_text())
    if {int(row["kid"]) for row in build["kernels"]} != BASE_SPARSE or any(row.get("status") != "ready" for row in build["kernels"]):
        raise ValueError("Base first24 build report must retain all ready sparse IDs")
    for kid, entry in base.gfx950_co_kernels_declared.items():
        if not (repo / MANIFEST_REL.parent / "gfx950" / entry["image"]).is_file():
            raise ValueError(f"Existing CO image missing: kid={kid}")
    manifest_hash = digest(repo / MANIFEST_REL)
    receipt = {"version": 1, "mode": "check_only" if not args.apply else "apply_isolated_manifest",
               "status": "checked", "base_repo": str(repo), "base_experiment_dir": str(base_dir),
               "base_manifest_sha256": manifest_hash, "base_common_sha256": digest(repo / COMMON_REL),
               "source_report": str(report_path), "source_report_sha256": digest(report_path),
               "source_build_report": str(base_build_report), "source_build_report_sha256": digest(base_build_report),
               "identity": identity, "points": points, "new_gpu_validation_pending": True,
               "summary": {"new_co": len(ids), "shapes": len(repeat_plan["shapes"]),
                           "new_and_seed_candidate_pairs": sum(len(item["candidates"]) for item in repeat_plan["shapes"]),
                           "source_shapes": report["summary"]["repeat_complete_shapes"],
                           "source_csv_wins": report["summary"]["repeat_wins_csv_median"]},
               "compiled_by_tool": False, "gpu_used_by_tool": False,
               "active_metadata_scope": "Only round2 points active; round1 metadata retained in history, all804 old entries unchanged"}
    out.mkdir(parents=True)
    dump(out / "receipt.json", receipt)
    dump(out / "selection_plan.json", plan)
    dump(out / "base_manifest.json", base_doc)
    dump(out / "proposed_manifest.json", document)
    dump(out / "helper_repeat_plan.json", repeat_plan)
    write_inputs(out, extended.gfx950_co_kernels_declared, points)
    write_launchers(out, repo, ids, base_build_report)
    if args.apply:
        if digest(repo / MANIFEST_REL) != manifest_hash or digest(repo / COMMON_REL) != receipt["base_common_sha256"]:
            raise ValueError("Isolated source changed after CPU check")
        backup = out / "backups/co_kernels_gfx950.json"
        backup.parent.mkdir()
        shutil.copy2(repo / MANIFEST_REL, backup)
        receipt["manifest_backup"] = str(backup)
        try:
            atomic_write(repo / MANIFEST_REL, (json.dumps(document, indent=2) + "\n").encode())
            applied = load_common(repo, label="extension_applied")
            check_identity(base, applied, ids)
        except Exception:
            atomic_write(repo / MANIFEST_REL, backup.read_bytes())
            receipt["status"] = "rolled_back"
            dump(out / "receipt.json", receipt)
            raise
        receipt["status"] = "extended_isolated_manifest"
        receipt["manifest_after_sha256"] = digest(repo / MANIFEST_REL)
        dump(out / "receipt.json", receipt)
    print(json.dumps({"out_dir": str(out), "status": receipt["status"], "identity": identity,
                      "summary": receipt["summary"], "compiled_by_tool": False, "gpu_used_by_tool": False}, indent=2))


if __name__ == "__main__":
    main()
