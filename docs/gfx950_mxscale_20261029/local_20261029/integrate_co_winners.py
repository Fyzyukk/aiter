#!/usr/bin/env python3
"""Audit six-round new CO improvements and optionally integrate selected images.

Default/check-only writes a proposed manifest, common source, and receipt under
local_20261029. It never builds or uses a GPU. --apply is explicit, preserves
all 780 original COs and 0..48211 identities, and backs up both public files.
"""

import argparse
import copy
import dataclasses
import glob
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from prepare_co_experiment import (LOCAL, ROOT, COMMON_REL, MANIFEST_REL,
                                   JIT_BASE_LINE, FROZEN_BASE_LINE, load_common,
                                   registry, prepare_manifest, check_identity)
from prepare_opus_repeat import DATA, DOC, KEYS, csv_rows, key, shape_row
from summarize_step1_repeat import boolean, number, jsonl_rows, write_csv

ROUNDS = set(range(6))
ORIGINAL_DIGEST = "37ea3a12fbdf5c797f9c62af0e9d0caf7f008ff58d39a0606107741c2a88d206"
TRAIT_FIELDS = ("traits", "body_func", "body_args", "pad_m", "wide_output",
                "fixed_k", "grid_layout", "launch_bounds", "device_flags")
SOURCE_SUFFIXES = {".py", ".cuh", ".h", ".hpp", ".cc", ".cpp", ".hip", ".cu"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def equal_number(left, right):
    if left is None or right is None:
        return left is None and right is None
    left, right = number(left), number(right)
    return left is not None and right is not None and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def registry_metadata(common):
    metadata = {}
    for kid, instance in registry(common).items():
        entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
        co = instance.bpreshuffle_co is not None
        traits = json.dumps({name: entry.get(name) for name in TRAIT_FIELDS},
                            sort_keys=True, separators=(",", ":")) if co else entry["traits"]
        metadata[kid] = {"kernelName": instance.name, "traits": traits,
                         "kind": "CO" if co else "JIT", "pipeline": entry["pipeline"],
                         "geometry": [entry["pipeline"], instance.B_M, instance.B_N,
                                      instance.B_K, instance.T_M, instance.T_N]}
    return metadata


def audit_candidate(row, records, baseline, err_ratio=.05):
    """Recompute report statistics from raw records, retaining rejection reasons."""
    complete = set(records) == ROUNDS
    valid = [event for event in records.values() if boolean(event.get("correct"))
             and number(event.get("us")) is not None and float(event["us"]) > 0
             and number(event.get("errRatio")) is not None
             and 0 <= float(event["errRatio"]) <= err_ratio and not event.get("failure")]
    times = [float(event["us"]) for event in valid]
    correct = complete and len(valid) == 6
    timing = complete and all(event.get("rotate_args") == 0 for event in records.values())
    expected = {"rounds_written": len(records), "rounds_complete": complete,
                "correct_rounds": len(valid), "all_rounds_correct": correct,
                "timing_matches_tuner": timing,
                "median_us": statistics.median(times) if times else None,
                "min_us": min(times) if times else None,
                "max_us": max(times) if times else None,
                "pstdev_us": statistics.pstdev(times) if times else None,
                "max_errRatio": max((number(e.get("errRatio")) or 0 for e in records.values()), default=None),
                "csv_baseline_us": baseline}
    for field, value in expected.items():
        if field not in row:
            raise ValueError(f"Candidate report lacks {field}: kid={row['kid']}")
        same = equal_number(row[field], value) if field.endswith("_us") or field in ("max_errRatio",) else row[field] == value
        if not same:
            raise ValueError(f"Candidate report disagrees with raw {field}: kid={row['kid']}, report={row[field]!r}, raw={value!r}")
    # pstdev_us also ends in _us and is checked above. Failure text never grants
    # correctness even if an upstream report happened to set correct=true.
    win_median = bool(correct and timing and expected["median_us"] < baseline)
    win_every = bool(correct and timing and expected["max_us"] < baseline)
    for field, value in (("wins_csv_median", win_median), ("wins_csv_every_round", win_every)):
        if row.get(field) is not value:
            raise ValueError(f"Candidate report disagrees with raw {field}: kid={row['kid']}")
    reasons = []
    if not complete:
        reasons.append("incomplete_six_rounds")
    if not correct:
        reasons.append("incorrect_or_failed_round")
    if not timing:
        reasons.append("timing_does_not_match_tuner")
    return {**expected, "valid": not reasons, "reasons": reasons,
            "wins_csv_median": win_median, "records": records}


def compare_pair(new, parent):
    reasons = ["new_" + reason for reason in new["reasons"]]
    reasons.extend("parent_" + reason for reason in parent["reasons"])
    comparisons = []
    if not reasons:
        for index in sorted(ROUNDS):
            a, b = new["records"][index], parent["records"][index]
            if a.get("gpu_logical_id") != b.get("gpu_logical_id"):
                reasons.append("paired_round_gpu_mismatch")
            if type(a.get("position")) is not int or type(b.get("position")) is not int or a["position"] == b["position"]:
                reasons.append("paired_round_positions_missing_or_equal")
            comparisons.append({"round": index, "new_us": float(a["us"]),
                                "parent_us": float(b["us"]), "new_faster": float(a["us"]) < float(b["us"])})
        if "paired_round_positions_missing_or_equal" not in reasons:
            for index in (0, 2, 4):
                first = new["records"][index]["position"] < parent["records"][index]["position"]
                second = new["records"][index + 1]["position"] < parent["records"][index + 1]["position"]
                if first == second:
                    reasons.append("paired_round_order_not_alternated")
        if not new["median_us"] < parent["median_us"]:
            reasons.append("new_median_not_lower")
        if sum(item["new_faster"] for item in comparisons) < 4:
            reasons.append("fewer_than_four_paired_round_wins")
    return {"stable_improvement": not reasons, "reasons": sorted(set(reasons)),
            "paired_round_wins": sum(item["new_faster"] for item in comparisons),
            "new_median_us": new["median_us"], "parent_median_us": parent["median_us"],
            "parent_speedup": parent["median_us"] / new["median_us"] if new["valid"] and parent["valid"] else None,
            "wins_csv_median": new["wins_csv_median"], "rounds": comparisons}


def audit_repeat(report, paths, baseline, metadata, err_ratio):
    summary = report["summary"]
    if summary.get("rounds") != 6 or summary.get("pending_shapes") != 0:
        raise ValueError("Integration requires a completed six-round report")
    rows = {}
    name_ids = {}
    for row in report["candidates"]:
        if any(field not in row for field in KEYS):
            raise ValueError("Candidate report lacks explicit six-field shape identity")
        shape, kid = key(row), int(row["kid"])
        if shape[0] != "gfx950" or shape[1] != 256 or shape[5] != "128x128" or shape not in baseline:
            raise ValueError(f"Unexpected shape/baseline: {shape}")
        if (shape, kid) in rows:
            raise ValueError(f"Duplicate reported shape/kid: {shape}, {kid}")
        if kid not in metadata or any(row.get(field) != value for field, value in metadata[kid].items()):
            raise ValueError(f"Reported name/traits/geometry differ from experiment registry: {shape}, kid={kid}")
        owner = name_ids.setdefault((shape, row["kernelName"]), kid)
        if owner != kid:
            raise ValueError("Same shape and complete name has conflicting IDs")
        rows[(shape, kid)] = row
    events = defaultdict(dict)
    shape_files, shape_gpus = defaultdict(set), defaultdict(set)
    positions = defaultdict(set)
    shape_counts = Counter(shape for shape, _ in rows)
    for path in paths:
        for event, location in jsonl_rows(path):
            if any(field not in event.get("shape", {}) for field in KEYS):
                raise ValueError(f"Raw record lacks explicit six-field shape identity at {location}")
            if type(event.get("round")) is not int or type(event.get("kid")) is not int:
                raise ValueError(f"Raw kid/round must be explicit integers at {location}")
            shape, kid, index = key(event["shape"]), int(event["kid"]), int(event["round"])
            if (shape, kid) not in rows or index not in ROUNDS or index in events[(shape, kid)]:
                raise ValueError(f"Unreported/duplicate/unexpected shape/kid/round at {location}")
            row = rows[(shape, kid)]
            if event.get("kernelName") != row["kernelName"] or event.get("kind", row["kind"]) != row["kind"]:
                raise ValueError(f"Raw candidate identity disagrees at {location}")
            if not equal_number(event.get("csv_baseline_us"), baseline[shape]):
                raise ValueError(f"Raw record differs from fixed CSV baseline at {location}")
            if type(event.get("gpu_logical_id")) is not int or event["gpu_logical_id"] not in range(4):
                raise ValueError(f"Raw record is missing a logical GPU 0..3 identity at {location}")
            position = event.get("position")
            if type(position) is not int or position not in range(shape_counts[shape]) or position in positions[(shape, index)]:
                raise ValueError(f"Raw round has missing/duplicate/invalid candidate position at {location}")
            positions[(shape, index)].add(position)
            events[(shape, kid)][index] = event
            shape_files[shape].add(str(path))
            shape_gpus[shape].add(event["gpu_logical_id"])
    if any(len(shape_files[shape]) != 1 or len(shape_gpus[shape]) != 1 for shape, _ in rows):
        raise ValueError("A measured shape is missing or split across files/GPU/batches")
    if set(events) != set(rows):
        raise ValueError("Raw candidate coverage differs from report")
    if any(positions[(shape, index)] != set(range(count)) for shape, count in shape_counts.items() for index in ROUNDS):
        raise ValueError("Raw per-round candidate positions are incomplete")
    return {identity: audit_candidate(row, events[identity], baseline[identity[0]], err_ratio)
            for identity, row in rows.items()}


def resolve_files(patterns):
    paths = set()
    for pattern in patterns:
        found = glob.glob(str(pattern))
        if not found:
            raise ValueError(f"No raw repeat file matches {pattern}")
        paths.update(Path(path).resolve() for path in found)
    if not paths:
        raise ValueError("Report must provide raw summary.files or --repeat-jsonl")
    return sorted(paths)


def source_parity(experiment_repo):
    records = []
    for relative in (Path("csrc/opus_gemm"), Path("csrc/include")):
        left = {path.relative_to(ROOT) for path in (ROOT / relative).rglob("*") if path.is_file() and path.suffix in SOURCE_SUFFIXES}
        right = {path.relative_to(experiment_repo) for path in (experiment_repo / relative).rglob("*") if path.is_file() and path.suffix in SOURCE_SUFFIXES}
        if left != right:
            raise ValueError(f"Experiment/public source file set differs under {relative}")
        for rel in sorted(left):
            a, b = ROOT / rel, experiment_repo / rel
            if rel == COMMON_REL:
                text = a.read_text()
                if text.count(JIT_BASE_LINE) != 1 or b.read_text() != text.replace(JIT_BASE_LINE, FROZEN_BASE_LINE):
                    raise ValueError("Experimental common source has changes beyond frozen JIT base")
            elif a.read_bytes() != b.read_bytes():
                raise ValueError(f"Experiment/public code differs: {rel}")
            records.append({"path": str(rel), "public_sha256": digest(a), "experiment_sha256": digest(b)})
    return records


def build_image(kid, entry, build_rows, experiment_repo):
    row = build_rows.get(kid)
    if not row or row.get("status") != "ready" or row.get("image") != entry["image"] or row.get("symbol") != entry["symbol"]:
        raise ValueError(f"Selected kid={kid} lacks matching ready build evidence")
    source = experiment_repo / MANIFEST_REL.parent / "gfx950" / entry["image"]
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Selected image missing or symlinked: {source}")
    with source.open("rb") as stream:
        header = stream.read(64)
    machine = int.from_bytes(header[18:20], "little")
    processor = int.from_bytes(header[48:52], "little") & 255
    if len(header) < 64 or header[:6] != b"\x7fELF\x02\x01" or machine != 224 or processor != 0x4F:
        raise ValueError(f"Selected image is not an AMDGPU gfx950 ELF64: {source}")
    for field, value in (("offload_arch", "gfx950"), ("elf_machine", 224), ("elf_processor_id", 0x4F), ("undefined_symbols", [])):
        if row.get(field) != value:
            raise ValueError(f"Selected kid={kid} lacks valid build {field}")
    sha = digest(source)
    if row.get("co_sha256") is not None and row["co_sha256"] != sha:
        raise ValueError(f"Selected kid={kid} image differs from build hash")
    destination = ROOT / MANIFEST_REL.parent / "gfx950" / entry["image"]
    if destination.exists():
        raise ValueError(f"Selected new image already exists publicly; review collision: {destination}")
    return {"kid": kid, "kernelName": entry["symbol"], "image": entry["image"],
            "source": str(source), "destination": str(destination), "sha256": sha,
            "build_report_contains_sha256": "co_sha256" in row, "build_evidence": row}


def m_class(m):
    return "lt128" if m < 128 else "128_1023" if m < 1024 else "1024_8191" if m < 8192 else "ge8192"


def prepare(args):
    experiment = args.experiment_dir.resolve()
    experiment_repo = experiment / ".cache/experiment_repo"
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text())
    plan_path, identity_path = experiment / "selection_plan.json", experiment / "identity_and_points.json"
    plan, identity = json.loads(plan_path.read_text()), json.loads(identity_path.read_text())
    public_doc = json.loads((ROOT / MANIFEST_REL).read_text())
    experiment_doc = json.loads((experiment_repo / MANIFEST_REL).read_text())
    points = identity["points"]
    expected_doc, regenerated_points = prepare_manifest(public_doc, plan, len(points))
    common = load_common(experiment_repo, label="integration_experiment")
    # prepare_co_experiment.main adds the complete name after CPU loading.
    # Its points list is also referenced by manifest extension metadata, so the
    # on-disk manifest contains these names. Regenerate that exact final form.
    for point in regenerated_points:
        point["kernelName"] = registry(common)[point["new_kid"]].name
    if experiment_doc != expected_doc:
        raise ValueError("Experiment manifest differs from exactly regenerated single-variable plan")
    public = load_common(ROOT, label="integration_public")
    ids = [point["new_kid"] for point in regenerated_points]
    identity_check = check_identity(public, common, ids)
    if identity_check["original_config_digest"] != ORIGINAL_DIGEST or identity.get("identity", {}).get("original_config_digest") != ORIGINAL_DIGEST:
        raise ValueError("Original 48212 identity digest differs from audited base")
    if len(points) != len(regenerated_points):
        raise ValueError("Identity receipt point count differs from selection plan")
    for point, expected in zip(points, regenerated_points):
        if point != {**expected, "kernelName": registry(common)[expected["new_kid"]].name}:
            raise ValueError("Identity receipt parent/axis/name differs from regenerated plan")
    parity = source_parity(experiment_repo)
    paths = resolve_files(args.repeat_jsonl or report["summary"].get("files", []))
    baseline_rows, shape_rows = csv_rows(args.baseline), csv_rows(args.shapes)
    full_shapes = {key(row) for row in shape_rows}
    if len(full_shapes) != 421 or len(full_shapes) != len(shape_rows):
        raise ValueError("Integration expansion needs the unique complete 421 shapes")
    # The fixed CSV also contains unrelated gfx1250/model shapes. Match only
    # this 421-shape cohort using all six identity fields and reject ambiguous
    # or invalid baseline rows within that cohort.
    baseline = {}
    for row in baseline_rows:
        shape = key(row)
        if shape not in full_shapes:
            continue
        us = number(row.get("us"))
        if shape in baseline or us is None or us <= 0:
            raise ValueError(f"Duplicate or invalid fixed baseline for {shape}")
        baseline[shape] = us
    if set(baseline) != full_shapes:
        raise ValueError("Fixed baseline lacks a complete six-field match for each of 421 shapes")
    stats = audit_repeat(report, paths, baseline, registry_metadata(common), args.err_ratio)
    comparisons, stable_shapes = [], defaultdict(set)
    for point in points:
        kid, parent = point["new_kid"], point["seed_kid"]
        probe = copy.deepcopy(registry(common)[kid])
        probe.bpreshuffle_co.pop("supported_shapes", None)
        for mnk in point["shapes"]:
            shape = ("gfx950", 256, *map(int, mnk), "128x128")
            if (shape, kid) not in stats or (shape, parent) not in stats:
                raise ValueError(f"New and exact parent must both be repeated for {mnk}, kid={kid}, parent={parent}")
            if not common.a8w8_mxscale_bpreshuffle_supports_shape(probe, *shape[2:5]):
                raise ValueError(f"Measured shape is illegal without supported_shapes bypass: {mnk}, kid={kid}")
            comparison = {**shape_row(shape), "new_kid": kid, "parent_kid": parent,
                          "kernelName": registry(common)[kid].name,
                          "parent_kernelName": registry(common)[parent].name,
                          "pipeline": point["pipeline"], "axis": point["axis"],
                          "csv_baseline_us": baseline[shape],
                          **compare_pair(stats[(shape, kid)], stats[(shape, parent)])}
            if args.require_csv_win and not comparison["wins_csv_median"]:
                comparison["stable_improvement"] = False
                comparison["reasons"].append("csv_win_required")
            comparisons.append(comparison)
            if comparison["stable_improvement"]:
                stable_shapes[kid].add(shape)
    selected_ids = sorted(stable_shapes)
    build_path = experiment / "new_co_build_info.json"
    build_doc = json.loads(build_path.read_text())
    build_rows = {int(row["kid"]): row for row in build_doc["kernels"]}
    if len(build_rows) != len(build_doc["kernels"]):
        raise ValueError("Duplicate IDs in new CO build report")
    document, scopes, images = copy.deepcopy(public_doc), [], []
    entries = {row["kid"]: row for row in experiment_doc["gfx950"]}
    for kid in selected_ids:
        tested = {key(row) for row in comparisons if row["new_kid"] == kid}
        stable = stable_shapes[kid]
        probe = copy.deepcopy(registry(common)[kid])
        probe.bpreshuffle_co.pop("supported_shapes", None)
        enabled = set(stable)
        if args.expand_scope != "none":
            for shape in full_shapes - tested:
                controls = [control for control in stable if control[3:5] == shape[3:5]]
                if args.expand_scope == "nk-mclass":
                    controls = [control for control in controls if m_class(control[2]) == m_class(shape[2])]
                if controls and common.a8w8_mxscale_bpreshuffle_supports_shape(probe, *shape[2:5]):
                    enabled.add(shape)
        entry = copy.deepcopy(entries[kid])
        entry["supported_shapes"] = [list(shape[2:5]) for shape in sorted(enabled)]
        if "launches" in entry:
            raise ValueError("New experiment unexpectedly uses launch overrides")
        document["gfx950"].append(entry)
        document["image_suffixes"][str(kid)] = experiment_doc["image_suffixes"][str(kid)]
        images.append(build_image(kid, common.gfx950_co_kernels_declared[kid], build_rows, experiment_repo))
        for shape in sorted(enabled):
            scopes.append({**shape_row(shape), "kid": kid, "kernelName": registry(common)[kid].name,
                           "m_class": m_class(shape[2]), "cpu_legal": True,
                           "gpu_validated_stable_improvement": shape in stable,
                           "gpu_validation_pending": shape not in stable})
    document["jit_kid_base"] = 780
    document["_integrated_extensions"] = {"source_co_count": 780, "sparse_id_start": 100000,
        "source_experiment": str(experiment), "source_report": str(report_path),
        "expand_scope": args.expand_scope,
        "definitions": {"stable": "Both exact configurations correct and tuner timing in all six rounds; new median lower and >=4 same-round wins",
                        "extension": "Same N/K CPU-legal shape, optionally same M class; excluded if measured without stable improvement; pending GPU"},
        "points": [point for point in points if point["new_kid"] in selected_ids]}
    with tempfile.TemporaryDirectory(prefix="integration_identity_", dir=LOCAL) as temporary:
        manifest = Path(temporary) / "co_kernels_gfx950.json"
        dump(manifest, document)
        integrated = load_common(ROOT, manifest=manifest, freeze=True, label="integration_proposed")
        proposal_identity = check_identity(public, integrated, selected_ids)
        for scope in scopes:
            if not integrated.a8w8_mxscale_bpreshuffle_supports_shape(registry(integrated)[scope["kid"]], scope["M"], scope["N"], scope["K"]):
                raise ValueError("Proposed scope was lost in integrated registry")
        for kid in selected_ids:
            original_new, proposed = registry(common)[kid], registry(integrated)[kid]
            left, right = dataclasses.asdict(original_new), dataclasses.asdict(proposed)
            left["bpreshuffle_co"].pop("supported_shapes", None)
            right["bpreshuffle_co"].pop("supported_shapes", None)
            if left != right or original_new.name != proposed.name:
                raise ValueError("Integrated new CO identity changed beyond supported shape restriction")
    if document["gfx950"][:780] != public_doc["gfx950"] or any(document["image_suffixes"][str(k)] != public_doc["image_suffixes"][str(k)] for k in range(780)):
        raise ValueError("Original 780 manifest entries or suffixes changed")
    source = (ROOT / COMMON_REL).read_text()
    if source.count(JIT_BASE_LINE) != 1:
        raise ValueError("Public source no longer has the reviewed original JIT-base line")
    inputs = [report_path, plan_path, identity_path, build_path, experiment_repo / MANIFEST_REL,
              args.baseline.resolve(), args.shapes.resolve(), *paths]
    receipt = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "checked",
               "mode": "apply" if args.apply else "check_only", "tool_used_gpu": False, "tool_built_co": False,
               "source_experiment": str(experiment), "source_report": str(report_path),
               "identity": proposal_identity, "public_files_before": {str(rel): digest(ROOT / rel) for rel in (COMMON_REL, MANIFEST_REL)},
               "inputs": [{"path": str(path), "sha256": digest(path)} for path in sorted(set(inputs))],
               "source_parity": parity, "selected_new_ids": selected_ids,
               "summary": {"tested_new_shape_pairs": len(comparisons),
                           "tested_unique_shapes": len({key(row) for row in comparisons}),
                           "stable_improvement_pairs": sum(row["stable_improvement"] for row in comparisons),
                           "stable_improvement_unique_shapes": len({key(row) for row in comparisons if row["stable_improvement"]}),
                           "stable_improvement_csv_win_pairs": sum(row["stable_improvement"] and row["wins_csv_median"] for row in comparisons),
                           "stable_improvement_csv_win_unique_shapes": len({key(row) for row in comparisons if row["stable_improvement"] and row["wins_csv_median"]}),
                           "selected_new_co": len(selected_ids), "enabled_shape_pairs": len(scopes),
                           "gpu_validated_shape_pairs": sum(row["gpu_validated_stable_improvement"] for row in scopes),
                           "gpu_pending_shape_pairs": sum(row["gpu_validation_pending"] for row in scopes),
                           "rejection_reasons": dict(Counter(reason for row in comparisons for reason in row["reasons"]))},
               "parent_comparisons": comparisons, "new_kernel_scopes": scopes, "images": images,
               "limitations": ["Pending extensions require GPU validation and are excluded from CSV win/completion counts.",
                               "This batch targets the original 780-CO base; sparse-ID collisions or existing integrations are rejected for explicit review."]}
    return receipt, document, source.replace(JIT_BASE_LINE, FROZEN_BASE_LINE)


def atomic_write(path, content):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".integrate_", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply(receipt, document, source, out):
    if not receipt["selected_new_ids"]:
        raise ValueError("No stable new CO was selected; no public integration to apply")
    for relative, sha in receipt["public_files_before"].items():
        if digest(ROOT / relative) != sha:
            raise ValueError("Public source/manifest changed after CPU validation")
    for item in receipt["inputs"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"An audit input changed after validation: {item['path']}")
    experiment_repo = Path(receipt["source_experiment"]) / ".cache/experiment_repo"
    for item in receipt["source_parity"]:
        if digest(ROOT / item["path"]) != item["public_sha256"] or digest(experiment_repo / item["path"]) != item["experiment_sha256"]:
            raise ValueError(f"A checked source file changed after validation: {item['path']}")
    for item in receipt["images"]:
        if digest(item["source"]) != item["sha256"] or Path(item["destination"]).exists():
            raise ValueError("New image changed or destination exists after validation")
    backups, copied = {}, []
    backup_dir = out / "backups"
    for relative in (COMMON_REL, MANIFEST_REL):
        backup = backup_dir / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, backup)
        backups[str(relative)] = str(backup)
    receipt["backups"] = backups
    receipt["status"] = "applying"
    dump(out / "receipt.json", receipt)
    try:
        for item in receipt["images"]:
            destination = Path(item["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation avoids overwriting a concurrent publication.
            with destination.open("xb") as target:
                copied.append(destination)
                with Path(item["source"]).open("rb") as source_stream:
                    shutil.copyfileobj(source_stream, target)
            if digest(destination) != item["sha256"]:
                raise ValueError("Copied CO differs from validated source image")
        # Freeze first: adding manifest entries must never shift the JIT base.
        atomic_write(ROOT / COMMON_REL, source.encode())
        atomic_write(ROOT / MANIFEST_REL, (json.dumps(document, indent=2) + "\n").encode())
        post = load_common(ROOT, label="integration_applied")
        if post._bpreshuffle_jit_kid_base != 780 or any(registry(post)[item["kid"]].name != item["kernelName"] for item in receipt["images"]):
            raise ValueError("Applied registry differs from validated candidate identities")
        receipt["status"] = "applied"
        receipt["public_files_after"] = {str(rel): digest(ROOT / rel) for rel in (COMMON_REL, MANIFEST_REL)}
    except Exception as error:
        # Restore the manifest before unfreezing the original common source.
        for relative in (MANIFEST_REL, COMMON_REL):
            atomic_write(ROOT / relative, Path(backups[str(relative)]).read_bytes())
        for destination in copied:
            destination.unlink(missing_ok=True)
        receipt["status"] = "rolled_back"
        receipt["apply_error"] = str(error)
        dump(out / "receipt.json", receipt)
        raise
    dump(out / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="Completed summarize_existing_repeat report.json")
    parser.add_argument("--repeat-jsonl", nargs="+", help="Optional raw paths/globs override report summary.files")
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--shapes", type=Path, default=DOC / "shapes_421.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh review/receipt directory below local_20261029")
    parser.add_argument("--expand-scope", choices=("none", "nk-mclass", "nk"), default="none")
    parser.add_argument("--require-csv-win", action="store_true", help="Optionally require CSV victory in addition to stable parent improvement")
    parser.add_argument("--err-ratio", type=float, default=.05)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Default: write review artifacts, no public changes")
    mode.add_argument("--apply", action="store_true", help="Explicitly back up and integrate only selected new COs")
    args = parser.parse_args()
    if not math.isfinite(args.err_ratio) or not 0 <= args.err_ratio <= .05:
        parser.error("err-ratio must be finite and within [0,0.05]")
    out = args.out_dir.resolve()
    if not out.is_relative_to(LOCAL) or out == LOCAL or out.exists():
        parser.error("Choose a fresh output directory below local_20261029")
    receipt, document, source = prepare(args)
    out.mkdir(parents=True)
    dump(out / "receipt.json", receipt)
    dump(out / "proposed_co_kernels_gfx950.json", document)
    (out / "proposed_opus_gemm_common.py").write_text(source)
    for filename, rows in (("parent_comparisons.csv", receipt["parent_comparisons"]),
                           ("new_kernel_scopes.csv", receipt["new_kernel_scopes"])):
        if rows:
            write_csv(out / filename, [{field: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                      for field, value in row.items()} for row in rows])
    if args.apply:
        apply(receipt, document, source, out)
    print(json.dumps({"status": receipt["status"], "mode": receipt["mode"], "out_dir": str(out),
                      "selected_new_ids": receipt["selected_new_ids"], "summary": receipt["summary"],
                      "tool_used_gpu": False, "tool_built_co": False}, indent=2))


if __name__ == "__main__":
    main()
