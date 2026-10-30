#!/usr/bin/env python3
"""CPU audit and optional publication of independent winners from two CO rounds.

Round1 uses the retained804 snapshot; round2 uses its816 proposal. Raw batches
remain separate. Sparse immediate parents are evidence, not image dependencies.
Default/check-only never changes public files or uses a GPU/build.
"""

import argparse
import copy
import dataclasses
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import integrate_co_winners as single
import extend_co_experiment as extension
from prepare_co_experiment import (LOCAL, ROOT, COMMON_REL, MANIFEST_REL,
                                   JIT_BASE_LINE, FROZEN_BASE_LINE, load_common,
                                   registry, prepare_manifest, check_identity)
from prepare_opus_repeat import DATA, DOC, csv_rows, key, shape_row, number
from summarize_step1_repeat import write_csv


def baseline_cohort(baseline_path, shapes_path):
    rows = csv_rows(shapes_path)
    shapes = {key(row) for row in rows}
    if len(rows) != 421 or len(shapes) != 421:
        raise ValueError("Expected the unique complete421 cohort")
    baseline = {}
    for row in csv_rows(baseline_path):
        shape = key(row)
        if shape not in shapes:
            continue
        us = number(row.get("us"))
        if shape in baseline or us is None or us <= 0:
            raise ValueError("Duplicate or invalid fixed cohort baseline")
        baseline[shape] = us
    if set(baseline) != shapes:
        raise ValueError("Fixed CSV lacks full six-field cohort coverage")
    return baseline, shapes


def read_round(path, snapshot_common, baseline):
    report = json.loads(path.read_text())
    files = single.resolve_files(report["summary"].get("files", []))
    audited = single.audit_repeat(report, files, baseline, single.registry_metadata(snapshot_common), .05)
    return {"report": report, "report_path": path, "files": files, "audited": audited}


def validate_snapshots(args, baseline):
    first, second = args.round1_dir.resolve(), args.round2_dir.resolve()
    receipt_path = second / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    repo = Path(receipt["base_repo"]).resolve()
    if not repo.is_relative_to(LOCAL) or repo == ROOT or receipt.get("status") != "extended_isolated_manifest":
        raise ValueError("Round2 must identify its applied isolated repo")
    snapshot1_path, snapshot2_path = second / "base_manifest.json", second / "proposed_manifest.json"
    snapshot1, snapshot2 = json.loads(snapshot1_path.read_text()), json.loads(snapshot2_path.read_text())
    if single.digest(snapshot1_path) != receipt["base_manifest_sha256"] or single.digest(snapshot2_path) != receipt["manifest_after_sha256"]:
        raise ValueError("Retained snapshots differ from round2 apply receipt")
    if json.loads((repo / MANIFEST_REL).read_text()) != snapshot2:
        raise ValueError("Live clone changed beyond the816 snapshot; supply a separately reviewed later-round mechanism")
    if snapshot2["gfx950"][:804] != snapshot1["gfx950"] or len(snapshot1["gfx950"]) != 804 or len(snapshot2["gfx950"]) != 816:
        raise ValueError("Round2 snapshots do not preserve exact804→816 entries")
    first_identity_path = first / "identity_and_points.json"
    first_identity = json.loads(first_identity_path.read_text())
    first_plan_path, second_plan_path = first / "selection_plan.json", second / "selection_plan.json"
    first_plan, second_plan = json.loads(first_plan_path.read_text()), json.loads(second_plan_path.read_text())
    public_doc = json.loads((ROOT / MANIFEST_REL).read_text())
    public = load_common(ROOT, label="rounds_public")
    common1 = load_common(repo, manifest=snapshot1_path, label="rounds_snapshot1")
    common2 = load_common(repo, manifest=snapshot2_path, label="rounds_snapshot2")
    document1, points1 = prepare_manifest(public_doc, first_plan, 24)
    for point in points1:
        point["kernelName"] = registry(common1)[point["new_kid"]].name
    if document1 != snapshot1 or points1 != first_identity["points"]:
        raise ValueError("Round1 snapshot differs from exactly regenerated original selected plan")
    if first_identity["identity"].get("original_config_digest") != single.ORIGINAL_DIGEST:
        raise ValueError("Round1 original identity evidence differs")
    round1 = read_round(args.round1_report.resolve(), common1, baseline)
    if single.digest(round1["report_path"]) != receipt["source_report_sha256"]:
        raise ValueError("Round2 seed evidence refers to a different round1 report")
    document2, points2 = extension.prepare_document(snapshot1, second_plan, 24)
    extension.validate_points(common1, common2, points2, baseline, round1["audited"])
    if document2 != snapshot2 or points2 != receipt["points"]:
        raise ValueError("Round2 snapshot differs from exactly regenerated single-variable plan/evidence")
    history = snapshot2.get("_experimental_extension_history", [])
    if history != [{"round": 1, "metadata": snapshot1["_experimental_extensions"]}]:
        raise ValueError("Round1 metadata history changed or was confused with active round2 metadata")
    previous_identity = extension.check_identity(common1, common2, [point["new_kid"] for point in points2])
    if previous_identity != receipt["identity"]:
        raise ValueError("Existing48212+24 identity differs from the recorded extension receipt")
    all_ids = [point["new_kid"] for point in points1 + points2]
    original_identity = check_identity(public, common2, all_ids)
    if original_identity["original_config_digest"] != single.ORIGINAL_DIGEST:
        raise ValueError("Current clone does not preserve original48212 identity")
    parity = single.source_parity(repo)
    if single.digest(repo / COMMON_REL) != receipt["base_common_sha256"]:
        raise ValueError("Isolated common source changed since round2 preparation")
    inputs = [receipt_path, snapshot1_path, snapshot2_path, first_identity_path, first_plan_path,
              second_plan_path, repo / MANIFEST_REL, *round1["files"], round1["report_path"]]
    if single.digest(Path(receipt["source_build_report"])) != receipt["source_build_report_sha256"]:
        raise ValueError("Round1 build report changed since round2 preparation")
    return {"repo": repo, "public": public, "public_doc": public_doc,
            "common": {1: common1, 2: common2}, "snapshots": {1: snapshot1, 2: snapshot2},
            "points": {1: points1, 2: points2}, "rounds": {1: round1},
            "directories": {1: first, 2: second}, "inputs": inputs,
            "source_parity": parity, "source_identity": original_identity,
            "previous_identity": previous_identity}


def ancestry(kid, points):
    chain, visited = [], set()
    current = kid
    while current >= 780:
        if current in visited or current not in points:
            raise ValueError("Sparse immediate-parent provenance is cyclic or unknown")
        visited.add(current)
        current = points[current]["seed_kid"]
        chain.append(current)
    return chain


def evaluate_rounds(points_by_round, rounds, require_csv=False, selected_filter=None):
    """Select using exact within-batch comparisons; later measurements supersede."""
    point_index = {point["new_kid"]: point for points in points_by_round.values() for point in points}
    if len(point_index) != sum(len(points) for points in points_by_round.values()):
        raise ValueError("A sparse new ID appears in more than one source round")
    if selected_filter is not None and not set(selected_filter) <= set(point_index):
        raise ValueError("Requested new IDs are absent from validated round provenance")
    comparisons, stable_shapes, latest_stats = [], defaultdict(set), {}
    for origin, points in sorted(points_by_round.items()):
        for point in points:
            kid, parent = point["new_kid"], point["seed_kid"]
            ancestors = ancestry(kid, point_index)
            for mnk in point["shapes"]:
                shape = ("gfx950", 256, *mnk, "128x128")
                first = rounds[origin]["audited"]
                if (shape, kid) not in first or (shape, parent) not in first:
                    raise ValueError(f"Origin round lacks new/exact immediate parent for kid={kid}, {mnk}")
                decision = single.compare_pair(first[(shape, kid)], first[(shape, parent)])
                stable_round = origin
                current, current_round = first[(shape, kid)], origin
                pair_history = [{"round": origin, **decision}]
                for index in sorted(rounds):
                    if index <= origin or (shape, kid) not in rounds[index]["audited"]:
                        continue
                    batch = rounds[index]["audited"]
                    current, current_round = batch[(shape, kid)], index
                    if (shape, parent) in batch:
                        decision = single.compare_pair(current, batch[(shape, parent)])
                        stable_round = index
                        pair_history.append({"round": index, **decision})
                    elif not current["valid"]:
                        decision = {**decision, "stable_improvement": False,
                                    "reasons": ["latest_new_" + reason for reason in current["reasons"]]}
                    # A valid later new-only sample updates timing, retaining
                    # its previously established exact-parent stable basis.
                eligible = decision["stable_improvement"] and current["valid"]
                reasons = list(decision["reasons"])
                if require_csv and not current["wins_csv_median"]:
                    eligible = False
                    reasons.append("csv_win_required")
                if selected_filter is not None and kid not in selected_filter:
                    eligible = False
                    reasons.append("outside_requested_new_ids")
                base_kid = ancestors[-1]
                controls = [(index, batch["audited"][(shape, base_kid)]) for index, batch in sorted(rounds.items())
                            if index <= current_round and (shape, base_kid) in batch["audited"] and batch["audited"][(shape, base_kid)]["valid"]]
                if controls:
                    control_round, control = controls[-1]
                    base_control = {"kid": base_kid, "round": control_round, "median_us": control["median_us"],
                                    "source_report": str(rounds[control_round]["report_path"]),
                                    "comparison_mode": "same_round_control" if control_round == current_round else "historical_separate_batch_control"}
                    if control_round == current_round:
                        base_control["same_round_comparison"] = single.compare_pair(current, control)
                else:
                    base_control = {"kid": base_kid, "comparison_mode": "not_measured_in_supplied_reports"}
                row = {**shape_row(shape), "new_kid": kid, "parent_kid": parent,
                       "kernelName": point["kernelName"], "origin_round": origin,
                       "stable_basis_round": stable_round, "current_timing_round": current_round,
                       "current_median_us": current["median_us"], "csv_baseline_us": current["csv_baseline_us"],
                       "wins_csv_median": current["wins_csv_median"], "selected_stable_improvement": eligible,
                       "reasons": sorted(set(reasons)), "immediate_parent_pair_history": pair_history,
                       "original_co_ancestor_chain": ancestors, "original_co_control": base_control}
                comparisons.append(row)
                latest_stats[(shape, kid)] = current
                if eligible:
                    stable_shapes[kid].add(shape)
    return comparisons, stable_shapes, latest_stats


def extension_controls(mode, pipeline, fixed_k, stable, shape):
    if mode == "family":
        mode = "k-narrow" if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap") and fixed_k == 7168 else \
               "k-wide" if pipeline == "pin_agpr" and shape[4] == 7168 else "nk"
    if mode == "k-narrow":
        if pipeline not in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap") or fixed_k != 7168 or shape[3] not in (768, 1536, 2048) or shape[4] != 7168:
            return []
        return [control for control in stable if control[3] in (768, 1536, 2048) and control[4] == 7168]
    if mode == "k-wide":
        if pipeline != "pin_agpr" or shape[3] not in (6144, 7168) or shape[4] != 7168:
            return []
        return [control for control in stable if control[3] in (6144, 7168) and control[4] == 7168]
    controls = [control for control in stable if control[3:5] == shape[3:5]]
    if mode == "nk-mclass":
        controls = [control for control in controls if single.m_class(control[2]) == single.m_class(shape[2])]
    return controls


def prepare(args):
    baseline, full_shapes = baseline_cohort(args.baseline, args.shapes)
    state = validate_snapshots(args, baseline)
    if args.snapshot_check_only:
        return {"mode": "snapshot_check_only", "status": "checked", "source_identity": state["source_identity"],
                "previous_identity": state["previous_identity"], "source_repo": str(state["repo"]),
                "round1_repeat_candidates": len(state["rounds"][1]["audited"]),
                "round2_gpu_report_pending": not args.round2_report.exists(),
                "tool_used_gpu": False, "tool_built_co": False}, None, None
    state["rounds"][2] = read_round(args.round2_report.resolve(), state["common"][2], baseline)
    state["inputs"].extend([state["rounds"][2]["report_path"], *state["rounds"][2]["files"]])
    comparisons, stable_shapes, latest_stats = evaluate_rounds(state["points"], state["rounds"],
                                                               args.require_csv_win, args.select_new_ids)
    selected_ids = sorted(stable_shapes)
    document = copy.deepcopy(state["public_doc"])
    images, scopes, provenance = [], [], []
    all_points = {point["new_kid"]: (index, point) for index, points in state["points"].items() for point in points}
    for index, directory in state["directories"].items():
        path = directory / "new_co_build_info.json"
        state["inputs"].append(path)
        build = json.loads(path.read_text())
        rows = {int(row["kid"]): row for row in build["kernels"]}
        if len(rows) != len(build["kernels"]) or set(rows) != {point["new_kid"] for point in state["points"][index]}:
            raise ValueError("Each round build report must contain exactly its separately declared new IDs")
        state["rounds"][index]["build_rows"] = rows
    for kid in selected_ids:
        origin, point = all_points[kid]
        common, snapshot = state["common"][origin], state["snapshots"][origin]
        probe = copy.deepcopy(registry(common)[kid])
        probe.bpreshuffle_co.pop("supported_shapes", None)
        stable, enabled = stable_shapes[kid], set(stable_shapes[kid])
        tested = {shape for batch in state["rounds"].values() for shape, tested_kid in batch["audited"] if tested_kid == kid}
        if args.expand_scope != "none":
            for shape in full_shapes - tested:
                controls = extension_controls(args.expand_scope, point["pipeline"], probe.bpreshuffle_fixed_k, stable, shape)
                if controls and common.a8w8_mxscale_bpreshuffle_supports_shape(probe, *shape[2:5]):
                    enabled.add(shape)
        entry = copy.deepcopy(next(row for row in snapshot["gfx950"] if row["kid"] == kid))
        if "launches" in entry:
            raise ValueError("Independent experimental CO unexpectedly carries per-shape launches")
        for shape in enabled:
            if not common.a8w8_mxscale_bpreshuffle_supports_shape(probe, *shape[2:5]):
                raise ValueError("Selected scope violates underlying shape guards")
        entry["supported_shapes"] = [list(shape[2:5]) for shape in sorted(enabled)]
        document["gfx950"].append(entry)
        document["image_suffixes"][str(kid)] = snapshot["image_suffixes"][str(kid)]
        images.append(single.build_image(kid, common.gfx950_co_kernels_declared[kid],
                                         state["rounds"][origin]["build_rows"], state["repo"]))
        provenance.append({**copy.deepcopy(point), "source_round": origin,
                           "source_report": str(state["rounds"][origin]["report_path"]),
                           "runtime_parent_required": False,
                           "integrated_shapes": [list(shape[2:5]) for shape in sorted(enabled)]})
        for shape in sorted(enabled):
            current = latest_stats.get((shape, kid))
            scopes.append({**shape_row(shape), "kid": kid, "kernelName": point["kernelName"],
                           "source_round": origin, "cpu_legal": True,
                           "gpu_validated_stable_improvement": shape in stable,
                           "gpu_validation_pending": shape not in stable,
                           "current_median_us": current["median_us"] if current else None,
                           "wins_csv_median": bool(current and current["wins_csv_median"] and shape in stable)})
    document["jit_kid_base"] = 780
    # Existing analyzer points require a live exact seed. These independent
    # images keep genuine parent provenance outside that active-pair schema.
    document["_integrated_extensions"] = {"source_co_count": 780, "sparse_id_start": 100000,
        "points": [], "standalone_points": provenance, "expand_scope": args.expand_scope,
        "runtime_parent_required": False,
        "definitions": {"stable": "Correct six rounds, tuner timing, median below immediate seed and >=4 paired round wins in one source batch",
                        "base_control": "Original CO ancestor shown with its report/round; historical controls are never paired with another batch",
                        "standalone": "Child traits/body link only repo headers/functions; sparse parent image is not a runtime dependency"}}
    with tempfile.TemporaryDirectory(prefix="rounds_publish_identity_", dir=LOCAL) as temporary:
        path = Path(temporary) / "manifest.json"
        single.dump(path, document)
        integrated = load_common(ROOT, manifest=path, freeze=True, label="rounds_proposed")
        identity = check_identity(state["public"], integrated, selected_ids)
        for kid in selected_ids:
            origin, _ = all_points[kid]
            left = dataclasses.asdict(registry(state["common"][origin])[kid])
            right = dataclasses.asdict(registry(integrated)[kid])
            left["bpreshuffle_co"].pop("supported_shapes", None)
            right["bpreshuffle_co"].pop("supported_shapes", None)
            if left != right:
                raise ValueError("Independent selected child identity changed beyond scope")
    if document["gfx950"][:780] != state["public_doc"]["gfx950"] or any(document["image_suffixes"][str(kid)] != state["public_doc"]["image_suffixes"][str(kid)] for kid in range(780)):
        raise ValueError("Public original780 manifest entries/suffixes changed")
    source = (ROOT / COMMON_REL).read_text()
    if source.count(JIT_BASE_LINE) != 1:
        raise ValueError("Public base is not the original reviewed780 source")
    state["inputs"].extend([args.baseline.resolve(), args.shapes.resolve()])
    receipt = {"version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "checked", "mode": "apply" if args.apply else "check_only",
               "source_experiment": str(args.round1_dir.resolve()), "source_repo": str(state["repo"]),
               "source_rounds": [{"round": index, "report": str(batch["report_path"]),
                                  "raw_files": [str(path) for path in batch["files"]],
                                  "summary": batch["report"]["summary"]} for index, batch in state["rounds"].items()],
               "identity": identity, "source_identity": state["source_identity"],
               "public_files_before": {str(rel): single.digest(ROOT / rel) for rel in (COMMON_REL, MANIFEST_REL)},
               "inputs": [{"path": str(path), "sha256": single.digest(path)} for path in sorted(set(state["inputs"]))],
               "source_parity": state["source_parity"], "selected_new_ids": selected_ids,
               "summary": {"tested_new_shape_pairs": len(comparisons),
                           "tested_unique_shapes": len({key(row) for row in comparisons}),
                           "selected_new_co": len(selected_ids),
                           "stable_improvement_pairs": sum(row["selected_stable_improvement"] for row in comparisons),
                           "stable_improvement_unique_shapes": len({key(row) for row in comparisons if row["selected_stable_improvement"]}),
                           "stable_improvement_csv_win_unique_shapes": len({key(row) for row in comparisons if row["selected_stable_improvement"] and row["wins_csv_median"]}),
                           "gpu_validated_shape_pairs": sum(row["gpu_validated_stable_improvement"] for row in scopes),
                           "gpu_pending_shape_pairs": sum(row["gpu_validation_pending"] for row in scopes),
                           "rejection_reasons": dict(Counter(reason for row in comparisons for reason in row["reasons"]))},
               "parent_comparisons": comparisons, "new_kernel_scopes": scopes, "images": images,
               "standalone_provenance": provenance, "tool_used_gpu": False, "tool_built_co": False,
               "limitations": ["Pending scope extensions require GPU validation and do not count as CSV wins.",
                               "Historical original-CO controls retain their batch identity and do not satisfy paired stability.",
                               "Active integrated pair metadata is empty; independent CO provenance is stored as standalone_points."]}
    return receipt, document, source.replace(JIT_BASE_LINE, FROZEN_BASE_LINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1-dir", type=Path, default=LOCAL / "co_round1")
    parser.add_argument("--round2-dir", type=Path, default=LOCAL / "co_round2")
    parser.add_argument("--round1-report", type=Path, default=LOCAL / "co_round1/repeat_report/report.json")
    parser.add_argument("--round2-report", type=Path, default=LOCAL / "co_round2/repeat_report/report.json")
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--shapes", type=Path, default=DOC / "shapes_421.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expand-scope", choices=("none", "nk-mclass", "nk", "k-narrow", "k-wide", "family"), default="none")
    parser.add_argument("--require-csv-win", action="store_true")
    parser.add_argument("--select-new-ids", nargs="+", type=int, help="Optional subset after full evidence validation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--snapshot-check-only", action="store_true", help="Validate both snapshots and round1 CPU evidence while round2 GPU report is pending")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if not out.is_relative_to(LOCAL) or out == LOCAL or out.exists():
        parser.error("Choose a fresh local review/receipt directory")
    receipt, document, source = prepare(args)
    out.mkdir(parents=True)
    single.dump(out / "receipt.json", receipt)
    if document is not None:
        single.dump(out / "proposed_co_kernels_gfx950.json", document)
        (out / "proposed_opus_gemm_common.py").write_text(source)
        for name in ("parent_comparisons", "new_kernel_scopes"):
            if receipt[name]:
                write_csv(out / (name + ".csv"), [{field: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                                  for field, value in row.items()} for row in receipt[name]])
        if args.apply:
            single.apply(receipt, document, source, out)
    print(json.dumps({key: receipt[key] for key in ("status", "mode", "selected_new_ids", "summary", "tool_used_gpu", "tool_built_co") if key in receipt}, indent=2))


if __name__ == "__main__":
    main()
