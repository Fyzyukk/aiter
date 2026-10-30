#!/usr/bin/env python3
"""Pure multi-round selection fixtures plus read-only review-artifact checks."""

import copy
import json
from pathlib import Path

import integrate_co_rounds as rounds
from summarize_opus_repeat import candidate_statistics
from argparse import Namespace


SHAPE = ("gfx950", 256, 8192, 7168, 7168, "128x128")


def statistics_for(kid, time, slot, valid=True, count=2):
    events = {index: {"round": index, "us": time, "errRatio": .001, "correct": valid,
                      "rotate_args": 0, "gpu_logical_id": 0, "failure": None,
                      "position": slot if index % 2 == 0 else count - 1 - slot} for index in range(6)}
    row = candidate_statistics(SHAPE, kid, {"kernelName": str(kid), "kind": "CO"}, events, 20,
                               Namespace(rounds=6, err_ratio=.05))
    return rounds.single.audit_candidate(row, events, 20)


def main():
    p1 = {"new_kid": 100000, "seed_kid": 392, "kernelName": "fixture100000", "shapes": [list(SHAPE[2:5])]}
    p2 = {"new_kid": 100024, "seed_kid": 100000, "kernelName": "fixture100024", "shapes": [list(SHAPE[2:5])]}
    points = {1: [p1], 2: [p2]}
    evidence = {1: {"report_path": Path("fixture_r1"), "audited": {(SHAPE, 392): statistics_for(392, 10, 0),
                       (SHAPE, 100000): statistics_for(100000, 9, 1)}},
                2: {"report_path": Path("fixture_r2"), "audited": {(SHAPE, 100000): statistics_for(100000, 9, 0),
                       (SHAPE, 100024): statistics_for(100024, 8, 1)}}}
    rows, stable, _ = rounds.evaluate_rounds(points, evidence, selected_filter=[100024])
    assert set(stable) == {100024}
    child = next(row for row in rows if row["new_kid"] == 100024)
    parent = next(row for row in rows if row["new_kid"] == 100000)
    assert child["original_co_ancestor_chain"] == [100000, 392]
    assert child["stable_basis_round"] == child["current_timing_round"] == 2
    assert child["original_co_control"]["comparison_mode"] == "historical_separate_batch_control"
    assert parent["stable_basis_round"] == 1 and parent["current_timing_round"] == 2
    checks = {"independent_child_does_not_require_parent_selection": True,
              "separate_round_indices_retained": True, "historical_base_is_not_a_paired_gate": True}
    next_evidence = copy.deepcopy(evidence)
    next_evidence[2]["audited"][(SHAPE, 392)] = statistics_for(392, 8, 2, count=3)
    next_evidence[2]["audited"][(SHAPE, 100000)] = statistics_for(100000, 9, 0, count=3)
    next_evidence[2]["audited"][(SHAPE, 100024)] = statistics_for(100024, 8, 1, count=3)
    rows, stable, _ = rounds.evaluate_rounds(points, next_evidence)
    row = next(row for row in rows if row["new_kid"] == 100000)
    assert row["stable_basis_round"] == 2 and not row["selected_stable_improvement"]
    checks["latest_same_parent_regression_overrides_old_win"] = True
    invalid = copy.deepcopy(evidence)
    invalid[2]["audited"][(SHAPE, 100000)] = statistics_for(100000, 9, 0, valid=False)
    rows, stable, _ = rounds.evaluate_rounds(points, invalid)
    row = next(row for row in rows if row["new_kid"] == 100000)
    assert not row["selected_stable_improvement"] and any(reason.startswith("latest_new_") for reason in row["reasons"])
    assert not stable
    checks["latest_new_only_failure_invalidates_old_win_and_child_parent"] = True
    none = copy.deepcopy(evidence)
    none[2]["audited"].pop((SHAPE, 100000))
    try:
        rounds.evaluate_rounds(points, none)
    except ValueError:
        checks["missing_origin_exact_parent_rejected"] = True
    else:
        raise AssertionError("Missing exact parent accepted")
    assert rounds.extension_controls("k-wide", "pin_agpr", 0, {SHAPE}, ("gfx950", 256, 2048, 6144, 7168, "128x128"))
    assert not rounds.extension_controls("k-wide", "pin_agpr", 0, {SHAPE}, ("gfx950", 256, 2048, 7168, 16384, "128x128"))
    narrow = ("gfx950", 256, 4096, 1536, 7168, "128x128")
    assert rounds.extension_controls("k-narrow", "prefetch_b_vgpr_overlap", 7168, {narrow}, ("gfx950", 256, 8192, 768, 7168, "128x128"))
    assert not rounds.extension_controls("k-narrow", "prefetch_b_vgpr_overlap", 384, {narrow}, ("gfx950", 256, 8192, 768, 7168, "128x128"))
    checks["finite_family_scope_requires_pipeline_K_N_controls"] = True
    review = rounds.LOCAL / "co_rounds_csv_family_review"
    receipt = json.loads((review / "receipt.json").read_text())
    proposal = json.loads((review / "proposed_co_kernels_gfx950.json").read_text())
    ids = {row["kid"] for row in proposal["gfx950"]}
    assert ids == set(range(780)) | {100003, 100024, 100025, 100030, 100031}
    assert 100001 not in ids and 100022 not in ids and 100023 not in ids
    assert receipt["identity"]["original_ids_verified"] == 48212
    assert proposal["_integrated_extensions"]["points"] == []
    assert all(not point["runtime_parent_required"] for point in proposal["_integrated_extensions"]["standalone_points"])
    enabled = {(rounds.key(row), row["kid"]) for row in receipt["new_kernel_scopes"]}
    assert all((rounds.key(row), row["new_kid"]) not in enabled for row in receipt["parent_comparisons"] if not row["selected_stable_improvement"])
    assert all(not row["wins_csv_median"] and not row["gpu_validated_stable_improvement"] for row in receipt["new_kernel_scopes"] if row["gpu_validation_pending"])
    checks["actual_five_CO_proposal_omits_unselected_sparse_parents"] = True
    checks["actual_proposal_preserves_original48212_and_excludes_tested_losses"] = True
    checks["pending_extensions_not_counted_as_validated_or_CSV_wins"] = True
    output = {"fixture_only": True, "gpu_used": False, "build_used": False, "public_applied": False,
              "checks": checks, "actual_review_summary": receipt["summary"]}
    rounds.single.dump(rounds.LOCAL / "co_rounds_cpu_fixture_report.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
