#!/usr/bin/env python3
"""CPU-only proof that every candidate in the current final plan was measured.

Reuse the reviewed focused strict current-batch validator without its cohort
classification. Historical records and profile timings cannot fill any round.
"""

import argparse
import json
import sys
from pathlib import Path

import finish_workflow as accounting
import prepare_opus_repeat as metadata_helper
from analyze_focused_results import repeat_results
from prepare_final_repeat import known_csv_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True, help="Only the current final repeat files or quoted globs")
    parser.add_argument("--expected", type=Path, default=metadata_helper.DOC / "shapes_421.csv")
    parser.add_argument("--baseline", type=Path, default=metadata_helper.DATA / "csv_baseline.csv")
    parser.add_argument("--known-csv", type=Path, default=metadata_helper.LOCAL / "focused_expansion_known_complete/known_winners.csv")
    parser.add_argument("--output-dir", type=Path, default=metadata_helper.LOCAL / "final_batch_audit")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--physical-gpus", help="Require the current raw logical-to-physical GPU mapping")
    args = parser.parse_args()
    # Explicit --files makes this fallback irrelevant; strict validator still
    # accepts the same Namespace as the focused accounting tool.
    args.run_dir = metadata_helper.LOCAL / "final"
    try:
        source_rows = metadata_helper.csv_rows(args.expected)
        expected = {accounting.strict_key(row, args.expected) for row in source_rows}
        if not expected or len(expected) != len(source_rows):
            raise ValueError("Expected shapes must be nonempty with unique complete shape keys")
        baseline = {}
        for row in metadata_helper.csv_rows(args.baseline):
            shape = accounting.strict_key(row, args.baseline, require_scope=False)
            if shape not in expected:
                continue
            us = accounting.finite(row.get("us"))
            if shape in baseline or us is None or us <= 0:
                raise ValueError("Invalid or duplicate fixed CSV baseline")
            baseline[shape] = us
        if set(baseline) != expected:
            raise ValueError("Fixed CSV does not cover exactly every expected shape")
        common = metadata_helper.pure_common()
        metadata, _ = metadata_helper.registry_metadata()
        shapes, candidates, raw, batch, planned = repeat_results(args, expected, baseline, metadata, common)
        if args.physical_gpus:
            physical_gpus = [int(value) for value in args.physical_gpus.split(",")]
            if len(physical_gpus) != 4 or len(set(physical_gpus)) != 4 or any(gpu not in range(8) for gpu in physical_gpus):
                raise ValueError("Select four distinct physical GPUs from 0 to 7")
            for rounds in raw.values():
                for record in rounds.values():
                    if record.get("gpu_physical_id") != physical_gpus[record["gpu_logical_id"]]:
                        raise ValueError("Current raw logical/physical GPU mapping differs from the authorized GPUs")
        known, outside = known_csv_candidates([args.known_csv.resolve()], expected, metadata, common)
        known_missing = []
        for shape, items in known.items():
            for kid in items:
                candidate = candidates.get(shape, {}).get(kid)
                reason = ("not_in_current_plan" if kid not in planned[shape] else
                          "incorrect_or_incomplete_current_rounds" if not candidate or not candidate["correct"] else None)
                if reason:
                    known_missing.append({**accounting.shape_row(shape), "kid": kid, "reason": reason})
        summary = {"expected_shapes": len(expected),
                   "completed_shapes": sum(row["repeat_complete"] for row in shapes),
                   "pending_shapes": sum(not row["repeat_complete"] for row in shapes),
                   "no_correct_opus": sum(row["status"] == "no_correct_opus" for row in shapes),
                   "wins_csv": sum(row["wins_csv"] for row in shapes),
                   "losses_csv": sum(row["status"] == "slower_than_csv" for row in shapes),
                   "known_csv_required_candidates": sum(len(items) for items in known.values()),
                   "known_csv_missing_incorrect_or_incomplete": len(known_missing),
                   "known_csv_outside_expected_rows": outside,
                   "current_raw_records": sum(len(rounds) for rounds in raw.values()),
                   **batch,
                   "torch_imported": "torch" in sys.modules,
                   "aiter_imported": "aiter" in sys.modules,
                   "gpu_used_by_tool": False,
                   "jit_or_co_built": False,
                   "physical_gpu_mapping_checked": args.physical_gpus,
                   "definition": "Only this exact current plan and supplied batch; every planned candidate must have all six rounds; known CSV candidates must be in the plan and correct in all six rounds"}
        if summary["torch_imported"] or summary["aiter_imported"]:
            raise ValueError("CPU audit unexpectedly imported GPU/JIT modules")
        if args.require_complete and (summary["pending_shapes"] or summary["no_correct_opus"] or known_missing):
            raise ValueError("Current final batch is incomplete, has no correct winner, or lacks a fully correct known CSV candidate")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, value in (("summary", summary), ("report", {"summary": summary, "shapes": shapes, "known_missing": known_missing})):
            accounting.atomic_text(args.output_dir / (name + ".json"), json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        accounting.atomic_csv(args.output_dir / "known_missing.csv", known_missing,
                              accounting.SHAPE_FIELDS + ("kid", "reason"))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
