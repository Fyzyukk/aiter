#!/usr/bin/env python3
"""Select finalists across campaigns; never combine their timing rounds."""
import argparse
import json
from pathlib import Path
from prepare_opus_repeat import registry_metadata, key, csv_rows, shape_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    metadata, pairs = registry_metadata()
    source = campaign.parent / "step4_final_recovery_gpu56_v2_20261030/full421_final_report/remaining_losses_sorted.csv"
    rows = csv_rows(source)
    controls = {key(row): int(row["kernelId"]) for row in rows}
    selected = {shape: {kid: {"current_step4_best_control"}} for shape, kid in controls.items()}
    sources = [campaign / "transfer/confirm", campaign / "co_round3/run/confirm"]
    for directory in sources:
        audit = json.loads((directory / "audit/summary.json").read_text())
        assert audit["completed_shapes"] == 160 and not audit["pending_shapes"]
        candidates = {}
        for path in (directory / "repeats").glob("gpu*.summary.json"):
            for row in json.loads(path.read_text())["candidates"]:
                if row["all_rounds_correct"]:
                    candidates.setdefault(key(row["shape"]), []).append(row)
        for shape, items in candidates.items():
            for item in sorted(items, key=lambda row: row["median_us"])[:2]:
                kid = item["kid"]
                assert item["kernelName"] == metadata[kid]["kernelName"]
                selected[shape].setdefault(kid, set()).add("top2_from_" + directory.parents[0].name)
    plan = {"version": 2, "family": "a8w8_blockscale_bpreshuffle", "metadata_module": "prepare_opus_repeat",
            "correctness_limit": .05, "selection": "Top2 from each completed campaign plus step4 control; all finalists start six fresh rounds on one GPU per shape",
            "source_campaigns": list(map(str, sources)), "shapes": []}
    baseline = {key(row): float(row["csv_baseline_us"]) for row in rows}
    for shape, kids in sorted(selected.items()):
        candidates = [{**metadata[kid], "roles": sorted(roles), "exact_opposite_kids": pairs[kid],
                       "us": None, "errRatio": None, "correct": False} for kid, roles in sorted(kids.items())]
        plan["shapes"].append({"shape": shape_row(shape), "csv_baseline_us": baseline[shape],
                              "remote_winner": None, "candidates": candidates})
    (campaign / "final160_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"finalist_shapes": 160, "candidate_groups": sum(len(item["candidates"]) for item in plan["shapes"])}))


if __name__ == "__main__":
    main()
