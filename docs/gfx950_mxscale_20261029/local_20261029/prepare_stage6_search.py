#!/usr/bin/env python3
"""Select complete traits sweeps around measured geometries for stage5 losses."""
import json
import math
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

from prepare_opus_repeat import csv_rows, key, registry_metadata, pure_common, shape_row

LOCAL = Path(__file__).resolve().parent
SOURCE = LOCAL / "stage5_optimize160_20261030"
OUT = LOCAL / "stage6_optimize117_20261030"


def main():
    OUT.mkdir()
    metadata, pairs = registry_metadata()
    common = pure_common()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    previous_best = csv_rows(SOURCE / "result/public_best_opus421.csv")
    losses = [row for row in previous_best if row["status"] == "slower_than_csv"]
    assert len(losses) == 117 and len(metadata) == 48240
    observed, tested = defaultdict(list), defaultdict(set)
    for phase in ("transfer/screen", "co_round3/run/screen", "final160/confirm"):
        for path in (SOURCE / phase / "repeats").glob("gpu*.summary.json"):
            for row in json.loads(path.read_text())["candidates"]:
                shape = key(row["shape"])
                tested[shape].add(row["kid"])
                if row["all_rounds_correct"] and row["kid"] in metadata:
                    observed[shape].append(row)
    by_nk = defaultdict(set)
    for row in previous_best:
        by_nk[(int(row["N"]), int(row["K"]))].add(int(row["kernelId"]))
    items, details = [], []
    for row in losses:
        shape = key(row)
        control = int(row["kernelId"])
        legal = {kid for kid, instance in registry.items()
                 if common.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape[2:5])}
        geometries = {tuple(metadata[control]["geometry"])}
        ranked = sorted(observed[shape], key=lambda item: item["median_us"])
        # Complete the traits search in the fastest measured non-CO geometry.
        # Also include the fastest alternative JIT pipeline within 15% of the
        # reference, avoiding the previous four-variant geometry sampling.
        seen = set()
        for candidate in ranked:
            kid = candidate["kid"]
            if metadata[kid]["kind"] != "JIT" or candidate["median_us"] > float(row["us"]) * 1.15:
                continue
            family = metadata[kid]["geometry"][0]
            if family in seen:
                continue
            geometries.add(tuple(metadata[kid]["geometry"]))
            seen.add(family)
            if len(seen) == 2:
                break
        chosen = {control: {"current_step4_best_control"}}
        for kid in sorted(legal):
            if tuple(metadata[kid]["geometry"]) in geometries and kid not in tested[shape]:
                chosen.setdefault(kid, set()).add("complete_traits_in_measured_geometry")
        for kid in by_nk[shape[3:5]] & legal:
            chosen.setdefault(kid, set()).add("current_winner_other_M")
        # Retain the best previously screened non-control candidates as warm
        # controls, while giving previously unmeasured variants a real trial.
        for candidate in ranked[:3]:
            kid = candidate["kid"]
            if kid in legal:
                chosen.setdefault(kid, set()).add("previous_screen_top3")
        candidates = [{**metadata[kid], "roles": sorted(roles), "exact_opposite_kids": pairs[kid],
                       "us": None, "errRatio": None, "correct": False}
                      for kid, roles in sorted(chosen.items())]
        items.append({"shape": shape_row(shape), "csv_baseline_us": float(row["csv_baseline_us"]),
                      "remote_winner": None, "candidates": candidates})
        details.append({"shape": shape_row(shape), "current_control": control,
                        "geometries": sorted(geometries), "candidates": len(candidates),
                        "new_to_stage5_screen": len(chosen.keys() - tested[shape])})
    plan = {"version": 2, "family": "a8w8_blockscale_bpreshuffle", "metadata_module": "prepare_opus_repeat",
            "correctness_limit": .05, "selection": "Complete untested legal traits in measured promising geometries; current stage5 best as control",
            "reference": str(SOURCE), "shapes": items}
    (OUT / "search_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    receipt = {"shape_count": 117, "candidate_groups": sum(len(item["candidates"]) for item in items),
               "new_to_stage5_groups": sum(row["new_to_stage5_screen"] for row in details), "details": details}
    (OUT / "search_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    state = json.loads((LOCAL / "workflow_state.json").read_text())
    state["active_optimization117"] = {"directory": str(OUT), "phase": "search_preparing",
       "reference_wins": 304, "reference_losses": 117, "physical_gpus": [5, 6],
       "started_utc": datetime.now(timezone.utc).isoformat(), "candidate_groups": receipt["candidate_groups"]}
    (LOCAL / "workflow_state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "details"}))


if __name__ == "__main__":
    main()
