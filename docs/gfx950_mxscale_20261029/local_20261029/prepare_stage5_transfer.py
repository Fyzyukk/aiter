#!/usr/bin/env python3
"""CPU-only expansion for all 160 step4 losses using already compiled OPUSs."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from prepare_opus_repeat import csv_rows, key, registry_metadata, pure_common

LOCAL = Path(__file__).resolve().parent
FINAL = LOCAL / "step4_final_recovery_gpu56_v2_20261030"
OUT = LOCAL / "stage5_optimize160_20261030"


def main():
    meta, pairs = registry_metadata()
    common = pure_common()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    union = set(json.loads((LOCAL / "full421_union/union.json").read_text())["union_kids"])
    best = csv_rows(FINAL / "full421_final_report/best_opus.csv")
    losses = [row for row in best if row["status"] == "slower_than_csv"]
    assert len(losses) == 160
    previous = {key(item["shape"]): {row["kid"] for row in item["candidates"]}
                for item in json.loads((FINAL / "repeat_plan.json").read_text())["shapes"]}
    by_nk = defaultdict(set)
    for row in best:
        by_nk[(int(row["N"]), int(row["K"]))].add(int(row["kernelId"]))
    # Stable winners from other M values and nearest complete-traits variants
    # can be overlooked by a noisy single-pass Top5. Geometry expansion uses
    # only the compiled union, so the production binary is never rebuilt here.
    items, receipt = [], []
    for row in losses:
        shape = key(row)
        m, n, k = shape[2:5]
        control = int(row["kernelId"])
        legal = {kid for kid in union if common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], m, n, k)}
        chosen = {control: {"current_step4_best_control"}}
        for kid in by_nk[(n, k)] & legal:
            chosen.setdefault(kid, set()).add("six_round_winner_at_other_M")
        control_geometry = meta[control]["geometry"]
        neighbors = []
        for kid in legal - previous[shape]:
            geometry = meta[kid]["geometry"]
            if geometry != control_geometry:
                continue
            left, right = meta[kid]["traits"], meta[control]["traits"]
            distance = sum(a != b for a, b in zip(left.split(","), right.split(","))) + abs(len(left.split(",")) - len(right.split(",")))
            neighbors.append((distance, kid))
        for _, kid in sorted(neighbors)[:12]:
            chosen.setdefault(kid, set()).add("same_geometry_unconfirmed_traits_neighbor")
        groups = defaultdict(list)
        for kid in legal:
            geometry = tuple(meta[kid]["geometry"])
            groups[geometry].append(kid)
        geometries = defaultdict(list)
        for geometry in groups:
            family, bm, bn, bk, tm, tn = geometry
            occupancy = math.ceil(m / bm) * math.ceil(n / bn) / 256
            padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
            score = padding - 1 + 2 * max(0, 1 - occupancy) + .1 * (math.ceil(occupancy) / occupancy - 1) + 8 / bm + 8 / bn
            geometries[family].append((score, geometry))
        # Pick at most two geometry groups per pipeline which are absent from
        # the final six-round plan; sample four distinct complete variants.
        added = 0
        for family, options in sorted(geometries.items()):
            used = 0
            for _, geometry in sorted(options):
                unconfirmed = sorted(set(groups[geometry]) - previous[shape] - chosen.keys())
                if not unconfirmed:
                    continue
                indexes = {0, len(unconfirmed) // 3, 2 * len(unconfirmed) // 3, len(unconfirmed) - 1}
                for index in sorted(indexes):
                    chosen.setdefault(unconfirmed[index], set()).add("compiled_geometry_expansion")
                    added += 1
                used += 1
                if used == 2:
                    break
        candidates = [{**meta[kid], "roles": sorted(roles), "exact_opposite_kids": pairs[kid],
                       "us": None, "errRatio": None, "correct": False}
                      for kid, roles in sorted(chosen.items())]
        items.append({"shape": dict(zip(("gfx", "cu_num", "M", "N", "K", "w_scale_block"), shape)),
                      "csv_baseline_us": float(row["csv_baseline_us"]), "remote_winner": None,
                      "candidates": candidates})
        receipt.append({"shape": items[-1]["shape"], "candidates": len(candidates),
                        "new_to_six_round_plan": len(chosen.keys() - previous[shape]),
                        "step4_control_kid": control, "step4_control_us": float(row["us"]),
                        "compiled_legal": len(legal)})
    plan = {"version": 2, "family": "a8w8_blockscale_bpreshuffle", "metadata_module": "prepare_opus_repeat",
            "correctness_limit": .05, "selection": "Cross-M six-round winners, traits neighbors, sampled compiled geometry expansion; two-round screen then fresh six",
            "source_step4": str(FINAL), "shapes": items}
    (OUT / "transfer_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (OUT / "transfer_receipt.json").write_text(json.dumps({"shapes": len(items), "candidate_groups": sum(len(i["candidates"]) for i in items),
             "new_to_six_round_plan_groups": sum(r["new_to_six_round_plan"] for r in receipt), "details": receipt}, indent=2) + "\n")
    print(json.dumps({"shapes": len(items), "candidate_groups": sum(len(i["candidates"]) for i in items), "new_to_six_round_plan_groups": sum(r["new_to_six_round_plan"] for r in receipt)}))


if __name__ == "__main__":
    main()
