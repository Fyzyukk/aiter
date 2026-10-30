#!/usr/bin/env python3
"""Attribute fresh six-round gains to a matched kernel scheduling change."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from prepare_opus_repeat import key, shape_row, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    identity = json.loads((campaign / "co_experiment/identity_and_points.json").read_text())
    points = {p["new_kid"]: p for p in identity["points"]}
    records = defaultdict(dict)
    for path in (campaign / "final/confirm/repeats").glob("gpu*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            records[(key(row["shape"]), row["kid"])][row["round"]] = row
    shapes = {shape for shape, kid in records}
    rows = []
    for pair in identity.get("matched_controls", []):
        a, b = [points[kid] for kid in pair]
        aa, ba = a["axis"].copy(), b["axis"].copy()
        first_a = aa.pop("PrefetchBBeforeLDSReadA")
        first_b = ba.pop("PrefetchBBeforeLDSReadA")
        assert aa == ba and first_a != first_b and a["seed_kid"] == b["seed_kid"]
        after, before = pair if first_a else pair[::-1]
        for shape in sorted(shapes):
            ar, br = records.get((shape, after), {}), records.get((shape, before), {})
            if set(ar) != set(br) or set(ar) != set(range(6)):
                continue
            assert all(r["correct"] and not r["failure"] for r in list(ar.values()) + list(br.values()))
            assert all((ar[i]["position"] < br[i]["position"]) !=
                       (ar[i + 1]["position"] < br[i + 1]["position"]) for i in (0, 2, 4))
            ratios = [br[i]["us"] / ar[i]["us"] for i in range(6)]
            before_us = statistics.median(r["us"] for r in br.values())
            after_us = statistics.median(r["us"] for r in ar.values())
            speedup = (statistics.median(ratios) - 1) * 100
            faster = sum(r > 1 for r in ratios)
            rows.append({**shape_row(shape), "before_kid": before, "after_kid": after,
                "before_us": before_us, "after_us": after_us,
                "paired_median_speedup_pct": speedup, "paired_faster_rounds": faster,
                "stable_material_schedule_gain": speedup >= .5 and faster >= 4 and after_us < before_us,
                "evidence": "same_seed_geometry_stages_depth_and_streaming_six_fresh_rounds"})
    result = campaign / "result"
    if rows:
        write_csv(result / "matched_schedule_comparisons.csv", rows)
    receipt = {"matched_candidate_shape_pairs": len(rows),
        "stable_material_schedule_gains": sum(r["stable_material_schedule_gain"] for r in rows),
        "stable_material_schedule_gain_shapes": len({key(r) for r in rows if r["stable_material_schedule_gain"]}),
        "definition": "Only measured six-round pairs with identical geometry, stages, B depth and streaming; changes only B-before-A issue order. Other combined-candidate gains are separate."}
    (result / "matched_schedule_metrics.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
