#!/usr/bin/env python3
"""Publish a usable 421-shape CSV using only available public-registry kernels."""
import argparse
import json
import shlex
import statistics
from collections import defaultdict
from pathlib import Path

from prepare_opus_repeat import csv_rows, key, shape_row, registry_metadata, pure_common, write_csv

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    final = campaign / "final160/confirm"
    result = campaign / "result"
    metadata, pairs = registry_metadata()
    registry = pure_common().get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    raw = defaultdict(list)
    for path in (final / "repeats").glob("gpu*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            raw[(key(row["shape"]), row["kid"])].append(row)
    measured = defaultdict(list)
    for (shape, kid), rounds in raw.items():
        if kid not in metadata or not pure_common().a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], *shape[2:5]):
            continue
        if len(rounds) != 6 or any(not row["correct"] for row in rounds):
            continue
        assert all(row["kernelName"] == metadata[kid]["kernelName"] for row in rounds)
        us = statistics.median(row["us"] for row in rounds)
        baseline = rounds[0]["csv_baseline_us"]
        measured[shape].append({**shape_row(shape), "libtype": "opus", "kernelId": kid, "splitK": 1,
            "us": us, "kernelName": metadata[kid]["kernelName"], "errRatio": max(row["errRatio"] for row in rounds),
            "status": "wins_csv" if us < baseline else "slower_than_csv", "csv_baseline_us": baseline,
            "wins_csv": us < baseline, "evidence_kind": "stage5_final_fresh_six_round_public_registry",
            "wins_csv_every_round": all(row["us"] < baseline for row in rounds)})
    assert len(measured) == 160
    original = csv_rows(campaign.parent / "step4_final_recovery_gpu56_v2_20261030/full421_final_report/best_opus.csv")
    kept = [row for row in original if key(row) not in measured]
    assert len(kept) == 261
    best = {shape: min(rows, key=lambda row: row["us"]) for shape, rows in measured.items()}
    current = list(best.values())
    fields = list(current[0])
    rows = [{field: row[field] for field in fields} for row in kept] + current
    rows.sort(key=key)
    write_csv(result / "public_best_opus421.csv", rows)
    items = []
    for row in rows:
        shape, kid = key(row), int(row["kernelId"])
        assert kid in metadata and row["kernelName"] == metadata[kid]["kernelName"]
        assert pure_common().a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], *shape[2:5])
        items.append({"shape": shape_row(shape), "csv_baseline_us": float(row["csv_baseline_us"]),
            "remote_winner": None, "candidates": [{**metadata[kid], "roles": ["current_step4_best_control"],
            "exact_opposite_kids": pairs[kid], "us": float(row["us"]), "errRatio": float(row["errRatio"]), "correct": True}]})
    plan = {"version": 2, "family": "a8w8_blockscale_bpreshuffle", "metadata_module": "prepare_opus_repeat",
            "correctness_limit": .05, "selection": "Publish available public kernels; identity dry-run and precompile only, no extra benchmark",
            "shapes": items}
    (result / "public421_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    wins = sum(row["wins_csv"] for row in current)
    prior = {key(row): row for row in original}
    comparisons = []
    for shape, row in best.items():
        control_kid = int(prior[shape]["kernelId"])
        control_rounds = sorted(raw[(shape, control_kid)], key=lambda r: r["round"])
        winner_rounds = sorted(raw[(shape, row["kernelId"])], key=lambda r: r["round"])
        control_us = statistics.median(r["us"] for r in control_rounds)
        ratios = [c["us"] / w["us"] for c, w in zip(control_rounds, winner_rounds)]
        faster = sum(ratio > 1 for ratio in ratios)
        stable = row["kernelId"] != control_kid and row["us"] < control_us and faster >= 4
        comparisons.append({**shape_row(shape), "winner_kid": row["kernelId"], "winner_us": row["us"],
            "control_kid": control_kid, "control_us": control_us, "baseline_us": row["csv_baseline_us"],
            "wins_csv": row["wins_csv"], "control_wins_csv": control_us < row["csv_baseline_us"],
            "winner_gain_vs_control_pct": (1-row["us"]/control_us)*100,
            "paired_median_speedup_pct": (statistics.median(ratios)-1)*100,
            "paired_faster_rounds": faster, "stable_candidate_improvement": stable,
            "winner_slower_than_csv_pct": (row["us"]/row["csv_baseline_us"]-1)*100})
    write_csv(result / "public_final160_comparisons.csv", comparisons)
    losses = sorted([row for row in comparisons if not row["wins_csv"]], key=lambda row: row["winner_slower_than_csv_pct"], reverse=True)
    if losses:
        write_csv(result / "public_remaining_losses.csv", losses)
    gains = [row["winner_gain_vs_control_pct"] for row in comparisons if row["stable_candidate_improvement"]]
    public = {"original261_wins_retained": 261, "fresh160_wins": wins, "fresh160_losses": 160-wins,
              "combined421_wins": 261+wins, "combined421_losses": 160-wins,
              "public_registered_count": len(metadata), "new_co_selected_shapes": sum(row["kernelId"] >= 100036 for row in current),
              "stable_candidate_improved_shapes": sum(row["stable_candidate_improvement"] for row in comparisons),
              "new_win_with_control_still_slower": sum(row["wins_csv"] and not row["control_wins_csv"] for row in comparisons),
              "wins_where_old_control_also_wins_now": sum(row["wins_csv"] and row["control_wins_csv"] for row in comparisons),
              "max_stable_candidate_time_reduction_pct": max(gains) if gains else 0,
              "median_stable_candidate_time_reduction_pct": statistics.median(gains) if gains else 0,
              "remaining_losses_median_pct": statistics.median(row["winner_slower_than_csv_pct"] for row in losses) if losses else 0,
              "remaining_losses_max_pct": max((row["winner_slower_than_csv_pct"] for row in losses), default=0),
              "remaining_loss_below_pct": {str(limit): sum(row["winner_slower_than_csv_pct"]<limit for row in losses) for limit in (1,3,5,10)},
              "csv": str(result / "public_best_opus421.csv"), "jit_cache": str(campaign / ".cache/public_jit"),
              "definition": "Stable new COs integrated; weak/non-integrated experimental COs excluded; choose best public candidate from the exact same final six rounds."}
    (result / "public_metrics.json").write_text(json.dumps(public, indent=2) + "\n")
    env = ("source " + shlex.quote(str(LOCAL / "env.sh")) + "\n"
           + "export ROCR_VISIBLE_DEVICES=4,5,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0\n"
           + "export AITER_JIT_DIR=" + shlex.quote(public["jit_cache"]) + "\n")
    (campaign / "public_env.sh").write_text(env)
    print(json.dumps(public))


if __name__ == "__main__":
    main()
