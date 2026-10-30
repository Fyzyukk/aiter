#!/usr/bin/env python3
"""Summarize the final fresh six-round batch and distinguish control drift."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

from prepare_opus_repeat import csv_rows, key, shape_row

FIELDS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    final = campaign / "final160/confirm"
    summary = json.loads((final / "audit/summary.json").read_text())
    assert summary["completed_shapes"] == 160 and summary["pending_shapes"] == 0
    shapes = json.loads((final / "audit/report.json").read_text())["shapes"]
    plan = json.loads((final / "repeat_plan.json").read_text())
    metadata = {(key(item["shape"]), candidate["kid"]): candidate
                for item in plan["shapes"] for candidate in item["candidates"]}
    raw = defaultdict(dict)
    for path in (final / "repeats").glob("gpu*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            identity = key(row["shape"]), row["kid"]
            assert row["round"] not in raw[identity]
            assert row["gpu_pci_bus"] == {5: "0000:95:00.0", 6: "0000:e5:00.0"}[row["gpu_physical_id"]]
            raw[identity][row["round"]] = row
    prior_path = campaign.parent / "step4_final_recovery_gpu56_v2_20261030/full421_final_report/best_opus.csv"
    prior = {key(row): row for row in csv_rows(prior_path)}
    comparisons = []
    current_csv = []
    for row in shapes:
        shape = key(row)
        control = int(prior[shape]["kernelId"])
        winner = int(row["kid"])
        winner_rounds, control_rounds = raw[(shape, winner)], raw[(shape, control)]
        assert set(winner_rounds) == set(control_rounds) == set(range(6))
        assert all(r["correct"] for r in winner_rounds.values()) and all(r["correct"] for r in control_rounds.values())
        control_us = statistics.median(r["us"] for r in control_rounds.values())
        ratios = [control_rounds[r]["us"] / winner_rounds[r]["us"] for r in range(6)]
        faster_rounds = sum(ratio > 1 for ratio in ratios)
        if winner != control:
            assert all((winner_rounds[r]["position"] < control_rounds[r]["position"])
                       != (winner_rounds[r+1]["position"] < control_rounds[r+1]["position"]) for r in (0, 2, 4))
        stable = winner != control and row["us"] < control_us and faster_rounds >= 4
        comparisons.append({**shape_row(shape), "winner_kid": winner, "winner_pipeline": metadata[(shape, winner)]["geometry"][0],
             "winner_us": row["us"], "control_kid": control, "control_us": control_us,
             "step4_us": float(prior[shape]["us"]), "baseline_us": row["csv_baseline_us"],
             "wins_csv": row["wins_csv"], "wins_csv_every_round": row["wins_csv_every_round"],
             "control_wins_csv": control_us < row["csv_baseline_us"],
             "winner_gain_vs_control_pct": (1 - row["us"] / control_us) * 100,
             "paired_median_speedup_pct": (statistics.median(ratios) - 1) * 100,
             "paired_faster_rounds": faster_rounds, "stable_candidate_improvement": stable,
             "new_experimental_co": winner >= 100036,
             "winner_slower_than_csv_pct": (row["us"] / row["csv_baseline_us"] - 1) * 100})
        current_csv.append({**shape_row(shape), "libtype": "opus", "kernelId": winner, "splitK": 1,
             "us": row["us"], "kernelName": row["kernelName"], "errRatio": row["errRatio"],
             "status": row["status"], "csv_baseline_us": row["csv_baseline_us"],
             "wins_csv": row["wins_csv"], "evidence_kind": "stage5_final_fresh_six_round_median",
             "wins_csv_every_round": row["wins_csv_every_round"]})
    assert len(prior) == 421 and len(comparisons) == 160
    kept = [row for shape, row in prior.items() if shape not in {key(row) for row in comparisons}]
    assert len(kept) == 261 and all(row["status"] == "wins_csv" for row in kept)
    all_csv = [{field: row[field] for field in current_csv[0]} for row in kept] + current_csv
    all_csv.sort(key=key)
    wins = sum(row["wins_csv"] for row in comparisons)
    results = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "step4_reference": {"wins": 261, "losses": 160},
        "retained_step4_wins": 261, "fresh_final_retested_shapes": 160,
        "fresh_final_wins": wins, "fresh_final_losses": 160 - wins,
        "combined421_wins": 261 + wins, "combined421_losses": 160 - wins,
        "stable_candidate_improved_shapes": sum(row["stable_candidate_improvement"] for row in comparisons),
        "new_win_with_control_still_slower": sum(row["wins_csv"] and not row["control_wins_csv"] for row in comparisons),
        "wins_where_old_control_also_wins_now": sum(row["wins_csv"] and row["control_wins_csv"] for row in comparisons),
        "experimental_co_selected_shapes": sum(row["new_experimental_co"] for row in comparisons),
        "experimental_co_csv_wins": sum(row["new_experimental_co"] and row["wins_csv"] for row in comparisons),
        "final_audit": summary,
        "timing_definition": "160 original losses: all finalists and original control measured together for six fresh rounds; the prior261 wins are retained without retesting. Control drift is reported separately from candidate gains.",
        "baseline_definition": "Fixed FlyDSL CSV joined by gfx/cu_num/M/N/K/w_scale_block; FlyDSL never tuned.",
        "kernel_repo": str(campaign / "co_round3/.cache/experiment_repo"),
        "run_environment": str(campaign / "co_round3/experiment_env.sh")}
    out = campaign / "result"
    out.mkdir(exist_ok=False)
    dump(out / "metrics.json", results)
    write_csv(out / "final160_comparisons.csv", comparisons)
    write_csv(out / "best_opus421.csv", all_csv)
    losses = sorted([row for row in comparisons if not row["wins_csv"]], key=lambda row: row["winner_slower_than_csv_pct"], reverse=True)
    if losses:
        write_csv(out / "remaining_losses.csv", losses)
    groups = defaultdict(list)
    for row in comparisons:
        groups[(row["N"], row["K"])].append(row)
    group_rows = [{"N": n, "K": k, "original_losses": len(rows),
                   "new_wins": sum(row["wins_csv"] for row in rows),
                   "remaining_losses": sum(not row["wins_csv"] for row in rows),
                   "stable_candidate_gains": sum(row["stable_candidate_improvement"] for row in rows)}
                  for (n, k), rows in sorted(groups.items())]
    write_csv(out / "groups.csv", group_rows)
    # Preserve clearly slow experimental candidates for later review even when
    # they were rejected by screening and never reached the final shortlist.
    experimental_screen = []
    for path in (campaign / "co_round3/run/screen/repeats").glob("gpu*.summary.json"):
        for row in json.loads(path.read_text())["candidates"]:
            if row["kid"] < 100036 or not row["all_rounds_correct"]:
                continue
            experimental_screen.append({**shape_row(key(row["shape"])), "kid": row["kid"],
                "kernelName": row["kernelName"], "two_round_median_us": row["median_us"],
                "baseline_us": row["csv_baseline_us"],
                "slower_than_csv_pct": (row["median_us"] / row["csv_baseline_us"] - 1) * 100,
                "evidence": "two_round_screen_only"})
    if experimental_screen:
        write_csv(out / "new_co_screen_candidates.csv", experimental_screen)
        outliers = [row for row in experimental_screen if row["slower_than_csv_pct"] >= 20]
        if outliers:
            write_csv(out / "new_co_screen_outliers20pct.csv", outliers)
    note = [f"本轮合并结果：{261 + wins} 领先 / {160 - wins} 落后。", "",
        "原来的261个领先项保留；原160个落后项全部做了新的最终六轮确认。没有将不同批次的候选轮次拼接。",
        f"160项中本轮{wins}项快过固定CSV，{results['stable_candidate_improved_shapes']}项的候选稳定优于同批次原最优（至少4/6轮更快）。",
        f"其中{results['new_win_with_control_still_slower']}项是新候选越线且原最优仍落后；{results['wins_where_old_control_also_wins_now']}项在本轮原最优自身也越线，需要与候选优化收益分开看。", "",
        "结果使用隔离实验仓库，环境见 co_round3/experiment_env.sh。原公用注册和第四步正式报告保持可对照。",
        "FlyDSL仅使用固定CSV；A端2GiB寻址范围没有改动。", "",
        "| N | K | 原落后 | 本轮越线 | 仍落后 | 稳定候选收益 |", "|---:|---:|---:|---:|---:|---:|"]
    note += [f"| {row['N']} | {row['K']} | {row['original_losses']} | {row['new_wins']} | {row['remaining_losses']} | {row['stable_candidate_gains']} |" for row in group_rows]
    (out / "result_zh.md").write_text("\n".join(note) + "\n")
    print(json.dumps({k: results[k] for k in ("combined421_wins", "combined421_losses", "stable_candidate_improved_shapes", "experimental_co_selected_shapes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
