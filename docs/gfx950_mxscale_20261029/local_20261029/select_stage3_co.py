#!/usr/bin/env python3
"""Select a bounded CO experiment from completed stage3 alternated medians.

Only creates a JSON plan and CPU validation receipt. No build/GPU invocation.
4wave picks the best panel/PAD/WideOutput parent for each measured M control,
merges identical runtime-K parents, then tests GroupM4/8. B/overlap picks
the best repeated strict CO pair at each narrow-N group's largest current gap,
then tests single-variable XCD, NB, and informative BD points within the budget.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from prepare_co_experiment import (LOCAL, ROOT, MANIFEST_REL, load_common,
                                   registry, prepare_manifest, transform)
from inspect_co_parents import peer_key

FOURWAVE_NK = ((6144, 7168), (7168, 7168), (7168, 16384), (7168, 4096))
NARROW_NK = ((768, 7168), (1536, 7168), (2048, 7168))
B_PIPES = ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=LOCAL / "step3_existing_round1/repeat_report/report.json")
    parser.add_argument("--output", type=Path, default=LOCAL / "selected_co_plan.json")
    parser.add_argument("--max-new-co", type=int, default=24)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["summary"]["pending_shapes"]:
        parser.error("Every selected shape must finish alternated repeats before selecting parents")
    document = json.loads((ROOT / MANIFEST_REL).read_text())
    entries = {entry["kid"]: entry for entry in document["gfx950"]}
    if set(entries) != set(range(780)):
        parser.error("This plan selector targets the preserved original 780 COs")
    common = load_common(ROOT, label="stage3_plan")
    reg = registry(common)
    valid = [row for row in report["candidates"] if row["kind"] == "CO"
             and row["all_rounds_correct"] and row["timing_matches_tuner"]]
    by_shape = defaultdict(dict)
    for row in valid:
        if row["kid"] not in entries or row["kernelName"] != reg[row["kid"]].name:
            raise ValueError("Repeated CO identity differs from preserved source")
        by_shape[(row["M"], row["N"], row["K"])][row["kid"]] = row
    shapes = {(row["M"], row["N"], row["K"]): row for row in report["shapes"]}
    reasons, experiments_by_key, skipped = [], {}, []
    for n, k in FOURWAVE_NK:
        selected = sorted(shape for shape in shapes if shape[1:] == (n, k))
        if not selected:
            reasons.append({"N": n, "K": k, "status": "no_measured_controls"})
            continue
        for shape in selected:
            eligible = [(row["median_us"], kid) for kid, row in by_shape[shape].items()
                        if row["pipeline"] == "pin_agpr"]
            if not eligible:
                raise ValueError(f"No six-round-correct 4wave parent for {shape}")
            parent_us, kid = min(eligible)
            reasons.append({"shape": list(shape), "parent_kid": kid,
                            "parent_median_us": parent_us,
                            "csv_baseline_us": shapes[shape]["csv_baseline_us"],
                            "wins_csv": by_shape[shape][kid]["wins_csv_median"],
                            "policy": "Each measured M uses its own best repeated 4wave parent; identical runtime-K parents are merged"})
            for group in (4, 8):
                point = experiments_by_key.setdefault((kid, group), {"seed_kid": kid,
                    "vary": {"GroupM": group}, "shapes": [], "reason": ""})
                point["shapes"].append(list(shape))
                point["reason"] += (f" {list(shape)}: six-round-correct parent kid{kid} is the fastest 4wave "
                                     f"({parent_us:.6f}us); keep panel/PAD/WideOutput, runtime K and Unroll2; test only GroupM{group}.")
    narrow_loss_groups = sum(any(shape[1:] == nk and row["repeat_status"] == "slower_than_csv"
                                 for shape, row in shapes.items()) for nk in NARROW_NK)
    reserved_b_co = 2 * narrow_loss_groups  # At least one strict pair per current narrow-N loss group.
    parent_budget = (args.max_new_co - reserved_b_co) // 2
    seed_shapes = {seed: {tuple(shape) for point in experiments_by_key.values()
                         if point["seed_kid"] == seed for shape in point["shapes"]}
                   for seed, _ in experiments_by_key}
    chosen_seeds = set()
    # Preserve every represented N/K, favoring the largest-M/longest-K control,
    # before allocating another M parent. This prevents 4wave points exhausting
    # the complete budget and suppressing all narrow-N B/overlap experiments.
    for nk in sorted(FOURWAVE_NK, key=lambda group: (group[1], group[0]), reverse=True):
        covered = lambda seed: any(shape[1:] == nk for shape in seed_shapes[seed])
        if any(covered(seed) for seed in chosen_seeds):
            continue
        candidates = [seed for seed in seed_shapes if covered(seed)]
        if not candidates:
            continue
        if len(chosen_seeds) == parent_budget:
            parser.error("Budget cannot preserve 4wave N/K coverage and the reserved narrow-N strict pairs")
        chosen_seeds.add(max(candidates, key=lambda seed: (max(shape[0] * shape[2] for shape in seed_shapes[seed] if shape[1:] == nk), -seed)))
    remaining_seeds = sorted(set(seed_shapes) - chosen_seeds,
                             key=lambda seed: (max(shape[0] * shape[2] for shape in seed_shapes[seed]),
                                               len(seed_shapes[seed]), -seed), reverse=True)
    chosen_seeds.update(remaining_seeds[:max(0, parent_budget - len(chosen_seeds))])
    experiments = [point for point in experiments_by_key.values() if point["seed_kid"] in chosen_seeds]
    for seed in set(seed_shapes) - chosen_seeds:
        skipped.append({"seed_kid": seed, "shapes": [list(s) for s in sorted(seed_shapes[seed])],
                        "reason": "Finite 4wave budget: preserve large-M/long-K and N/K coverage while reserving one strict B/overlap point per current narrow-N loss group"})
    for reason in reasons:
        if "parent_kid" in reason:
            reason["selected_for_group_tests"] = reason["parent_kid"] in chosen_seeds
    for point in experiments:
        # Each selected runtime-K CO can compare GroupM on every legal M control
        # of its represented N/K groups without adding another device kernel.
        nk_groups = {shape[1:] for shape in seed_shapes[point["seed_kid"]]}
        point["shapes"] = sorted(shape for shape in shapes if shape[1:] in nk_groups
                                 and common.a8w8_mxscale_bpreshuffle_supports_shape(reg[point["seed_kid"]], *shape))
        point["shapes"] = [list(shape) for shape in point["shapes"]]
    paired_groups = defaultdict(lambda: defaultdict(list))
    for kid, entry in entries.items():
        if entry["pipeline"] in B_PIPES:
            paired_groups[peer_key(entry)][entry["pipeline"]].append(kid)
    b_to_overlap = {kid: pipes[B_PIPES[1]][0] for pipes in paired_groups.values()
                    if len(pipes.get(B_PIPES[0], [])) == 1 and len(pipes.get(B_PIPES[1], [])) == 1
                    for kid in pipes[B_PIPES[0]]}
    base_pairs = []
    for n, k in NARROW_NK:
        loss_shapes = sorted(shape for shape, row in shapes.items()
                             if shape[1:] == (n, k) and row["repeat_status"] == "slower_than_csv")
        if not loss_shapes:
            skipped.append({"N": n, "K": k, "reason": "No currently measured narrow-N loss after existing-parameter optimization"})
            continue
        priority = max(loss_shapes, key=lambda shape: shapes[shape]["best_repeat_median_us"] / shapes[shape]["csv_baseline_us"])
        pairs = []
        for b_kid, overlap_kid in b_to_overlap.items():
            if b_kid not in by_shape[priority] or overlap_kid not in by_shape[priority]:
                continue
            if not all(common.a8w8_mxscale_bpreshuffle_supports_shape(reg[kid], *shape)
                       for shape in loss_shapes for kid in (b_kid, overlap_kid)):
                continue
            value = min(by_shape[priority][b_kid]["median_us"], by_shape[priority][overlap_kid]["median_us"])
            pairs.append((value, b_kid, overlap_kid))
        if not pairs:
            skipped.append({"N": n, "K": k, "reason": "No six-round-correct strict B/overlap CO pair in the shortlist"})
            continue
        value, b_kid, overlap_kid = min(pairs)
        base_pairs.append({"b_seed_kid": b_kid, "overlap_seed_kid": overlap_kid,
                           "shapes": [list(shape) for shape in loss_shapes],
                           "priority_shape": list(priority), "pair_best_median_us": value,
                           "baseline_us": shapes[priority]["csv_baseline_us"]})
    proposed = []
    # Round-robin axes preserve cross-group coverage before extra points. JIT
    # already compared NB2/3/4; each CO pair first adds NB3, then actual BD and
    # finally NB2 when legal and within the finite budget.
    for axis in ("XcdOrder", "NB3", "BD", "NB2"):
        for pair in base_pairs:
            entry = entries[pair["b_seed_kid"]]
            traits = entry["traits"]["args"]
            vary = {"XcdOrder": traits[5] != "true"} if axis == "XcdOrder" else \
                   {"NB": 3 if axis == "NB3" else 2} if axis.startswith("NB") else \
                   {"BD": 2 if int(traits[3]) == 1 else 1}
            try:
                _, _, schedule = transform(entry, vary, pair["shapes"], pair["b_seed_kid"])
                transform(entries[pair["overlap_seed_kid"]], vary, pair["shapes"], pair["overlap_seed_kid"])
            except ValueError as exc:
                skipped.append({"axis": axis, "b_seed_kid": pair["b_seed_kid"], "reason": str(exc)})
                continue
            before, after = schedule["before"], schedule["after"]
            proposed.append({"b_seed_kid": pair["b_seed_kid"], "overlap_seed_kid": pair["overlap_seed_kid"],
                "vary": vary, "shapes": pair["shapes"],
                "reason": (f"Six-round-correct strict pair at {pair['priority_shape']}: "
                           f"pair median{pair['pair_best_median_us']:.6f}us vs CSV{pair['baseline_us']:.6f}us; "
                           f"only {axis} changes, actual B_AHEAD {before['B_AHEAD']}->{after['B_AHEAD']}, "
                           f"LDS {before['LDS_BYTES']}->{after['LDS_BYTES']}; geometry/pin/Contiguous/cache/output fixed. "
                           "Ordinary PublishM=-1 and overlap PublishM=0 are measured together.")})
    # Fixed-K parents are runtime-M/N kernels. Identical parent/axis choices in
    # several narrow-N groups need one paired CO point with a union of shapes,
    # rather than repeated device kernels that only differ in host restrictions.
    merged_pairs = {}
    for point in proposed:
        identity = (point["b_seed_kid"], point["overlap_seed_kid"],
                    json.dumps(point["vary"], sort_keys=True))
        if identity not in merged_pairs:
            merged_pairs[identity] = {**point, "shapes": list(point["shapes"])}
        else:
            merged_pairs[identity]["shapes"].extend(point["shapes"])
            merged_pairs[identity]["reason"] += " " + point["reason"]
    unique_pairs = list(merged_pairs.values())
    for point in unique_pairs:
        point["shapes"] = [list(shape) for shape in sorted({tuple(shape) for shape in point["shapes"]})]
    paired_budget = (args.max_new_co - len(experiments)) // 2
    chosen = unique_pairs[:paired_budget]
    skipped.extend({"reason": "finite CO budget", "point": point} for point in unique_pairs[paired_budget:])
    plan = {"source_report": str(args.report.resolve()), "source_status": "completed six-round medians",
            "budget": {"max_new_co": args.max_new_co, "reserved_B_overlap_co": reserved_b_co,
                       "max_4wave_parents": parent_budget, "selected_4wave_parents": sorted(chosen_seeds)},
            "experiments": experiments, "paired_experiments": chosen,
            "parent_selection_evidence": reasons, "skipped_points": skipped}
    _, validation = prepare_manifest(document, plan, args.max_new_co)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    receipt = {"plan": str(args.output), "new_co_count": len(validation),
               "4wave_points": len(experiments), "paired_B_overlap_points": len(chosen),
               "points": validation, "compiled": False, "gpu_used": False}
    args.output.with_suffix(".cpu_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "points"}, indent=2))


if __name__ == "__main__":
    main()
