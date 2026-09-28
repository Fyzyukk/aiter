#!/usr/bin/env python3
"""Compare new variants using completed, same-shape, same-GPU measurements.

Positive improvement means lower latency: (1 - variant_us / rival_us) * 100.
Aggregate only dimensionless speedups, never absolute latencies across shapes.
This script only reads measurement results and writes comparison tables.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD_KIDS = {9000, 9010, 9011, 9012, 9020}
RIVALS = ("control", "old_opus", "cktile")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def shape(row):
    return tuple(int(row[key]) for key in ("M", "N", "K"))


def is_control(variant):
    return int(variant['id']) == int(variant['base_kid'])


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def task_comparisons(task):
    run_path = Path(task["run_manifest"])
    if task.get("status") != "passed" or task.get("exit_code") != 0:
        raise ValueError("Every experiment task must finish successfully before comparison")
    if sha256(run_path) != task["run_sha256"]:
        raise ValueError(f"Run manifest changed: {run_path}")
    run = json.loads(run_path.read_text())
    expected = {tuple(s) for s in task["shapes"]}
    if (run.get("status") != "passed" or run.get("rounds") != 5
            or run.get("gpu") != task["gpu"] or run.get("gpu_uuid") != task["gpu_uuid"]
            or run.get("family") != task["family"] or run.get("completed_shapes") != len(expected)
            or {tuple(s) for s in run["shapes"]} != expected):
        raise ValueError(f"Task identity, rounds, or coverage mismatch: {run_path}")
    summary_path = Path(run["outputs"]["summary"])
    if sha256(summary_path) != run["output_sha256"]["summary"]:
        raise ValueError(f"Measured summary changed: {summary_path}")
    by_shape = defaultdict(dict)
    for row in read_csv(summary_path):
        key = shape(row)
        if key not in expected or row["name"] in by_shape[key]:
            raise ValueError("Unexpected shape or duplicate measured candidate")
        us = float(row["median_us"])
        if int(row["samples"]) != run["rounds"] or not math.isfinite(us) or us <= 0:
            raise ValueError("Only complete finite five-round medians can be compared")
        by_shape[key][row["name"]] = dict(row, median_us=us, kid=int(row["kid"]))
    if by_shape.keys() != expected:
        raise ValueError("Measured summary does not cover every assigned shape")
    variants = run["variants"]  # Freeze variant/control metadata to the measured batch.
    controls = {int(v["id"]): v for v in variants if is_control(v)}
    new_variants = [v for v in variants if not is_control(v)]
    if not new_variants:
        raise ValueError("The measured experiment contains no new variants")
    result = []
    for key, rows in by_shape.items():
        def best(lib, kids=None):
            candidates = [r for r in rows.values() if r["lib"] == lib
                          and (kids is None or r["kid"] in kids)]
            if not candidates:
                raise ValueError(f"Missing same-batch {lib} reference for {key}")
            return min(candidates, key=lambda r: (r["median_us"], r["name"]))
        old, cktile = best("opus", OLD_KIDS), best("cktile")
        for variant in new_variants:
            if int(variant["fixed_k"]) != key[2]:
                continue
            control_meta = controls.get(int(variant["base_kid"]))
            if control_meta is None or int(control_meta["fixed_k"]) != key[2]:
                raise ValueError(f"No same-library base_kid control for {variant['name']}")
            measured, control = rows[variant["name"]], rows[control_meta["name"]]
            if (measured["lib"] != "experiment" or control["lib"] != "experiment"
                    or measured["kid"] != int(variant["id"])
                    or control["kid"] != int(variant["base_kid"])):
                raise ValueError("Variant/control metadata differs from measured candidates")
            out = dict(M=key[0], N=key[1], K=key[2], gpu=run["gpu"], gpu_uuid=run["gpu_uuid"],
                       family=run["family"], variant=variant["name"], variant_id=int(variant["id"]),
                       base_kid=int(variant["base_kid"]), variant_us=measured["median_us"])
            for label, rival in (("control", control), ("old_opus", old), ("cktile", cktile)):
                ratio = rival["median_us"] / measured["median_us"]
                out.update({label: rival["name"], f"{label}_us": rival["median_us"],
                            f"vs_{label}_speedup": ratio,
                            f"vs_{label}_improvement_pct": (1 - 1 / ratio) * 100})
            result.append(out)
    return result, expected


def grouped_comparisons(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["K"], row["family"], row["variant"], row["variant_id"], row["base_kid"])].append(row)
    result = []
    for (k, family, name, kid, base_kid), values in sorted(grouped.items()):
        out = dict(K=k, family=family, variant=name, variant_id=kid, base_kid=base_kid, shapes=len(values))
        for rival in RIVALS:
            ratios = [row[f"vs_{rival}_speedup"] for row in values]
            improvements = [row[f"vs_{rival}_improvement_pct"] for row in values]
            geometric_mean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            out.update({
                f"vs_{rival}_geomean_speedup": geometric_mean,
                f"vs_{rival}_geomean_improvement_pct": (1 - 1 / geometric_mean) * 100,
                f"vs_{rival}_min_improvement_pct": min(improvements),
                f"vs_{rival}_max_improvement_pct": max(improvements),
                f"vs_{rival}_faster_shapes": sum(value > 1 for value in ratios),
                f"vs_{rival}_equal_shapes": sum(value == 1 for value in ratios),
                f"vs_{rival}_slower_shapes": sum(value < 1 for value in ratios),
            })
        result.append(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=HERE / "experiments_r5")
    parser.add_argument("--output-dir", type=Path, help="fresh directory; default: <batch>/variant_comparison")
    args = parser.parse_args()
    batch = args.batch.resolve()
    launch_path = batch / "launch.json"
    launch = json.loads(launch_path.read_text())
    if (launch.get("status") != "passed" or launch.get("rounds") != 5
            or launch.get("completed_shapes") != launch.get("shape_count")):
        raise ValueError("Complete all target measurements successfully before summarizing")
    rows, covered = [], set()
    for worker in launch["workers"]:
        for task in worker["tasks"]:
            comparisons, shapes = task_comparisons(task)
            if covered & shapes:
                raise ValueError("Cannot mix repeated measurements of one shape across tasks or GPUs")
            covered |= shapes
            rows.extend(comparisons)
    if covered != {tuple(s) for s in launch["shapes"]} or len(covered) != launch["shape_count"] or not rows:
        raise ValueError("Measured coverage differs from the requested targets")
    rows.sort(key=lambda row: (row["K"], row["N"], row["M"], row["variant_id"]))
    grouped = grouped_comparisons(rows)
    output = (args.output_dir or batch / "variant_comparison").resolve()
    output.mkdir(exist_ok=False)
    write_csv(output / "variants_by_shape.csv", rows)
    write_csv(output / "variants_by_k.csv", grouped)
    report = dict(
        status="passed", launch_manifest=str(launch_path), launch_sha256=sha256(launch_path),
        shapes=len(covered), new_variant_shape_pairs=len(rows), groups=grouped,
        control_policy="Each new variant uses its own base_kid control from the same experiments.so and measurement task",
        reference_policy="Fastest old OPUS and CKTile medians measured on the same shape, GPU, and batch",
        improvement_definition="(1 - variant_us / rival_us) * 100; positive means less time",
        aggregation="Geometric mean of per-shape rival_us/variant_us; no absolute latency pooling",
        output_sha256={p.name: sha256(p) for p in sorted(output.glob("*.csv"))},
    )
    with (output / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output_dir": str(output), "shapes": len(covered),
                      "new_variant_shape_pairs": len(rows), "groups": grouped}, indent=2))


if __name__ == "__main__":
    main()
