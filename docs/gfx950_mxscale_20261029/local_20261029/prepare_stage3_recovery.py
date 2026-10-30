#!/usr/bin/env python3
"""Prepare, optionally run, and merge recovery for missing correct stage3 shapes.

Without --run this program only writes CPU-verified recovery inputs and a
launcher. --run is for root's unified GPU scheduler after the original run
ends. Original measurements remain unchanged; only correct current-identity
records enter profile_confirmable.csv, and all 19 expected shapes must have a
correct OPUS record before that file is published.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

from prepare_opus_repeat import LOCAL, ROOT, KEYS, csv_rows, key, pure_common, shape_row


def write_rows(path, fields, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def correct_records(path, expected, common, registry, err_limit):
    rows = csv_rows(path)
    result, rejected = {}, []
    for line, row in enumerate(rows, 2):
        if row.get("libtype") != "opus":
            continue
        try:
            shape = key(row)
            kid, us, err = int(row["kernelId"]), float(row["us"]), float(row["errRatio"])
        except (KeyError, TypeError, ValueError):
            rejected.append({"line": line, "reason": "invalid shape/ID/time/error"})
            continue
        if shape not in expected:
            raise ValueError(f"{path}:{line}: unexpected shape {shape}")
        if not math.isfinite(us) or us <= 0 or not math.isfinite(err) or not 0 <= err <= err_limit:
            rejected.append({"line": line, "shape": shape_row(shape), "kid": kid, "reason": "failed correctness/time"})
            continue
        if kid not in registry or row.get("kernelName") != registry[kid].name:
            raise ValueError(f"{path}:{line}: current kid/full-name identity mismatch for {kid}")
        if not common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], *shape[2:5]):
            raise ValueError(f"{path}:{line}: illegal current shape/kid contract for {shape}/{kid}")
        result[(shape, kid)] = row  # Last correct observation; never let a dummy overwrite it.
    return rows, result, rejected


def write_launcher(out, geometry, shapes):
    command = ['"$OPUS_RUN_PYTHON"',
               shlex.quote(str(ROOT / "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py")),
               "--mxscale_table --libtype opus --mp 4 --shape_grouped --all",
               "--opus_top_geometries 0 --iters 21 --timeout 7200",
               "--opus_geometry_file", shlex.quote(str(geometry)),
               "-i", shlex.quote(str(shapes)), "-o", shlex.quote(str(out / "tuned.csv")),
               "-o2", shlex.quote(str(out / "profile.csv"))]
    launcher = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                f"source {shlex.quote(str(LOCAL / 'env.sh'))}\n"
                "export AITER_REBUILD=0\n"
                f"export OPUS_TUNE_PROGRESS_FILE={shlex.quote(str(out / 'progress.jsonl'))}\n"
                f"export OPUS_TUNE_CANDIDATE_PROGRESS_DIR={shlex.quote(str(out / 'candidates'))}\n"
                + " ".join(command) + "\n")
    path = out / "run.sh"
    path.write_text(launcher)
    path.chmod(0o755)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=LOCAL / "step3_existing_round1")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--confirmable", type=Path)
    parser.add_argument("--err-ratio", type=float, default=.05)
    parser.add_argument("--run", action="store_true", help="Root only: execute the prepared recovery using GPUs 0-3")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = (args.out_dir or run_dir / "recovery").resolve()
    confirmable = (args.confirmable or run_dir / "profile_confirmable.csv").resolve()
    expected_rows = csv_rows(run_dir / "shapes.csv")
    expected = {key(row) for row in expected_rows}
    if len(expected) != 19 or len(expected_rows) != 19:
        parser.error("Stage3 original shape list must have 19 distinct expected shapes")
    if any(shape[:2] != ("gfx950", 256) or shape[-1] != "128x128" for shape in expected):
        parser.error("This recovery is restricted to gfx950/CU256 NT MXscale128x128")
    original = run_dir / "profile.csv"
    if not original.exists():
        parser.error("Original profile has not been published; wait for the original process to exit")
    common = pure_common()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    original_rows, original_correct, rejected = correct_records(original, expected, common, registry, args.err_ratio)
    original_coverage = {shape for shape, _ in original_correct}
    missing = expected - original_coverage
    receipt = {"source_profile": str(original), "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
               "expected_shapes": len(expected), "original_correct_shapes": len(original_coverage),
               "original_correct_candidates": len(original_correct),
               "recovery_needed_shapes": [shape_row(shape) for shape in sorted(missing)],
               "original_rejected_candidates": len(rejected), "recovery_dir": str(out),
               "confirmable_profile": str(confirmable), "gpu_run_requested": args.run, "gpu_executed": False,
               "timing": "Recovery screen21 iterations; subsequent alternated confirmation101 iterations/6 rounds",
               "correctness_limit": args.err_ratio, "torch_imported": "torch" in sys.modules}
    out.mkdir(parents=True, exist_ok=True)
    recovery_correct = {}
    if missing:
        source_geometry = csv_rows(run_dir / "geometries.csv")
        geometry_rows = [row for row in source_geometry if (int(row["M"]), int(row["N"]), int(row["K"])) in
                         {shape[2:5] for shape in missing}]
        if {tuple(int(row[field]) for field in ("M", "N", "K")) for row in geometry_rows} != {shape[2:5] for shape in missing}:
            raise ValueError("Original geometries do not cover every recovery shape")
        write_rows(out / "shapes.csv", KEYS, [shape_row(shape) for shape in sorted(missing)])
        write_rows(out / "geometries.csv", list(source_geometry[0]), geometry_rows)
        launcher = write_launcher(out, out / "geometries.csv", out / "shapes.csv")
        receipt.update(phase="prepared", launcher=str(launcher), recovery_iters=21, recovery_timeout=7200)
        recovery_profile = out / "profile.csv"
        if not recovery_profile.exists() and args.run:
            if (out / "progress.jsonl").exists() or (out / "tuned.csv").exists() or (out / "tune.log").exists():
                raise ValueError("Partial recovery outputs exist; use a fresh --out-dir to preserve them")
            receipt["gpu_executed"] = True
            (out / "status.json").write_text(json.dumps({**receipt, "phase": "running"}, indent=2) + "\n")
            with (out / "tune.log").open("x") as stream:
                subprocess.run(["bash", str(launcher)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if recovery_profile.exists():
            _, recovery_correct, recovery_rejected = correct_records(recovery_profile, missing, common, registry, args.err_ratio)
            receipt.update(recovery_correct_candidates=len(recovery_correct),
                           recovery_correct_shapes=len({shape for shape, _ in recovery_correct}),
                           recovery_rejected_candidates=len(recovery_rejected),
                           recovery_profile=str(recovery_profile),
                           recovery_source_sha256=hashlib.sha256(recovery_profile.read_bytes()).hexdigest())
        elif not args.run:
            (out / "status.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt, indent=2))
            return
    merged = {**original_correct, **recovery_correct}
    coverage = {shape for shape, _ in merged}
    receipt.update(final_correct_shapes=len(coverage), final_correct_candidates=len(merged),
                   missing_correct_shapes=[shape_row(shape) for shape in sorted(expected - coverage)])
    if coverage != expected:
        receipt["phase"] = "failed_correctness_coverage"
        (out / "status.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise SystemExit("Recovery did not establish correct OPUS coverage for all19 shapes; confirmable profile was not published")
    fields = list(dict.fromkeys(field for row in original_rows + list(recovery_correct.values()) for field in row))
    confirmable.parent.mkdir(parents=True, exist_ok=True)
    write_rows(confirmable, fields, [merged[index] for index in sorted(merged)])
    receipt.update(phase="complete", confirmable_sha256=hashlib.sha256(confirmable.read_bytes()).hexdigest())
    (out / "status.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
