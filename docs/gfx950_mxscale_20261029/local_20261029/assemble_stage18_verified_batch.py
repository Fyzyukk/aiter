#!/usr/bin/env python3
"""Replace six noisy shapes with complete cross-card measurements and re-finalize."""
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_opus_repeat import key

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
STAGE = LOCAL / "stage18_remote_final84_20261030"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_records(folder):
    return [json.loads(line) for path in sorted(folder.glob("gpu*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def main():
    verify = STAGE / "verification_six_shapes/confirm"
    assert json.loads((STAGE / "verification_six_shapes/completed.json").read_text())["exit_code"] == 0
    report = json.loads((verify / "audit/summary.json").read_text())
    assert report["completed_shapes"] == 6 and report["pending_shapes"] == 0
    plan = json.loads((STAGE / "verification_plan.json").read_text())
    affected = {key(item["shape"]) for item in plan["shapes"]}
    replacement = read_records(verify / "repeats")
    assert len(replacement) == sum(len(i["candidates"]) for i in plan["shapes"]) * 6 == 354
    assert all(r["correct"] and not r["failure"] and r["gpu_physical_id"] in (0, 3, 6) for r in replacement)
    snapshot = STAGE / "initial_verification_snapshot"
    original_folder = snapshot / "final_confirm/repeats" if snapshot.exists() else STAGE / "final/confirm/repeats"
    original = read_records(original_folder)
    merged = [r for r in original if key(r["shape"]) not in affected] + replacement
    assert len(original) == len(merged) == 4794
    seen = collections.Counter((key(r["shape"]), r["kid"], r["round"]) for r in merged)
    assert max(seen.values()) == 1

    # Preserve the complete original validation before replacing any measurement.
    if not snapshot.exists():
        snapshot.mkdir()
        for name in ("completion_receipt.json", "public_env.sh", "controller_exit_code"):
            shutil.copy2(STAGE / name, snapshot / name)
        (STAGE / "result").rename(snapshot / "result")
        (STAGE / "integration").rename(snapshot / "integration")
        (STAGE / "final/confirm").rename(snapshot / "final_confirm")
        (STAGE / ".cache/public_jit").rename(STAGE / ".cache/public_jit_initial_verification")
    else:
        assert not (STAGE / "result").exists() and not (STAGE / "integration").exists()
    current = STAGE / "final/confirm"
    current.mkdir(exist_ok=True)
    (current / "repeats").mkdir(exist_ok=True)
    for name in ("repeat_plan.json", "expected.csv", "known.csv", "compile_receipt.json", "pci_preflight.json"):
        shutil.copy2(snapshot / "final_confirm" / name, current / name)
    physical = {0: 0, 1: 3, 2: 6, 3: 5}
    shards = collections.defaultdict(list)
    for row in merged:
        assert physical[row["gpu_logical_id"]] == row["gpu_physical_id"]
        shards[row["gpu_logical_id"]].append(row)
    for logical, rows in shards.items():
        rows.sort(key=lambda r: (*key(r["shape"]), r["round"], r["position"]))
        (current / "repeats" / f"gpu{logical}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    dump(current / "assembly_receipt.json", {"original_final_batch": str(snapshot / "final_confirm"),
         "verification_batch": str(verify), "superseded_records": 354, "replacement_records": 354,
         "retained_original_records": 4440, "selected_final_records": 4794,
         "replaced_shapes": [dict(item["shape"]) for item in plan["shapes"]],
         "selection": "Entire six-round candidate set per affected shape replaced; no minimum-of-batches selection"})
    experiment_helper = Path(os.environ["OPUS_RUN_REPO"]) / "docs/gfx950_mxscale_20261029/local_20261029"
    subprocess.run([sys.executable, str(experiment_helper / "audit_final_repeat.py"), "--plan", str(current / "repeat_plan.json"),
        "--files", str(current / "repeats/gpu*.jsonl"), "--expected", str(current / "expected.csv"),
        "--known-csv", str(current / "known.csv"), "--output-dir", str(current / "audit"),
        "--physical-gpus", "0,3,6,5", "--require-complete"], check=True)

    # Re-run normal stable integration against the preserved original public manifest.
    receipt = json.loads((snapshot / "integration/receipt.json").read_text())
    manifest = ROOT / "csrc/opus_gemm/gen_co/co_kernels_gfx950.json"
    before = snapshot / "integration/public_manifest_before.json"
    assert sha(manifest) == receipt["manifest_after_sha256"]
    assert sha(before) == receipt["manifest_before_sha256"]
    targets = {Path(name): digest for name, digest in receipt["headers"].items()}
    targets.update({Path(r["destination"]): r["sha256"] for r in receipt["images"]})
    for path, digest in targets.items():
        assert sha(path) == digest
    manifest.write_bytes(before.read_bytes())
    for path in targets:
        path.unlink()
    (STAGE / "completion_receipt.json").unlink()
    dump(STAGE / "measurement_verification_assembly.json", {
        "assembled": True, "first_validated_result_preserved": str(snapshot),
        "normal_stable_integration_pending": True, "same_optimization_round": 18,
        "superseded_measurements": 354, "additional_verification_measurements": 354,
        "final_batch_records": 4794, "accepted_screen_and_final_records": 8742,
        "actual_total_performance_measurements": 9096})


if __name__ == "__main__":
    main()
