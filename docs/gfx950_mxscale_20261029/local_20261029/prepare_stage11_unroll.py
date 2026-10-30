#!/usr/bin/env python3
"""Test compact/long MFMA loop schedules on the remaining wide-N cases."""
import copy
import dataclasses
import hashlib
import json
import shlex
import shutil
from pathlib import Path

from prepare_co_experiment import load_common, registry, fingerprint
from prepare_opus_repeat import csv_rows, key

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
OUT = LOCAL / "stage11_unroll76_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
TRAITS = "opus_gemm_traits_stage11_unroll_gfx950.cuh"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    reference = LOCAL / "stage10_prefetch20_20261030/result/public_best_opus421.csv"
    assert (reference.parents[1] / "completion_receipt.json").exists()
    rows = [r for r in csv_rows(reference) if r["status"] == "slower_than_csv" and int(r["N"]) >= 6144]
    assert len(rows) == 76
    before = registry(load_common(ROOT, label="stage11_before"))
    assert len(before) == 48251
    document = json.loads((ROOT / REL).read_text())
    original = copy.deepcopy(document)
    entries = {e["kid"]: e for e in document["gfx950"]}
    points = []
    # Previous campaigns covered unroll2/4. Unroll1 reduces the loop footprint
    # and scalar lifetimes; unroll8 exposes a longer MFMA scheduling window.
    # Both keep the same actual matrix/scales pipeline and all memory waits.
    for parent in (392, 100083, 100025, 395):
        for unroll in (1, 8):
            for policy in (0, 2):
                entry = copy.deepcopy(entries[parent])
                assert entry["pipeline"] == "pin_agpr"
                for field in ("kid", "configKey", "supported_shapes"):
                    entry.pop(field, None)
                kid = 100600 + len(points)
                entry.update(kid=kid, traits_header=TRAITS,
                    pipeline_header="opus_gemm_pipeline_stage7_pin_agpr_gfx950.cuh",
                    body_func="opus_stage7_pin_body",
                    body_args=[f"opus_stage11_pin_traits<{entry['body_args'][0]}, {unroll}, {policy}>"])
                document["gfx950"].append(entry)
                digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
                document["image_suffixes"][str(kid)] = f"pm{int(entry['pad_m'])}_stage11_u{unroll}_seed{parent}_{digest}"
                points.append({"new_kid": kid, "seed_kid": parent, "pipeline": "pin_agpr",
                    "axis": {"LoopUnroll": unroll, "SchedulePolicy": policy,
                             "hardware_memory_waits_and_pipeline_unchanged": True}})
    OUT.mkdir()
    experiment = OUT / "co_experiment"
    experiment.mkdir()
    dump(experiment / "proposed_manifest.json", document)
    module = load_common(ROOT, manifest=experiment / "proposed_manifest.json", label="stage11_extended")
    after = registry(module)
    assert set(after) == set(before) | {p["new_kid"] for p in points}
    digest = hashlib.sha256()
    for kid, instance in before.items():
        assert dataclasses.asdict(instance) == dataclasses.asdict(after[kid]), kid
        digest.update(f"{kid}:{fingerprint(instance)}\n".encode())
    covered = set()
    for point in points:
        instance = after[point["new_kid"]]
        shapes = [list(key(row)[2:5]) for row in rows
                  if module.a8w8_mxscale_bpreshuffle_supports_shape(instance, *key(row)[2:5])]
        assert shapes, point
        point["shapes"], point["kernelName"] = shapes, instance.name
        covered.update(tuple(s) for s in shapes)
        next(e for e in document["gfx950"] if e["kid"] == point["new_kid"])["supported_shapes"] = shapes
    assert len(covered) == 76
    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    dump(experiment / "proposed_manifest.json", document)
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    shutil.copy2(experiment / "proposed_manifest.json", repo / REL)
    traits = '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage7_gfx950.cuh"

template<class Base, int Unroll, int Policy>
struct opus_stage11_pin_traits : Base {
    static_assert(Unroll == 1 || Unroll == 8);
    static_assert(Policy == 0 || Policy == 2);
    static constexpr int LOOP_UNROLL = Unroll;
    static constexpr int SCHEDULE_POLICY = Policy;
};
'''
    (repo / "csrc/opus_gemm/include/gfx950" / TRAITS).write_text(traits)
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029"
    helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path, helpers / path.name)
    dump(experiment / "identity_and_points.json", {"previous_ids_verified": len(before),
        "previous_config_digest": digest.hexdigest(), "jit_kid_base": 780, "repo": str(repo),
        "points": points, "reference": str(reference), "target_only_shapes": len(rows),
        "new_source_changes": ["csrc/opus_gemm/include/gfx950/" + TRAITS],
        "new_source_sha256": {TRAITS: hashlib.sha256(traits.encode()).hexdigest()},
        "hardware_waits_barriers_descriptors_and_address_widths_preserved": True})
    (experiment / "experiment_env.sh").write_text("source " + shlex.quote(str(LOCAL / "env.sh"))
        + "\nexport OPUS_RUN_REPO=" + shlex.quote(str(repo)) + '\nexport PYTHONPATH="$OPUS_RUN_REPO"'
        + "\nexport AITER_JIT_DIR=" + shlex.quote(str(experiment / ".cache/aiter_jit"))
        + '\nexport ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0 OPUS_TARGET_NEW_ONLY=1\ncd "$OPUS_RUN_REPO"\n')
    (experiment / "build_new_co.sh").write_text("set -euo pipefail\nsource " + shlex.quote(str(experiment / "experiment_env.sh"))
        + '\n"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
        + " ".join(str(p["new_kid"]) for p in points)
        + ' --jobs 8 --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report '
        + shlex.quote(str(experiment / "build_info.json")) + "\n")
    print(json.dumps({"new_co_points": len(points), "previous_ids_verified": len(before), "target_shapes": len(covered)}))


if __name__ == "__main__":
    main()
