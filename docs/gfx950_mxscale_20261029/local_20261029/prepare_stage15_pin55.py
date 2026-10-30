#!/usr/bin/env python3
"""Prioritize all 55 remaining pin_agpr cases, including narrow N."""
import copy
import dataclasses
import hashlib
import json
import shlex
import shutil
from pathlib import Path

from prepare_co_experiment import load_common, registry, fingerprint
from prepare_opus_repeat import csv_rows, key
from prepare_stage12_scale import verify_scale_cache

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
OUT = LOCAL / "stage15_pin55_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
TRAITS = "opus_gemm_traits_stage15_pin_unroll_gfx950.cuh"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    reference = LOCAL / "stage14_chunk_scale_20261030/result/public_best_opus421.csv"
    assert (reference.parents[1] / "completion_receipt.json").exists()
    losses = csv_rows(reference.parent / "public_remaining_losses.csv")
    target = {key(r) for r in losses if r["winner_pipeline"] == "pin_agpr"}
    rows = [r for r in csv_rows(reference) if key(r) in target and r["status"] == "slower_than_csv"]
    assert len(rows) == len(target) == 55
    before = registry(load_common(ROOT, label="stage15_before"))
    assert len(before) == 48259
    document = json.loads((ROOT / REL).read_text())
    original = copy.deepcopy(document)
    entries = {e["kid"]: e for e in document["gfx950"]}
    points, matches = [], []
    for parent in (100901, 100906, 100911, 395):
        for panel in (64, 32):
            baseline = None
            for unroll in (8, 4, 2):
                entry = copy.deepcopy(entries[parent])
                for field in ("kid", "configKey", "supported_shapes"):
                    entry.pop(field, None)
                kid = 101000 + len(points)
                entry.update(kid=kid, traits_header=TRAITS,
                    pipeline_header="opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh",
                    body_func="opus_stage14_pin_body",
                    body_args=[f"opus_stage15_pin_traits<{entry['body_args'][0]}, {panel}, {unroll}>"])
                document["gfx950"].append(entry)
                digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
                document["image_suffixes"][str(kid)] = f"pm{int(entry['pad_m'])}_stage15_pin_u{unroll}_p{panel}_seed{parent}_{digest}"
                points.append({"new_kid": kid, "seed_kid": parent, "pipeline": "pin_agpr",
                    "axis": {"ScalePanelK128": panel, "LoopUnroll": unroll, "SchedulePolicy": 0,
                             "LocalSFATemporary": True, "ChunkedScaleRefill": True},
                    "comparison_only": unroll == 8})
                if unroll == 8:
                    baseline = kid
                else:
                    matches.append([baseline, kid])
    OUT.mkdir()
    experiment = OUT / "co_experiment"
    experiment.mkdir()
    dump(experiment / "proposed_manifest.json", document)
    module = load_common(ROOT, manifest=experiment / "proposed_manifest.json", label="stage15_extended")
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
    assert len(covered) == 55
    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    dump(experiment / "proposed_manifest.json", document)
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    shutil.copy2(experiment / "proposed_manifest.json", repo / REL)
    traits = '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage14_chunk_scale_gfx950.cuh"

template<class Base, int Panel, int Unroll>
struct opus_stage15_pin_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits<Base, Panel> {
    static_assert(Panel == 32 || Panel == 64);
    static_assert(Unroll == 2 || Unroll == 4 || Unroll == 8);
    static constexpr bool GUARD_SFB_LANE = true;
    static constexpr int LOOP_UNROLL = Unroll;
    static constexpr int SCHEDULE_POLICY = 0;
};
'''
    (repo / "csrc/opus_gemm/include/gfx950" / TRAITS).write_text(traits)
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029"
    helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path, helpers / path.name)
    dump(experiment / "scale_cache_invariants.json", {"checks": [verify_scale_cache(panel, k)
        for panel in (32, 64) for k in sorted({int(r['K']) // 128 for r in rows})],
        "unchanged_chunk_body": "opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh"})
    dump(experiment / "identity_and_points.json", {"previous_ids_verified": len(before),
        "previous_config_digest": digest.hexdigest(), "jit_kid_base": 780, "repo": str(repo),
        "points": points, "reference": str(reference), "target_only_shapes": len(rows),
        "priority_pipeline": "pin_agpr", "all_other_pipeline_shapes_untested": 30,
        "matched_controls": matches, "require_matched_control": True,
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
    print(json.dumps({"candidate_points": len(points), "modified_pin_kernels": 16,
        "comparison_only_points": 8, "previous_ids_verified": len(before), "target_shapes": len(covered)}))


if __name__ == "__main__":
    main()
