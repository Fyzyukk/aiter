#!/usr/bin/env python3
"""Reduce the scale cache around the validated unroll8 pinned MFMA loop."""
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
OUT = LOCAL / "stage12_scale70_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
TRAITS = "opus_gemm_traits_stage12_scale_gfx950.cuh"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_scale_cache(panel, groups):
    """Verify register-current/cache-next handoff at all panel boundaries.

    The pipeline preloads current scales into registers, refills before the
    next scale load, then carries that next value into the following panel.
    Include the final two drained K tiles, partial caches and producer lanes.
    """
    cache = [i if i < groups else None for i in range(panel)]
    current = 0
    boundaries = []
    for tile in range(groups):
        assert current == tile
        next_tile = tile + 1
        if next_tile % panel == 0:
            cache = [next_tile + i if next_tile + i < groups else None for i in range(panel)]
            boundaries.append(next_tile)
        if next_tile < groups:
            assert cache[next_tile % panel] == next_tile
            current = cache[next_tile % panel]
    # SFB stores a four-byte packed scale per producer lane in each N128 half.
    # Guarding lanes >=panel prevents the small cache from receiving aliases.
    sfb_bytes = panel * 2 * 4
    offsets = {(lane * 2 + half) * 4 for half in range(2) for lane in range(panel)}
    assert len(offsets) == panel * 2 and max(offsets) + 4 == sfb_bytes
    assert panel % 16 == 0
    return {"panel": panel, "K128_groups": groups, "refill_boundaries": boundaries,
            "current_register_next_cache_and_sfb_bounds_verified": True,
            "scale_A_cache_bytes": panel * 256, "scale_B_cache_bytes": sfb_bytes}


def main():
    reference = LOCAL / "stage11_unroll76_20261030/result/public_best_opus421.csv"
    assert (reference.parents[1] / "completion_receipt.json").exists()
    rows = [r for r in csv_rows(reference) if r["status"] == "slower_than_csv" and int(r["N"]) >= 6144]
    assert len(rows) == 70
    before = registry(load_common(ROOT, label="stage12_before"))
    assert len(before) == 48254
    document = json.loads((ROOT / REL).read_text())
    original = copy.deepcopy(document)
    entries = {e["kid"]: e for e in document["gfx950"]}
    points, matches = [], []
    for parent in (100602, 100606, 100610, 395):
        group_ids = []
        for panel in (64, 32, 16):
            entry = copy.deepcopy(entries[parent])
            assert entry["pipeline"] == "pin_agpr"
            for field in ("kid", "configKey", "supported_shapes"):
                entry.pop(field, None)
            kid = 100700 + len(points)
            entry.update(kid=kid, traits_header=TRAITS,
                pipeline_header="opus_gemm_pipeline_stage7_pin_agpr_gfx950.cuh",
                body_func="opus_stage7_pin_body",
                body_args=[f"opus_stage12_scale_traits<{entry['body_args'][0]}, {panel}, 8>"])
            document["gfx950"].append(entry)
            digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
            document["image_suffixes"][str(kid)] = f"pm{int(entry['pad_m'])}_stage12_panel{panel}_seed{parent}_{digest}"
            points.append({"new_kid": kid, "seed_kid": parent, "pipeline": "pin_agpr",
                "axis": {"ScalePanelK128": panel, "LoopUnroll": 8, "SchedulePolicy": 0,
                         "GUARD_SFB_LANE": True, "hardware_memory_waits_and_pipeline_unchanged": True},
                "comparison_only": panel == 64})
            group_ids.append(kid)
        matches.extend([[group_ids[0], group_ids[1]], [group_ids[0], group_ids[2]]])
    OUT.mkdir()
    experiment = OUT / "co_experiment"
    experiment.mkdir()
    dump(experiment / "proposed_manifest.json", document)
    module = load_common(ROOT, manifest=experiment / "proposed_manifest.json", label="stage12_extended")
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
    assert len(covered) == 70
    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    dump(experiment / "proposed_manifest.json", document)
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    shutil.copy2(experiment / "proposed_manifest.json", repo / REL)
    traits = '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage11_unroll_gfx950.cuh"

template<class Base, int Panel, int Unroll>
struct opus_stage12_scale_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits<Base, Panel> {
    static_assert(Panel == 16 || Panel == 32 || Panel == 64);
    static_assert(Unroll == 8);
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
    checks = [verify_scale_cache(panel, k) for panel in (16, 32, 64)
              for k in sorted({1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 168}
                              | {int(r['K']) // 128 for r in rows})]
    dump(experiment / "scale_cache_invariants.json", {"verified_schedules": len(checks), "checks": checks})
    dump(experiment / "identity_and_points.json", {"previous_ids_verified": len(before),
        "previous_config_digest": digest.hexdigest(), "jit_kid_base": 780, "repo": str(repo),
        "points": points, "reference": str(reference), "target_only_shapes": len(rows),
        "matched_controls": matches,
        "require_matched_control": True,
        "new_source_changes": ["csrc/opus_gemm/include/gfx950/" + TRAITS],
        "new_source_sha256": {TRAITS: hashlib.sha256(traits.encode()).hexdigest()},
        "comparison_only_definition": "Unroll8/panel64 clones establish the unchanged-scale reference; they can enter final comparisons but cannot be integrated as new kernel improvements.",
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
    print(json.dumps({"candidate_points": len(points), "modified_scale_kernels": 8,
        "comparison_only_points": 4, "previous_ids_verified": len(before),
        "target_shapes": len(covered), "verified_scale_schedules": len(checks)}))


if __name__ == "__main__":
    main()
