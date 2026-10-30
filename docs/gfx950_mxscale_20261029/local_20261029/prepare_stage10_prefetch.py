#!/usr/bin/env python3
"""Move next B requests before streamed LDS A reads, with matched controls."""
import copy
import dataclasses
import hashlib
import json
import shlex
import shutil
from pathlib import Path

from prepare_co_experiment import load_common, registry, fingerprint, b_metadata
from prepare_opus_repeat import csv_rows, key, registry_metadata, traits_args
from prepare_stage9_compact import verify_schedule

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
OUT = LOCAL / "stage10_prefetch20_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
INCLUDE = Path("csrc/opus_gemm/include/gfx950")
HEADER = "opus_gemm_pipeline_stage10_prefetch_b_gfx950.cuh"
TRAITS = "opus_gemm_traits_stage10_gfx950.cuh"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    reference = LOCAL / "stage9_compact22_20261030/result/public_best_opus421.csv"
    assert (reference.parents[1] / "completion_receipt.json").exists()
    rows = [r for r in csv_rows(reference) if r["status"] == "slower_than_csv"
            and int(r["N"]) <= 2048 and int(r["K"]) == 7168]
    assert len(rows) == 20
    md, _ = registry_metadata()
    before = registry(load_common(ROOT, label="stage10_before"))
    assert len(before) == 48251
    document = json.loads((ROOT / REL).read_text())
    original = copy.deepcopy(document)
    entries = {e["kid"]: e for e in document["gfx950"]}
    points = []

    def append(parent, nb, depth, b_first=True):
        if md[parent]["kind"] == "CO":
            entry = copy.deepcopy(entries[parent])
            args = entry["traits"]["args"].copy()
        else:
            name, values = traits_args(md[parent]["traits"])
            args = list(values)
            entry = {"pipeline": md[parent]["geometry"][0], "traits": {"name": name}}
        args[2] = str(nb)
        if int(args[3]) >= nb:
            args[3] = str(nb - 1)
        args[11], args[14], args[15] = "0", "false", "false"
        resources = b_metadata(args)
        assert depth <= resources["B_MAX_AHEAD"]
        entry["traits"]["args"] = args
        entry.update(block_size=int(args[7]) * 64,
            tile=[int(args[0]), int(args[1]), int(args[12])], wave_layout=[1, int(args[7])],
            stages=nb, launch_bounds=[int(args[7]) * 64, int(args[8])], device_flags="flags_00",
            pad_m=True, wide_output=True, max_k=int(args[6]), fixed_k=int(args[6]),
            traits_header=TRAITS, pipeline_header=HEADER,
            body_func="opus_stage10_b_detail::" + ("bulk_panel_overlap_body" if entry["pipeline"].endswith("overlap") else "bulk_panel_body"),
            body_args=[f"opus_stage10_b_traits<opus_stage7_b_traits<opus_stage6_b_queue_traits<CoTraits, {depth}>, true, true, 0>, {str(b_first).lower()}>"],
            grid_layout="xcd" if args[5] == "true" else "linear", pin_variant=True, pin_mask=0)
        for field in ("kid", "configKey", "supported_shapes"):
            entry.pop(field, None)
        kid = 100500 + len(points)
        entry["kid"] = kid
        document["gfx950"].append(entry)
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
        document["image_suffixes"][str(kid)] = f"pm1_stage10_prefetch_seed{parent}_{digest}"
        schedule = verify_schedule(nb, resources["B_FRAGMENTS"], depth + 1,
            int(args[6]) // int(args[12]), int(args[0]) * int(args[12]) // (int(args[7]) * 64 * 16))
        points.append({"new_kid": kid, "seed_kid": parent, "pipeline": entry["pipeline"],
            "axis": {"AStages": nb, "ActualBAhead": depth, "StreamA": True,
                     "StaticBRing": True, "PrefetchBBeforeLDSReadA": b_first}, "schedule": schedule})

    for nb, depth in ((2, 3), (2, 7), (3, 3), (3, 7), (3, 15), (4, 7), (4, 15)):
        append(41672, nb, depth)
    append(41672, 4, 7, False)  # Exact stage9 winner's schedule, generic shape scope.
    for nb in (2, 3):
        append(41816, nb, 7)
    for depth in (7, 15):
        for first in (False, True):
            append(100224, 2, depth, first)
    for first in (False, True):
        append(41672, 2, 15, first)
    assert len(points) == 16
    OUT.mkdir()
    experiment = OUT / "co_experiment"
    experiment.mkdir()
    dump(experiment / "proposed_manifest.json", document)
    module = load_common(ROOT, manifest=experiment / "proposed_manifest.json", label="stage10_extended")
    after = registry(module)
    assert set(after) == set(before) | {p["new_kid"] for p in points}
    digest = hashlib.sha256()
    for kid, instance in before.items():
        assert dataclasses.asdict(instance) == dataclasses.asdict(after[kid]), kid
        digest.update(f"{kid}:{fingerprint(instance)}\n".encode())
    for point in points:
        instance = after[point["new_kid"]]
        shapes = [list(key(row)[2:5]) for row in rows
                  if module.a8w8_mxscale_bpreshuffle_supports_shape(instance, *key(row)[2:5])]
        assert shapes, point
        point["shapes"], point["kernelName"] = shapes, instance.name
        next(e for e in document["gfx950"] if e["kid"] == point["new_kid"])["supported_shapes"] = shapes
    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    dump(experiment / "proposed_manifest.json", document)
    original_body = (ROOT / INCLUDE / "opus_gemm_pipeline_stage9_compact_b_gfx950.cuh").read_text()
    source = original_body.replace("opus_stage9_b_detail", "opus_stage10_b_detail")
    before_read = """            if constexpr (STREAM_A && n == 0)
                read_a(number<ks>{});"""
    after_read = """            if constexpr (STREAM_A && n == 0 && !T::PREFETCH_B_BEFORE_A)
                read_a(number<ks>{});"""
    assert source.count(before_read) == 1
    source = source.replace(before_read, after_read)
    before_b = """                issue_b(number<((ROTATE_RING ? fragment : absolute) + BD) % BS>{}, number<(fragment + BD) % F>{}, t + (fragment + BD) / F);"""
    assert source.count(before_b) == 1
    source = source.replace(before_b, before_b + """
            // Start the next global B request before the current streamed A
            // LDS reads; the existing VMEM/LDS waits still precede MFMA.
            if constexpr (STREAM_A && n == 0 && T::PREFETCH_B_BEFORE_A)
                read_a(number<ks>{});""")
    traits = '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage7_gfx950.cuh"

template<class Base, bool BFirst>
struct opus_stage10_b_traits : Base {
    static constexpr bool PREFETCH_B_BEFORE_A = BFirst;
};
'''
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    shutil.copy2(experiment / "proposed_manifest.json", repo / REL)
    headers = {HEADER: source, TRAITS: traits}
    for name, body in headers.items():
        (repo / INCLUDE / name).write_text(body)
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029"
    helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path, helpers / path.name)
    dump(experiment / "identity_and_points.json", {"previous_ids_verified": len(before),
        "previous_config_digest": digest.hexdigest(), "jit_kid_base": 780, "repo": str(repo),
        "points": points, "reference": str(reference), "target_only_shapes": len(rows),
        "new_source_changes": [str(INCLUDE / name) for name in headers],
        "new_source_sha256": {name: hashlib.sha256(body.encode()).hexdigest() for name, body in headers.items()},
        "hardware_waits_barriers_descriptors_and_address_widths_preserved": True,
        "matched_controls": [[100505, 100507], [100510, 100511], [100512, 100513], [100514, 100515]]})
    (experiment / "experiment_env.sh").write_text("source " + shlex.quote(str(LOCAL / "env.sh"))
        + "\nexport OPUS_RUN_REPO=" + shlex.quote(str(repo)) + '\nexport PYTHONPATH="$OPUS_RUN_REPO"'
        + "\nexport AITER_JIT_DIR=" + shlex.quote(str(experiment / ".cache/aiter_jit"))
        + '\nexport ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0 OPUS_TARGET_NEW_ONLY=1\ncd "$OPUS_RUN_REPO"\n')
    (experiment / "build_new_co.sh").write_text("set -euo pipefail\nsource " + shlex.quote(str(experiment / "experiment_env.sh"))
        + '\n"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
        + " ".join(str(p["new_kid"]) for p in points)
        + ' --jobs 8 --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report '
        + shlex.quote(str(experiment / "build_info.json")) + "\n")
    print(json.dumps({"new_co_points": len(points), "previous_ids_verified": len(before), "target_shapes": len(rows)}))


if __name__ == "__main__":
    main()
