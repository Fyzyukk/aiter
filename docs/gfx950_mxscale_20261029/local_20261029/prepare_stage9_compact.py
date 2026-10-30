#!/usr/bin/env python3
"""Isolate compact A/B ring periods and a small narrow-N candidate set."""
import copy
import dataclasses
import hashlib
import json
import math
import shlex
import shutil
from pathlib import Path

from prepare_co_experiment import load_common, registry, fingerprint, b_metadata
from prepare_opus_repeat import csv_rows, key, registry_metadata, traits_args

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
OUT = LOCAL / "stage9_compact22_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
INCLUDE = Path("csrc/opus_gemm/include/gfx950")
HEADER = "opus_gemm_pipeline_stage9_compact_b_gfx950.cuh"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_schedule(nb, f, bs, panels, copies):
    """Check live ring contents and vmcnt bounds across all three loop regions.

    Each request is assigned an issue-order tag. A wait may leave at most the
    number of newer requests behind the operand it is about to consume.
    Speculative tail B requests count as requests, as in the unchanged body;
    only in-range fragments are consumed.
    """
    ahead, initial = bs - 1, nb - 1
    assert ahead < panels * f and initial <= panels
    period = math.lcm(nb, bs // math.gcd(bs, f))
    start = min(initial, panels - initial)
    end = start + (panels - initial - start) // period * period
    a_ring, b_ring, last_a, last_b = {}, {}, {}, {}
    issued = 0

    def issue_a(slot, t):
        nonlocal issued
        assert slot == t % nb and t < panels
        a_ring[slot] = t
        issued += copies
        last_a[t] = issued

    def issue_b(slot, absolute, consumed):
        nonlocal issued
        assert slot == absolute % bs
        previous = b_ring.get(slot)
        assert previous is None or previous < consumed or previous >= panels * f
        b_ring[slot] = absolute
        issued += 2
        last_b[absolute] = issued

    for t in range(initial):
        issue_a(t, t)
    for fragment in range(ahead):
        issue_b(fragment % bs, fragment, 0)
    for t in range(panels):
        phase = start + (t - start) % period if start <= t < end else t
        assert phase % nb == t % nb and phase * f % bs == t * f % bs
        assert a_ring[phase % nb] == t
        issued_b = min(phase * f + ahead, panels * f)
        before_a = ahead + (phase - initial) * f + 1
        newer_b_a = issued_b if phase < initial else max(issued_b - before_a, 0)
        newer_a = initial - phase - 1 if phase < initial else min(panels - phase - 1, initial - 1)
        wa = min(newer_b_a * 2 + newer_a * copies, 63)
        assert wa <= issued - last_a[t], (nb, f, bs, panels, t, phase, "wait A")
        for fragment in range(f):
            absolute, real = phase * f + fragment, t * f + fragment
            if absolute + ahead < panels * f:
                issue_b((absolute + ahead) % bs, real + ahead, real)
            assert b_ring[absolute % bs] == real
            newer_b = min(panels * f - absolute - 1, ahead)
            first_a = 0 if absolute < ahead else (absolute - ahead + f - 1) // f
            last = min(phase - (fragment == 0), panels - initial - 1)
            pending_a = max(last - first_a + 1, 0) * copies
            wait = min(newer_b * 2 + pending_a, 63)
            assert wait <= issued - last_b[real], (nb, f, bs, panels, t, phase, fragment, "wait B")
            if phase + initial < panels and fragment == 0:
                issue_a((phase + nb - 1) % nb, t + nb - 1)
    return {"NB": nb, "F": f, "BS": bs, "panels": panels,
            "old_period": math.lcm(nb, bs), "compact_period": period,
            "ring_contents_and_request_wait_bounds_verified": True}


def main():
    # Exhaust boundary/startup/drain schedules, including periods which cannot
    # enter the steady loop. This checks the changed phase reuse independently
    # of compiler output and numerical GPU validation.
    checks = []
    for nb in range(2, 6):
        for f in (1, 2, 3, 4, 6, 8, 16):
            for bs in (2, 4, 8, 12, 16):
                for panels in sorted({nb - 1, nb, 2 * nb, 28, 56, 128}):
                    if bs - 1 < panels * f:
                        checks.append(verify_schedule(nb, f, bs, panels, 8))
    OUT.mkdir()
    experiment = OUT / "co_experiment"
    experiment.mkdir()
    reference = LOCAL / "stage8_longk100_20261030/result/public_best_opus421.csv"
    rows = [r for r in csv_rows(reference) if r["status"] == "slower_than_csv"
            and int(r["N"]) <= 2048 and int(r["K"]) == 7168]
    assert len(rows) == 22
    md, _ = registry_metadata()
    before = registry(load_common(ROOT, label="stage9_before"))
    assert len(before) == 48250
    document = json.loads((ROOT / REL).read_text())
    original = copy.deepcopy(document)
    entries = {e["kid"]: e for e in document["gfx950"]}
    points = []

    def append(parent, nb, depth, stream, wg=None):
        if md[parent]["kind"] == "CO":
            entry = copy.deepcopy(entries[parent])
            args = entry["traits"]["args"].copy()
        else:
            family = md[parent]["geometry"][0]
            name, original_args = traits_args(md[parent]["traits"])
            args = list(original_args)
            entry = {"pipeline": family, "traits": {"name": name}}
        args[2] = str(nb)
        if int(args[3]) >= nb:
            args[3] = str(nb - 1)
        if wg is not None:
            args[8] = str(wg)
        args[11], args[14], args[15] = "0", "false", "false"
        resources = b_metadata(args)
        assert depth <= resources["B_MAX_AHEAD"]
        entry["traits"]["args"] = args
        overlap = entry["pipeline"].endswith("overlap")
        entry.update(block_size=int(args[7]) * 64,
            tile=[int(args[0]), int(args[1]), int(args[12])],
            wave_layout=[1, int(args[7])], stages=nb,
            launch_bounds=[int(args[7]) * 64, int(args[8])], device_flags="flags_00",
            pad_m=True, wide_output=True, max_k=int(args[6]), fixed_k=int(args[6]),
            traits_header="opus_gemm_traits_stage7_gfx950.cuh", pipeline_header=HEADER,
            body_func="opus_stage9_b_detail::" + ("bulk_panel_overlap_body" if overlap else "bulk_panel_body"),
            body_args=[f"opus_stage7_b_traits<opus_stage6_b_queue_traits<CoTraits, {depth}>, {str(stream).lower()}, true, 0>"],
            grid_layout="xcd" if args[5] == "true" else "linear", pin_variant=True, pin_mask=0)
        for field in ("kid", "configKey", "supported_shapes"):
            entry.pop(field, None)
        kid = 100400 + len(points)
        entry["kid"] = kid
        document["gfx950"].append(entry)
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
        document["image_suffixes"][str(kid)] = f"pm1_stage9_compact_seed{parent}_{digest}"
        schedule = verify_schedule(nb, resources["B_FRAGMENTS"], depth + 1,
                                   int(args[6]) // int(args[12]),
                                   int(args[0]) * int(args[12]) // (int(args[7]) * 64 * 16))
        points.append({"new_kid": kid, "seed_kid": parent, "pipeline": entry["pipeline"],
            "axis": {"AStages": nb, "ActualBAhead": depth, "StreamA": stream,
                     "StaticBRing": True, "WGPerCU": int(args[8]), "CompactPeriod": True},
            "schedule": schedule})

    for depth in (3, 7):
        for stream in (False, True):
            append(41816, 4, depth, stream)
        for nb in (2, 3):
            append(41816, nb, depth, True)
        append(41816, 2, depth, True, 2)
        append(41672, 4, depth, True)
        append(100224, 2, depth, True)
        append(48078, 3, depth, False)
    assert len(points) == 16
    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    dump(experiment / "proposed_manifest.json", document)
    module = load_common(ROOT, manifest=experiment / "proposed_manifest.json", label="stage9_extended")
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
    dump(experiment / "proposed_manifest.json", document)
    source = (ROOT / INCLUDE / "opus_gemm_pipeline_stage7_b_vgpr_gfx950.cuh").read_text()
    source = source.replace("opus_stage7_b_detail", "opus_stage9_b_detail")
    old = """        int a = NB, b = BS;
        while (b) { int r = a % b; a = b; b = r; }
        return NB / a * BS;"""
    new = """        // The B ring advances by F fragments per panel. Reuse a static
        // phase once both A stages and B fragment slots repeat.
        int a = BS, b = F;
        while (b) { int r = a % b; a = b; b = r; }
        const int b_panels = BS / a;
        a = NB; b = b_panels;
        while (b) { int r = a % b; a = b; b = r; }
        return NB / a * b_panels;"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    shutil.copy2(experiment / "proposed_manifest.json", repo / REL)
    (repo / INCLUDE / HEADER).write_text(source)
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029"
    helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path, helpers / path.name)
    dump(experiment / "schedule_invariants.json", {"verified_schedules": len(checks),
        "checks": checks, "candidate_schedules": [p["schedule"] for p in points],
        "hardware_waits_barriers_descriptors_and_address_widths_preserved": True})
    dump(experiment / "identity_and_points.json", {"previous_ids_verified": len(before),
        "previous_config_digest": digest.hexdigest(), "jit_kid_base": 780, "repo": str(repo),
        "points": points, "reference": str(reference), "new_source_changes": [str(INCLUDE / HEADER)],
        "new_source_sha256": hashlib.sha256(source.encode()).hexdigest(), "target_only_shapes": 22})
    (experiment / "experiment_env.sh").write_text("source " + shlex.quote(str(LOCAL / "env.sh"))
        + "\nexport OPUS_RUN_REPO=" + shlex.quote(str(repo)) + '\nexport PYTHONPATH="$OPUS_RUN_REPO"'
        + "\nexport AITER_JIT_DIR=" + shlex.quote(str(experiment / ".cache/aiter_jit"))
        + '\nexport ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0 OPUS_TARGET_NEW_ONLY=1\ncd "$OPUS_RUN_REPO"\n')
    (experiment / "build_new_co.sh").write_text("set -euo pipefail\nsource " + shlex.quote(str(experiment / "experiment_env.sh"))
        + '\n"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
        + " ".join(str(p["new_kid"]) for p in points)
        + ' --jobs 8 --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report '
        + shlex.quote(str(experiment / "build_info.json")) + "\n")
    print(json.dumps({"new_co_points": len(points), "previous_ids_verified": len(before),
                     "target_shapes": len(rows), "verified_schedules": len(checks)}))


if __name__ == "__main__":
    main()
