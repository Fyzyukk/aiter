#!/usr/bin/env python3
"""Advance future matrix requests and shorten tail temporaries in pin_agpr."""
import copy
import dataclasses
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path

from prepare_co_experiment import load_common, registry, fingerprint
from prepare_opus_repeat import csv_rows, key
from prepare_stage14_chunk_scale import verify_chunk_order
from prepare_stage12_scale import verify_scale_cache

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
OUT = LOCAL / "stage16_pin_compute54_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
INCLUDE = Path("csrc/opus_gemm/include/gfx950")
HEADER = "opus_gemm_pipeline_stage16_pin_compute_gfx950.cuh"
TRAITS = "opus_gemm_traits_stage16_pin_compute_gfx950.cuh"
SCOPE = "a8w8_blockscale_bpreshuffle"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def compute_body():
    original = (ROOT / INCLUDE / "opus_gemm_pipeline_stage14_chunk_scale_gfx950.cuh").read_text()
    begin = original.index("        // A half 0 x B half 0 -> C[0][0]", original.index("// ===== Main loop ====="))
    end = original.index("        v_sfb[1] = v_sfb_next[1];", begin)
    loop = original[begin:end]
    early_loop = loop
    relocated = {}
    for issue, c_index in zip(range(12, 16), (8, 10, 12, 15)):
        request = f"        issue_matrix_prefetch(opus::number<{issue}>{{}}, stage, future_matrix_offset);\n"
        early_loop = replace_once(early_loop, request, "")
        anchor = f"        mma_scale_group<T, {c_index}, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);\n"
        early_loop = replace_once(early_loop, anchor, anchor + request)
        relocated[issue] = anchor, request
    c11_begin = loop.index("        // A half 1 x B half 1 -> C[1][1]")
    c11 = loop[c11_begin:]
    b_temp = """        const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
            s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
"""
    a_temp = """        const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
            s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
"""
    compact = replace_once(replace_once(c11, b_temp, ""), a_temp, "")
    tail_anchor = "        mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);\n"
    compact = replace_once(compact, tail_anchor, b_temp + a_temp + tail_anchor)

    # Keep each memory handoff and every accumulator's K reduction unchanged.
    # Resolve the four compile-time variants for a dataflow audit below.
    variants = {}
    for early in (False, True):
        for tail in (False, True):
            body = early_loop if early else loop
            if tail:
                body = replace_once(body, c11, compact)
            variants[f"early{int(early)}_compact{int(tail)}"] = verify_compute_order(body)

    emitted = loop
    for issue, (anchor, request) in relocated.items():
        # The existing VMEM group at the old site applies only to its old request.
        emitted = replace_once(emitted, request,
            "        if constexpr (!T::EARLY_MATRIX_PREFETCH) {\n" + request + "        }\n")
        emitted = replace_once(emitted, anchor, anchor
            + "        if constexpr (T::EARLY_MATRIX_PREFETCH) {\n" + request
            + "            __builtin_amdgcn_sched_group_barrier(0x800, 1, 1);\n        }\n")
    c01_begin = emitted.index("        // A half 0 x B half 1 -> C[0][1]")
    c11_begin = emitted.index("        // A half 1 x B half 1 -> C[1][1]")
    c01 = emitted[c01_begin:c11_begin]
    c01 = c01.replace("if constexpr (T::SCHEDULE_POLICY == 0 || (T::SCHEDULE_POLICY == 2 && 0x800 == 0x008))",
        "if constexpr (!T::EARLY_MATRIX_PREFETCH && (T::SCHEDULE_POLICY == 0 || (T::SCHEDULE_POLICY == 2 && 0x800 == 0x008)))")
    emitted = emitted[:c01_begin] + c01 + emitted[c11_begin:]
    emitted = replace_once(emitted, c11,
        "        if constexpr (T::COMPACT_TAIL_STAGING) {\n" + compact
        + "        } else {\n" + c11 + "        }\n")
    source = original[:begin] + emitted + original[end:]
    source = source.replace("opus_stage14_pin_detail", "opus_stage16_pin_detail").replace(
        "opus_stage14_pin_body", "opus_stage16_pin_body")
    waits = r"s_waitcnt_\w+\([^;]*?\)|opus::sync_threads\(\)"
    assert re.findall(waits, original) == re.findall(waits, source)
    return source, variants


def verify_compute_order(body):
    """Every current fragment is consumed before its AGPR slot is replaced."""
    a = [[0] * 4 for _ in range(2)]
    b = [[0] * 4 for _ in range(2)]
    completed = {c: [] for c in ("c00", "c10", "c01", "c11")}
    requests = []
    synchronized = False
    events = []
    for line in body.splitlines():
        if "opus::sync_threads()" in line:
            synchronized = True
        call = re.search(r"mma_scale_group<T, (\d+), (\d+)(?:, [^>]+)?>(.*);", line)
        if call:
            c = re.search(r", (c00|c10|c01|c11),", call[3])[1]
            ah, bh = int(c[1]), int(c[2])
            for ci in range(int(call[1]), int(call[1]) + int(call[2])):
                assert a[ah][ci // 4] == b[bh][ci % 4] == 0, (c, ci, a, b)
                completed[c].append(ci)
                events.append([c, ci])
        req = re.search(r"issue_matrix_prefetch\(opus::number<(\d+)>", line)
        if req:
            assert synchronized
            requests.append(int(req[1]))
        dest = re.search(r"load_operand_fragment_pinned<[^>]+>\(opus::number<\d+>\{\}, (v_a\[(\d+)\]\[(\d+)\]|v_b(?:_second)?\[(\d+)\])", line)
        if dest:
            if dest[2] is not None:
                a[int(dest[2])][int(dest[3])] = 1
            else:
                b[int("second" in dest[1])][int(dest[4])] = 1
        if "v_a[1][3] = a1_m3_next" in line:
            a[1][3] = 1
        if "v_b_second[3] = b1_n3_next" in line:
            b[1][3] = 1
    assert all(sorted(indices) == list(range(16)) for indices in completed.values())
    assert sorted(requests) == list(range(16)) and all(x == 1 for half in a + b for x in half)
    return {"MFMA_updates": len(events), "all_current_operand_uses_before_overwrite": True,
            "each_accumulator_updated_once": True, "future_requests": requests,
            "all_future_requests_after_stage_release_barrier": True}


def main(*, reference=None, first_kid=101100, trait_type="opus_stage16_pin_traits",
         body_func="opus_stage16_pin_body", traits_source=None,
         axis_names=("EarlyMatrixPrefetch", "CompactTailStaging"), experiment_tag="stage16"):
    reference = reference or LOCAL / "stage15_pin55_20261030/result/public_best_opus421.csv"
    assert (reference.parents[1] / "completion_receipt.json").exists()
    targets = {key(r) for r in csv_rows(reference.parent / "public_remaining_losses.csv") if r["winner_pipeline"] == "pin_agpr"}
    rows = [r for r in csv_rows(reference) if key(r) in targets]
    assert len(rows) == len(targets) == 54
    before = registry(load_common(ROOT, label="stage16_before"))
    assert len(before) == 48261
    source, audits = compute_body()
    traits = traits_source or '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage15_pin_unroll_gfx950.cuh"

template<class Base, bool EarlyMatrix, bool CompactTail>
struct opus_stage16_pin_traits : Base {
    static constexpr bool EARLY_MATRIX_PREFETCH = EarlyMatrix;
    static constexpr bool COMPACT_TAIL_STAGING = CompactTail;
};
'''
    document = json.loads((ROOT / REL).read_text())
    original = copy.deepcopy(document)
    entries = {e["kid"]: e for e in document["gfx950"]}
    points, matches = [], []
    for parent in (101011, 101020, 100901, 100906, 100911):
        baseline = None
        for early, tail in ((False, False), (True, False), (False, True), (True, True)):
            entry = copy.deepcopy(entries[parent])
            for field in ("kid", "configKey", "supported_shapes"):
                entry.pop(field, None)
            kid = first_kid + len(points)
            entry["kid"] = kid
            if early or tail:
                entry.update(traits_header=TRAITS, pipeline_header=HEADER, body_func=body_func,
                    body_args=[f"{trait_type}<{entry['body_args'][0]}, {str(early).lower()}, {str(tail).lower()}>"])
            document["gfx950"].append(entry)
            digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
            document["image_suffixes"][str(kid)] = f"pm{int(entry['pad_m'])}_{experiment_tag}_a{int(early)}_b{int(tail)}_seed{parent}_{digest}"
            points.append({"new_kid": kid, "seed_kid": parent, "pipeline": "pin_agpr",
                "axis": {axis_names[0]: early, axis_names[1]: tail},
                "comparison_only": not (early or tail)})
            if not (early or tail):
                baseline = kid
            else:
                matches.append([baseline, kid])
    OUT.mkdir()
    experiment = OUT / "co_experiment"
    experiment.mkdir()
    dump(experiment / "proposed_manifest.json", document)
    after_module = load_common(ROOT, manifest=experiment / "proposed_manifest.json", label="stage16_extended")
    after = registry(after_module)
    assert set(after) == set(before) | {p["new_kid"] for p in points}
    digest = hashlib.sha256()
    for kid, instance in before.items():
        assert dataclasses.asdict(instance) == dataclasses.asdict(after[kid]), kid
        digest.update(f"{kid}:{fingerprint(instance)}\n".encode())
    covered = set()
    for point in points:
        instance = after[point["new_kid"]]
        shapes = [list(key(r)[2:5]) for r in rows if after_module.a8w8_mxscale_bpreshuffle_supports_shape(instance, *key(r)[2:5])]
        assert shapes
        point["shapes"], point["kernelName"] = shapes, instance.name
        covered.update(tuple(s) for s in shapes)
        next(e for e in document["gfx950"] if e["kid"] == point["new_kid"])["supported_shapes"] = shapes
    assert len(covered) == 54
    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    dump(experiment / "proposed_manifest.json", document)
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    shutil.copy2(experiment / "proposed_manifest.json", repo / REL)
    headers = {HEADER: source, TRAITS: traits}
    for name, content in headers.items():
        (repo / INCLUDE / name).write_text(content)
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029"
    helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path, helpers / path.name)
    groups = sorted({int(r["K"]) // 128 for r in rows})
    dump(experiment / "compute_schedule_invariants.json", {"variants": audits,
        "chunk_checks": [verify_chunk_order(panel, k) for panel in (32, 64) for k in groups],
        "scale_checks": [verify_scale_cache(panel, k) for panel in (32, 64) for k in groups],
        "hardware_waits_barriers_descriptors_and_address_widths_preserved": True})
    dump(experiment / "identity_and_points.json", {"previous_ids_verified": len(before),
        "previous_config_digest": digest.hexdigest(), "jit_kid_base": 780, "repo": str(repo),
        "points": points, "reference": str(reference), "target_only_shapes": len(rows),
        "priority_pipeline": "pin_agpr", "compile_family_only": SCOPE,
        "matched_controls": matches, "require_matched_control": True,
        "new_source_changes": [str(INCLUDE / name) for name in headers],
        "new_source_sha256": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in headers.items()},
        "hardware_waits_barriers_descriptors_and_address_widths_preserved": True})
    (experiment / "experiment_env.sh").write_text("source " + shlex.quote(str(LOCAL / "env.sh"))
        + "\nexport OPUS_RUN_REPO=" + shlex.quote(str(repo)) + '\nexport PYTHONPATH="$OPUS_RUN_REPO"'
        + "\nexport AITER_JIT_DIR=" + shlex.quote(str(experiment / ".cache/aiter_jit"))
        + "\nexport OPUS_COMPILE_FAMILY_ONLY=" + SCOPE
        + '\nexport ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0 OPUS_TARGET_NEW_ONLY=1\ncd "$OPUS_RUN_REPO"\n')
    (experiment / "build_new_co.sh").write_text("set -euo pipefail\nsource " + shlex.quote(str(experiment / "experiment_env.sh"))
        + '\n"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
        + " ".join(str(p["new_kid"]) for p in points)
        + ' --jobs 8 --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report '
        + shlex.quote(str(experiment / "build_info.json")) + "\n")
    print(json.dumps({"candidate_points": len(points), "modified_compute_kernels": 15,
        "comparison_only_points": 5, "previous_ids_verified": len(before), "target_shapes": len(covered),
        "compute_dataflow_variants_verified": len(audits), "compile_family_only": SCOPE}), flush=True)


if __name__ == "__main__":
    main()
