#!/usr/bin/env python3
"""Prepare isolated shape-specialized pin and explicit B-queue CO experiments."""
import copy
import dataclasses
import hashlib
import json
import shlex
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from prepare_opus_repeat import csv_rows, key, registry_metadata, traits_args
from prepare_co_experiment import load_common, registry, fingerprint, b_metadata

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
SOURCE = LOCAL / "stage5_optimize160_20261030"
OUT = LOCAL / "stage6_optimize117_20261030/co_experiment"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
HEADER = "opus_gemm_traits_stage6_gfx950.cuh"
HEADER_TEXT = '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_overlap_gfx950.cuh"

// Only explicitly scoped COs use these wrappers. Existing pipelines and traits
// retain their definitions, input addressing and output bounds.
template<class Base, int StaticM, int StaticN>
struct opus_stage6_static_shape_traits : Base {
    static_assert(StaticM >= 0 && StaticN > 0);
    static constexpr int STATIC_M = StaticM;
    static constexpr int STATIC_N = StaticN;
};

// Actual fragment queue depth, independent of the BD multiplier's clipping.
// B_SETS changes with B_AHEAD; the existing pipeline recomputes its ring period
// and prologue/steady-state/epilogue waits from these two values.
template<class Base, int Ahead>
struct opus_stage6_b_queue_traits : Base {
    static_assert(Ahead >= 1 && Ahead <= Base::B_MAX_AHEAD);
    static_assert(Ahead < Base::PANELS * Base::B_FRAGMENTS);
    static constexpr int B_AHEAD = Ahead;
    static constexpr int B_SETS = Ahead + 1;
};
'''


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    OUT.mkdir()
    document = json.loads((ROOT / REL).read_text())
    original_doc = copy.deepcopy(document)
    metadata, _ = registry_metadata()
    base = load_common(ROOT, label="stage6_base")
    before = registry(base)
    assert len(before) == 48240
    entries = {entry["kid"]: entry for entry in document["gfx950"]}
    losses = csv_rows(SOURCE / "result/public_remaining_losses.csv")
    best = csv_rows(SOURCE / "result/public_best_opus421.csv")
    points = []

    def append(entry, parent, reason, axis, restrict_n=None, restrict_mnk=None):
        entry = copy.deepcopy(entry)
        entry.pop("kid", None)
        entry.pop("configKey", None)
        entry.pop("supported_shapes", None)
        kid = 100071 + len(points)
        entry["kid"] = kid
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
        document["gfx950"].append(entry)
        document["image_suffixes"][str(kid)] = f"pm{int(entry['pad_m'])}_stage6_seed{parent}_{digest}"
        points.append({"new_kid": kid, "seed_kid": parent, "reason": reason, "axis": axis,
                       "pipeline": entry["pipeline"], "restrict_n": restrict_n, "restrict_mnk": restrict_mnk})

    pin_pairs = Counter((int(row["winner_kid"]), int(row["N"])) for row in losses
                        if metadata[int(row["winner_kid"])]["geometry"][0] == "pin_agpr")
    selected = []
    for n in sorted({n for _, n in pin_pairs}):
        count = 3 if n == 7168 else 2 if n in (1536, 6144) else 1
        selected += [pair for pair, _ in sorted(((pair, freq) for pair, freq in pin_pairs.items() if pair[1] == n),
                                               key=lambda item: item[1], reverse=True)[:count]]
    for parent, n in selected:
        entry = copy.deepcopy(entries[parent])
        entry["body_args"] = [f"opus_stage6_static_shape_traits<{entry['body_args'][0]}, 0, {n}>"]
        entry["traits_header"] = HEADER
        entry["static_m"], entry["static_n"] = 0, n
        append(entry, parent, "Eliminate dynamic N/output stride while preserving current panel, unroll and grouped order",
               {"StaticN": n}, restrict_n=n)
    pin_losses = [row for row in losses if metadata[int(row["winner_kid"])]["geometry"][0] == "pin_agpr"]
    for row in pin_losses[:3]:
        parent = int(row["winner_kid"])
        m, n, k = key(row)[2:5]
        entry = copy.deepcopy(entries[parent])
        entry["body_args"] = [f"opus_stage6_static_shape_traits<{entry['body_args'][0]}, {m}, {n}>"]
        entry["traits_header"] = HEADER
        entry["static_m"], entry["static_n"] = m, n
        append(entry, parent, "Worst current pin loss: eliminate dynamic M/N divisions and descriptor/scale stride inputs",
               {"StaticM": m, "StaticN": n}, restrict_mnk=(m, n, k))

    # A compiler scheduling alternative for long-K pin kernels. This remains a
    # separate axis; shape-specialization and compiler flags are not bundled.
    flags = document["device_flag_sets"]["flags_00"]
    modified = []
    i = 0
    while i < len(flags):
        if flags[i:i+2] == ["-mllvm", "--amdgpu-mfma-vgpr-form"]:
            i += 2
        else:
            modified.append(flags[i]); i += 1
    assert len(modified) == len(flags) - 2
    document["device_flag_sets"]["flags_stage6_agpr_form"] = modified
    long_parents = Counter(int(row["winner_kid"]) for row in pin_losses if int(row["K"]) >= 16384)
    for parent, _ in long_parents.most_common(4):
        entry = copy.deepcopy(entries[parent]); entry["device_flags"] = "flags_stage6_agpr_form"
        append(entry, parent, "Long K: measure MFMA accumulator form and register pressure with the same device algorithm",
               {"mfma_vgpr_form": False})

    # Two remaining worst-case geometries and the successful larger-tile
    # geometry from stage5; include the exact baseline depth as a CO control.
    b_seeds = []
    for row in losses + best:
        parent = int(row.get("winner_kid", row.get("kernelId")))
        info = metadata[parent]
        if (parent in b_seeds or info["kind"] != "JIT" or int(row["N"]) > 2048
                or info["geometry"][0] not in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap")):
            continue
        b_seeds.append(parent)
        if len(b_seeds) == 3:
            break
    for parent in b_seeds:
        info = metadata[parent]
        name, args = traits_args(info["traits"]); args = list(args)
        if len(args) == 10:
            args += ["2", "0", "256", "false", "false", "false", "0"]
        resources = b_metadata(args)
        family = info["geometry"][0]
        bm, bn, bk, wn, wg = (int(args[index]) for index in (0, 1, 12, 7, 8))
        base_entry = {"pipeline": family, "block_size": wn*64, "tile": [bm,bn,bk], "wave_layout": [1,wn],
            "stages": int(args[2]), "launch_bounds": [wn*64,wg], "device_flags": "flags_00",
            "pad_m": True, "wide_output": True, "max_k": int(args[6]), "fixed_k": int(args[6]),
            "traits": {"name": name, "args": args}, "traits_header": HEADER,
            "pipeline_header": "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh",
            "body_func": "opus_gemm_prefetch_b_vgpr_detail::" + ("bulk_panel_overlap_body" if family.endswith("overlap") else "bulk_panel_body"),
            "body_args": ["CoTraits"], "grid_layout": "xcd" if args[5] == "true" else "linear",
            "pin_variant": False, "pin_mask": 0}
        depths = sorted({int(resources["B_AHEAD"]), 3, 7})
        for depth in depths:
            assert depth <= resources["B_MAX_AHEAD"]
            entry = copy.deepcopy(base_entry)
            entry["body_args"] = [f"opus_stage6_b_queue_traits<CoTraits, {depth}>"]
            entry["b_ahead_override"] = depth
            append(entry, parent, "Narrow N: explicit actual B queue depth; baseline-depth CO distinguishes compiler-route effects",
                   {"ActualBAhead": depth, "baseline_ahead": resources["B_AHEAD"]}, restrict_n="narrow")
        entry = copy.deepcopy(base_entry)
        entry["traits"]["args"][11] = "2"
        append(entry, parent, "Narrow N: packed output avoids LDS writeback stripes at otherwise identical scheduling traits",
               {"Output": 2}, restrict_n="narrow")

    document["_stage6_experiment"] = {"source_registry": 48240, "points": points}
    document["jit_kid_base"] = 780
    assert document["gfx950"][:len(entries)] == original_doc["gfx950"]
    dump(OUT / "proposed_manifest.json", document)
    extended = load_common(ROOT, manifest=OUT/"proposed_manifest.json", label="stage6_extended")
    after = registry(extended)
    assert set(after) == set(before) | {point["new_kid"] for point in points}
    digest = hashlib.sha256()
    for kid, instance in before.items():
        assert dataclasses.asdict(instance) == dataclasses.asdict(after[kid]), kid
        digest.update(f"{kid}:{fingerprint(instance)}\n".encode())
    for point in points:
        instance = after[point["new_kid"]]
        shapes = []
        for row in losses:
            mnk = key(row)[2:5]
            if point["restrict_mnk"] and tuple(point["restrict_mnk"]) != mnk:
                continue
            if point["restrict_n"] == "narrow" and mnk[1] > 2048:
                continue
            if isinstance(point["restrict_n"], int) and mnk[1] != point["restrict_n"]:
                continue
            if extended.a8w8_mxscale_bpreshuffle_supports_shape(instance, *mnk):
                shapes.append(list(mnk))
        assert shapes, point
        point["shapes"] = shapes
        point["kernelName"] = instance.name
        entry = next(entry for entry in document["gfx950"] if entry["kid"] == point["new_kid"])
        entry["supported_shapes"] = shapes
    dump(OUT/"proposed_manifest.json", document)
    repo = OUT/".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    (repo/REL).write_text((OUT/"proposed_manifest.json").read_text())
    header_path = repo/"csrc/opus_gemm/include/gfx950"/HEADER
    header_path.write_text(HEADER_TEXT)
    helpers = repo/"docs/gfx950_mxscale_20261029/local_20261029"; helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path, helpers/path.name)
    dump(OUT/"identity_and_points.json", {"previous_ids_verified": 48240, "previous_config_digest": digest.hexdigest(),
         "jit_kid_base": 780, "repo": str(repo), "points": points, "new_header": str(header_path),
         "old_manifest_sha256": hashlib.sha256((ROOT/REL).read_bytes()).hexdigest()})
    env = ("source " + shlex.quote(str(LOCAL/"env.sh")) + "\nexport OPUS_RUN_REPO=" + shlex.quote(str(repo))
       + '\nexport PYTHONPATH="$OPUS_RUN_REPO"\nexport AITER_JIT_DIR=' + shlex.quote(str(OUT/".cache/aiter_jit"))
       + '\nexport ROCR_VISIBLE_DEVICES=4,5,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0\ncd "$OPUS_RUN_REPO"\n')
    (OUT/"experiment_env.sh").write_text(env)
    (OUT/"build_new_co.sh").write_text("set -euo pipefail\nsource " + shlex.quote(str(OUT/"experiment_env.sh"))
       + '\n"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
       + " ".join(str(point["new_kid"]) for point in points) + ' --jobs 8 --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report '
       + shlex.quote(str(OUT/"build_info.json")) + "\n")
    print(json.dumps({"new_co_points": len(points), "previous48240_preserved": True, "repo": str(repo)}))


if __name__ == "__main__":
    main()
