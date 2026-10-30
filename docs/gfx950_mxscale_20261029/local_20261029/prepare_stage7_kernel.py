#!/usr/bin/env python3
"""Build isolated kernel-body experiments for the 103 remaining shapes."""
import copy
import dataclasses
import hashlib
import json
import re
import shlex
import shutil
from collections import Counter
from pathlib import Path

from prepare_co_experiment import load_common, registry, fingerprint, b_metadata
from prepare_opus_repeat import csv_rows, key, registry_metadata, traits_args

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
OUT = LOCAL / "stage7_kernel103_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
INCLUDE = Path("csrc/opus_gemm/include/gfx950")
TRAITS = "opus_gemm_traits_stage7_gfx950.cuh"
B_HEADER = "opus_gemm_pipeline_stage7_b_vgpr_gfx950.cuh"
PIN_HEADER = "opus_gemm_pipeline_stage7_pin_agpr_gfx950.cuh"
B_BASE = "opus_gemm_traits_stage7_b_base_gfx950.cuh"
B_ORIGINAL = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh"
PIN_ORIGINAL = "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def kernel_headers():
    # Pad scale-cache rows to an entire packed M64 group. This permits M32
    # tiles without writing packed scale rows outside their LDS allocation.
    base = (ROOT / INCLUDE / "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh").read_text()
    base = base.replace("opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits", "opus_stage7_prefetch_b_base_traits")
    base = replace_once(base,
        "static constexpr int SA_COPIES = (SA_ROWS * SCALE_COLS + BLOCK_SIZE * VEC_SCALE_A - 1) / (BLOCK_SIZE * VEC_SCALE_A);",
        "static constexpr int SA_COPIES = (((BM + 63) / 64 * 64) * SCALE_COLS + BLOCK_SIZE * VEC_SCALE_A - 1) / (BLOCK_SIZE * VEC_SCALE_A);")
    b = (ROOT / INCLUDE / B_ORIGINAL).read_text()
    b = b.replace("opus_gemm_prefetch_b_vgpr_detail", "opus_stage7_b_detail")
    b = replace_once(b, "constexpr int KS = BK / T::W_K;",
        "constexpr int KS = BK / T::W_K;\n    constexpr bool ROTATE_RING = T::ROTATE_B && !T::STATIC_B_RING;\n    constexpr bool STREAM_A = T::ROTATE_B || T::STREAM_A;")
    b = replace_once(b, "if constexpr (T::ROTATE_B) return 1;", "if constexpr (ROTATE_RING) return 1;")
    b = replace_once(b, "const int slot = T::ROTATE_B ? t % NB : phase % NB;", "const int slot = ROTATE_RING ? t % NB : phase % NB;")
    b = b.replace("T::ROTATE_B ? 1 : KS", "STREAM_A ? 1 : KS")
    b = b.replace("T::ROTATE_B ? 0 : ks", "STREAM_A ? 0 : ks")
    b = replace_once(b, "if constexpr (!T::ROTATE_B)", "if constexpr (!STREAM_A)")
    b = b.replace("T::ROTATE_B && n == 0", "STREAM_A && n == 0")
    b = b.replace("T::ROTATE_B ? fragment : absolute", "ROTATE_RING ? fragment : absolute")
    b = b.replace("T::ROTATE_B ? fragment : phase * F + fragment", "ROTATE_RING ? fragment : phase * F + fragment")
    # Constant phase slots repeat at the LCM of A and B ring lengths. A remains
    # streamed one K128 fragment at a time; no B tuple permutation is needed.
    b = replace_once(b, "if constexpr (T::ROTATE_B) {", "if constexpr (ROTATE_RING) {")
    b = b.replace("T::ROTATE_B ? t + NB - 1 : phase + NB - 1", "ROTATE_RING ? t + NB - 1 : phase + NB - 1")
    b = replace_once(b, "if constexpr (T::OUTPUT == 2) {",
        "if constexpr (T::OUTPUT == 3) {\n"
        "        // Each MFMA lane owns four contiguous BF16 values. The LDS\n"
        "        // scatter layout supplies row/column coordinates for direct\n"
        "        // stores; the existing tile descriptor bounds partial M rows.\n"
        "        static_for<MR * NR>([&](auto ci_i) {\n"
        "            constexpr int ci = decltype(ci_i)::value;\n"
        "            const unsigned offset = u_sc.offsets[ci];\n"
        "            const unsigned rr = offset / (BN + T::C_PADDING);\n"
        "            const unsigned cc = offset % (BN + T::C_PADDING);\n"
        "            if (rr < static_cast<unsigned>(valid_rows))\n"
        "                store<4>(g_c, cast<bf16_t>(r_c[ci]), rr * stride_c + cc, 0, number<T::STORE_CACHE>{});\n"
        "        });\n"
        "    } else if constexpr (T::OUTPUT == 2) {")
    # All hardware memory waits, thread barriers, address widths and descriptor
    # bounds in the original body are retained. Only compiler schedule groups
    # are relaxed in the pin experiment.
    pin = (ROOT / INCLUDE / PIN_ORIGINAL).read_text()
    pin = pin.replace("opus_gemm_blockscale_mxscale_bpreshuffle_pin_detail", "opus_stage7_pin_detail")
    pin = pin.replace("void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr(", "void opus_stage7_pin_body(")
    pattern = r"__builtin_amdgcn_sched_group_barrier\((0x[0-9a-fA-F]+), ([0-9]+), ([0-9]+)\);"
    count = len(re.findall(pattern, pin))
    assert count > 60
    pin = re.sub(pattern, lambda m:
        "if constexpr (T::SCHEDULE_POLICY == 0 || (T::SCHEDULE_POLICY == 2 && "
        + m[1] + " == 0x008)) " + m[0], pin)
    wrappers = '''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_stage6_gfx950.cuh"
#include "opus_gemm_traits_stage7_b_base_gfx950.cuh"

template<int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false,
    int FixedK = 7168, int WN = 4, int WGPerCU = 1, int ACache = 0, int StoreCache = 2,
    int Output = 0, int BlockK = 256, bool Contiguous = false,
    bool PinA = false, bool PinC = false, int GroupM = 0, int PublishM = 0>
struct opus_stage7_prefetch_b_overlap_base_traits
    : opus_stage7_prefetch_b_base_traits<BM, BN, NB, BD, BC, XCD, FixedK, WN,
        WGPerCU, ACache, StoreCache, Output, BlockK, Contiguous, PinA, PinC, GroupM> {
    static constexpr int PUBLISH_M = PublishM;
    static_assert(PublishM >= 0 && PublishM < BM / 16 - 1);
};

template<class Base, bool StreamA, bool StaticRing, int Output>
struct opus_stage7_b_traits : Base {
    static constexpr bool STREAM_A = StreamA;
    static constexpr bool STATIC_B_RING = StaticRing;
    static constexpr int OUTPUT = Output < 0 ? Base::OUTPUT : Output;
    static_assert(OUTPUT == 0 || OUTPUT == 2 || OUTPUT == 3);
};

template<class Base, int Policy>
struct opus_stage7_pin_traits : Base {
    static constexpr int SCHEDULE_POLICY = Policy;
    static_assert(Policy >= 0 && Policy <= 2);
};
'''
    return {B_BASE: base, B_HEADER: b, PIN_HEADER: pin, TRAITS: wrappers}, count


def main():
    OUT.mkdir()
    experiment = OUT / "co_experiment"
    experiment.mkdir()
    headers, groups = kernel_headers()
    md, _ = registry_metadata()
    before = registry(load_common(ROOT, label="stage7_before"))
    assert len(before) == 48246
    rows = csv_rows(LOCAL / "stage6_optimize117_20261030/result/public_remaining_losses.csv")
    assert len(rows) == 103
    document = json.loads((ROOT / REL).read_text())
    original = copy.deepcopy(document)
    entries = {entry["kid"]: entry for entry in document["gfx950"]}
    points = []

    def append(entry, parent, axis, target):
        entry = copy.deepcopy(entry)
        for field in ("kid", "configKey", "supported_shapes"):
            entry.pop(field, None)
        kid = 100200 + len(points)
        entry["kid"] = kid
        entry["pin_variant"], entry["pin_mask"] = True, 0
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
        document["gfx950"].append(entry)
        document["image_suffixes"][str(kid)] = f"pm{int(entry['pad_m'])}_stage7_seed{parent}_{digest}"
        points.append({"new_kid": kid, "seed_kid": parent, "axis": axis, "target": target,
                       "pipeline": entry["pipeline"]})

    pin_parents = Counter(int(r["winner_kid"]) for r in rows
        if int(r["K"]) >= 16384 and int(r["winner_kid"]) in entries
        and entries[int(r["winner_kid"])]["pipeline"] == "pin_agpr")
    for parent, _ in pin_parents.most_common():
        for policy in (1, 2):
            entry = copy.deepcopy(entries[parent])
            entry["pipeline_header"] = PIN_HEADER
            entry["traits_header"] = TRAITS
            entry["body_func"] = "opus_stage7_pin_body"
            entry["body_args"] = [f"opus_stage7_pin_traits<{entry['body_args'][0]}, {policy}>"]
            entry["pin_variant"] = False
            append(entry, parent, {"SchedulePolicy": policy, "memory_waits_preserved": True}, "long")

    b_seeds = []
    for row in rows:
        parent = int(row["winner_kid"])
        if parent in b_seeds or md[parent]["kind"] != "JIT" or int(row["N"]) > 2048:
            continue
        if md[parent]["geometry"][0] not in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
            continue
        b_seeds.append(parent)
    for parent in b_seeds:
        info = md[parent]
        family = info["geometry"][0]
        name, args = traits_args(info["traits"])
        args = list(args)
        if len(args) == 10:
            args += ["2", "0", "256", "false", "false", "false", "0"]
        resources = b_metadata(args)
        base_entry = {"pipeline": family, "block_size": int(args[7])*64,
            "tile": [int(args[0]), int(args[1]), int(args[12])], "wave_layout": [1,int(args[7])],
            "stages": int(args[2]), "launch_bounds": [int(args[7])*64,int(args[8])],
            "device_flags": "flags_00", "pad_m": True, "wide_output": True,
            "max_k": int(args[6]), "fixed_k": int(args[6]),
            "traits": {"name": name, "args": args}, "traits_header": TRAITS,
            "pipeline_header": B_HEADER,
            "body_func": "opus_stage7_b_detail::" + ("bulk_panel_overlap_body" if family.endswith("overlap") else "bulk_panel_body"),
            "grid_layout": "xcd" if args[5] == "true" else "linear", "pin_variant": False, "pin_mask": 0}
        variants = [(False, False, 3)]
        if int(args[12]) == 256 and not resources["ROTATE_B"]:
            variants += [(True, False, -1), (True, False, 3)]
        if resources["ROTATE_B"]:
            variants += [(False, True, -1), (False, True, 3)]
        for stream, ring, output in variants:
            entry = copy.deepcopy(base_entry)
            entry["body_args"] = [f"opus_stage7_b_traits<CoTraits, {str(stream).lower()}, {str(ring).lower()}, {output}>"]
            append(entry, parent, {"StreamA": stream, "StaticBRing": ring, "DirectOutput": output == 3}, "narrow")
        if parent == b_seeds[0]:
            for nb in (3, 4):
                variant = args.copy()
                variant[0], variant[1], variant[2], variant[3] = "32", "64", str(nb), "1"
                variant[7], variant[8], variant[14], variant[15] = "2", "2", "false", "false"
                if family.endswith("overlap"):
                    variant[17] = "0"
                entry = copy.deepcopy(base_entry)
                entry.update(block_size=128, tile=[32,64,256], wave_layout=[1,2], stages=nb, launch_bounds=[128,2])
                entry["traits"] = {"name": "opus_stage7_prefetch_b_overlap_base_traits" if family.endswith("overlap") else "opus_stage7_prefetch_b_base_traits", "args": variant}
                entry["body_args"] = ["opus_stage7_b_traits<CoTraits, true, true, 3>"]
                append(entry, parent, {"BM":32,"BN":64,"waves":2,"WGPerCU":2,"AStages":nb,
                    "StreamA":True,"StaticBRing":True,"DirectOutput":True,"padded_M64_scale_cache":True}, "narrow")

    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    document["jit_kid_base"] = 780
    dump(experiment / "proposed_manifest.json", document)
    extended = load_common(ROOT, manifest=experiment / "proposed_manifest.json", label="stage7_extended")
    after = registry(extended)
    assert set(after) == set(before) | {p["new_kid"] for p in points}
    digest = hashlib.sha256()
    for kid, instance in before.items():
        assert dataclasses.asdict(instance) == dataclasses.asdict(after[kid]), kid
        digest.update(f"{kid}:{fingerprint(instance)}\n".encode())
    for point in points:
        instance = after[point["new_kid"]]
        shapes = [list(key(row)[2:5]) for row in rows
            if (int(row["K"]) >= 16384 if point["target"] == "long" else int(row["N"]) <= 2048)
            and extended.a8w8_mxscale_bpreshuffle_supports_shape(instance, *key(row)[2:5])]
        assert shapes, point
        point["shapes"], point["kernelName"] = shapes, instance.name
        entry = next(e for e in document["gfx950"] if e["kid"] == point["new_kid"])
        entry["supported_shapes"] = shapes
    dump(experiment / "proposed_manifest.json", document)
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    shutil.copy2(experiment / "proposed_manifest.json", repo / REL)
    for name, source in headers.items():
        (repo / INCLUDE / name).write_text(source)
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029"
    helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path, helpers / path.name)
    dump(experiment / "identity_and_points.json", {"previous_ids_verified":len(before),
        "previous_config_digest":digest.hexdigest(),"jit_kid_base":780,"repo":str(repo),"points":points,
        "headers":list(headers),"relaxed_compiler_schedule_groups":groups,
        "reference":str(LOCAL / "stage6_optimize117_20261030/result/public_best_opus421.csv")})
    (experiment / "experiment_env.sh").write_text("source " + shlex.quote(str(LOCAL / "env.sh"))
        + "\nexport OPUS_RUN_REPO=" + shlex.quote(str(repo)) + '\nexport PYTHONPATH="$OPUS_RUN_REPO"'
        + "\nexport AITER_JIT_DIR=" + shlex.quote(str(experiment / ".cache/aiter_jit"))
        + '\nexport ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0\ncd "$OPUS_RUN_REPO"\n')
    (experiment / "build_new_co.sh").write_text("set -euo pipefail\nsource " + shlex.quote(str(experiment / "experiment_env.sh"))
        + '\n"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
        + " ".join(str(p["new_kid"]) for p in points) + ' --jobs 8 --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report '
        + shlex.quote(str(experiment / "build_info.json")) + "\n")
    print(json.dumps({"new_kernel_bodies":len(points),"previous_ids_verified":len(before),"shape_targets":dict(Counter(p["target"] for p in points)),"repo":str(repo)}))


if __name__ == "__main__":
    main()
