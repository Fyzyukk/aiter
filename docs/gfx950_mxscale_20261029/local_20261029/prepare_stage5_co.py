#!/usr/bin/env python3
"""Freeze the current registry and prepare finite new COs in a byte-copy repo."""
import copy
import dataclasses
import hashlib
import importlib.util
import json
import shlex
import shutil
from pathlib import Path

from prepare_co_experiment import load_common, registry, fingerprint, pin_wrapper, b_metadata
from extend_co_experiment import device_identity, transform
from prepare_opus_repeat import csv_rows, key, registry_metadata, traits_args, pure_common

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
OUT = LOCAL / "stage5_optimize160_20261030/co_round3"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    OUT.mkdir()
    document = json.loads((ROOT / REL).read_text())
    baseline_document = copy.deepcopy(document)
    base = load_common(ROOT, label="stage5_base")
    meta, _ = registry_metadata()
    rows = csv_rows(LOCAL / "step4_final_recovery_gpu56_v2_20261030/full421_final_report/remaining_losses_sorted.csv")
    entries = {entry["kid"]: entry for entry in document["gfx950"]}
    owners = {device_identity(entry): entry["kid"] for entry in document["gfx950"]}
    points = []

    def append(entry, parent, reason, variation):
        identity = device_identity(entry)
        if identity in owners:
            return
        kid = 100036 + len(points)
        owners[identity] = kid
        entry["kid"] = kid
        entry.pop("configKey", None)
        suffix = f"pm{int(entry['pad_m'])}_stage5_seed{parent}_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        document["gfx950"].append(entry)
        document["image_suffixes"][str(kid)] = suffix
        points.append({"new_kid": kid, "seed_kid": parent, "reason": reason, "variation": variation,
                       "pipeline": entry["pipeline"]})

    # U4 exists in the C++ traits but all current COs use U2. Cover panel size,
    # padding and GroupM independently using the exact current U2 parent.
    for parent in (387, 388, 389, 390, 391, 392, 393, 394, 395,
                   100000, 100001, 100002, 100003, 100004, 100005, 100024, 100025):
        if parent not in entries:
            continue
        entry, _, change = transform(entries[parent], {"Unroll": 4}, [], parent)
        entry.pop("supported_shapes", None)
        append(entry, parent, "Remaining pin_agpr losses; compare existing U2 with C++-supported U4", change)

    # Pin A/C for the actual narrow-N JIT winners. Copy all cache, prefetch,
    # wave and publication traits; the CO compiler path is an additional
    # difference from JIT and is recorded in the attribution.
    chosen = []
    for row in rows:
        kid = int(row["kernelId"])
        info = meta[kid]
        if (kid in chosen or info["kind"] != "JIT" or int(row["N"]) > 2048
                or info["geometry"][0] not in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap")):
            continue
        chosen.append(kid)
        if len(chosen) == 6:
            break
    defaults = ["2", "0", "256", "false", "false", "false", "0"]
    for parent in chosen:
        info = meta[parent]
        family = info["geometry"][0]
        name, args = traits_args(info["traits"])
        args = list(args)
        if len(args) == 10:
            args += defaults
        if len(args) != (18 if family.endswith("overlap") else 17):
            raise ValueError(f"Unexpected source traits for {parent}")
        for pin_a, pin_c in ((True, False), (False, True), (True, True)):
            variant = args.copy()
            variant[14], variant[15] = str(pin_a).lower(), str(pin_c).lower()
            traits = b_metadata(variant)
            bm, bn, bk = int(variant[0]), int(variant[1]), int(variant[12])
            waves, wg = int(variant[7]), int(variant[8])
            # Both bodies are implemented by the shared b_vgpr pipeline header.
            header = "b_vgpr"
            entry = {"pipeline": family, "block_size": waves * 64, "tile": [bm, bn, bk],
                     "stages": int(variant[2]), "wave_layout": [1, waves],
                     "launch_bounds": [waves * 64, wg], "device_flags": "flags_00",
                     "pad_m": True, "wide_output": True, "max_k": int(variant[6]),
                     "fixed_k": int(variant[6]), "traits": {"name": name, "args": variant},
                     "body_func": "opus_gemm_prefetch_b_vgpr_detail::" + ("bulk_panel_overlap_body" if family.endswith("overlap") else "bulk_panel_body"),
                     "body_args": ["CoTraits"],
                     "pipeline_header": "opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_" + header + "_gfx950.cuh",
                     "grid_layout": "xcd" if variant[5] == "true" else "linear",
                     "pin_variant": True, "pin_mask": int(pin_a) + 4 * int(pin_c)}
            append(entry, parent, "Narrow-N current JIT winner: test pinned A/C at identical complete scheduling traits",
                   {"PinA": pin_a, "PinC": pin_c, "resources": traits,
                    "attribution": "pinning plus isolated CO compiler route; not a pure JIT pin-only comparison"})
    document["jit_kid_base"] = 780
    document["_stage5_experiment"] = {"points": points, "source_co_count": len(entries)}
    assert document["gfx950"][:len(entries)] == baseline_document["gfx950"]
    dump(OUT / "proposed_manifest.json", document)
    extended = load_common(ROOT, manifest=OUT / "proposed_manifest.json", label="stage5_extended")
    left, right = registry(base), registry(extended)
    assert set(right) == set(left) | {point["new_kid"] for point in points}
    digest = hashlib.sha256()
    for kid, instance in left.items():
        assert dataclasses.asdict(instance) == dataclasses.asdict(right[kid]), kid
        digest.update(f"{kid}:{fingerprint(instance)}\n".encode())
    assert base._bpreshuffle_jit_kid_base == extended._bpreshuffle_jit_kid_base == 780
    assert len({instance.name for instance in right.values()}) == len(right)
    for point in points:
        instance = right[point["new_kid"]]
        legal = [key(row)[2:5] for row in rows if extended.a8w8_mxscale_bpreshuffle_supports_shape(instance, *key(row)[2:5])
                 and (point["pipeline"] == "pin_agpr" or int(row["N"]) <= 2048)]
        assert legal, point
        point["shapes"] = [list(shape) for shape in legal]
        point["kernelName"] = instance.name
        entry = next(entry for entry in document["gfx950"] if entry["kid"] == point["new_kid"])
        entry["supported_shapes"] = point["shapes"]
    dump(OUT / "proposed_manifest.json", document)
    repo = OUT / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info", "*.so", "*.o"))
    (repo / REL).write_text((OUT / "proposed_manifest.json").read_text())
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029"
    helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path, helpers / path.name)
    copied = load_common(repo, label="stage5_copied")
    assert all(dataclasses.asdict(instance) == dataclasses.asdict(registry(copied)[kid]) for kid, instance in left.items())
    dump(OUT / "identity_and_points.json", {"previous_ids_verified": len(left), "jit_kid_base": 780,
         "previous_config_digest": digest.hexdigest(), "points": points, "repo": str(repo),
         "original_source_sha256": {str(ROOT / path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                     for path in (REL, Path("csrc/opus_gemm/opus_gemm_common.py"))}})
    env = (f"source {shlex.quote(str(LOCAL / 'env.sh'))}\n"
           f"export OPUS_RUN_REPO={shlex.quote(str(repo))}\nexport PYTHONPATH=\"$OPUS_RUN_REPO\"\n"
           f"export AITER_JIT_DIR={shlex.quote(str(OUT / '.cache/aiter_jit'))}\n"
           "export ROCR_VISIBLE_DEVICES=4,5,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0\n"
           "cd \"$OPUS_RUN_REPO\"\n")
    (OUT / "experiment_env.sh").write_text(env)
    command = ("set -euo pipefail\nsource " + shlex.quote(str(OUT / "experiment_env.sh")) + "\n"
               '"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
               + " ".join(str(point["new_kid"]) for point in points)
               + ' --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --jobs 8 --report '
               + shlex.quote(str(OUT / "new_co_build_info.json")) + "\n")
    (OUT / "build_new_co.sh").write_text(command)
    print(json.dumps({"new_co_count": len(points), "previous_ids_verified": len(left), "repo": str(repo)}))


if __name__ == "__main__":
    main()
