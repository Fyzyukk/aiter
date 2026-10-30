#!/usr/bin/env python3
"""Prepare an isolated, sparse-ID CO experiment; never compile or use a GPU.

Requires an explicit plan based on measured step2 results. The production
source/manifest are only read. A source copy under local_20261029 freezes the
existing JIT base at 780 and appends a few new CO IDs from 100000. Every old
0..48211 ID, complete name and dataclass configuration is compared on CPU.
"""

import argparse
import copy
import csv
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
COMMON_REL = Path("csrc/opus_gemm/opus_gemm_common.py")
MANIFEST_REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")
JIT_BASE_LINE = "_bpreshuffle_jit_kid_base = len(gfx950_co_kernels_declared)"
FROZEN_BASE_LINE = "_bpreshuffle_jit_kid_base = 780  # Frozen original IDs; experimental CO IDs are sparse."
TRAITS_ROOT = "opus_gemm_blockscale_mxscale_bpreshuffle_"
B_FIELDS = ("BM", "BN", "NB", "BD", "CacheB", "XcdOrder", "FixedK", "WaveN", "WGPerCU", "CacheA", "StoreCache", "Output", "BlockK", "Contiguous", "PinA", "PinC", "GroupM", "PublishM")


def load_common(root, *, manifest=None, freeze=False, label="original"):
    source = (root / COMMON_REL).read_text()
    if freeze:
        if source.count(JIT_BASE_LINE) != 1:
            raise ValueError("JIT-base source changed; review explicit base before preparing experiment")
        source = source.replace(JIT_BASE_LINE, FROZEN_BASE_LINE)
    if manifest is not None:
        old = 'GFX950_CO_KERNELS_JSON = os.path.join(\n    os.path.dirname(__file__), "gen_co", "co_kernels_gfx950.json"\n)'
        if source.count(old) != 1:
            raise ValueError("Manifest loader source changed; inspect before CPU override")
        source = source.replace(old, "GFX950_CO_KERNELS_JSON = " + repr(str(manifest)))
    name = "_co_exp_common_" + label
    spec = importlib.util.spec_from_file_location(name, root / COMMON_REL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(root / COMMON_REL), "exec"), module.__dict__)
    return module


def registry(module):
    return module.KERNEL_REGISTRY["a8w8_blockscale_bpreshuffle"]["gfx950"][256]


def fingerprint(instance):
    return hashlib.sha256(json.dumps(dataclasses.asdict(instance), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_identity(original, extended, new_ids):
    left, right = registry(original), registry(extended)
    if set(left) != set(range(48212)):
        raise ValueError(f"Source is not the original 0..48211 registry ({len(left)} entries); review the experiment base")
    if set(right) != set(left) | set(new_ids):
        raise ValueError("Extended registry shifted/dropped original IDs or added unexpected IDs")
    digest = hashlib.sha256()
    for kid, old in left.items():
        if old.name != right[kid].name or dataclasses.asdict(old) != dataclasses.asdict(right[kid]):
            raise ValueError(f"Original identity changed for kid={kid}: {old.name} -> {right[kid].name}")
        digest.update(f"{kid}:{fingerprint(old)}\n".encode())
    names = [instance.name for instance in right.values()]
    if len(names) != len(set(names)):
        raise ValueError("Experimental and original configurations share a complete name")
    return {"original_ids_verified": len(left), "original_id_range": [0, 48211],
            "original_config_digest": digest.hexdigest(), "jit_kid_base": extended._bpreshuffle_jit_kid_base,
            "extended_declared_ids": len(right), "new_ids": list(new_ids),
            "complete_names_unique": True, "comparison": "Every original ID, full name and complete dataclass config equals source"}


def b_metadata(args):
    vals = dict(zip(B_FIELDS, args))
    number = lambda name: int(vals[name])
    boolean = lambda name: str(vals[name]).lower() == "true"
    bm, bn, bk, wn, nb, bd, k, wg = (number(field) for field in ("BM", "BN", "BlockK", "WaveN", "NB", "BD", "FixedK", "WGPerCU"))
    if bm <= 0 or bm % 16 or wn not in (1, 2, 4, 8) or bn <= 0 or bn % (wn * 16) or bn > 256:
        raise ValueError("Invalid B geometry")
    mr, nr, threads = bm // 16, bn // (wn * 16), wn * 64
    if nb not in (2, 3, 4) or bd < 1 or bd >= nb or bk not in (128, 256) or k <= 0 or k % bk or k // bk < nb - 1:
        raise ValueError("Invalid B NB/BD/fixed-K configuration")
    if wg not in (1, 2) or mr * nr * 4 > 256 or bm * bk % (threads * 16):
        raise ValueError("Invalid B workgroup/register geometry")
    if boolean("PinA") and (bk // 128) * mr * 8 > 128 or boolean("PinC") and mr * nr * 4 > 128:
        raise ValueError("Invalid pinned AGPR range")
    scale_cols = k // 128
    sa_bytes = ((bm * scale_cols + threads * 16 - 1) // (threads * 16)) * threads * 16
    panel_pitch = ((mr + 3) // 4) * 16 * 4
    sb_rows = (bn + 127) // 128
    sb_slots = scale_cols if scale_cols % 4 else scale_cols // 4
    lds = nb * bm * bk + sa_bytes + threads * 4
    if panel_pitch * scale_cols > sa_bytes or sb_rows * sb_slots > threads or ((lds + 1279) // 1280 * 1280) * wg > 160 * 1024:
        raise ValueError("Invalid B scale panel/LDS occupancy")
    b_fragments = (bk // 128) * nr
    register_budget = 256 if wg == 2 else 512
    register_room = register_budget - mr * nr * 4 - (bk // 128) * mr * 8 - 48
    max_ahead = max(int(register_room / 8) - 1, 1)
    rotate = wg == 2 or (bk // 128) * mr >= 16
    schedule_ahead = (1 if nb == 2 else 4) if rotate else 3 if bk == 128 and nr >= 4 and boolean("PinA") and k >= 1536 else max_ahead
    ahead = min(bd * b_fragments, schedule_ahead)
    return {**vals, "MR": mr, "NR": nr, "B_FRAGMENTS": b_fragments,
            "ROTATE_B": rotate, "B_REGISTER_ROOM": register_room, "B_MAX_AHEAD": max_ahead,
            "B_SCHEDULE_AHEAD": schedule_ahead, "B_AHEAD": ahead, "B_SETS": ahead + 1, "LDS_BYTES": lds}


def pin_wrapper(entry):
    root = TRAITS_ROOT + "4wave_pin_agpr_traits"
    match = re.fullmatch(re.escape(root) + r"<([^<>]+)>", entry["body_args"][0])
    if not match:
        raise ValueError("Unknown 4wave wrapper; do not infer GroupM slot")
    args = [value.strip() for value in match[1].split(",")]
    defaults = ["CoTraits", TRAITS_ROOT + "pin_matrix_view::tile", TRAITS_ROOT + "pin_output::lds", "false", "false", "0"]
    if not 1 <= len(args) <= len(defaults) or args[0] != "CoTraits":
        raise ValueError("Unexpected 4wave wrapper arguments")
    args.extend(defaults[len(args):])
    if args[1:3] != defaults[1:3] or args[3].lower() != "false" or args[4].lower() not in ("false", "true"):
        raise ValueError("Only existing tile/LDS 4wave route is supported by this experiment generator")
    if (args[4].lower() == "true") != bool(entry.get("wide_output", False)):
        raise ValueError("4wave WideOutput metadata differs from the actual wrapper")
    return root, args


def transform(entry, vary, shapes, seed_kid, allow_equal=False):
    if len(vary) != 1:
        raise ValueError("Each experimental point must change exactly one explicit variable; no Cartesian expansion")
    new = copy.deepcopy(entry)
    new.pop("kid", None)
    new.pop("configKey", None)
    axis, value = next(iter(vary.items()))
    pipeline = new["pipeline"]
    if pipeline == "pin_agpr":
        if axis != "GroupM" or type(value) is not int or value not in (4, 8):
            raise ValueError("4wave experiments only add explicit GroupM4/8")
        root, before_args = pin_wrapper(new)
        after_args = list(before_args)
        after_args[-1] = str(value)
        if before_args == after_args:
            raise ValueError("Requested GroupM is unchanged")
        new["body_args"][0] = root + "<" + ", ".join(after_args) + ">"
        before = {"GroupM": int(before_args[-1]), "WideOutput": before_args[4].lower() == "true",
                  "panel_traits": list(new["traits"]["args"]), "runtime_K": True}
        after = {**before, "GroupM": value}
        suffix = f"exp_nt_seed{seed_kid}_sp{new['traits']['args'][0]}_u{new['traits']['args'][1]}_pm{int(new['pad_m'])}_wide{int(before['WideOutput'])}_gm{value}"
    elif pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        if axis not in ("NB", "BD", "XcdOrder"):
            raise ValueError("B/overlap points only change NB, BD or XcdOrder")
        if axis == "XcdOrder" and type(value) is not bool or axis != "XcdOrder" and type(value) is not int:
            raise ValueError("Use a JSON boolean for XcdOrder and integers for NB/BD")
        args = list(new["traits"]["args"])
        required = 18 if pipeline == "prefetch_b_vgpr_overlap" else 17
        if len(args) != required:
            raise ValueError("Expected canonical 17/18-parameter pinned B/overlap traits")
        before = b_metadata(args)
        slot = {"NB": 2, "BD": 3, "XcdOrder": 5}[axis]
        encoded = str(value).lower()
        if str(args[slot]).lower() == encoded:
            raise ValueError(f"Requested {axis} is unchanged")
        args[slot] = encoded
        after = b_metadata(args)
        if axis == "BD" and before["B_AHEAD"] == after["B_AHEAD"] and not allow_equal:
            raise ValueError("BD is clipped to the same actual B_AHEAD; select an informative seed or explicitly allow_equal_b_ahead")
        if pipeline == "prefetch_b_vgpr_overlap" and not 0 <= int(args[17]) < int(args[0]) // 16 - 1:
            raise ValueError("Overlap PublishM outside legal M repeat range")
        new["traits"]["args"] = args
        new["stages"] = int(args[2])
        new["grid_layout"] = "xcd" if args[5] == "true" else "linear"
        pub = args[17] if len(args) == 18 else "minus1"
        suffix = (f"exp_nt_seed{seed_kid}_p{args[2]}_bd{args[3]}_ba{after['B_AHEAD']}_xcd{int(args[5]=='true')}_contig{int(args[13]=='true')}"
                  f"_k{args[6]}_gm{args[16]}_ca{args[9]}_cb{args[4]}_pinA{int(args[14]=='true')}C{int(args[15]=='true')}_pub{pub}")
    else:
        raise ValueError(f"Unsupported experimental pipeline {pipeline}")
    if shapes:
        new["supported_shapes"] = sorted({tuple(int(v) for v in shape) for shape in shapes})
        new["supported_shapes"] = [list(shape) for shape in new["supported_shapes"]]
    new_hash = hashlib.sha256(json.dumps(new, sort_keys=True).encode()).hexdigest()[:12]
    return new, suffix + "_h" + new_hash, {"axis": axis, "before": before, "after": after,
                                          "strict_pair_note": "NB may also change actual B_AHEAD; compare before/after fields before attributing a pure stage effect" if axis == "NB" else None}


def prepare_manifest(original_doc, plan, max_new):
    seeds = {row["kid"]: row for row in original_doc["gfx950"]}
    if set(seeds) != set(range(780)):
        raise ValueError("Original manifest must contain the preserved 780 CO IDs")
    expanded = [(copy.deepcopy(item), None) for item in plan.get("experiments", [])]
    for index, pair in enumerate(plan.get("paired_experiments", [])):
        b, overlap = seeds[pair["b_seed_kid"]], seeds[pair["overlap_seed_kid"]]
        if b["pipeline"] != "prefetch_b_vgpr" or overlap["pipeline"] != "prefetch_b_vgpr_overlap":
            raise ValueError("Paired seeds must be ordinary B and overlap")
        if b["traits"]["args"] != overlap["traits"]["args"][:17]:
            raise ValueError("B/overlap seed pair differs in complete base traits; never compare bundled changes as a strict pair")
        for field in ("tile", "wave_layout", "launch_bounds", "device_flags", "grid_layout", "pin_mask", "pad_m", "wide_output", "fixed_k", "max_k"):
            if b.get(field) != overlap.get(field):
                raise ValueError(f"B/overlap seed pair differs in {field}")
        group = f"paired_point_{index}"
        for seed in (pair["b_seed_kid"], pair["overlap_seed_kid"]):
            expanded.append(({**pair, "seed_kid": seed}, group))
    if not expanded or len(expanded) > max_new:
        raise ValueError(f"Plan must explicitly contain 1..{max_new} new COs, got {len(expanded)}")
    doc, records = copy.deepcopy(original_doc), []
    for index, (point, group) in enumerate(expanded):
        if not point.get("reason"):
            raise ValueError("Every point needs a measured-result selection reason")
        seed_kid = int(point["seed_kid"])
        if seed_kid not in seeds:
            raise ValueError(f"Unknown original seed kid={seed_kid}")
        shapes = point.get("shapes", [])
        if not shapes or any(len(shape) != 3 or min(map(int, shape)) <= 0 for shape in shapes):
            raise ValueError("Every point requires explicit positive M,N,K shapes from current measured losses")
        entry, suffix, metadata = transform(seeds[seed_kid], point["vary"], shapes, seed_kid, point.get("allow_equal_b_ahead", False))
        new_kid = 100000 + index
        entry["kid"] = new_kid
        doc["gfx950"].append(entry)
        doc["image_suffixes"][str(new_kid)] = suffix
        records.append({"new_kid": new_kid, "seed_kid": seed_kid, "pipeline": entry["pipeline"],
                        "paired_point": group, "reason": point["reason"], "shapes": shapes, **metadata})
    doc["jit_kid_base"] = 780
    doc["_experimental_extensions"] = {"sparse_id_start": 100000, "source_co_count": 780, "points": records}
    if doc["gfx950"][:780] != original_doc["gfx950"] or any(doc["image_suffixes"][str(k)] != original_doc["image_suffixes"][str(k)] for k in seeds):
        raise ValueError("Original manifest entries/suffixes changed")
    return doc, records


def mechanism_plan():
    # CPU validation fixtures only, never selected performance experiments.
    return {"experiments": [{"seed_kid": 0, "vary": {"GroupM": 4}, "shapes": [[8192, 6144, 7168]], "reason": "CPU mechanism fixture; not a performance selection"}],
            "paired_experiments": [{"b_seed_kid": 451, "overlap_seed_kid": 643, "vary": {"BD": 2}, "shapes": [[1280, 7168, 384]], "reason": "CPU strict-pair fixture; not a performance selection"}]}


def write_inputs(out, declared, records):
    shapes, geometries = set(), set()
    for record in records:
        entry = declared[record["new_kid"]]
        for shape in record["shapes"]:
            mnk = tuple(map(int, shape))
            shapes.add(("gfx950", 256, *mnk, "128x128"))
            geometries.add((*mnk, entry["pipeline"], *entry["tile"], *entry["wave_layout"]))
    for filename, fields, rows in (("experiment_shapes.csv", ("gfx", "cu_num", "M", "N", "K", "w_scale_block"), shapes),
                                   ("experiment_geometries.csv", ("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN"), geometries)):
        with (out / filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(sorted(rows))


def write_launchers(out, new_ids):
    repo = out / ".cache/experiment_repo"
    env = (f"#!/usr/bin/env bash\nsource {shlex.quote(str(LOCAL / 'env.sh'))}\n"
           f"export OPUS_RUN_REPO={shlex.quote(str(repo))}\nexport OPUS_RUN_DIR={shlex.quote(str(out))}\n"
           'export PYTHONPATH="$OPUS_RUN_REPO"\nexport AITER_JIT_DIR="$OPUS_RUN_DIR/.cache/aiter_jit"\n'
           'export FLYDSL_RUNTIME_CACHE_DIR="$OPUS_RUN_DIR/.cache/flydsl"\nexport TRITON_CACHE_DIR="$OPUS_RUN_DIR/.cache/triton"\n'
           'unset OPUS_GEN_CO_DIR\ncd "$OPUS_RUN_REPO" || return 1\n')
    (out / "experiment_env.sh").write_text(env)
    prefix = f"#!/usr/bin/env bash\nset -euo pipefail\nsource {shlex.quote(str(out / 'experiment_env.sh'))}\n"
    build = (prefix + '"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" '
             + "--kids " + " ".join(map(str, new_ids))
             + ' --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report "$OPUS_RUN_DIR/new_co_build_info.json" --jobs 8\n')
    (out / "build_new_co.sh").write_text(build)
    tune = (prefix + 'export AITER_REBUILD=1\nexport OPUS_TUNE_PROGRESS_FILE="$OPUS_RUN_DIR/progress.jsonl"\n'
            'export OPUS_TUNE_CANDIDATE_PROGRESS_DIR="$OPUS_RUN_DIR/candidates"\n'
            '"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py '
            '--mxscale_table --libtype opus --mp 4 --shape_grouped --opus_top_geometries 0 '
            '--opus_geometry_file "$OPUS_RUN_DIR/experiment_geometries.csv" -i "$OPUS_RUN_DIR/experiment_shapes.csv" '
            '-o "$OPUS_RUN_DIR/tuned.csv" -o2 "$OPUS_RUN_DIR/profile.csv"\n')
    (out / "run_tune.sh").write_text(tune)
    for filename in ("experiment_env.sh", "build_new_co.sh", "run_tune.sh"):
        (out / filename).chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out-dir", type=Path, default=LOCAL / "co_experiment")
    parser.add_argument("--max-new-co", type=int, default=24)
    parser.add_argument("--check-only", action="store_true", help="Validate a selected plan on CPU without copying a repo")
    parser.add_argument("--mechanism-check", action="store_true", help="CPU fixtures prove stable IDs/pair validation; no performance plan is produced")
    args = parser.parse_args()
    if args.mechanism_check:
        plan = mechanism_plan()
    elif args.plan:
        plan = json.loads(args.plan.read_text())
    else:
        parser.error("An explicit --plan based on current step2 measurements is required")
    original_doc = json.loads((ROOT / MANIFEST_REL).read_text())
    original = load_common(ROOT)
    document, records = prepare_manifest(original_doc, plan, args.max_new_co)
    new_ids = [row["new_kid"] for row in records]
    with tempfile.TemporaryDirectory(prefix="co_identity_check_", dir=LOCAL) as temporary:
        manifest = Path(temporary) / "co_kernels_gfx950.json"
        manifest.write_text(json.dumps(document, indent=2) + "\n")
        extended = load_common(ROOT, manifest=manifest, freeze=True, label="extended")
        receipt = check_identity(original, extended, new_ids)
        for record in records:
            instance = registry(extended)[record["new_kid"]]
            record["kernelName"] = instance.name
            # Remove selection restriction to validate each selected shape using
            # the actual common support predicates rather than bypassing guards.
            probe = copy.deepcopy(instance)
            probe.bpreshuffle_co.pop("supported_shapes", None)
            for shape in record["shapes"]:
                if not extended.a8w8_mxscale_bpreshuffle_supports_shape(probe, *map(int, shape)):
                    raise ValueError(f"Selected shape {shape} is illegal for new kid={record['new_kid']}")
    report = {"identity": receipt, "points": records, "compiled": False, "gpu_used": False,
              "mode": "mechanism_fixture_only" if args.mechanism_check else "selected_plan_cpu_check" if args.check_only else "prepared_isolated_source"}
    if args.mechanism_check or args.check_only:
        print(json.dumps(report, indent=2))
        return
    out = args.out_dir.resolve()
    if not out.is_relative_to(LOCAL) or out == LOCAL:
        parser.error("The isolated experiment must be a fresh subdirectory of local_20261029")
    if out.exists():
        parser.error(f"Output exists; choose a fresh directory to preserve previous experiments: {out}")
    out.mkdir(parents=True)
    # Byte copies preserve source independence: never hard-link mutable sources.
    repo = out / ".cache/experiment_repo"
    shutil.copytree(ROOT, repo, symlinks=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".cache", "reports", "local_20261029", "build", "*.egg-info"))
    common_path = repo / COMMON_REL
    common_source = common_path.read_text()
    if common_source.count(JIT_BASE_LINE) != 1:
        raise ValueError("Copied source differs from validated source")
    common_path.write_text(common_source.replace(JIT_BASE_LINE, FROZEN_BASE_LINE))
    (repo / MANIFEST_REL).write_text(json.dumps(document, indent=2) + "\n")
    (out / "selection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "identity_and_points.json").write_text(json.dumps(report, indent=2) + "\n")
    write_inputs(out, extended.gfx950_co_kernels_declared, records)
    write_launchers(out, new_ids)
    copied = load_common(repo, label="copied")
    copied_receipt = check_identity(original, copied, new_ids)
    (out / "copied_source_identity.json").write_text(json.dumps(copied_receipt, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), "new_ids": new_ids, "original_identities_verified": 48212,
                      "compiled": False, "gpu_used": False,
                      "next_commands": [f"bash {out}/build_new_co.sh", f"bash {out}/run_tune.sh"]}, indent=2))


if __name__ == "__main__":
    main()
