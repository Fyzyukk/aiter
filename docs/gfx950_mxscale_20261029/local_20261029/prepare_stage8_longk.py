#!/usr/bin/env python3
"""Expand the validated streaming-A/static-B kernel to long-K geometries."""
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
OUT = LOCAL / "stage8_longk100_20261030"
REL = Path("csrc/opus_gemm/gen_co/co_kernels_gfx950.json")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    OUT.mkdir()
    experiment = OUT / "co_experiment"
    experiment.mkdir()
    reference = LOCAL / "stage7_kernel103_20261030/result/public_best_opus421.csv"
    rows = [r for r in csv_rows(reference) if r["status"] == "slower_than_csv"]
    assert len(rows) == 100
    before = registry(load_common(ROOT, label="stage8_before"))
    assert len(before) == 48250
    document = json.loads((ROOT / REL).read_text())
    original = copy.deepcopy(document)
    points = []
    # Tile/wave/LDS occupancy, K panel and operand pinning change independently
    # of the validated body. All use a real three-fragment B queue, streamed A
    # and constant ring slots. C keeps the existing contiguous LDS writeback.
    geometries = [(64,128,256,4,2,3), (64,256,256,4,2,2),
        (96,128,256,4,2,2), (96,256,128,4,2,3),
        (128,256,256,4,1,2), (128,128,256,4,1,2),
        (64,128,128,2,2,3), (64,256,128,4,2,3)]
    for fixed_k in (16384, 21504):
        for bm,bn,bk,waves,wg,stages in geometries:
            for pin in (False,True):
                kid = 100300 + len(points)
                args = [str(v) for v in (bm,bn,stages,1,1)] + ["true",str(fixed_k),str(waves),str(wg),"17","2","0",str(bk),"false",str(pin).lower(),str(pin).lower(),"0"]
                parent = 392
                entry = {"kid":kid,"pipeline":"prefetch_b_vgpr","block_size":waves*64,
                    "tile":[bm,bn,bk],"wave_layout":[1,waves],"stages":stages,
                    "launch_bounds":[waves*64,wg],"device_flags":"flags_00",
                    "pad_m":True,"wide_output":True,"max_k":fixed_k,"fixed_k":fixed_k,
                    "traits":{"name":"opus_stage7_prefetch_b_base_traits","args":args},
                    "traits_header":"opus_gemm_traits_stage7_gfx950.cuh",
                    "pipeline_header":"opus_gemm_pipeline_stage7_b_vgpr_gfx950.cuh",
                    "body_func":"opus_stage7_b_detail::bulk_panel_body",
                    "body_args":["opus_stage7_b_traits<opus_stage6_b_queue_traits<CoTraits, 3>, true, true, 0>"],
                    "grid_layout":"xcd","pin_variant":True,"pin_mask":0}
                document["gfx950"].append(entry)
                digest = hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest()[:16]
                document["image_suffixes"][str(kid)] = f"pm1_stage8_stream_k{fixed_k}_{digest}"
                points.append({"new_kid":kid,"seed_kid":parent,"pipeline":"prefetch_b_vgpr",
                    "axis":{"BM":bm,"BN":bn,"BK":bk,"waves":waves,"WGPerCU":wg,"AStages":stages,
                        "ActualBAhead":3,"StreamA":True,"StaticBRing":True,"PinA":pin,"PinC":pin,"FixedK":fixed_k}})
    assert document["gfx950"][:len(original["gfx950"])] == original["gfx950"]
    dump(experiment / "proposed_manifest.json", document)
    module = load_common(ROOT,manifest=experiment / "proposed_manifest.json",label="stage8_extended")
    after = registry(module)
    assert set(after) == set(before) | {p["new_kid"] for p in points}
    digest = hashlib.sha256()
    for kid, instance in before.items():
        assert dataclasses.asdict(instance) == dataclasses.asdict(after[kid]), kid
        digest.update(f"{kid}:{fingerprint(instance)}\n".encode())
    for point in points:
        instance = after[point["new_kid"]]
        shapes = [list(key(row)[2:5]) for row in rows if int(row["K"])==point["axis"]["FixedK"]
            and module.a8w8_mxscale_bpreshuffle_supports_shape(instance,*key(row)[2:5])]
        assert shapes, point
        point["shapes"],point["kernelName"] = shapes,instance.name
        next(e for e in document["gfx950"] if e["kid"]==point["new_kid"])["supported_shapes"] = shapes
    dump(experiment / "proposed_manifest.json",document)
    repo = experiment / ".cache/experiment_repo"
    shutil.copytree(ROOT,repo,symlinks=True,ignore=shutil.ignore_patterns(
        ".git","__pycache__",".cache","reports","local_20261029","build","*.egg-info","*.so","*.o"))
    shutil.copy2(experiment / "proposed_manifest.json",repo / REL)
    helpers = repo / "docs/gfx950_mxscale_20261029/local_20261029"
    helpers.mkdir()
    for path in LOCAL.glob("*.py"):
        shutil.copy2(path,helpers / path.name)
    dump(experiment / "identity_and_points.json",{"previous_ids_verified":len(before),
        "previous_config_digest":digest.hexdigest(),"jit_kid_base":780,"repo":str(repo),"points":points,
        "reference":str(reference),"new_source_changes":[],"expands_validated_body":"stage7_streaming_A_static_B"})
    (experiment / "experiment_env.sh").write_text("source " + shlex.quote(str(LOCAL / "env.sh"))
        + "\nexport OPUS_RUN_REPO=" + shlex.quote(str(repo)) + '\nexport PYTHONPATH="$OPUS_RUN_REPO"'
        + "\nexport AITER_JIT_DIR=" + shlex.quote(str(experiment / ".cache/aiter_jit"))
        + '\nexport ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0\ncd "$OPUS_RUN_REPO"\n')
    (experiment / "build_new_co.sh").write_text("set -euo pipefail\nsource " + shlex.quote(str(experiment / "experiment_env.sh"))
        + '\n"$OPUS_RUN_PYTHON" csrc/opus_gemm/gen_co/build_co.py --arch gfx950 --llvm-bin "$HIP_CLANG_PATH" --kids '
        + " ".join(str(p["new_kid"]) for p in points) + ' --jobs 8 --out-dir "$OPUS_RUN_REPO/csrc/opus_gemm/gen_co/gfx950" --report '
        + shlex.quote(str(experiment / "build_info.json")) + "\n")
    print(json.dumps({"new_co_points":len(points),"previous_ids_verified":len(before),"long_K_target_shapes":sum(int(r['K'])>=16384 for r in rows)}))


if __name__ == "__main__":
    main()
