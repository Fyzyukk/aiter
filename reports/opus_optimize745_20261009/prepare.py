#!/usr/bin/env python3
"""Prepare the complete current-source OPUS / raw-scale FlyDSL BMM experiment.

Standard library and scalar configuration imports only; never initializes HIP.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from csrc.opus_gemm.opus_gemm_bpreshuffle_config import (
    config_from_legacy_kid, config_identity, config_supports_shape,
    construct_config, pipeline_configs, kernel_instance_from_config,
)
from csrc.opus_gemm.opus_gemm_common import (
    bpreshuffle_candidate_split_k,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape(row):
    return tuple(int(row[x]) for x in ("M", "N", "K"))


def serialize(config):
    return {"pipeline": config.pipeline, "compile_params": json.loads(config.to_json()),
            "legacy_kid": config.legacy_kid, "identity": config_identity(config)}


def extra_configs():
    """Explore supported geometry/queue/stage axes, keeping mechanism choices explicit."""
    configs = {}
    rejected = []

    def add(pipeline, **params):
        try:
            c = construct_config(pipeline, **params)
            configs[config_identity(c)] = c
        except ValueError as error:
            rejected.append({"pipeline": pipeline, "parameters": params, "reason": str(error)})

    for tm, tn, wk in ((16, 16, 1), (16, 32, 1), (32, 32, 1),
                       (32, 64, 1), (16, 32, 2), (32, 32, 2)):
        for depth in (2, 3, 4, 5, 6, 8):
            add("register", tile_m=tm, tile_n=tn, wave_m=1, wave_n=1,
                wave_k=wk, prefetch=depth, runtime_split_k=True)
    for tm in (64, 96, 128, 160):
        for stages in (2, 3):
            for fk, panel in ((384, 8), (768, 8), (1536, 16), (0, 32)):
                add("tiled", tile_m=tm, tile_n=128, wave_m=2, wave_n=2,
                    schedule=2, stages=stages, fixed_k=fk, scale_panel=panel)
    # A cooperative LDS ring with independent B registers, and fine geometry
    # with runtime global splitting, are different movement mechanisms.
    for tm, tn, wm, wn in ((32, 64, 1, 4), (64, 64, 2, 2),
                           (64, 128, 2, 2), (96, 128, 2, 2), (32, 128, 1, 4)):
        for stages in (2, 3, 4):
            for ahead in (1, 2, 3):
                add("lds", tile_m=tm, tile_n=tn, wave_m=wm, wave_n=wn,
                    stages=stages, cluster=1, b_direct=True, b_ahead=ahead,
                    register_scales=True, early_scale_loads=False,
                    prefetch_before_read=False, read_only_drain=False,
                    specializations=(), dispatch=(), runtime_split_k=False,
                    split_k=1, fine_m_loads=True, output_mode=2,
                    sfa_alignment=16, c_alignment=16)
    for tm, tn, wm, wn in ((48, 64, 1, 4), (64, 64, 2, 2),
                           (64, 128, 2, 2), (96, 128, 2, 2)):
        for stages in (2, 3, 4, 5, 6):
            add("lds", tile_m=tm, tile_n=tn, wave_m=wm, wave_n=wn,
                stages=stages, cluster=1, runtime_split_k=True,
                register_scales=False, early_scale_loads=True,
                prefetch_before_read=True, read_only_drain=True,
                specializations=(), dispatch=(), fine_m_loads=True, output_mode=2)
    add("large_output", tile_m=192, tile_n=256, b_direct=True, c_chunk_rows=64)
    return tuple(configs.values()), rejected


def main():
    destination = HERE / "plan.json"
    if destination.exists():
        raise SystemExit("Plan already exists; use a fresh experiment directory")
    history = ROOT / "reports/opus_flydsl_resume_20261009/historical_745_best_per_shape.csv"
    profile = ROOT / "reports/opus_clang23_mixed_retune_20261008/profile.csv"
    flycsv = ROOT / "reports/opus_flydsl_comparison_20261008/upstream_mxscale_main.csv"
    hs = list(csv.DictReader(history.open()))
    assert len(hs) == len({shape(r) for r in hs}) == 745
    best = {}
    for r in csv.DictReader(profile.open()):
        if r["libtype"] != "opus" or float(r["errRatio"]) != 0:
            continue
        s = shape(r)
        c = config_from_legacy_kid(int(r["kernelId"]))
        if not config_supports_shape(c, *s) or not math.isfinite(float(r["us"])):
            continue
        if s not in best or float(r["us"]) < float(best[s]["us"]):
            best[s] = r
    bmm = {}
    for r in csv.DictReader(flycsv.open()):
        if r["kernelId"] == "bmm" and r["w_scale_block"] == "128x128":
            assert shape(r) not in bmm
            bmm[shape(r)] = r
    extras, rejected = extra_configs()
    all_configs = {config_identity(c): c for c in extras}
    cases = []
    for h in hs:
        s = shape(h)
        baseline = config_from_legacy_kid(int(best[s]["kernelId"]))
        assert s in bmm
        candidates = {config_identity(c): c for c in (*pipeline_configs(s), *extras)
                      if config_supports_shape(c, *s)}
        candidates[config_identity(baseline)] = baseline
        all_configs.update(candidates)
        tasks = []
        for ident, c in candidates.items():
            instance = kernel_instance_from_config(c)
            for split in bpreshuffle_candidate_split_k(instance, *s):
                tasks.append({"config": ident, "split_k": split})
        cases.append({"M": s[0], "N": s[1], "K": s[2],
                      "baseline": {"config": config_identity(baseline), "split_k": 0},
                      "flydsl_name": bmm[s]["kernelName"],
                      "historical_opus_us": float(best[s]["us"]),
                      "historical_bmm_us": float(bmm[s]["us"]),
                      "candidates": tasks})
    # Prior device restriction is retained: physical SMI 0..3; one worker
    # claims whichever of those is actually owner-free. Other cards are sampled.
    sources = {p for directory in (ROOT / "csrc/opus_gemm", ROOT / "csrc/include",
                                  ROOT / "aiter/ops/opus", ROOT / "aiter/jit",
                                  ROOT / "aiter/ops/flydsl")
               for p in directory.rglob("*")
               if p.is_file() and p.suffix in (".py", ".json", ".h", ".hpp", ".cuh")}
    sources.update((ROOT / "aiter/test_common.py", ROOT / "aiter/utility/mp_tuner.py"))
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)}
    oldenv = json.loads((ROOT / "reports/opus_flydsl_resume_20261009/queue_four.json").read_text())["env"]
    env = {**oldenv, "AITER_JIT_DIR": str(HERE / "jit"),
           "FLYDSL_RUNTIME_CACHE_DIR": str(HERE / "flydsl_cache"),
           "PYTHONDONTWRITEBYTECODE": "1", "MAX_JOBS": "4", "AITER_LOG_MORE": "0"}
    plan = {"schema": 1, "source_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scope": "All 745 shapes; native raw E8M0 scales; FlyDSL BMM only",
            "allowed_smi_indices": [0, 1, 2, 3], "env": env,
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (history, profile, flycsv)},
            "source_sha256": source_hashes,
            "configs": {i: serialize(c) for i, c in all_configs.items()}, "cases": cases}
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    (HERE / "extra_configs.json").write_text(json.dumps([serialize(c) for c in extras], indent=2) + "\n")
    receipt = {"shapes": len(cases), "configs": len(all_configs), "extra_configs": len(extras),
               "shape_config_split_combinations": sum(len(c["candidates"]) for c in cases),
               "historical_bmm_wins": sum(c["historical_opus_us"] < c["historical_bmm_us"] for c in cases),
               "historical_bmm_losses": sum(c["historical_opus_us"] >= c["historical_bmm_us"] for c in cases),
               "by_pipeline": dict(collections.Counter(c.pipeline for c in all_configs.values())),
               "rejected_parameters": rejected, "plan_sha256": sha(destination),
               "gpu_queries": 0, "library_loads": 0, "kernel_launches": 0}
    (HERE / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rejected_parameters"}))


if __name__ == "__main__":
    main()
