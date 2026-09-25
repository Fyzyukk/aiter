"""Enumerate current registered candidates against the upstream gfx950 shape set."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(ROCR_VISIBLE_DEVICES="GPU-5ff36708541c8ec0", AITER_AOT_IMPORT="1",
                  AITER_REBUILD="0", AITER_JIT_DIR=str(HERE / "jit"),
                  GPU_ARCHS="gfx950", CU_NUM="256", OPUS_TUNE_GPU_INDEX="7")
sys.path.insert(0, str(ROOT))
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from bootstrap import load_tune
from benchmark import candidates_for_shape
tune = load_tune()
tuner = tune.OpusMxscaleBpreshuffleTuner()
baseline = ROOT / "aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
all_shapes = sorted({tuple(int(r[k]) for k in ("M", "N", "K"))
                    for r in csv.DictReader(baseline.open())
                    if r["gfx"] == "gfx950" and r["cu_num"] == "256" and int(r["M"]) >= 1024},
                   key=lambda s: (s[1], s[2], s[0]))
active, deferred = [], []
for shape in all_shapes:
    (active if tune.candidate_kids_for_shape("gfx950", *shape) else deferred).append(shape)
assert len(all_shapes) == 305 and len(active) == 295 and len(deferred) == 10
def write(name, rows, keys):
    with (HERE / name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)
def shape_rows(shapes):
    return [dict(gfx="gfx950", cu_num=256, **dict(zip(("M","N","K"), s))) for s in shapes]
for name, shapes in (("all_model_shapes.csv", all_shapes), ("shapes.csv", active), ("deferred_shapes.csv", deferred)):
    write(name, shape_rows(shapes), ["gfx", "cu_num", "M", "N", "K"])
rows = []
for shape in active:
    for c in candidates_for_shape(tune, tuner, shape):
        r = c.row
        rows.append(dict(gfx="gfx950", cu_num=256, M=r["M"], N=r["N"], K=r["K"],
                         libtype=r["lib"], kernelId=r["kid"], splitK=r["splitK"], kernelName=r["kernelName"]))
assert len(rows) == 27690
write("expected_all_candidates.csv", rows, list(rows[0]))
workers = []
for i,gpu in enumerate(range(4,8)):
    subset = active[i::4]
    name = f"gpu{gpu}_shapes.csv"
    write(name, shape_rows(subset), ["gfx","cu_num","M","N","K"])
    workers.append(dict(gpu=gpu, shapes_file=str(HERE / name), shapes=len(subset),
                        prefix=str(HERE / f"gpu{gpu}_r3")))
plan = dict(status="prepared", rounds=3, total_shapes=len(active), total_candidates=len(rows),
            candidates_by_backend=dict(Counter(r["libtype"] for r in rows)), workers=workers,
            jit_dir=str(HERE / "jit"), build_manifest=str(HERE / "build.json"),
            harness_sha256=hashlib.sha256((HERE / "benchmark.py").read_bytes()).hexdigest(),
            baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
            scope="Current upstream-based source, fresh builds, new local timings; old timings only track transitions")
(HERE / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(json.dumps(plan, indent=2))
