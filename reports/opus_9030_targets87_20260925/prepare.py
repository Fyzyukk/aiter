"""Enumerate all registered backends for exactly the historical 87 targets."""
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(ROCR_VISIBLE_DEVICES="GPU-23a6cd0d658d72b6", OPUS_TUNE_GPU_INDEX="4",
                  AITER_AOT_IMPORT="1", AITER_REBUILD="0", AITER_JIT_DIR=str(HERE / "jit"),
                  GPU_ARCHS="gfx950", CU_NUM="256")
sys.path.insert(0, str(ROOT))
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from bootstrap import load_tune
from benchmark import candidates_for_shape

tune = load_tune()
tuner = tune.OpusMxscaleBpreshuffleTuner()
shapes = [tuple(int(r[k]) for k in ("M", "N", "K"))
          for r in csv.DictReader((HERE / "shapes.csv").open())]
assert len(shapes) == len(set(shapes)) == 87
rows = []
for shape in shapes:
    assert set(range(9030, 9034)) <= set(tune.candidate_kids_for_shape("gfx950", *shape))
    for c in candidates_for_shape(tune, tuner, shape):
        rows.append(dict(c.row))
with (HERE / "expected_all_candidates.csv").open("x", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
workers = [dict(gpu=gpu, shapes_file=str(HERE / f"gpu{gpu}_shapes.csv"),
                shapes=len(shapes[i::4]), prefix=str(HERE / f"gpu{gpu}_r5"))
           for i, gpu in enumerate(range(4, 8))]
plan = dict(status="prepared", rounds=5, total_shapes=87, total_candidates=len(rows),
            candidates_by_backend=dict(Counter(r["lib"] for r in rows)),
            opus_candidates_by_id=dict(Counter(r["kid"] for r in rows if r["lib"] == "opus")),
            workers=workers, jit_dir=str(HERE / "jit"), build_manifest=str(HERE / "build.json"),
            harness_sha256=hashlib.sha256((HERE / "benchmark.py").read_bytes()).hexdigest(),
            scope="Historical 87 losing shapes; all registered OPUS including 9030-9033 plus CK/CKTile/ASM; five fresh rounds per valid candidate")
(HERE / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(json.dumps(plan, indent=2))
