"""Audit four complete disjoint sweeps, preserving physical-GPU provenance."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEYS = ("M", "N", "K")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def shape(row):
    return tuple(int(row[k]) for k in KEYS)


plan = json.loads((HERE / "plan.json").read_text())
assert sha(HERE / "benchmark.py") == plan["harness_sha256"]
manifests, seen, provenance = [], set(), []
for worker in plan["workers"]:
    path = Path(worker["prefix"] + "_run.json")
    run = json.loads(path.read_text())
    assert run["status"] == "passed" and run["mode"] == "sweep", path
    assert run["rounds"] == 3 and run["completed_shapes"] == worker["shapes"], path
    assert run["gpu"] == worker["gpu"] and run["harness_sha256"] == plan["harness_sha256"]
    own = {tuple(s) for s in run["shapes"]}
    assert len(own) == worker["shapes"] and not seen.intersection(own)
    assert own == {shape(r) for r in read(worker["shapes_file"])}
    assert sha(run["shapes_file"]) == run["shapes_sha256"]
    seen.update(own)
    for key in ("source_sha256", "binary_sha256", "reference_sha256", "jit_dir",
                "num_warmup", "num_iters", "num_rotate_args", "numerical_contract",
                "timing", "timing_scope", "candidate_policy", "selection_policy"):
        if manifests:
            assert run[key] == manifests[0][key], key
    for kind, digest in run["output_sha256"].items():
        assert sha(run["outputs"][kind]) == digest, (path, kind)
    for kind in ("candidates", "raw", "correctness", "summary", "choices"):
        rows = read(run["outputs"][kind])
        assert {shape(r) for r in rows} <= own, (path, kind)
        if kind in ("candidates", "correctness", "summary", "choices"):
            assert {shape(r) for r in rows} == own, (path, kind)
        if kind == "choices":
            assert len(rows) == len(own)
    provenance.append(dict(path=str(path), sha256=sha(path), gpu=run["gpu"],
                           gpu_uuid=run["gpu_uuid"], gpu_bus=run["gpu_bus"], shapes=len(own)))
    manifests.append(run)
assert len(seen) == 295 and seen == {shape(r) for r in read(HERE / "shapes.csv")}
first = manifests[0]
for path, digest in first["source_sha256"].items():
    assert sha(ROOT / path) == digest, path
for path, digest in first["binary_sha256"].items():
    assert sha(path) == digest, path

outputs = {}
for kind in ("candidates", "raw", "correctness", "summary", "choices"):
    combined = []
    for run in manifests:
        for row in read(run["outputs"][kind]):
            combined.append(dict(row, gpu=run["gpu"], gpu_uuid=run["gpu_uuid"]))
    combined.sort(key=lambda r: (int(r["N"]), int(r["K"]), int(r["M"])))
    path = HERE / f"full_r3_{kind}.csv"
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, list(combined[0]))
        w.writeheader()
        w.writerows(combined)
    outputs[kind] = str(path)

keys = ("rounds", "jit_dir", "harness_sha256", "source_sha256", "binary_sha256",
        "reference_sha256", "num_warmup", "num_iters", "num_rotate_args", "timing",
        "timing_scope", "numerical_contract", "inputs", "candidate_policy", "selection_policy")
merged = {key: first[key] for key in keys}
merged.update(schema_version=2, status="passed", mode="sweep", aggregate=True,
              aggregation="Union of disjoint shape sweeps; comparisons within one GPU only",
              shards=provenance, gpus=[r["gpu"] for r in manifests], shapes=sorted(seen),
              completed_shapes=295, start_time=min(r["start_time"] for r in manifests),
              end_time=max(r["end_time"] for r in manifests), outputs=outputs,
              output_sha256={kind: sha(p) for kind, p in outputs.items()})
for key in ("correctness_checks", "excluded_checks", "timing_records"):
    merged[key] = sum(r[key] for r in manifests)
with (HERE / "full_r3_run.json").open("x") as f:
    json.dump(merged, f, indent=2)
    f.write("\n")
print(json.dumps({k: merged[k] for k in ("status", "completed_shapes", "timing_records", "gpus")}))
