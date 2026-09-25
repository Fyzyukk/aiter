"""Retest the occupied GPU-7 shard on GPUs 4--6, preserving valid base shards.

Each shape is measured completely on one physical GPU. Complete subruns on
the same GPU are concatenated with provenance and hash checks before the
ordinary audit and confirmation pipeline. No GPU-7 timing enters the union.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data):
    temp = Path(str(path)+".tmp")
    temp.write_text(json.dumps(data, indent=2)+"\n")
    temp.replace(path)
def write(path, rows):
    with Path(path).open("x", newline="") as f:
        w=csv.DictWriter(f,list(rows[0])); w.writeheader(); w.writerows(rows)
old_plan = json.loads((HERE / "plan.json").read_text())
assert not (HERE / "recovery.json").exists()
failed = json.loads((HERE / "gpu7_r3_run.json").read_text())
assert failed["status"] == "failed" and "External GPU activity" in failed["error"]
shutil.copy2(HERE / "plan.json", HERE / "plan_before_gpu7_recovery.json")
todo = read(HERE / "gpu7_shapes.csv")
state = dict(status="running", start_time=time.time(),
             reason="External process occupied physical GPU 7; discard its entire initial shard and remeasure on GPUs 4--6",
             excluded_run=str(HERE / "gpu7_r3_run.json"), workers=[])
jobs = {}
for i,gpu in enumerate((4,5,6)):
    shapes = HERE / f"gpu{gpu}_recovery_shapes.csv"
    subset = todo[i::3]
    write(shapes, subset)
    state["workers"].append(dict(gpu=gpu, status="waiting_for_base_shard", shapes=len(subset),
                                 shapes_file=str(shapes), prefix=str(HERE / f"gpu{gpu}_recovery_r3")))
def save():
    dump(HERE / "recovery.json", state)
save()

def combine(gpu, worker):
    paths = [HERE / f"gpu{gpu}_r3_run.json", Path(worker["prefix"]+"_run.json")]
    runs = [json.loads(p.read_text()) for p in paths]
    first = runs[0]
    stable = ["gpu", "gpu_uuid", "gpu_bus", "rounds", "jit_dir", "harness_sha256",
              "source_sha256", "binary_sha256", "reference_sha256", "num_warmup", "num_iters",
              "num_rotate_args", "timing", "timing_scope", "numerical_contract", "inputs",
              "candidate_policy", "selection_policy"]
    seen=set()
    for run in runs:
        assert run["status"]=="passed" and run["mode"]=="sweep" and run["rounds"]==3
        assert all(run[k]==first[k] for k in stable)
        own={tuple(s) for s in run["shapes"]}
        assert len(own)==run["completed_shapes"] and not seen.intersection(own)
        seen.update(own)
        for kind,h in run["output_sha256"].items():
            assert sha(run["outputs"][kind])==h, (gpu,kind)
    prefix=HERE / f"gpu{gpu}_complete_r3"
    outputs={}
    for kind in ("candidates","raw","correctness","summary","choices"):
        rows=[r for run in runs for r in read(run["outputs"][kind])]
        path=Path(f"{prefix}_{kind}.csv"); write(path,rows); outputs[kind]=str(path)
    log=Path(f"{prefix}_gpu.jsonl")
    with log.open("x") as f:
        for run in runs:
            f.write(Path(run["outputs"]["gpu"]).read_text())
    outputs["gpu"]=str(log)
    shapes=HERE / f"gpu{gpu}_complete_shapes.csv"
    sorted_shapes=sorted(seen,key=lambda s:(s[1],s[2],s[0]))
    write(shapes,[dict(gfx="gfx950",cu_num=256,**dict(zip(("M","N","K"),s))) for s in sorted_shapes])
    merged=dict(first)
    merged.update(schema_version=3,aggregate=True,
                  aggregation="Union of complete disjoint shape subruns on the same physical GPU",
                  subruns=[dict(path=str(p),sha256=sha(p)) for p in paths],
                  shapes=sorted_shapes,shapes_file=str(shapes),shapes_sha256=sha(shapes),
                  completed_shapes=len(seen),outputs=outputs,
                  output_sha256={k:sha(p) for k,p in outputs.items()},
                  start_time=min(r["start_time"] for r in runs),end_time=max(r["end_time"] for r in runs))
    for key in ("correctness_checks","excluded_checks","timing_records"):
        merged[key]=sum(r[key] for r in runs)
    merged["all_shapes_opus_faster"]=all(r["all_shapes_opus_faster"] for r in runs)
    merged["all_shapes_9010_9011_faster"]=all(r["all_shapes_9010_9011_faster"] for r in runs)
    dump(Path(str(prefix)+"_run.json"),merged)
    return dict(gpu=gpu,shapes_file=str(shapes),shapes=len(seen),prefix=str(prefix))

try:
    while True:
        for worker in state["workers"]:
            gpu=worker["gpu"]
            if gpu not in jobs:
                base=json.loads((HERE / f"gpu{gpu}_r3_run.json").read_text())
                if base["status"]=="running": continue
                assert base["status"]=="passed", (gpu,base.get("error"))
                cmd=[sys.executable,"-u",str(HERE/"benchmark.py"),"--sweep","--rounds","3",
                     "--prefix",worker["prefix"],"--shapes",worker["shapes_file"],"--jit-dir",old_plan["jit_dir"]]
                with Path(worker["prefix"]+".log").open("x") as log:
                    p=subprocess.Popen(cmd,env=dict(os.environ,OPUS_TUNE_GPU_INDEX=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                jobs[gpu]=p;worker.update(status="running",pid=p.pid)
            p=jobs[gpu]
            run_path=Path(worker["prefix"]+"_run.json")
            if run_path.exists():
                run=json.loads(run_path.read_text())
                for key in ("status","completed_shapes","timing_records","error"):
                    if key in run: worker[key]=run[key]
            worker["exit_code"]=p.poll()
            if p.poll() is not None:
                assert p.returncode==0 and worker["status"]=="passed", worker
        save()
        if len(jobs)==3 and all(p.poll() is not None for p in jobs.values()): break
        time.sleep(10)
    # The original launcher reports its failed GPU-7 child; retain that record.
    while json.loads((HERE/"workflow.json").read_text())["status"]=="running":
        time.sleep(5)
    shutil.copy2(HERE/"workflow.json",HERE/"workflow_before_gpu7_recovery.json")
    shutil.copy2(HERE/"launcher.json",HERE/"launcher_before_gpu7_recovery.json")
    plan=dict(old_plan)
    plan["workers"]=[combine(w["gpu"],w) for w in state["workers"]]
    assert sum(w["shapes"] for w in plan["workers"])==295
    plan["recovery"]=dict(reason=state["reason"], excluded_run=state["excluded_run"], active_gpus=[4,5,6])
    dump(HERE/"plan.json",plan)
    dump(HERE/"launcher.json",dict(status="passed",aggregation=True,
                                   recovery=str(HERE/"recovery.json"),workers=plan["workers"]))
    dump(HERE/"workflow.json",dict(status="running",stage="audit_confirmation_replay",start_time=state["start_time"],
                                   recovery=str(HERE/"recovery.json")))
    state["stage"]="audit_confirmation_replay";save()
    with (HERE/"finish.log").open("x") as log:
        r=subprocess.run([sys.executable,"-u",str(HERE/"finish.py")],stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,"finish pipeline failed"
    state.update(status="passed",stage="complete")
    dump(HERE/"workflow.json",dict(status="passed",stage="complete",start_time=state["start_time"],end_time=time.time(),
                                   recovery=str(HERE/"recovery.json")))
except BaseException as exc:
    state.update(status="failed",error=repr(exc))
    raise
finally:
    state["end_time"]=time.time();save()
