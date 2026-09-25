"""Finish on GPUs 5/6 after an external workload also occupied GPU 4."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(path):
    with Path(path).open(newline="") as f: return list(csv.DictReader(f))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def shape(r): return tuple(int(r[k]) for k in ("M","N","K"))
def dump(path,data):
    tmp=Path(str(path)+".tmp");tmp.write_text(json.dumps(data,indent=2)+"\n");tmp.replace(path)
def write(path,rows):
    with Path(path).open("x",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

plan=json.loads((HERE/"final_recovery_plan.json").read_text())
original=json.loads((HERE/"plan_before_gpu7_recovery.json").read_text())
state=dict(plan,status="running",stage="remeasuring",start_time=time.time())
assert not (HERE/"final_recovery.json").exists()
shutil.copy2(HERE/"workflow.json",HERE/"workflow_before_final_recovery.json")
dump(HERE/"workflow.json",dict(status="running",stage="final_recovery",start_time=state["start_time"]))
def save(): dump(HERE/"final_recovery.json",state)
save()

def coalesce(gpu,parts):
    runs=[];sources=[];seen=set()
    combined={k:[] for k in ("candidates","raw","correctness","summary","choices")}
    stable=("gpu","gpu_uuid","gpu_bus","rounds","jit_dir","harness_sha256","source_sha256",
            "binary_sha256","reference_sha256","num_warmup","num_iters","num_rotate_args",
            "timing","timing_scope","numerical_contract","inputs","candidate_policy","selection_policy")
    for path,allowed in parts:
        run=json.loads(path.read_text());assert run["status"]=="passed" and run["mode"]=="sweep"
        assert run["gpu"]==gpu and run["rounds"]==3
        if runs: assert all(run[k]==runs[0][k] for k in stable)
        own={tuple(s) for s in run["shapes"]}
        if allowed is None: allowed=own
        assert allowed<=own and not seen.intersection(allowed)
        seen.update(allowed)
        for kind,h in run["output_sha256"].items():assert sha(run["outputs"][kind])==h,(path,kind)
        for kind in combined:
            rows=[r for r in read(run["outputs"][kind]) if shape(r) in allowed]
            assert {shape(r) for r in rows}==allowed,(path,kind)
            combined[kind].extend(rows)
        sources.append(dict(path=str(path),sha256=sha(path),retained_shapes=sorted(allowed),
                            excluded_shapes=sorted(own-allowed)))
        runs.append(run)
    first=runs[0]
    for p,h in first["source_sha256"].items(): assert sha(ROOT/p)==h,p
    for p,h in first["binary_sha256"].items(): assert sha(p)==h,p
    prefix=HERE/f"gpu{gpu}_complete_r3";outputs={}
    for kind,rows in combined.items():
        path=Path(f"{prefix}_{kind}.csv");write(path,rows);outputs[kind]=str(path)
    path=Path(f"{prefix}_gpu.jsonl")
    with path.open("x") as f:
        for run in runs:f.write(Path(run["outputs"]["gpu"]).read_text())
    outputs["gpu"]=str(path)
    shapes=HERE/f"gpu{gpu}_complete_shapes.csv"
    sorted_shapes=sorted(seen,key=lambda s:(s[1],s[2],s[0]))
    write(shapes,[dict(gfx="gfx950",cu_num=256,**dict(zip(("M","N","K"),s))) for s in sorted_shapes])
    merged=dict(first,schema_version=4,aggregate=True,
                aggregation="Union of successful complete shape measurements on the same GPU; interrupted runs excluded in full",
                subruns=sources,shapes=sorted_shapes,shapes_file=str(shapes),shapes_sha256=sha(shapes),
                completed_shapes=len(seen),outputs=outputs,output_sha256={k:sha(p) for k,p in outputs.items()},
                correctness_checks=len(combined["correctness"]),
                excluded_checks=sum(r["status"]!="passed" for r in combined["correctness"]),
                timing_records=len(combined["raw"]),
                start_time=min(r["start_time"] for r in runs),end_time=max(r["end_time"] for r in runs),
                all_shapes_opus_faster=all(r["opus_faster"]=="True" for r in combined["choices"]),
                all_shapes_9010_9011_faster=all(r["target_faster"]=="True" for r in combined["choices"]))
    dump(Path(str(prefix)+"_run.json"),merged)
    return dict(gpu=gpu,shapes_file=str(shapes),shapes=len(seen),prefix=str(prefix))

jobs=[]
try:
    for worker in state["workers"]:
        cmd=[sys.executable,"-u",str(HERE/"benchmark.py"),"--sweep","--rounds","3", "--prefix",worker["prefix"],
             "--shapes",worker["shapes_file"],"--jit-dir",original["jit_dir"]]
        with Path(worker["prefix"]+".log").open("x") as log:
            p=subprocess.Popen(cmd,env=dict(os.environ,OPUS_TUNE_GPU_INDEX=str(worker["gpu"])),stdout=log,stderr=subprocess.STDOUT)
        jobs.append(p);worker.update(pid=p.pid,status="running")
    save()
    while True:
        for p,w in zip(jobs,state["workers"]):
            path=Path(w["prefix"]+"_run.json")
            if path.exists():
                run=json.loads(path.read_text())
                for k in ("status","completed_shapes","error"):
                    if k in run:w[k]=run[k]
            w["exit_code"]=p.poll()
            if p.poll() is not None:assert p.returncode==0 and w["status"]=="passed",w
        save()
        if all(p.poll() is not None for p in jobs):break
        time.sleep(10)
    migrated={tuple(s) for s in plan["migrated_gpu4_close_shapes"]}
    base4=HERE/"gpu4_r3_run.json"
    retained4={tuple(s) for s in json.loads(base4.read_text())["shapes"]}-migrated
    workers=[coalesce(4,[(base4,retained4)])]
    for gpu in (5,6):
        workers.append(coalesce(gpu,[(HERE/f"gpu{gpu}_r3_run.json",None),
                                     (HERE/f"gpu{gpu}_recovery_r3_run.json",None),
                                     (HERE/f"gpu{gpu}_final_recovery_r3_run.json",None)]))
    assert sum(w["shapes"] for w in workers)==295
    active=dict(original,workers=workers,recovery=str(HERE/"final_recovery.json"))
    dump(HERE/"plan.json",active)
    shutil.copy2(HERE/"launcher.json",HERE/"launcher_before_final_recovery.json")
    dump(HERE/"launcher.json",dict(status="passed",aggregation=True,workers=workers,recovery=str(HERE/"final_recovery.json")))
    state["stage"]="audit_confirmation_replay";save()
    dump(HERE/"workflow.json",dict(status="running",stage=state["stage"],start_time=state["start_time"]))
    with (HERE/"finish.log").open("x") as log:
        r=subprocess.run([sys.executable,"-u",str(HERE/"finish.py")],stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,"finish pipeline failed"
    state.update(status="passed",stage="complete")
    dump(HERE/"workflow.json",dict(status="passed",stage="complete",recovery=str(HERE/"final_recovery.json"),
                                   start_time=state["start_time"],end_time=time.time()))
except BaseException as exc:
    state.update(status="failed",error=repr(exc));raise
finally:
    state["end_time"]=time.time();save()
