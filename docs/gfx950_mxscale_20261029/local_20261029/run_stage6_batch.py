#!/usr/bin/env python3
"""Stage6 runner on explicitly selected PCI-verified physical GPU0/3/6/5."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from run_stage5_batch import csv_write, dump, shortlist

LOCAL = Path(__file__).resolve().parent
PHYSICAL = [0,3,6,5]
PCI = ["0000:05:00.0", "0000:75:00.0", "0000:e5:00.0", "0000:95:00.0"]


def preflight(output, gpus):
    assert os.environ.get("ROCR_VISIBLE_DEVICES") == "1,0,6,7"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0,1,2,3"
    snapshot = json.loads(subprocess.check_output(["rocm-smi","--showuse","--showmeminfo","vram","--showbus","--json"], text=True))
    probe = '''import ctypes,json
h=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so'); assert h.hipInit(0)==0
f=h.hipDeviceGetPCIBusId; f.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]; f.restype=ctypes.c_int
rows=[]
for i in range(4):
 b=ctypes.create_string_buffer(32); assert f(b,32,i)==0; rows.append(b.value.decode().lower())
print(json.dumps(rows))
'''
    actual = json.loads(subprocess.check_output([sys.executable,"-c",probe],text=True))
    assert actual == PCI, actual
    for gpu in gpus:
        card = snapshot[f"card{PHYSICAL[gpu]}"]
        assert card["PCI Bus"].lower() == PCI[gpu]
        assert int(card["GPU use (%)"]) <= 10, (PHYSICAL[gpu],card)
        assert int(card["VRAM Total Memory (B)"])-int(card["VRAM Total Used Memory (B)"]) > 128*1024**3
    dump(output/"pci_preflight.json",{"physical_mapping":PHYSICAL,"pci":actual,"active_logical":gpus,"smi":snapshot})


def run(plan, output, rounds, iters, gpus):
    output.mkdir(); (output/"repeats").mkdir(); (output/"worker_plans").mkdir()
    preflight(output,gpus)
    dump(output/"repeat_plan.json",plan)
    subprocess.run([sys.executable,str(LOCAL/"precompile_stage5_plan.py"),"--plan",str(output/"repeat_plan.json"),
        "--receipt",str(output/"compile_receipt.json"),"--gpu",str(gpus[0])],check=True)
    csv_write(output/"expected.csv",[item["shape"] for item in plan["shapes"]])
    known=[]
    for item in plan["shapes"]:
        for candidate in item["candidates"]:
            if "current_step4_best_control" in candidate.get("roles",[]):
                known.append({**item["shape"],**{f:candidate[f] for f in ("kid","kernelName","kind","traits")},
                              "geometry":json.dumps(candidate["geometry"])})
    csv_write(output/"known.csv",known)
    shards={gpu:[] for gpu in gpus};costs={gpu:0. for gpu in gpus}
    def cost(item):
        s=item["shape"];return len(item["candidates"])*max(1.,s["M"]*s["N"]*s["K"]/1e10)
    for item in sorted(plan["shapes"],key=cost,reverse=True):
        gpu=min(costs,key=costs.get);shards[gpu].append(item);costs[gpu]+=cost(item)
    processes={};logs=[]
    try:
        for gpu,items in shards.items():
            if not items:continue
            path=output/f"worker_plans/gpu{gpu}_plan.json"
            dump(path,{**plan,"shapes":items,"gpu_physical_id":PHYSICAL[gpu],"gpu_pci_bus":PCI[gpu]})
            cmd=[sys.executable,str(LOCAL/"repeat_opus_candidates.py"),"--plan",str(path),"--output",str(output/f"repeats/gpu{gpu}.jsonl"),
                 "--gpu",str(gpu),"--rounds",str(rounds),"--iters",str(iters)]
            subprocess.run(cmd+["--dry-run"],check=True)
            log=(output/f"repeats/gpu{gpu}.log").open("x");logs.append(log)
            processes[gpu]=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            print(f"Started logical{gpu}/physical{PHYSICAL[gpu]}: shapes={len(items)}, pid={processes[gpu].pid}",flush=True)
        statuses={gpu:process.wait() for gpu,process in processes.items()}
    except BaseException:
        for process in processes.values():
            if process.poll() is None:process.terminate()
        raise
    finally:
        for log in logs:log.close()
    dump(output/"worker_status.json",statuses)
    if any(statuses.values()):raise RuntimeError(statuses)
    if rounds==6:
        subprocess.run([sys.executable,str(LOCAL/"audit_final_repeat.py"),"--plan",str(output/"repeat_plan.json"),
           "--files",str(output/"repeats/gpu*.jsonl"),"--expected",str(output/"expected.csv"),
           "--known-csv",str(output/"known.csv"),"--output-dir",str(output/"audit"),
           "--physical-gpus",",".join(map(str,PHYSICAL)),"--require-complete"],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--logical-gpus",default="0,1,2,3");parser.add_argument("--screen-first",action="store_true")
    args=parser.parse_args();gpus=[int(value) for value in args.logical_gpus.split(",")]
    assert gpus and len(gpus)==len(set(gpus)) and set(gpus)<=set(range(4))
    args.output.mkdir();plan=json.loads(args.plan.read_text())
    dump(args.output/"controller.json",{"pid":os.getpid(),"gpus":gpus,"started_utc":datetime.now(timezone.utc).isoformat()})
    if args.screen_first:
        run(plan,args.output/"screen",2,21,gpus);plan=shortlist(plan,args.output/"screen")
    run(plan,args.output/"confirm",6,101,gpus)
    dump(args.output/"completed.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"exit_code":0})


if __name__=="__main__":main()
