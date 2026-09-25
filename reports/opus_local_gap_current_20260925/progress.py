"""Print compact progress without touching GPU work."""
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
def read(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else {}
workflow = read("workflow.json")
build = read("build.json")
out = dict(workflow=workflow.get("status"), stage=workflow.get("stage"),
           elapsed_s=round(time.time()-workflow.get("start_time", time.time())),
           build=build.get("status"), building=build.get("building"))
for name in ("launcher.json", "confirmation.json"):
    data = read(name)
    if data:
        out[name] = dict(status=data["status"], workers=[
            {k:w.get(k) for k in ("gpu","status","completed_shapes","shapes","exit_code","error")}
            for w in data["workers"]])
for phase in ("r3", "confirm_r5"):
    states = []
    for gpu in range(4,8):
        data = read(f"gpu{gpu}_{phase}_run.json")
        if data:
            states.append({k:data.get(k) for k in ("gpu","status","completed_shapes","timing_records","error")})
    if states:
        out[phase] = states
if (HERE / "pipeline.json").exists():
    data = read("pipeline.json")
    out["pipeline"] = {k:data.get(k) for k in ("status","stage","error")}
import csv
choices=[]
migrated=set()
if (HERE/"final_recovery_plan.json").exists():
    migrated={tuple(s) for s in read("final_recovery_plan.json")["migrated_gpu4_close_shapes"]}
for p in HERE.glob("gpu[456]_r3_choices.csv"):
    for r in csv.DictReader(p.open()):
        if p.name.startswith("gpu4_") and tuple(int(r[k]) for k in ("M","N","K")) in migrated: continue
        choices.append(r)
for p in list(HERE.glob("gpu[56]_recovery_r3_choices.csv"))+list(HERE.glob("gpu[56]_final_recovery_r3_choices.csv")):
    choices.extend(csv.DictReader(p.open()))
compact=dict(stage=out.get("pipeline", {}).get("stage",out["stage"]), workflow=out["workflow"],
             build=out["build"],elapsed_s=out["elapsed_s"],
             per_gpu={r["gpu"]:r.get("completed_shapes") for r in out.get("r3",[])},
             completed=len(choices), provisional_wins=sum(r["opus_faster"]=="True" for r in choices),
             provisional_losses=sum(r["opus_faster"]=="False" for r in choices),
             errors=[r for r in out.get("r3",[])+out.get("confirm_r5",[]) if r.get("error")])
if "confirmation.json" in out: compact["confirmation"]=out["confirmation.json"]
if (HERE/"final_recovery.json").exists():
    recovery=read("final_recovery.json")
    compact["workflow"]=recovery["status"]
    compact["recovery"]={k:recovery.get(k) for k in ("status","stage","error")}
    compact["recovery"]["workers"]=[{k:w.get(k) for k in ("gpu","status","completed_shapes","shapes","error")} for w in recovery["workers"]]
    compact["errors"]=[r for r in compact["errors"] if r["gpu"]!=7]
print(json.dumps(compact, ensure_ascii=False))
