#!/usr/bin/env python3
"""CPU-only finalist selection from completed batches; all rounds restart."""
import argparse
import json
from pathlib import Path
from prepare_opus_repeat import csv_rows, key, shape_row, registry_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--sources",nargs="+",type=Path,required=True)
    args=parser.parse_args();metadata,pairs=registry_metadata()
    losses=[row for row in csv_rows(args.reference) if row["status"]=="slower_than_csv"]
    controls={key(row):int(row["kernelId"]) for row in losses}
    selected={shape:{kid:{"current_step4_best_control"}} for shape,kid in controls.items()}
    for index,directory in enumerate(args.sources):
        audit=json.loads((directory/"audit/summary.json").read_text())
        assert audit["completed_shapes"]==len(losses) and not audit["pending_shapes"]
        by_shape={}
        for path in (directory/"repeats").glob("gpu*.summary.json"):
            for row in json.loads(path.read_text())["candidates"]:
                if row["all_rounds_correct"]:
                    shape=key(row["shape"]);assert shape in selected
                    by_shape.setdefault(shape,[]).append(row)
        assert set(by_shape)==set(selected)
        for shape,rows in by_shape.items():
            for row in sorted(rows,key=lambda row:row["median_us"])[:2]:
                kid=row["kid"];assert row["kernelName"]==metadata[kid]["kernelName"]
                selected[shape].setdefault(kid,set()).add(f"source{index}_top2")
    baseline={key(row):float(row["csv_baseline_us"]) for row in losses}
    items=[]
    for shape,kids in sorted(selected.items()):
        candidates=[{**metadata[kid],"roles":sorted(roles),"exact_opposite_kids":pairs[kid],
                    "us":None,"errRatio":None,"correct":False} for kid,roles in sorted(kids.items())]
        items.append({"shape":shape_row(shape),"csv_baseline_us":baseline[shape],"remote_winner":None,"candidates":candidates})
    plan={"version":2,"family":"a8w8_blockscale_bpreshuffle","metadata_module":"prepare_opus_repeat",
          "correctness_limit":.05,"selection":"Top2 from each completed source and current reference best; six fresh rounds",
          "reference":str(args.reference),"sources":list(map(str,args.sources)),"shapes":items}
    (args.campaign/"final_plan.json").write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({"shapes":len(items),"candidate_groups":sum(len(item['candidates']) for item in items)}))


if __name__=="__main__":main()
