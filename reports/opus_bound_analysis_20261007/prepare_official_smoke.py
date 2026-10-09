#!/usr/bin/env python3
"""CPU-only selection of current 26 parent representatives and known gaps."""
import csv
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def main():
    build=json.loads((OUT/'current_build.json').read_text())
    table=ROOT/'reports/opus_current745_tables_20260930/shape_comparison_745.csv'
    rows=list(csv.DictReader(table.open(encoding='utf-8-sig')))
    targets=[]
    def add(kid,shape,purpose,reference=None):
        target={'kid':kid,'shape':shape,'seed':17+len(targets),'signed':True,
                'purpose':purpose,'private_baselines':{}}
        if reference is not None:
            target['historical_source_run']=reference['source_run']
            target['historical_us']=float(reference['opus_us'])
        # These libraries mimic the complete current public dispatch or are
        # byte-verified runtime kernels. A parent ID alone is insufficient.
        m,n,k=shape
        if kid in [9021,9022]:
            target['private_baselines']['compute_prologue']=str(OUT/'compute_prologue/baseline/experiments.so')
        if kid in [9060,9061,9062,9063]:
            target['private_baselines']['fine_wait']=str(OUT/'fine_wait/baseline/experiments.so')
        if kid in [9040,9042,9051,9052,9053,9054] and not(k==7168 and kid in [9042,9052,9053]):
            target['private_baselines']['register_reuse']=str(OUT/'register_reuse/baseline/experiments.so')
        if kid==9062 and k==16384 and (m+95)//96 >= (m+79)//80:
            target['private_baselines']['fine_n64']=str(OUT/'fine_n64/baseline/experiments.so')
        targets.append(target)
    for kid in build['kids']:
        matches=[row for row in rows if int(row['opus_kernelId'])==kid]
        if not matches: raise ValueError(f'No historical representative for {kid}')
        chosen=min(matches,key=lambda row:int(row['M'])*int(row['K'])+int(row['N'])*int(row['K'])+2*int(row['M'])*int(row['N']))
        add(kid,[int(chosen[name]) for name in ['M','N','K']],
            'minimum-logical-bytes historical winner for this current parent',chosen)
    for m in [192,384]:
        add(9042,[m,768,7168],'largest known same-run performance gap; official K7168 branch')
    add(9052,[17,256,384],'runtime register-waveK private/current ABI agreement')
    add(9062,[144,256,16384],'fixed-K M80 split2 private/current ABI agreement')
    plan={'source_head':build['source_head'],'official_binary':build['binary'],
        'official_binary_sha256':build['binary_sha256'],
        'current_parent_count':len(build['kids']), 'historical_table':str(table),
        'historical_table_sha256':hashlib.sha256(table.read_bytes()).hexdigest(),
        'selection':'One small-storage historical chosen-kid shape per current default parent, two 9042 gaps, two private alignment shapes.',
        'targets':targets}
    path=OUT/'plans/official_smoke.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'status':'prepared_no_gpu_execution','targets':len(targets),
                      'parents':len(build['kids']),'output':str(path)}))

if __name__=='__main__': main()
