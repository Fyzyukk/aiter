#!/usr/bin/env python3
"""CPU review of completed finite official API gates and strict owner epochs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from audit_integration import HERE, ROOT, module, new_write, read, require, sha
from prepare_api_gate import EXPERIMENT, LAUNCHER, RUNNER


def analyze(directory):
    plan_path,queue_path = directory/'official_api_plan.json',directory/'official_api_queue.json'
    cpu_path = directory/'api_plan_cpu_audit.json'
    plan,queue,cpu = read(plan_path),read(queue_path),read(cpu_path)
    require(cpu['status']=='passed_CPU_API_plan_identity_pending_GPU' and cpu['plan_sha256']==sha(plan_path)
            and cpu['queue_sha256']==sha(queue_path),'CPU prepared API plan changed')
    identity_path = Path(plan['identity_audit_path'])
    identity = read(identity_path)
    require(sha(identity_path)==plan['identity_audit_sha256']==cpu['identity_audit_sha256']
            and identity['status']=='passed_pending_official_API_no_adoption','Complete CPU identity changed')
    require(sha(plan['official_binary'])==plan['official_binary_sha256']==identity['candidate_binary_sha256'],
            'Official binary changed')
    for field,path in [('runner_sha256',RUNNER),('experiment_runner_sha256',EXPERIMENT),('owner_launcher_sha256',LAUNCHER)]:
        require(sha(path)==plan[field],'GPU runner/launcher changed after preparation')
    require(len(plan['targets'])==len(queue['commands'])==len(cpu['rows'])==cpu['target_count'],
            'Prepared target count mismatch')
    claims = [json.loads(x) for x in Path(queue['claim_log']).read_text().splitlines() if x.strip()]
    require(claims and all(a['time']<=b['time'] for a,b in zip(claims,claims[1:])),'Claim chronology invalid')
    helper = module('remaining_api_strict_claim',HERE.parent/'diagnostics/merged_att_analysis.py')
    official = {'path':plan['official_binary'],'sha256':plan['official_binary_sha256']}
    rows = []
    labels = Counter()
    for index,(target,command,prepared) in enumerate(zip(plan['targets'],queue['commands'],cpu['rows'])):
        argv = command['argv']
        require('--check-only' in argv and '--profile' not in argv and
                argv[argv.index('--repetitions')+1]=='8' and argv[argv.index('--target-index')+1]==str(index),
                'GPU API protocol differs')
        output = Path(argv[argv.index('--output')+1])
        app = read(output)
        require(app['status']=='passed' and app['plan']==plan and app['source_head']==plan['source_head']
                and app['official_binary_sha256']==official['sha256'],'Executed API plan/module mismatch')
        require(app['runner_sha256']==sha(RUNNER) and app['experiment_runner_sha256']==sha(EXPERIMENT)
                and not app['profiling_only'] and len(app['rows'])==1,'GPU API runner/target mismatch')
        clean = helper.clean_claim(command['name'],app,command,queue,claims)
        require(not any(r.get('event')=='start' and clean['start']['time']<r['time']<clean['end']['time']
                        for r in claims),'Another command overlaps API epoch')
        row = app['rows'][0]
        require(row['target_index']==index==prepared['target_index'] and row['kid']==target['kid']==prepared['parent_id']
                and row['shape']==target['shape']==prepared['shape'] and row['split']==prepared['split_k'],
                'Executed exact API target/branch differs')
        require(row['actual_official_module']==official and row['timings']==[] and target['signed']
                and target['seed']==17,'Loaded official SHA/check-only/signed contract differs')
        expected = {'official':official}
        expected.update({label:{'path':path,'sha256':sha(path)} for label,path in target['private_baselines'].items()})
        require(row['libraries']==expected and set(row['correctness'])==set(expected),'Private/official label set differs')
        for label,correct in row['correctness'].items():
            require(correct['repetitions']==8 and correct['errRatio']==0 and correct['repeatable']
                    and correct['output_guards'] and correct['workspace_guards'],'Signed8 reference/guard gate failed')
            labels[label] += 8
        rows.append({'target_index':index,'parent_id':row['kid'],'shape':row['shape'],
                     'symbol':prepared['actual_candidate_symbol'],'changed':prepared['changed'],
                     'split_k':row['split'],'signed8_reference_repeatability_guards_passed':True,
                     'actual_official_module':row['actual_official_module'],'labels':list(expected),
                     'result_path':str(output),'result_sha256':sha(output),'clean_claim':clean})
    require({r['parent_id'] for r in rows}=={p['parent_id'] for p in identity['parents']},'All26 public API parents not covered')
    require({r['symbol'] for r in rows if r['changed']}==set(identity['selected_devices']),
            'Official API misses changed selected entries')
    require('torch' not in sys.modules,'CPU result analysis imported torch')
    return {'status':'passed_official_API_signed8_loaded_identity_all26parents','generated_utc':datetime.now(timezone.utc).isoformat(),
            'cpu_only_analysis':True,'new_GPU_execution':False,'new_build_execution':False,'production_modified':False,
            'target_count':len(rows),'public_parent_count':26,'changed_entries_checked':len(identity['selected_devices']),
            'numerical_calls_by_label':dict(labels),'official_module':official,'rows':rows,
            'evidence':{str(p):sha(p) for p in (plan_path,queue_path,cpu_path,identity_path,Path(queue['claim_log']))},
            'script_sha256':sha(__file__),
            'limits':['API correctness/guards and exact loaded module only; no performance inference.',
                      'Finite representative/boundary coverage does not test every supported shape.',
                      'Root must still apply/verify selected production sources and preserve historical artifacts.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-dir',type=Path,default=HERE/'api_gate')
    parser.add_argument('--output',type=Path,default=HERE/'api_gate/gpu_api_analysis.json')
    args = parser.parse_args()
    report = analyze(args.api_dir.resolve())
    new_write(args.output,report)
    print(json.dumps({'status':report['status'],'target_count':report['target_count'],'output':str(args.output)}))


if __name__ == '__main__':
    main()
