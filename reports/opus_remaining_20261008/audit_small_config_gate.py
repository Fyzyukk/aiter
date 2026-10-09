#!/usr/bin/env python3
"""CPU audit of all33 small actual-config numerical gates, no perf claim."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DIAG=HERE/'diagnostics'
spec=importlib.util.spec_from_file_location('small_gate_claim_cpu',HERE/'small_att_analysis.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}


def main():
    plan_path=DIAG/'all_config_gate_plan.json';result_path=DIAG/'all_config_gate_results.json'
    queue_path=DIAG/'all_config_gate_queue.json';claim_path=DIAG/'all_config_gate_claim.jsonl'
    plan=json.loads(plan_path.read_text());result=json.loads(result_path.read_text());queue=json.loads(queue_path.read_text())
    claims=[json.loads(x) for x in claim_path.read_text().splitlines() if x.strip()]
    assert len(queue['commands'])==1
    own=common.clean_claim(queue['commands'][0],result,claims)
    assert result['status']=='passed' and result['plan']==plan and len(plan['targets'])==len(result['rows'])==44
    assert result['profiling_only'] is False and result['official_binary_sha256']==plan['official_binary_sha256']==sha(plan['official_binary'])
    assert result['runner_sha256']==sha(ROOT/'reports/opus_bound_analysis_20261007/official_smoke.py')
    assert result['experiment_runner_sha256']==sha(ROOT/'reports/opus_bound_analysis_20261007/experiment_runner.py')
    review=json.loads((HERE/'small_family_review.json').read_text());configs={c['symbol']:c for c in review['configurations']}
    rows=[]
    descriptive_trait_label_corrections=[]
    for index,(target,row) in enumerate(zip(plan['targets'],result['rows'])):
        assert row['target_index']==index and row['kid']==target['kid'] and row['shape']==target['shape'] and row['timings']==[]
        assert row['actual_official_module']=={'path':plan['official_binary'],'sha256':plan['official_binary_sha256']}
        assert all(row['correctness']['official'].get(k)==v for k,v in
                   {'repetitions':2,'repeatable':True,'errRatio':0,'output_guards':True,'workspace_guards':True}.items())
        assert target['signed'] is True
        if target['symbol'] not in configs:continue
        config=configs[target['symbol']]
        assert target['instruction_sha256']==config['instruction_sha256']
        if target['traits']!=config['actual_selected_traits']:
            assert config['actual_configuration_id'] in [9042,9053,9054]
            assert target['traits'].endswith('false, false>') and config['actual_selected_traits']==target['traits'][:-len('false>')]+'true>'
            descriptive_trait_label_corrections.append({'target_index':index,'plan_label':target['traits'],
                'authoritative_actual_trait':config['actual_selected_traits'],'symbol_and_instruction_SHA_exact':True,
                'reason':'Plan description retained prereusefalse; mangledselectedsymbol andhash point to acceptedtrue alias.'})
        assert any(w['shape']==target['shape'] and w['parent_id']==target['kid'] for w in config['winner_shapes'])
        rows.append({'target_index':index,'symbol':target['symbol'],'parent_id':target['kid'],
                     'actual_configuration_id':config['actual_configuration_id'],'shape':target['shape'],
                     'exact_instruction_sha256':target['instruction_sha256'],'signed2repeat_reference_guards_passed':True,
                     'performance_measured':False})
    assert len(rows)==33 and {r['symbol'] for r in rows}==set(configs)
    report={'status':'passed_all33_small_actual_config_numerical_gate','cpu_only':True,'GPU_execution_by_this_audit':False,
            'wholeGPU_performance_measured':False,'total_formal_gate_targets':44,'small_actual_configs':33,
            'gate_coverage':'Exactly one retained actualwinner per33 currentsmallproducerconfig; no perconfig performance/bottleneck inference.',
            'descriptive_trait_label_corrections':descriptive_trait_label_corrections,
            'clean_claim':own,'rows':rows,'evidence':{'plan':ref(plan_path),'result':ref(result_path),'queue':ref(queue_path),
                                                    'claim':ref(claim_path)},'script':ref(Path(__file__))}
    (HERE/'small_config_gate_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'configs':33,'total_gate_targets':44}))


if __name__=='__main__':main()
