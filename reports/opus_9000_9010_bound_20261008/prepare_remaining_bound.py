from pathlib import Path
import json,collections
ROOT=Path.cwd();OUT=ROOT/'reports/opus_9000_9010_bound_20261008';OLD=ROOT/'reports/opus_bound_analysis_20261007'
inv=json.loads((ROOT/'reports/opus_remaining_20261008/inventory.json').read_text());hist=json.loads((OLD/'shape_inventory.json').read_text())['rows'];prior=json.loads((ROOT/'reports/opus_remaining_20261008/small_att_analysis.json').read_text())
covered={c['symbol'] for c in prior['captures']};targets=[]
for e in inv['entries']:
 if e['parent_id']<9040 or e['symbol'] in covered:continue
 shape=e['representative'];row=next(r for r in hist if r['parent_id']==e['parent_id'] and [r['M'],r['N'],r['K']]==shape)
 targets.append({'kid':e['parent_id'],'shape':shape,'seed':17,'signed':True,'purpose':'own actual-config bound classification representative','private_baselines':{},'symbol':e['symbol'],'kernel_function':e['kernel_function'],'traits':e['traits'],'expected_mfma_total':row['padded_mfma_instruction_count_estimate'],'waves':row['waves'],'producer_workgroups':row['producer_workgroups'],'actual_ids':e['actual_configuration_ids']})
module=json.loads((OUT/'source_manifest.json').read_text())['baseline'];plan={'source_head':inv['source_head'],'current_parent_count':26,'official_binary':module['path'],'official_binary_sha256':module['sha256'],'targets':targets}
folder=OUT/'remaining_bound_diagnostics';folder.mkdir(exist_ok=True);(folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
t=json.loads((OUT/'baseline_diagnostics/att_queue.json').read_text());commands=[]
prefix=['/opt/venv/bin/python3',str(OLD/'official_smoke.py'),'--plan',str(folder/'plan.json')]
commands.append({'name':'additional_config_correctness','argv':prefix+['--output',str(folder/'correctness.json'),'--check-only','--repetitions','2'],'log':str(folder/'correctness.log')})
for i,target in enumerate(targets):
 name=f'target{i}_kid{target["kid"]}_att';d=folder/name;env=t['commands'][1]['env'].copy();env.update(ROCPROF_OUTPUT_FILE_NAME=name,ROCPROF_OUTPUT_PATH=str(d),ROCPROF_KERNEL_FILTER_INCLUDE_REGEX=target['kernel_function'],ROCPROF_ATT_PARAM_TARGET_CU='1')
 commands.append({'name':name,'argv':prefix+['--output',str(d/'application.json'),'--profile','--target-index',str(i),'--iters','11','--profile-rotation','1'],'env':env,'log':str(d/'rocprof.log')})
(folder/'att_queue.json').write_text(json.dumps({'env':t['env'],'commands':commands},indent=2)+'\n')
print('remaining configs',len(targets),[(t['kid'],t['actual_ids'],t['shape']) for t in targets])
