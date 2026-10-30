#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 3
export OPUS_CAMPAIGN_DIR="$1" OPUS_REFERENCE_CSV="$2" OPUS_EXPECTED_SIZE="$3"
source "$OPUS_CAMPAIGN_DIR/co_experiment/experiment_env.sh"
export OPUS_PUBLIC_REPO=/root/workspace/aiter-fyz-10-29-aiter
export OPUS_PUBLIC_LOCAL="$OPUS_PUBLIC_REPO/docs/gfx950_mxscale_20261029/local_20261029"
"$OPUS_RUN_PYTHON" - <<'PY'
import json,sys,shutil,os
from pathlib import Path
helper=Path.cwd()/'docs/gfx950_mxscale_20261029/local_20261029'
sys.path.insert(0,str(helper))
from prepare_opus_repeat import registry_metadata,pure_common,csv_rows,key,shape_row
out=Path(os.environ['OPUS_CAMPAIGN_DIR']);local=Path(os.environ['OPUS_PUBLIC_LOCAL'])
identity=json.loads((out/'co_experiment/identity_and_points.json').read_text())
build=json.loads((out/'co_experiment/build_info.json').read_text())['kernels']
active={r['kid'] for r in build if r['status']=='ready'}
assert active and active<={p['new_kid'] for p in identity['points']}
(out/'build_selection.json').write_text(json.dumps({'ready_ids':sorted(active),
 'rejected':[r for r in build if r['status']!='ready'],'definition':'Only successfully compiled, available images enter the GPU plan; rejected diagnostics retained.'},indent=2)+'\n')
md,pairs=registry_metadata();common=pure_common();registry=common.get_registry('a8w8_blockscale_bpreshuffle','gfx950',256)
parents={p['new_kid']:p['seed_kid'] for p in identity['points']}
existing=identity.get('existing_candidates_by_shape',{})
reference=Path(os.environ['OPUS_REFERENCE_CSV'])
rows=[r for r in csv_rows(reference) if r['status']=='slower_than_csv'];assert rows
items=[]
for row in rows:
 shape=key(row);control=int(row['kernelId']);chosen={control:{'current_step4_best_control'}}
 for item in existing.get(','.join(map(str,shape[2:5])),[]):
  kid=int(item['kid'])
  assert kid in registry and item['kernelName']==registry[kid].name
  assert common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid],*shape[2:5])
  chosen.setdefault(kid,set()).add(item['role'])
 for kid in sorted(active):
  if not common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid],*shape[2:5]):continue
  chosen.setdefault(kid,set()).add('new_kernel_body')
  parent=parents[kid]
  if common.a8w8_mxscale_bpreshuffle_supports_shape(registry[parent],*shape[2:5]):chosen.setdefault(parent,set()).add('exact_experiment_parent')
 if os.environ.get('OPUS_TARGET_NEW_ONLY')=='1' and not any('new_kernel_body' in roles for roles in chosen.values()):continue
 candidates=[{**md[kid],'roles':sorted(roles),'exact_opposite_kids':pairs[kid],'us':None,'errRatio':None,'correct':False} for kid,roles in sorted(chosen.items())]
 items.append({'shape':shape_row(shape),'csv_baseline_us':float(row['csv_baseline_us']),'remote_winner':None,'candidates':candidates})
plan={'version':2,'family':'a8w8_blockscale_bpreshuffle','metadata_module':'prepare_opus_repeat','correctness_limit':.05,
 'selection':'Kernel-body family expansion with exact current best and parent controls; two-round selection then six fresh rounds',
 'reference':str(reference),'target_new_only':os.environ.get('OPUS_TARGET_NEW_ONLY')=='1',
 'matched_controls':identity.get('matched_controls',[]),'shapes':items}
assert items
(out/'kernel_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
cache=Path(os.environ['AITER_JIT_DIR']);cache.mkdir(parents=True,exist_ok=True)
shutil.copy2(reference.parents[1]/'.cache/public_jit/module_aiter_core.so',cache/'module_aiter_core.so')
print(json.dumps({'reference_losses':len(rows),'shapes':len(items),'new_kernel_bodies':len(active),'candidate_groups':sum(len(i['candidates']) for i in items)}),flush=True)
PY
"$OPUS_RUN_PYTHON" "$OPUS_RUN_REPO/docs/gfx950_mxscale_20261029/local_20261029/run_stage6_batch.py" \
  --plan "$OPUS_CAMPAIGN_DIR/kernel_plan.json" --output "$OPUS_CAMPAIGN_DIR/final" --logical-gpus 0,1,2,3 --screen-first
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/finish_kernel_campaign.py" --campaign "$OPUS_CAMPAIGN_DIR" \
  --reference "$OPUS_REFERENCE_CSV" --expected-registry "$OPUS_EXPECTED_SIZE"
source "$OPUS_CAMPAIGN_DIR/public_env.sh"
"$OPUS_RUN_PYTHON" - <<'PY'
import os,shutil
from pathlib import Path
cache=Path(os.environ['AITER_JIT_DIR']);cache.mkdir(parents=True,exist_ok=True)
shutil.copy2(Path(os.environ['OPUS_REFERENCE_CSV']).parents[1]/'.cache/public_jit/module_aiter_core.so',cache/'module_aiter_core.so')
PY
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/repeat_opus_candidates.py" --plan "$OPUS_CAMPAIGN_DIR/result/public421_plan.json" --gpu 0 --dry-run
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/precompile_stage5_plan.py" --plan "$OPUS_CAMPAIGN_DIR/result/public421_plan.json" \
  --receipt "$OPUS_CAMPAIGN_DIR/result/public_compile_receipt.json" --gpu 0
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/validate_public_campaign.py" --campaign "$OPUS_CAMPAIGN_DIR"
