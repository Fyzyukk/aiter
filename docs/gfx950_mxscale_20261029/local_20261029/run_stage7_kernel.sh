#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage7_kernel103_20261030/co_experiment/experiment_env.sh
export OPUS_STAGE7_DIR=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage7_kernel103_20261030
export OPUS_PUBLIC_REPO=/root/workspace/aiter-fyz-10-29-aiter
export OPUS_PUBLIC_LOCAL="$OPUS_PUBLIC_REPO/docs/gfx950_mxscale_20261029/local_20261029"
"$OPUS_RUN_PYTHON" - <<'PY'
import json,sys,shutil,os
from pathlib import Path
helper=Path.cwd()/'docs/gfx950_mxscale_20261029/local_20261029'
sys.path.insert(0,str(helper))
from prepare_opus_repeat import registry_metadata,pure_common,csv_rows,key,shape_row
out=Path(os.environ['OPUS_STAGE7_DIR']);local=Path(os.environ['OPUS_PUBLIC_LOCAL'])
identity=json.loads((out/'co_experiment/identity_and_points.json').read_text())
active={r['kid'] for r in json.loads((out/'co_experiment/build_info.json').read_text())['kernels'] if r['status']=='ready'}
assert active=={p['new_kid'] for p in identity['points']}
md,pairs=registry_metadata();common=pure_common();registry=common.get_registry('a8w8_blockscale_bpreshuffle','gfx950',256)
parents={p['new_kid']:p['seed_kid'] for p in identity['points']}
reference=local/'stage6_optimize117_20261030/result/public_best_opus421.csv'
rows=[r for r in csv_rows(reference) if r['status']=='slower_than_csv'];assert len(rows)==103
items=[]
for row in rows:
 shape=key(row);control=int(row['kernelId']);chosen={control:{'current_step4_best_control'}}
 for kid in sorted(active):
  if not common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid],*shape[2:5]):continue
  chosen.setdefault(kid,set()).add('stage7_new_kernel_body')
  parent=parents[kid]
  if common.a8w8_mxscale_bpreshuffle_supports_shape(registry[parent],*shape[2:5]):chosen.setdefault(parent,set()).add('exact_experiment_parent')
 candidates=[{**md[kid],'roles':sorted(roles),'exact_opposite_kids':pairs[kid],'us':None,'errRatio':None,'correct':False} for kid,roles in sorted(chosen.items())]
 items.append({'shape':shape_row(shape),'csv_baseline_us':float(row['csv_baseline_us']),'remote_winner':None,'candidates':candidates})
plan={'version':2,'family':'a8w8_blockscale_bpreshuffle','metadata_module':'prepare_opus_repeat','correctness_limit':.05,
 'selection':'New kernel bodies: direct BF16 output, streaming A, constant B ring, M32 padded scales and relaxed pin scheduling; two-round selection then six fresh rounds',
 'reference':str(reference),'shapes':items}
(out/'kernel_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
cache=Path(os.environ['AITER_JIT_DIR']);cache.mkdir(parents=True,exist_ok=True)
shutil.copy2(local/'stage6_optimize117_20261030/.cache/public_jit/module_aiter_core.so',cache/'module_aiter_core.so')
print(json.dumps({'shapes':103,'new_kernel_bodies':len(active),'candidate_groups':sum(len(i['candidates']) for i in items)}),flush=True)
PY
"$OPUS_RUN_PYTHON" "$OPUS_RUN_REPO/docs/gfx950_mxscale_20261029/local_20261029/run_stage6_batch.py" \
  --plan "$OPUS_STAGE7_DIR/kernel_plan.json" --output "$OPUS_STAGE7_DIR/final" --logical-gpus 0,1,2,3 --screen-first
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/finish_kernel_campaign.py" --campaign "$OPUS_STAGE7_DIR" \
  --reference "$OPUS_PUBLIC_LOCAL/stage6_optimize117_20261030/result/public_best_opus421.csv" --expected-registry 48246
source "$OPUS_STAGE7_DIR/public_env.sh"
"$OPUS_RUN_PYTHON" - <<'PY'
import os,shutil
from pathlib import Path
cache=Path(os.environ['AITER_JIT_DIR']);cache.mkdir(parents=True,exist_ok=True)
shutil.copy2(Path(os.environ['OPUS_PUBLIC_LOCAL'])/'stage6_optimize117_20261030/.cache/public_jit/module_aiter_core.so',cache/'module_aiter_core.so')
PY
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/repeat_opus_candidates.py" --plan "$OPUS_STAGE7_DIR/result/public421_plan.json" --gpu 0 --dry-run
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/precompile_stage5_plan.py" --plan "$OPUS_STAGE7_DIR/result/public421_plan.json" \
  --receipt "$OPUS_STAGE7_DIR/result/public_compile_receipt.json" --gpu 0
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/validate_public_campaign.py" --campaign "$OPUS_STAGE7_DIR"
