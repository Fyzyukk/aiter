#!/usr/bin/env bash
set -euo pipefail
export OPUS_CAMPAIGN_DIR=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030 OPUS_REFERENCE_CSV=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage17_pin_invariants54_20261030/result/public_best_opus421.csv OPUS_EXPECTED_SIZE=48261
source "$OPUS_CAMPAIGN_DIR/co_experiment/experiment_env.sh"
export OPUS_PUBLIC_REPO=/root/workspace/aiter-fyz-10-29-aiter
export OPUS_PUBLIC_LOCAL=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029
trap 'task_exit_code=$?; printf "%s\n" "$task_exit_code" > "$OPUS_CAMPAIGN_DIR/controller_resume_exit_code"; if test "$task_exit_code" -eq 0; then printf "0\n" > "$OPUS_CAMPAIGN_DIR/controller_exit_code"; fi' EXIT
"$OPUS_RUN_PYTHON" - <<'RESUME_PY'
import sys,json,os
from pathlib import Path
from datetime import datetime,timezone
helper=Path(os.environ['OPUS_RUN_REPO'])/'docs/gfx950_mxscale_20261029/local_20261029'
sys.path.insert(0,str(helper))
from run_stage6_batch import run,dump
from run_stage5_batch import shortlist
c=Path(os.environ['OPUS_CAMPAIGN_DIR']);out=c/'final'
assert json.loads((out/'screen/worker_status.json').read_text())=={'0':0,'1':0,'2':0,'3':0}
failed=out/'confirm'
assert failed.is_dir() and not any(p.is_file() for p in failed.rglob('*'))
failed.rename(out/'confirm_preflight_stopped')
original=(c/'controller_exit_code').read_text().strip()
dump(c/'resume_receipt.json',{'original_controller_exit_code':int(original),'original_stop':'GPU5 utilization preflight at screen-to-confirm boundary; all screen workers exited 0','resumed_utc':datetime.now(timezone.utc).isoformat(),'screen_reused':True})
plan=shortlist(json.loads((c/'kernel_plan.json').read_text()),out/'screen')
run(plan,out/'confirm',6,101,[0,1,2,3])
dump(out/'completed.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'exit_code':0,'resumed_after_preflight':True})
RESUME_PY
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
