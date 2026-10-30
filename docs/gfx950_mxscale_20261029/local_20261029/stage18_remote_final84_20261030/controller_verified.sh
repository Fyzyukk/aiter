#!/usr/bin/env bash
set -euo pipefail
export OPUS_CAMPAIGN_DIR=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030 OPUS_REFERENCE_CSV=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage17_pin_invariants54_20261030/result/public_best_opus421.csv OPUS_EXPECTED_SIZE=48261
source "$OPUS_CAMPAIGN_DIR/co_experiment/experiment_env.sh"
export OPUS_PUBLIC_REPO=/root/workspace/aiter-fyz-10-29-aiter OPUS_PUBLIC_LOCAL=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029
trap 'task_exit_code=$?; printf "%s\n" "$task_exit_code" > "$OPUS_CAMPAIGN_DIR/controller_verified_exit_code"; if test "$task_exit_code" -eq 0; then printf "0\n" > "$OPUS_CAMPAIGN_DIR/controller_exit_code"; fi' EXIT
"$OPUS_RUN_PYTHON" "$OPUS_PUBLIC_LOCAL/assemble_stage18_verified_batch.py"
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
"$OPUS_RUN_PYTHON" - <<'PY_RECEIPT'
from pathlib import Path
import os,json
c=Path(os.environ['OPUS_CAMPAIGN_DIR'])
r=json.loads((c/'completion_receipt.json').read_text());a=json.loads((c/'measurement_verification_assembly.json').read_text())
assert r['actual_measurements']==a['accepted_screen_and_final_records']
r['accepted_screen_and_final_measurements']=r['actual_measurements'];r['actual_measurements']=a['actual_total_performance_measurements'];r['measurement_verification']=a;r['superseded_measurements']=a['superseded_measurements']
a['normal_stable_integration_pending']=False
(c/'measurement_verification_assembly.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
(c/'completion_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
PY_RECEIPT
