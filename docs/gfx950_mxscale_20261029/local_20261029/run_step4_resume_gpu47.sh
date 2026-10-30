#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/env.sh
export ROCR_VISIBLE_DEVICES=4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3
export AITER_REBUILD=0 AITER_MAX_JOBS=32
export OPUS_RECOVERY_DIR="$OPUS_RUN_DIR/recovery_20261030T021903Z"
export OPUS_RESUME_WORKFLOW="$OPUS_RUN_DIR/step4_resume_gpu47_20261030"
mkdir "$OPUS_RESUME_WORKFLOW"
printf '%s\n' "$$" > "$OPUS_RESUME_WORKFLOW/launcher.pid"
trap 'printf "%s\n" "$?" > "$OPUS_RESUME_WORKFLOW/exit_code"' EXIT

set_phase() {
  printf '%s\n' "$1" > "$OPUS_RESUME_WORKFLOW/phase"
  "$OPUS_RUN_PYTHON" - "$1" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path(os.environ['OPUS_RUN_DIR']) / 'workflow_state.json'
s = json.loads(p.read_text())
s.update(updated_utc=datetime.now(timezone.utc).isoformat(), current_step=4,
         phase=sys.argv[1], gpu_ids=[4,5,6,7], scope='NT only',
         tuning_libtype='opus', baseline='handoff_data/csv_baseline.csv')
s['active_resume'] = {'directory':os.environ['OPUS_RESUME_WORKFLOW'],
                      'physical_gpu_ids':[4,5,6,7], 'logical_gpu_ids':[0,1,2,3],
                      'phase':sys.argv[1], 'preserved_profile_shapes':100,
                      'resume_profile_shapes':321}
s['steps']['4'] = 'in_progress: OPUS-only GPU4-7; preserving100 profiles and resuming321; then37 supplements and final six rounds'
temp = p.with_suffix('.json.tmp')
temp.write_text(json.dumps(s, ensure_ascii=False, indent=2)+'\n')
temp.replace(p)
PY
  printf '%s: %s\n' "$(date -u +%FT%TZ)" "$1"
}

set_phase preflight
"$OPUS_RUN_PYTHON" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
r = Path(os.environ['OPUS_RUN_DIR'])
c = json.loads((r/'full421_union/compile_receipt.json').read_text())
assert c['compiled'] and c['union_unique_kids']==26322
for path, expected in c['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, path
snapshot = json.loads(subprocess.check_output(
    ['rocm-smi','--showuse','--showmeminfo','vram','--json'],text=True))
for gpu in (4,5,6,7):
    row=snapshot[f'card{gpu}']
    assert int(row['GPU use (%)'])<=5, (gpu,row)
    assert int(row['VRAM Total Memory (B)'])-int(row['VRAM Total Used Memory (B)'])>128*1024**3, (gpu,row)
import torch
assert torch.cuda.device_count()==4
devices=[]
for gpu in range(4):
    p=torch.cuda.get_device_properties(gpu)
    assert p.gcnArchName.startswith('gfx950') and p.multi_processor_count==256
    devices.append({'logical_id':gpu,'physical_id':gpu+4,'name':p.name,'arch':p.gcnArchName,'cu_num':p.multi_processor_count})
(Path(os.environ['OPUS_RESUME_WORKFLOW'])/'preflight.json').write_text(
    json.dumps({'visible_devices':devices,'source_hashes_match':True,'jit_rebuilt':False,
                'baseline':str(r.parent/'handoff_data/csv_baseline.csv'),
                'gpu_snapshot':snapshot},indent=2)+'\n')
print('Four gfx950 devices mapped to physical GPU4-7; compiled source identities unchanged.',flush=True)
PY

OPUS_RESUME_FULL="$OPUS_RESUME_WORKFLOW/full421_top8"
OPUS_RESUME_FALLBACK="$OPUS_RESUME_WORKFLOW/winner_fallback_top0"
mkdir "$OPUS_RESUME_FULL"
cp "$OPUS_RECOVERY_DIR/saved100_profile.csv" "$OPUS_RESUME_FULL/profile.csv"
cp "$OPUS_RECOVERY_DIR/saved100_tuned.csv" "$OPUS_RESUME_FULL/tuned.csv"
cp "$OPUS_RECOVERY_DIR/saved100_progress.jsonl" "$OPUS_RESUME_FULL/progress.jsonl"
cp "$OPUS_RECOVERY_DIR/resume_shapes_321.csv" "$OPUS_RESUME_FULL/shapes.csv"
export OPUS_TUNE_PROGRESS_FILE="$OPUS_RESUME_FULL/progress.jsonl"
export OPUS_TUNE_CANDIDATE_PROGRESS_DIR="$OPUS_RESUME_FULL/candidates"
set_phase full421_top8_opus_only
"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype opus --mp 4 --shape_grouped \
  --opus_top_geometries 8 --iters 21 --timeout 7200 --batch 4 --all \
  -i "$OPUS_RESUME_FULL/shapes.csv" \
  -o "$OPUS_RESUME_FULL/tuned.csv" -o2 "$OPUS_RESUME_FULL/profile.csv" \
  > "$OPUS_RESUME_FULL/tune.log" 2>&1 &
OPUS_RESUME_TUNER_PID=$!
printf '%s\n' "$OPUS_RESUME_TUNER_PID" > "$OPUS_RESUME_FULL/tune.pid"
wait "$OPUS_RESUME_TUNER_PID"
printf '0\n' > "$OPUS_RESUME_FULL/exit_code"
export OPUS_RESUME_FULL
"$OPUS_RUN_PYTHON" - <<'PY'
import csv,json,os
from pathlib import Path
r=Path(os.environ['OPUS_RESUME_FULL'])
fields=('gfx','cu_num','M','N','K','w_scale_block')
key=lambda row:tuple(str(row[f]) for f in fields)
events=[json.loads(line) for line in (r/'progress.jsonl').read_text().splitlines() if line.strip()]
assert len(events)==len({key(e['shape']) for e in events})==421
assert all(e.get('best_opus') and e['best_opus']['us']>0 and 0<=e['best_opus']['errRatio']<=.05 for e in events)
assert len({key(row) for row in csv.DictReader((r/'profile.csv').open())})==421
print('Full421 OPUS coverage and all-candidate profile persistence complete.',flush=True)
PY

set_phase winner_fallback_top0_opus_only
mkdir "$OPUS_RESUME_FALLBACK"
export OPUS_TUNE_PROGRESS_FILE="$OPUS_RESUME_FALLBACK/progress.jsonl"
export OPUS_TUNE_CANDIDATE_PROGRESS_DIR="$OPUS_RESUME_FALLBACK/candidates"
"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype opus --mp 4 --shape_grouped \
  --opus_top_geometries 0 --iters 21 --timeout 7200 --batch 4 --all \
  --opus_geometry_file "$OPUS_RUN_DIR/top8_public_after22_audit/winner_fallback_geometries.csv" \
  -i "$OPUS_RUN_DIR/top8_public_after22_audit/winner_fallback_shapes.csv" \
  -o "$OPUS_RESUME_FALLBACK/tuned.csv" -o2 "$OPUS_RESUME_FALLBACK/profile.csv" \
  > "$OPUS_RESUME_FALLBACK/tune.log" 2>&1
printf '0\n' > "$OPUS_RESUME_FALLBACK/exit_code"

OPUS_RESUME_PROFILES=("$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" "$OPUS_RESUME_FULL/profile.csv" "$OPUS_RESUME_FALLBACK/profile.csv")
OPUS_RESUME_COVERAGE=(--full-progress "$OPUS_RESUME_FULL/progress.jsonl" --fallback-progress "$OPUS_RESUME_FALLBACK/progress.jsonl" --fallback-expected "$OPUS_RUN_DIR/top8_public_after22_audit/winner_fallback_shapes.csv")
set_phase discovery_audit
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --profile "${OPUS_RESUME_PROFILES[@]}" "${OPUS_RESUME_COVERAGE[@]}" \
  --physical-gpus 4,5,6,7 --current-registry \
  --output-dir "$OPUS_RESUME_WORKFLOW/discovery_report"
"$OPUS_RUN_PYTHON" - <<'PY'
import json,os
from pathlib import Path
s=json.loads((Path(os.environ['OPUS_RESUME_WORKFLOW'])/'discovery_report/summary.json').read_text())
assert s['full_top8_completed_shapes']==421
assert s['fallback_completed_shapes']==s['fallback_expected_shapes']==37
assert s['full_validation_complete'] and not s['missing_inputs'] and not s['no_correct_opus']
PY

set_phase prepare_final_repeat
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_final_repeat.py" \
  --profile "${OPUS_RESUME_PROFILES[@]}" \
  --known-csv "$OPUS_RUN_DIR/focused_expansion_known_complete/known_winners.csv" \
  --known-kids 389 100001 100003 100022 100023 100024 100025 100030 100031 \
  --known-repeat "$OPUS_RUN_DIR/step1_repeat_shard*.jsonl" "$OPUS_RUN_DIR/step2_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step3_existing_repeats/gpu*.jsonl" "$OPUS_RUN_DIR/co_round1/repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/co_round2/repeats/gpu*.jsonl" "$OPUS_RUN_DIR/focused_expansion_known_complete/repeats/gpu*.jsonl" \
  --current-report "$OPUS_RUN_DIR/co_rounds_family_integration/receipt.json" \
  --output-dir "$OPUS_RESUME_WORKFLOW/final_repeat_plan"
"$OPUS_RUN_PYTHON" - <<'PY'
import json,os
from pathlib import Path
s=json.loads((Path(os.environ['OPUS_RESUME_WORKFLOW'])/'final_repeat_plan/receipt.json').read_text())
assert s['repeat_shapes']==s['profile_complete_shapes']==421 and s['cpu_dry_run_passed']
assert s['known_csv_required_candidates']==s['known_csv_candidates_selected']==225
assert s['known_csv_omission_count']==0
PY

set_phase final_six_rounds_opus_only
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/run_repeat_shards.py" \
  --plan "$OPUS_RESUME_WORKFLOW/final_repeat_plan/repeat_plan.json" \
  --out-dir "$OPUS_RESUME_WORKFLOW/final/repeats" --rounds 6 --physical-gpus 4,5,6,7
set_phase final_batch_audit
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/audit_final_repeat.py" \
  --plan "$OPUS_RESUME_WORKFLOW/final_repeat_plan/repeat_plan.json" \
  --files "$OPUS_RESUME_WORKFLOW/final/repeats/gpu*.jsonl" \
  --physical-gpus 4,5,6,7 --require-complete --output-dir "$OPUS_RESUME_WORKFLOW/final_batch_audit"
set_phase final_report
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --profile "${OPUS_RESUME_PROFILES[@]}" "${OPUS_RESUME_COVERAGE[@]}" \
  --repeat "$OPUS_RESUME_WORKFLOW/final/repeats/gpu*.jsonl" \
  --physical-gpus 4,5,6,7 --current-registry --require-repeat-winners --finalize \
  --workflow-state "$OPUS_RUN_DIR/workflow_state.json" \
  --output-dir "$OPUS_RESUME_WORKFLOW/full421_final_report"
printf 'complete\n' > "$OPUS_RESUME_WORKFLOW/phase"
printf '%s: workflow complete\n' "$(date -u +%FT%TZ)"
