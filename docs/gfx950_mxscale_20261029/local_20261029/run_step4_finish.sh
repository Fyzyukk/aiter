#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/env.sh
export AITER_REBUILD=0
OPUS_FINISH_DIR="$OPUS_RUN_DIR/step4_finish"
if ! mkdir "$OPUS_FINISH_DIR"; then
  printf 'Finish launcher already exists: %s\n' "$OPUS_FINISH_DIR" >&2
  exit 12
fi
printf '%s\n' "$$" > "$OPUS_FINISH_DIR/launcher.pid"
OPUS_FINISH_RC=0
trap 'OPUS_FINISH_RC=$?; printf "%s\n" "$OPUS_FINISH_RC" > "$OPUS_FINISH_DIR/exit_code"' EXIT
printf '%s\n' 'waiting_for_full421' > "$OPUS_FINISH_DIR/phase"
OPUS_FULL_LAUNCHER=$(cat "$OPUS_RUN_DIR/step4_full421_top8/launcher.pid")
while [[ ! -f "$OPUS_RUN_DIR/step4_full421_top8/exit_code" ]]; do
  if ! kill -0 "$OPUS_FULL_LAUNCHER" 2>/dev/null; then
    if [[ -f "$OPUS_RUN_DIR/step4_full421_top8/exit_code" ]]; then
      break
    fi
    printf 'Full421 launcher exited without a completion receipt.\n' >&2
    exit 1
  fi
  sleep 5
done
test "$(cat "$OPUS_RUN_DIR/step4_full421_top8/exit_code")" = 0
"$OPUS_RUN_PYTHON" - <<'PY'
import json, os
from pathlib import Path
base = Path(os.environ['OPUS_RUN_DIR'])
rows = [json.loads(line) for line in (base/'step4_full421_top8/progress.jsonl').read_text().splitlines() if line.strip()]
fields = ('gfx', 'cu_num', 'M', 'N', 'K', 'w_scale_block')
assert len(rows) == len({tuple(r['shape'][f] for f in fields) for r in rows}) == 421
assert all(r.get('best_opus') and r['best_opus']['us'] > 0 and 0 <= r['best_opus']['errRatio'] <= .05 for r in rows)
print('Full421 completed with correct OPUS results; starting 37 winner supplements.', flush=True)
PY
printf '%s\n' 'winner_fallback_top0' > "$OPUS_FINISH_DIR/phase"
bash "$OPUS_RUN_DIR/run_winner_fallback.sh" "$OPUS_RUN_DIR/top8_public_after22_audit"
printf '%s\n' 'discovery_audit' > "$OPUS_FINISH_DIR/phase"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --profile "$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" \
    "$OPUS_RUN_DIR/step4_full421_top8/profile.csv" \
    "$OPUS_RUN_DIR/step4_winner_fallback_top0/profile.csv" \
  --full-progress "$OPUS_RUN_DIR/step4_full421_top8/progress.jsonl" \
  --fallback-progress "$OPUS_RUN_DIR/step4_winner_fallback_top0/progress.jsonl" \
  --fallback-expected "$OPUS_RUN_DIR/top8_public_after22_audit/winner_fallback_shapes.csv" \
  --current-registry --output-dir "$OPUS_RUN_DIR/step4_discovery_report"
"$OPUS_RUN_PYTHON" - <<'PY'
import json, os
from pathlib import Path
s = json.loads((Path(os.environ['OPUS_RUN_DIR'])/'step4_discovery_report/summary.json').read_text())
assert s['full_top8_completed_shapes'] == 421
assert s['fallback_completed_shapes'] == s['fallback_expected_shapes'] == 37
assert s['full_validation_complete'] and not s['missing_inputs'] and not s['no_correct_opus']
PY
printf '%s\n' 'prepare_final_repeat' > "$OPUS_FINISH_DIR/phase"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_final_repeat.py" \
  --profile "$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" \
    "$OPUS_RUN_DIR/step4_full421_top8/profile.csv" \
    "$OPUS_RUN_DIR/step4_winner_fallback_top0/profile.csv" \
  --known-csv "$OPUS_RUN_DIR/focused_expansion_known_complete/known_winners.csv" \
  --known-kids 389 100001 100003 100022 100023 100024 100025 100030 100031 \
  --known-repeat "$OPUS_RUN_DIR/step1_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step2_repeat_shard*.jsonl" \
    "$OPUS_RUN_DIR/step3_existing_repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/co_round1/repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/co_round2/repeats/gpu*.jsonl" \
    "$OPUS_RUN_DIR/focused_expansion_known_complete/repeats/gpu*.jsonl" \
  --current-report "$OPUS_RUN_DIR/co_rounds_family_integration/receipt.json" \
  --output-dir "$OPUS_RUN_DIR/final_repeat_plan"
"$OPUS_RUN_PYTHON" - <<'PY'
import json, os
from pathlib import Path
base = Path(os.environ['OPUS_RUN_DIR'])
s = json.loads((base/'final_repeat_plan/receipt.json').read_text())
assert s['repeat_shapes'] == s['profile_complete_shapes'] == 421 and s['cpu_dry_run_passed']
assert s['known_csv_required_candidates'] == s['known_csv_candidates_selected'] == 225
assert s['known_csv_omission_count'] == 0
p = json.loads((base/'final_repeat_plan/repeat_plan.json').read_text())
assert any(c['kid'] == 48096 for item in p['shapes'] for c in item['candidates'])
print('Complete421 final shortlist and all 225 known candidates validated.', flush=True)
PY
printf '%s\n' 'final_six_rounds' > "$OPUS_FINISH_DIR/phase"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/run_repeat_shards.py" \
  --plan "$OPUS_RUN_DIR/final_repeat_plan/repeat_plan.json" \
  --out-dir "$OPUS_RUN_DIR/final/repeats" --rounds 6
printf '%s\n' 'final_batch_audit' > "$OPUS_FINISH_DIR/phase"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/audit_final_repeat.py" \
  --plan "$OPUS_RUN_DIR/final_repeat_plan/repeat_plan.json" \
  --files "$OPUS_RUN_DIR/final/repeats/gpu*.jsonl" \
  --require-complete --output-dir "$OPUS_RUN_DIR/final_batch_audit"
printf '%s\n' 'final_report' > "$OPUS_FINISH_DIR/phase"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/finish_workflow.py" \
  --profile "$OPUS_RUN_DIR/focused_expansion_known_complete/profile.csv" \
    "$OPUS_RUN_DIR/step4_full421_top8/profile.csv" \
    "$OPUS_RUN_DIR/step4_winner_fallback_top0/profile.csv" \
  --full-progress "$OPUS_RUN_DIR/step4_full421_top8/progress.jsonl" \
  --fallback-progress "$OPUS_RUN_DIR/step4_winner_fallback_top0/progress.jsonl" \
  --fallback-expected "$OPUS_RUN_DIR/top8_public_after22_audit/winner_fallback_shapes.csv" \
  --repeat "$OPUS_RUN_DIR/final/repeats/gpu*.jsonl" \
  --current-registry --require-repeat-winners --finalize \
  --workflow-state "$OPUS_RUN_DIR/workflow_state.json" \
  --output-dir "$OPUS_RUN_DIR/full421_final_report"
printf '%s\n' 'complete' > "$OPUS_FINISH_DIR/phase"
