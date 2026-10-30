#!/usr/bin/env bash
# Run only after the full step1 batch has exited and both CSVs are complete.
set -euo pipefail
source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/env.sh
OPUS_STEP1_TUNE_PID="${1:-3873907}"
if kill -0 "$OPUS_STEP1_TUNE_PID" 2>/dev/null; then
  printf 'Step1 tuning PID %s is still alive; no GPU repeat was started.\n' "$OPUS_STEP1_TUNE_PID" >&2
  exit 12
fi
export AITER_REBUILD=0
"$OPUS_RUN_PYTHON" - <<'PY'
import csv, os
from pathlib import Path
root = Path(os.environ['OPUS_RUN_DIR'])
for name in ('step1_profile.csv', 'step1_tuned.csv'):
    with (root / name).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    keys = {(r['gfx'], r['cu_num'], r['M'], r['N'], r['K'], r['w_scale_block']) for r in rows}
    if len(keys) != 32:
        raise SystemExit(f'{name}: expected 32 complete shapes, got {len(keys)}')
print('Both step1 CSVs contain all 32 shapes; preparing repeat shards.', flush=True)
PY
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/paired_mrepeat.py"
"$OPUS_RUN_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['OPUS_RUN_DIR'])
plan = json.loads((root / 'step1_paired/repeat_plan.json').read_text())
if len(plan['shapes']) != 32:
    raise SystemExit('Repeat plan does not cover all 32 shapes')
shards, loads = [[], [], [], []], [0, 0, 0, 0]
for item in sorted(plan['shapes'], key=lambda item: len(item['candidates']), reverse=True):
    gpu = min(range(4), key=lambda index: (loads[index], len(shards[index])))
    shards[gpu].append(item)
    loads[gpu] += len(item['candidates'])
keys = [(s['shape']['M'], s['shape']['N'], s['shape']['K']) for shard in shards for s in shard]
if len(keys) != len(set(keys)) or len(keys) != 32:
    raise SystemExit('Shard coverage is not disjoint and complete')
seen_targets = {c['kid'] for shard in shards for s in shard for c in s['candidates'] if 48198 <= c['kid'] <= 48211}
if seen_targets != set(range(48198, 48212)):
    raise SystemExit(f'Missing target false configurations: {set(range(48198, 48212)) - seen_targets}')
for gpu, shard in enumerate(shards):
    output = root / f'step1_repeat_shard{gpu}.jsonl'
    if output.exists():
        raise SystemExit(f'Refusing to overwrite existing repeat output {output}')
    shard_plan = {**plan, 'gpu_logical_id': gpu, 'shapes': shard}
    (root / f'step1_repeat_shard{gpu}_plan.json').write_text(json.dumps(shard_plan, indent=2) + '\n')
    print(json.dumps({'gpu': gpu, 'shapes': len(shard), 'candidates': loads[gpu]}), flush=True)
PY
for OPUS_REPEAT_GPU in 0 1 2 3; do
  "$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/repeat_top_candidates.py" \
    --plan "$OPUS_RUN_DIR/step1_repeat_shard${OPUS_REPEAT_GPU}_plan.json" --dry-run
done
OPUS_REPEAT_PIDS=()
for OPUS_REPEAT_GPU in 0 1 2 3; do
  "$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/repeat_top_candidates.py" \
    --plan "$OPUS_RUN_DIR/step1_repeat_shard${OPUS_REPEAT_GPU}_plan.json" \
    --output "$OPUS_RUN_DIR/step1_repeat_shard${OPUS_REPEAT_GPU}.jsonl" \
    --gpu "$OPUS_REPEAT_GPU" --rounds 6 \
    > "$OPUS_RUN_DIR/step1_repeat_shard${OPUS_REPEAT_GPU}.log" 2>&1 &
  OPUS_REPEAT_PIDS+=("$!")
  printf '%s\n' "$!" > "$OPUS_RUN_DIR/step1_repeat_shard${OPUS_REPEAT_GPU}.pid"
done
OPUS_REPEAT_STATUS=0
for OPUS_REPEAT_PID in "${OPUS_REPEAT_PIDS[@]}"; do
  if ! wait "$OPUS_REPEAT_PID"; then
    OPUS_REPEAT_STATUS=1
  fi
done
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/summarize_step1_repeat.py" \
  "$OPUS_RUN_DIR/step1_repeat_shard*.jsonl"
exit "$OPUS_REPEAT_STATUS"
