#!/usr/bin/env bash
# Continue only after the existing 19-shape batch has fully finished.
set -euo pipefail
source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/env.sh
export AITER_REBUILD=0
export AITER_MAX_JOBS=96
OPUS_EXISTING_PID="${1:-4162470}"
while kill -0 "$OPUS_EXISTING_PID" 2>/dev/null; do
  sleep 10
done
"$OPUS_RUN_PYTHON" - <<'PY'
import csv, os
from pathlib import Path
p = Path(os.environ['OPUS_RUN_DIR']) / 'step3_existing_round1'
for name in ('profile.csv',):
    with (p / name).open() as stream:
        shapes = {(r['gfx'], r['cu_num'], r['M'], r['N'], r['K'], r['w_scale_block'])
                  for r in csv.DictReader(stream)}
    if len(shapes) != 19:
        raise SystemExit(f'{name}: {len(shapes)} complete shapes, expected 19')
PY
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_stage3_recovery.py" --run
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_existing_repeat.py" \
  --profile "$OPUS_RUN_DIR/step3_existing_round1/profile_confirmable.csv"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/run_repeat_shards.py" \
  --plan "$OPUS_RUN_DIR/step3_existing_round1/cpu_analysis/repeat_plan.json" \
  --out-dir "$OPUS_RUN_DIR/step3_existing_repeats"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/summarize_existing_repeat.py" \
  "$OPUS_RUN_DIR/step3_existing_repeats/gpu*.jsonl" \
  --plan "$OPUS_RUN_DIR/step3_existing_round1/cpu_analysis/repeat_plan.json" \
  --output-dir "$OPUS_RUN_DIR/step3_existing_round1/repeat_report"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/select_stage3_co.py"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_co_experiment.py" \
  --plan "$OPUS_RUN_DIR/selected_co_plan.json" --check-only \
  > "$OPUS_RUN_DIR/selected_co_check.json"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_DIR/prepare_co_experiment.py" \
  --plan "$OPUS_RUN_DIR/selected_co_plan.json" \
  --out-dir "$OPUS_RUN_DIR/co_round1"
OPUS_EXPERIMENT_DIR="$OPUS_RUN_DIR/co_round1"
OPUS_EXPERIMENT_HELPERS="$OPUS_EXPERIMENT_DIR/.cache/experiment_repo/docs/gfx950_mxscale_20261029/local_20261029"
mkdir -p "$OPUS_EXPERIMENT_HELPERS"
# Keep helper-derived repository paths inside the isolated source copy.
cp "$OPUS_RUN_DIR"/*.py "$OPUS_EXPERIMENT_HELPERS/"
bash "$OPUS_EXPERIMENT_DIR/build_new_co.sh" > "$OPUS_EXPERIMENT_DIR/build.log" 2>&1
bash "$OPUS_EXPERIMENT_DIR/run_tune.sh" > "$OPUS_EXPERIMENT_DIR/tune.log" 2>&1
"$OPUS_RUN_PYTHON" "$OPUS_EXPERIMENT_HELPERS/prepare_existing_repeat.py" \
  --run-dir "$OPUS_EXPERIMENT_DIR" \
  --profile "$OPUS_EXPERIMENT_DIR/profile.csv" \
  --shapes "$OPUS_EXPERIMENT_DIR/experiment_shapes.csv"
source "$OPUS_EXPERIMENT_DIR/experiment_env.sh"
export AITER_REBUILD=0
"$OPUS_RUN_PYTHON" "$OPUS_EXPERIMENT_HELPERS/run_repeat_shards.py" \
  --plan "$OPUS_EXPERIMENT_DIR/cpu_analysis/repeat_plan.json" \
  --out-dir "$OPUS_EXPERIMENT_DIR/repeats"
"$OPUS_RUN_PYTHON" "$OPUS_EXPERIMENT_HELPERS/summarize_existing_repeat.py" \
  "$OPUS_EXPERIMENT_DIR/repeats/gpu*.jsonl" \
  --plan "$OPUS_EXPERIMENT_DIR/cpu_analysis/repeat_plan.json" \
  --output-dir "$OPUS_EXPERIMENT_DIR/repeat_report"
printf 'Stage3 existing comparison and finite CO experiment finished.\n'
