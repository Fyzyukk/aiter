#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/env.sh
OPUS_FULL_DIR="$OPUS_RUN_DIR/step4_full421_top8"
if [[ -e "$OPUS_FULL_DIR" ]]; then
  printf 'Full421 output already exists: %s\n' "$OPUS_FULL_DIR" >&2
  exit 12
fi
mkdir -p "$OPUS_FULL_DIR"
export AITER_REBUILD=0
export AITER_MAX_JOBS=96
export OPUS_TUNE_PROGRESS_FILE="$OPUS_FULL_DIR/progress.jsonl"
export OPUS_TUNE_CANDIDATE_PROGRESS_DIR="$OPUS_FULL_DIR/candidates"
printf '%s\n' "$$" > "$OPUS_FULL_DIR/launcher.pid"
export OPUS_FULL_DIR
"$OPUS_RUN_PYTHON" - <<'PY'
import csv, json, os
from pathlib import Path
root, output = Path(os.environ['OPUS_RUN_REPO']), Path(os.environ['OPUS_FULL_DIR'])
fields = ('gfx', 'cu_num', 'M', 'N', 'K', 'w_scale_block')
key = lambda row: tuple(str(row[field]) for field in fields)
with (root / 'docs/gfx950_mxscale_20261029/shapes_421.csv').open() as stream:
    reader = csv.DictReader(stream)
    columns, rows = reader.fieldnames, list(reader)
with (root / 'docs/gfx950_mxscale_20261029/handoff_data/csv_baseline.csv').open() as stream:
    baseline = {key(row): float(row['us']) for row in csv.DictReader(stream)}
receipt = Path(os.environ['OPUS_RUN_DIR']) / 'full421_union/per_shape_receipt.json'
counts = {}
if receipt.exists():
    counts = {(r['M'], r['N'], r['K']): r['candidate_count']
              for r in json.loads(receipt.read_text())['shapes'] if r['source'] == 'full_top8'}
if len(rows) != 421 or len({key(row) for row in rows}) != 421:
    raise SystemExit('Expected 421 distinct input shapes')
def cost(row):
    mnk = tuple(int(row[field]) for field in ('M', 'N', 'K'))
    return counts.get(mnk, 2000) * (0.015 + 42 * baseline[key(row)] / 1e6)
rows.sort(key=cost, reverse=True)
with (output / 'shapes.csv').open('x') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
print('Scheduled all 421 shapes, larger estimated tasks first.', flush=True)
PY
"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype both --mp 4 --shape_grouped \
  --opus_top_geometries 8 --iters 21 --timeout 7200 \
  -i "$OPUS_FULL_DIR/shapes.csv" \
  -o "$OPUS_FULL_DIR/tuned.csv" -o2 "$OPUS_FULL_DIR/profile.csv" \
  > "$OPUS_FULL_DIR/tune.log" 2>&1 &
OPUS_FULL_PID="$!"
printf '%s\n' "$OPUS_FULL_PID" > "$OPUS_FULL_DIR/tune.pid"
OPUS_FULL_RC=0
wait "$OPUS_FULL_PID" || OPUS_FULL_RC="$?"
printf '%s\n' "$OPUS_FULL_RC" > "$OPUS_FULL_DIR/exit_code"
exit "$OPUS_FULL_RC"
