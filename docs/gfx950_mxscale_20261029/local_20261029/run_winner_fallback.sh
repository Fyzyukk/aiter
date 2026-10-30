#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/env.sh
OPUS_FALLBACK_AUDIT="${1:-$OPUS_RUN_DIR/top8_current_audit}"
OPUS_FALLBACK_DIR="$OPUS_RUN_DIR/step4_winner_fallback_top0"
if [[ -e "$OPUS_FALLBACK_DIR" ]]; then
  printf 'Fallback output already exists: %s\n' "$OPUS_FALLBACK_DIR" >&2
  exit 12
fi
mkdir -p "$OPUS_FALLBACK_DIR"
export AITER_REBUILD=0
export AITER_MAX_JOBS=96
export OPUS_TUNE_PROGRESS_FILE="$OPUS_FALLBACK_DIR/progress.jsonl"
export OPUS_TUNE_CANDIDATE_PROGRESS_DIR="$OPUS_FALLBACK_DIR/candidates"
"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype opus --mp 4 --shape_grouped \
  --opus_top_geometries 0 --iters 21 --timeout 7200 \
  --opus_geometry_file "$OPUS_FALLBACK_AUDIT/winner_fallback_geometries.csv" \
  -i "$OPUS_FALLBACK_AUDIT/winner_fallback_shapes.csv" \
  -o "$OPUS_FALLBACK_DIR/tuned.csv" -o2 "$OPUS_FALLBACK_DIR/profile.csv" \
  > "$OPUS_FALLBACK_DIR/tune.log" 2>&1 &
OPUS_FALLBACK_PID="$!"
printf '%s\n' "$OPUS_FALLBACK_PID" > "$OPUS_FALLBACK_DIR/tune.pid"
OPUS_FALLBACK_RC=0
wait "$OPUS_FALLBACK_PID" || OPUS_FALLBACK_RC="$?"
printf '%s\n' "$OPUS_FALLBACK_RC" > "$OPUS_FALLBACK_DIR/exit_code"
exit "$OPUS_FALLBACK_RC"
