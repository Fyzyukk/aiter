#!/usr/bin/env bash
set -euo pipefail
source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/env.sh
export AITER_REBUILD=1
export OPUS_TUNE_PROGRESS_FILE="$OPUS_RUN_DIR/step1_progress.jsonl"
export OPUS_TUNE_CANDIDATE_PROGRESS_DIR="$OPUS_RUN_DIR/step1_candidates"
"$OPUS_RUN_PYTHON" csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype both --mp 4 --shape_grouped \
  --opus_top_geometries 0 \
  --opus_geometry_file docs/gfx950_mxscale_20261029/handoff_data/m_repeat_32_geometries.csv \
  -i docs/gfx950_mxscale_20261029/handoff_data/m_repeat_32_shapes.csv \
  -o "$OPUS_RUN_DIR/step1_tuned.csv" \
  -o2 "$OPUS_RUN_DIR/step1_profile.csv"
