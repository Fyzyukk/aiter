#!/usr/bin/env bash
set -euo pipefail
OPUS_STAGE14_LOCAL=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029
OPUS_STAGE14_PREVIOUS="$OPUS_STAGE14_LOCAL/stage13_local_scale69_20261030"
OPUS_STAGE14_CAMPAIGN="$OPUS_STAGE14_LOCAL/stage14_chunk_scale_20261030"
trap 'opus_stage14_exit=$?; if [[ -d "$OPUS_STAGE14_CAMPAIGN" ]]; then printf "%s\n" "$opus_stage14_exit" > "$OPUS_STAGE14_CAMPAIGN/controller_exit_code"; fi' EXIT
while [[ ! -f "$OPUS_STAGE14_PREVIOUS/completion_receipt.json" ]]; do
    if [[ -f "$OPUS_STAGE14_PREVIOUS/controller_exit_code" ]] && [[ "$(cat "$OPUS_STAGE14_PREVIOUS/controller_exit_code")" != 0 ]]; then
        exit 1
    fi
    sleep 5
done
source "$OPUS_STAGE14_LOCAL/env.sh"
"$OPUS_RUN_PYTHON" "$OPUS_STAGE14_LOCAL/prepare_stage14_chunk_scale.py"
bash "$OPUS_STAGE14_CAMPAIGN/co_experiment/build_new_co.sh"
bash "$OPUS_STAGE14_LOCAL/run_kernel_campaign.sh" "$OPUS_STAGE14_CAMPAIGN" \
  "$OPUS_STAGE14_PREVIOUS/result/public_best_opus421.csv" 48255
